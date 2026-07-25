from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from .calibration import SecondMomentState
from .config import AdapterConfig
from .inject import (
    adapter_modules,
    attention_group_key,
    inject_attention_adapters,
)
from .layers import FilteredLoRALinear


@dataclass(frozen=True)
class InitializationRecord:
    method: str
    module_count: int
    exact_function_preserving: bool
    calibration_required: bool
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _scaled_factors_from_svd(
    matrix: torch.Tensor,
    *,
    rank: int,
    scaling: float,
    select: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if rank > min(matrix.shape):
        raise ValueError(f"rank {rank} exceeds matrix shape {tuple(matrix.shape)}")
    work = matrix.detach().to(device="cpu", dtype=torch.float64)
    left, singular_values, right_h = torch.linalg.svd(work, full_matrices=False)
    if select == "largest":
        left = left[:, :rank]
        singular_values = singular_values[:rank]
        right_h = right_h[:rank, :]
    elif select == "smallest":
        left = left[:, -rank:]
        singular_values = singular_values[-rank:]
        right_h = right_h[-rank:, :]
    else:
        raise ValueError(f"Unknown singular-value selection: {select}")
    roots = torch.sqrt(torch.clamp(singular_values / scaling, min=0.0))
    lora_B = left * roots.unsqueeze(0)
    lora_A = roots.unsqueeze(1) * right_h
    return lora_A, lora_B


@torch.no_grad()
def initialize_lora_null(
    model: nn.Module,
    moments: Mapping[str, SecondMomentState],
) -> InitializationRecord:
    """Apply the official LoRA-Null low-activation-subspace initialization.

    Track B differs from the official repository only in target scope: this
    function operates on already-injected attention-only modules.
    """

    dimensions: Dict[str, int] = {}
    count = 0
    for name, module in adapter_modules(model):
        group = attention_group_key(name)
        if group not in moments:
            raise KeyError(f"Missing LoRA-Null calibration moment for {group}")
        covariance = moments[group].matrix.to(device="cpu", dtype=torch.float64)
        _, eigenvectors = torch.linalg.eigh(
            0.5 * (covariance + covariance.transpose(0, 1))
        )
        if module.rank > eigenvectors.shape[1]:
            raise ValueError(
                f"rank {module.rank} exceeds input dimension for {name}"
            )
        low_basis = eigenvectors[:, : module.rank]
        weight = module.base_layer.weight.detach().to(device="cpu", dtype=torch.float64)
        low_component = weight @ low_basis @ low_basis.transpose(0, 1)
        lora_A, lora_B = _scaled_factors_from_svd(
            low_component,
            rank=module.rank,
            scaling=module.scaling,
            select="largest",
        )
        module.set_decomposition_initialization(lora_A, lora_B)
        dimensions[name] = module.rank
        count += 1
    return InitializationRecord(
        method="lora_null",
        module_count=count,
        exact_function_preserving=True,
        calibration_required=True,
        details={
            "subspace": "lowest_activation_energy",
            "target_scope": "attention_only",
            "dimensions": dimensions,
        },
    )


@torch.no_grad()
def initialize_milora(model: nn.Module) -> InitializationRecord:
    count = 0
    for _, module in adapter_modules(model):
        weight = module.base_layer.weight.detach()
        lora_A, lora_B = _scaled_factors_from_svd(
            weight,
            rank=module.rank,
            scaling=module.scaling,
            select="smallest",
        )
        module.set_decomposition_initialization(lora_A, lora_B)
        count += 1
    return InitializationRecord(
        method="milora",
        module_count=count,
        exact_function_preserving=True,
        calibration_required=False,
        details={"singular_values": "smallest", "target_scope": "attention_only"},
    )


def build_native_adapter(
    model: nn.Module,
    config: AdapterConfig,
    *,
    calibration_moments: Optional[Mapping[str, SecondMomentState]] = None,
) -> Tuple[Dict[str, FilteredLoRALinear], InitializationRecord]:
    wrapped = inject_attention_adapters(model, config)
    if config.method == "lora_null":
        if calibration_moments is None:
            raise ValueError("LoRA-Null requires calibration moments")
        record = initialize_lora_null(model, calibration_moments)
    elif config.method == "milora":
        record = initialize_milora(model)
    else:
        record = InitializationRecord(
            method=config.method,
            module_count=len(wrapped),
            exact_function_preserving=True,
            calibration_required=config.method == "lora_nf",
            details={
                "initialization": "standard_lora",
                "runtime_filter": config.method == "lora_nf",
                "target_scope": "attention_only",
            },
        )
    return wrapped, record


def build_peft_baseline(model: nn.Module, config: AdapterConfig, method: str) -> nn.Module:
    """Create audited PEFT baselines for LoRA, DoRA, PiSSA, and CorDA.

    CorDA preprocessing is deliberately not hidden here; callers must run the
    method-specific PEFT preprocessing callback before calling get_peft_model.
    """

    from peft import LoraConfig, TaskType, get_peft_model

    if method not in {"lora", "dora", "pissa"}:
        raise ValueError(f"Unsupported direct PEFT baseline: {method}")
    kwargs: Dict[str, Any] = {}
    if method == "dora":
        kwargs["use_dora"] = True
    if method == "pissa":
        if config.dropout != 0.0:
            raise ValueError("PiSSA requires dropout=0.0 in the locked protocol")
        kwargs["init_lora_weights"] = "pissa_niter_16"
    peft_config = LoraConfig(
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        target_modules=list(config.target_modules),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        **kwargs,
    )
    return get_peft_model(model, peft_config)
