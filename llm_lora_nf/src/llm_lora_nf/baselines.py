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
    work = matrix.detach().to(dtype=torch.float32)
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


def _scaled_factors_from_projected_weight(
    weight: torch.Tensor,
    low_basis: torch.Tensor,
    *,
    scaling: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Factor W U U^T through the much smaller W U matrix.

    If U is orthonormal, WU and WUU^T have identical nonzero singular
    values. This produces the same square-root SVD parameterization as a
    dense SVD of WUU^T without materializing the d_out x d_in projection.
    """

    compressed = weight @ low_basis
    left, singular_values, compressed_right_h = torch.linalg.svd(
        compressed,
        full_matrices=False,
    )
    right_h = compressed_right_h @ low_basis.transpose(0, 1)
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

    The runner supplies moments collected with the official raw-character-span
    sampler and signed-global-maximum normalization. This function operates on
    already-injected attention-only modules, which is the intentional Track B
    target-scope difference.
    """

    dimensions: Dict[str, int] = {}
    low_bases: Dict[str, torch.Tensor] = {}
    count = 0
    for name, module in adapter_modules(model):
        group = attention_group_key(name)
        if group not in moments:
            raise KeyError(f"Missing LoRA-Null calibration moment for {group}")
        if group not in low_bases:
            covariance = moments[group].matrix.to(
                device="cpu",
                dtype=torch.float32,
            )
            left, _, _ = torch.linalg.svd(covariance, full_matrices=False)
            if module.rank > left.shape[1]:
                raise ValueError(
                    f"rank {module.rank} exceeds input dimension for {name}"
                )
            low_bases[group] = left[:, -module.rank :]
        low_basis = low_bases[group]
        weight = module.base_layer.weight.detach().to(
            device="cpu",
            dtype=torch.float32,
        )
        lora_A, lora_B = _scaled_factors_from_projected_weight(
            weight,
            low_basis,
            scaling=module.scaling,
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
            "moment_decomposition": "torch.linalg.svd_left_tail",
            "moment_decomposition_dtype": "float32",
            "unique_moment_decompositions": len(low_bases),
            "calibration_sampling": "official_raw_character_spans",
            "calibration_normalization": "abs(signed_global_max)",
            "dimensions": dimensions,
        },
    )


@torch.no_grad()
def initialize_milora(model: nn.Module) -> InitializationRecord:
    count = 0
    devices = set()
    for _, module in adapter_modules(model):
        weight = module.base_layer.weight.detach()
        devices.add(str(weight.device))
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
        details={
            "singular_values": "smallest",
            "target_scope": "attention_only",
            "svd_dtype": "float32",
            "svd_devices": sorted(devices),
            "official_commit": "c3c94693b26c800a96dba84a1fe92d7384b7c28d",
        },
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
    return get_peft_model(
        model,
        peft_config,
        autocast_adapter_dtype=True,
    )


def build_corda_baseline(
    model: nn.Module,
    config: AdapterConfig,
    *,
    run_calibration,
    cache_file: Optional[str] = None,
    covariance_file: Optional[str] = None,
    mode: str = "kpm",
) -> nn.Module:
    from peft import LoraConfig, TaskType, get_peft_model
    from peft.tuners.lora.config import CordaConfig
    from peft.tuners.lora.corda import preprocess_corda

    if mode not in {"kpm", "ipm"}:
        raise ValueError("CorDA mode must be 'kpm' or 'ipm'")
    corda_config = CordaConfig(
        cache_file=cache_file,
        covariance_file=covariance_file,
        corda_method=mode,
        verbose=True,
        use_float16_for_covariance=False,
        prune_temporary_fields=True,
    )
    peft_config = LoraConfig(
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        target_modules=list(config.target_modules),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        init_lora_weights="corda",
        corda_config=corda_config,
    )
    preprocess_corda(model, peft_config, run_model=run_calibration)
    return get_peft_model(
        model,
        peft_config,
        autocast_adapter_dtype=True,
    )
