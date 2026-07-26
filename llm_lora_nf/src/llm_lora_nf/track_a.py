"""Isolated LoRA-Null Track A reproduction primitives.

Track A follows the public LoRA-Null implementation where every decoder
projection linear is adapted. It is intentionally separate from the
attention-only :class:`AdapterConfig` used by the controlled Track B study.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from .baselines import (
    InitializationRecord,
    _scaled_factors_from_projected_weight,
)
from .calibration import SecondMomentState
from .inject import attention_group_key
from .layers import FilteredLoRALinear


TRACK_A_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
TRACK_A_OFFICIAL_COMMIT = "1e6808abb81fe10e50b8172c40ac9a8ab4f11e83"
TRACK_A_CHECKPOINT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class TrackAAdapterConfig:
    method: str
    rank: int = 128
    alpha: float = 128.0
    dropout: float = 0.0
    target_modules: Tuple[str, ...] = TRACK_A_TARGETS

    def __post_init__(self) -> None:
        if self.method not in {"lora", "lora_null"}:
            raise ValueError("Track A supports only LoRA and LoRA-Null")
        if self.rank <= 0:
            raise ValueError("rank must be positive")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if self.dropout != 0.0:
            raise ValueError("The locked LoRA-Null Track A protocol uses dropout=0")
        if tuple(self.target_modules) != TRACK_A_TARGETS:
            raise ValueError(
                "Track A target_modules must exactly match the official seven "
                "decoder projections"
            )

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["target_modules"] = list(self.target_modules)
        return payload


def track_a_config_from_mapping(payload: Mapping[str, Any]) -> TrackAAdapterConfig:
    return TrackAAdapterConfig(
        method=str(payload["method"]),
        rank=int(payload["rank"]),
        alpha=float(payload["alpha"]),
        dropout=float(payload.get("dropout", 0.0)),
        target_modules=tuple(payload.get("target_modules", TRACK_A_TARGETS)),
    )


def _parent_and_leaf(model: nn.Module, module_name: str) -> Tuple[nn.Module, str]:
    parent_name, leaf = module_name.rsplit(".", 1)
    return model.get_submodule(parent_name), leaf


def track_a_linear_candidates(model: nn.Module) -> Dict[str, nn.Linear]:
    """Resolve the official all-linear-except-lm-head scope safely.

    Llama and Qwen decoder blocks expose exactly the seven locked projection
    leaf names. Rejecting unexpected linear modules prevents a new architecture
    from silently changing the Track A parameter budget.
    """

    candidates = {
        name: module
        for name, module in model.named_modules()
        if name and "lm_head" not in name and isinstance(module, nn.Linear)
    }
    if not candidates:
        raise ValueError("No non-lm_head Linear modules found for Track A")
    leaves = {name.rsplit(".", 1)[-1] for name in candidates}
    unexpected = leaves - set(TRACK_A_TARGETS)
    missing = set(TRACK_A_TARGETS) - leaves
    if unexpected or missing:
        details = []
        if unexpected:
            details.append("unexpected=" + ",".join(sorted(unexpected)))
        if missing:
            details.append("missing=" + ",".join(sorted(missing)))
        raise ValueError("Unsupported Track A model layout: " + "; ".join(details))
    return candidates


def track_a_adapter_modules(
    model: nn.Module,
) -> Iterator[Tuple[str, FilteredLoRALinear]]:
    for name, module in model.named_modules():
        if isinstance(module, FilteredLoRALinear):
            yield name, module


def track_a_moment_group_key(module_name: str) -> str:
    prefix, leaf = module_name.rsplit(".", 1)
    if leaf in {"q_proj", "k_proj", "v_proj"}:
        return f"{prefix}.qkv_shared"
    if leaf in {"gate_proj", "up_proj"}:
        return f"{prefix}.gate_up_shared"
    if leaf in {"o_proj", "down_proj"}:
        return module_name
    raise ValueError(f"Not a locked Track A projection: {module_name}")


def inject_track_a_native_adapters(
    model: nn.Module,
    config: TrackAAdapterConfig,
) -> Dict[str, FilteredLoRALinear]:
    """Freeze a model and wrap the official seven decoder projections."""

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    candidates = track_a_linear_candidates(model)
    wrapped: Dict[str, FilteredLoRALinear] = {}
    for name, module in candidates.items():
        parent, leaf = _parent_and_leaf(model, name)
        adapter = FilteredLoRALinear(
            module,
            rank=config.rank,
            alpha=config.alpha,
            dropout=config.dropout,
            use_filter=False,
        )
        setattr(parent, leaf, adapter)
        wrapped[name] = adapter
    return wrapped


class OfficialLoRANullMomentAccumulator:
    """Exact relevant second-moment arithmetic from the public repository."""

    def __init__(self, input_dim: int, expected_samples: int) -> None:
        if expected_samples <= 0:
            raise ValueError("expected_samples must be positive")
        self.input_dim = int(input_dim)
        self.expected_samples = int(expected_samples)
        self.matrix: Optional[torch.Tensor] = None
        self.samples = 0

    @torch.no_grad()
    def update(self, activations: torch.Tensor) -> None:
        if activations.ndim != 3 or activations.shape[0] != 1:
            raise ValueError(
                "Official LoRA-Null calibration requires batches shaped "
                "(1, sequence, hidden)"
            )
        if activations.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected activation dim {self.input_dim}, "
                f"got {activations.shape[-1]}"
            )
        values = activations.detach().squeeze(0)
        # This deliberately reproduces ``torch.max(input).abs()`` from commit
        # 1e6808a. It is not equivalent to ``input.abs().max()``.
        denominator = torch.max(values).abs()
        normalized = values / denominator
        if not torch.isfinite(normalized).all():
            raise FloatingPointError(
                "Official LoRA-Null signed-maximum normalization produced "
                "non-finite activations"
            )
        covariance = normalized.transpose(0, 1) @ normalized
        if not torch.isfinite(covariance).all():
            raise FloatingPointError(
                "Official LoRA-Null calibration produced a non-finite moment"
            )
        contribution = covariance / self.expected_samples
        if self.matrix is None:
            self.matrix = contribution
        else:
            self.matrix.add_(contribution)
        self.samples += 1

    def finalize(self) -> SecondMomentState:
        if self.samples != self.expected_samples:
            raise RuntimeError(
                f"Expected {self.expected_samples} calibration samples, "
                f"observed {self.samples}"
            )
        if self.matrix is None:
            raise RuntimeError("No Track A moments were accumulated")
        return SecondMomentState(
            matrix=self.matrix.to(device="cpu"),
            observations=self.samples,
        )


class OfficialLoRANullCalibrator:
    """Collect per-linear Track A moments without mutating the base model."""

    def __init__(self, model: nn.Module, *, expected_samples: int) -> None:
        self.model = model
        self.candidates = track_a_linear_candidates(model)
        self.group_representatives: Dict[str, Tuple[str, nn.Linear]] = {}
        for name, module in self.candidates.items():
            group = track_a_moment_group_key(name)
            if group in self.group_representatives:
                representative = self.group_representatives[group][1]
                if representative.in_features != module.in_features:
                    raise ValueError(
                        f"Track A shared-input group {group} has inconsistent "
                        "input dimensions"
                    )
            else:
                self.group_representatives[group] = (name, module)
        self.accumulators = {
            group: OfficialLoRANullMomentAccumulator(
                module.in_features,
                expected_samples,
            )
            for group, (_, module) in self.group_representatives.items()
        }

    @torch.no_grad()
    def collect(
        self,
        batches: Iterable[Mapping[str, torch.Tensor]],
        *,
        device: torch.device,
    ) -> Dict[str, SecondMomentState]:
        handles = []
        for group, (_, module) in self.group_representatives.items():
            accumulator = self.accumulators[group]

            def hook(_module, inputs, _accumulator=accumulator):
                del _module
                _accumulator.update(inputs[0])

            handles.append(module.register_forward_pre_hook(hook))
        was_training = self.model.training
        self.model.eval()
        seen = 0
        try:
            for batch in batches:
                moved = {
                    key: value.to(device) if torch.is_tensor(value) else value
                    for key, value in batch.items()
                    if key != "labels"
                }
                if int(moved["input_ids"].shape[0]) != 1:
                    raise ValueError(
                        "Official LoRA-Null calibration requires batch_size=1"
                    )
                self.model(**moved)
                seen += 1
        finally:
            for handle in handles:
                handle.remove()
            self.model.train(was_training)
        if seen == 0:
            raise ValueError("Calibration iterable produced no batches")
        grouped = {
            group: accumulator.finalize()
            for group, accumulator in self.accumulators.items()
        }
        return {
            name: grouped[track_a_moment_group_key(name)]
            for name in self.candidates
        }


class OfficialLoRANullAttentionCalibrator:
    """Official arithmetic restricted to the controlled Track B scope."""

    def __init__(self, model: nn.Module, *, expected_samples: int) -> None:
        self.model = model
        self.candidates: Dict[str, Tuple[str, nn.Linear]] = {}
        for name, module in model.named_modules():
            leaf = name.rsplit(".", 1)[-1]
            if leaf not in {"q_proj", "k_proj", "v_proj", "o_proj"}:
                continue
            if not isinstance(module, nn.Linear):
                continue
            group = attention_group_key(name)
            if group not in self.candidates:
                self.candidates[group] = (name, module)
        if not self.candidates:
            raise ValueError("No attention projections found for LoRA-Null")
        self.accumulators = {
            group: OfficialLoRANullMomentAccumulator(
                module.in_features,
                expected_samples,
            )
            for group, (_, module) in self.candidates.items()
        }

    @torch.no_grad()
    def collect(
        self,
        batches: Iterable[Mapping[str, torch.Tensor]],
        *,
        device: torch.device,
    ) -> Dict[str, SecondMomentState]:
        handles = []
        for group, (_, module) in self.candidates.items():
            accumulator = self.accumulators[group]

            def hook(_module, inputs, _accumulator=accumulator):
                del _module
                _accumulator.update(inputs[0])

            handles.append(module.register_forward_pre_hook(hook))
        was_training = self.model.training
        self.model.eval()
        seen = 0
        try:
            for batch in batches:
                moved = {
                    key: value.to(device) if torch.is_tensor(value) else value
                    for key, value in batch.items()
                    if key != "labels"
                }
                if int(moved["input_ids"].shape[0]) != 1:
                    raise ValueError(
                        "Official LoRA-Null calibration requires batch_size=1"
                    )
                self.model(**moved)
                seen += 1
        finally:
            for handle in handles:
                handle.remove()
            self.model.train(was_training)
        if seen == 0:
            raise ValueError("Calibration iterable produced no batches")
        return {
            group: accumulator.finalize()
            for group, accumulator in self.accumulators.items()
        }


@torch.no_grad()
def initialize_track_a_lora_null(
    model: nn.Module,
    moments: Mapping[str, SecondMomentState],
) -> InitializationRecord:
    """Install the official low-activation-subspace decomposition."""

    count = 0
    low_bases: Dict[str, torch.Tensor] = {}
    for name, module in track_a_adapter_modules(model):
        if name not in moments:
            raise KeyError(f"Missing Track A LoRA-Null moment for {name}")
        group = track_a_moment_group_key(name)
        if group not in low_bases:
            covariance = moments[name].matrix.to(
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
        count += 1
    return InitializationRecord(
        method="lora_null",
        module_count=count,
        exact_function_preserving=True,
        calibration_required=True,
        details={
            "protocol_track": "A",
            "official_commit": TRACK_A_OFFICIAL_COMMIT,
            "target_scope": "all_decoder_linears",
            "normalization": "abs(signed_global_max)",
            "sample_divisor": "declared_calibration_samples",
            "subspace": "lowest_activation_energy",
            "moment_decomposition": "torch.linalg.svd_left_tail",
            "moment_decomposition_dtype": "float32",
            "unique_moment_decompositions": len(low_bases),
        },
    )


def build_track_a_native_adapter(
    model: nn.Module,
    config: TrackAAdapterConfig,
    *,
    calibration_moments: Optional[Mapping[str, SecondMomentState]] = None,
) -> Tuple[Dict[str, FilteredLoRALinear], InitializationRecord]:
    wrapped = inject_track_a_native_adapters(model, config)
    if config.method == "lora_null":
        if calibration_moments is None:
            raise ValueError("Track A LoRA-Null requires calibration moments")
        record = initialize_track_a_lora_null(model, calibration_moments)
    else:
        record = InitializationRecord(
            method="lora",
            module_count=len(wrapped),
            exact_function_preserving=True,
            calibration_required=False,
            details={
                "protocol_track": "A",
                "official_commit": TRACK_A_OFFICIAL_COMMIT,
                "target_scope": "all_decoder_linears",
                "initialization": "standard_lora",
            },
        )
    return wrapped, record


def build_track_a_peft_lora(
    model: nn.Module,
    config: TrackAAdapterConfig,
) -> nn.Module:
    """Build the official PEFT LoRA comparator for dropout zero."""

    if config.method != "lora":
        raise ValueError("The PEFT Track A builder is only for LoRA")
    # Validate the architecture before PEFT replaces the linear modules.
    track_a_linear_candidates(model)
    from peft import LoraConfig, TaskType, get_peft_model

    peft_config = LoraConfig(
        r=config.rank,
        lora_alpha=config.alpha,
        init_lora_weights=True,
        target_modules=list(config.target_modules),
        lora_dropout=config.dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(model, peft_config)


def _module_payload(module: FilteredLoRALinear) -> Dict[str, torch.Tensor]:
    return {
        "lora_A": module.lora_A.weight.detach().cpu(),
        "lora_B": module.lora_B.weight.detach().cpu(),
        "base_offset_A": module.base_offset_A.detach().cpu(),
        "base_offset_B": module.base_offset_B.detach().cpu(),
    }


def save_track_a_native_adapter(
    model: nn.Module,
    output_dir: str,
    *,
    adapter_config: TrackAAdapterConfig,
    metadata: Mapping[str, Any],
) -> Tuple[Path, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    modules = {
        name: _module_payload(module)
        for name, module in track_a_adapter_modules(model)
    }
    if not modules:
        raise ValueError("No Track A native adapter modules found")
    weights_path = destination / "adapter.pt"
    manifest_path = destination / "adapter_manifest.json"
    torch.save(
        {
            "format_version": TRACK_A_CHECKPOINT_FORMAT_VERSION,
            "protocol_track": "A",
            "modules": modules,
        },
        weights_path,
    )
    manifest = {
        "format_version": TRACK_A_CHECKPOINT_FORMAT_VERSION,
        "protocol_track": "A",
        "official_commit": TRACK_A_OFFICIAL_COMMIT,
        "adapter_config": adapter_config.to_dict(),
        "module_names": sorted(modules),
        "metadata": dict(metadata),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return weights_path, manifest_path


@torch.no_grad()
def load_track_a_native_adapter(
    model: nn.Module,
    checkpoint_dir: str,
    *,
    adapter_config: TrackAAdapterConfig,
) -> Dict[str, Any]:
    source = Path(checkpoint_dir)
    manifest = json.loads(
        (source / "adapter_manifest.json").read_text(encoding="utf-8")
    )
    payload = torch.load(
        source / "adapter.pt",
        map_location="cpu",
        weights_only=True,
    )
    if (
        payload.get("format_version") != TRACK_A_CHECKPOINT_FORMAT_VERSION
        or payload.get("protocol_track") != "A"
    ):
        raise ValueError("Unsupported Track A native adapter checkpoint")
    wrapped = inject_track_a_native_adapters(model, adapter_config)
    saved_modules = payload["modules"]
    if set(wrapped) != set(saved_modules):
        raise ValueError("Track A checkpoint module set does not match the model")
    for name, module in wrapped.items():
        saved = saved_modules[name]
        offset_A = saved["base_offset_A"]
        offset_B = saved["base_offset_B"]
        if offset_A.shape[0] > 0:
            module.restore_base_offset(offset_A, offset_B)
        module.lora_A.weight.copy_(
            saved["lora_A"].to(
                device=module.lora_A.weight.device,
                dtype=module.lora_A.weight.dtype,
            )
        )
        module.lora_B.weight.copy_(
            saved["lora_B"].to(
                device=module.lora_B.weight.device,
                dtype=module.lora_B.weight.dtype,
            )
        )
    return manifest
