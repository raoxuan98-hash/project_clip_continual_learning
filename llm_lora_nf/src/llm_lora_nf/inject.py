import hashlib
import json
from typing import Any, Dict, Iterable, Iterator, Tuple

import torch
import torch.nn as nn

from .config import AdapterConfig
from .filters import HardLeakyFilter
from .layers import FilteredLoRALinear


def attention_group_key(module_name: str) -> str:
    prefix, leaf = module_name.rsplit(".", 1)
    if leaf in {"q_proj", "k_proj", "v_proj"}:
        return f"{prefix}.qkv_shared"
    if leaf == "o_proj":
        return module_name
    raise ValueError(f"Not an attention projection: {module_name}")


def _parent_and_leaf(model: nn.Module, module_name: str) -> Tuple[nn.Module, str]:
    if "." not in module_name:
        return model, module_name
    parent_name, leaf = module_name.rsplit(".", 1)
    return model.get_submodule(parent_name), leaf


def inject_attention_adapters(
    model: nn.Module,
    config: AdapterConfig,
) -> Dict[str, FilteredLoRALinear]:
    """Freeze the model and replace only configured attention projections."""

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    target_set = set(config.target_modules)
    candidates = [
        (name, module)
        for name, module in model.named_modules()
        if name
        and name.rsplit(".", 1)[-1] in target_set
        and isinstance(module, nn.Linear)
    ]
    if not candidates:
        raise ValueError(
            "No attention projection modules matched "
            + ", ".join(sorted(target_set))
        )

    wrapped: Dict[str, FilteredLoRALinear] = {}
    for name, module in candidates:
        parent, leaf = _parent_and_leaf(model, name)
        adapter = FilteredLoRALinear(
            module,
            rank=config.rank,
            alpha=config.alpha,
            dropout=config.dropout,
            use_filter=config.method == "lora_nf",
        )
        setattr(parent, leaf, adapter)
        wrapped[name] = adapter
    return wrapped


def adapter_modules(model: nn.Module) -> Iterator[Tuple[str, FilteredLoRALinear]]:
    for name, module in model.named_modules():
        if isinstance(module, FilteredLoRALinear):
            yield name, module


def assign_group_filters(
    model: nn.Module,
    group_filters: Dict[str, HardLeakyFilter],
) -> None:
    missing = []
    prepared: Dict[str, HardLeakyFilter] = {}
    for name, module in adapter_modules(model):
        key = attention_group_key(name)
        filter_module = group_filters.get(key)
        if filter_module is None:
            missing.append(key)
            continue
        if key not in prepared:
            filter_module.to(
                device=module.base_layer.weight.device,
                dtype=module.base_layer.weight.dtype,
            )
            filter_module.configure_runtime_reuse(
                3 if key.endswith(".qkv_shared") else 1
            )
            prepared[key] = filter_module
        module.share_filter(prepared[key])
    if missing:
        raise KeyError("Missing filters for groups: " + ", ".join(sorted(set(missing))))


@torch.no_grad()
def merge_native_adapters(model: nn.Module) -> Dict[str, nn.Linear]:
    """Merge native adapters and restore standard ``nn.Linear`` modules."""

    merged: Dict[str, nn.Linear] = {}
    for name, module in list(adapter_modules(model)):
        module.merge()
        parent, leaf = _parent_and_leaf(model, name)
        setattr(parent, leaf, module.base_layer)
        merged[name] = module.base_layer
    if not merged:
        raise ValueError("No native adapter modules found")
    return merged


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {"total": total, "trainable": trainable}


def audit_trainable_parameter_scope(
    model: nn.Module,
    *,
    target_modules: Iterable[str],
) -> Dict[str, Any]:
    """Seal the exact trainable scope shared by native and PEFT adapters."""

    expected_targets = tuple(str(target) for target in target_modules)
    if not expected_targets or len(set(expected_targets)) != len(expected_targets):
        raise ValueError("target_modules must be a non-empty unique sequence")
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable:
        raise RuntimeError("Adapter model has no trainable parameters")

    names = sorted(name for name, _ in trainable)
    observed_targets = sorted(
        {
            target
            for name in names
            for target in expected_targets
            if f".{target}." in f".{name}."
        }
    )
    unexpected_names = [
        name
        for name in names
        if not any(
            f".{target}." in f".{name}."
            for target in expected_targets
        )
    ]
    missing_targets = sorted(set(expected_targets) - set(observed_targets))
    if unexpected_names or missing_targets:
        raise RuntimeError(
            "Trainable adapter scope mismatch: "
            f"unexpected={unexpected_names}, missing_targets={missing_targets}"
        )

    dtype_elements: Dict[str, int] = {}
    dtype_tensors: Dict[str, int] = {}
    for _, parameter in trainable:
        dtype = str(parameter.dtype)
        dtype_elements[dtype] = dtype_elements.get(dtype, 0) + parameter.numel()
        dtype_tensors[dtype] = dtype_tensors.get(dtype, 0) + 1
    encoded_names = json.dumps(
        names,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "target_modules": list(expected_targets),
        "observed_target_modules": observed_targets,
        "trainable_tensor_count": len(trainable),
        "trainable_parameter_count": sum(
            parameter.numel() for _, parameter in trainable
        ),
        "trainable_dtype_elements": dict(sorted(dtype_elements.items())),
        "trainable_dtype_tensors": dict(sorted(dtype_tensors.items())),
        "trainable_parameter_names_sha256": hashlib.sha256(
            encoded_names
        ).hexdigest(),
    }


@torch.no_grad()
def audit_finite_trainable_parameters(model: nn.Module) -> Dict[str, Any]:
    """Reject a checkpoint if any trainable tensor contains NaN or infinity."""

    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable:
        raise RuntimeError("Adapter model has no trainable parameters")
    non_finite = [
        name
        for name, parameter in trainable
        if not bool(torch.isfinite(parameter.detach()).all().item())
    ]
    if non_finite:
        raise FloatingPointError(
            "Non-finite trainable parameters after training: "
            + ", ".join(non_finite)
        )
    return {
        "all_finite": True,
        "checked_tensor_count": len(trainable),
        "checked_parameter_count": sum(
            parameter.numel() for _, parameter in trainable
        ),
    }
