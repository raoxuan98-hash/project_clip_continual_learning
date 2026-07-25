from typing import Dict, Iterable, Iterator, Tuple

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
    for name, module in adapter_modules(model):
        key = attention_group_key(name)
        filter_module = group_filters.get(key)
        if filter_module is None:
            missing.append(key)
            continue
        module.set_filter(filter_module)
    if missing:
        raise KeyError("Missing filters for groups: " + ", ".join(sorted(set(missing))))


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {"total": total, "trainable": trainable}
