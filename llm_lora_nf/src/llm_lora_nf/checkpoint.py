import json
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import torch
import torch.nn as nn

from .config import AdapterConfig
from .filters import HardLeakyFilter
from .inject import (
    adapter_modules,
    assign_group_filters,
    attention_group_key,
    inject_attention_adapters,
)
from .layers import FilteredLoRALinear


CHECKPOINT_FORMAT_VERSION = 2
SUPPORTED_CHECKPOINT_FORMATS = {1, 2}


def _module_payload(module: FilteredLoRALinear) -> Dict[str, torch.Tensor]:
    return {
        "lora_A": module.lora_A.weight.detach().cpu(),
        "lora_B": module.lora_B.weight.detach().cpu(),
        "base_offset_A": module.base_offset_A.detach().cpu(),
        "base_offset_B": module.base_offset_B.detach().cpu(),
    }


def save_native_adapter(
    model: nn.Module,
    output_dir: str,
    *,
    adapter_config: AdapterConfig,
    metadata: Mapping[str, Any],
) -> Tuple[Path, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    modules = {
        name: _module_payload(module)
        for name, module in adapter_modules(model)
    }
    if not modules:
        raise ValueError("No native adapter modules found")
    filters: Dict[str, Dict[str, torch.Tensor]] = {}
    for name, module in adapter_modules(model):
        group = attention_group_key(name)
        if group not in filters:
            filters[group] = {
                "protected_basis": module.filter.protected_basis.detach().cpu(),
                "leakage": module.filter.leakage.detach().cpu(),
            }
    weights_path = destination / "adapter.pt"
    manifest_path = destination / "adapter_manifest.json"
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "modules": modules,
            "filters": filters,
        },
        weights_path,
    )
    manifest = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "adapter_config": adapter_config.to_dict(),
        "module_names": sorted(modules),
        "filter_groups": sorted(filters),
        "metadata": dict(metadata),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return weights_path, manifest_path


@torch.no_grad()
def load_native_adapter(
    model: nn.Module,
    checkpoint_dir: str,
    *,
    adapter_config: AdapterConfig,
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
    format_version = payload.get("format_version")
    if format_version not in SUPPORTED_CHECKPOINT_FORMATS:
        raise ValueError("Unsupported native adapter checkpoint format")
    wrapped = inject_attention_adapters(model, adapter_config)
    saved_modules = payload["modules"]
    if set(wrapped) != set(saved_modules):
        raise ValueError("Checkpoint module set does not match model target modules")

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
        if format_version == 1:
            module.filter.set_basis(
                saved["protected_basis"].to(
                    device=module.base_layer.weight.device,
                    dtype=module.base_layer.weight.dtype,
                ),
                leakage=float(saved["leakage"].item()),
            )
    if format_version == 2:
        filters = {
            group: HardLeakyFilter(
                input_dim=int(saved["protected_basis"].shape[0]),
                protected_basis=saved["protected_basis"],
                leakage=float(saved["leakage"].item()),
            )
            for group, saved in payload["filters"].items()
        }
        assign_group_filters(model, filters)
    return manifest
