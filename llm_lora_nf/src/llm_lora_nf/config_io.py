import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

from .config import AdapterConfig, FilterConfig


def _deep_merge(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, Mapping)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml_config(path: str) -> Dict[str, Any]:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {config_path}")
    parent = payload.pop("extends", None)
    if parent is not None:
        parent_path = (config_path.parent / str(parent)).resolve()
        payload = _deep_merge(load_yaml_config(str(parent_path)), payload)
    return payload


def canonical_config_hash(config: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_comparison_config_hash(config: Mapping[str, Any]) -> str:
    """Hash the training fields that must be identical across methods.

    Only run identity/seed and explicitly permitted method selectors are
    removed. Any other accidental method-specific override changes this hash
    and therefore blocks paired aggregation.
    """

    normalized = copy.deepcopy(dict(config))
    run = normalized.get("run")
    if isinstance(run, dict):
        run.pop("seed", None)
        run.pop("name", None)
    adapter = normalized.get("adapter")
    if isinstance(adapter, dict):
        adapter.pop("method", None)
        adapter.pop("corda_mode", None)
    return canonical_config_hash(normalized)


def adapter_config_from_mapping(payload: Mapping[str, Any]) -> AdapterConfig:
    filter_payload = dict(payload.get("filter", {}))
    adapter_payload = {
        key: value
        for key, value in payload.items()
        if key in {"method", "rank", "alpha", "dropout", "target_modules"}
    }
    adapter_payload["filter"] = FilterConfig(**filter_payload)
    return AdapterConfig(**adapter_payload)
