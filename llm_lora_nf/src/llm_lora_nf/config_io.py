import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

from .config import AdapterConfig, FilterConfig


def load_yaml_config(path: str) -> Dict[str, Any]:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {config_path}")
    return payload


def canonical_config_hash(config: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def adapter_config_from_mapping(payload: Mapping[str, Any]) -> AdapterConfig:
    filter_payload = dict(payload.get("filter", {}))
    adapter_payload = {
        key: value
        for key, value in payload.items()
        if key in {"method", "rank", "alpha", "dropout", "target_modules"}
    }
    adapter_payload["filter"] = FilterConfig(**filter_payload)
    return AdapterConfig(**adapter_payload)
