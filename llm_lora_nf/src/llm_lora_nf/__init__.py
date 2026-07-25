"""Attention-only LoRA-NF for LLM adaptation."""

from .config import AdapterConfig, FilterConfig
from .filters import HardLeakyFilter, ProjectionBuildResult, build_hard_leaky_filter
from .inject import (
    adapter_modules,
    attention_group_key,
    count_parameters,
    inject_attention_adapters,
)
from .layers import FilteredLoRALinear

__all__ = [
    "AdapterConfig",
    "FilterConfig",
    "FilteredLoRALinear",
    "HardLeakyFilter",
    "ProjectionBuildResult",
    "adapter_modules",
    "attention_group_key",
    "build_hard_leaky_filter",
    "count_parameters",
    "inject_attention_adapters",
]

