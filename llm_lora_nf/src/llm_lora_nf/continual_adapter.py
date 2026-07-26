import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from .baselines import build_peft_baseline
from .calibration import SecondMomentState, merge_phase_moments
from .config import AdapterConfig
from .inject import adapter_modules, attention_group_key
from .layers import FilteredLoRALinear


CUMULATIVE_ADAPTER_FORMAT_VERSION = 1
CUMULATIVE_ADAPTER_WEIGHTS = "cumulative_adapter.pt"
CUMULATIVE_ADAPTER_MANIFEST = "cumulative_adapter_manifest.json"
SUPPORTED_CONTINUAL_BACKENDS = {"native_factor_stack", "peft_state_stack"}


@dataclass
class CumulativeAdapterState:
    method: str
    backend: str
    adapter_config: Dict[str, Any]
    branches: List[Dict[str, Any]] = field(default_factory=list)
    protection_state: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        method: str,
        backend: str,
        adapter_config: AdapterConfig,
    ) -> "CumulativeAdapterState":
        if backend not in SUPPORTED_CONTINUAL_BACKENDS:
            raise ValueError(f"Unsupported continual adapter backend: {backend}")
        if method != adapter_config.method:
            raise ValueError("Continual method must match AdapterConfig.method")
        return cls(
            method=method,
            backend=backend,
            adapter_config=adapter_config.to_dict(),
        )

    @property
    def tasks(self) -> List[str]:
        return [str(branch["task"]) for branch in self.branches]

    def add_branch(self, branch: Dict[str, Any]) -> None:
        task = str(branch.get("task", ""))
        if not task:
            raise ValueError("Continual adapter branch task must be non-empty")
        if task in self.tasks:
            raise ValueError(f"Duplicate continual adapter task: {task}")
        self.branches.append(branch)


def _cpu_clone(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu").clone()


def _factor_payload(module: FilteredLoRALinear) -> Dict[str, torch.Tensor]:
    if module.has_base_offset and module.use_filter:
        raise ValueError(
            "Filtered decomposition initialization is not a supported method"
        )
    return {
        "merge_A": _cpu_clone(module.effective_lora_A()),
        "merge_B": _cpu_clone(module.lora_B.weight),
        "subtract_A": _cpu_clone(module.base_offset_A),
        "subtract_B": _cpu_clone(module.base_offset_B),
    }


@torch.no_grad()
def capture_and_merge_native_task(
    model: nn.Module,
    state: CumulativeAdapterState,
    *,
    task: str,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    if state.backend != "native_factor_stack":
        raise ValueError("Native capture requires native_factor_stack state")
    modules = {
        name: _factor_payload(module)
        for name, module in adapter_modules(model)
    }
    if not modules:
        raise ValueError("No native adapter modules found")
    branch = {
        "task": task,
        "modules": modules,
        "metadata": dict(metadata or {}),
    }
    state.add_branch(branch)
    for _, module in adapter_modules(model):
        module.merge_and_reset()
    return branch


@torch.no_grad()
def capture_native_protection_state(
    model: nn.Module,
    *,
    moment_phases: Optional[Mapping[str, int]] = None,
    summaries: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    filters: Dict[str, Dict[str, Any]] = {}
    for name, module in adapter_modules(model):
        group = attention_group_key(name)
        if group in filters:
            continue
        filters[group] = {
            "protected_basis": _cpu_clone(module.filter.protected_basis),
            "leakage": float(module.filter.leakage.item()),
            "selected_dimension": int(module.filter.protected_basis.shape[1]),
            "moment_phases": (
                int(moment_phases[group])
                if moment_phases is not None and group in moment_phases
                else None
            ),
            "summary": (
                dict(summaries[group])
                if summaries is not None and group in summaries
                else {}
            ),
        }
    if not filters:
        raise ValueError("No native adapter filters found")
    return {"filters": filters}


def update_continual_moment_map(
    current: Mapping[str, SecondMomentState],
    task_moments: Mapping[str, SecondMomentState],
    *,
    mode: str,
) -> Dict[str, SecondMomentState]:
    if mode not in {"reference_fixed", "reference_plus_history"}:
        raise ValueError(f"Unsupported protection-state update mode: {mode}")
    if set(current) != set(task_moments):
        raise ValueError("Continual moment group set changed between phases")
    if mode == "reference_fixed":
        return {
            group: SecondMomentState(
                matrix=state.matrix.detach().clone(),
                observations=state.observations,
                phases=state.phases,
            )
            for group, state in current.items()
        }
    return {
        group: merge_phase_moments(current[group], task_moments[group])
        for group in current
    }


def _validate_state_config(
    state: CumulativeAdapterState,
    adapter_config: AdapterConfig,
) -> None:
    if state.method != adapter_config.method:
        raise ValueError("Cumulative adapter method mismatch")
    if state.adapter_config != adapter_config.to_dict():
        raise ValueError("Cumulative adapter configuration mismatch")


@torch.no_grad()
def replay_native_factor_stack(
    model: nn.Module,
    state: CumulativeAdapterState,
    *,
    adapter_config: AdapterConfig,
) -> nn.Module:
    if state.backend != "native_factor_stack":
        raise ValueError("Native replay requires native_factor_stack state")
    _validate_state_config(state, adapter_config)
    if any(True for _ in adapter_modules(model)):
        raise ValueError("Native factor replay requires an unwrapped base model")
    expected_modules: Optional[set] = None
    for branch in state.branches:
        branch_modules = branch["modules"]
        names = set(branch_modules)
        if expected_modules is None:
            expected_modules = names
        elif names != expected_modules:
            raise ValueError("Cumulative native module set changed between tasks")
        for name, factors in branch_modules.items():
            target = model.get_submodule(name)
            if not isinstance(target, nn.Linear):
                raise TypeError(f"Cumulative target is not nn.Linear: {name}")
            merge_A = factors["merge_A"].to(dtype=torch.float32)
            merge_B = factors["merge_B"].to(dtype=torch.float32)
            if (
                tuple(merge_A.shape)
                != (adapter_config.rank, target.in_features)
                or tuple(merge_B.shape)
                != (target.out_features, adapter_config.rank)
            ):
                raise ValueError(f"Cumulative factor shape mismatch: {name}")
            delta = merge_B @ merge_A
            subtract_A = factors["subtract_A"].to(dtype=torch.float32)
            subtract_B = factors["subtract_B"].to(dtype=torch.float32)
            if subtract_A.numel() or subtract_B.numel():
                if (
                    tuple(subtract_A.shape)
                    != (adapter_config.rank, target.in_features)
                    or tuple(subtract_B.shape)
                    != (target.out_features, adapter_config.rank)
                ):
                    raise ValueError(
                        f"Cumulative subtraction shape mismatch: {name}"
                    )
                delta.sub_(subtract_B @ subtract_A)
            target.weight.add_(
                (adapter_config.scaling * delta).to(
                    device=target.weight.device,
                    dtype=target.weight.dtype,
                )
            )
    return model


def capture_and_merge_peft_task(
    model: nn.Module,
    state: CumulativeAdapterState,
    *,
    task: str,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    if state.backend != "peft_state_stack":
        raise ValueError("PEFT capture requires peft_state_stack state")
    from peft import get_peft_model_state_dict

    adapter_state = {
        name: _cpu_clone(tensor)
        for name, tensor in get_peft_model_state_dict(model).items()
    }
    if not adapter_state:
        raise ValueError("No PEFT adapter parameters found")
    branch = {
        "task": task,
        "modules": adapter_state,
        "metadata": dict(metadata or {}),
    }
    state.add_branch(branch)
    base_model = model.merge_and_unload(safe_merge=True)
    # PEFT 0.17.1 leaves this adapter-only attribute on the plain base model
    # returned by merge_and_unload. Re-wrapping that base for the next
    # continual task is valid, but get_peft_model otherwise emits a misleading
    # "multiple adapters" warning. The replay state above is the authoritative
    # adapter record, so stale wrapper metadata must not leak across tasks.
    if hasattr(base_model, "peft_config"):
        delattr(base_model, "peft_config")
    return base_model, branch


def replay_peft_state_stack(
    model: nn.Module,
    state: CumulativeAdapterState,
    *,
    adapter_config: AdapterConfig,
) -> nn.Module:
    if state.backend != "peft_state_stack":
        raise ValueError("PEFT replay requires peft_state_stack state")
    _validate_state_config(state, adapter_config)
    from peft import set_peft_model_state_dict

    current = model
    for branch in state.branches:
        wrapped = build_peft_baseline(
            current,
            adapter_config,
            state.method,
        )
        load_result = set_peft_model_state_dict(
            wrapped,
            branch["modules"],
        )
        unexpected = list(getattr(load_result, "unexpected_keys", []))
        if unexpected:
            raise ValueError(
                "Unexpected PEFT state keys during cumulative replay: "
                + ", ".join(unexpected[:20])
            )
        current = wrapped.merge_and_unload(safe_merge=True)
        if hasattr(current, "peft_config"):
            delattr(current, "peft_config")
    return current


def save_cumulative_adapter(
    state: CumulativeAdapterState,
    output_dir: str,
    *,
    metadata: Mapping[str, Any],
) -> Tuple[Path, Path]:
    if not state.branches:
        raise ValueError("Cannot save an empty cumulative adapter")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    weights_path = destination / CUMULATIVE_ADAPTER_WEIGHTS
    manifest_path = destination / CUMULATIVE_ADAPTER_MANIFEST
    torch.save(
        {
            "format_version": CUMULATIVE_ADAPTER_FORMAT_VERSION,
            "method": state.method,
            "backend": state.backend,
            "adapter_config": state.adapter_config,
            "branches": state.branches,
            "protection_state": state.protection_state,
        },
        weights_path,
    )
    module_names = sorted(state.branches[0]["modules"])
    manifest = {
        "format_version": CUMULATIVE_ADAPTER_FORMAT_VERSION,
        "method": state.method,
        "backend": state.backend,
        "adapter_config": state.adapter_config,
        "tasks": state.tasks,
        "branch_count": len(state.branches),
        "module_names": module_names,
        "checkpoint_semantics": (
            "one_final_replayable_adapter_no_intermediate_model_checkpoints"
        ),
        "metadata": dict(metadata),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return weights_path, manifest_path


def load_cumulative_adapter(
    checkpoint_dir: str,
) -> Tuple[CumulativeAdapterState, Dict[str, Any]]:
    source = Path(checkpoint_dir)
    manifest = json.loads(
        (source / CUMULATIVE_ADAPTER_MANIFEST).read_text(encoding="utf-8")
    )
    payload = torch.load(
        source / CUMULATIVE_ADAPTER_WEIGHTS,
        map_location="cpu",
        weights_only=True,
    )
    if (
        manifest.get("format_version") != CUMULATIVE_ADAPTER_FORMAT_VERSION
        or payload.get("format_version") != CUMULATIVE_ADAPTER_FORMAT_VERSION
    ):
        raise ValueError("Unsupported cumulative adapter checkpoint format")
    for key in ("method", "backend", "adapter_config"):
        if manifest.get(key) != payload.get(key):
            raise ValueError(f"Cumulative adapter manifest mismatch: {key}")
    if payload["backend"] not in SUPPORTED_CONTINUAL_BACKENDS:
        raise ValueError("Unsupported cumulative adapter backend")
    state = CumulativeAdapterState(
        method=payload["method"],
        backend=payload["backend"],
        adapter_config=payload["adapter_config"],
        branches=payload["branches"],
        protection_state=payload["protection_state"],
    )
    if (
        manifest.get("tasks") != state.tasks
        or manifest.get("branch_count") != len(state.branches)
        or manifest.get("module_names")
        != sorted(state.branches[0]["modules"])
    ):
        raise ValueError("Cumulative adapter branch manifest mismatch")
    return state, manifest
