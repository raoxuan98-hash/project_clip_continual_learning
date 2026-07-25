from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from .filters import ProjectionBuildResult, build_hard_leaky_filter
from .inject import assign_group_filters, attention_group_key
from .layers import FilteredLoRALinear


@dataclass
class SecondMomentState:
    matrix: torch.Tensor
    observations: int
    phases: int = 1


class SecondMomentAccumulator:
    def __init__(self, input_dim: int) -> None:
        self.input_dim = int(input_dim)
        self.sum_xtx = torch.zeros(input_dim, input_dim, dtype=torch.float32)
        self.observations = 0

    @torch.no_grad()
    def update(
        self,
        activations: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> None:
        if activations.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected activation dim {self.input_dim}, got {activations.shape[-1]}"
            )
        values = activations.detach()
        if values.ndim == 2:
            flat = values
        elif values.ndim == 3:
            if attention_mask is None:
                flat = values.reshape(-1, values.shape[-1])
            else:
                mask = attention_mask.to(device=values.device, dtype=torch.bool)
                if tuple(mask.shape) != tuple(values.shape[:2]):
                    raise ValueError(
                        f"Mask shape {tuple(mask.shape)} does not match "
                        f"activations {tuple(values.shape[:2])}"
                    )
                flat = values[mask]
        else:
            raise ValueError("Expected 2D or 3D activations")
        if flat.numel() == 0:
            return
        flat_cpu = flat.to(device="cpu", dtype=torch.float32)
        self.sum_xtx.add_(flat_cpu.transpose(0, 1) @ flat_cpu)
        self.observations += int(flat_cpu.shape[0])

    def finalize(self, ridge: float = 0.0) -> SecondMomentState:
        if self.observations <= 0:
            raise RuntimeError("No activations were accumulated")
        matrix = self.sum_xtx / self.observations
        matrix = 0.5 * (matrix + matrix.transpose(0, 1))
        if ridge:
            matrix = matrix + ridge * torch.eye(self.input_dim, dtype=matrix.dtype)
        return SecondMomentState(matrix=matrix, observations=self.observations)


def _projection_candidates(model: nn.Module) -> "OrderedDict[str, Tuple[str, nn.Module]]":
    groups: "OrderedDict[str, Tuple[str, nn.Module]]" = OrderedDict()
    for name, module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in {"q_proj", "k_proj", "v_proj", "o_proj"}:
            continue
        if not isinstance(module, (nn.Linear, FilteredLoRALinear)):
            continue
        group = attention_group_key(name)
        if group not in groups:
            input_dim = (
                module.in_features
                if isinstance(module, nn.Linear)
                else module.in_features
            )
            groups[group] = (name, module)
    if not groups:
        raise ValueError("No attention projections found for calibration")
    return groups


class ActivationCalibrator:
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.candidates = _projection_candidates(model)
        self.accumulators: Dict[str, SecondMomentAccumulator] = {}
        self._attention_mask: Optional[torch.Tensor] = None
        for group, (_, module) in self.candidates.items():
            self.accumulators[group] = SecondMomentAccumulator(module.in_features)

    @torch.no_grad()
    def collect(
        self,
        batches: Iterable[Mapping[str, torch.Tensor]],
        *,
        device: torch.device,
        max_batches: Optional[int] = None,
    ) -> Dict[str, SecondMomentState]:
        handles = []
        for group, (_, module) in self.candidates.items():
            accumulator = self.accumulators[group]

            def hook(_module, inputs, _group=group, _accumulator=accumulator):
                del _module, _group
                _accumulator.update(inputs[0], self._attention_mask)

            handles.append(module.register_forward_pre_hook(hook))

        was_training = self.model.training
        self.model.eval()
        try:
            for batch_index, batch in enumerate(batches):
                if max_batches is not None and batch_index >= max_batches:
                    break
                moved = {
                    key: value.to(device) if torch.is_tensor(value) else value
                    for key, value in batch.items()
                    if key != "labels"
                }
                self._attention_mask = moved.get("attention_mask")
                self.model(**moved)
        finally:
            self._attention_mask = None
            for handle in handles:
                handle.remove()
            self.model.train(was_training)

        return {
            group: accumulator.finalize()
            for group, accumulator in self.accumulators.items()
        }


def merge_phase_moments(
    old: SecondMomentState,
    new: SecondMomentState,
) -> SecondMomentState:
    if old.matrix.shape != new.matrix.shape:
        raise ValueError("Cannot merge second moments with different shapes")
    phases = old.phases + new.phases
    matrix = (old.matrix * old.phases + new.matrix * new.phases) / phases
    return SecondMomentState(
        matrix=matrix,
        observations=old.observations + new.observations,
        phases=phases,
    )


def build_and_assign_filters(
    model: nn.Module,
    moments: Dict[str, SecondMomentState],
    *,
    energy_fraction: float,
    leakage: float,
    ridge: float,
) -> Dict[str, ProjectionBuildResult]:
    results = {
        group: build_hard_leaky_filter(
            state.matrix,
            energy_fraction=energy_fraction,
            leakage=leakage,
            ridge=ridge,
        )
        for group, state in moments.items()
    }
    assign_group_filters(
        model,
        {group: result.filter for group, result in results.items()},
    )
    return results

