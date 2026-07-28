from dataclasses import asdict, dataclass
from typing import Any, Dict

import torch


@dataclass(frozen=True)
class RunMetadata:
    run_id: str
    execution_mode: str
    method: str
    model_id: str
    seed: int
    commit_sha: str

    def __post_init__(self) -> None:
        if self.execution_mode not in {"gpu_formal", "cpu_smoke_only"}:
            raise ValueError(f"Unknown execution_mode: {self.execution_mode}")

    @property
    def formal_result_eligible(self) -> bool:
        return self.execution_mode == "gpu_formal"

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["formal_result_eligible"] = self.formal_result_eligible
        return payload


def require_formal_result(metadata: RunMetadata) -> None:
    if not metadata.formal_result_eligible:
        raise ValueError(
            f"Run {metadata.run_id} is {metadata.execution_mode} and cannot enter formal results"
        )


@torch.no_grad()
def tensor_difference_summary(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> Dict[str, Any]:
    """Record numerical drift without relying only on an assert tolerance."""

    if reference.shape != candidate.shape:
        raise ValueError(
            "Difference summary requires equal tensor shapes: "
            f"{tuple(reference.shape)} != {tuple(candidate.shape)}"
        )
    reference_float = reference.detach().float()
    candidate_float = candidate.detach().float()
    if (
        not bool(torch.isfinite(reference_float).all().item())
        or not bool(torch.isfinite(candidate_float).all().item())
    ):
        raise ValueError("Difference summary tensors must be finite")
    difference = candidate_float - reference_float
    rmse = torch.sqrt(torch.mean(difference.square()))
    reference_rms = torch.sqrt(torch.mean(reference_float.square()))
    denominator = torch.clamp(
        reference_rms,
        min=torch.finfo(reference_float.dtype).eps,
    )
    return {
        "element_count": difference.numel(),
        "max_absolute_error": float(difference.abs().max().item()),
        "mean_absolute_error": float(difference.abs().mean().item()),
        "rmse": float(rmse.item()),
        "reference_rms": float(reference_rms.item()),
        "relative_rmse": float((rmse / denominator).item()),
    }
