from dataclasses import dataclass
from typing import Optional
import weakref

import torch
import torch.nn as nn


@dataclass
class ProjectionBuildResult:
    filter: "HardLeakyFilter"
    eigenvalues: torch.Tensor
    tail_dimension: int
    protected_dimension: int
    captured_tail_energy: float


class HardLeakyFilter(nn.Module):
    """Exact factorized application of a hard leaky null-space filter.

    For a protected high-energy basis U, this module applies

        P = I - (1 - leakage) U U^T

    without materializing the dense d x d matrix.
    """

    def __init__(
        self,
        input_dim: int,
        protected_basis: Optional[torch.Tensor] = None,
        leakage: float = 1.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if not 0.0 <= leakage <= 1.0:
            raise ValueError("leakage must lie in [0, 1]")
        if protected_basis is None:
            protected_basis = torch.empty(input_dim, 0)
        self._validate_basis(protected_basis, input_dim)
        self.input_dim = int(input_dim)
        self.register_buffer("protected_basis", protected_basis.detach().clone())
        self.register_buffer(
            "leakage",
            torch.tensor(float(leakage), dtype=torch.float32),
        )
        self._reuse_limit = 1
        self._cached_input_ref = None
        self._cached_output: Optional[torch.Tensor] = None
        self._cached_uses = 0

    @staticmethod
    def _validate_basis(basis: torch.Tensor, input_dim: int) -> None:
        if basis.ndim != 2 or basis.shape[0] != input_dim:
            raise ValueError(
                f"protected_basis must have shape ({input_dim}, k), got {tuple(basis.shape)}"
            )

    @property
    def protected_dimension(self) -> int:
        return int(self.protected_basis.shape[1])

    @property
    def is_identity(self) -> bool:
        return self.protected_dimension == 0 or float(self.leakage.item()) == 1.0

    @torch.no_grad()
    def set_basis(self, basis: torch.Tensor, leakage: Optional[float] = None) -> None:
        self._validate_basis(basis, self.input_dim)
        if leakage is not None and not 0.0 <= leakage <= 1.0:
            raise ValueError("leakage must lie in [0, 1]")
        self.protected_basis = basis.detach().clone()
        if leakage is not None:
            self.leakage.fill_(float(leakage))
        self.clear_runtime_cache()

    def configure_runtime_reuse(self, calls_per_input: int) -> None:
        """Reuse one exact factorized projection for shared q/k/v inputs."""

        if calls_per_input <= 0:
            raise ValueError("calls_per_input must be positive")
        self._reuse_limit = int(calls_per_input)
        self.clear_runtime_cache()

    def clear_runtime_cache(self) -> None:
        self._cached_input_ref = None
        self._cached_output = None
        self._cached_uses = 0

    def _apply_filter(self, x: torch.Tensor) -> torch.Tensor:
        basis = self.protected_basis.to(device=x.device, dtype=x.dtype)
        coefficient = 1.0 - self.leakage.to(device=x.device, dtype=x.dtype)
        return x - coefficient * ((x @ basis) @ basis.transpose(0, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected last dimension {self.input_dim}, got {x.shape[-1]}"
            )
        if self.is_identity:
            return x
        cached_input = (
            self._cached_input_ref()
            if self._cached_input_ref is not None
            else None
        )
        if (
            self._reuse_limit > 1
            and cached_input is x
            and self._cached_output is not None
        ):
            output = self._cached_output
            self._cached_uses += 1
            if self._cached_uses >= self._reuse_limit:
                self.clear_runtime_cache()
            return output
        output = self._apply_filter(x)
        if self._reuse_limit > 1:
            self._cached_input_ref = weakref.ref(x)
            self._cached_output = output
            self._cached_uses = 1
        return output

    def dense(self, *, device=None, dtype=None) -> torch.Tensor:
        device = self.protected_basis.device if device is None else device
        dtype = self.protected_basis.dtype if dtype is None else dtype
        identity = torch.eye(self.input_dim, device=device, dtype=dtype)
        if self.is_identity:
            return identity
        basis = self.protected_basis.to(device=device, dtype=dtype)
        coefficient = 1.0 - self.leakage.to(device=device, dtype=dtype)
        return identity - coefficient * (basis @ basis.transpose(0, 1))


def _tail_dimension(eigenvalues: torch.Tensor, energy_fraction: float) -> int:
    dimension = int(eigenvalues.numel())
    if energy_fraction <= 0.0:
        return 0
    if energy_fraction >= 1.0:
        return dimension
    total = eigenvalues.sum()
    if not torch.isfinite(total) or float(total.item()) <= 0.0:
        return dimension
    threshold = energy_fraction * total
    index = int(torch.searchsorted(torch.cumsum(eigenvalues, dim=0), threshold).item())
    return min(dimension, index + 1)


@torch.no_grad()
def build_hard_leaky_filter(
    second_moment: torch.Tensor,
    *,
    energy_fraction: float = 0.20,
    leakage: float = 0.02,
    ridge: float = 1e-4,
    basis_dtype: Optional[torch.dtype] = None,
) -> ProjectionBuildResult:
    """Build the intended inclusive-threshold hard LoRA-NF filter.

    The current CLIP implementation uses the index at which cumulative energy
    crosses the threshold as a slice endpoint. That excludes the crossing vector.
    This implementation uses the mathematically intended inclusive definition
    (index + 1) and records the choice in METHOD_SPEC.md.
    """

    if second_moment.ndim != 2 or second_moment.shape[0] != second_moment.shape[1]:
        raise ValueError("second_moment must be a square matrix")
    if not 0.0 <= energy_fraction <= 1.0:
        raise ValueError("energy_fraction must lie in [0, 1]")
    if not 0.0 <= leakage <= 1.0:
        raise ValueError("leakage must lie in [0, 1]")
    if ridge < 0.0:
        raise ValueError("ridge must be non-negative")

    original_device = second_moment.device
    source_dtype = second_moment.dtype
    matrix = second_moment.detach().to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(matrix).all():
        raise ValueError("second_moment contains non-finite values")
    matrix = 0.5 * (matrix + matrix.transpose(0, 1))
    if ridge:
        matrix = matrix + ridge * torch.eye(matrix.shape[0], dtype=matrix.dtype)

    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    spectral_scale = max(
        1.0,
        float(eigenvalues.abs().max().item()),
    )
    minimum = float(eigenvalues.min().item())
    if minimum < -1e-6 * spectral_scale:
        raise ValueError(
            "second_moment is materially non-PSD after symmetrization/ridge: "
            f"minimum_eigenvalue={minimum}, spectral_scale={spectral_scale}"
        )
    eigenvalues = eigenvalues.clamp_min(0.0)
    tail_dimension = _tail_dimension(eigenvalues, energy_fraction)
    protected_basis = eigenvectors[:, tail_dimension:]
    output_dtype = source_dtype if basis_dtype is None else basis_dtype
    protected_basis = protected_basis.to(device=original_device, dtype=output_dtype)

    total = eigenvalues.sum()
    if float(total.item()) > 0.0 and tail_dimension > 0:
        captured = float((eigenvalues[:tail_dimension].sum() / total).item())
    else:
        captured = 0.0

    filter_module = HardLeakyFilter(
        input_dim=second_moment.shape[0],
        protected_basis=protected_basis,
        leakage=leakage,
    )
    return ProjectionBuildResult(
        filter=filter_module,
        eigenvalues=eigenvalues.to(dtype=torch.float32),
        tail_dimension=tail_dimension,
        protected_dimension=second_moment.shape[0] - tail_dimension,
        captured_tail_energy=captured,
    )
