import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .filters import HardLeakyFilter


class FilteredLoRALinear(nn.Module):
    """Frozen Linear + trainable LoRA branch with a persistent input filter."""

    def __init__(
        self,
        base_layer: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        use_filter: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("base_layer must be torch.nn.Linear")
        if rank <= 0:
            raise ValueError("rank must be positive")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")

        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.use_filter = bool(use_filter)

        for parameter in self.base_layer.parameters():
            parameter.requires_grad_(False)

        device = base_layer.weight.device
        dtype = base_layer.weight.dtype
        adapter_dtype = torch.float32
        self.lora_A = nn.Linear(
            self.in_features,
            self.rank,
            bias=False,
            device=device,
            dtype=adapter_dtype,
        )
        self.lora_B = nn.Linear(
            self.rank,
            self.out_features,
            bias=False,
            device=device,
            dtype=adapter_dtype,
        )
        self.dropout = nn.Dropout(dropout)
        self.filter = HardLeakyFilter(self.in_features, leakage=1.0).to(
            device=device, dtype=dtype
        )
        self.register_buffer(
            "base_offset_A",
            torch.empty(0, self.in_features, device=device, dtype=adapter_dtype),
        )
        self.register_buffer(
            "base_offset_B",
            torch.empty(self.out_features, 0, device=device, dtype=adapter_dtype),
        )
        self.register_buffer("merged", torch.tensor(False, device=device))
        self._merged_python = False
        self.reset_lora_parameters()

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        # Loading is cold-path code, so synchronize the persisted tensor once
        # rather than calling ``merged.item()`` on every CUDA forward.
        self._merged_python = bool(self.merged.detach().cpu().item())

    def reset_lora_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    @torch.no_grad()
    def set_filter(self, filter_module: HardLeakyFilter) -> None:
        if filter_module.input_dim != self.in_features:
            raise ValueError(
                f"Filter input_dim {filter_module.input_dim} does not match "
                f"layer input {self.in_features}"
            )
        self.filter.set_basis(
            filter_module.protected_basis.to(
                device=self.base_layer.weight.device,
                dtype=self.base_layer.weight.dtype,
            ),
            leakage=float(filter_module.leakage.item()),
        )

    @torch.no_grad()
    def share_filter(self, filter_module: HardLeakyFilter) -> None:
        """Attach a shared group filter without cloning its protected basis."""

        if filter_module.input_dim != self.in_features:
            raise ValueError(
                f"Filter input_dim {filter_module.input_dim} does not match "
                f"layer input {self.in_features}"
            )
        filter_module.to(
            device=self.base_layer.weight.device,
            dtype=self.base_layer.weight.dtype,
        )
        self.filter = filter_module

    def adapter_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_filter:
            x = self.filter(x)
        return self.dropout(x)

    @property
    def has_base_offset(self) -> bool:
        return self.base_offset_A.shape[0] > 0

    @torch.no_grad()
    def set_decomposition_initialization(
        self,
        lora_A: torch.Tensor,
        lora_B: torch.Tensor,
    ) -> None:
        """Install a numerically exact-function decomposition initialization.

        The official LoRA-Null/MiLoRA reparameterization subtracts the initial
        low-rank product from the frozen base weight and adds the same product
        through the adapter.  Performing those as separate deep-network
        matmuls accumulates avoidable FP32 cancellation error.  We keep the
        original base weight unchanged and represent the mathematically
        identical update as ``B A - B_0 A_0`` instead.  At initialization the
        two factor paths are bit-identical, so the adapter update is exactly
        zero while gradients with respect to the trainable ``A`` and ``B`` are
        unchanged.
        """

        if tuple(lora_A.shape) != tuple(self.lora_A.weight.shape):
            raise ValueError(
                f"Expected A shape {tuple(self.lora_A.weight.shape)}, "
                f"got {tuple(lora_A.shape)}"
            )
        if tuple(lora_B.shape) != tuple(self.lora_B.weight.shape):
            raise ValueError(
                f"Expected B shape {tuple(self.lora_B.weight.shape)}, "
                f"got {tuple(lora_B.shape)}"
            )
        if self.has_base_offset:
            raise RuntimeError("A decomposition initialization is already installed")
        offset_A = lora_A.to(
            device=self.lora_A.weight.device,
            dtype=self.lora_A.weight.dtype,
        )
        offset_B = lora_B.to(
            device=self.lora_B.weight.device,
            dtype=self.lora_B.weight.dtype,
        )
        self.lora_A.weight.copy_(offset_A)
        self.lora_B.weight.copy_(offset_B)
        self.base_offset_A = offset_A.detach().clone()
        self.base_offset_B = offset_B.detach().clone()

    @torch.no_grad()
    def restore_base_offset(self, lora_A: torch.Tensor, lora_B: torch.Tensor) -> None:
        """Reapply a saved decomposition offset to a fresh base checkpoint."""

        self.set_decomposition_initialization(lora_A, lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(x)
        if self._merged_python:
            return base
        adapter_input = self.adapter_input(x).to(dtype=self.lora_A.weight.dtype)
        adapter = self.lora_B(self.lora_A(adapter_input))
        if self.has_base_offset:
            initial_adapter = F.linear(
                F.linear(adapter_input, self.base_offset_A),
                self.base_offset_B,
            )
            adapter = adapter - initial_adapter
        return base + (self.scaling * adapter).to(dtype=base.dtype)

    def delta_weight(self) -> torch.Tensor:
        delta = self.lora_B.weight @ self.effective_lora_A()
        if self.has_base_offset:
            delta = delta - (
                self.base_offset_B
                @ self._effective_lora_A(self.base_offset_A)
            )
        return self.scaling * delta

    def effective_lora_A(self) -> torch.Tensor:
        """Return the factorized right-hand update after the runtime filter.

        ``B @ effective_lora_A()`` is exactly the unscaled dense update used
        by :meth:`delta_weight`.  Keeping this factorized representation lets
        continual runs replay every merged task from the original checkpoint
        without storing dense deltas or every historical filter basis.
        """

        return self._effective_lora_A(self.lora_A.weight)

    def _effective_lora_A(self, factor: torch.Tensor) -> torch.Tensor:
        effective = factor
        if self.use_filter and not self.filter.is_identity:
            basis = self.filter.protected_basis.to(
                device=effective.device, dtype=effective.dtype
            )
            coefficient = 1.0 - self.filter.leakage.to(
                device=effective.device, dtype=effective.dtype
            )
            effective = effective - coefficient * (
                (effective @ basis) @ basis.transpose(0, 1)
            )
        return effective

    @torch.no_grad()
    def merge(self) -> None:
        if self._merged_python:
            return
        self.base_layer.weight.add_(self.delta_weight())
        self.merged.fill_(True)
        self._merged_python = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if not self._merged_python:
            return
        self.base_layer.weight.sub_(self.delta_weight())
        self.merged.fill_(False)
        self._merged_python = False

    @torch.no_grad()
    def merge_and_reset(self) -> None:
        self.merge()
        self.reset_lora_parameters()
        self.base_offset_A = torch.empty(
            0,
            self.in_features,
            device=self.lora_A.weight.device,
            dtype=self.lora_A.weight.dtype,
        )
        self.base_offset_B = torch.empty(
            self.out_features,
            0,
            device=self.lora_B.weight.device,
            dtype=self.lora_B.weight.dtype,
        )
        self.merged.fill_(False)
        self._merged_python = False

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, use_filter={self.use_filter}"
        )
