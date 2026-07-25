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
        self.lora_A = nn.Linear(
            self.in_features, self.rank, bias=False, device=device, dtype=dtype
        )
        self.lora_B = nn.Linear(
            self.rank, self.out_features, bias=False, device=device, dtype=dtype
        )
        self.dropout = nn.Dropout(dropout)
        self.filter = HardLeakyFilter(self.in_features, leakage=1.0).to(
            device=device, dtype=dtype
        )
        self.register_buffer("merged", torch.tensor(False, device=device))
        self.reset_lora_parameters()

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

    def adapter_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_filter:
            x = self.filter(x)
        return self.dropout(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(x)
        if bool(self.merged.item()):
            return base
        adapter = self.lora_B(self.lora_A(self.adapter_input(x)))
        return base + self.scaling * adapter

    def delta_weight(self) -> torch.Tensor:
        delta = self.lora_B.weight @ self.lora_A.weight
        if self.use_filter and not self.filter.is_identity:
            basis = self.filter.protected_basis.to(
                device=delta.device, dtype=delta.dtype
            )
            coefficient = 1.0 - self.filter.leakage.to(
                device=delta.device, dtype=delta.dtype
            )
            delta = delta - coefficient * ((delta @ basis) @ basis.transpose(0, 1))
        return self.scaling * delta

    @torch.no_grad()
    def merge(self) -> None:
        if bool(self.merged.item()):
            return
        self.base_layer.weight.add_(self.delta_weight())
        self.merged.fill_(True)

    @torch.no_grad()
    def unmerge(self) -> None:
        if not bool(self.merged.item()):
            return
        self.base_layer.weight.sub_(self.delta_weight())
        self.merged.fill_(False)

    @torch.no_grad()
    def merge_and_reset(self) -> None:
        self.merge()
        self.reset_lora_parameters()
        self.merged.fill_(False)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, use_filter={self.use_filter}"
        )

