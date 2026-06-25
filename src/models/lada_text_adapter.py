import math
from typing import Iterable, Optional

import torch
import torch.nn as nn


class LADAAdaptFormer(nn.Module):
    """LADA-style AdaptFormer block for CLIP text transformer layers."""

    def __init__(self, hidden_size: int, bottle_dim: int = 16, dtype=None):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_size, dtype=dtype)
        self.down_proj = nn.Linear(hidden_size, bottle_dim, dtype=dtype)
        self.relu = nn.ReLU(inplace=True)
        self.up_proj = nn.Linear(bottle_dim, hidden_size, dtype=dtype)
        self.scale = nn.Parameter(torch.ones(1, dtype=dtype))

        nn.init.kaiming_normal_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln(x)
        x = self.down_proj(x)
        x = self.relu(x)
        x = self.up_proj(x)
        return x * self.scale

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)
        nn.init.ones_(self.scale)


class LADAAdaptFormerEncoderLayer(nn.Module):
    """Wrap a HF CLIPEncoderLayer and insert AdaptFormer in the FFN residual path."""

    def __init__(self, base_layer: nn.Module, bottle_dim: int = 16, scale: float = 0.1):
        super().__init__()
        self.base_layer = base_layer
        hidden_size = base_layer.layer_norm2.normalized_shape[0]
        dtype = base_layer.layer_norm2.weight.dtype
        self.adaptformer = LADAAdaptFormer(hidden_size, bottle_dim=bottle_dim, dtype=dtype)
        self.scale = float(scale)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ):
        residual = hidden_states

        hidden_states = self.base_layer.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.base_layer.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.base_layer.layer_norm2(hidden_states)
        hidden_states = self.base_layer.mlp(hidden_states)
        hidden_states = hidden_states + self.scale * self.adaptformer(residual)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs


class LADAAdaptFormerCLIPTextTransformer(nn.Module):
    """HF CLIPTextTransformer wrapper using the text AdaptFormer from official LADA."""

    def __init__(
        self,
        clip_text_model: nn.Module,
        adapter_dim: int = 16,
        adapter_scale: float = 0.1,
        adapter_layers: Optional[Iterable[int]] = None,
    ):
        super().__init__()
        for p in clip_text_model.parameters():
            p.requires_grad_(False)

        self.clip_text_model = clip_text_model
        self.adapter_dim = int(adapter_dim)
        self.adapter_scale = float(adapter_scale)
        self.adapter_layers = (
            list(adapter_layers)
            if adapter_layers is not None
            else list(range(len(clip_text_model.encoder.layers)))
        )
        self.adaptformer_modules = nn.ModuleDict()

        for idx in self.adapter_layers:
            base_layer = clip_text_model.encoder.layers[idx]
            wrapped = LADAAdaptFormerEncoderLayer(
                base_layer,
                bottle_dim=self.adapter_dim,
                scale=self.adapter_scale,
            )
            clip_text_model.encoder.layers[idx] = wrapped
            self.adaptformer_modules[f"layer_{idx}_adaptformer"] = wrapped.adaptformer

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        return self.clip_text_model(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    def get_params(self):
        return [p for p in self.adaptformer_modules.parameters() if p.requires_grad]

    def get_module_names(self):
        return list(self.adaptformer_modules.keys())

    def reset_adapters(self) -> None:
        for module in self.adaptformer_modules.values():
            module.reset_parameters()

    def regularization_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=next(self.parameters()).device)
