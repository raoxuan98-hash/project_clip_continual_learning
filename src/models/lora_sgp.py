import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Iterable, List, Optional
import logging

try:
    from transformers.models.clip.modeling_clip import eager_attention_forward, ALL_ATTENTION_FUNCTIONS
except Exception:
    eager_attention_forward = None
    ALL_ATTENTION_FUNCTIONS = {}

def compute_weights(x: torch.Tensor, weight_kind="log1p", beta=1.0, weight_p=1.0, weight_alpha=0.5, weight_kappa=2.0) -> torch.Tensor:
    if weight_kind == "exp":
        return torch.exp(-beta * x)

    elif weight_kind == "rational1":
        return 1.0 / (1.0 + beta * x)

    elif weight_kind == "rational2":
        return 1.0 / (1.0 + beta * (x ** 2))

    elif weight_kind == "sqrt_rational2":
        return 1.0 / torch.sqrt(1.0 + beta * (x ** 2))

    elif weight_kind == "log1p":
        return 1.0 / (1.0 + beta * torch.log1p(x**weight_p))

    elif weight_kind == "power_family":
        return (1.0 + beta * (x ** weight_p)) ** (-weight_alpha)

    elif weight_kind == "stretched_exp":
        return torch.exp(- (beta * x) ** weight_kappa)

    elif weight_kind == "band_pass":
        # 抑制两端，只允许中间带宽更新
        w_high = 1.0 / (1.0 + beta * torch.log1p(x**weight_p))
        w_low = 1.0 / (1.0 + weight_alpha * torch.log1p(x**(-weight_kappa)))
        return w_high * w_low

    elif weight_kind == "high_cut":
        # 允许中间+小端，只抑制大端（带阈值 τ）
        tau = max(weight_alpha, 1e-6)
        x_scaled = x / tau
        return 1.0 / (1.0 + beta * (x_scaled ** weight_p))

    else:
        raise ValueError(
            f"Unknown weight_kind='{weight_kind}'. "
            f"Choose from ['exp','rational1','rational2','sqrt_rational2','log1p','power_family','stretched_exp',"
            f"'band_pass','high_cut']")

def compute_proj_init(
    W: torch.Tensor,
    cov: torch.Tensor,
    r: int,
    window: str = "tail",
    s_ratio: float = 0.33,
):
    """Proj-Σ 初始化：从协方差特征分解取 V_small → 投影 W → SVD 分解为 A, B。

    Args:
        W: 当前合并后的权重 [d_out, d_in]
        cov: 特征 Gram 矩阵 [d_in, d_in]
        r: LoRA rank
        window: 'tail'（最小特征值方向）或 'middle'（中间特征值方向）
        s_ratio: 'middle' 窗口起始位置比例（默认 0.33）
    Returns:
        A_init: [r, d_in], B_init: [d_out, r], W_residual: [d_out, d_in]
    """
    # 协方差预处理（同 build_projection）
    cov_double = cov.to(torch.float64)
    cov_double = (cov_double + cov_double.t()) / 2.0
    safe_eps = 1e-4
    eye = torch.eye(cov_double.size(0), device=cov_double.device, dtype=torch.float64)
    cov_double = cov_double + safe_eps * eye

    eigvals_double, eigvecs_double = torch.linalg.eigh(cov_double)
    eigvecs = eigvecs_double.to(dtype=W.dtype, device=W.device)

    d = cov.size(0)
    if window == "tail":
        V_small = eigvecs[:, :r]                     # 最小 r 个特征向量
    elif window == "middle":
        s = max(1, min(int(d * s_ratio), d - r - 1))
        V_small = eigvecs[:, s:s + r]                # 中间 r 个特征向量
    else:
        raise ValueError(f"Unknown window '{window}'. Choose from ['tail', 'middle']")

    # BA_init = W @ V_small @ V_small^T
    BA_init = W @ V_small @ V_small.T

    # SVD 分解 → A_init, B_init
    U_svd, D_svd, Vh_svd = torch.linalg.svd(BA_init, full_matrices=False)
    r_eff = min(r, U_svd.size(1))                    # BA_init 的有效秩

    A_init = torch.diag(D_svd[:r_eff] ** 0.5) @ Vh_svd[:r_eff, :]
    B_init = U_svd[:, :r_eff] @ torch.diag(D_svd[:r_eff] ** 0.5)
    W_residual = W - BA_init

    # 若有效秩 < r，补零对齐
    if r_eff < r:
        pad_A = torch.zeros(r - r_eff, A_init.size(1), device=A_init.device, dtype=A_init.dtype)
        pad_B = torch.zeros(B_init.size(0), r - r_eff, device=B_init.device, dtype=B_init.dtype)
        A_init = torch.cat([A_init, pad_A], dim=0)
        B_init = torch.cat([B_init, pad_B], dim=1)

    return A_init, B_init, W_residual


def compute_tail_basis(
    cov: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """从 activation Gram 矩阵提取前 k 个最小特征值方向作为 tail basis。

    Args:
        cov: 协方差/activation Gram 矩阵 [d_in, d_in]
        k: 要保留的 tail 方向数
    Returns:
        U_h: [d_in, k]，列向量为 orthonormal tail basis
    """
    cov_double = cov.to(torch.float64)
    cov_double = (cov_double + cov_double.t()) / 2.0
    safe_eps = 1e-4
    eye = torch.eye(cov_double.size(0), device=cov_double.device, dtype=torch.float64)
    cov_double = cov_double + safe_eps * eye

    _, eigvecs_double = torch.linalg.eigh(cov_double)
    eigvecs = eigvecs_double.to(dtype=cov.dtype, device=cov.device)
    return eigvecs[:, :k]


def compute_weight_svd_init(
    W: torch.Tensor,
    r: int,
    window: str = "tail",
    s_ratio: float = 0.33,
):
    """权重 SVD 初始化：对 W 做 SVD → 取窗口分量 → 分解为 A, B。

    Args:
        W: 当前合并后的权重 [d_out, d_in]
        r: LoRA rank
        window: 'tail'（最小奇异值，MiLoRA 风格）或 'middle'（中间奇异值，Least 风格）
        s_ratio: 'middle' 窗口起始位置比例（默认 0.33）
    Returns:
        A_init: [r, d_in], B_init: [d_out, r], W_residual: [d_out, d_in]
    """
    U, D, Vh = torch.linalg.svd(W, full_matrices=False)
    k = min(W.shape)

    if window == "tail":
        s = max(0, k - r)
        U_small = U[:, s:]
        D_small = D[s:]
        V_small = Vh[s:, :]
    elif window == "middle":
        s = max(1, min(int(k * s_ratio), k - r))
        U_small = U[:, s:s + r]
        D_small = D[s:s + r]
        V_small = Vh[s:s + r, :]
    else:
        raise ValueError(f"Unknown window '{window}'. Choose from ['tail', 'middle']")

    r_eff = min(r, U_small.size(1))
    A_init = torch.diag(D_small[:r_eff] ** 0.5) @ V_small[:r_eff, :]
    B_init = U_small[:, :r_eff] @ torch.diag(D_small[:r_eff] ** 0.5)
    BA_init = U_small[:, :r_eff] @ torch.diag(D_small[:r_eff]) @ V_small[:r_eff, :]
    W_residual = W - BA_init

    if r_eff < r:
        pad_A = torch.zeros(r - r_eff, A_init.size(1), device=A_init.device, dtype=A_init.dtype)
        pad_B = torch.zeros(B_init.size(0), r - r_eff, device=B_init.device, dtype=B_init.dtype)
        A_init = torch.cat([A_init, pad_A], dim=0)
        B_init = torch.cat([B_init, pad_B], dim=1)

    return A_init, B_init, W_residual


class FixedProjection(nn.Module):
    def __init__(self, P: torch.Tensor):
        super().__init__()
        self.register_buffer("P", P)

    def forward(self):
        return self.P


class FusedQKVCLIPAttention(nn.Module):
    """Timm 风格的 fused QKV Attention。

    将 q_proj/k_proj/v_proj 合并为单个 qkv_proj，输出维度为 3*embed_dim，
    再 split 为 q/k/v 走与原 CLIPAttention 完全一致的 attention 计算。
    该模块会保留原 out_proj 对象，方便外部继续对其做 LoRA/DoRA 包装。
    """

    def __init__(self, orig_attn: nn.Module):
        super().__init__()
        self.config = orig_attn.config
        self.embed_dim = orig_attn.embed_dim
        self.num_heads = getattr(orig_attn, 'num_heads', orig_attn.config.num_attention_heads)
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got embed_dim={self.embed_dim}, "
                f"num_heads={self.num_heads})."
            )
        self.scale = self.head_dim ** -0.5
        self.dropout = orig_attn.dropout
        self.is_causal = getattr(orig_attn, 'is_causal', False)

        # 用原 q/k/v 的权重初始化一个 qkv_proj
        has_bias = orig_attn.q_proj.bias is not None
        self.qkv_proj = nn.Linear(self.embed_dim, self.embed_dim * 3, bias=has_bias)
        with torch.no_grad():
            self.qkv_proj.weight.copy_(torch.cat([
                orig_attn.q_proj.weight.data,
                orig_attn.k_proj.weight.data,
                orig_attn.v_proj.weight.data,
            ], dim=0))
            if has_bias:
                self.qkv_proj.bias.copy_(torch.cat([
                    orig_attn.q_proj.bias.data,
                    orig_attn.k_proj.bias.data,
                    orig_attn.v_proj.bias.data,
                ], dim=0))

        # out_proj 保持原对象，后续会被 LoRA 包装替换
        self.out_proj = orig_attn.out_proj

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        batch_size, seq_length, embed_dim = hidden_states.shape

        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split(embed_dim, dim=-1)

        queries = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        keys = k.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        values = v.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # 与 CLIPAttention 完全一致的 attention mask / interface 处理
        if self.config._attn_implementation == "flash_attention_2":
            self.is_causal = causal_attention_mask is not None
        else:
            if attention_mask is not None and causal_attention_mask is not None:
                attention_mask = attention_mask + causal_attention_mask
            elif causal_attention_mask is not None:
                attention_mask = causal_attention_mask

        if self.config._attn_implementation != "eager" and ALL_ATTENTION_FUNCTIONS:
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        else:
            attention_interface = eager_attention_forward

        attn_output, attn_weights = attention_interface(
            self,
            queries,
            keys,
            values,
            attention_mask,
            scaling=self.scale,
            dropout=0.0 if not self.training else self.dropout,
            is_causal=self.is_causal,
            output_attentions=output_attentions,
        )

        attn_output = attn_output.reshape(batch_size, seq_length, embed_dim).contiguous()
        attn_output = self.out_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights


class SGPBaseLoRA(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        r: int,
        proj: nn.Module,
        projection_param_mode: str = "full",
        basis_rank: Optional[int] = None):

        super().__init__()
        self.linear = linear
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.r = r
        self.P = proj
        self.projection_param_mode = projection_param_mode
        self.basis_rank = basis_rank
        self.register_buffer("basis_ready", torch.tensor(False))

        device = linear.weight.device
        dtype = linear.weight.dtype

        # 始终保留 full-mode 的 A/B，用于 Task 1 没有 history covariance 时的回退
        self.A = nn.Parameter(torch.zeros(r, self.in_features, device=device, dtype=dtype))
        self.B = nn.Parameter(torch.zeros(self.out_features, r, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

        if projection_param_mode == "fixed_basis":
            if basis_rank is None:
                raise ValueError("fixed_basis mode requires basis_rank")
            k = basis_rank
            self.register_buffer("U_h", torch.zeros(self.in_features, k, device=device, dtype=dtype))
            self.B_basis = nn.Parameter(torch.zeros(self.out_features, k, device=device, dtype=dtype))
            nn.init.zeros_(self.B_basis)
        elif projection_param_mode == "core_basis":
            if basis_rank is None:
                raise ValueError("core_basis mode requires basis_rank")
            k = basis_rank
            self.register_buffer("U_h", torch.zeros(self.in_features, k, device=device, dtype=dtype))
            self.C = nn.Parameter(torch.zeros(r, k, device=device, dtype=dtype))
            nn.init.kaiming_uniform_(self.C, a=math.sqrt(5))
        elif projection_param_mode != "full":
            raise ValueError(f"Unknown projection_param_mode '{projection_param_mode}'")

        if linear.bias is not None:
            self.bias = linear.bias
        else:
            self.register_buffer("bias", None)

        self.register_buffer("lora_active", torch.tensor(True, device=device))

    def _compute_lora_delta(self) -> torch.Tensor:
        """根据 projection_param_mode 计算 LoRA delta (d_out, d_in)"""
        if self.projection_param_mode == "full" or not self.basis_ready.item():
            P_scaled = self.P()
            return self.B @ self.A @ P_scaled
        elif self.projection_param_mode == "fixed_basis":
            return self.B_basis @ self.U_h.T
        elif self.projection_param_mode == "core_basis":
            return self.B @ self.C @ self.U_h.T
        else:
            raise ValueError(f"Unknown projection_param_mode '{self.projection_param_mode}'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.lora_active:
            lora_delta = self._compute_lora_delta()
            adapted_weight = self.linear.weight + lora_delta
            return F.linear(x, adapted_weight, self.bias)
        else:
           return self.linear(x)

    def merge_lora_weights(self, lora_active: bool=True) -> None:
        """将 LoRA 权重合并到原始权重中"""
        with torch.no_grad():
            delta = self._compute_lora_delta()
            self.linear.weight.data.add_(delta)
            if self.projection_param_mode == "full" or not self.basis_ready.item():
                nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
                self.B.data.zero_()
            if self.projection_param_mode == "fixed_basis":
                self.B_basis.data.zero_()
            if self.projection_param_mode == "core_basis":
                nn.init.kaiming_uniform_(self.C, a=math.sqrt(5))
                self.B.data.zero_()
            self.lora_active = torch.tensor(lora_active)

    def get_effective_weight(self) -> torch.Tensor:
        """获取当前有效基础权重（用于 LoRA-Null 等初始化）"""
        return self.linear.weight.data

    def set_effective_weight(self, W: torch.Tensor) -> None:
        """设置当前有效基础权重（用于 LoRA-Null 等初始化）"""
        self.linear.weight.data.copy_(W)

    def set_lora_init(self, A_init: torch.Tensor, B_init: torch.Tensor) -> None:
        """设置 LoRA-Null/SVD-W 初始化后的参数。

        对 full 模式：A=A_init, B=B_init。
        对 fixed_basis：B_basis = W U_h = B_init A_init U_h。
        对 core_basis：C = A_init U_h, B = B_init。
        """
        with torch.no_grad():
            if self.projection_param_mode == "full":
                self.A.data.copy_(A_init)
                self.B.data.copy_(B_init)
            elif self.projection_param_mode == "fixed_basis":
                B_basis_init = B_init @ A_init @ self.U_h
                self.B_basis.data.copy_(B_basis_init)
            elif self.projection_param_mode == "core_basis":
                C_init = A_init @ self.U_h
                self.C.data.copy_(C_init)
                self.B.data.copy_(B_init)


class SGPBaseDoRA(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        r: int,
        proj: nn.Module,
        projection_param_mode: str = "full",
        basis_rank: Optional[int] = None):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.r = r
        self.P = proj
        self.projection_param_mode = projection_param_mode
        self.basis_rank = basis_rank
        self.register_buffer("basis_ready", torch.tensor(False))

        device = linear.weight.device
        dtype = linear.weight.dtype

        # 始终保留 full-mode 的 A/B，用于 Task 1 回退
        self.A = nn.Parameter(torch.zeros(r, self.in_features, device=device, dtype=dtype))
        self.B = nn.Parameter(torch.zeros(self.out_features, r, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

        if projection_param_mode == "fixed_basis":
            if basis_rank is None:
                raise ValueError("fixed_basis mode requires basis_rank")
            k = basis_rank
            self.register_buffer("U_h", torch.zeros(self.in_features, k, device=device, dtype=dtype))
            self.B_basis = nn.Parameter(torch.zeros(self.out_features, k, device=device, dtype=dtype))
            nn.init.zeros_(self.B_basis)
        elif projection_param_mode == "core_basis":
            if basis_rank is None:
                raise ValueError("core_basis mode requires basis_rank")
            k = basis_rank
            self.register_buffer("U_h", torch.zeros(self.in_features, k, device=device, dtype=dtype))
            self.C = nn.Parameter(torch.zeros(r, k, device=device, dtype=dtype))
            nn.init.kaiming_uniform_(self.C, a=math.sqrt(5))
        elif projection_param_mode != "full":
            raise ValueError(f"Unknown projection_param_mode '{projection_param_mode}'")

        # DoRA 参数：方向 + 幅度
        with torch.no_grad():
            weight = linear.weight.data
            weight_norm = weight.norm(p=2, dim=1, keepdim=True) + 1e-8
            self.weight_directions = nn.Parameter(
                weight / weight_norm, requires_grad=False)
            self.magnitude = nn.Parameter(weight_norm.clone(), requires_grad=True)

        if linear.bias is not None:
            self.bias = linear.bias
        else:
            self.register_buffer("bias", None)

        self.register_buffer("lora_active", torch.tensor(True))

    def _compute_lora_delta(self) -> torch.Tensor:
        """根据 projection_param_mode 计算 LoRA delta (d_out, d_in)"""
        if self.projection_param_mode == "full" or not self.basis_ready.item():
            P_scaled = self.P()
            return self.B @ self.A @ P_scaled
        elif self.projection_param_mode == "fixed_basis":
            return self.B_basis @ self.U_h.T
        elif self.projection_param_mode == "core_basis":
            return self.B @ self.C @ self.U_h.T
        else:
            raise ValueError(f"Unknown projection_param_mode '{self.projection_param_mode}'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.lora_active:
            lora_delta = self._compute_lora_delta()
            adapted_weight = (self.weight_directions + lora_delta) * self.magnitude
        else:
            adapted_weight = self.weight_directions * self.magnitude
        return F.linear(x, adapted_weight, self.bias)

    def merge_lora_weights(self, lora_active: bool=True) -> None:
        with torch.no_grad():
            lora_delta = self._compute_lora_delta()
            self.weight_directions.data.add_(lora_delta)
            if self.projection_param_mode == "full" or not self.basis_ready.item():
                nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
                self.B.data.zero_()
            if self.projection_param_mode == "fixed_basis":
                self.B_basis.data.zero_()
            if self.projection_param_mode == "core_basis":
                nn.init.kaiming_uniform_(self.C, a=math.sqrt(5))
                self.B.data.zero_()
            self.lora_active = torch.tensor(lora_active)

    def get_effective_weight(self) -> torch.Tensor:
        """获取当前有效基础权重（direction * magnitude）"""
        return self.weight_directions.data * self.magnitude.data

    def set_effective_weight(self, W: torch.Tensor) -> None:
        """设置当前有效基础权重，并重新分解为 direction * magnitude"""
        with torch.no_grad():
            new_norm = W.norm(p=2, dim=1, keepdim=True) + 1e-8
            self.weight_directions.data.copy_(W / new_norm)
            self.magnitude.data.copy_(new_norm)

    def set_lora_init(self, A_init: torch.Tensor, B_init: torch.Tensor) -> None:
        """设置 LoRA-Null/SVD-W 初始化后的参数（DoRA 下按 magnitude 缩放）。

        DoRA 下 effective delta = magnitude * lora_delta，而 LoRA-Null 给出的是
        期望的 effective delta BA_init。因此把 lora_delta 按 1/magnitude 缩放。
        """
        with torch.no_grad():
            # magnitude: [d_out, 1]
            scale = 1.0 / (self.magnitude.data + 1e-8)
            if self.projection_param_mode == "full":
                self.A.data.copy_(A_init)
                self.B.data.copy_(B_init * scale)
            elif self.projection_param_mode == "fixed_basis":
                B_basis_init = B_init @ A_init @ self.U_h
                self.B_basis.data.copy_(B_basis_init * scale)
            elif self.projection_param_mode == "core_basis":
                C_init = A_init @ self.U_h
                self.C.data.copy_(C_init)
                self.B.data.copy_(B_init * scale)

class LoRACLIPVisionTransformer(nn.Module):
    def __init__(
        self,
        clip_vision_model: nn.Module,
        r: int,
        lora_layer: Optional[Iterable[int]] = None,
        use_soft_projection: bool = True,
        weight_temp: float = 1.0,
        weight_kind: str = "log1p",
        weight_p: float = 1.0,
        nsp_eps: float = 0.05,
        nsp_weight: float = 0.02,
        lora_class: type = SGPBaseDoRA,
        include_norm: bool = False,
        projection_param_mode: str = "full",
        basis_rank: Optional[int] = None,
        target_modules: Optional[List[str]] = None,
        fused_qkv: bool = False):

        super().__init__()
        assert r > 0, "LoRA rank r must be positive"
        self.fused_qkv = fused_qkv
        self.share_qkv = os.environ.get("SHARE_QKV", "1").lower() in ("1", "true", "yes")
        self.r = r
        self.feature_dim = clip_vision_model.embeddings.patch_embedding.out_channels  #768

        self.use_soft_projection = use_soft_projection
        self.weight_temp = weight_temp
        self.weight_kind = weight_kind
        self.weight_p = weight_p

        self.nsp_eps = nsp_eps
        self.nsp_weight = nsp_weight
        self.projection_param_mode = projection_param_mode
        self.basis_rank = basis_rank

        for n, p in clip_vision_model.named_parameters():
            if include_norm and ("norm" in n or "layernorm" in n.lower()):
                p.requires_grad_(True)
            else:
                p.requires_grad_(False)

        self.lora_layer = list(lora_layer) if lora_layer is not None else list(range(len(clip_vision_model.encoder.layers)))
        self.target_modules = target_modules if target_modules is not None else ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
        self.lora_modules = nn.ModuleDict()

        # 设备和 dtype 推断
        dev = clip_vision_model.embeddings.patch_embedding.weight.device
        dtype = clip_vision_model.embeddings.patch_embedding.weight.dtype

        def make_placeholder(d):
            return FixedProjection(torch.eye(d, device=dev, dtype=dtype))

        # 遍历每一层 Transformer
        for idx, layer in enumerate(clip_vision_model.encoder.layers):
            if idx not in self.lora_layer:
                continue

            # === Self-Attention Projections ===
            # 若启用 fused_qkv 且 q/k/v 三者都存在，则替换为 fused attention；否则走方案 A（共享 P）。
            attn_proj_names = ["q_proj", "k_proj", "v_proj", "out_proj"]
            active_attn = [p for p in attn_proj_names if p in self.target_modules]
            qkv_all_present = all(p in self.target_modules for p in ["q_proj", "k_proj", "v_proj"])
            use_fused = fused_qkv and qkv_all_present

            if use_fused:
                fused_attn = FusedQKVCLIPAttention(layer.self_attn)
                layer.self_attn = fused_attn

                qkv_linear = fused_attn.qkv_proj
                proj = make_placeholder(qkv_linear.in_features)
                lora_mod = lora_class(
                    qkv_linear, r, proj,
                    projection_param_mode=projection_param_mode,
                    basis_rank=basis_rank)
                fused_attn.qkv_proj = lora_mod
                self.lora_modules[f"layer_{idx}_attn_qkv_proj"] = lora_mod

                if "out_proj" in self.target_modules:
                    linear = fused_attn.out_proj
                    proj = make_placeholder(linear.in_features)
                    lora_mod = lora_class(
                        linear, r, proj,
                        projection_param_mode=projection_param_mode,
                        basis_rank=basis_rank)
                    fused_attn.out_proj = lora_mod
                    self.lora_modules[f"layer_{idx}_attn_out_proj"] = lora_mod
            else:
                # 方案 A / 独立 P：q/k/v 输入空间相同，可选共享同一个占位 P；out_proj 保持独立。
                qkv_names = [p for p in ["q_proj", "k_proj", "v_proj"] if p in active_attn]
                shared_qkv_proj = None
                if qkv_names and self.share_qkv:
                    ref_linear = getattr(layer.self_attn, qkv_names[0])
                    shared_qkv_proj = make_placeholder(ref_linear.in_features)

                for proj_name in active_attn:
                    linear = getattr(layer.self_attn, proj_name)
                    proj = shared_qkv_proj if (proj_name in qkv_names and self.share_qkv) else make_placeholder(linear.in_features)
                    lora_mod = lora_class(
                        linear, r, proj,
                        projection_param_mode=projection_param_mode,
                        basis_rank=basis_rank)
                    setattr(layer.self_attn, proj_name, lora_mod)
                    self.lora_modules[f"layer_{idx}_attn_{proj_name}"] = lora_mod

            # === MLP ===
            for mlp_name in ["fc1", "fc2"]:
                if mlp_name not in self.target_modules:
                    continue
                linear = getattr(layer.mlp, mlp_name)
                proj = make_placeholder(linear.in_features)
                lora_mod = lora_class(
                    linear, r, proj,
                    projection_param_mode=projection_param_mode,
                    basis_rank=basis_rank)
                setattr(layer.mlp, mlp_name, lora_mod)
                self.lora_modules[f"layer_{idx}_mlp_{mlp_name}"] = lora_mod

        self.clip_vision_model = clip_vision_model

    @torch.no_grad()
    def _ensure_merged_before_rebuild(self):
        self.merge_lora_weights()

    @torch.no_grad()
    def update_projection_matrices(self, covariances: Dict[str, torch.Tensor]) -> None:
        # 多个模块可能共享同一个 FixedProjection，只需构建一次 P
        seen_proj_ids = set()
        for name, cov in covariances.items():
            if name not in self.lora_modules:
                continue
            module = self.lora_modules[name]
            proj_id = id(module.P)
            if proj_id in seen_proj_ids:
                continue
            seen_proj_ids.add(proj_id)

            P = build_projection(
                cov,
                soft_projection=self.use_soft_projection,
                weight_temp=self.weight_temp,
                weight_kind=self.weight_kind,
                weight_p=self.weight_p,
                nsp_eps=self.nsp_eps,
                nsp_weight=self.nsp_weight)

            # 确保投影矩阵与 LoRA 模块在同一设备
            P = P.to(device=module.A.device, dtype=module.A.dtype)
            module.P = FixedProjection(P)

    @torch.no_grad()
    def update_basis_matrices(self, covariances: Dict[str, torch.Tensor]) -> None:
        """为 basis 模式更新 U_h 并设置 basis_ready=True。

        只在 projection_param_mode 为 fixed_basis / core_basis 时有效。
        """
        if self.projection_param_mode not in ["fixed_basis", "core_basis"]:
            return
        if self.basis_rank is None:
            return

        # 共享 P 的模块也共享 U_h，只需计算一次
        seen_uh_ids = set()
        for name, cov in covariances.items():
            if name not in self.lora_modules:
                continue
            module = self.lora_modules[name]
            uh_id = id(module.U_h)
            if uh_id in seen_uh_ids:
                continue
            seen_uh_ids.add(uh_id)
            U_h = compute_tail_basis(cov, k=self.basis_rank)
            module.U_h = U_h.to(device=module.A.device, dtype=module.A.dtype)
            module.basis_ready = torch.tensor(True)

    @torch.no_grad()
    def initialize_adapters_from_covariance(
        self,
        covariances: Dict[str, torch.Tensor],
        window: str = "tail",
        s_ratio: float = 0.33,
        set_p_to_identity: bool = True,
    ) -> None:
        """Proj-Σ 初始化 LoRA 适配器（支持 SGPBaseLoRA 和 SGPBaseDoRA）。

        对每层：Σ 特征分解 → 取 V_small → BA_init = W @ V_small @ V_small^T
        → SVD → A_init, B_init → W' = W - BA_init。

        Args:
            covariances: Dict[name, cov_matrix]
            window: 'tail'（最小特征值）或 'middle'（中间特征值）
            s_ratio: 'middle' 窗口起始位置比例
            set_p_to_identity: True 时把 P 设为单位阵（hist_null_init_only）；
                               False 时保留当前 P（hist_null_init_runtime）。
        """
        self._ensure_merged_before_rebuild()

        for name, cov in covariances.items():
            if name not in self.lora_modules:
                continue

            module = self.lora_modules[name]
            # 使用有效权重（DoRA 下是 direction * magnitude，LoRA 下是 linear.weight）
            W = module.get_effective_weight()

            A_init, B_init, W_residual = compute_proj_init(
                W, cov, self.r, window=window, s_ratio=s_ratio)

            # 写入初始化后的有效权重和 LoRA 参数
            module.set_effective_weight(W_residual)
            module.set_lora_init(A_init, B_init)

            if set_p_to_identity:
                # P 设为单位矩阵（无运行时投影）
                d = cov.size(0)
                device = module.A.device
                dtype = module.A.dtype
                module.P = FixedProjection(torch.eye(d, device=device, dtype=dtype))

    @torch.no_grad()
    def initialize_adapters_from_weight_svd(
        self,
        window: str = "tail",
        s_ratio: float = 0.33,
    ) -> None:
        """SVD-W 初始化 LoRA 适配器：对当前合并权重做 SVD → 取窗口分量 → 初始化 A, B。

        Args:
            window: 'tail'（最小奇异值）或 'middle'（中间奇异值）
            s_ratio: 'middle' 窗口起始比例
        """
        self._ensure_merged_before_rebuild()

        for name, module in self.lora_modules.items():
            # 使用有效权重（DoRA 下是 direction * magnitude，LoRA 下是 linear.weight）
            W = module.get_effective_weight()

            A_init, B_init, W_residual = compute_weight_svd_init(
                W, self.r, window=window, s_ratio=s_ratio)

            module.set_effective_weight(W_residual)
            module.set_lora_init(A_init, B_init)

            d = W.size(1)
            device = module.A.device
            dtype = module.A.dtype
            module.P = FixedProjection(torch.eye(d, device=device, dtype=dtype))

    def regularization_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=next(self.parameters()).device)

    def forward(self, pixel_values: torch.Tensor, **kwargs) -> torch.Tensor:
        kwargs.setdefault('output_attentions', False)
        return self.clip_vision_model(pixel_values, **kwargs)

    def get_module_names(self):
        return list(self.lora_modules.keys())

    def finalize_without_lora(self) -> None:
        self.eval()
        for _, mod in self.lora_modules.items():
            mod.merge_lora_weights(lora_active=False)

    def merge_lora_weights(self):
        for _, mod in self.lora_modules.items():
            mod.merge_lora_weights()

    def get_params(self):
        params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            else:
                params.append(param)
        return params


def build_projection(
    cov: torch.Tensor,
    soft_projection: bool = True,
    weight_temp: float = 5.0,
    nsp_eps = 0.05, 
    nsp_weight = 0.0,
    *,
    weight_kind: str = "log1p",
    weight_alpha: float = 0.5,
    weight_p: float = 2.0,
    weight_kappa: float = 2 ) -> torch.Tensor:

# --- [修改点] 提升计算精度和稳定性 ---
    # 1. 强制转换为 float64 (双精度)，这是解决 MKL Argument 8 错误的核心
    cov_double = cov.to(torch.float64)
    
    # 2. 检查并清理 NaN 或 Inf，防止脏数据导致崩溃
    if torch.isnan(cov_double).any() or torch.isinf(cov_double).any():
        cov_double = torch.nan_to_num(cov_double, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # 3. 确保矩阵绝对对称（理论上 cov 是对称的，但浮点误差会导致微小不对称，引发 eigh 报错）
    cov_double = (cov_double + cov_double.t()) / 2.0
    
    # 4. 增加正则化项 (岭系数)，将 1e-6 提高到 1e-4 以增强稳定性
    safe_eps = 1e-4
    cov_double = cov_double + safe_eps * torch.eye(cov_double.size(0), device=cov_double.device, dtype=torch.float64)
    
    try:
        # 5. 在双精度下进行特征值分解
        eigvals_double, eigvecs_double = torch.linalg.eigh(cov_double)
    except RuntimeError:
        # [备选方案] 如果 GPU 分解失败，尝试在 CPU 上分解（CPU 的 MKL 库通常比 GPU 更鲁棒）
        logging.warning("GPU eigh failed, falling back to CPU...")
        eigvals_double, eigvecs_double = torch.linalg.eigh(cov_double.cpu())
        eigvals_double = eigvals_double.to(cov.device)
        eigvecs_double = eigvecs_double.to(cov.device)
    
    # 6. 计算完成后，转回模型原本的精度（float16 或 float32）
    # [修改点] 将结果从双精度转回原精度，并移回 GPU (cuda)
    eigvals = eigvals_double.to(dtype=cov.dtype, device='cuda')
    eigvecs = eigvecs_double.to(dtype=cov.dtype, device='cuda')
    # --- [修改结束] ---
    eigvals = torch.abs(eigvals)
    d = cov.size(0)
    sum_vals = eigvals.sum()
    scale_ = d / (sum_vals + safe_eps)
    eigvals = eigvals * scale_


    if soft_projection:
        weights = compute_weights(eigvals, weight_kind, weight_temp, weight_p, weight_alpha, weight_kappa)
        max_weight = weights.max()
        weights = weights / max_weight
        diag_w = torch.diag(weights)
        P = eigvecs @ diag_w @ eigvecs.t()
    else:
        eps_hard = nsp_eps
        total = eigvals.sum()
        cumsum = torch.cumsum(eigvals, dim=0)
        ratio = cumsum / (total + 1e-12)
        idx = (ratio >= eps_hard).nonzero(as_tuple=False)
        m = idx[0].item() if idx.numel() > 0 else eigvals.numel()
        V_keep = eigvecs[:, :m]
        P = V_keep @ V_keep.t()
        I = torch.eye(P.size(0), device=P.device, dtype=P.dtype)
        P = (1 - nsp_weight) * P + nsp_weight * I
    
    # [修改点] 确保返回的 P 矩阵一定在显卡上，与模型权重设备对齐
    P = P.to(device='cuda')
    return P


class LoRACLIPTextTransformer(nn.Module):
    """
    CLIP 文本编码器的 LoRA 包装器（与 LoRACLIPVisionTransformer 对称）

    给 CLIPTextTransformer 的 12 层 Transformer 的 attention(q/k/v/out_proj)
    和 MLP(fc1/fc2) 添加 LoRA（默认 DoRA），共 72 个 LoRA 模块。
    支持 NSP/SGP 零空间投影，防止文本编码器灾难性遗忘。
    """

    def __init__(
        self,
        clip_text_model: nn.Module,
        r: int,
        lora_layer: Optional[Iterable[int]] = None,
        use_soft_projection: bool = True,
        weight_temp: float = 1.0,
        weight_kind: str = "log1p",
        weight_p: float = 1.0,
        nsp_eps: float = 0.05,
        nsp_weight: float = 0.02,
        lora_class: type = SGPBaseDoRA,
        projection_param_mode: str = "full",
        basis_rank: Optional[int] = None,
        target_modules: Optional[List[str]] = None,
        fused_qkv: bool = False,
    ):
        super().__init__()
        assert r > 0
        self.r = r
        self.fused_qkv = fused_qkv
        self.share_qkv = os.environ.get("SHARE_QKV", "1").lower() in ("1", "true", "yes")
        self.use_soft_projection = use_soft_projection
        self.weight_temp = weight_temp
        self.weight_kind = weight_kind
        self.weight_p = weight_p
        self.nsp_eps = nsp_eps
        self.nsp_weight = nsp_weight
        self.projection_param_mode = projection_param_mode
        self.basis_rank = basis_rank

        for p in clip_text_model.parameters():
            p.requires_grad_(False)

        self.lora_layer = list(lora_layer) if lora_layer is not None else list(
            range(len(clip_text_model.encoder.layers))
        )
        self.target_modules = target_modules if target_modules is not None else ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
        self.lora_modules = nn.ModuleDict()

        dev = clip_text_model.embeddings.token_embedding.weight.device
        dtype = clip_text_model.embeddings.token_embedding.weight.dtype

        def make_placeholder(d):
            return FixedProjection(torch.eye(d, device=dev, dtype=dtype))

        for idx, layer in enumerate(clip_text_model.encoder.layers):
            if idx not in self.lora_layer:
                continue

            # === Self-Attention Projections ===
            # 若启用 fused_qkv 且 q/k/v 三者都存在，则替换为 fused attention；否则走方案 A（共享 P）。
            attn_proj_names = ["q_proj", "k_proj", "v_proj", "out_proj"]
            active_attn = [p for p in attn_proj_names if p in self.target_modules]
            qkv_all_present = all(p in self.target_modules for p in ["q_proj", "k_proj", "v_proj"])
            use_fused = fused_qkv and qkv_all_present

            if use_fused:
                fused_attn = FusedQKVCLIPAttention(layer.self_attn)
                layer.self_attn = fused_attn

                qkv_linear = fused_attn.qkv_proj
                proj = make_placeholder(qkv_linear.in_features)
                lora_mod = lora_class(
                    qkv_linear, r, proj,
                    projection_param_mode=projection_param_mode,
                    basis_rank=basis_rank)
                fused_attn.qkv_proj = lora_mod
                self.lora_modules[f"layer_{idx}_attn_qkv_proj"] = lora_mod

                if "out_proj" in self.target_modules:
                    linear = fused_attn.out_proj
                    proj = make_placeholder(linear.in_features)
                    lora_mod = lora_class(
                        linear, r, proj,
                        projection_param_mode=projection_param_mode,
                        basis_rank=basis_rank)
                    fused_attn.out_proj = lora_mod
                    self.lora_modules[f"layer_{idx}_attn_out_proj"] = lora_mod
            else:
                # 方案 A / 独立 P：q/k/v 输入空间相同，可选共享同一个占位 P；out_proj 保持独立。
                qkv_names = [p for p in ["q_proj", "k_proj", "v_proj"] if p in active_attn]
                shared_qkv_proj = None
                if qkv_names and self.share_qkv:
                    ref_linear = getattr(layer.self_attn, qkv_names[0])
                    shared_qkv_proj = make_placeholder(ref_linear.in_features)

                for proj_name in active_attn:
                    linear = getattr(layer.self_attn, proj_name)
                    proj = shared_qkv_proj if (proj_name in qkv_names and self.share_qkv) else make_placeholder(linear.in_features)
                    lora_mod = lora_class(
                        linear, r, proj,
                        projection_param_mode=projection_param_mode,
                        basis_rank=basis_rank)
                    setattr(layer.self_attn, proj_name, lora_mod)
                    self.lora_modules[f"layer_{idx}_attn_{proj_name}"] = lora_mod

            # === MLP ===
            for mlp_name in ["fc1", "fc2"]:
                if mlp_name not in self.target_modules:
                    continue
                linear = getattr(layer.mlp, mlp_name)
                proj = make_placeholder(linear.in_features)
                lora_mod = lora_class(
                    linear, r, proj,
                    projection_param_mode=projection_param_mode,
                    basis_rank=basis_rank)
                setattr(layer.mlp, mlp_name, lora_mod)
                self.lora_modules[f"layer_{idx}_mlp_{mlp_name}"] = lora_mod

        self.clip_text_model = clip_text_model

    @torch.no_grad()
    def _ensure_merged_before_rebuild(self):
        self.merge_lora_weights()

    def update_projection_matrices(self, covariances: Dict[str, torch.Tensor]) -> None:
        self._ensure_merged_before_rebuild()
        # 多个模块可能共享同一个 FixedProjection，只需构建一次 P
        seen_proj_ids = set()
        for name, cov in covariances.items():
            if name not in self.lora_modules:
                continue
            module = self.lora_modules[name]
            proj_id = id(module.P)
            if proj_id in seen_proj_ids:
                continue
            seen_proj_ids.add(proj_id)
            P = build_projection(
                cov,
                soft_projection=self.use_soft_projection,
                weight_temp=self.weight_temp,
                weight_kind=self.weight_kind,
                weight_p=self.weight_p,
                nsp_eps=self.nsp_eps,
                nsp_weight=self.nsp_weight,
            )
            self.lora_modules[name].P = FixedProjection(P)

    @torch.no_grad()
    def update_basis_matrices(self, covariances: Dict[str, torch.Tensor]) -> None:
        if self.projection_param_mode not in ["fixed_basis", "core_basis"]:
            return
        if self.basis_rank is None:
            return
        # 共享 P 的模块也共享 U_h，只需计算一次
        seen_uh_ids = set()
        for name, cov in covariances.items():
            if name not in self.lora_modules:
                continue
            module = self.lora_modules[name]
            uh_id = id(module.U_h)
            if uh_id in seen_uh_ids:
                continue
            seen_uh_ids.add(uh_id)
            U_h = compute_tail_basis(cov, k=self.basis_rank)
            module.U_h = U_h.to(device=module.A.device, dtype=module.A.dtype)
            module.basis_ready = torch.tensor(True)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        kwargs.setdefault('output_attentions', False)
        return self.clip_text_model(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    def get_module_names(self):
        return list(self.lora_modules.keys())

    def merge_lora_weights(self):
        for mod in self.lora_modules.values():
            mod.merge_lora_weights()

    def get_params(self):
        return [p for p in self.parameters() if p.requires_grad]