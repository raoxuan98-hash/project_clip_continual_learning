# LoRA-NSP 中 Q/K/V 投影冗余与两种改造方案

**日期**: 2026-07-05  
**会话主题**: 当前实现里同一层的 q_proj/k_proj/v_proj 会重复收集协方差、重复做 eigendecomposition、重复存储投影矩阵。这里把两个改造方案（共享 P vs. 融合 QKV）写清楚，供后续实现和消融参考。

---

## 1. 问题确认

在 `src/models/lora_sgp.py` 中，`LoRACLIPVisionTransformer` / `LoRACLIPTextTransformer` 对每一层的 attention 投影分别包装：

```python
for proj_name in ["k_proj", "v_proj", "q_proj", "out_proj"]:
    linear = getattr(layer.self_attn, proj_name)
    proj = make_placeholder(linear.in_features)          # 每个模块一个 P
    lora_mod = lora_class(linear, r, proj, ...)
    self.lora_modules[f"layer_{idx}_attn_{proj_name}"] = lora_mod
```

在 `src/trainers/lora_nsp_trainer.py` 中，`_extract_covariances_from_modules` 对每个 `lora_modules` 的 key 单独挂 hook：

```python
for name in module_names:
    extractor = FeatureExtractorHook()
    hook = lora_modules[name].register_forward_hook(extractor)
    ...
    xtx_batch = batch_feats.t() @ batch_feats
    running_xtx[name] += xtx_batch
```

由于同一 transformer 层内 q/k/v 的输入张量完全相同（都是 layer-norm 后的 attention 输入），`X^T X` 对 q/k/v 来说完全相同， ensuing 的三次 eigendecomposition、三次 `build_projection`、三个 `d×d` 的 P 矩阵都是冗余。

---

## 2. 方案 A：仅共享 P 矩阵（推荐先实现）

### 核心思想
保持 q/k/v 为三个独立的 `nn.Linear` 和三个独立的 LoRA 参数（A/B），只让同层 q/k/v 共享同一个 `FixedProjection` 对象，并在协方差提取时只算一次。

### 为什么这是安全的
P 矩阵只依赖于输入空间的协方差。同一层 q/k/v 输入空间相同，因此它们对应的 P 本来就应该是同一个矩阵。共享 P 不引入任何新的学习约束。

### 具体改造点

#### 2.1 `src/models/lora_sgp.py`

在 `LoRACLIPVisionTransformer.__init__` 的层遍历中：

```python
for idx, layer in enumerate(clip_vision_model.encoder.layers):
    if idx not in self.lora_layer:
        continue

    # === Self-Attention Projections ===
    # 先确定本层哪些 attention 投影需要 LoRA
    attn_proj_names = ["q_proj", "k_proj", "v_proj", "out_proj"]
    active_attn = [p for p in attn_proj_names if p in self.target_modules]

    # q/k/v 共享输入空间，复用同一个占位 P
    qkv_names = [p for p in ["q_proj", "k_proj", "v_proj"] if p in active_attn]
    shared_qkv_proj = None
    if qkv_names:
        # 取任意一个的 in_features（三者相同）
        ref_linear = getattr(layer.self_attn, qkv_names[0])
        shared_qkv_proj = make_placeholder(ref_linear.in_features)

    for proj_name in active_attn:
        linear = getattr(layer.self_attn, proj_name)
        proj = shared_qkv_proj if proj_name in qkv_names else make_placeholder(linear.in_features)
        lora_mod = lora_class(linear, r, proj, ...)
        setattr(layer.self_attn, proj_name, lora_mod)
        self.lora_modules[f"layer_{idx}_attn_{proj_name}"] = lora_mod

    # === MLP（输入空间不同，保持独立 P）===
    for mlp_name in ["fc1", "fc2"]:
        if mlp_name not in self.target_modules:
            continue
        linear = getattr(layer.mlp, mlp_name)
        proj = make_placeholder(linear.in_features)
        lora_mod = lora_class(linear, r, proj, ...)
        setattr(layer.mlp, mlp_name, lora_mod)
        self.lora_modules[f"layer_{idx}_mlp_{mlp_name}"] = lora_mod
```

`LoRACLIPTextTransformer` 做对称修改。

> 注意：`ModuleDict` 中多个 key 引用同一个 `nn.Module`（`FixedProjection`）时，`state_dict()` 会按 key 重复保存该 buffer。如果希望 checkpoint 体积也下降，需要额外自定义 `state_dict`/`load_state_dict`，或改为只用一个 key 存共享 P。第一步可先不管 checkpoint 体积，先验证训练等价性。

#### 2.2 `src/trainers/lora_nsp_trainer.py`

在 `_extract_covariances_from_modules` 中按输入源去重 hook：

```python
def _extract_covariances_from_modules(self, lora_modules, forward_fn, data_iter, desc="Collecting features"):
    torch.cuda.empty_cache()
    self.model.eval()

    module_names = list(lora_modules.keys())

    # === 新增：按输入源分组 ===
    def _group_key(name):
        # 例如 "layer_0_attn_q_proj" -> "layer_0_attn_qkv"
        # out_proj / fc1 / fc2 保持独立
        import re
        m = re.match(r"layer_(\d+)_attn_(q_proj|k_proj|v_proj)$", name)
        if m:
            return f"layer_{m.group(1)}_attn_qkv_shared"
        return name

    groups = {}
    for name in module_names:
        gkey = _group_key(name)
        groups.setdefault(gkey, []).append(name)

    # 只给每个组挂一个 hook
    hooks, feature_extractors, running_xtx = {}, {}, {}
    representative = {}
    for gkey, names in groups.items():
        rep = names[0]
        representative[gkey] = rep
        extractor = FeatureExtractorHook()
        hook = lora_modules[rep].register_forward_hook(extractor)
        hooks[gkey] = hook
        feature_extractors[gkey] = extractor
        running_xtx[gkey] = None

    total_observations = 0
    for batch_data in tqdm(data_iter, desc=desc, leave=False):
        batch_input = batch_data[0] if isinstance(batch_data, (tuple, list)) else batch_data
        batch_input = batch_input.to(self.device)
        _ = forward_fn(batch_input)

        for gkey in groups:
            batch_feats = feature_extractors[gkey].get_features()
            if batch_feats is not None:
                batch_feats = batch_feats.to(torch.float32)
                if batch_feats.dim() == 3:
                    batch_feats = batch_feats.reshape(-1, batch_feats.shape[-1])
                xtx_batch = batch_feats.t() @ batch_feats
                if running_xtx[gkey] is None:
                    running_xtx[gkey] = xtx_batch
                else:
                    running_xtx[gkey] += xtx_batch
                feature_extractors[gkey].clear()
                total_observations += batch_feats.shape[0]

    covariances = {}
    for gkey, names in groups.items():
        if running_xtx[gkey] is not None:
            cov = running_xtx[gkey] / total_observations
            cov = (cov + cov.t()) / 2.0
            cov = cov + torch.eye(cov.shape[0], device=cov.device) * 1e-6
            if torch.isnan(cov).any() or torch.isinf(cov).any():
                cov = torch.nan_to_num(cov, nan=0.0, posinf=1.0, neginf=-1.0)
            # 同一协方差复制给组内所有模块名字
            for name in names:
                covariances[name] = cov.to('cpu')
        hooks[gkey].remove()

    return covariances
```

#### 2.3 `update_projection_matrices` / `update_basis_matrices`

由于多个模块现在引用同一个 `FixedProjection` 对象，`build_projection` 只需调用一次。实现上最简单的方式：

```python
@torch.no_grad()
def update_projection_matrices(self, covariances: Dict[str, torch.Tensor]) -> None:
    self._ensure_merged_before_rebuild()
    seen_proj_ids = set()
    for name, cov in covariances.items():
        if name not in self.lora_modules:
            continue
        module = self.lora_modules[name]
        proj_id = id(module.P)
        if proj_id in seen_proj_ids:
            continue
        seen_proj_ids.add(proj_id)
        P = build_projection(cov, soft_projection=self.use_soft_projection, ...)
        module.P = FixedProjection(P)
```

`update_basis_matrices` 同理，用 `id(module.U_h)` 去重。

### 预期收益
- 显存：每层 q/k/v 的 P 矩阵从 3 个降到 1 个。以 ViT-L/14（`d=1024`，24 层）为例，约省 48 个 `1024×1024` 矩阵 ≈ 192 MB。
- 后处理时间：covariance extraction 的 hook 数从 `3×L` 降到 `L`；`build_projection` 调用次数同比例下降。
- 精度：理论上无变化。

### 验证方式
- 在 `aircraft` 上跑 1-iteration smoke test，对比共享 P 前后的 `Transfer / Average / Last`，要求差异 < 0.1%。
- 检查 `nvidia-smi` 或 `torch.cuda.memory_summary` 确认显存下降。

---

## 3. 方案 B：Timm 式 QKV 融合

### 核心思想
不保持三个独立线性层，而是把 `q_proj/k_proj/v_proj` 合并成一个 `qkv_proj`（类似 Timm ViT），并配套自定义 Attention 模块。

### 这会改变什么
这不仅是“共享 P”，而是改变了 LoRA 的参数化和表达能力：
- 原来：3 个独立 rank-r LoRA delta，总参数量 `3 × r × (d + d) = 6rd`。
- 融合后：1 个 rank-r LoRA delta 作用在 `[3d, d]` 上，参数量 `r × (d + 3d) = 4rd`。
- 融合后 Q/K/V 的更新共享同一个低秩列空间，不能独立调整。

### 具体改造点

#### 3.1 新增 `FusedQKVCLIPAttention`

```python
class FusedQKVCLIPAttention(nn.Module):
    def __init__(self, orig_attn: nn.Module):
        super().__init__()
        self.num_heads = orig_attn.num_heads
        self.head_dim = orig_attn.head_dim
        self.scale = self.head_dim ** -0.5
        # 复用或新建 qkv_proj
        embed_dim = orig_attn.q_proj.in_features
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=orig_attn.q_proj.bias is not None)
        # 用原 q/k/v 的权重初始化 qkv_proj
        with torch.no_grad():
            self.qkv_proj.weight.copy_(torch.cat([
                orig_attn.q_proj.weight.data,
                orig_attn.k_proj.weight.data,
                orig_attn.v_proj.weight.data,
            ], dim=0))
            if orig_attn.q_proj.bias is not None:
                self.qkv_proj.bias.copy_(torch.cat([
                    orig_attn.q_proj.bias.data,
                    orig_attn.k_proj.bias.data,
                    orig_attn.v_proj.bias.data,
                ], dim=0))
        self.out_proj = orig_attn.out_proj
        # 其他：dropout、causal_attention_mask 等

    def forward(self, hidden_states, attention_mask=None, causal_attention_mask=None, output_attentions=False):
        bsz, tgt_len, embed_dim = hidden_states.size()
        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.reshape(bsz, tgt_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        query_states, key_states, value_states = qkv[0], qkv[1], qkv[2]
        # 后续与 CLIPAttention 一致：scale、softmax、dropout、out_proj
        ...
```

#### 3.2 在 `LoRACLIPVisionTransformer.__init__` 中替换 attention

当 `target_modules` 包含 `qkv_proj` 时，把 `layer.self_attn` 替换为 `FusedQKVCLIPAttention`：

```python
if "qkv_proj" in self.target_modules:
    layer.self_attn = FusedQKVCLIPAttention(layer.self_attn)
    qkv_linear = layer.self_attn.qkv_proj
    qkv_proj = make_placeholder(qkv_linear.in_features)
    lora_mod = lora_class(qkv_linear, r, qkv_proj, ...)
    layer.self_attn.qkv_proj = lora_mod
    self.lora_modules[f"layer_{idx}_attn_qkv_proj"] = lora_mod

    # out_proj 仍独立包装
    if "out_proj" in self.target_modules:
        ...
```

#### 3.3 CLI 兼容
- `--lora_target_modules qkv_proj,out_proj,fc1,fc2` 走融合分支。
- `--lora_target_modules q_proj,k_proj,v_proj,out_proj,fc1,fc2` 走独立分支（可复用方案 A 的共享 P）。

### 预期收益
- 参数量降到原来的 2/3（q/k/v 部分）。
- 融合后的 `qkv_proj` 是一个更大的 GEMM，对 GPU 更友好。

### 风险
- **表达能力变化**：Q/K/V 更新不再独立，可能损害性能。需要单独消融。
- **实现复杂**：需要重写 attention forward，处理 bias、head split、causal mask、`output_attentions`、文本/视觉 attention 差异等。
- **与现有最佳配置不可直接比较**：之前所有 lora_nsp 消融都基于独立 q/k/v，切换后可能需要重新调 rank/alpha。

### 验证方式
- 在 `aircraft` 上跑 1-iteration smoke test，确认输出 shape 与原始一致。
- 把 `qkv_proj` 作为新变体加入 layer 消融表，和 `layer_qk_only`、`layer_ffn_only`、`layer_attn_only` 等一起比较。

---

## 4. 建议的执行顺序

1. **先实施方案 A**。它直接解决冗余问题，改动小，不改变学习动态，可立即与当前最佳配置兼容。
2. **验证方案 A** 的精度等价性和显存收益。
3. **再决定是否实施方案 B**。方案 B 更适合作为“独立 q/k/v LoRA vs. 融合 qkv LoRA”的独立架构消融，而不是 NSP 投影优化的默认实现。

---

## 5. 当前阻塞

- `experiments/layer_rank_ablation/` 正在服务器运行，修改 `src/models/lora_sgp.py` 和 `src/trainers/lora_nsp_trainer.py` 不会影响已加载模块的进程，但为避免意外，建议在空闲 GPU（当前 GPU 2/5 空闲）上做 smoke test，或等 layer-rank 消融完成后再合并到主分支。

---

## 6. 相关文件

- `src/models/lora_sgp.py`
- `src/trainers/lora_nsp_trainer.py`
- `src/models/clip.py`（方案 B 需要透传新 target module）
- `main_incremental.py`（方案 B 需要兼容 `qkv_proj` CLI 选项）


---

## 7. 实现状态更新（2026-07-05 晚间）

### 已落地代码

- `src/models/lora_sgp.py`
  - 新增 `FusedQKVCLIPAttention`：将 `q_proj/k_proj/v_proj` 合并为单个 `qkv_proj`，复用原 CLIP attention 的 `eager_attention_forward` / `ALL_ATTENTION_FUNCTIONS` 接口。
  - `LoRACLIPVisionTransformer` / `LoRACLIPTextTransformer` 新增 `fused_qkv` 参数；当 `fused_qkv=True` 且 `q_proj/k_proj/v_proj` 同时在 `target_modules` 中时，替换为 fused attention，否则保持方案 A（共享 P）。
- `src/trainers/lora_nsp_trainer.py`
  - `_extract_covariances_from_modules` 的分组逻辑已能兼容 `qkv_proj` 模块名（作为独立组）。
- `src/models/clip.py`
  - 对 `lora_nsp` / `lora_sgp` 分支透传 `fused_qkv=getattr(args, 'fused_qkv', False)`。
- `main_incremental.py`
  - 新增 `--fused_qkv` 命令行参数（默认 False）。
- `main_joint.py`
  - 新增 `--lora_target_modules` 和 `--fused_qkv` 命令行参数（默认 False）。

### 验证结果

#### main_incremental.py（--fused_qkv）
- 命令：`aircraft / iterations=1 / --lora_target_modules q_proj,k_proj,v_proj,out_proj,fc1,fc2 --fused_qkv`
- 结果：跑完全流程无报错。
- LADA+ZS：Average 33.0 / Last 33.0。

#### main_joint.py（--fused_qkv）
- 命令：`--id_datasets aircraft --iterations 1 --tune_vision_encoder true --lora_target_modules q_proj,k_proj,v_proj,out_proj,fc1,fc2 --fused_qkv`
- 结果：跑完全流程无报错。
- ID Average：ZS 22.4% / Ensemble 34.0% / LADA+ZS 28.1%。
- OOD Average：ZS 64.0% / Ensemble 64.3% / LADA+ZS 0.0%（RGDA/LADA 分支为 0 是因为 joint 1-iteration 未充分训练分类器，属预期）。

#### main_incremental.py（默认非 fused）
- 命令：`aircraft / iterations=1 / --lora_target_modules q_proj,k_proj,v_proj,out_proj,fc1,fc2`
- 结果：跑完全流程无报错，行为与改动前一致。

### 使用方式

```bash
# 方案 A（默认）：独立 q/k/v，共享 P
python main_incremental.py ... --lora_target_modules q_proj,k_proj,v_proj,out_proj,fc1,fc2

# 方案 B：Timm 式 fused QKV
python main_incremental.py ... --lora_target_modules q_proj,k_proj,v_proj,out_proj,fc1,fc2 --fused_qkv

# 只有 q/k 时自动退回方案 A（无法融合）
python main_incremental.py ... --lora_target_modules q_proj,k_proj --fused_qkv
```

### 已知限制

- `FusedQKVCLIPAttention` 依赖 transformers 内部 API `eager_attention_forward` / `ALL_ATTENTION_FUNCTIONS`。当前环境 transformers 版本可用，但升级可能不兼容。如需彻底解耦，可后续把 eager attention 实现内嵌到本文件中。
- `ModuleDict` 中多个 key 引用同一个 `FixedProjection` 对象时，`state_dict()` 会按 key 重复保存 buffer；方案 A 的 checkpoint 体积未完全下降。这是已知问题，不影响训练。
- 融合后 Q/K/V 的 LoRA 更新共享同一个低秩列空间，是表达能力与参数效率的权衡，需要后续消融验证长期训练效果。
