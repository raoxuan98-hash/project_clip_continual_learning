# LoRA-NSP Basis/Null Ablation 数学公式梳理与 DoRA 初始化 Bug

**日期**: 2026-06-30
**会话概况**: 阅读 `docs/lora_null_basis_ablation_plan.md` 并从数学公式层面逐个讨论 6 个变体；发现当前 DoRA 实现下 `initialize_adapters_from_covariance` 存在 bug；确认 basis 参数化可与 DoRA 结合。

---

## 1. 关键讨论 / 决策

- **变体设计目标**: 本轮消融要回答两个问题：(1) `BAP` 能否重参数化为显式低秩 basis；(2) LoRA-Null-style 初始化能否替代/增强 runtime NSP。
- **current_nsp 实际默认配置**: `main_incremental.py` / `main_joint.py` 默认 `--lora_type lora_nsp`，对应 `use_soft_projection=False`，即 hard projection：`P = 0.98 * V_keep V_keep^T + 0.02 * I`，子空间维度 m 由 `nsp_eps=0.05` 自适应决定。不是 soft projection。
- **basis_fixed_tail 是强约束**: 由于 current_nsp hard 使用自适应 m 维子空间，而 basis_fixed_tail 使用固定 k 维子空间，两者不是简单参数化冗余关系。建议让 current_nsp 也支持固定维度以便公平对比。
- **k=128 是合理候选**: 基于 hard NSP 的 m 可能在 100-300 之间，k=128 可作为 basis 变体的起点，同时建议扫描 {32, 64, 128, 256}。
- **DoRA 可写成 W = W_0 + ΔW**: 当前代码 `adapted_weight = (weight_directions + lora_delta) * magnitude` 等价于 `W = W_0 + ||W_0|| * lora_delta`，因此 basis 参数化可以不改 DoRA 结构，只替换 `lora_delta` 计算。
- **保留 DoRA 进行 ablation**: 不必为了机制消融切回纯 LoRA；在 DoRA 下重新定义 `lora_delta` 更贴近主流程。

## 2. 重要发现

- **U_h 索引修正**: 代码中 `torch.linalg.eigh` 返回升序特征值，最小 k 个方向应为 `V[:, 0:k]`（0-based），不是 `V[:, 1:k]`。
- **soft projection 的 P 是满秩加权矩阵**: 默认 soft 模式下 `P = V diag(w_i) V^T` 且所有 `w_i > 0`，不是投影矩阵。只有 hard 模式（`use_soft_projection=False, nsp_weight=0`）才得到真正的投影。
- **LoRA-Null 初始化理解**: `BA_init = W_0 U_null U_null^T` 不是把 A 初始化为 `U_null^T`，而是把乘积 `BA` 初始化为 `W_0` 在激活 null/low-energy 子空间上的投影，然后通过 SVD 分解得到 A_init, B_init，保证 `W_0' + BA_init = W_0`。
- **当前 DoRA 下 history-null init 有 bug**: `initialize_adapters_from_covariance` 直接读取 `module.linear.weight.data`，但 DoRA 的 forward 实际使用 `weight_directions * magnitude`，`linear.weight` 在初始化后失效。导致 init 基于错误权重且写回无效内存。
- **DoRA 下 LoRA-Null init 需要缩放 B**: DoRA 的 effective delta 是 `magnitude * (B A)`，因此把 `BA_init` 写入 A/B 时，需要将 B 按 `1/magnitude_new` 缩放，才能保证 `W_residual + magnitude_new * (B A) = W_eff`。
- **DoRA merge 的当前实现未归一化**: `weight_directions += lora_delta` 后直接乘 magnitude，缺少原 DoRA 论文中的 `||direction + delta||` 归一化。但当前实现是自洽的，可视为一种简化版 DoRA。
- **修复已验证**: 新增 `get_effective_weight` / `set_effective_weight` / `set_lora_init` helper，并用 `test_dora_init_fix.py` 验证 SGPBaseLoRA 和 SGPBaseDoRA 的 init 前后功能等价。
- **basis 变体已实现**: `fixed_basis` (`ΔW = B U_h^T`) 和 `core_basis` (`ΔW = B C U_h^T`) 在 `SGPBaseLoRA` 和 `SGPBaseDoRA` 中实现，Task 1 无 history 时自动回退到 full 模式。
- **统一后处理逻辑**: `main_incremental.py` 根据 `projection_param_mode` 和 `null_init_mode` 自动决定更新 projection / basis / 都不更新。

## 3. 待办事项 / 遗留问题

- [x] 修复 `initialize_adapters_from_covariance` 对 DoRA 的支持：读取 `weight_directions * magnitude` 作为有效权重，并把 residual 写回 direction/magnitude。
- [x] 修复 `initialize_adapters_from_weight_svd` 对 DoRA 的支持。
- [x] 设计数值自检：init 前后 `W_eff' + effective_delta == W_eff`，forward 等价。
- [x] 实现 `projection_param_mode` CLI 参数：`full | fixed_basis | core_basis`。
- [x] 实现 `null_init_mode` CLI 参数：`none | history_init_only | history_init_runtime`。
- [x] 在 DoRA 结构下实现 basis 变体的 `lora_delta` 分支。
- [x] `SGPBaseLoRA` / `SGPBaseDoRA` 支持 `fixed_basis` 和 `core_basis`，Task 1 自动回退 full 模式。
- [x] `LoRACLIPVisionTransformer` / `LoRACLIPTextTransformer` 新增 `update_basis_matrices`。
- [x] `main_incremental.py` 集成新参数和统一的 task 后处理逻辑。
- [x] 新增 `test_basis_variants.py` 并通过测试。
- [ ] 决定是否让 current_nsp hard 支持固定子空间维度 `--nsp_fixed_dim` 以公平对比 basis 变体。
- [ ] 实际测量当前 hard NSP 在各层的 m 值，为 k 选择提供依据。

## 4. 相关文件

- `docs/lora_null_basis_ablation_plan.md`: 本次讨论的消融计划文档。
- `src/models/lora_sgp.py`: LoRA/DoRA/NSP 核心实现，包含 `compute_proj_init`、`initialize_adapters_from_covariance`、`SGPBaseLoRA`、`SGPBaseDoRA`。
- `src/models/clip.py`: `lora_nsp` / `lora_sgp` 类型映射与默认参数。
- `main_incremental.py` / `main_joint.py`: 默认 `--lora_type lora_nsp --nsp_eps 0.05 --nsp_weight 0.02`。
- `lora_null_llm/paper/2503.02659v2.pdf`: LoRA-Null 原始论文，明确 `BA_init = W_0 U_null U_null^T`。
- `test_dora_init_fix.py`: DoRA/LoRA init 等价性测试。
- `test_basis_variants.py`: basis 参数化变体正确性测试。

---

## 5. 实现完成与验证（会话末尾更新）

### 已落地代码

- `src/models/lora_sgp.py`: 完成 `SGPBaseLoRA` / `SGPBaseDoRA` 的 `full/fixed_basis/core_basis` 三模式支持；新增 `compute_tail_basis`、`update_basis_matrices`；修复并验证 DoRA 下 `initialize_adapters_from_covariance` / `initialize_adapters_from_weight_svd`。
- `src/models/clip.py`: `lora_nsp` / `lora_sgp` 均透传 `projection_param_mode` / `basis_rank`。
- `src/trainers/lora_nsp_trainer.py`: `update_covariance_history` / `update_text_covariance_history` 新增 `update_projection` / `update_basis` 开关。
- `main_incremental.py`: 新增 `--projection_param_mode`、`--basis_rank`、`--basis_window`、`--null_init_mode`；统一 task 前后处理；保留旧 `init_mode` 兼容。

### 测试命令与结果

```bash
python test_dora_init_fix.py
python test_basis_variants.py
```

全部通过：
- `test_dora_init_fix.py`: SGPBaseLoRA / SGPBaseDoRA 的 init 前后 output diff < 1e-6。
- `test_basis_variants.py`: fixed_basis / core_basis 在 LoRA 和 DoRA 下 merge 前后等价，effective delta rank 正确。

### 当前状态

代码已 ready，未提交（等待用户 review）。分支 `v2-text-lora` 上除本次改动外还有其他未提交文件。
