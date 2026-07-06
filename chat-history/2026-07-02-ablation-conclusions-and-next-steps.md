# 消融实验结论与下一步计划

**日期**: 2026-07-02
**背景**: 已完成 LoRA/DoRA 对比、优化器、SGD 学习率、AdamW 学习率、batch size、蒸馏/辅助头系数的单点消融。

---

## 一、当前可确认的结论

### 1. LoRA vs DoRA / 变体选择
- LoRA 在 Average/Last 上系统优于 DoRA（前期工作）。
- `current_nsp`（无 history null init）与 `hist_null_init_runtime` 表现非常平衡：
  - `hist_null_init_runtime` 的 Transfer 略高；
  - `current_nsp` 的 Last/Average 略高。
- 默认配置可继续以 `current_nsp` 为主。

### 2. 优化器
- **RMSprop、AdamW、Adam** 是第一梯队，指标接近。
- 按 `(Transfer + Last) / 2` 与 Last 综合看，**RMSprop** 略占优。
- SGD 并非不可用：默认 1e-3 学习率较弱；将 lr 提升到 **5e-3** 后，
  - `current_nsp` 的 Ens Average 达到 **69.99**，Ens Last 达到 **82.24**；
  - 但 Transfer 明显下降（ZS 57.44，Ens 57.59）。
- 因此 SGD 的最佳学习率落在 **1e-3 到 5e-3 之间**。

### 3. 学习率（AdamW / RMSprop）
- **1e-4 是最佳平衡点**：
  - ZS (T+L)/2 = 69.22，Ens (T+L)/2 = 70.59。
- 3e-4 的 Last 最高（ZS 79.56，Ens 81.87），但 Transfer 下降明显（58.25/58.29）。
- 5e-5 过小，6e-4 与 3e-4 接近但略差。

### 4. Batch Size
- 越大越好：
  - `bs=128` 的 Ens Average 最高（70.14），Ens Last 最高（81.01）。
  - `bs=64` 接近，`bs=32` 明显落后。
- **当前阶段保持 64**（训练/调试效率更高）；
- **最终大规模收集实验时使用 128**。

### 5. Feature Distillation (`fd_weight`)
- `fd_weight ∈ {0.0, 1.0, 2.0}` 的结果几乎无差别。
- 说明在当前 feature-level distillation 形式下，权重不敏感。
- 后续可保持默认值或设为 0.0。

### 6. Auxiliary Head (`aux_weight`)
- `aux=0.0` 的 Average 最好，但 `aux=1.0` 的 Last 略高。
- 综合便利性与收益，**默认权重取 0.0**。

### 7. Class Distillation (`cd_weight`)
- `cd=2.0` 的 Transfer 和 Last 均优于 `cd=1.0` 和 `cd=0.0`：
  - ZS Last 77.67，Ens Last 80.27。
- 但提升幅度有限，可能存在饱和。

---

## 二、尚未充分探索的问题

1. **SGD 学习率的精细搜索**
   - 当前仅测了 1e-3、5e-3 以及灾难性的 5e-2。
   - 需要在 1e-3 到 5e-3 之间做更细网格（如 2e-3、3e-3、4e-3），绘制 Transfer-Last Pareto 曲线。

2. **`cd_weight` 的饱和行为**
   - 已测 {0,1,2}，2.0 最好但提升有限。
   - 需要测试 3.0、4.0 甚至更高，确认是饱和还是继续上升。

3. **KL 散度的温度系数**
   - 当前 class distillation 使用默认温度（通常为 1.0）。
   - 温度系数会显著改变分布匹配的“软硬”程度，尚未扫描。

4. **其他散度/距离公式**
   - 除了 KL，可尝试反向 KL、JS 散度、Wasserstein 距离、对比散度等。
   - 也可尝试基于 logits 的 MSE 或 cosine 相似度。

5. **蒸馏对象/教师选择**
   - 当前蒸馏来自上一个任务或历史平均模型。
   - 可探讨是否使用任务-specific 教师、ensemble 教师或 frozen zero-shot 教师。

6. **超参数之间的交互**
   - 学习率与 batch size 的交互：bs=128 时最优 lr 是否仍是 1e-4？
   - cd_weight 与学习率的交互。

7. **多种子验证**
   - 当前全部为 seed=42 的单点结果。
   - 关键配置需要用多个种子验证稳定性。

8. **最终大规模配置**
   - 使用 `current_nsp + RMSprop/AdamW + lr=1e-4 + bs=128 + cd=2.0 + aux=0.0` 做完整跑分。
   - 与 `hist_null_init_runtime` 同配置做最终对比。

---

## 三、下一步具体计划

1. **近期（机制探索）**
   - SGD lr 精细搜索：{2e-3, 3e-3, 4e-3} × `current_nsp`。
   - `cd_weight` 扩展：{3.0, 4.0} × `current_nsp + AdamW + lr=1e-4`。
   - KL 温度系数：{0.5, 1.0, 2.0, 4.0} × `cd_weight=2.0`。

2. **中期（方法变体）**
   - 实现并测试 2–3 种替代 class distillation 散度（如 reverse KL、JS）。
   - 测试 `bs=128` 下最优 lr 是否需要调整。

3. **远期（最终验证）**
   - 选定 2–3 组最优配置，跑 3 个种子。
   - 生成最终对比表格与图表。

---

## 四、当前推荐默认配置

```text
variant          = current_nsp
optimizer        = RMSprop 或 AdamW
lr               = 1e-4
batch_size       = 64（调试）/ 128（最终）
fd_weight        = 0.0 或 1.0（不敏感）
cd_weight        = 2.0
aux_weight       = 0.0
lora_rank        = 4
use_dora         = false
```

---

*记录者: Kimi Code CLI*
