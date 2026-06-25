# main_incremental.py LADA Metric Alignment

**日期**: 2026-06-16
**会话概况**: 用户质疑 06-16 num_centers 实验的 Transfer/Average 与 LADA 不对齐。本次核查确认旧的 `src/experiments/run_continual_learning.py` 和 `src/utils/continual_metrics.py` 已精确对齐 LADA，但 `main_incremental.py` 的 06-16 路径只评估已学任务，现已修复为完整 KxK 评估矩阵。

---

## 1. 关键讨论 / 决策

- 用户记忆正确：仓库中已有 LADA 对齐实现，主要在 `src/utils/continual_metrics.py` 和 `src/experiments/run_continual_learning.py`。
- 06-16 num_centers 实验的问题不是项目整体未对齐，而是 `main_incremental.py` 新集成路径没有沿用完整矩阵评估。
- `main_incremental.py` 现在每个 incremental step 后评估全部任务，而不是只评估已学任务。
- `main_incremental.py` 现在使用 `ContinualLearningMetrics` 计算并保存 LADA 口径的 Transfer/Average/Last。
- `src/utils/main_utils.py:get_full_stats` 已恢复为完整 LADA matrix 的上三角 Transfer 和全列 Average 口径，避免后续误用。

## 2. 重要发现

- 旧的 `ContinualLearningMetrics` 仍是正确实现：`Transfer_k = mean_{j<k} a_k^(j)`，`Average_k = mean_{j=1..K} a_k^(j)`，`Last_k = a_k^(K)`。
- 旧的 `run_continual_learning.py` 已经每步评估所有任务，并将 accuracy matrix 存为 `[0, 1]` fraction。
- 06-16 的 `main_incremental.py` 此前只在 `range(i + 1)` 上评估已学任务，并把三角矩阵 padding 成方阵，因此旧的 06-16 Transfer/Average 不能与 LADA 比较。
- 修复后 `main_incremental.py` 构建全局 zero-shot classifier，并使用 dataset-sequence 全局 label offset 评估所有任务。
- JSON per-classifier 输出现在保存完整 KxK matrix，数值为 `[0, 1]` fraction，兼容 `scripts/summarize_incremental_metrics.py` 的 recompute 逻辑。

## 3. 待办事项 / 遗留问题

- [ ] 重新运行 06-16 num_centers/GMM/fit 配置；旧的 `Transfer=72.30`、`Average=81.57` 不再作为 LADA 对齐证据。
- [ ] 重新生成 `docs/ablation_num_centers.md`，明确旧表为 pre-fix / non-LADA-metric result。
- [ ] 如需 LADA 分类器增量对比，还需继续接通或审查 `--enable_lada` 在 `main_incremental.py` 中的完整逻辑；本次只修评估矩阵和指标口径。

## 4. 相关文件

- `main_incremental.py`: 改为全任务评估、全局 zero-shot classifier、`ContinualLearningMetrics` 指标和 fraction matrix 保存。
- `src/utils/main_utils.py`: 恢复 `get_full_stats` 的 LADA Transfer/Average 口径。
- `src/utils/continual_metrics.py`: 既有 LADA 对齐指标实现。
- `src/experiments/run_continual_learning.py`: 既有完整矩阵评估路径。
- `docs/ablation_num_centers.md`: 旧 06-16 表格需要后续重跑后更新。
