# SGD 学习率精细搜索总结（2026-07-02）

## 实验配置

- 方法：`current_nsp + LoRA (rank=4) + SGD`
- 学习率候选：`{2e-3, 3e-3, 4e-3}`
- 其他超参：`batch_size=64, iterations=800, weight_decay=3e-5, cd_weight=2.0, aux_weight=0.0, fd_weight=1.0`
- 任务序列：`aircraft, caltech101, dtd, eurosat, flowers, oxford_pets`
- 种子：`42`

## 结果

| 学习率 | ZS-Transfer (%) | LADA-Average (%) | LADA-Last (%) | Ensemble-Last (%) |
|--------|----------------|------------------|---------------|-------------------|
| 2e-3   | 63.42          | 41.62            | 77.13         | 78.01             |
| 3e-3   | 61.94          | 42.32            | 77.69         | 80.99             |
| 4e-3   | 60.55          | 43.28            | 78.77         | 81.99             |

## 关键观察

1. **Last / Average 随学习率增大而提升**：`4e-3` 在 LADA-Last 和 Ensemble-Last 上最高，说明更大的 SGD 学习率在该范围内增强了最终任务的拟合能力。
2. **ZS-Transfer 随学习率增大而下降**：`2e-3` 保留了最好的零样本迁移能力，而 `4e-3` 相比 `2e-3` 下降了约 2.9 个百分点。
3. **3e-3 是折中点**：在 Last 与 Transfer 之间取得相对平衡。

## 备注

- 结果文件中的 `transfer` 字段为 `0.0`，这是因为当前 LADA 评估流程对未来任务返回 `0.0`（LADA 分类器未包含未见类别）。
- 上表中的 **ZS-Transfer** 是从零样本分类器结果矩阵的上三角（每个任务被训练之前的平均准确率）重新计算的，符合 LADA 论文对 Transfer 的定义。

## 下一步

基于默认 AdamW (`lr=1e-4`) 配置，继续执行 `cd_divergence` 消融实验。
