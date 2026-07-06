# CD 散度 / 温度 / iterations / scheduler 消融进展

记录时间：2026-07-02 23:58 CST（北京时间）

## 已完成

- SGD 学习率精细搜索已结束，结果文件已存在。
- 散度形式消融（cd_temperature=2.0，cd_divergence ∈ {kl_forward, kl_reverse, js, mse, cosine, l1}）已完成。
- LADA 指标计算方式已核对，与 `src/utils/continual_metrics.py` 实现一致：
  - **Transfer** = mean(Transfer_k for k=2..K)，其中 Transfer_k 是学习任务 1..k-1 时在任务 k 上的平均准确率（forward transfer）。
  - **Average** = mean(Average_k for k=1..K)，其中 Average_k 是在所有训练步骤中任务 k 的平均准确率。
  - **Last** = mean(Last_k for k=1..K)，即最终模型在每个任务上的平均准确率。
- 零样本分类器（`zs_results.json`）与集成分类器（`ens_results.json`）结果已分开统计；方法自身的 `lada_results.json` 由于只记录了下三角矩阵，无法直接计算 Transfer，因此温度/iterations/scheduler 选择以 `zs_results.json` 的 Transfer 为准（与 orchestrate_cd_ablation.py 一致）。
- `kl_forward` 被选为最优散度形式（与 `kl_reverse` 几乎一致，Transfer 略高）。
- 温度调优 `temp=2.0` 已完成，结果文件：
  - `experiments/optimizer_ablation/current_nsp_adamw_cd_divergence_kl_forward_temp2.0_seed42_lada_results.json`
  - `experiments/optimizer_ablation/current_nsp_adamw_cd_divergence_kl_forward_temp2.0_seed42_zs_results.json`
  - `experiments/optimizer_ablation/current_nsp_adamw_cd_divergence_kl_forward_temp2.0_seed42_ens_results.json`
- 已扩展 `main_incremental.py` 与 `src/trainers/lora_nsp_trainer.py` 以支持 scheduler 消融：`linear`、`constant`、`cosine_with_warmup`（warmup steps = 10% iterations）。
- 已上传编排脚本 `scripts/orchestrate_cd_iter_scheduler_ablation.py`，用于温度实验完成后自动执行 iterations 与 scheduler 消融并生成总结。

## 进行中

- 温度调优（cd_divergence=kl_forward）：
  - `temp=1.0`：运行中（PID 2768885，CUDA_VISIBLE_DEVICES=2，物理 GPU2，已运行约 34 分钟）。
  - `temp=4.0`：运行中（PID 2784356，CUDA_VISIBLE_DEVICES=3，物理 GPU3，已运行约 23 分钟）。
  - `temp=2.0`：已完成。

## 当前 GPU 占用

- GPU0：空闲（可用）。
- GPU1：被占用（**不动**）。
- GPU2：被 temp=1.0 占用。
- GPU3：被 temp=4.0 占用。
- GPU4：空闲（可用）。
- GPU5：被占用（**不动**）。

## 待执行

1. 等待 `temp=1.0` 和 `temp=4.0` 完成后，收集三个温度的结果，写入 `chat-history/2026-07-02-cd-divergence-ablation-summary.md`。
2. 基于选出的最优 `cd_temperature`，执行 iterations 消融（iterations ∈ {400, 800, 1600}）。
3. 基于选出的最优 iterations，执行 scheduler 消融（scheduler ∈ {cosine, linear, cosine_with_warmup, constant}，warmup steps = 10% iterations）。
4. 最终写入 `chat-history/2026-07-02-iterations-scheduler-ablation-summary.md`。

## 约束

- 不动 GPU1 / GPU5。
- 若 GPU 被占用、实验异常中断或服务器断开超过 30 分钟，停止并报告。
# CD 消融进展更新

记录时间：2026-07-03 00:03 CST

## 当前状态

- **temp=1.0**：正在 Task 5 (flowers) 训练，进度约 24%，预计剩余 15–30 分钟。
- **temp=4.0**：正在 Task 3 (dtd) 训练，进度约 24%，预计剩余 30–60 分钟。
- **temp=2.0**：已完成。
- **编排器 (PID 2827424)**：处于 Phase 1，每分钟轮询一次结果文件，等待 temp=1.0 与 temp=4.0 完成。

## 下一步（由编排器自动执行）

1. temp=1.0 与 temp=4.0 完成后，自动生成 `chat-history/2026-07-02-cd-divergence-ablation-summary.md`。
2. 动态挑选 3 个空闲 GPU 启动 iterations 消融 `{400, 800, 1600}`。
3. 完成后启动 scheduler 消融 `{cosine, linear, cosine_with_warmup, constant}`。
4. 生成 `chat-history/2026-07-02-iterations-scheduler-ablation-summary.md`。

## 约束遵守

- 未触碰 GPU1/GPU5。
- 编排器会在无法获得足够 GPU 时停止并报错。

# CD 消融进展更新

记录时间：2026-07-03 00:10 CST

## 当前状态

- **temp=1.0**：Task 5 (flowers) 训练中，已训练约 180+ / 800 iterations，预计 5–8 分钟内完成 Task 5，再 3–5 分钟完成 Task 6。
- **temp=4.0**：Task 4 (eurosat) 已完成后处理，即将进入 Task 5，预计 8–12 分钟内完成剩余两个任务。
- **temp=2.0**：已完成。
- **编排器**：仍在 Phase 1 轮询。

## 约束遵守

- 未触碰 GPU1/GPU5。

# CD 消融进展更新

记录时间：2026-07-03 00:12 CST

## 当前状态

- **temp=1.0**：已进入最终任务 Task 6 (oxford_pets) 训练，预计 3–5 分钟内完成全部实验并写入结果文件。
- **temp=4.0**：已进入 Task 5 (flowers) 训练，完成 Task 5 后还需 Task 6，预计 6–10 分钟内完成。
- **temp=2.0**：已完成。
- **编排器**：仍在 Phase 1 轮询，等待 temp=1.0 与 temp=4.0 的结果文件。

## 约束遵守

- 未触碰 GPU1/GPU5。

# CD 消融进展更新

记录时间：2026-07-03 00:18 CST

## 当前状态

- **temp=1.0**：已完成，结果文件已生成。
- **temp=4.0**：Task 5 训练中，预计 5–8 分钟内完成。
- **temp=2.0**：已完成。
- **编排器**：仍在 Phase 1 轮询，等待 temp=4.0 完成。

## 约束遵守

- 未触碰 GPU1/GPU5。

# CD 消融进展更新

记录时间：2026-07-03 00:31 CST

## 当前状态

- **温度调优**：全部完成， 已生成。
- **最优温度**：（零样本 Transfer 60.31，集成分类器 Transfer 60.37）。
- **iterations 消融**：已启动，三个配置分别在物理 GPU0/2/3 上运行：
  - ：GPU0
  - ：GPU2
  - ：GPU3
- **scheduler 消融**：待 iterations 消融完成后自动启动。

## 约束遵守

- 未触碰 GPU1/GPU5。

# CD 消融进展更新（修正）

记录时间：2026-07-03 00:32 CST

## 当前状态

- **温度调优**：全部完成，summary 文件 chat-history/2026-07-02-cd-divergence-ablation-summary.md 已生成。
- **最优温度**：cd_temperature=4.0（零样本 Transfer 60.31，集成分类器 Transfer 60.37）。
- **iterations 消融**：已启动，三个配置分别在物理 GPU0/2/3 上运行：
  - iter400：GPU0
  - iter800：GPU2
  - iter1600：GPU3
- **scheduler 消融**：待 iterations 消融完成后自动启动。

## 约束遵守

- 未触碰 GPU1/GPU5。
