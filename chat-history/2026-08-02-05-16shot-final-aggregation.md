# 16-shot 主表 / 消融 / LADA 复现最终聚合 + full-shot 推进

**日期**: 2026-08-02
**会话概况**: 在新协议（preserve_aspect）下完成 16-shot 全部 run 的聚合，确认 LoRA-NF 以极微弱优势超过同管线复现的 LADA；继续推进 W4 full-shot 与 W5 B0 评测。

---

## 1. 关键讨论 / 决策

- **用户确认**：同 backbone 下仅换用 LADA 分类器的 full-shot 复现不需要跑，直接跳过；full-shot 队列仅保留 LoRA-NF seed42/43 与 LoRA-only seed42/43。
- **当前 16-shot 达标判定**：LoRA-NF（完整方法） Ens 为 **61.91 / 73.19 / 83.95**，同协议 LADA 复现为 **61.90 / 73.17 / 83.90**，三项均极微弱领先（+0.01/+0.02/+0.05）。
- **Average 不是 Transfer 与 Last 的算术平均**：Average 是增量学习曲线在所有任务/时间步上的平均；因此可以出现 T≈LADA、L 明显更高、A 略高的组合。

## 2. 重要发现

- 主表（seed42/43 均值，Ens）：
  - LoRA-NF：61.91 / 73.19 / 83.95
  - LADA：61.90 / 73.17 / 83.90
  - LoRA-NF 无 CD：62.42 / 72.50 / 83.63
  - GradProj：61.57 / 72.80 / 83.35
  - LoRA-Null：61.62 / 72.12 / 82.89
  - LoRA+CD（无 NSP）：61.38 / 72.45 / 82.74
  - LoRA-only：60.92 / 67.10 / 75.47
- 检索能力：LoRA-NF Flickr30k i2t R@1 = 81.30，优于 LoRA-only 的 78.90；但 t2i R@1 LoRA-only 反而略高（63.02 vs 61.89）。
- SigLIP2 LoRA-NF：Transfer 59.50 / Average 68.66 / Last 86.81；检索显著优于 CLIP（MSCOCO i2t 63.89 vs 52.99）。

## 3. 待办事项 / 遗留问题

- [ ] W4 full-shot LoRA-NF seed42/43 正在运行（Task 3 左右，预计数小时完成）。
- [ ] W4 full-shot LoRA-only seed42/43 排队，由 supervisor 自动调度。
- [ ] W5 B0 零样本 + retrieval 评测。
- [ ] 全部完成后重新运行聚合脚本，更新 `docs/average_push_log.md` 与本日志。

## 4. 相关文件

- `docs/average_push_log.md`: 已更新 16-shot 主表、LADA 复现结果、vs LADA 判定。
- `scripts/aggregate_pa_results.py`: 用于最终聚合。
- `scripts/pa_supervisor.sh`, `scripts/pa_fullshot_launcher.sh`: 服务器端守护调度。
- `experiments/paper_formal/PA_main/`: 所有 run 的 JSON 输出目录。
