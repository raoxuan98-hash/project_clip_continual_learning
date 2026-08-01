# PA 新协议 16-shot 主表双种子聚合完成

**日期**: 2026-08-01
**会话概况**: 继续推进 preserve_aspect 全量重跑计划；W1 27 个 run 全部完成，按 seed42/43 双种子重新聚合主表，结果写入 docs/average_push_log.md。

---

## 1. 关键讨论 / 决策

- **决策 1**: 在 SigLIP2/LADA/full-shot/B0 尚未完成前，先利用已完成的 W1 双种子结果做聚合，更新主表。
- **决策 2**: 继续由服务器 supervisor 自治推进后续波次；当前不干预运行中的 SigLIP2 retrieval。

## 2. 重要发现

- LoRA-NF（完整方法）双种子 Ens: **61.91±0.16 / 73.19±0.16 / 83.95±0.06**。
- LoRA-only: 61.10±0.03 / 68.54±0.06 / 79.80±0.80，与 LoRA-NF 差距明显。
- LoRA-NF 无 CD: Transfer 更高（62.42±0.03），但 Average 低于完整方法（72.50±0.25），说明 CD 提升 Average/Last。
- 检索 R@1（LoRA-NF）：MSCOCO i2t=52.99, t2i=33.70；Flickr30k i2t=81.30, t2i=61.89。
- **LADA 复现结果尚未生成，vs LADA 判定待 W3 完成后给出。**

## 3. 待办事项 / 遗留问题

- [ ] 等待 SigLIP2 LoRA-NF seed42/43 完成 retrieval 并生成 JSON。
- [ ] 等待 LADA 16-shot seed42/43 复现完成。
- [ ] 等待 full-shot 6 run 完成。
- [ ] 等待 B0 零样本 + retrieval 完成。
- [ ] 全部完成后重新聚合，给出 vs 同协议 LADA 复现的判定结论。

## 4. 相关文件

- `docs/average_push_log.md`: 当前聚合主表与观察。
- `experiments/paper_formal/PA_main/`: 所有 JSON 输出。
- `scripts/pa_supervisor.sh`: 服务器端自治调度。
- `scripts/aggregate_pa_results.py`: 聚合脚本。
