# full-shot 重启与 launcher 修复

**日期**: 2026-08-02
**会话概况**: 发现 full-shot LoRA-NF 因 DataLoader worker 误匹配导致大量重复进程，修复 launcher 后重启；增大 covariance batch size 加速。

---

## 1. 关键决策 / 修复

- **修复 launcher 进程检测**：DataLoader workers 继承父进程 cmdline，导致 `pgrep -f experiment_name` 把 workers 也计为独立实验进程。修改 `scripts/pa_*_launcher.sh`，通过 PPid 过滤只统计顶层主进程。
- **增大 cov_loader batch size**：`main_incremental.py` 中 cov_loader 从 `args.batch_size=32` 改为 `max(args.batch_size, 128)`，full-shot covariance 收集从每个 task 约 1.5 小时降至约 8–10 分钟。
- **重启 full-shot LoRA-NF seed42/43**：kill 所有重复/旧进程后，由 supervisor 重新调度，目前在 Task 2 正常运行。

## 2. 重要发现

- 16-shot 主表已聚合完成：LoRA-NF 完整方法 61.91 / 73.19 / 83.95，略高于同协议 LADA 复现 61.90 / 73.17 / 83.90。
- full-shot 仍在进行中；每个 task 训练约 2 分钟，covariance 收集约 8–10 分钟，评估约 2–3 分钟。

## 3. 待办事项 / 遗留问题

- [ ] 等待 full-shot LoRA-NF seed42/43 完成
- [ ] supervisor 自动调度 full-shot LoRA-only seed42/43
- [ ] supervisor 自动调度 B0 零样本 + retrieval 评测
- [ ] 全部 JSON 齐全后运行 `scripts/aggregate_pa_results.py` 并更新 `docs/average_push_log.md`

## 4. 相关文件

- `scripts/pa_fullshot_launcher.sh`：修复 DataLoader worker 误匹配
- `scripts/pa_wave_launcher.sh`、`scripts/pa_siglip2_launcher.sh`、`scripts/pa_lada_launcher.sh`：同步修复
- `main_incremental.py`：cov_loader batch size 128
- `docs/average_push_log.md`：16-shot 结果已记录
