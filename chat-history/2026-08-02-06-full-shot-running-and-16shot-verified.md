# full-shot 运行中 + 16-shot 聚合验证

**日期**: 2026-08-02
**会话概况**: 检查服务器确认 LoRA-NF full-shot seed42/43 健康运行；使用 aggregate_pa_results.py 重新聚合所有 16-shot JSON，数字与之前一致。

---

## 1. 关键状态

- **16-shot 主表/消融/敏感性/SigLIP2/LADA 复现**：全部完成并聚合完毕。
  - LoRA-NF Ens（seed42/43 均值）：**61.91 / 73.19 / 83.95**
  - 同协议 LADA 复现：**61.90 / 73.17 / 83.90**
  - 结论：LoRA-NF 以极微弱优势同时超过同管线 LADA（+0.01/+0.02/+0.05）。
- **full-shot W4**：
  - LoRA-NF seed43：已运行至 Task 2/10（caltech101），训练迭代约 150/600。
  - LoRA-NF seed42：同步在 GPU 1 运行。
  - LoRA-only seed42/43：排队，等待 supervisor 自动调度。
- **B0 W5**：未开始，将在 full-shot 全部完成后由 supervisor 自动启动。

## 2. 健康检查

- supervisor pid `3305934` 正常自守护，每 10 分钟检查一次。
- `pa_fullshot_launcher.sh` 的 ppid 过滤有效，只识别 2 个顶层主进程，不会重复启动。
- GPU 0/1 显存占用约 4 GB，utilization 当前为 0（主要在 covariance 收集/DataLoader），属正常。

## 3. 已验证的聚合命令

```bash
/home/raoxuan/ENTER/envs/raoxuan/bin/python scripts/aggregate_pa_results.py \
  --dir experiments/paper_formal/PA_main --per-dataset
```

输出已确认：主表各配置双种子 Ens 数字与 `docs/average_push_log.md` 一致。

## 4. 待办事项

- [ ] 等待 LoRA-NF full-shot seed42/43 完成（标志：`PA__fs__loranf_i600__seed*_ens_results.json` 出现）。
- [ ] 等待 supervisor 自动调度 LoRA-only full-shot seed42/43。
- [ ] 等待 supervisor 自动调度 B0 零样本 + retrieval 评测。
- [ ] 全部 JSON 齐全后再次运行聚合脚本，更新 `docs/average_push_log.md` 与本 chat-history。

## 5. 相关文件

- `docs/average_push_log.md`: 当前主表与进度。
- `scripts/aggregate_pa_results.py`: 聚合脚本。
- `scripts/pa_supervisor.sh`, `scripts/pa_fullshot_launcher.sh`: 服务器自守护调度。
- `artifacts/launch_logs/supervisor.log`: supervisor 运行日志。
