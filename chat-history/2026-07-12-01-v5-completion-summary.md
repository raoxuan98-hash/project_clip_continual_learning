# V5 10-task 消融实验完成总结

**日期**: 2026-07-12
**来源**: 服务器状态检查 (`raoxuan@10.20.34.30`)
**对应计划**: chat-history/2026-07-08-v5-plan.md

---

## 1. 服务器状态

- **SSH 连接**: 正常（通过 `C:\Users\Administrator\.ssh\raoxuan` 密钥）。
- **project_clip_continual_learning 进程**: 无 `main_incremental.py` 在运行。
- **GPU 状态**:
  - GPU 0,1,3,4,5 空闲（显存 ~15 MiB，利用率 0%）。
  - GPU 2 被其他项目占用（约 1.8 GiB，利用率 45%）。
- **代码状态**:
  - 最近 commit: `39c73fe fix: add missing --alpha_sweep_batch_size CLI arg`。
  - 存在 untracked 文件: `scripts/launch_v5_remaining.sh`。

## 2. V5 计划执行结果

2026-07-08-v5-plan.md 中 Phase 10–15 共 **12 个实验**已全部完成，最后一批结果生成于 **2026-07-10 00:14**。

| Phase | 实验内容 | 数量 | 状态 |
|:---|:---|---:|:---|
| 10 | LoRA Rank: {8, 16} + rank8×eps0.20 交互 | 3 | ✅ 完成 |
| 11 | Iterations: 1600 | 1 | ✅ 完成 |
| 12 | Warmup Ratio: {0%, 5%, 20%} | 3 | ✅ 完成 |
| 13 | FD Weight: {2.0, 4.0, 0.0} | 3 | ✅ 完成 |
| 14 | Weight Decay: {1e-5, 1e-4} | 2 | ✅ 完成 |
| **合计** | | **12** | **全部完成** |

## 3. 关键结果

### 基线（V4 最佳，未改变）

```text
LoRA+NSP, hard, nsp_eps=0.20, text=always, lr=1e-4, bs=32
cosine_with_warmup, eta_min=0.0, cd=2.0, fd=1.0, aux=0.0
cd_temperature=4.0, num_centers=4, rgda_train_iter=200, seed=43, maxshift
```

**ZS Average = 70.33 | Ens Average = 72.18 | Ens Last = 83.93**

### 各 Phase 最佳 vs 基线

| Phase | 变体 | ZS Avg | Ens Avg | Ens Last | 结论 |
|:---|:---|---:|---:|---:|:---|
| 10 | rank8 | 70.23 | 71.96 | 83.76 | 保持 rank=4 |
| 10 | rank16 | 70.24 | 71.97 | 83.81 | 保持 rank=4 |
| 10 | rank8 + eps0.20 | 70.22 | 71.95 | 83.77 | 无超加性，保持 rank=4 |
| 11 | iter1600 | 70.20 | 71.82 | 83.69 | 保持 iter=800 |
| 12 | warmup 0% | 70.15 | 72.07 | 83.66 | 保持 warmup=10% |
| 12 | warmup 5% | 70.06 | 71.96 | 83.72 | 保持 warmup=10% |
| 12 | warmup 20% | 70.20 | 72.06 | 83.72 | 保持 warmup=10% |
| 13 | fd=2.0 | 70.13 | 72.06 | 83.78 | 保持 fd=1.0 |
| 13 | fd=4.0 | 70.13 | 72.03 | 83.70 | 保持 fd=1.0 |
| 13 | fd=0.0 | 69.83 | 71.63 | 83.35 | 蒸馏有效，保持 fd=1.0 |
| 14 | wd=1e-5 | 70.12 | 72.03 | 83.81 | 保持 wd=3e-5 |
| 14 | wd=1e-4 | 70.12 | 72.03 | 83.77 | 保持 wd=3e-5 |

## 4. 结论

- **V4 最佳配置得到 V5 验证**：Phase 10–14 的所有变体均未超过基线。
- 当前 10-task 设置下的最优参数已经收敛，继续边际调参收益有限。
- 日志与结果文件位于服务器：
  - 结果：`/home/raoxuan/projects/project_clip_continual_learning/experiments/10task_reablation/`
  - 日志：`/home/raoxuan/projects/project_clip_continual_learning/logs/10task_reablation/`

## 5. 下一步建议

1. **Phase 16 最终汇总**：整理 V4+V5 全部实验为最终表格。
2. **版本控制**：将 `scripts/launch_v5_remaining.sh` commit。
3. **Seed 稳定性**：如投稿需要，可跑 seed=42/44 验证基线方差。
4. **论文图表**：基于当前最佳配置生成最终检索/可视化结果。

---

*由服务器状态检查后整理。*
