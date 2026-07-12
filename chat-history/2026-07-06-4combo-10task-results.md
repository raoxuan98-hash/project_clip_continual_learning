# 4 Combo 10-Task 结果 & Alpha Sweep 总结

**日期**: 2026-07-06
**协议**: 10-task 增量训练 (aircraft → caltech101 → dtd → eurosat → flowers → food101 → mnist → oxford_pets → stanford_cars → sun397), 16-shot, seed=42
**评估**: 同步 inline eval + alpha sensitivity sweep (21 alpha, 0.00~1.00)

---

## 1. 4 Combo 配置

| Combo | 关键差异 |
|---|---|
| C1 Balanced | current_nsp + AdamW 1e-4 + cwu + iter800 + bs128 |
| C2 Last Max | current_nsp + AdamW 3e-4 + linear + iter1600 + bs128 |
| C3 Transfer Max | hist_null_init_runtime + AdamW 1e-4 + cosine + iter800 + bs128 |
| C4 Init-Only | hist_null_init_only + AdamW 1e-4 + cwu + iter800 + bs128 |

共同配置: lora_nsp, rank=4, q/k/v/out/ffn, cd=2.0, fd=1.0, aux=0.0, kl_forward, temp=4.0

---

## 2. Zero-shot & Ensemble 精度（LADA 指标）

| Combo | ZS Transfer | ZS Average | ZS Last | Ens Transfer | Ens Average | Ens Last |
|---|---|---|---|---|---|---|
| **C1 Balanced** | 58.4 | **68.4** | **76.5** | 58.5 | **69.6** | **80.0** |
| C2 Last Max | 56.6 | 66.9 | 74.3 | 56.6 | 67.8 | 77.5 |
| C3 Transfer Max | **58.5** | 67.5 | 74.9 | **58.6** | 68.9 | 79.0 |
| C4 Init-Only | 58.4 | 66.5 | 71.6 | 58.5 | 67.6 | 74.7 |

### 逐任务明细（C1 Balanced）

#### Zero-shot
| Task | aircraft | caltech101 | dtd | eurosat | flowers | food101 | mnist | oxford_pets | stanford_cars | sun397 |
|---|---|---|---|---|---|---|---|---|---|---|
| Transfer | N/A | 73.3 | 36.2 | 32.2 | 61.2 | 81.6 | 43.2 | 84.8 | 52.0 | 60.6 |
| Average | 49.2 | 82.7 | 55.5 | 68.3 | 76.4 | 83.1 | 63.8 | 86.7 | 56.4 | 61.6 |
| Last | 47.3 | 82.4 | 56.1 | 81.8 | 84.2 | 84.3 | 94.4 | 90.9 | 73.6 | 69.8 |

#### Ensemble (alpha=0.05)
| Task | aircraft | caltech101 | dtd | eurosat | flowers | food101 | mnist | oxford_pets | stanford_cars | sun397 |
|---|---|---|---|---|---|---|---|---|---|---|
| Transfer | N/A | 73.4 | 36.3 | 32.3 | 61.3 | 81.7 | 43.7 | 84.9 | 52.1 | 60.9 |
| Average | 50.5 | 84.1 | 57.4 | 69.3 | 80.7 | 83.6 | 64.3 | 87.0 | 57.1 | 62.1 |
| Last | 48.9 | 85.2 | 60.5 | 90.5 | 91.6 | 85.5 | 95.5 | 92.1 | 76.8 | 72.9 |

---

## 3. Alpha Sensitivity Sweep（ZS, 按 Best Average）

| Combo | Best α | ZS Transfer | ZS Average | ZS Last |
|---|---|---|---|---|
| C1 Balanced | 0.30 | 0.74 | 0.70 | 0.82 |
| C2 Last Max | 0.20 | 0.70 | 0.68 | 0.79 |
| C3 Transfer Max | 0.30 | 0.73 | 0.70 | 0.82 |
| C4 Init-Only | 0.30 | 0.70 | 0.68 | 0.78 |

> Alpha sweep 值范围为 [0, 1]，×100 得百分比。与 LADA 指标口径不同，不可直接对比。

---

## 4. 关键结论

1. **C1 (Balanced) 全面最优** — ZS/Ens 的 Average 和 Last 均为最高，Transfer 仅比 C3 低 0.1pp。当前最佳配置。

2. **C2 (Last Max) 出乎意料地弱** — lr=3e-4 + iter1600 未推高 Last，反而比 C1 低 2.5pp（Ens Last: 77.5 vs 80.0）。高 lr + 长训练在 10-task 下可能过拟合。

3. **C3 (Transfer Max) 的 Transfer 优势微弱** — 仅比 C1 高 0.1pp，Average/Last 均不如。hist_null_init_runtime 的额外投影开销无实质收益。

4. **C4 (Init-Only) 垫底** — Ens Last 仅 74.7（比 C1 低 5.3pp），验证 runtime NSP 不可省略。但 null init alone 仍有一定效果（vs early null basis ablation 的矛盾——需结合 10-task 重新评估）。

5. **10-task vs 6-task** — C1 Ens Average 从 70.65 (6-task) 降至 69.6 (10-task)，降幅约 1pp，合理。

---

## 5. 当前状态

- **NSP 消融 Wave 1** 正在运行：nsp_eps ∈ {0.02, 0.08, 0.12, 0.20}，10-task，seed=43
- 后续 Wave 2 (nsp_weight)、Wave 3 (soft projection) 待启动
- Cron job `90038ef8` 自动跟进

---

## 6. 相关文件

- 日志: `logs/combo{1-4}_*.log`
- Alpha sweep 输出: 嵌入各 combo 日志末尾
- 消融脚本: `scripts/launch_nsp_ablation.sh`
