# 近一周消融实验综合回顾（2026-06-30 ~ 2026-07-05）

**日期**: 2026-07-05  
**数据来源**: 服务器 `10.20.34.30:/home/raoxuan/projects/project_clip_continual_learning/experiments/` 与本地 `chat-history/` 已同步文件。  
**基准任务序列**: `aircraft -> caltech101 -> dtd -> eurosat -> flowers -> oxford_pets`（X-TAIL 16-shot，seed=42）。  

---

## 1. 指标说明

所有指标均按 LADA 论文 Appendix A 的定义从 `*_zs_results.json` / `*_ens_results.json` 的 `metrics` 字段读取，已乘以 100 并以百分比形式保留三位小数：

- **Transfer**: 对已训练任务，计算学习后续任务时在前序任务上的平均准确率；对首个任务为 0。
- **Average**: 每个任务在所有训练步骤（或检查点）上的平均准确率的均值。
- **Last**: 最终模型在每个任务上的平均准确率。

> **重要**：Average **不是** `(Transfer + Last) / 2`。例如 `combo_balanced` 的 Zero-shot 结果：Transfer=59.962、Last=79.181，二者平均为 69.572，而实际 Average=69.609，二者相差 0.037。因此 Average 是独立按训练步骤平均后再求均值得来。

---

## 2. 公共基准配置

除非表格另有说明，下列实验默认使用：

```text
lora_type      = lora_nsp
use_dora       = false
lora_rank      = 4
target modules = q/k/v/out/ffn
optimizer      = AdamW
lr             = 1e-4
weight_decay   = 3e-5
batch_size     = 64
iterations     = 800
scheduler      = cosine
cd_weight      = 2.0
fd_weight      = 1.0
aux_weight     = 0.0
cd_divergence  = kl_forward
cd_temperature = 2.0（温度消融前）/ 4.0（温度消融后）
```

---

## 3. 优化器消融（LoRA）

| variant | 优化器 | ZS Transfer | ZS Average | ZS Last | Ens Transfer | Ens Average | Ens Last |
|---|---|---:|---:|---:|---:|---:|---:|
| current_nsp | Adagrad | 60.067 | 67.618 | 76.266 | 60.103 | 69.076 | 79.644 |
| current_nsp | Adam | 60.871 | 68.586 | 77.538 | 60.914 | 69.852 | 80.298 |
| current_nsp | AdamW | 60.886 | 68.590 | 77.545 | 60.924 | 69.846 | 80.246 |
| current_nsp | RMSprop | 60.870 | 68.637 | 77.578 | 60.887 | 69.888 | 80.363 |
| current_nsp | SGD (lr=1e-3) | 60.884 | 65.022 | 71.250 | 60.922 | 66.722 | 74.427 |
| hist_null_init_runtime | Adagrad | 60.446 | 67.936 | 76.329 | 60.492 | 69.261 | 79.518 |
| hist_null_init_runtime | Adam | 61.242 | 68.324 | 76.627 | 61.266 | 69.736 | 79.707 |
| hist_null_init_runtime | AdamW | 61.274 | 68.354 | 76.711 | 61.299 | 69.752 | 79.699 |
| hist_null_init_runtime | RMSprop | 60.859 | 68.516 | 77.655 | 60.905 | 69.739 | 80.235 |
| hist_null_init_runtime | SGD (lr=1e-3) | 61.275 | 66.129 | 72.261 | 61.355 | 67.842 | 75.579 |

**结论**：AdamW / Adam / RMSprop 是第一梯队；`hist_null_init_runtime` 的 Transfer 略高，`current_nsp` 的 Last/Average 略高。最终选定 **current_nsp + AdamW**。

---

## 4. 学习率消融

### 4.1 AdamW 学习率

| lr | ZS Transfer | ZS Average | ZS Last | Ens Transfer | Ens Average | Ens Last |
|---|---:|---:|---:|---:|---:|---:|
| 5e-5 | 61.754 | 67.235 | 74.688 | 61.812 | 68.771 | 77.861 |
| 1e-4 | 60.886 | 68.590 | 77.545 | 60.924 | 69.846 | 80.246 |
| 3e-4 | 58.250 | 68.746 | 79.563 | 58.294 | 69.795 | 81.874 |
| 6e-4 | 57.975 | 68.268 | 78.536 | 58.030 | 69.644 | 81.468 |

**结论**：**1e-4 是 Transfer/Average/Last 的最佳平衡点**；3e-4 的 Last 最高但 Transfer 明显下降。

### 4.2 SGD 精细学习率（current_nsp + SGD）

| lr | ZS Transfer | ZS Average | ZS Last | Ens Transfer | Ens Average | Ens Last |
|---|---:|---:|---:|---:|---:|---:|
| 1e-3 | 60.884 | 65.022 | 71.250 | 60.922 | 66.722 | 74.427 |
| 2e-3 | 60.744 | 67.174 | 74.492 | 60.820 | 68.883 | 78.013 |
| 3e-3 | 59.386 | 68.225 | 77.352 | 59.467 | 69.757 | 80.988 |
| 4e-3 | 58.159 | 68.536 | 79.018 | 58.257 | 69.925 | 81.995 |
| 5e-3 | 57.440 | 68.530 | 79.000 | 57.590 | 69.990 | 82.243 |
| 5e-2 | 61.354 | 26.387 | 0.174 | 61.354 | 38.187 | 70.950 |

**结论**：SGD 学习率越大 Last 越高、Transfer 越低。5e-3 在 Ens Last 上达到 82.243，但 Transfer 掉至 57.590；最终仍选用 AdamW 1e-4 作为默认。

---

## 5. Batch Size 消融

| batch_size | ZS Transfer | ZS Average | ZS Last | Ens Transfer | Ens Average | Ens Last |
|---|---:|---:|---:|---:|---:|---:|
| 32 | 61.662 | 67.782 | 75.797 | 61.682 | 69.315 | 79.022 |
| 64 | 60.886 | 68.590 | 77.545 | 60.924 | 69.846 | 80.246 |
| 128 | 60.131 | 69.061 | 78.689 | 60.178 | 70.139 | 81.007 |

**结论**：更大的 batch size 系统提升 Average/Last，但 Transfer 轻微下降。后续最佳组合实验统一使用 **batch_size=128**。

---

## 6. 蒸馏与辅助头权重消融

| 配置 | ZS Transfer | ZS Average | ZS Last | Ens Transfer | Ens Average | Ens Last |
|---|---:|---:|---:|---:|---:|---:|
| cd_weight=0.0 | 60.513 | 68.016 | 77.090 | 60.543 | 69.409 | 80.274 |
| cd_weight=1.0 | 60.886 | 68.590 | 77.545 | 60.924 | 69.846 | 80.246 |
| cd_weight=2.0 | 60.960 | 68.715 | 77.671 | 61.006 | 69.950 | 80.273 |
| fd_weight=0.0 | 60.915 | 68.613 | 77.569 | 60.970 | 69.865 | 80.322 |
| fd_weight=1.0 | 60.886 | 68.590 | 77.545 | 60.924 | 69.846 | 80.246 |
| fd_weight=2.0 | 60.864 | 68.587 | 77.533 | 60.899 | 69.847 | 80.180 |
| aux_weight=0.0 | 60.433 | 68.976 | 78.146 | 60.483 | 70.031 | 80.442 |
| aux_weight=1.0 | 60.886 | 68.590 | 77.545 | 60.924 | 69.846 | 80.246 |
| aux_weight=2.0 | 61.087 | 68.120 | 76.500 | 61.131 | 69.456 | 79.778 |

**结论**：
- `cd_weight=2.0` 在 Transfer 上最优；
- `fd_weight` 在 {0,1,2} 内几乎不敏感；
- `aux_weight=0.0` 的 Average 最好，`aux=1.0` 的 Last 略高，差异很小。默认保持 **cd=2.0, fd=1.0, aux=0.0**。

---

## 7. CD 散度形式消融（temp=2.0）

| cd_divergence | ZS Transfer | ZS Average | ZS Last | Ens Transfer | Ens Average | Ens Last |
|---|---:|---:|---:|---:|---:|---:|
| kl_forward | 60.267 | 68.983 | 78.344 | 60.303 | 70.086 | 80.718 |
| kl_reverse | 60.261 | 68.987 | 78.441 | 60.301 | 70.079 | 80.604 |
| js | 60.204 | 68.823 | 78.012 | 60.252 | 69.975 | 80.528 |
| mse | 59.971 | 68.467 | 77.806 | 59.999 | 69.525 | 80.113 |
| cosine | 60.043 | 68.597 | 77.856 | 60.071 | 69.752 | 80.418 |
| l1 | 59.987 | 68.505 | 77.699 | 60.024 | 69.714 | 80.322 |

**结论**：**kl_forward** 在 Transfer 上略优于其他散度，被选为默认。

---

## 8. CD 温度消融（kl_forward）

| cd_temperature | ZS Transfer | ZS Average | ZS Last | Ens Transfer | Ens Average | Ens Last |
|---|---:|---:|---:|---:|---:|---:|
| 1.0 | 60.304 | 68.827 | 77.934 | 60.344 | 70.001 | 80.561 |
| 2.0 | 60.267 | 68.983 | 78.344 | 60.303 | 70.086 | 80.718 |
| 4.0 | 60.309 | 69.046 | 78.404 | 60.367 | 70.137 | 80.704 |

**结论**：温度 4.0 在 ZS/Ens Transfer 上均最优，同时 Average/Last 也保持前列。默认改为 **cd_temperature=4.0**。

---

## 9. Iterations × Scheduler 消融（kl_forward, temp=4.0）

| iter | scheduler | ZS Transfer | ZS Average | ZS Last | Ens Transfer | Ens Average | Ens Last |
|---:|---|---:|---:|---:|---:|---:|---:|
| 400 | cosine | 61.497 | 67.068 | 74.787 | 61.551 | 68.600 | 77.832 |
| 400 | cosine_with_warmup | 61.501 | 67.331 | 75.213 | 61.548 | 68.919 | 78.119 |
| 400 | linear | 61.538 | 67.097 | 74.713 | 61.584 | 68.706 | 77.916 |
| 400 | constant | 60.701 | 68.979 | 78.176 | 60.751 | 70.113 | 80.566 |
| 800 | cosine | 60.680 | 69.155 | 78.286 | 60.731 | 70.248 | 80.675 |
| 800 | cosine_with_warmup | 60.505 | 69.217 | 78.316 | 60.554 | 70.314 | 80.798 |
| 800 | linear | 60.698 | 69.142 | 78.218 | 60.747 | 70.261 | 80.624 |
| 800 | constant | 59.697 | 67.099 | 75.553 | 59.928 | 68.945 | 78.615 |
| 1600 | cosine | 58.981 | 69.560 | 79.499 | 59.095 | 70.400 | 81.166 |
| 1600 | cosine_with_warmup | 58.896 | 69.416 | 79.362 | 58.971 | 70.282 | 80.939 |
| 1600 | linear | 59.037 | 69.606 | 79.533 | 59.157 | 70.502 | 81.263 |
| 1600 | constant | 58.929 | 67.570 | 76.168 | 59.162 | 69.226 | 78.666 |

**结论**：
- iterations 从 400 增加到 800 时 Average/Last 显著提升；800→1600 时 Last 继续小幅提升，但 Transfer 下降。
- `cosine`、`linear`、`cosine_with_warmup` 在 iter=800/1600 上表现接近，均优于 `constant`。
- 平衡配置选 **iter=800 + cosine_with_warmup**；若只看 Last，可选 **iter=1600 + linear**。

---

## 10. 主干微调层消融（rank=4）

| target modules | ZS Transfer | ZS Average | ZS Last | Ens Transfer | Ens Average | Ens Last |
|---|---:|---:|---:|---:|---:|---:|
| q/k/v/out/ffn (All) | 59.962 | 69.609 | 79.181 | 60.041 | 70.653 | 81.405 |
| q/k only | 61.173 | 63.972 | 69.848 | 61.368 | 66.037 | 73.904 |
| v/out/ffn (FFN only) | 59.746 | 68.701 | 77.591 | 59.907 | 70.120 | 80.448 |
| q/k/v/out (Attention only) | 61.291 | 67.788 | 75.348 | 61.390 | 69.206 | 78.445 |

**结论**：同时微调 QKV+FFN 仍是最佳；只微调 FFN 的 Ens Average 可达 70.120，仅比 All 低 0.533；只微调 QK 或 Attention 效果较差。

---

## 11. LoRA Rank 消融（target modules = q/k/v/out/ffn）

| rank | ZS Transfer | ZS Average | ZS Last | Ens Transfer | Ens Average | Ens Last |
|---|---:|---:|---:|---:|---:|---:|
| 2 | 60.383 | 68.403 | 76.949 | 60.486 | 69.790 | 79.779 |
| 4 | 59.866 | 68.946 | 78.150 | 59.974 | 70.116 | 80.571 |
| 8 | 59.323 | 69.298 | 79.191 | 59.459 | 70.391 | 81.556 |
| 16 | 59.475 | 69.628 | 79.522 | 59.628 | 70.595 | 81.434 |

**结论**：rank 越大 Last/Average 越高，但 Transfer 越低。rank=8/16 的 Ens Average 接近且均高于 rank=4。在效率与效果之间，**rank=4 仍是默认**；若追求 Last 可提升至 rank=8/16。

---

## 12. QKV 投影方案消融

| 方案 | 实现说明 | ZS Transfer | ZS Average | ZS Last | Ens Transfer | Ens Average | Ens Last |
|---|---|---:|---:|---:|---:|---:|---:|
| 方案 A：共享 P（当前默认）| Q/K/V 共享同一个投影矩阵 P | 59.962 | 69.609 | 79.181 | 60.041 | 70.653 | 81.405 |
| 旧实现：独立 P | Q/K/V 各自收集协方差并独立 SVD | 60.057 | 69.360 | 78.542 | 60.180 | 70.547 | 81.121 |
| 方案 B：fused QKV | 把 Q/K/V 合成一个大矩阵后再投影 | 56.562 | 65.331 | 72.781 | 56.658 | 66.064 | 74.247 |

**结论**：共享 P（方案 A）在三者中 Ens Average 最高，且去除了独立 P 的冗余协方差/SVD 计算；fused QKV 明显落后，不采用。

---

## 13. 最佳配置组合验证

| 组合 | 关键配置 | ZS Transfer | ZS Average | ZS Last | Ens Transfer | Ens Average | Ens Last |
|---|---|---:|---:|---:|---:|---:|---:|
| Combo 1: Best Balanced | current_nsp + AdamW 1e-4 + iter800 + cwu + bs128 | 59.962 | 69.609 | 79.181 | 60.041 | 70.653 | 81.405 |
| Combo 2: Last Max | current_nsp + AdamW 3e-4 + iter1600 + linear + bs128 | 57.254 | 69.040 | 80.025 | 57.320 | 69.850 | 81.685 |
| Combo 3: Transfer Max | hist_null_init_runtime + AdamW 1e-4 + iter800 + cosine + bs128 | 60.399 | 69.249 | 78.154 | 60.482 | 70.435 | 80.709 |
| Combo 4: Init-Only | hist_null_init_only + AdamW 1e-4 + iter800 + cwu + bs128 | 59.927 | 67.798 | 75.219 | 59.981 | 69.059 | 78.016 |

**结论**：按 Ens Average 排序：**Combo 1 > Combo 3 > Combo 2 > Combo 4**。

---

## 14. 按 Ens Average 排序的 Top 单点/组合

| 配置 | Ens Average | Ens Transfer | Ens Last | 备注 |
|---|---:|---:|---:|---|
| Combo 1: current_nsp + AdamW 1e-4 + bs128 + iter800 + cwu | **70.653** | 60.041 | 81.405 | 当前最佳组合 |
| rank=16 + All layers | 70.595 | 59.628 | 81.434 | rank 消融 |
| rank=8 + All layers | 70.391 | 59.459 | 81.556 | rank 消融 |
| Combo 3: hist_runtime + cosine | 70.435 | 60.482 | 80.709 | Transfer 优先 |
| Layer All + rank=4（baseline） | 70.653 | 60.041 | 81.405 | 与 Combo 1 相同 |
| FFN only + rank=4 | 70.120 | 59.907 | 80.448 | 层消融 |
| temp=4.0 + iter800 + cwu | 70.314 | 60.554 | 80.798 | 温度/iter 消融 |
| bs128 + AdamW 1e-4 | 70.139 | 60.178 | 81.007 | batch size 消融 |

---

## 15. 当前推荐默认配置

```text
variant               = current_nsp
lora_type             = lora_nsp
use_dora              = false
lora_rank             = 4
target modules        = q/k/v/out/ffn
projection sharing    = 方案 A（Q/K/V 共享 P）
optimizer             = AdamW
lr                    = 1e-4
weight_decay          = 3e-5
batch_size            = 128
iterations            = 800
scheduler             = cosine_with_warmup
fd_weight             = 1.0
cd_weight             = 2.0
aux_weight            = 0.0
cd_divergence         = kl_forward
cd_temperature        = 4.0
```

在该配置下，seed=42 的结果为：

- Zero-shot: Transfer=59.962 / Average=69.609 / Last=79.181
- Ensemble: Transfer=60.041 / Average=70.653 / Last=81.405

---

## 16. 遗留问题与下一步

1. **多种子验证**：当前全部为 seed=42，关键配置需要 2–3 个种子确认稳定性。
2. **cd_weight 饱和**：已测 {0,1,2}，2.0 最优；是否需要继续测试 3.0/4.0？
3. **DoRA 定量对比**：现有结论为“LoRA 系统优于 DoRA”，但缺少保留三位小数的直接对比表格。
4. **rank 与层选择交互**：rank=8/16 与 FFN-only 的组合尚未测试。
5. **更大 batch / 更长训练**：bs=128 + iter=1600 的组合尚未充分探索。

---

*记录者: Kimi Code CLI*  
*同步路径: `/home/raoxuan/projects/project_clip_continual_learning/chat-history/2026-07-05-weekly-ablation-review.md`*
