# Iterations × Scheduler 消融最终汇总

**日期**: 2026-07-04  
**配置**: current_nsp + AdamW + lr=1e-4 + cd_divergence=kl_forward + cd_temperature=4.0 + cd_weight=2.0 + fd_weight=1.0 + aux_weight=0.0 + batch_size=64 + seed=42  
**指标**: Transfer / Average / Last（按 LADA 论文 Appendix A 公式计算，百分比，保留两位小数）

## Zero-shot 分类器

| iter / scheduler | cosine | cosine_with_warmup | linear | constant |
|---|---:|---:|---:|---:|
| iter400 | 60.97 / 64.25 / 71.44 | 61.50 / 67.33 / 75.21 | 61.54 / 67.10 / 74.71 | 60.70 / 68.98 / 78.18 |
| iter800 | 60.68 / 69.16 / 78.29 | 60.51 / 69.22 / 78.32 | 60.70 / 69.14 / 78.22 | 59.70 / 67.10 / 75.55 |
| iter1600 | 58.98 / 69.56 / 79.50 | 58.90 / 69.42 / 79.36 | 59.04 / 69.61 / 79.53 | 58.93 / 67.57 / 76.17 |

## Ensemble 分类器

| iter / scheduler | cosine | cosine_with_warmup | linear | constant |
|---|---:|---:|---:|---:|
| iter400 | 61.11 / 66.61 / 75.45 | 61.55 / 68.92 / 78.12 | 61.58 / 68.71 / 77.92 | 60.75 / 70.11 / 80.57 |
| iter800 | 60.73 / 70.25 / 80.67 | 60.55 / 70.31 / 80.80 | 60.75 / 70.26 / 80.62 | 59.93 / 68.94 / 78.61 |
| iter1600 | 59.10 / 70.40 / 81.17 | 58.97 / 70.28 / 80.94 | 59.16 / 70.50 / 81.26 | 59.16 / 69.23 / 78.67 |

## 主要观察

- **iterations**: 从 400 增加到 800，Average 和 Last 显著提升；从 800 到 1600，Last 继续小幅提升，但 Transfer 轻微下降。
- **scheduler**: 在 iter800/1600 上，`cosine`、`linear`、`cosine_with_warmup` 三者表现非常接近，均优于 `constant`。
- **constant scheduler**: 仅在 iter400 凭借较高的 Last 表现突出，但不随迭代次数扩展，在 iter800/1600 明显落后。
- 综合考虑 Transfer、Average、Last，`iter800 + cosine_with_warmup` / `iter800 + linear` / `iter800 + cosine` 都是强候选；若更看重 Last，可倾向 `iter1600 + linear`。
