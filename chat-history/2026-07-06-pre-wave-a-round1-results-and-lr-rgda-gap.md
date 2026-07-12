# Pre-Wave A Round 1 初步结果与 LR-RGDA 配置差异发现

**日期**: 2026-07-06
**会话**: Pre-Wave A 执行中

---

## Round 1 结果（seed=42, maxshift）

| 实验 | ZS Transfer | ZS Average | ZS Last | Ens Transfer | Ens Average | Ens Last |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| lora_freeze | 58.78 | 67.45 | 74.71 | 58.87 | 68.77 | 79.17 |
| lora_lowlr | 58.89 | 69.13 | 77.81 | 59.01 | 70.07 | 80.40 |
| **lora_always** | **59.79** | **70.01** | **80.02** | **59.90** | **70.58** | **81.51** |
| dora_freeze | 59.67 | 67.01 | 72.90 | 59.76 | 68.48 | 77.35 |
| dora_lowlr | — | — | — | — | — | — |
| dora_always | — | — | — | — | — | — |

### 初步观察

1. **LoRA vs DoRA**：在 freeze_after 下，LoRA ZS Average 67.45 > DoRA 67.01。DoRA 在 10-task 上未表现出优势。

2. **text_schedule 排序（LoRA 侧）**：`always > low_lr_after > freeze_after`。always 的 ZS Average 比 freeze_after 高 2.56pp，Ens Last 高 2.34pp。

3. **lora_always 的 Ens Last 81.51** 已经接近历史 82.79 水平。

---

## 关键发现：LR-RGDA 配置差异

Pre-Wave A 使用了 CLI 默认的 LR-RGDA 配置：

| 参数 | Pre-Wave A（当前） | 历史 82.79 |
|------|:---:|:---:|
| `--num_centers` | **1**（单中心） | **4**（多中心） |
| `--rgda_train_iter` | **0**（解析式，无微调） | **200**（GMM 采样微调） |
| `--rgda_fit_source` | gmm_sample | gmm_sample |
| `--rgda_train_lr` | 0.01（未使用） | 0.01 |

历史最优 Ensemble 结果是 `ens_mc4ft200_a0p05`（m=4, ft=200），比当前解析式单中心 LR-RGDA 强 **1-2pp**（基于 6/18 的实验经验）。

**因此当前 Pre-Wave A 的 Ensemble 指标是下界**——后续 Waves 恢复多中心微调后，所有 Ens 指标预期会上涨。ZS 指标不受影响（ZS 分类器不依赖 LR-RGDA）。

### 影响

- Pre-Wave A 的 ZS 指标可直接用于 backbone 和 text_schedule 的选择（不受分类器影响）
- Ensemble 指标的比较仅在同分类器配置下有效，不可直接与历史 82.79 对标
- 后续 Waves 应在公共配置中加入 `--num_centers 4 --rgda_train_iter 200 --rgda_train_lr 0.01 --rgda_fit_source gmm_sample`

### 教训

启动脚本应显式传入所有关键参数，而非依赖 CLI 默认值。`launch_pre_wave_a.sh` 中遗漏了 LR-RGDA 分类器配置。
