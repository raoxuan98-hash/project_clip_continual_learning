# LADA Combo 消融实验

**日期**: 2026-06-23
**目的**: 在最优配置（lada_hybrid）基础上，逐个替换 LADA 训练参数，找出兼容/冲突项

## 基线配置

```
lr=1e-4, weight_decay=3e-5, batch_size=32, scheduler=cosine
train_budget=uniform(800 iter), text_classifier_mode=lada_hybrid
num_centers=4, alpha=0.05, rgda_train_iter=200
gmm_sample_mode=mean, text_tuning_schedule=low_lr_after, fd=cd=1.0
```

## 实验矩阵

| 实验 | lr | wd | bs | scheduler | train_budget |
|------|:---:|:---:|:---:|:---:|:---:|
| Baseline | 1e-4 | 3e-5 | 32 | cosine | uniform |
| lr=1e-3 | **1e-3** | 3e-5 | 32 | cosine | uniform |
| wd=5e-4 | 1e-4 | **5e-4** | 32 | cosine | uniform |
| bs=64 | 1e-4 | 3e-5 | **64** | cosine | uniform |
| onecycle | 1e-4 | 3e-5 | 32 | **onecycle** | uniform |
| lada_epochs | 1e-4 | 3e-5 | 32 | cosine | **lada_epochs** |
| combo_best | 1e-4 | 3e-5 | 64 | onecycle | lada_epochs |
| LADA combo | 1e-3 | 5e-4 | 64 | onecycle | lada_epochs |

## Ensemble 整体对比

| 实验 | Transfer | Average | Last | vs Baseline |
|------|:---:|:---:|:---:|:---:|
| Baseline | 60.6 | 70.5 | 82.3 | — |
| lr=1e-3 | 57.0 | 66.6 | 77.8 | **-4.5** ❌ |
| wd=5e-4 | 60.6 | 70.5 | 82.2 | -0.1 |
| bs=64 | 60.2 | 70.6 | 82.6 | +0.3 |
| onecycle | 60.4 | 70.5 | 82.6 | +0.3 |
| lada_epochs | 60.3 | 70.7 | 82.6 | +0.3 |
| combo_best | 60.5 | 70.7 | 82.5 | +0.2 |
| LADA combo | 56.7 | 66.1 | 78.0 | **-4.3** ❌ |

## Per-dataset 详细数据

### Baseline (lada_hybrid)

**Zero-shot**

| | aircraft | caltech101 | dtd | eurosat | flowers | food101 | mnist | oxford_pets | stanford_cars | sun397 | **Overall** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Transfer | N/A | 76.3 | 35.3 | 37.4 | 64.6 | 82.2 | 44.7 | 84.2 | 55.6 | 61.9 | **60.3** |
| Average | 43.7 | 81.0 | 54.6 | 66.9 | 75.3 | 83.7 | 63.8 | 86.1 | 58.3 | 62.6 | **67.6** |
| Last | 43.0 | 80.4 | 58.3 | 71.4 | 82.2 | 85.3 | 92.1 | 90.3 | 68.8 | 68.8 | **74.0** |

**LR-RGDA**

| | aircraft | caltech101 | dtd | eurosat | flowers | food101 | mnist | oxford_pets | stanford_cars | sun397 | **Overall** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Average | 48.5 | 82.7 | 56.2 | 63.2 | 58.5 | 41.7 | 37.9 | 26.6 | 16.4 | 7.4 | **43.9** |
| Last | 47.2 | 90.3 | 67.6 | 88.1 | 97.1 | 83.1 | 94.5 | 87.9 | 81.5 | 74.1 | **81.1** |

**Ensemble (α=0.05)**

| | aircraft | caltech101 | dtd | eurosat | flowers | food101 | mnist | oxford_pets | stanford_cars | sun397 | **Overall** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Transfer | N/A | 76.3 | 35.5 | 37.5 | 64.8 | 82.2 | 46.6 | 84.6 | 55.7 | 62.2 | **60.6** |
| Average | 50.1 | 85.6 | 58.8 | 68.5 | 81.4 | 83.8 | 65.6 | 86.6 | 61.2 | 63.6 | **70.5** |
| Last | 50.0 | 89.4 | 69.1 | 87.7 | 96.1 | 85.4 | 94.3 | 91.2 | 83.2 | 76.2 | **82.3** |

### lr=1e-3

**Zero-shot**

| | aircraft | caltech101 | dtd | eurosat | flowers | food101 | mnist | oxford_pets | stanford_cars | sun397 | **Overall** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Transfer | N/A | 74.0 | 34.6 | 28.6 | 61.7 | 78.1 | 47.2 | 82.7 | 52.4 | 61.3 | **57.8** |
| Average | 44.3 | 81.1 | 56.5 | 64.7 | 76.6 | 83.7 | 65.8 | 85.8 | 57.9 | 62.5 | **67.9** |
| Last | 43.0 | 80.4 | 58.3 | 71.4 | 82.2 | 85.3 | 92.1 | 90.3 | 68.8 | 68.8 | **74.0** |

**LR-RGDA**

| | aircraft | caltech101 | dtd | eurosat | flowers | food101 | mnist | oxford_pets | stanford_cars | sun397 | **Overall** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Average | 49.2 | 83.5 | 56.5 | 49.3 | 59.5 | 39.9 | 35.4 | 27.5 | 16.7 | 8.5 | **42.6** |
| Last | 47.9 | 90.7 | 67.7 | 88.2 | 97.4 | 83.6 | 94.0 | 88.8 | 81.9 | 74.7 | **81.5** |

**Ensemble (α=0.05)**

| | aircraft | caltech101 | dtd | eurosat | flowers | food101 | mnist | oxford_pets | stanford_cars | sun397 | **Overall** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Transfer | 0 | 74.3 | 33.4 | 29.0 | 60.7 | 78.0 | 45.4 | 82.7 | 51.5 | 59.9 | **57.0** |
| Average | 50.5 | 85.0 | 56.4 | 45.2 | 82.6 | 80.2 | 62.2 | 84.6 | 56.6 | 62.3 | **66.6** |
| Last | 46.7 | 88.2 | 66.7 | 62.1 | 97.6 | 78.7 | 92.9 | 88.0 | 81.9 | 75.1 | **77.8** |

### wd=5e-4

**Zero-shot**

| | aircraft | caltech101 | dtd | eurosat | flowers | food101 | mnist | oxford_pets | stanford_cars | sun397 | **Overall** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Transfer | N/A | 76.3 | 35.3 | 37.4 | 64.6 | 82.2 | 44.7 | 84.2 | 55.6 | 61.9 | **60.3** |
| Average | 43.7 | 81.0 | 54.6 | 66.9 | 75.3 | 83.7 | 63.8 | 86.1 | 58.3 | 62.6 | **67.6** |
| Last | 43.0 | 80.4 | 58.3 | 71.4 | 82.2 | 85.3 | 92.1 | 90.3 | 68.8 | 68.8 | **74.0** |

**LR-RGDA**

| | aircraft | caltech101 | dtd | eurosat | flowers | food101 | mnist | oxford_pets | stanford_cars | sun397 | **Overall** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Average | 48.5 | 82.7 | 56.2 | 63.2 | 58.5 | 41.7 | 37.9 | 26.6 | 16.4 | 7.4 | **43.9** |
| Last | 47.2 | 90.3 | 67.6 | 88.1 | 97.1 | 83.1 | 94.5 | 87.9 | 81.5 | 74.1 | **81.1** |

**Ensemble (α=0.05)**

| | aircraft | caltech101 | dtd | eurosat | flowers | food101 | mnist | oxford_pets | stanford_cars | sun397 | **Overall** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Transfer | N/A | 76.3 | 35.3 | 37.2 | 64.5 | 82.3 | 46.3 | 84.6 | 55.8 | 62.2 | **60.6** |
| Average | 50.1 | 85.6 | 58.8 | 68.5 | 81.5 | 83.8 | 65.5 | 86.6 | 61.2 | 63.6 | **70.5** |
| Last | 50.1 | 89.4 | 69.4 | 87.5 | 96.2 | 85.4 | 94.1 | 91.2 | 83.2 | 76.1 | **82.2** |

### bs=64

**Zero-shot**

| | aircraft | caltech101 | dtd | eurosat | flowers | food101 | mnist | oxford_pets | stanford_cars | sun397 | **Overall** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Transfer | N/A | 76.4 | 35.0 | 37.4 | 64.5 | 82.4 | 44.6 | 84.2 | 55.6 | 61.8 | **60.2** |
| Average | 43.7 | 81.0 | 54.7 | 67.0 | 75.7 | 83.8 | 63.3 | 86.2 | 58.4 | 62.8 | **67.7** |
| Last | 43.0 | 80.4 | 58.3 | 71.4 | 82.2 | 85.3 | 92.1 | 90.3 | 68.8 | 68.8 | **74.0** |

**LR-RGDA**

| | aircraft | caltech101 | dtd | eurosat | flowers | food101 | mnist | oxford_pets | stanford_cars | sun397 | **Overall** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Average | 48.5 | 82.9 | 56.5 | 63.5 | 58.1 | 40.4 | 37.5 | 26.4 | 16.5 | 7.4 | **43.8** |
| Last | 47.2 | 90.5 | 67.9 | 88.1 | 97.1 | 82.7 | 94.3 | 88.0 | 81.7 | 74.2 | **81.1** |

**Ensemble (α=0.05)**

| | aircraft | caltech101 | dtd | eurosat | flowers | food101 | mnist | oxford_pets | stanford_cars | sun397 | **Overall** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Transfer | N/A | 76.2 | 35.2 | 37.3 | 64.6 | 82.1 | 45.4 | 84.5 | 55.6 | 61.8 | **60.2** |
| Average | 49.8 | 85.5 | 58.9 | 69.1 | 82.2 | 83.9 | 64.7 | 86.8 | 61.2 | 63.7 | **70.6** |
| Last | 49.9 | 89.7 | 70.4 | 88.4 | 97.3 | 85.4 | 94.2 | 91.4 | 83.7 | 76.4 | **82.6** |

### onecycle

**Ensemble (α=0.05)**

| | aircraft | caltech101 | dtd | eurosat | flowers | food101 | mnist | oxford_pets | stanford_cars | sun397 | **Overall** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Transfer | N/A | 76.3 | 35.3 | 37.5 | 64.8 | 82.4 | 46.4 | 84.7 | 55.9 | 62.2 | **60.4** |
| Average | 50.0 | 85.2 | 58.8 | 68.4 | 81.8 | 83.7 | 65.6 | 86.6 | 61.3 | 63.6 | **70.5** |
| Last | 49.7 | 89.4 | 69.3 | 88.0 | 96.4 | 85.5 | 94.4 | 91.2 | 83.3 | 76.4 | **82.6** |

### lada_epochs

**Ensemble (α=0.05)**

| | aircraft | caltech101 | dtd | eurosat | flowers | food101 | mnist | oxford_pets | stanford_cars | sun397 | **Overall** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Transfer | N/A | 76.4 | 35.6 | 37.6 | 64.8 | 82.5 | 46.4 | 84.7 | 55.7 | 62.0 | **60.3** |
| Average | 50.1 | 85.4 | 58.7 | 68.5 | 82.2 | 83.8 | 65.6 | 86.8 | 61.4 | 63.9 | **70.7** |
| Last | 49.9 | 89.3 | 69.6 | 88.2 | 97.0 | 85.5 | 94.5 | 91.3 | 83.7 | 76.4 | **82.6** |

### combo_best（bs=64 + onecycle + lada_epochs）

**Ensemble (α=0.05)**

| | aircraft | caltech101 | dtd | eurosat | flowers | food101 | mnist | oxford_pets | stanford_cars | sun397 | **Overall** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Transfer | N/A | 75.4 | 35.6 | 39.1 | 64.5 | 81.9 | 49.0 | 84.1 | 53.1 | 61.9 | **60.5** |
| Average | 51.9 | 83.1 | 57.7 | 72.1 | 83.5 | 83.3 | 67.0 | 85.7 | 59.4 | 63.3 | **70.7** |
| Last | 51.2 | 89.2 | 69.0 | 88.6 | 97.1 | 85.1 | 94.3 | 89.6 | 84.5 | 76.3 | **82.5** |

### LADA combo（全部替换）

**Zero-shot**

| | aircraft | caltech101 | dtd | eurosat | flowers | food101 | mnist | oxford_pets | stanford_cars | sun397 | **Overall** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Transfer | N/A | 73.6 | 33.7 | 28.3 | 62.6 | 78.2 | 47.5 | 82.8 | 51.5 | 61.0 | **57.6** |
| Average | 44.3 | 81.1 | 56.8 | 63.5 | 77.7 | 84.0 | 66.6 | 85.5 | 56.3 | 62.6 | **67.8** |
| Last | 43.0 | 80.4 | 58.3 | 71.4 | 82.2 | 85.3 | 92.1 | 90.3 | 68.8 | 68.8 | **74.0** |

**LR-RGDA**

| | aircraft | caltech101 | dtd | eurosat | flowers | food101 | mnist | oxford_pets | stanford_cars | sun397 | **Overall** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Average | 49.2 | 83.7 | 57.0 | 50.5 | 59.0 | 35.5 | 38.6 | 27.9 | 17.1 | 8.4 | **42.7** |
| Last | 47.6 | 91.2 | 68.0 | 88.2 | 97.4 | 83.4 | 94.8 | 88.7 | 82.2 | 74.9 | **81.6** |

**Ensemble (α=0.05)**

| | aircraft | caltech101 | dtd | eurosat | flowers | food101 | mnist | oxford_pets | stanford_cars | sun397 | **Overall** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Transfer | N/A | 74.0 | 33.1 | 29.2 | 60.1 | 77.8 | 44.5 | 82.6 | 49.0 | 60.2 | **56.7** |
| Average | 50.1 | 85.0 | 56.1 | 45.2 | 82.6 | 79.8 | 59.9 | 84.4 | 56.1 | 61.8 | **66.1** |
| Last | 45.2 | 87.5 | 65.5 | 65.0 | 97.0 | 80.7 | 93.4 | 87.2 | 82.7 | 75.9 | **78.0** |

## 结论

| 参数 | 单独效果 | 结论 |
|------|:---:|------|
| lr=1e-3 | **-4.5** | ❌ 绝对不能用，LoRA-NSP 需要 1e-4 |
| wd=5e-4 | -0.1 | ⬜ 无影响 |
| bs=64 | +0.3 | ✅ 采纳 |
| onecycle | +0.3 | ✅ 采纳 |
| lada_epochs | +0.3 | ✅ 采纳 |
| 三个收益组合 | +0.2 | ❌ 不叠加，互相抵消 |
| **全部 combo** | **-4.3** | ❌ lr=1e-3 一个参数毁所有 |

**当前最优配置：Baseline (lada_hybrid) + bs=64 → Transfer 60.2 / Average 70.6 / Last 82.6**


---

## 视觉/文本解耦消融（仅文本端改 LADA 参数）

**日期**: 2026-06-24

### 目的

之前消融发现 LADA 训练配方不兼容我们——但改动都是视觉和文本**同时生效**。本次将参数解耦，视觉端保持我们的配置不变，**仅在文本端**逐个替换 LADA 参数。

### 基线配置

**视觉端**：lr=1e-4, wd=3e-5, scheduler=cosine, budget=uniform(800 iter)  
**文本端**：lr=1e-4→2e-5(Task2+), wd=3e-5, scheduler=cosine, budget=uniform(800 iter)  
**通用**：bs=64, text_classifier_mode=lada_hybrid, num_centers=4, alpha=0.05, rgda_train_iter=200, gmm_sample_mode=mean, fd=cd=1.0

### 代码改动

在 `lora_nsp_trainer.py` 中将 vision / text 的 optimizer 和 scheduler 拆成两套独立控制。新增 CLI 参数：`--text_lr`、`--text_weight_decay`、`--text_scheduler`、`--text_train_budget_mode`。改动 ~50 行。

### 实验矩阵

| 实验 | 视觉端 | 文本端 | 变量 |
|------|------|------|------|
| Baseline | 同上 | 同上 | — |
| text_lr=1e-3 | 同上 | lr=**1e-3**→2e-4(Task2+) | lr |
| text_wd=5e-4 | 同上 | wd=**5e-4** | wd |
| text_onecycle | 同上 | scheduler=**onecycle** | scheduler |
| text_lada_epochs | 同上 | budget=**lada_epochs** | budget |
| text_combo | 同上 | lr=1e-3, wd=5e-4, onecycle, lada_epochs | 全部 |

### Ensemble 整体对比

| 实验 | Transfer | Average | Last | vs Baseline |
|------|:---:|:---:|:---:|:---:|
| Baseline | 60.2 | 70.6 | 82.5 | — |
| text_lr=1e-3 | 60.6 | 67.8 | 82.4 | -0.1 |
| text_wd=5e-4 | 60.2 | 70.6 | 82.5 | 0 |
| text_onecycle | 60.1 | 70.5 | 82.6 | +0.1 |
| text_lada_epochs | 60.1 | 70.6 | 82.5 | 0 |
| text_combo | 60.5 | 67.9 | 82.3 | -0.2 |

### 结论

1. **解耦几乎没用**——只改文本端的结果和之前的结论完全一致
2. **text_lr=1e-3 只伤 Average（-2.8），不动 Last（-0.1）**——和双端 lr=1e-3 的 Last -4.5 对比，证明退步来自视觉端被破坏
3. **wd、onecycle、lada_epochs 零影响**——不管同时改还是只改文本端，都毫无效果
4. **LADA 的 training tricks 对我们的框架完全无效**，收益来源是别处（NSP + GMM fit + lada_hybrid）
