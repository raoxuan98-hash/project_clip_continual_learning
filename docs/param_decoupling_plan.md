# 视觉端/文本端参数解耦计划

**日期**: 2026-06-24

## 1. 当前参数状态

| 参数 | 视觉端 | 文本端 | 解耦？ | 代码位置 |
|------|------|------|:---:|------|
| **lr** | `args.lr` = 1e-4 | `text_lr` 独立传入 | ✅ 已解耦 | trainer L326/330 |
| **wd** | `args.weight_decay` = 3e-5 | 同左 | ❌ | trainer L340/343 |
| **scheduler** | cosine | 同左 | ❌ | trainer L347-351 |
| **train_budget** | `args.iterations` = 800 | 同左 | ❌ | main_incremental L731 |
| **text_classifier_mode** | — | `lada_hybrid` / `current` | ✅ | 仅文本端概念 |

## 2. 需要解耦的参数

### scheduler

当前：一个 `CosineAnnealingLR` 管所有 param_groups

解耦方案：
- 视觉：独立 optimizer + CosineAnnealingLR
- 文本：独立 optimizer + OneCycleLR（LADA 配方）

改动量：~15 行（trainer 的 `train()` 方法）

### train_budget

当前：`args.iterations` 统一控制

解耦方案：
- 视觉：`--iterations` 保持 800（uniform）
- 文本：新增 `--text_iterations` 或复用 `--train_budget_mode lada_epochs` 仅作用于文本端

改动量：~10 行（`_resolve_task_train_iterations` 函数 + `train()` 方法）

### wd

当前：`args.weight_decay` 全局共享

解耦方案：
- 视觉：`--weight_decay` = 3e-5
- 文本：新增 `--text_weight_decay`，默认 = args.weight_decay

改动量：~5 行（trainer param_groups）

### 总计改动：~30 行，均在 `src/trainers/lora_nsp_trainer.py`

## 3. 解耦后的消融计划

基线方法保持不变：

**视觉端**：lr=1e-4, wd=3e-5, scheduler=cosine, budget=uniform(800 iter)  
**文本端 baseline**：lr=1e-4 (Task1)→2e-5 (Task2+), wd=3e-5, scheduler=cosine, budget=uniform(800 iter)  
**通用**：bs=64, text_classifier_mode=lada_hybrid, num_centers=4, alpha=0.05, rgda_train_iter=200, gmm_sample_mode=mean, fd=cd=1.0

### 单变量消融（仅改文本端，视觉端同上）

| 实验 | 视觉端 | 文本端 | 变量 |
|------|------|------|------|
| Baseline | 同上 | 同上 | — |
| text_lr=1e-3 | 同上 | lr=**1e-3** (Task1)→2e-4 (Task2+) | lr |
| text_wd=5e-4 | 同上 | wd=**5e-4** | wd |
| text_onecycle | 同上 | scheduler=**onecycle** | scheduler |
| text_lada_epochs | 同上 | budget=**lada_epochs** | budget |

### LADA text combo

| 实验 | 视觉端 | 文本端 |
|------|------|------|
| text_combo | 同上 | lr=1e-3 (Task1)→2e-4 (Task2+), wd=5e-4, scheduler=onecycle, budget=lada_epochs |

## 4. 预期

如果 LADA combo 退步来自视觉端（之前两边都改 lr=1e-3 时崩了），那只在文本端上 LADA 配方应该不会有负影响——LADA 本身就是纯文本端操作。如果仍然退步，说明文本端这些参数和我们的 LoRA-NSP 体系也存在冲突。
