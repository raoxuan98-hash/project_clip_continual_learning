# LADA Combo 消融 & 视觉/文本参数解耦

**日期**: 2026-06-22 ~ 2026-06-25

## 1. 背景

v4 代码合并后，最优基线为 lada_hybrid 模式 + num_centers=4 + rgda_train_iter=200 + gmm_sample_mode=mean，Ensemble Last=82.3。在此基础上探索 LADA 训练配方（lr=1e-3, wd=5e-4, onecycle, lada_epochs）是否能进一步提升。

## 2. 双端消融（视觉和文本同时改）

固定基线：lada_hybrid, bs=32, lr=1e-4, wd=3e-5, scheduler=cosine, budget=uniform(800)

逐个替换 LADA 参数，6 个实验 + 1 个组合：

| 实验 | 改动 | Ensemble Last | Δ |
|------|------|:---:|:---:|
| Baseline | — | 82.3 | — |
| lr=1e-3 | lr→1e-3 | 77.8 | **-4.5** |
| wd=5e-4 | wd→5e-4 | 82.2 | -0.1 |
| bs=64 | bs→64 | **82.6** | +0.3 |
| onecycle | scheduler→onecycle | 82.6 | +0.3 |
| lada_epochs | budget→lada_epochs | 82.6 | +0.3 |
| combo_best | bs64+onecycle+lada_epochs | 82.5 | +0.2 |
| LADA combo | all 5 params | 78.0 | **-4.3** |

### 结论

- lr=1e-3 是 LADA combo 退步的**唯一罪魁祸首**（-4.5）
- wd/bs/onecycle/lada_epochs 各自单独微涨 +0.3，但组合不叠加
- **当前最优：Baseline + bs=64 → Last 82.6**

## 3. 视觉/文本参数解耦

### 动机

双端消融中 lr=1e-3 两边都改了。LADA 本身只训文本端，怀疑退步来自视觉端被破坏。将 trainer 中 optimizer/scheduler 拆成 vision/text 两套独立控制。

### 代码改动

- `lora_nsp_trainer.py`: ~50 行，vision / text 各自独立 optimizer + scheduler，训练循环分开 step
- `main_incremental.py`: ~20 行，新增 `--text_lr`、`--text_weight_decay`、`--text_scheduler`、`--text_train_budget_mode`

### 解耦消融（视觉端保持基线，仅改文本端）

| 实验 | 文本端改动 | Ensemble Last | Δ |
|------|------|:---:|:---:|
| Baseline | — | 82.5 | — |
| text_lr=1e-3 | lr=1e-3→2e-4(Task2+) | 82.4 | -0.1 |
| text_wd=5e-4 | wd=5e-4 | 82.5 | 0 |
| text_onecycle | scheduler=onecycle | 82.6 | +0.1 |
| text_lada_epochs | budget=lada_epochs | 82.5 | 0 |
| text_combo | 四个全改 | 82.3 | -0.2 |

### 结论

- 解耦几乎没用：只改文本端的结果和双端消融完全一致
- text_lr=1e-3 只伤 Average（-2.8），不动 Last（-0.1）——与双端 lr=1e-3 的 Last -4.5 对比，证明退步确实来自视觉端被高 lr 破坏
- wd/onecycle/lada_epochs 无论双端还是仅文本端都零影响
- **LADA training tricks 对我们的框架完全无效。收益来自 NSP + GMM fit + lada_hybrid，与训练配方无关**
- 解耦代码已回退，保留 lada_hybrid + bs=64 配置

## 4. LADA 源码复现

用 LADA 官方脚本在 X-TAIL 16-shot Order1 上复现：
- LADA 论文：Transfer 56.7 / Average 68.9 / Last 83.1
- 我们复现 (seed=42)：Transfer 61.6 / Average 72.5 / Last **82.9**
- 仅差 0.2，论文数据可信

## 5. Alpha Sweep

在 Exp3（num_centers=4 + GMM mean replay + rgda_train_iter=200）基础上扫 α∈[0,1]，α=0.05 仍然最优。

## 6. 当前最优配置

```
lada_hybrid text classifier
num_centers=4, rgda_train_iter=200, gmm_sample_mode=mean
alpha=0.05, bs=64, lr=1e-4 (vision) + 1e-4→2e-5 (text)
scheduler=cosine, budget=uniform(800), fd=cd=1.0
Ensemble Last = 82.6
```

## 相关文件

- `docs/combo_ablation.md` — 完整消融实验记录
- `docs/param_decoupling_plan.md` — 解耦计划
- `docs/all_experiments.md` — 全部 34 个实验结果排序
- `main_incremental.py` — 当前版本（解耦已回退）
