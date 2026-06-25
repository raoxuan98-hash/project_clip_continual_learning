# Visual Tuning Anchor Sweep Results

**日期**: 2026-06-18
**实验目录**:

```text
experiments/visual_tuning_anchor_sweep_20260618_0112
```

---

## 1. 完成状态

本轮目标是固定“文本侧微调 + 每任务后缓存 tuned text prototypes / semantic anchors”，比较视觉侧不同训练方式的边际收益。

三个配置均已完成训练和 full-test 离线 sweep：

| config | artifacts | sweep JSON | summary |
|---|---:|---:|---|
| `text_only_anchor_seed42` | 10/10 | 22/22 | yes |
| `vanilla_vision_anchor_seed42` | 10/10 | 22/22 | yes |
| `nsp_fd_cd_vision_anchor_seed42` | 10/10 | 22/22 | yes |

总计：

```text
artifact.pt: 30
result JSON: 66
rgda_sweep_summary.csv: 3
final reports: generated
```

最终报告：

```text
experiments/visual_tuning_anchor_sweep_20260618_0112/reports/
  visual_tuning_anchor_comparison_wide.{csv,md}
  visual_tuning_anchor_comparison_deltas.{csv,md}
  visual_tuning_anchor_comparison_best_ensemble.{csv,md}
  visual_tuning_anchor_comparison_status.json
```

`status.json` 中 `missing={}`，说明三个预期配置均被汇总脚本加载。

---

## 2. 协议确认

本轮仍然遵守严格增量设定：

- 不使用过去任务 real-feature replay；
- LR-RGDA analytical classifier 使用 compact stats；
- classifier-only fine-tune 使用 `gmm_memory` 采样的 pseudo-features；
- unseen classes 不接收 LR-RGDA logits，只由 zero-shot classifier 覆盖；
- seen classes 的文本分类器使用每任务后缓存的 tuned text prototypes；
- unseen classes 使用 frozen CLIP text features。

离线 sweep 方法：

```text
zero_shot
rgda_sc
rgda_mc4
rgda_mc4ft200: m=4, ft=200, lr=0.01, source=gmm_sample
ensemble alpha = 0,0.05,0.1,0.2,0.5,1.0
```

---

## 3. 工程问题与修复

### 3.1 full-test sweep 的 OOM

原始离线 evaluator 在每个数据集抽完 full-test features 后，会一次性执行：

```python
rgda_logits = rgda_classifier.forward(features)
```

在 `sun397` 等大测试集、step 后期类别数较多时，LR-RGDA 内部的 `einsum` 会产生很大的临时张量。旧进程在 GPU0 上出现 OOM：

```text
torch.OutOfMemoryError: CUDA out of memory
```

这不是协议错误，也不是 artifact 错误，而是 evaluator 没有 chunk LR-RGDA forward。

修复：

```text
scripts/evaluate_incremental_rgda_sweep_artifacts.py
  --rgda_eval_chunk_size 512
```

新增 `_rgda_forward_chunked()`，对 LR-RGDA logits 分块计算后拼接。修复后：

- `text_only_anchor_seed42` 用 chunked evaluator 在 GPU1 重跑成功；
- `vanilla_vision_anchor_seed42` 用 chunked evaluator 在 GPU5 重跑成功；
- `nsp_fd_cd_vision_anchor_seed42` 旧进程已先完成，结果有效。

### 3.2 fast eval 开关

为了后续快速诊断，还新增：

```text
--eval_shots_per_class 32
--eval_subset_seed 0
```

语义：

- 默认 `eval_shots_per_class=0`，仍为 full-test；
- 设为 `32` 时，每个测试数据集每类最多抽 32 个 test samples；
- 抽样按 `eval_subset_seed` 固定，保证同一 dataset 在所有 incremental steps 使用同一 subset；
- 结果 JSON 会记录 `eval_shots_per_class` 和 `eval_subset_seed`，避免和 full-test 混淆。

本次正式结果使用 full-test，不是 fast32。

---

## 4. 核心 full-test 结果

### 4.1 Zero-shot

| config | Transfer | Average | Last |
|---|---:|---:|---:|
| `text_only_anchor_seed42` | 61.15 | 58.02 | 64.00 |
| `vanilla_vision_anchor_seed42` | 57.37 | 60.44 | 72.07 |
| `nsp_fd_cd_vision_anchor_seed42` | 59.85 | 68.54 | 77.72 |

观察：

- 普通视觉侧微调相对 text-only：Transfer -3.78，Average +2.41，Last +8.07。
- LoRA-NSP + FD/CD 相对 text-only：Transfer -1.30，Average +10.51，Last +13.72。
- LoRA-NSP + FD/CD 相对 vanilla vision：Transfer +2.49，Average +8.10，Last +5.65。

这说明视觉侧不是没有收益；关键在于普通视觉微调的 open-vocabulary Transfer 损伤较大，而 NSP+FD/CD 明显缓解了该问题。

### 4.2 Best ensemble by Average

三个配置按 Average 选择的 diagnostic best ensemble 都是：

```text
ens_mc4ft200_a0p05
```

| config | method | Transfer | Average | Last |
|---|---|---:|---:|---:|
| `text_only_anchor_seed42` | `ens_mc4ft200_a0p05` | 61.27 | 59.98 | 80.55 |
| `vanilla_vision_anchor_seed42` | `ens_mc4ft200_a0p05` | 57.47 | 62.03 | 78.22 |
| `nsp_fd_cd_vision_anchor_seed42` | `ens_mc4ft200_a0p05` | 60.42 | 70.38 | 82.79 |

相对 text-only best ensemble：

| comparison | delta Transfer | delta Average | delta Last |
|---|---:|---:|---:|
| vanilla - text_only | -3.79 | +2.05 | -2.32 |
| NSP+FD/CD - text_only | -0.85 | +10.40 | +2.25 |
| NSP+FD/CD - vanilla | +2.94 | +8.35 | +4.57 |

按照先前 gate：

```text
视觉侧配置若 Average 和 Last 均提升 >= 1.0，且 Transfer 下降 <= 1.0-1.5，则认为有明确收益空间。
```

NSP+FD/CD 满足这一趋势：

- Average 大幅提升；
- Last 提升；
- Transfer 损伤约 0.85，在可接受范围。

vanilla vision 的结果更像 tradeoff：

- Average 比 text-only 高；
- zero-shot Last 明显高；
- 但 best ensemble Last 低于 text-only，Transfer 损伤明显。

---

## 5. 解释

### 5.1 text-only anchor 的性质

`text_only_anchor_seed42` 固定视觉编码器，只微调文本侧并缓存每任务 tuned text prototypes。这能保持较高 Transfer：

```text
zero_shot Transfer = 61.15
best ensemble Transfer = 61.27
```

但其 Average/Last 相对低，说明只调文本侧仍有不足，尤其是视觉特征本身未适配 seen-task 数据分布。

### 5.2 vanilla vision tuning 的性质

普通视觉侧微调提高了 zero-shot Average/Last：

```text
zero_shot Average: 58.02 -> 60.44
zero_shot Last:    64.00 -> 72.07
```

但 Transfer 明显下降：

```text
61.15 -> 57.37
```

在 best ensemble 下，vanilla 的 Average 上升到 62.03，但 Last 反而低于 text-only best ensemble：

```text
80.55 -> 78.22
```

因此 vanilla vision tuning 不是当前最优方向；它证明视觉侧有收益空间，但代价控制不好。

### 5.3 LoRA-NSP + FD/CD 的性质

`nsp_fd_cd_vision_anchor_seed42` 是本轮最强配置。

zero-shot:

```text
Transfer = 59.85
Average  = 68.54
Last     = 77.72
```

best ensemble:

```text
Transfer = 60.42
Average  = 70.38
Last     = 82.79
```

它相对 text-only 的关键增益：

```text
zero_shot Average +10.51
zero_shot Last    +13.72
best Average      +10.40
best Last         +2.25
Transfer only     -0.85 for best ensemble
```

这说明视觉侧微调本身有明确收益，但需要 NSP/FD/CD 之类约束来控制 Transfer 损伤。

---

## 6. 与先前假设的关系

先前讨论中，我们怀疑：

```text
视觉侧同时微调可能会增加 seen-task separability，但也可能破坏 zero-shot alignment。
```

本轮结果支持这个判断，但进一步说明：

- `vanilla_vision_anchor` 的确有较强 tradeoff；
- `nsp_fd_cd_vision_anchor` 把 tradeoff 明显推向有利方向；
- 因此“不微调视觉侧”不是最优结论；
- 更合理的结论是：视觉侧可以微调，但必须配合 NSP/FD/CD 或类似保持结构的约束。

---

## 7. 下一步建议

### 7.1 保留 NSP+FD/CD 视觉侧路线

当前 evidence 足够支持继续推进：

```text
tune_text_encoder=true
tune_vision_encoder=true
lora_type=lora_nsp
fd_weight=1.0
cd_weight=1.0
text_classifier_mode=lada_hybrid
ensemble = mc4ft200, alpha=0.05
```

这是当前最强的 full-test 配置。

### 7.2 不建议单独押注 vanilla vision tuning

普通视觉侧微调可以作为 ablation 保留，但不应作为主方案：

- Transfer 损伤较大；
- best ensemble Last 不如 text-only；
- 说明没有约束的视觉适配会破坏部分泛化结构。

### 7.3 继续测试 lr=5e-4

仍建议对视觉侧做低学习率 follow-up：

- 对 vanilla vision：看能否缓解 Transfer -3.78 的损伤；
- 对 NSP+FD/CD：看是否进一步把 Transfer 从 60.42 拉近 text-only 61.27，同时保持 Average/Last。

优先级：

```text
1. nsp_fd_cd_vision_anchor, lr=5e-4
2. vanilla_vision_anchor, lr=5e-4
```

### 7.4 fixed alpha 与 adaptive alpha

本轮 best Average 都落在：

```text
ens_mc4ft200_a0p05
```

这说明在 `mc4ft200 + gmm_sample` 下，小 alpha 仍然是合理默认值。

后续可以做：

```text
fixed alpha = 0.05 as main report
best alpha as diagnostic upper bound
adaptive alpha as follow-up
```

---

## 8. 需要注意的边界

本轮结果是：

```text
seed=42
10 datasets
full-test evaluation
```

结论强于之前 4-dataset ablation，但仍需要多 seed 验证稳定性。

另外，本轮比较的是“视觉侧训练策略”在 LADA-style text anchors + LR-RGDA ensemble 下的效果，不应直接外推到完全不同 classifier 或没有 text anchors 的设定。

