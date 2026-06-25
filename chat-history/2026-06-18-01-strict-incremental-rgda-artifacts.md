# Strict Incremental LR-RGDA Artifacts and Multi-M Sweep

**日期**: 2026-06-18
**主题**: 对齐严格增量设定下的 LR-RGDA classifier fine-tune 数据来源，并扩展异步 artifact 以支持不保存 real features 的多中心数离线评估。

---

## 1. 关键结论

严格增量学习中，训练到新任务时不能继续保存或使用过去任务的 real features 作为 replay。

因此当前协议区分两类信息：

- LR-RGDA analytical classifier 构建：
  - 使用当前任务真实特征在任务结束时构建 compact statistics；
  - 保存并累积 `global_stats_dict`、`global_center_means`、`dataset_balanced_global_cov`；
  - 这些是分布统计，不是旧任务 real-feature replay。

- LR-RGDA classifier-only fine-tune：
  - 如果 `rgda_train_iter > 0`，不再默认使用旧 real features；
  - 默认使用 per-class spherical GMM compact memory 生成 pseudo-features；
  - 可选来源：
    - `gmm_sample`
    - `gmm_mean`
    - `center_replay`
    - `none`

这点修正了之前 diagnostic sweep 中使用 artifact 内 `classifier_features/classifier_labels` 的潜在不严格问题。

---

## 2. main_incremental.py 更新

新增参数：

```text
--rgda_fit_source none|center_replay|gmm_mean|gmm_sample
```

语义：

```text
只影响 LR-RGDA classifier-only fine-tune。
不影响 LR-RGDA analytical mean/cov/multi-center stats 构建。
默认值为 gmm_sample。
```

新增参数：

```text
--artifact_num_centers 1,4
```

语义：

```text
训练主分类器仍由 --num_centers 决定。
artifact_num_centers 只控制 step artifact 额外保存哪些 M 的 compact LR-RGDA stats。
--num_centers 会被自动纳入保存集合。
```

例子：

```bash
python main_incremental.py \
  --num_centers 4 \
  --artifact_num_centers 1,4 \
  --save_step_artifacts \
  --skip_inline_eval
```

这会：

- inline / 训练时实际 LR-RGDA 使用 `M=4`；
- artifact 额外保存 `M=1` 和 `M=4` 两套 compact stats；
- 不保存旧任务 real features。

---

## 3. Artifact 协议

原有主字段仍保留：

```text
global_stats_dict
global_center_means
dataset_balanced_global_cov
lr_rgda.state_dict
gmm_memory
```

新增：

```text
rgda_stats_by_m:
  1:
    global_stats_dict
    global_center_means
    dataset_balanced_global_cov
  4:
    global_stats_dict
    global_center_means
    dataset_balanced_global_cov
```

解释：

- `lr_rgda.state_dict` 表示训练入口当时实际使用的 LR-RGDA classifier。
- `rgda_stats_by_m` 表示可供离线 evaluator 重新构建 classifier 的 compact statistics。
- `rgda_stats_by_m` 不包含 real features。
- evaluator 不能从 `M=1` artifact 凭空重建 `M=4`，除非 artifact 里已有 `rgda_stats_by_m[4]`。

---

## 4. Sweep Evaluator 更新

`scripts/evaluate_incremental_rgda_sweep_artifacts.py` 已更新：

- 优先读取 `rgda_stats_by_m[variant.m]`；
- 如果没有 `rgda_stats_by_m`，再使用旧的 primary compact fields；
- 如果是旧 artifact 且仍有 `classifier_features/classifier_labels`，可以 fallback；
- classifier-only fine-tune 使用 compact pseudo-features：

```text
source=gmm_sample
source=gmm_mean
source=center_replay
source=none
```

variant 示例：

```text
sc:m=1:ft=0
mc4:m=4:ft=0
mc4ft200:m=4:ft=200:lr=0.01:source=gmm_sample
```

---

## 5. Launcher 更新

`scripts/run_text_only_rgda_sweep.sh` 默认改为：

```text
NUM_CENTERS=4
ARTIFACT_NUM_CENTERS=1,4
VARIANTS=sc:m=1:ft=0,mc4:m=4:ft=0,mc4ft200:m=4:ft=200:lr=0.01:source=gmm_sample
```

这表示：

- 训练主 LR-RGDA 用 `M=4`；
- artifact 同时保存 `M=1` 和 `M=4` compact stats；
- sweep 在同一个训练产物上比较 single-center 和 multi-center；
- `mc4ft200` 的 fine-tune 数据来源为 GMM pseudo-features。

---

## 6. 验证

本地验证通过：

```bash
python -m py_compile main_incremental.py scripts/evaluate_incremental_rgda_sweep_artifacts.py
bash -n scripts/run_text_only_rgda_sweep.sh
```

dry-run 验证命令展开：

```text
train:
  --num_centers 4
  --artifact_num_centers 1,4

sweep:
  sc:m=1:ft=0
  mc4:m=4:ft=0
  mc4ft200:m=4:ft=200:lr=0.01:source=gmm_sample
```

远程 2-task smoke：

```text
OUT_DIR=experiments/strict_multi_m_artifact_smoke_20260618
TASKS=aircraft caltech101
ITERATIONS=2
GPU=5
ALPHAS=0,0.05
NUM_CENTERS=4
ARTIFACT_NUM_CENTERS=1,4
```

结果：

- training 阶段完成；
- sweep evaluator 完成；
- 生成 `zero_shot`、`rgda_sc`、`rgda_mc4`、`rgda_mc4ft200` 以及对应 ensemble JSON；
- `step_02_caltech101/artifact.pt` 中：

```text
rgda_stats_by_m keys: 1, 4
M=1 classes=200 center_means=None cov_shape=(512, 512)
M=4 classes=200 center_means=present cov_shape=(512, 512)
gmm_memory classes=200
```

smoke 指标仅用于确认流程，不用于科学结论：

| method | Transfer | Average | Last |
|---|---:|---:|---:|
| zero_shot | 81.50 | 52.33 | 52.39 |
| rgda_sc | 0.00 | 42.32 | 65.08 |
| ens_sc_a0p05 | 82.07 | 55.58 | 56.18 |
| rgda_mc4 | 0.00 | 35.54 | 56.43 |
| ens_mc4_a0p05 | 81.87 | 55.35 | 55.98 |
| rgda_mc4ft200 | 0.00 | 43.57 | 67.09 |
| ens_mc4ft200_a0p05 | 84.02 | 63.87 | 66.27 |

---

## 7. 当前建议

后续 text-only / LADA-hybrid + LR-RGDA 诊断实验应优先使用：

```text
--num_centers 4
--artifact_num_centers 1,4
--rgda_fit_source gmm_sample
```

这样可以在单次训练后离线比较：

- zero-shot；
- single-center LR-RGDA；
- multi-center LR-RGDA；
- multi-center + GMM pseudo-feature classifier fine-tune；
- 多个 alpha 的 ensemble。
