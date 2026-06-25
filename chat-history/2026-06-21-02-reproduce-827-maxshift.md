# 2026-06-21 - Reproducing the Previous 82.7 Result

## Goal

Reproduce the previous best full-test result:

```text
experiments/visual_tuning_anchor_sweep_20260618_0112
run: nsp_fd_cd_vision_anchor_seed42
method: ens_mc4ft200_a0p05
old result: Transfer 60.42 / Average 70.38 / Last 82.79
```

The run used:

```text
batch_size = 64
iterations = 800 per task
train_budget_mode = uniform
lr = default 1e-4
weight_decay = default 3e-5
scheduler = default cosine
lora_type = lora_nsp
init_mode = lora_nsp
fd_weight = 1.0
cd_weight = 1.0
tune_vision_encoder = true
tune_text_encoder = true
text_classifier_mode = lada_hybrid
```

It did not use the later LADA-style optimizer recipe (`lr=1e-3`,
`weight_decay=5e-4`, `scheduler=onecycle`).

## First Re-Evaluation: Current Default Fusion

Command:

```text
python scripts/evaluate_incremental_rgda_sweep_artifacts.py \
  --async_eval_dir experiments/visual_tuning_anchor_sweep_20260618_0112/async/nsp_fd_cd_vision_anchor_seed42 \
  --output_dir experiments/reproduce_827_fulltest_20260621/nsp_fd_cd_vision_anchor_seed42 \
  --experiment_name reproduce_827_fulltest_nsp_fd_cd_seed42 \
  --device cuda:0 \
  --variants mc4ft200:m=4:ft=200:lr=0.01:source=gmm_sample \
  --alphas 0.05 \
  --rgda_eval_chunk_size 512
```

Because `--ensemble_normalize` was not specified, this used the current default:

```text
ensemble_normalize = zscore
```

Result:

```text
zero_shot:       Transfer 59.85 / Average 68.54 / Last 77.72
rgda_mc4ft200:  Transfer  0.00 / Average 43.98 / Last 81.20
ens_a0p05:      Transfer 59.34 / Average 69.54 / Last 79.13
```

This did not reproduce the old `82.79` Last.

## Diagnosis

The old result JSON did not contain `ensemble_normalize`, while current
results include:

```text
ensemble_normalize = zscore
```

The relevant note is in:

```text
chat-history/2026-06-19-01-lada-text-adaptformer-ensemble-fix.md
```

It records that the evaluator was changed after the old result:

```text
--ensemble_normalize zscore|maxshift|prob|raw
Default is now zscore.
maxshift is retained for reproducing old numbers.
```

Therefore, the old `82.79` result was produced by the earlier `maxshift`
fusion behavior, not by the current `zscore` default.

This is also supported by the result pattern:

- zero-shot reproduced exactly: `59.85 / 68.54 / 77.72`;
- LR-RGDA alone reproduced to small numerical variation;
- only the ensemble result changed strongly.

So the mismatch was not caused by training artifacts, dataset order, or
zero-shot classifier construction. It was caused by changed ensemble fusion.

## Second Re-Evaluation: Old maxshift Fusion

Command:

```text
python scripts/evaluate_incremental_rgda_sweep_artifacts.py \
  --async_eval_dir experiments/visual_tuning_anchor_sweep_20260618_0112/async/nsp_fd_cd_vision_anchor_seed42 \
  --output_dir experiments/reproduce_827_fulltest_20260621/nsp_fd_cd_vision_anchor_seed42_maxshift \
  --experiment_name reproduce_827_fulltest_nsp_fd_cd_seed42_maxshift \
  --device cuda:0 \
  --variants mc4ft200:m=4:ft=200:lr=0.01:source=gmm_sample \
  --alphas 0.05 \
  --ensemble_normalize maxshift \
  --rgda_eval_chunk_size 512
```

Result:

```text
zero_shot:       Transfer 59.85 / Average 68.54 / Last 77.72
rgda_mc4ft200:  Transfer  0.00 / Average 43.96 / Last 81.15
ens_a0p05:      Transfer 60.40 / Average 70.36 / Last 82.63
```

This essentially reproduces the old result:

```text
old:        Transfer 60.42 / Average 70.38 / Last 82.79
reproduced: Transfer 60.40 / Average 70.36 / Last 82.63
delta:      Transfer -0.02 / Average -0.02 / Last -0.16
```

## Remaining Small Difference

The remaining small difference is consistent with stochastic pseudo-feature
sampling in the classifier fine-tuning path:

```text
variant = mc4ft200:m=4:ft=200:lr=0.01:source=gmm_sample
```

In `scripts/evaluate_incremental_rgda_sweep_artifacts.py`, `gmm_sample` uses
`torch.randn(...)` when sampling pseudo-features. The evaluator does not fix a
global random seed at entry, and the old sweep did not record the random state.
Therefore, exact bit-level reproduction of the old `mc4ft200` fine-tuned
classifier is not guaranteed.

The important conclusion is that the previous `82.7` number is reproducible
only under the old `maxshift` ensemble mode. Running the same artifacts with
the current default `zscore` fusion gives a much lower Last accuracy.

## Practical Takeaways

For historical comparisons against the 2026-06-18 `82.79` result, use:

```text
--ensemble_normalize maxshift
```

For current experiments after the 2026-06-19 ensemble fix, use:

```text
--ensemble_normalize zscore
```

Do not compare `maxshift` historical results directly against `zscore` results
without explicitly noting the fusion-mode change.
