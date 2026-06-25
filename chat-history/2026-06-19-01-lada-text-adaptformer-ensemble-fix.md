# 2026-06-19 LADA text AdaptFormer and ensemble normalization diagnosis

## Context

We tested whether replacing our text-side LoRA with official-LADA-style text
AdaptFormer can remove the suspicious EuroSAT collapse observed in the
`text_only_anchor_seed42` setting.

The remote experiment directory is:

```text
experiments/lada_text_adaptformer_10task_20260618_180541
```

The run used:

```text
10-task X-TAIL, seed=42, num_shots=16
vision frozen
text_adapter_type=lada_adaptformer
text_adapter_dim=16
text_adapter_scale=0.1
text_classifier_mode=lada_hybrid
fd_weight=0, cd_weight=0, aux_weight=0
use_lada_recipe_defaults=true
batch_size=64, lr=1e-3, weight_decay=5e-4
scheduler=onecycle, train_budget_mode=lada_epochs
num_centers=4, artifact_num_centers=1,4
```

Training finished task 10, but saving the step-10 artifact failed with:

```text
RuntimeError: PytorchStreamWriter failed writing file data/...: file write failed
```

Valid artifacts exist only for steps 1-9. Therefore any 10-task `Last` metric
from these artifacts is invalid and appears as 0.00 because the step-10 row is
missing.

## Implementation

Added official-LADA-style text AdaptFormer support:

```text
src/models/lada_text_adapter.py
src/models/clip.py
src/trainers/lora_nsp_trainer.py
main_incremental.py
```

Important behavior:

- `--text_adapter_type lada_adaptformer` uses AdaptFormer on the text encoder.
- In `text_classifier_mode=lada_hybrid`, after each task we cache the tuned
  text features for current-task classes.
- For LADA AdaptFormer, adapters are reset after caching, matching the intended
  LADA behavior that stores per-task text anchors rather than accumulating one
  text tuner forever.

The remote CLIP cache required:

```bash
export CLIP_MODEL_NAME=/mnt/raoxuan/.cache/huggingface/hub/models--openai--clip-vit-base-patch16/snapshots/57c216476eefef5ab752ec549e440a49ae4ae5f3
export CLIP_LOCAL_FILES_ONLY=1
export CLIP_USE_SAFETENSORS=0
```

## EuroSAT diagnostic

A EuroSAT-only 32-shot-per-class diagnostic over valid step-1 to step-9
artifacts showed:

| step | task | zero-shot | LR-RGDA mc4ft200 | old ensemble alpha=0.05 |
|---:|---|---:|---:|---:|
| 1 | aircraft | 35.94 | 0.00 | 35.94 |
| 2 | caltech101 | 35.94 | 0.00 | 35.94 |
| 3 | dtd | 35.94 | 0.00 | 35.94 |
| 4 | eurosat | 12.81 | 85.00 | 12.81 |
| 5 | flowers | 12.81 | 84-85 | 12.81 |
| 6 | food101 | 12.81 | 84-86 | 12.81 |
| 7 | mnist | 15.62 | 84-85 | 15.62 |
| 8 | oxford_pets | 15.62 | 84-85 | 15.62 |
| 9 | stanford_cars | 15.62 | 84-85 | 15.62 |

This means LADA text AdaptFormer avoids the previous extreme 1.14% EuroSAT
collapse, but the zero-shot anchor remains far below official LADA's reported
EuroSAT transfer numbers. LR-RGDA itself is healthy on EuroSAT.

## Ensemble bug

The old ensemble used max-shifted logits:

```python
zs_logits_norm = zs_logits - zs_logits.max(dim=-1, keepdim=True).values
rgda_logits_norm = rgda_logits - rgda_logits.max(dim=-1, keepdim=True).values
ensemble = (1 - alpha) * zs_logits_norm
ensemble[:, :current_num_classes] += alpha * rgda_logits_norm
```

This is structurally flawed when fusing a global zero-shot classifier with an
ID-only LR-RGDA classifier.

Reason:

- max-shift makes each classifier's best logit exactly 0;
- future/unseen zero-shot classes keep their score in the global classifier;
- LR-RGDA is added only to the seen class block;
- therefore, if zero-shot predicts a future class, that future class has score
  0, and every seen class has score at most 0 for any `alpha < 1`;
- LR-RGDA cannot pull an ID sample back from a future-class zero-shot mistake
  unless `alpha=1`.

This exactly explained why EuroSAT ensemble equaled zero-shot for alpha values
0.05 through 0.7, despite LR-RGDA being 84-86%.

## Fix

Added configurable ensemble normalization:

```text
--ensemble_normalize zscore|maxshift|prob|raw
```

Default is now `zscore`.

Touched files:

```text
src/utils/main_utils.py
scripts/evaluate_incremental_rgda_sweep_artifacts.py
main_incremental.py
```

`zscore` means per-sample standardization before fusion:

```python
(logits - logits.mean(dim=-1, keepdim=True)) / logits.std(dim=-1, keepdim=True)
```

`maxshift` is retained for reproducing old numbers. `prob` was tested but is
not suitable as a default because it badly damages transfer on unseen datasets:
RGDA has no future-class outputs, yet probability fusion still injects mass
into seen classes.

## Fast sweep results

Using valid 9 artifacts only, 32-shot-per-class fast eval, z-score fusion:

```text
output:
experiments/lada_text_adaptformer_10task_20260618_180541/sweep/lada_text_adaptformer_10task_seed42_zscore
```

| method | Transfer | Average | Last |
|---|---:|---:|---:|
| zero_shot | 61.69 | 57.53 | 0.00 |
| rgda_mc4ft200 | 0.00 | 37.36 | 0.00 |
| ens_mc4ft200_a0p05 | 61.38 | 59.28 | 0.00 |
| ens_mc4ft200_a0p10 | 60.90 | 61.31 | 0.00 |
| ens_mc4ft200_a0p20 | 58.55 | 63.40 | 0.00 |
| ens_mc4ft200_a0p30 | 51.16 | 60.88 | 0.00 |
| ens_mc4ft200_a0p50 | 22.74 | 47.02 | 0.00 |

Again, `Last=0.00` is not meaningful because the step-10 artifact is missing.

Interpretation:

- The old max-shift ensemble underestimated LR-RGDA contribution.
- With z-score fusion, alpha around 0.1-0.2 is a better range for this setting.
- alpha=0.2 gives the best Average in this fast sweep but sacrifices about
  3.1 Transfer points relative to zero-shot.
- Larger alpha values begin to damage Transfer strongly.

