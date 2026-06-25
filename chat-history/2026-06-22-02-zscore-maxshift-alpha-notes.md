# 2026-06-22 - zscore/maxshift Alpha Notes

## Context

After reproducing the old 82.7 result, the ensemble default was changed back to:

```text
--ensemble_normalize maxshift
```

This matches the historical evaluation definition used by:

```text
experiments/visual_tuning_anchor_sweep_20260618_0112
nsp_fd_cd_vision_anchor_seed42 / ens_mc4ft200_a0p05
```

## Existing Full-Test Results

Historical maxshift full-test sweep for `mc4ft200`:

| method | Transfer | Average | Last |
|---|---:|---:|---:|
| ens_mc4ft200_a0p00 | 59.85 | 68.54 | 77.72 |
| ens_mc4ft200_a0p05 | 60.42 | 70.38 | 82.79 |
| ens_mc4ft200_a0p10 | 60.45 | 70.18 | 82.27 |
| ens_mc4ft200_a0p20 | 60.47 | 69.98 | 81.76 |
| ens_mc4ft200_a0p50 | 60.48 | 69.80 | 81.25 |

Reproduced maxshift full-test:

```text
ens_mc4ft200_a0p05:
Transfer 60.40 / Average 70.36 / Last 82.63
```

Current zscore full-test re-evaluation only covered `alpha=0.05`:

```text
ens_mc4ft200_a0p05:
Transfer 59.34 / Average 69.54 / Last 79.13
```

Therefore, on full-test we have not yet swept zscore alpha values.

## Existing 32-Shot zscore Alpha Sweep

For the old `nsp_fd_cd_vision_anchor_seed42` artifacts under 32-shot-per-class
evaluation:

| method | Transfer | Average | Last |
|---|---:|---:|---:|
| ens_mc4ft200_a0p05 | 60.42 | 70.45 | 79.86 |
| ens_mc4ft200_a0p10 | 59.49 | 70.64 | 80.85 |
| ens_mc4ft200_a0p20 | 55.44 | 68.92 | 82.13 |
| ens_mc4ft200_a0p30 | 46.68 | 64.20 | 82.73 |
| ens_mc4ft200_a0p50 | 23.09 | 52.15 | 82.72 |

Observation:

- Under zscore, `alpha=0.10` is a better balanced point than `alpha=0.05`
  on 32-shot eval.
- Larger zscore alpha values can improve Last but quickly damage Transfer and
  Average.

## Interpretation

The zscore alpha behavior does not directly transfer to maxshift.

For maxshift full-test, the historical sweep already shows that:

```text
alpha=0.05 is better than alpha=0.10 for Average and Last.
```

For zscore, `alpha=0.05` appears too conservative in 32-shot eval, and
`alpha=0.10` is worth testing on full-test if we want to keep exploring zscore.

## Current Working Defaults

The code default is now:

```text
--ensemble_normalize maxshift
```

Use this for historical 82.7-style comparisons.

To explicitly test the post-06-19 zscore fusion, pass:

```text
--ensemble_normalize zscore
```

Do not compare maxshift and zscore results without stating the fusion mode.
