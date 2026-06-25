# 2026-06-22 - Ensemble Default Changed Back to maxshift

## Context

The previous best result:

```text
nsp_fd_cd_vision_anchor_seed42 / ens_mc4ft200_a0p05
Transfer 60.42 / Average 70.38 / Last 82.79
```

was reproduced only when using the old ensemble fusion mode:

```text
--ensemble_normalize maxshift
```

The current `zscore` default produced a substantially lower full-test result:

```text
zscore:   Transfer 59.34 / Average 69.54 / Last 79.13
maxshift: Transfer 60.40 / Average 70.36 / Last 82.63
```

## Change

The default ensemble normalization was changed back from `zscore` to
`maxshift` in:

```text
main_incremental.py
scripts/evaluate_incremental_rgda_sweep_artifacts.py
src/utils/main_utils.py
```

`zscore` remains available explicitly:

```text
--ensemble_normalize zscore
```

## Practical Implication

Historical 82.7-style comparisons now use the matching default again. New
experiments that intend to use the post-06-19 z-score fusion must pass
`--ensemble_normalize zscore` explicitly.
