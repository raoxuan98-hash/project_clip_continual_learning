# Experiment and Evaluation Scripts

`scripts/` is the canonical home for experiment launchers, evaluation workers,
result summarizers, protocol audits, and remote-run helpers. Repository-root
files with matching names are compatibility entry points only; new code should
import or invoke the implementation in this directory.

## Main reusable entry points

- `run_cached_experiment.py`: run classifier experiments from cached features;
- `evaluate_incremental_artifacts.py`: evaluate queued incremental checkpoints;
- `evaluate_incremental_rgda_sweep_artifacts.py`: evaluate LR-RGDA and ensemble sweeps;
- `merge_incremental_async_results.py`: merge asynchronous evaluation results;
- `evaluate_retrieval_artifacts.py`: evaluate cross-modal retrieval artifacts;
- `verify_publication_package.sh`: verify the publication result package.

## Script categories

- `launch_*`, `run_*`, `remote_*`: experiment launch and remote execution;
- `evaluate_*`: model or artifact evaluation;
- `summarize_*`, `generate_*`, `extract_*`, `merge_*`: result processing;
- `audit_*`, `verify_*`, `selftest_*`: protocol and publication checks;
- `monitor_*`, `wait_*`: long-running experiment coordination;
- `plot_*`: visualization generation.

Many launchers encode a specific historical ablation. Do not delete or rename
one until its references in `chat-history/`, experiment logs, and result
artifacts have been audited.

## Subdirectories

- `debug/`: manual diagnostic and exploratory scripts;
- `legacy/`: superseded entry points retained for historical reproducibility.
