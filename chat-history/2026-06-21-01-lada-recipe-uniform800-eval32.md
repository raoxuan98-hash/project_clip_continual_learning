# 2026-06-21 LADA recipe with uniform budget and 32-shot eval

## Goal

Test whether the negative result from the previous combined LADA-recipe run was caused by LADA's per-dataset training budget tuning. This run keeps the common LADA recipe pieces but removes data-specific training budgets.

## Change From Previous Full Run

Previous run:

```text
experiments/lada_recipe_textlow_full_20260619
use_lada_recipe_defaults = true
train_budget_mode = lada_epochs
eval_shots_per_class = 0
```

New run:

```text
experiments/lada_recipe_uniform800_eval32_20260621
train_budget_mode = uniform
iterations = 800 per task
eval_shots_per_class = 32
```

Important: `--use_lada_recipe_defaults` is not used in this run because it forcibly sets `train_budget_mode=lada_epochs`. Instead, the shared LADA recipe settings are expanded explicitly.

## Training Recipe

Run name:

```text
lada_recipe_uniform800_nsp_fdcd_textlow_after2_eval32_seed42
```

Remote repo:

```text
raoxuan@10.20.34.30:/home/raoxuan/projects/project_clip_continual_learning
```

Output directory:

```text
experiments/lada_recipe_uniform800_eval32_20260621
```

Pipeline script:

```text
experiments/lada_recipe_uniform800_eval32_20260621/run_pipeline.sh
```

Launch PID:

```text
503397
```

GPU:

```text
CUDA_VISIBLE_DEVICES=1
```

Dataset sequence:

```text
aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397
```

Training settings:

```text
num_shots = 16
batch_size = 64
lr = 1e-3
weight_decay = 5e-4
scheduler = onecycle
train_budget_mode = uniform
iterations = 800
```

Model/tuning settings:

```text
lora_type = lora_nsp
init_mode = lora_nsp
fd_weight = 1.0
cd_weight = 1.0
tune_vision_encoder = true
tune_text_encoder = true
text_classifier_mode = lada_hybrid
text_tuning_schedule = low_lr_after
text_schedule_switch_task = 2
text_lr_scale_after_task = 0.2
```

Text LR semantics:

```text
tasks 1-2: text_lr = 1e-3
tasks 3-10: text_lr = 2e-4
```

Artifact settings:

```text
save_step_artifacts = true
skip_inline_eval = true
num_centers = 4
artifact_num_centers = 1,4
```

## Evaluation

Post-hoc classifier sweep only; retrieval is not run in this experiment because the requested change is classification evaluation on 32-shot test subsets.

Sweep script:

```text
scripts/evaluate_incremental_rgda_sweep_artifacts.py
```

Classifier variants:

```text
sc:m=1:ft=0
mc4:m=4:ft=0
mc4ft200:m=4:ft=200:lr=0.01:source=gmm_sample
```

Alpha sweep:

```text
0, 0.05, 0.1, 0.2, 0.5, 1.0
```

Evaluation subset:

```text
eval_shots_per_class = 32
eval_subset_seed = 0
```

## Initial Verification

The first status check confirmed that the process was alive and the training log printed:

```text
Task 1 training budget: 800 iterations (uniform)
Task train schedule: vision_lora=True, text_adapter=True, train_text=True, text_lr=0.001
```

This confirms that the data-specific LADA epoch schedule was removed while the shared recipe remained active.

There is a cosmetic date-format warning in the logs, similar to the previous run:

```text
date: extra operand '%T'
```

It does not stop training.
