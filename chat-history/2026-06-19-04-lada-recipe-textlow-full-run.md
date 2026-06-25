# 2026-06-19 LADA recipe + text-low schedule full run

## Goal

Run one high-value 10-task test instead of broad ablations. This test combines the currently most plausible useful pieces:

- LADA-style training recipe defaults.
- NSP visual tuning with FD/CD.
- Text encoder tuning enabled.
- Text LR reduced after the second task.
- Saved step artifacts with separate post-hoc LR-RGDA/ensemble sweep.
- Retrieval evaluation on English COCO/Flickr30K datasets.

This run should be interpreted as a combined recipe test, not as a causal isolation of any single component.

## Remote run

Remote repo:

```text
raoxuan@10.20.34.30:/home/raoxuan/projects/project_clip_continual_learning
```

Output directory:

```text
experiments/lada_recipe_textlow_full_20260619
```

Run name:

```text
lada_recipe_nsp_fdcd_textlow_after2_seed42
```

Pipeline script:

```text
experiments/lada_recipe_textlow_full_20260619/run_pipeline.sh
```

Launch:

```bash
CUDA_VISIBLE_DEVICES=1 nohup bash experiments/lada_recipe_textlow_full_20260619/run_pipeline.sh \
  > experiments/lada_recipe_textlow_full_20260619/logs/nohup.log 2>&1 &
```

Pipeline PID at launch:

```text
3873906
```

## Training config

Dataset sequence:

```text
aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397
```

Core settings:

```text
seed = 42
num_shots = 16
batch_size = 64
num_centers = 4
artifact_num_centers = 1,4
skip_inline_eval = true
save_step_artifacts = true
```

Backbone/tuning settings:

```text
lora_type = lora_nsp
init_mode = lora_nsp
fd_weight = 1.0
cd_weight = 1.0
tune_vision_encoder = true
tune_text_encoder = true
text_classifier_mode = lada_hybrid
```

LADA recipe defaults:

```text
use_lada_recipe_defaults = true
lr = 1e-3
weight_decay = 5e-4
scheduler = onecycle
train_budget_mode = lada_epochs
```

Text LR schedule:

```text
text_tuning_schedule = low_lr_after
text_schedule_switch_task = 2
text_lr_scale_after_task = 0.2
```

Semantics:

```text
tasks 1-2: text_lr = 1e-3
tasks 3-10: text_lr = 2e-4
```

At the first status check, the training log confirmed:

```text
Task 1 training budget: 40 epochs x 25 steps/epoch = 1000 iterations
Task train schedule: vision_lora=True, text_adapter=True, train_text=True, text_lr=0.001
```

## Classifier sweep

Post-hoc sweep after training:

```text
scripts/evaluate_incremental_rgda_sweep_artifacts.py
```

Variants:

```text
sc:m=1:ft=0
mc4:m=4:ft=0
mc4ft200:m=4:ft=200:lr=0.01:source=gmm_sample
```

Alpha sweep:

```text
0, 0.05, 0.1, 0.2, 0.5, 1.0
```

Other settings:

```text
eval_shots_per_class = 0
rgda_eval_chunk_size = 512
```

## Retrieval evaluation

Post-hoc retrieval evaluation after classifier sweep:

```text
scripts/evaluate_retrieval_artifacts.py
```

Datasets:

```text
mscoco_2014_5k: /mnt/raoxuan/open_datasets/mscoco_2014_5k_test_hf
flickr30k_hf:   /mnt/raoxuan/open_datasets/flickr30k_hf
```

Retrieval settings:

```text
include_frozen_baseline = true
batch_size = 128
text_batch_size = 256
num_workers = 4
```

Important interpretation: retrieval evaluates the saved encoder state. It does not evaluate cached class anchors or the LR-RGDA classifier.

## Comparison target

Prior strongest full-test baseline:

```text
experiments/visual_tuning_anchor_sweep_20260618_0112
nsp_fd_cd_vision_anchor_seed42
best ensemble ens_mc4ft200_a0p05:
Transfer = 60.42
Average  = 70.38
Last     = 82.79
```

## Interpretation rules

Strong positive:

```text
Average >= 70.38 and Last >= 82.79, with Transfer not below 59.0
```

Useful positive:

```text
Average improves or stays close, Last improves, Transfer drop <= 1.5
```

Tradeoff:

```text
Average/Last improve but Transfer drops > 1.5
```

Negative:

```text
Average or Last drop by > 2 points without retrieval benefit
```

Do not report Last as valid unless the step10 artifact exists.

## Failure fallback

If classifier sweep OOMs:

```text
rerun sweep with --rgda_eval_chunk_size 256
```

If retrieval OOMs:

```text
rerun retrieval with --batch_size 64 --text_batch_size 128
```

If training OOMs:

```text
do not silently lower batch_size, because batch_size=64 is part of the LADA recipe being tested
```

## Initial status

The pipeline was alive on GPU1 after launch. The first status check found no artifacts yet and Task 1 at early training. A second check confirmed progress from roughly 39/1000 to 57/1000 iterations, so the process was not stalled.

There is a cosmetic shell quoting bug in the pipeline date logging:

```text
date: extra operand '%T'
```

This only affects printed timestamps in pipeline/nohup logs and did not prevent training from starting.
