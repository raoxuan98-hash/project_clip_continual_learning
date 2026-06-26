#!/bin/bash
export PATH=$PATH:/home/raoxuan/.local/bin
cd /home/raoxuan/projects/project_clip_continual_learning

arbor run "Phase 1: Verify the 4 key experiments at 800-iter/16-shot/seed=42 on aircraft caltech101 dtd eurosat. These are a 2x2 grid: (baseline=current_nsp, best=hist_null_init_only) x (vision_only=FdCd=0, full_recipe=FdCd=1). Use main_incremental.py with --lora_type lora_nsp --classifier_feature_transform test --batch_size 64. Output to experiments/null_basis_ablation/iter800/. Run experiments in parallel on separate GPUs. Phase 2: After verifying, autonomously research to improve continual learning performance. Explore text encoder schedules, distillation weights, LoRA ranks, auxiliary loss configurations, projection/basis setups. Goal: maximize Ensemble Average on B_dev (aircraft caltech101 dtd) and B_test (add eurosat). Data at /data1/open_datasets/X-TAIL." \
  --yes --yes-cwd /home/raoxuan/projects/project_clip_continual_learning \
  --mode auto --no-dashboard-input --no-webui --no-followup \
  --max-cycles 20 --allow-non-base-branch
