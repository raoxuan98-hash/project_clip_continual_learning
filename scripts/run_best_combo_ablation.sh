#!/bin/bash
set -euo pipefail

PROJECT=/home/raoxuan/projects/project_clip_continual_learning
cd "$PROJECT"
PYTHON=/home/raoxuan/ENTER/envs/raoxuan/bin/python
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

OUT_DIR=experiments/best_combo_ablation
LOG_DIR=$OUT_DIR/logs
mkdir -p "$LOG_DIR"

COMMON="--root /data1/open_datasets/X-TAIL --dataset_sequence aircraft caltech101 dtd eurosat flowers oxford_pets --num_shots 16 --batch_size 128 --lora_type lora_nsp --lora_rank 4 --use_dora false --train_budget_mode uniform --cd_weight 2.0 --aux_weight 0.0 --fd_weight 1.0 --cd_divergence kl_forward --cd_temperature 4.0 --weight_decay 3e-5 --seed 42 --output_dir $OUT_DIR"

launch() {
  local name=$1
  local gpu=$2
  shift 2
  local log=$LOG_DIR/${name}.log
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching $name on physical GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu nohup $PYTHON main_incremental.py $COMMON "$@" --gpu 0 --experiment_name "$name" > "$log" 2>&1 &
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] PID $! for $name"
}

# Combo 1: Best Balanced (current_nsp, AdamW, lr=1e-4, iter=800, cosine_with_warmup)
launch combo_balanced_adamw_1e-4_iter800_cwu 0   --projection_param_mode full --null_init_mode none   --optimizer adamw --lr 1e-4 --iterations 800 --scheduler cosine_with_warmup

# Combo 2: Last Max (current_nsp, AdamW, lr=3e-4, iter=1600, linear)
launch combo_last_adamw_3e-4_iter1600_linear 2   --projection_param_mode full --null_init_mode none   --optimizer adamw --lr 3e-4 --iterations 1600 --scheduler linear

# Combo 3: Transfer Max (hist_null_init_runtime, AdamW, lr=1e-4, iter=800, cosine)
launch combo_transfer_hist_runtime_adamw_1e-4_iter800_cosine 3   --projection_param_mode full --null_init_mode history_init_runtime   --optimizer adamw --lr 1e-4 --iterations 800 --scheduler cosine

# Combo 4: Init-Only (hist_null_init_only, AdamW, lr=1e-4, iter=800, cosine_with_warmup)
launch combo_init_only_hist_only_adamw_1e-4_iter800_cwu 4   --projection_param_mode full --null_init_mode history_init_only   --optimizer adamw --lr 1e-4 --iterations 800 --scheduler cosine_with_warmup

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All 4 combo experiments launched."
echo "Monitor with: tail -f $LOG_DIR/*.log"
