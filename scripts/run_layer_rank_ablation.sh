#!/bin/bash
set -euo pipefail

PROJECT=/home/raoxuan/projects/project_clip_continual_learning
cd "$PROJECT"
PYTHON=/home/raoxuan/ENTER/envs/raoxuan/bin/python
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

OUT_DIR=experiments/layer_rank_ablation
LOG_DIR=$OUT_DIR/logs
mkdir -p "$LOG_DIR"

COMMON="--root /data1/open_datasets/X-TAIL --dataset_sequence aircraft caltech101 dtd eurosat flowers oxford_pets --num_shots 16 --batch_size 128 --lora_type lora_nsp --use_dora false --train_budget_mode uniform --cd_weight 2.0 --aux_weight 0.0 --fd_weight 1.0 --cd_divergence kl_forward --cd_temperature 4.0 --optimizer adamw --lr 1e-4 --weight_decay 3e-5 --iterations 800 --scheduler cosine_with_warmup --projection_param_mode full --null_init_mode none --seed 42 --output_dir $OUT_DIR"

launch() {
  local name=$1
  local gpu=$2
  shift 2
  local log=$LOG_DIR/${name}.log
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching $name on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu nohup $PYTHON main_incremental.py $COMMON "$@" --gpu 0 --experiment_name "$name" > "$log" 2>&1 &
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] PID $! for $name"
}

# Wave 1: Layer selection (rank=4)
launch layer_all_rank4 0 --lora_rank 4 --lora_target_modules q_proj,k_proj,v_proj,out_proj,fc1,fc2
launch layer_qk_only 2 --lora_rank 4 --lora_target_modules q_proj,k_proj
launch layer_ffn_only 3 --lora_rank 4 --lora_target_modules fc1,fc2
launch layer_attn_only 4 --lora_rank 4 --lora_target_modules q_proj,k_proj,v_proj,out_proj

wait
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Wave 1 (layer selection) finished."

# Wave 2: Rank ablation (default modules)
launch rank_1 0 --lora_rank 1 --lora_target_modules q_proj,k_proj,v_proj,out_proj,fc1,fc2
launch rank_2 2 --lora_rank 2 --lora_target_modules q_proj,k_proj,v_proj,out_proj,fc1,fc2
launch rank_4 3 --lora_rank 4 --lora_target_modules q_proj,k_proj,v_proj,out_proj,fc1,fc2
launch rank_8 4 --lora_rank 8 --lora_target_modules q_proj,k_proj,v_proj,out_proj,fc1,fc2

wait
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Wave 2 (rank ablation) finished."
echo "All experiments done. Monitor with: tail -f $LOG_DIR/*.log"
