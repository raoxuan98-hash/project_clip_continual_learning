#!/bin/bash
set -euo pipefail

PROJECT=/home/raoxuan/projects/project_clip_continual_learning
cd "$PROJECT"
PYTHON=/home/raoxuan/ENTER/envs/raoxuan/bin/python
GPU0=0
GPU2=2
GPU3=3
GPU4=4

COMMON_ARGS="--root /data1/open_datasets/X-TAIL --dataset_sequence aircraft caltech101 dtd eurosat flowers oxford_pets --num_shots 16 --batch_size 64 --iterations 800 --lr 1e-4 --weight_decay 3e-5 --optimizer adamw --lora_type lora_nsp --lora_rank 4 --use_dora false --train_budget_mode uniform --projection_param_mode full --null_init_mode none --cd_weight 2.0 --aux_weight 0.0 --fd_weight 1.0 --seed 42 --output_dir experiments/optimizer_ablation"

LOG_DIR=experiments/optimizer_ablation/logs_cd_divergence_ablation
mkdir -p "$LOG_DIR"

launch() {
  local div=$1
  local temp=$2
  local gpu=$3
  local name=current_nsp_adamw_cd_divergence_${div}_temp${temp}_seed42
  local log=$LOG_DIR/${name}.log
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launch $name on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu nohup $PYTHON main_incremental.py $COMMON_ARGS --cd_divergence "$div" --cd_temperature "$temp" --gpu 0 --experiment_name "$name" > "$log" 2>&1 &
}

# Wave 1: 4 forms on 4 GPUs
launch kl_forward 2.0 $GPU0
launch kl_reverse 2.0 $GPU2
launch js 2.0 $GPU3
launch mse 2.0 $GPU4
wait
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Wave 1 finished"

# Wave 2: remaining 2 forms on first two GPUs
launch cosine 2.0 $GPU0
launch l1 2.0 $GPU2
wait
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Wave 2 finished"
echo "All cd_divergence ablation jobs completed"
