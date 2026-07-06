#!/bin/bash
# Waves B-F: launched individually after Wave A completes
# Usage: bash launch_10task_reablation_remaining.sh <wave> <winner_args>

set -e

PROJECT=/home/raoxuan/projects/project_clip_continual_learning
PYTHON=/home/raoxuan/ENTER/envs/raoxuan/bin/python
LOGDIR="$PROJECT/logs/10task_reablation"
OUTDIR="$PROJECT/experiments/10task_reablation"
mkdir -p "$LOGDIR" "$OUTDIR"

DATASET_SEQUENCE="aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397"

launch_bg() {
    local gpu=$1
    local name=$2
    local extra=$3
    echo "Launching $name on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu nohup "$PYTHON" -u "$PROJECT/main_incremental.py"       --root /data1/open_datasets/X-TAIL --dataset_sequence $DATASET_SEQUENCE --num_shots 16       --lora_type lora_nsp --lora_rank 4 --lora_target_modules q_proj,k_proj,v_proj,out_proj,fc1,fc2       --projection_param_mode full --null_init_mode none --seed 43 --eval_batch_size 128       --output_dir "$OUTDIR" $extra --experiment_name $name > "$LOGDIR/${name}.log" 2>&1 &
}

WAVE=$1
WINNER=${2:-""}

case $WAVE in
  B)
    launch_bg 4 winner_adamw1e4 "$WINNER --optimizer adamw --lr 1e-4 --batch_size 64 --iterations 800 --scheduler cosine --cd_temperature 2.0 --aux_weight 0.0"
    launch_bg 4 winner_sgd1e3 "$WINNER --optimizer sgd --lr 1e-3 --batch_size 64 --iterations 800 --scheduler cosine --cd_temperature 2.0 --aux_weight 0.0"
    ;;
  C)
    launch_bg 4 winner_bs32 "$WINNER --batch_size 32"
    launch_bg 4 winner_bs64 "$WINNER --batch_size 64"
    launch_bg 4 winner_bs128 "$WINNER --batch_size 128"
    ;;
  D)
    launch_bg 4 winner_temp1p0 "$WINNER --cd_temperature 1.0"
    launch_bg 4 winner_temp2p0 "$WINNER --cd_temperature 2.0"
    launch_bg 4 winner_temp4p0 "$WINNER --cd_temperature 4.0"
    ;;
  E)
    launch_bg 4 winner_aux0 "$WINNER --aux_weight 0.0"
    launch_bg 4 winner_aux1 "$WINNER --aux_weight 1.0"
    ;;
  F)
    launch_bg 4 winner_text_freeze "$WINNER --text_tuning_schedule freeze_after"
    launch_bg 4 winner_text_lowlr "$WINNER --text_tuning_schedule low_lr_after"
    launch_bg 4 winner_text_always "$WINNER --text_tuning_schedule always"
    launch_bg 4 winner_text_never "$WINNER --text_tuning_schedule never"
    ;;
  *)
    echo "Unknown wave: $WAVE"
    exit 1
    ;;
esac

wait
echo "Wave $WAVE all done"
