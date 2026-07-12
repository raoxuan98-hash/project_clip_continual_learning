#!/bin/bash
set -e

PROJECT=/home/raoxuan/projects/project_clip_continual_learning
PYTHON=/home/raoxuan/ENTER/envs/raoxuan/bin/python
LOGDIR="$PROJECT/artifacts/logs/10task_reablation"
OUTDIR="$PROJECT/experiments/10task_reablation"
mkdir -p "$LOGDIR" "$OUTDIR"

DATASET_SEQUENCE="aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397"
NAME=dora_anchor_repro

CUDA_VISIBLE_DEVICES=4 nohup "$PYTHON" -u "$PROJECT/main_incremental.py"   --root /data1/open_datasets/X-TAIL   --dataset_sequence $DATASET_SEQUENCE   --num_shots 16   --batch_size 64   --iterations 800   --scheduler cosine   --lora_type lora_nsp   --use_dora true   --lora_rank 4   --lora_target_modules q_proj,k_proj,v_proj,out_proj,fc1,fc2   --projection_param_mode full   --null_init_mode none   --optimizer adamw   --lr 1e-4   --weight_decay 3e-5   --cd_weight 2.0   --fd_weight 1.0   --aux_weight 0.0   --cd_divergence kl_forward   --cd_temperature 2.0   --seed 43   --eval_batch_size 128   --output_dir "$OUTDIR"   --experiment_name $NAME   > "$LOGDIR/${NAME}.log" 2>&1 &

echo "Launched $NAME on GPU 4, PID=$!"
