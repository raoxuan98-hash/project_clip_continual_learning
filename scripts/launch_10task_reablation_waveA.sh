#!/bin/bash
# Wave A: DoRA vs LoRA x lr (excluding dora_lr1e4 which is Pre-Wave A)

set -e

PROJECT=/home/raoxuan/projects/project_clip_continual_learning
PYTHON=/home/raoxuan/ENTER/envs/raoxuan/bin/python
LOGDIR="$PROJECT/artifacts/logs/10task_reablation"
OUTDIR="$PROJECT/experiments/10task_reablation"
mkdir -p "$LOGDIR" "$OUTDIR"

DATASET_SEQUENCE="aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397"

COMMON="--root /data1/open_datasets/X-TAIL --dataset_sequence $DATASET_SEQUENCE --num_shots 16 --batch_size 64 --iterations 800 --scheduler cosine --lora_type lora_nsp --lora_rank 4 --lora_target_modules q_proj,k_proj,v_proj,out_proj,fc1,fc2 --projection_param_mode full --null_init_mode none --optimizer adamw --weight_decay 3e-5 --cd_weight 2.0 --fd_weight 1.0 --aux_weight 0.0 --cd_divergence kl_forward --cd_temperature 2.0 --seed 43 --eval_batch_size 128 --output_dir "$OUTDIR""

launch_bg() {
    local gpu=$1
    local name=$2
    local extra=$3
    echo "Launching $name on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu nohup "$PYTHON" -u "$PROJECT/main_incremental.py" $COMMON $extra --experiment_name $name > "$LOGDIR/${name}.log" 2>&1 &
}

# Pair 1 on GPU 4
launch_bg 4 dora_lr5e5 "--use_dora true --lr 5e-5"
launch_bg 4 lora_lr5e5 "--use_dora false --lr 5e-5"
wait

# Pair 2 on GPU 4 (dora_lr1e4 is Pre-Wave A, skip)
launch_bg 4 lora_lr1e4 "--use_dora false --lr 1e-4"
launch_bg 4 dora_lr3e4 "--use_dora true --lr 3e-4"
wait

# Last one on GPU 4
launch_bg 4 lora_lr3e4 "--use_dora false --lr 3e-4"
wait

echo "Wave A all done"
