#!/bin/bash
set -euo pipefail

# Exclusive lock to prevent concurrent invocations
LOCK_FILE=/tmp/10task_reablation_master.lock
exec 200>"$LOCK_FILE"
if ! flock -n 200; then
    echo "Another instance of 10-task reablation launcher is already running. Exiting."
    exit 1
fi

PROJECT=/home/raoxuan/projects/project_clip_continual_learning
PYTHON=/home/raoxuan/ENTER/envs/raoxuan/bin/python
LOGDIR="$PROJECT/logs/10task_reablation"
OUTDIR="$PROJECT/experiments/10task_reablation"
mkdir -p "$LOGDIR" "$OUTDIR"

DATASET_SEQUENCE="aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397"

COMMON="--root /data1/open_datasets/X-TAIL --dataset_sequence $DATASET_SEQUENCE --num_shots 16 --batch_size 64 --iterations 800 --scheduler cosine --lora_type lora_nsp --lora_rank 4 --lora_target_modules q_proj,k_proj,v_proj,out_proj,fc1,fc2 --projection_param_mode full --null_init_mode none --optimizer adamw --weight_decay 3e-5 --cd_weight 2.0 --fd_weight 1.0 --aux_weight 0.0 --cd_divergence kl_forward --cd_temperature 2.0 --seed 43 --eval_batch_size 128 --num_workers 2 --output_dir $OUTDIR"

is_running() {
    local name=$1
    # Look for actual python processes with this experiment_name, excluding pgrep/grep/bash
    pgrep -af "/home/raoxuan/ENTER/envs/raoxuan/bin/python -u $PROJECT/main_incremental.py" 2>/dev/null | grep -q "experiment_name $name" || return 1
}

launch_one() {
    local gpu=$1
    local name=$2
    if [ -f "$OUTDIR/${name}_ens_results.json" ] && [ -f "$OUTDIR/${name}_zs_results.json" ]; then
        echo "[skip] $name already has results"
        return 0
    fi
    if is_running "$name"; then
        echo "[skip] $name is already running"
        return 0
    fi

    local extra=""
    case $name in
        dora_anchor_repro) extra="--use_dora true --lr 1e-4" ;;
        dora_lr5e5)        extra="--use_dora true --lr 5e-5" ;;
        dora_lr3e4)        extra="--use_dora true --lr 3e-4" ;;
        lora_lr5e5)        extra="--use_dora false --lr 5e-5" ;;
        lora_lr1e4)        extra="--use_dora false --lr 1e-4" ;;
        lora_lr3e4)        extra="--use_dora false --lr 3e-4" ;;
        winner_adamw1e4)   extra="--use_dora true --lr 1e-4 --optimizer adamw" ;;
        winner_sgd1e3)     extra="--use_dora true --lr 1e-4 --optimizer sgd" ;;
        winner_bs32)       extra="--use_dora true --lr 1e-4 --batch_size 32" ;;
        winner_bs64)       extra="--use_dora true --lr 1e-4 --batch_size 64" ;;
        winner_bs128)      extra="--use_dora true --lr 1e-4 --batch_size 128" ;;
        winner_temp1p0)    extra="--use_dora true --lr 1e-4 --cd_temperature 1.0" ;;
        winner_temp2p0)    extra="--use_dora true --lr 1e-4 --cd_temperature 2.0" ;;
        winner_temp4p0)    extra="--use_dora true --lr 1e-4 --cd_temperature 4.0" ;;
        winner_aux0)       extra="--use_dora true --lr 1e-4 --aux_weight 0.0" ;;
        winner_aux1)       extra="--use_dora true --lr 1e-4 --aux_weight 1.0" ;;
        winner_text_freeze) extra="--use_dora true --lr 1e-4 --text_tuning_schedule freeze_after" ;;
        winner_text_lowlr)  extra="--use_dora true --lr 1e-4 --text_tuning_schedule low_lr_after" ;;
        winner_text_always) extra="--use_dora true --lr 1e-4 --text_tuning_schedule always" ;;
        winner_text_never)  extra="--use_dora true --lr 1e-4 --text_tuning_schedule never" ;;
        *) echo "Unknown experiment $name"; return 1 ;;
    esac

    echo "[launch] $name on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu nohup "$PYTHON" -u "$PROJECT/main_incremental.py" $COMMON $extra --experiment_name $name > "$LOGDIR/${name}.log" 2>&1 &
}

# Run a queue of experiment names sequentially on a given GPU
run_queue() {
    local gpu=$1
    shift
    for name in "$@"; do
        launch_one "$gpu" "$name"
    done
    wait
    echo "GPU $gpu queue done"
}

# 5 free GPUs (0-4); GPU 5 reserved for vLLM. Distribute 20 experiments.
run_queue 0 dora_anchor_repro winner_bs32 winner_temp1p0 winner_aux0 &
run_queue 1 dora_lr5e5 winner_bs64 winner_temp2p0 winner_aux1 &
run_queue 2 dora_lr3e4 winner_bs128 winner_temp4p0 winner_text_freeze &
run_queue 3 lora_lr5e5 winner_adamw1e4 winner_text_lowlr winner_text_always &
run_queue 4 lora_lr1e4 winner_sgd1e3 winner_text_never lora_lr3e4 &

wait
echo "All 10-task reablation queues finished"
