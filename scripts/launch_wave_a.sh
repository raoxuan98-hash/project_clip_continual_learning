#!/bin/bash
# ============================================================
# Wave A: DoRA vs LoRA × lr sweep
# Plan: chat-history/2026-07-06-10task-reablation-plan.md (v4) §7
#
# Prerequisites: Pre-Wave A completed, winner backbone + text_sched decided.
# The text_tuning_schedule below should match the Pre-A winner (expected: low_lr_after).
# ============================================================
set -euo pipefail

PROJECT=/home/raoxuan/projects/project_clip_continual_learning
PYTHON=/home/raoxuan/ENTER/envs/raoxuan/bin/python
LOGDIR="$PROJECT/artifacts/logs/10task_reablation"
OUTDIR="$PROJECT/experiments/10task_reablation"
mkdir -p "$LOGDIR" "$OUTDIR"

# ---- common arguments (plan v4 §5 public config) ----
# Differs from Pre-Wave A: scheduler=cosine_with_warmup, cd_weight=2.0, seed=43
DATASET_SEQUENCE="aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397"

COMMON="--root /data1/open_datasets/X-TAIL"
COMMON="$COMMON --dataset_sequence $DATASET_SEQUENCE"
COMMON="$COMMON --num_shots 16"
COMMON="$COMMON --batch_size 64"
COMMON="$COMMON --iterations 800"
COMMON="$COMMON --scheduler cosine_with_warmup"
COMMON="$COMMON --lora_type lora_nsp"
COMMON="$COMMON --lora_rank 4"
COMMON="$COMMON --lora_target_modules q_proj,k_proj,v_proj,out_proj,fc1,fc2"
COMMON="$COMMON --projection_param_mode full"
COMMON="$COMMON --null_init_mode none"
COMMON="$COMMON --optimizer adamw"
COMMON="$COMMON --weight_decay 3e-5"
COMMON="$COMMON --cd_weight 2.0"
COMMON="$COMMON --fd_weight 1.0"
COMMON="$COMMON --aux_weight 0.0"
COMMON="$COMMON --cd_divergence kl_forward"
COMMON="$COMMON --cd_temperature 2.0"
COMMON="$COMMON --seed 43"
COMMON="$COMMON --eval_batch_size 128"
COMMON="$COMMON --output_dir $OUTDIR"

# ---- text_tuning_schedule: set to Pre-A winner (expected: low_lr_after) ----
TEXT_SCHED="${TEXT_SCHED:-low_lr_after}"

# ---- cleanup ----
pkill -f main_incremental.py 2>/dev/null || true
sleep 2

launch_one() {
    local name=$1; local use_dora=$2; local lr=$3; local gpu=$4
    local log="$LOGDIR/${name}.log"
    echo "[$(date '+%H:%M:%S')] Launching $name on GPU $gpu (dora=$use_dora, lr=$lr)"
    CUDA_VISIBLE_DEVICES=$gpu nohup "$PYTHON" -u "$PROJECT/main_incremental.py" \
        $COMMON \
        --use_dora $use_dora \
        --lr $lr \
        --text_tuning_schedule $TEXT_SCHED \
        --experiment_name $name \
        > "$log" 2>&1 &
}

# ---- 6 experiments on 4 GPUs ----
# GPU queue: each GPU runs its assigned experiments sequentially
# Round 1: 4 experiments on GPU 0,1,2,4
# Round 2: 2 experiments on GPU 0,1

echo "=== Wave A ==="
echo "text_tuning_schedule = $TEXT_SCHED"

# Round 1
launch_one dora_lr5e5 true  5e-5 0
launch_one dora_lr1e4 true  1e-4 1
launch_one dora_lr3e4 true  3e-4 2
launch_one lora_lr5e5 false 5e-5 4
wait
echo "[$(date '+%H:%M:%S')] Round 1 complete"

# Round 2
launch_one lora_lr1e4 false 1e-4 0
launch_one lora_lr3e4 false 3e-4 1
wait
echo "[$(date '+%H:%M:%S')] Round 2 complete"

# ---- verify ----
echo ""
echo "=== Verification ==="
ALL_OK=true
for name in dora_lr5e5 dora_lr1e4 dora_lr3e4 lora_lr5e5 lora_lr1e4 lora_lr3e4; do
    zs="$OUTDIR/${name}_zs_results.json"
    ens="$OUTDIR/${name}_ens_results.json"
    if [ -s "$zs" ] && [ -s "$ens" ]; then
        echo "  [OK]    $name"
    else
        echo "  [FAIL]  $name"
        ALL_OK=false
    fi
done
$ALL_OK && echo "Wave A complete." || echo "Some experiments FAILED."
