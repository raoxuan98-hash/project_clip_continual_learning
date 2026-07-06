#!/bin/bash
# ============================================================
# Post Pre-Wave A: seed variance + Wave A (simplified LoRA × lr)
# Plan: chat-history/2026-07-06-10task-reablation-plan.md (v4)
#
# Pre-Wave A winner: LoRA + always
# ============================================================
set -euo pipefail

PROJECT=/home/raoxuan/projects/project_clip_continual_learning
PYTHON=/home/raoxuan/ENTER/envs/raoxuan/bin/python
LOGDIR="$PROJECT/logs/10task_reablation"
OUTDIR="$PROJECT/experiments/10task_reablation"
mkdir -p "$LOGDIR" "$OUTDIR"

DATASET_SEQUENCE="aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397"

# ---- Public config (updated from Pre-Wave A findings) ----
COMMON="--root /data1/open_datasets/X-TAIL"
COMMON="$COMMON --dataset_sequence $DATASET_SEQUENCE"
COMMON="$COMMON --num_shots 16"
COMMON="$COMMON --batch_size 64"
COMMON="$COMMON --iterations 800"
COMMON="$COMMON --scheduler cosine_with_warmup"
COMMON="$COMMON --lora_type lora_nsp"
COMMON="$COMMON --use_dora false"
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
COMMON="$COMMON --text_tuning_schedule always"
COMMON="$COMMON --num_centers 4"
COMMON="$COMMON --rgda_train_iter 200"
COMMON="$COMMON --rgda_train_lr 0.01"
COMMON="$COMMON --rgda_fit_source gmm_sample"
COMMON="$COMMON --seed 43"
COMMON="$COMMON --eval_batch_size 128"
COMMON="$COMMON --output_dir $OUTDIR"

# ---- cleanup ----
pkill -f main_incremental.py 2>/dev/null || true
sleep 2

launch_one() {
    local name=$1; local gpu=$2; shift 2
    local log="$LOGDIR/${name}.log"
    echo "[$(date '+%H:%M:%S')] Launching $name on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu nohup "$PYTHON" -u "$PROJECT/main_incremental.py" \
        $COMMON --experiment_name $name "$@" > "$log" 2>&1 &
}

echo "=== Seed Variance + Wave A ==="

# ---- 1. Seed variance ----
# lora_always seed=43 (vs Pre-A seed=42) to estimate sigma_seed
launch_one lora_always_s43 0

# ---- 2. Wave A: LoRA × lr ----
launch_one wa_lora_lr5e5 1 --lr 5e-5
launch_one wa_lora_lr1e4 2 --lr 1e-4
launch_one wa_lora_lr3e4 4 --lr 3e-4

wait
echo "[$(date '+%H:%M:%S')] All done."

# ---- verify ----
echo ""
for name in lora_always_s43 wa_lora_lr5e5 wa_lora_lr1e4 wa_lora_lr3e4; do
    zs="$OUTDIR/${name}_zs_results.json"
    ens="$OUTDIR/${name}_ens_results.json"
    if [ -s "$zs" ] && [ -s "$ens" ]; then
        echo "  [OK]    $name"
    else
        echo "  [FAIL]  $name"
    fi
done
