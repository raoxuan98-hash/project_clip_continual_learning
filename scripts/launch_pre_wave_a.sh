#!/bin/bash
set -e

# ============================================================
# Pre-Wave A: backbone × text_schedule 全交叉 (6 experiments)
# Plan: chat-history/2026-07-06-10task-reablation-plan.md (v4)
# ============================================================

PROJECT=/home/raoxuan/projects/project_clip_continual_learning
PYTHON=/home/raoxuan/ENTER/envs/raoxuan/bin/python
LOGDIR="$PROJECT/artifacts/logs/10task_reablation"
OUTDIR="$PROJECT/experiments/10task_reablation"
mkdir -p "$LOGDIR" "$OUTDIR"

# ---- common arguments (aligned with historical 82.79 config) ----
COMMON="--root /data1/open_datasets/X-TAIL"
COMMON="$COMMON --dataset_sequence aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397"
COMMON="$COMMON --num_shots 16"
COMMON="$COMMON --batch_size 64"
COMMON="$COMMON --iterations 800"
COMMON="$COMMON --scheduler cosine"
COMMON="$COMMON --lora_type lora_nsp"
COMMON="$COMMON --lora_rank 4"
COMMON="$COMMON --lora_target_modules q_proj,k_proj,v_proj,out_proj,fc1,fc2"
COMMON="$COMMON --projection_param_mode full"
COMMON="$COMMON --null_init_mode none"
COMMON="$COMMON --optimizer adamw"
COMMON="$COMMON --lr 1e-4"
COMMON="$COMMON --weight_decay 3e-5"
COMMON="$COMMON --cd_weight 1.0"
COMMON="$COMMON --fd_weight 1.0"
COMMON="$COMMON --aux_weight 0.0"
COMMON="$COMMON --cd_divergence kl_forward"
COMMON="$COMMON --cd_temperature 2.0"
COMMON="$COMMON --seed 42"
COMMON="$COMMON --eval_batch_size 128"
COMMON="$COMMON --output_dir $OUTDIR"

# ---- cleanup ----
pkill -f main_incremental.py 2>/dev/null || true
sleep 2

# ---- experiment launcher ----
launch_one() {
    local name=$1
    local use_dora=$2
    local text_sched=$3
    local gpu=$4

    local extra="--use_dora $use_dora --text_tuning_schedule $text_sched --experiment_name $name"
    local log="$LOGDIR/${name}.log"

    echo "[$(date '+%H:%M:%S')] Launching $name on GPU $gpu (dora=$use_dora, text=$text_sched)"
    CUDA_VISIBLE_DEVICES=$gpu nohup "$PYTHON" -u "$PROJECT/main_incremental.py" \
        $COMMON $extra \
        > "$log" 2>&1 &
    echo "  PID=$!"
}

# ---- Round 1: 4 experiments on GPU 0,1,2,4 ----
echo "=== Pre-Wave A Round 1 ==="

launch_one "lora_freeze"  false freeze_after  0
launch_one "lora_lowlr"   false low_lr_after  1
launch_one "lora_always"  false always        2
launch_one "dora_freeze"  true  freeze_after  4

echo ""
echo "Waiting for Round 1 to finish..."
wait
echo "[$(date '+%H:%M:%S')] Round 1 complete."

# ---- Round 2: 2 experiments on GPU 0,1 ----
echo ""
echo "=== Pre-Wave A Round 2 ==="

launch_one "dora_lowlr"   true  low_lr_after  0
launch_one "dora_always"  true  always        1

echo ""
echo "Waiting for Round 2 to finish..."
wait
echo "[$(date '+%H:%M:%S')] Round 2 complete."

# ---- verify results ----
echo ""
echo "=== Verification ==="
ALL_OK=true
for name in lora_freeze lora_lowlr lora_always dora_freeze dora_lowlr dora_always; do
    zs_file="$OUTDIR/${name}_zs_results.json"
    ens_file="$OUTDIR/${name}_ens_results.json"
    log_file="$LOGDIR/${name}.log"

    zs_ok=false; ens_ok=false
    [ -s "$zs_file" ] && zs_ok=true
    [ -s "$ens_file" ] && ens_ok=true

    if $zs_ok && $ens_ok; then
        echo "  [OK]    $name"
    else
        echo "  [FAIL]  $name  (zs=$zs_ok, ens=$ens_ok)"
        ALL_OK=false
        # print last 5 lines of log for diagnosis
        echo "         tail: $(tail -n 3 "$log_file" 2>/dev/null | tr '\n' ' ')"
    fi
done

echo ""
if $ALL_OK; then
    echo "All 6 experiments completed successfully."
else
    echo "Some experiments FAILED. Check logs in $LOGDIR"
fi
