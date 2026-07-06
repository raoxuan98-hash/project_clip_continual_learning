#!/bin/bash
# ============================================================
# Waves B–F: optimizer, bs, temperature, aux, text schedule
# Plan: chat-history/2026-07-06-10task-reablation-plan.md (v4) §8–§12
#
# Usage:
#   bash launch_waves_b_f.sh B "<use_dora> <lr> <opt> <text_sched> <bs> <temp> <aux>"
#   bash launch_waves_b_f.sh C "<use_dora> <lr> <opt> <text_sched>"
#   bash launch_waves_b_f.sh D "<use_dora> <lr> <opt> <text_sched> <bs>"
#   bash launch_waves_b_f.sh E "<use_dora> <lr> <opt> <text_sched> <bs> <temp>"
#   bash launch_waves_b_f.sh F "<use_dora> <lr> <opt> <text_sched> <bs> <temp> <aux>"
#
# Each <param> passed as a string of space-separated winner values.
# ============================================================
set -euo pipefail

WAVE=${1:-}
shift || { echo "Usage: $0 <B|C|D|E|F> '<winner_args...>'"; exit 1; }
WINNER_ARGS=(${1:-})

PROJECT=/home/raoxuan/projects/project_clip_continual_learning
PYTHON=/home/raoxuan/ENTER/envs/raoxuan/bin/python
LOGDIR="$PROJECT/logs/10task_reablation"
OUTDIR="$PROJECT/experiments/10task_reablation"
mkdir -p "$LOGDIR" "$OUTDIR"

DATASET_SEQUENCE="aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397"

# Parse winner args (positional, order defined in usage above)
# Convention: use_dora lr optimizer text_sched [bs] [temp] [aux]
W_USE_DORA="${WINNER_ARGS[0]:-true}"
W_LR="${WINNER_ARGS[1]:-1e-4}"
W_OPT="${WINNER_ARGS[2]:-adamw}"
W_TEXT="${WINNER_ARGS[3]:-low_lr_after}"
W_BS="${WINNER_ARGS[4]:-64}"
W_TEMP="${WINNER_ARGS[5]:-2.0}"
W_AUX="${WINNER_ARGS[6]:-0.0}"

echo "=== Wave $WAVE ==="
echo "Winner: use_dora=$W_USE_DORA lr=$W_LR opt=$W_OPT text=$W_TEXT bs=$W_BS temp=$W_TEMP aux=$W_AUX"

# ---- cleanup ----
pkill -f main_incremental.py 2>/dev/null || true
sleep 2

launch_one() {
    local gpu=$1; local name=$2; shift 2
    local log="$LOGDIR/${name}.log"
    # Skip if results already exist
    if [ -s "$OUTDIR/${name}_zs_results.json" ] && [ -s "$OUTDIR/${name}_ens_results.json" ]; then
        echo "[skip] $name (results exist)"
        return 0
    fi
    echo "[$(date '+%H:%M:%S')] Launching $name on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu nohup "$PYTHON" -u "$PROJECT/main_incremental.py" \
        --root /data1/open_datasets/X-TAIL \
        --dataset_sequence $DATASET_SEQUENCE \
        --num_shots 16 \
        --iterations 800 \
        --scheduler cosine_with_warmup \
        --lora_type lora_nsp \
        --lora_rank 4 \
        --lora_target_modules q_proj,k_proj,v_proj,out_proj,fc1,fc2 \
        --projection_param_mode full \
        --null_init_mode none \
        --cd_weight 2.0 --fd_weight 1.0 \
        --cd_divergence kl_forward \
        --seed 43 --eval_batch_size 128 \
        --output_dir "$OUTDIR" \
        --experiment_name "$name" \
        "$@" \
        > "$log" 2>&1 &
}

case $WAVE in
  B)
    # Optimizer comparison: AdamW + RMSprop (lr=1e-4) + SGD (lr=5e-3)
    launch_one 0 winner_adamw1e4  --use_dora $W_USE_DORA --lr 1e-4   --optimizer adamw   --text_tuning_schedule $W_TEXT --batch_size 64 --cd_temperature 2.0 --aux_weight 0.0 --weight_decay 3e-5 --fd_weight 1.0
    launch_one 1 winner_rmsprop1e4 --use_dora $W_USE_DORA --lr 1e-4   --optimizer rmsprop  --text_tuning_schedule $W_TEXT --batch_size 64 --cd_temperature 2.0 --aux_weight 0.0 --weight_decay 3e-5 --fd_weight 1.0
    launch_one 2 winner_sgd5e3     --use_dora $W_USE_DORA --lr 5e-3   --optimizer sgd      --text_tuning_schedule $W_TEXT --batch_size 64 --cd_temperature 2.0 --aux_weight 0.0 --weight_decay 3e-5 --fd_weight 1.0 --momentum 0.9
    wait
    ;;

  C)
    # Batch size: {32, 64, 128}
    launch_one 0 winner_bs32   --use_dora $W_USE_DORA --lr $W_LR --optimizer $W_OPT --text_tuning_schedule $W_TEXT --batch_size 32  --cd_temperature $W_TEMP --aux_weight $W_AUX --weight_decay 3e-5 --fd_weight 1.0
    launch_one 1 winner_bs64   --use_dora $W_USE_DORA --lr $W_LR --optimizer $W_OPT --text_tuning_schedule $W_TEXT --batch_size 64  --cd_temperature $W_TEMP --aux_weight $W_AUX --weight_decay 3e-5 --fd_weight 1.0
    launch_one 2 winner_bs128  --use_dora $W_USE_DORA --lr $W_LR --optimizer $W_OPT --text_tuning_schedule $W_TEXT --batch_size 128 --cd_temperature $W_TEMP --aux_weight $W_AUX --weight_decay 3e-5 --fd_weight 1.0
    wait
    ;;

  D)
    # CD temperature: {1.0, 2.0, 4.0}
    launch_one 0 winner_temp1p0 --use_dora $W_USE_DORA --lr $W_LR --optimizer $W_OPT --text_tuning_schedule $W_TEXT --batch_size $W_BS --cd_temperature 1.0 --aux_weight $W_AUX --weight_decay 3e-5 --fd_weight 1.0
    launch_one 1 winner_temp2p0 --use_dora $W_USE_DORA --lr $W_LR --optimizer $W_OPT --text_tuning_schedule $W_TEXT --batch_size $W_BS --cd_temperature 2.0 --aux_weight $W_AUX --weight_decay 3e-5 --fd_weight 1.0
    launch_one 2 winner_temp4p0 --use_dora $W_USE_DORA --lr $W_LR --optimizer $W_OPT --text_tuning_schedule $W_TEXT --batch_size $W_BS --cd_temperature 4.0 --aux_weight $W_AUX --weight_decay 3e-5 --fd_weight 1.0
    wait
    ;;

  E)
    # aux_weight: {0.0, 1.0}
    launch_one 0 winner_aux0 --use_dora $W_USE_DORA --lr $W_LR --optimizer $W_OPT --text_tuning_schedule $W_TEXT --batch_size $W_BS --cd_temperature $W_TEMP --aux_weight 0.0 --weight_decay 3e-5 --fd_weight 1.0
    launch_one 1 winner_aux1 --use_dora $W_USE_DORA --lr $W_LR --optimizer $W_OPT --text_tuning_schedule $W_TEXT --batch_size $W_BS --cd_temperature $W_TEMP --aux_weight 1.0 --weight_decay 3e-5 --fd_weight 1.0
    wait
    ;;

  F)
    # Text tuning schedule validation: {freeze_after, low_lr_after, always, never}
    launch_one 0 winner_text_freeze --use_dora $W_USE_DORA --lr $W_LR --optimizer $W_OPT --text_tuning_schedule freeze_after --batch_size $W_BS --cd_temperature $W_TEMP --aux_weight $W_AUX --weight_decay 3e-5 --fd_weight 1.0
    launch_one 1 winner_text_lowlr  --use_dora $W_USE_DORA --lr $W_LR --optimizer $W_OPT --text_tuning_schedule low_lr_after --batch_size $W_BS --cd_temperature $W_TEMP --aux_weight $W_AUX --weight_decay 3e-5 --fd_weight 1.0
    launch_one 2 winner_text_always --use_dora $W_USE_DORA --lr $W_LR --optimizer $W_OPT --text_tuning_schedule always       --batch_size $W_BS --cd_temperature $W_TEMP --aux_weight $W_AUX --weight_decay 3e-5 --fd_weight 1.0
    launch_one 4 winner_text_never  --use_dora $W_USE_DORA --lr $W_LR --optimizer $W_OPT --text_tuning_schedule never        --batch_size $W_BS --cd_temperature $W_TEMP --aux_weight $W_AUX --weight_decay 3e-5 --fd_weight 1.0
    wait
    ;;

  *)
    echo "Unknown wave: $WAVE (expected B|C|D|E|F)"
    exit 1
    ;;
esac

# ---- verify ----
echo ""
echo "=== Verification ==="
for exp_name in $(jobs -l | grep -oP 'experiment_name \K\S+' | sort -u); do
    :  # no-op, we check results via files below
done
echo "Check $LOGDIR for logs and $OUTDIR for results."
echo "Wave $WAVE done."
