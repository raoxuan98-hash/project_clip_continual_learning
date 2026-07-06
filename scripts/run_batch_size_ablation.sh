#!/usr/bin/env bash
# LoRA + current_nsp + AdamW batch_size 消融（reference_batch_size 同步）
# Usage: run_batch_size_ablation.sh GPU_ID
set -euo pipefail

PROJECT_DIR="/home/raoxuan/projects/project_clip_continual_learning"
DATA_ROOT="/data1/open_datasets/X-TAIL"
OUT_DIR="experiments/optimizer_ablation"
LOG_DIR="${OUT_DIR}/logs_batch_size_ablation"
PYTHON="/home/raoxuan/ENTER/envs/raoxuan/bin/python"

GPU_ID="${1:?GPU_ID required}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

cd "${PROJECT_DIR}"
mkdir -p "${LOG_DIR}"

# 等待 SGD 5e-3 实验结束，避免 GPU 冲突
while pgrep -af main_incremental.py | grep -q "sgd_lr5e-3"; do
    echo "[GPU${GPU_ID}] Waiting for sgd_lr5e-3 experiments to finish..."
    sleep 60
done

VARIANT="current_nsp"
OPT="adamw"
LR="1e-4"
BSS=(32 64 128)

for bs in "${BSS[@]}"; do
    exp_name="${VARIANT}_${OPT}_bs${bs}_seed42"
    ref_bs="${bs}"
    echo "[GPU${GPU_ID}] Starting ${exp_name} (ref_bs=${ref_bs}) at $(date -Iseconds)"
    ${PYTHON} main_incremental.py \
        --root "${DATA_ROOT}" \
        --dataset_sequence aircraft caltech101 dtd eurosat flowers oxford_pets \
        --num_shots 16 \
        --batch_size "${bs}" \
        --reference_batch_size "${ref_bs}" \
        --iterations 800 \
        --lr "${LR}" \
        --weight_decay 3e-5 \
        --optimizer "${OPT}" \
        --lora_type lora_nsp \
        --lora_rank 4 \
        --use_dora false \
        --seed 42 \
        --gpu 0 \
        --output_dir "${OUT_DIR}" \
        --experiment_name "${exp_name}" \
        --train_budget_mode uniform \
        --projection_param_mode full \
        --null_init_mode none \
        > "${LOG_DIR}/${exp_name}.log" 2>&1
    echo "[GPU${GPU_ID}] Finished ${exp_name} at $(date -Iseconds)"
done

echo "[GPU${GPU_ID}] Batch size ablation completed at $(date -Iseconds)"
