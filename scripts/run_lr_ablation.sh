#!/usr/bin/env bash
# LoRA + current_nsp + AdamW 学习率消融
# Usage: run_lr_ablation.sh GPU_ID
set -euo pipefail

PROJECT_DIR="/home/raoxuan/projects/project_clip_continual_learning"
DATA_ROOT="/data1/open_datasets/X-TAIL"
OUT_DIR="experiments/optimizer_ablation"
LOG_DIR="${OUT_DIR}/logs_lr_ablation"
PYTHON="/home/raoxuan/ENTER/envs/raoxuan/bin/python"

GPU_ID="${1:?GPU_ID required}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

cd "${PROJECT_DIR}"
mkdir -p "${LOG_DIR}"

VARIANT="current_nsp"
OPT="adamw"
LRS=(5e-5 1e-4 3e-4 6e-4)

for lr in "${LRS[@]}"; do
    # safe experiment name: replace . with p and e with e
    lr_name="${lr/e/e}"
    lr_name="${lr_name/./p}"
    exp_name="${VARIANT}_${OPT}_lr${lr_name}_seed42"
    echo "[GPU${GPU_ID}] Starting ${exp_name} at $(date -Iseconds)"
    ${PYTHON} main_incremental.py \
        --root "${DATA_ROOT}" \
        --dataset_sequence aircraft caltech101 dtd eurosat flowers oxford_pets \
        --num_shots 16 \
        --batch_size 64 \
        --iterations 800 \
        --lr "${lr}" \
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

echo "[GPU${GPU_ID}] LR ablation completed at $(date -Iseconds)"
