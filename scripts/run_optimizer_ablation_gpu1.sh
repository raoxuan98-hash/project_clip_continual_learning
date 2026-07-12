#!/usr/bin/env bash
# Optimizer ablation on GPU 1: hist_null_init_runtime with 5 optimizers.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/raoxuan/projects/project_clip_continual_learning}"
DATA_ROOT="${DATA_ROOT:-/data1/open_datasets/X-TAIL}"
OUT_DIR="${OUT_DIR:-experiments/optimizer_ablation}"
LOG_DIR="${LOG_DIR:-${OUT_DIR}/logs}"
PYTHON="${PYTHON:-/home/raoxuan/ENTER/envs/raoxuan/bin/python}"

SEED="${SEED:-42}"
ITERATIONS="${ITERATIONS:-800}"
BATCH_SIZE="${BATCH_SIZE:-64}"
WD="${WD:-3e-5}"
LORA_RANK="${LORA_RANK:-4}"
GPU_ID="${GPU_ID:-1}"
USE_DORA="${USE_DORA:-false}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

cd "${PROJECT_DIR}"
mkdir -p "${LOG_DIR}"

declare -A OPT_LR=(
    [adamw]=1e-4
    [adam]=1e-4
    [sgd]=1e-3
    [adagrad]=1e-3
    [rmsprop]=1e-4
)

VARIANT="hist_null_init_runtime"

for opt in adamw adam sgd adagrad rmsprop; do
    lr="${OPT_LR[$opt]}"
    exp_name="${VARIANT}_${opt}_seed${SEED}"
    echo "[GPU${GPU_ID}] Starting ${exp_name} (lr=${lr}, optimizer=${opt}) at $(date -Iseconds)"
    ${PYTHON} main_incremental.py \
        --root "${DATA_ROOT}" \
        --dataset_sequence aircraft caltech101 dtd eurosat flowers oxford_pets \
        --num_shots 16 \
        --batch_size "${BATCH_SIZE}" \
        --iterations "${ITERATIONS}" \
        --lr "${lr}" \
        --weight_decay "${WD}" \
        --optimizer "${opt}" \
        --lora_type lora_nsp \
        --lora_rank "${LORA_RANK}" \
        --use_dora "${USE_DORA}" \
        --seed "${SEED}" \
        --gpu 0 \
        --output_dir "${OUT_DIR}" \
        --experiment_name "${exp_name}" \
        --train_budget_mode uniform \
        --projection_param_mode full \
        --null_init_mode history_init_runtime \
        > "${LOG_DIR}/${exp_name}.log" 2>&1
    echo "[GPU${GPU_ID}] Finished ${exp_name} at $(date -Iseconds)"
done

echo "[GPU${GPU_ID}] All optimizer ablation variants completed at $(date -Iseconds)"
