#!/usr/bin/env bash
# Phase 1 null/basis ablation on GPU 0.
# Runs: current_nsp, basis_fixed_tail, basis_core_tail
# Usage: bash scripts/run_null_basis_phase1_gpu0.sh
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/raoxuan/projects/project_clip_continual_learning}"
DATA_ROOT="${DATA_ROOT:-/data1/open_datasets/X-TAIL}"
OUT_DIR="${OUT_DIR:-experiments/null_basis_ablation_phase1}"
LOG_DIR="${LOG_DIR:-${OUT_DIR}/logs}"
PYTHON="${PYTHON:-/home/raoxuan/ENTER/envs/raoxuan/bin/python}"

SEED="${SEED:-42}"
ITERATIONS="${ITERATIONS:-800}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-1e-4}"
WD="${WD:-3e-5}"
LORA_RANK="${LORA_RANK:-4}"
GPU_ID="${GPU_ID:-0}"
USE_DORA="${USE_DORA:-true}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

cd "${PROJECT_DIR}"
mkdir -p "${LOG_DIR}"

run_variant() {
    local name="$1"
    shift
    local extra_args=("$@")

    echo "[GPU${GPU_ID}] Starting ${name} at $(date -Iseconds)"
    ${PYTHON} main_incremental.py \
        --root "${DATA_ROOT}" \
        --dataset_sequence aircraft caltech101 dtd eurosat flowers oxford_pets \
        --num_shots 16 \
        --batch_size "${BATCH_SIZE}" \
        --iterations "${ITERATIONS}" \
        --lr "${LR}" \
        --weight_decay "${WD}" \
        --optimizer adamw \
        --lora_type lora_nsp \
        --lora_rank "${LORA_RANK}" \
        --use_dora "${USE_DORA}" \
        --seed "${SEED}" \
        --gpu 0 \
        --output_dir "${OUT_DIR}" \
        --experiment_name "${name}_seed${SEED}" \
        --train_budget_mode uniform \
        "${extra_args[@]}" \
        > "${LOG_DIR}/${name}_seed${SEED}.log" 2>&1
    echo "[GPU${GPU_ID}] Finished ${name} at $(date -Iseconds)"
}

run_variant "current_nsp" \
    --projection_param_mode full \
    --null_init_mode none

run_variant "basis_fixed_tail" \
    --projection_param_mode fixed_basis \
    --basis_rank "${LORA_RANK}" \
    --basis_window tail

run_variant "basis_core_tail" \
    --projection_param_mode core_basis \
    --basis_rank 16 \
    --basis_window tail

echo "[GPU0] All variants completed at $(date -Iseconds)"
