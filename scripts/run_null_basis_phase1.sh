#!/usr/bin/env bash
# Phase 1: 6 datasets x 5 LoRA variants (DoRA + AdamW), 2 GPUs parallel.
# Datasets: aircraft caltech101 dtd eurosat flowers oxford_pets
# Two idle GPUs are auto-selected via nvidia-smi before launch.
#
# Usage (local or remote):
#   bash scripts/run_null_basis_phase1.sh
#
# Dry-run to verify GPU selection:
#   DRY_RUN=1 bash scripts/run_null_basis_phase1.sh
#
# To override defaults:
#   PROJECT_DIR=... DATA_ROOT=... OUT_DIR=... bash scripts/run_null_basis_phase1.sh
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/raoxuan/projects/project_clip_continual_learning}"
OUT_DIR="${OUT_DIR:-experiments/null_basis_ablation_phase1}"
LOG_DIR="${LOG_DIR:-${OUT_DIR}/logs}"

SEED="${SEED:-42}"
ITERATIONS="${ITERATIONS:-800}"
NUM_GPUS="${NUM_GPUS:-2}"
DRY_RUN="${DRY_RUN:-0}"

export PROJECT_DIR DATA_ROOT OUT_DIR LOG_DIR SEED ITERATIONS BATCH_SIZE LR WD LORA_RANK PYTHON DRY_RUN

cd "${PROJECT_DIR}"
mkdir -p "${LOG_DIR}"

select_idle_gpus() {
    local n="$1"
    if ! command -v nvidia-smi &>/dev/null; then
        echo "nvidia-smi not found; cannot auto-select idle GPUs" >&2
        return 1
    fi
    local gpus
    gpus=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null | \
        awk -F',' '{gsub(/ /,"",$1); gsub(/ /,"",$2); gsub(/ /,"",$3); print $1,$2,$3}' | \
        sort -k2,2n -k3,3n | \
        head -n "$n" | \
        awk '{print $1}')
    local count
    count=$(echo "$gpus" | grep -c '^[0-9]\+$' || true)
    if [ "$count" -lt "$n" ]; then
        echo "Need $n idle GPUs, found $count" >&2
        return 1
    fi
    echo "$gpus"
}

mapfile -t IDLE_GPUS < <(select_idle_gpus "${NUM_GPUS}")
GPU0="${IDLE_GPUS[0]}"
GPU1="${IDLE_GPUS[1]}"

echo "============================================"
echo "Phase 1 null/basis ablation"
echo "  Datasets: aircraft caltech101 dtd eurosat flowers oxford_pets"
echo "  Iterations: ${ITERATIONS}"
echo "  Seed: ${SEED}"
echo "  Selected GPUs: ${GPU0}, ${GPU1}"
echo "  Output: ${OUT_DIR}"
echo "  Logs:   ${LOG_DIR}"
echo "============================================"

# Syntax check both scripts before launching.
bash -n scripts/run_null_basis_phase1_gpu0.sh
bash -n scripts/run_null_basis_phase1_gpu1.sh

if [ "$DRY_RUN" -eq 1 ]; then
    echo ""
    echo "DRY_RUN: would launch with"
    echo "  GPU_ID=${GPU0} bash scripts/run_null_basis_phase1_gpu0.sh"
    echo "  GPU_ID=${GPU1} bash scripts/run_null_basis_phase1_gpu1.sh"
    echo ""
    echo "Selected CUDA_VISIBLE_DEVICES: ${GPU0},${GPU1}"
    exit 0
fi

GPU0_LOG="${LOG_DIR}/gpu${GPU0}_master_seed${SEED}.log"
GPU1_LOG="${LOG_DIR}/gpu${GPU1}_master_seed${SEED}.log"

GPU_ID="${GPU0}" nohup bash scripts/run_null_basis_phase1_gpu0.sh > "${GPU0_LOG}" 2>&1 &
GPU0_PID=$!
GPU_ID="${GPU1}" nohup bash scripts/run_null_basis_phase1_gpu1.sh > "${GPU1_LOG}" 2>&1 &
GPU1_PID=$!

echo ""
echo "Launched GPU ${GPU0} pipeline (PID ${GPU0_PID}); log: ${GPU0_LOG}"
echo "Launched GPU ${GPU1} pipeline (PID ${GPU1_PID}); log: ${GPU1_LOG}"
echo ""
echo "Wait for completion:"
echo "  tail -f ${GPU0_LOG}"
echo "  tail -f ${GPU1_LOG}"
echo ""
echo "Check running processes:"
echo "  pgrep -af run_null_basis_phase1"
