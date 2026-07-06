#!/usr/bin/env bash
# Phase 1 null/basis ablation under plain LoRA (SGPBaseLoRA).
# Reuses the same auto-GPU-selection launcher but sets USE_DORA=false.
#
# Usage:
#   bash scripts/run_null_basis_phase1_lora.sh
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/raoxuan/projects/project_clip_continual_learning}"

export USE_DORA="false"
export OUT_DIR="${OUT_DIR:-experiments/null_basis_ablation_phase1_lora}"

cd "${PROJECT_DIR}"

bash scripts/run_null_basis_phase1.sh "$@"
