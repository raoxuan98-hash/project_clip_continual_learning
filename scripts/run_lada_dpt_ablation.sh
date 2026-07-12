#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"
ROOT="${XTAIL_ROOT:-/data1/open_datasets/X-TAIL}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="experiments/lada_dpt_ablation_${STAMP}"

mkdir -p "${OUT_DIR}"

COMMON_ARGS=(
  --dataset_sequence aircraft caltech101
  --root "${ROOT}"
  --gpu "${GPU}"
  --num_shots 16
  --batch_size 64
  --iterations 800
  --lr 0.001
  --lada_k 16
  --prototype_k 4
  --lada_alpha 1.0
  --image_prototypes_weight_coef 64.0
  --lada_official_mode
)

for MODE in none mean dpt; do
  python scripts/legacy/main_incremental_lada.py \
    "${COMMON_ARGS[@]}" \
    --lada_replay_mode "${MODE}" \
    2>&1 | tee "${OUT_DIR}/${MODE}.log"
done

echo "Logs saved to ${OUT_DIR}"
