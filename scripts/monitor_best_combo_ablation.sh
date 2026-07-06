#!/bin/bash
set -euo pipefail

PROJECT=/home/raoxuan/projects/project_clip_continual_learning
OUT_DIR=$PROJECT/experiments/best_combo_ablation
NAMES=(
  combo_balanced_adamw_1e-4_iter800_cwu
  combo_last_adamw_3e-4_iter1600_linear
  combo_transfer_hist_runtime_adamw_1e-4_iter800_cosine
  combo_init_only_hist_only_adamw_1e-4_iter800_cwu
)

wait_for_file() {
  local path=$1
  local name=$2
  echo "Waiting for $name ..."
  while [ ! -f "$path" ]; do
    sleep 60
  done
  echo "$name done."
}

for name in "${NAMES[@]}"; do
  wait_for_file "$OUT_DIR/${name}_zs_results.json" "$name (zs)"
  wait_for_file "$OUT_DIR/${name}_ens_results.json" "$name (ens)"
done

$PROJECT/ENTER/envs/raoxuan/bin/python $PROJECT/scripts/summarize_best_combo_ablation.py
