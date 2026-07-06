#!/bin/bash
set -euo pipefail

PROJECT=/home/raoxuan/projects/project_clip_continual_learning
OUT_DIR=$PROJECT/experiments/layer_rank_ablation
NAMES=(
  layer_all_rank4
  layer_qk_only
  layer_ffn_only
  layer_attn_only
  rank_1
  rank_2
  rank_4
  rank_8
)

wait_for_file() {
  local path=$1
  while [ ! -f "$path" ]; do
    sleep 60
  done
}

for name in "${NAMES[@]}"; do
  echo "Waiting for $name ..."
  wait_for_file "$OUT_DIR/${name}_zs_results.json"
  wait_for_file "$OUT_DIR/${name}_ens_results.json"
  echo "$name done."
done

source $PROJECT/ENTER/envs/raoxuan/etc/profile.d/conda.sh || true
conda activate raoxuan || true
/home/raoxuan/ENTER/envs/raoxuan/bin/python $PROJECT/scripts/summarize_layer_rank_ablation.py
