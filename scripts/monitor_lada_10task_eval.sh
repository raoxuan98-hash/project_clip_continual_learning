#!/bin/bash
set -e
PROJECT_DIR=/home/raoxuan/projects/project_clip_continual_learning
export HF_ENDPOINT=https://hf-mirror.com

COMBOS=(combo1_balanced_10task combo2_lastmax_10task combo3_transfermax_10task combo4_initonly_10task)
PIDS=($1 $2 $3 $4)

cd "$PROJECT_DIR"

# Wait for all training processes to finish
for i in "${!COMBOS[@]}"; do
    echo "Waiting for ${COMBOS[$i]} (PID ${PIDS[$i]})..."
    tail --pid=${PIDS[$i]} -f /dev/null
done

echo "All training finished. Alpha sweep was run inline during training."
echo "Results are in experiments/lada_10task_eval/combo*_train.log and combo*.json"
