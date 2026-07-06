#!/usr/bin/env bash
# Master script to run optimizer ablation on 3 most available GPUs in parallel.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/raoxuan/projects/project_clip_continual_learning}"
PYTHON="${PYTHON:-/home/raoxuan/ENTER/envs/raoxuan/bin/python}"

# Pick three least-utilized GPUs.
GPU_IDS=$(${PYTHON} - <<'PY'
import subprocess
out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                               "--format=csv,noheader,nounits"], text=True)
rows = []
for line in out.strip().split("\n"):
    parts = [x.strip() for x in line.split(",")]
    idx, util, mem_used, mem_total = parts
    util = int(util)
    mem_ratio = int(mem_used) / max(int(mem_total), 1)
    rows.append((util, mem_ratio, int(idx)))
rows.sort(key=lambda x: (x[0], x[1]))
print(",".join(str(r[2]) for r in rows[:3]))
PY
)

IFS=',' read -r GPU0 GPU1 GPU2 <<< "${GPU_IDS}"

echo "Optimizer ablation will run on GPUs: ${GPU_IDS}"

# 10 experiments balanced across 3 GPUs. Estimated per-variant runtime:
# current_nsp ~53 min, hist_null_init_runtime ~41 min.
"${PROJECT_DIR}/scripts/run_optimizer_ablation_pipeline.sh" "${GPU0}" \
    "current_nsp:adamw current_nsp:adam current_nsp:sgd" &
PID0=$!

"${PROJECT_DIR}/scripts/run_optimizer_ablation_pipeline.sh" "${GPU1}" \
    "current_nsp:adagrad current_nsp:rmsprop hist_null_init_runtime:adamw" &
PID1=$!

"${PROJECT_DIR}/scripts/run_optimizer_ablation_pipeline.sh" "${GPU2}" \
    "hist_null_init_runtime:adam hist_null_init_runtime:sgd hist_null_init_runtime:adagrad hist_null_init_runtime:rmsprop" &
PID2=$!

wait "${PID0}"
wait "${PID1}"
wait "${PID2}"

echo "Optimizer ablation completed on GPUs ${GPU_IDS}."
