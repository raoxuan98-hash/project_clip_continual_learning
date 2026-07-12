#!/usr/bin/env bash
set -euo pipefail

if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" && -z "${CONDA_DEFAULT_ENV:-}" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-raoxuan}"
fi

ROOT="${ROOT:-/data1/open_datasets/X-TAIL}"
OUT_DIR="${OUT_DIR:-experiments/incremental_text_schedule_stress_20260614}"
SEEDS="${SEEDS:-42}"
ITERATIONS="${ITERATIONS:-800}"
BATCH_SIZE="${BATCH_SIZE:-64}"
GPUS="${GPUS:-0 1 2}"
DRY_RUN="${DRY_RUN:-0}"
EXPECTED_TASKS="${EXPECTED_TASKS:-aircraft caltech101 dtd eurosat}"

export CLIP_USE_SAFETENSORS="${CLIP_USE_SAFETENSORS:-0}"
export CLIP_LOCAL_FILES_ONLY="${CLIP_LOCAL_FILES_ONLY:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

mkdir -p "${OUT_DIR}/logs" "${OUT_DIR}/runs"

CONFIG_NAMES=(
  text_always
  text_task1_freeze
  text_task1_lr_1_5
)
CONFIG_TEXT_SCHEDULE=(always freeze_after low_lr_after)
CONFIG_SWITCH_TASK=(1 1 1)
CONFIG_TEXT_LR_SCALE=(1.0 0.0 0.2)
CONFIG_TUNE_TEXT=(true true true)

COMMON_ARGS=(
  --root "${ROOT}"
  --task_sequence ${EXPECTED_TASKS}
  --num_shots 16
  --iterations "${ITERATIONS}"
  --batch_size "${BATCH_SIZE}"
  --eval_max_samples 0
  --method lora_nsp
  --fd_weight 1.0
  --cd_weight 1.0
  --tune_vision_encoder true
)

read -r -a GPU_LIST <<< "${GPUS}"
if [[ "${#GPU_LIST[@]}" -lt "${#CONFIG_NAMES[@]}" ]]; then
  echo "Need at least ${#CONFIG_NAMES[@]} GPUs in GPUS, got: ${GPUS}" >&2
  exit 1
fi

write_manifest() {
  local manifest="${OUT_DIR}/manifest.txt"
  {
    echo "created_at: $(date '+%F %T %Z')"
    echo "hostname: $(hostname)"
    echo "pwd: $(pwd)"
    echo "python: $(command -v python)"
    echo "conda_env: ${CONDA_DEFAULT_ENV:-}"
    echo "root: ${ROOT}"
    echo "out_dir: ${OUT_DIR}"
    echo "seeds: ${SEEDS}"
    echo "iterations: ${ITERATIONS}"
    echo "batch_size: ${BATCH_SIZE}"
    echo "gpus: ${GPUS}"
    echo "dry_run: ${DRY_RUN}"
    echo "expected_tasks: ${EXPECTED_TASKS}"
    echo "CLIP_USE_SAFETENSORS: ${CLIP_USE_SAFETENSORS}"
    echo "CLIP_LOCAL_FILES_ONLY: ${CLIP_LOCAL_FILES_ONLY}"
    echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS}"
    echo "MKL_NUM_THREADS: ${MKL_NUM_THREADS}"
    echo "OPENBLAS_NUM_THREADS: ${OPENBLAS_NUM_THREADS}"
    echo "NUMEXPR_NUM_THREADS: ${NUMEXPR_NUM_THREADS}"
    echo
    echo "[git]"
    git rev-parse HEAD 2>/dev/null || true
    git status --short 2>/dev/null || true
    echo
    echo "[common_args]"
    printf '%q ' "${COMMON_ARGS[@]}"
    echo
    echo
    echo "[configs]"
    for idx in "${!CONFIG_NAMES[@]}"; do
      printf '%s method=lora_nsp fd=1.0 cd=1.0 tune_vision=true tune_text=%s text_schedule=%s switch_task=%s text_lr_scale=%s gpu=%s\n' \
        "${CONFIG_NAMES[$idx]}" \
        "${CONFIG_TUNE_TEXT[$idx]}" \
        "${CONFIG_TEXT_SCHEDULE[$idx]}" \
        "${CONFIG_SWITCH_TASK[$idx]}" \
        "${CONFIG_TEXT_LR_SCALE[$idx]}" \
        "${GPU_LIST[$idx]:-}"
    done
  } > "${manifest}"
}

run_config() {
  local gpu="$1"
  local name="$2"
  local schedule="$3"
  local switch_task="$4"
  local scale="$5"
  local tune_text="$6"

  for seed in ${SEEDS}; do
    local run_dir="${OUT_DIR}/runs/${name}_seed${seed}"
    local log="${OUT_DIR}/logs/${name}_seed${seed}.log"
    local result="${OUT_DIR}/${name}_seed${seed}_results.json"
    local nested_result="${run_dir}/${name}_results.json"
    if [[ -f "${result}" ]]; then
      echo "[$(date '+%F %T')] skip ${name} seed=${seed}; result exists: ${result}" | tee -a "${OUT_DIR}/logs/launch.log"
      continue
    fi
    mkdir -p "${run_dir}"
    echo "[$(date '+%F %T')] start text-schedule stress ${name} seed=${seed} gpu=${gpu}" | tee -a "${OUT_DIR}/logs/launch.log"
    if [[ "${DRY_RUN}" == "1" ]]; then
      printf 'DRY_RUN CUDA_VISIBLE_DEVICES=%q python src/experiments/run_continual_learning.py ' "${gpu}" > "${log}"
      printf '%q ' "${COMMON_ARGS[@]}" >> "${log}"
      printf '%q ' \
        --tune_text_encoder "${tune_text}" \
        --text_tuning_schedule "${schedule}" \
        --text_schedule_switch_task "${switch_task}" \
        --text_lr_scale_after_task "${scale}" \
        --seed "${seed}" \
        --experiment_name "${name}" \
        --output_dir "${run_dir}" \
        >> "${log}"
      echo >> "${log}"
      echo "[$(date '+%F %T')] dry-run text-schedule stress ${name} seed=${seed}" | tee -a "${OUT_DIR}/logs/launch.log"
      continue
    fi
    CUDA_VISIBLE_DEVICES="${gpu}" python src/experiments/run_continual_learning.py \
      "${COMMON_ARGS[@]}" \
      --tune_text_encoder "${tune_text}" \
      --text_tuning_schedule "${schedule}" \
      --text_schedule_switch_task "${switch_task}" \
      --text_lr_scale_after_task "${scale}" \
      --seed "${seed}" \
      --experiment_name "${name}" \
      --output_dir "${run_dir}" \
      > "${log}" 2>&1
    cp "${nested_result}" "${result}"
    echo "[$(date '+%F %T')] done text-schedule stress ${name} seed=${seed}" | tee -a "${OUT_DIR}/logs/launch.log"
  done
}

write_manifest

PIDS=()
for idx in "${!CONFIG_NAMES[@]}"; do
  run_config \
    "${GPU_LIST[$idx]}" \
    "${CONFIG_NAMES[$idx]}" \
    "${CONFIG_TEXT_SCHEDULE[$idx]}" \
    "${CONFIG_SWITCH_TASK[$idx]}" \
    "${CONFIG_TEXT_LR_SCALE[$idx]}" \
    "${CONFIG_TUNE_TEXT[$idx]}" &
  PIDS+=("$!")
done

STATUS=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    STATUS=1
  fi
done

if [[ "${STATUS}" -ne 0 ]]; then
  echo "[$(date '+%F %T')] at least one text-schedule stress job failed" | tee -a "${OUT_DIR}/logs/launch.log" >&2
  exit "${STATUS}"
fi

python scripts/summarize_incremental_metrics.py \
  "${OUT_DIR}" \
  --output_csv "${OUT_DIR}/incremental_summary.csv" \
  --output_markdown "${OUT_DIR}/incremental_summary.md" \
  --aggregate_csv "${OUT_DIR}/incremental_aggregate.csv" \
  --aggregate_markdown "${OUT_DIR}/incremental_aggregate.md" \
  --expected_tasks "${EXPECTED_TASKS}" \
  | tee "${OUT_DIR}/logs/incremental_summary.log"

