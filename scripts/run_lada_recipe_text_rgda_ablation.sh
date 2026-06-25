#!/usr/bin/env bash
set -euo pipefail

if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" && -z "${CONDA_DEFAULT_ENV:-}" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-raoxuan}"
fi

ROOT="${ROOT:-/data1/open_datasets/X-TAIL}"
OUT_DIR="${OUT_DIR:-experiments/lada_recipe_text_rgda_ablation_$(date +%Y%m%d_%H%M%S)}"
SEEDS="${SEEDS:-42}"
BATCH_SIZE="${BATCH_SIZE:-64}"
ITERATIONS="${ITERATIONS:-800}"
GPUS="${GPUS:-0 1 2 3}"
DRY_RUN="${DRY_RUN:-0}"
TASKS="${TASKS:-aircraft caltech101}"
PARALLEL_JOBS="${PARALLEL_JOBS:-4}"
CONFIG_FILTER="${CONFIG_FILTER:-}"

export CLIP_USE_SAFETENSORS="${CLIP_USE_SAFETENSORS:-0}"
export CLIP_LOCAL_FILES_ONLY="${CLIP_LOCAL_FILES_ONLY:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

mkdir -p "${OUT_DIR}/logs" "${OUT_DIR}/runs"

CONFIG_NAMES=(
  base_current
  recipe_current
  no_fd_cd_current
  no_fd_cd_lada_hybrid
  vanilla_no_fd_cd_current
  vanilla_no_fd_cd_lada_hybrid
  text_only_no_fd_cd_lada_hybrid
)
CONFIG_LORA=(lora_nsp lora_nsp lora_nsp lora_nsp lora_vanilla lora_vanilla lora_vanilla)
CONFIG_FD=(1.0 1.0 0.0 0.0 0.0 0.0 0.0)
CONFIG_CD=(1.0 1.0 0.0 0.0 0.0 0.0 0.0)
CONFIG_TUNE_VISION=(true true true true true true false)
CONFIG_TUNE_TEXT=(true true true true true true true)
CONFIG_TEXT_MODE=(current current current lada_hybrid current lada_hybrid lada_hybrid)
CONFIG_RECIPE=(0 1 0 0 0 0 0)

should_run_config() {
  local name="$1"
  local selected
  if [[ -z "${CONFIG_FILTER}" ]]; then
    return 0
  fi
  for selected in ${CONFIG_FILTER}; do
    if [[ "${selected}" == "${name}" ]]; then
      return 0
    fi
  done
  return 1
}

COMMON_ARGS=(
  --root "${ROOT}"
  --dataset_sequence ${TASKS}
  --num_shots 16
  --batch_size "${BATCH_SIZE}"
  --iterations "${ITERATIONS}"
  --eval_max_samples 0
  --num_centers 1
  --alpha 0.05
)

read -r -a GPU_LIST <<< "${GPUS}"
if [[ "${#GPU_LIST[@]}" -lt 1 ]]; then
  echo "GPUS must contain at least one GPU id" >&2
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
    echo "tasks: ${TASKS}"
    echo "iterations: ${ITERATIONS}"
    echo "batch_size: ${BATCH_SIZE}"
    echo "gpus: ${GPUS}"
    echo "parallel_jobs: ${PARALLEL_JOBS}"
    echo "config_filter: ${CONFIG_FILTER}"
    echo "dry_run: ${DRY_RUN}"
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
      printf '%s lora=%s fd=%s cd=%s tune_vision=%s tune_text=%s text_mode=%s recipe=%s\n' \
        "${CONFIG_NAMES[$idx]}" \
        "${CONFIG_LORA[$idx]}" \
        "${CONFIG_FD[$idx]}" \
        "${CONFIG_CD[$idx]}" \
        "${CONFIG_TUNE_VISION[$idx]}" \
        "${CONFIG_TUNE_TEXT[$idx]}" \
        "${CONFIG_TEXT_MODE[$idx]}" \
        "${CONFIG_RECIPE[$idx]}"
    done
  } > "${manifest}"
}

run_one() {
  local gpu="$1"
  local name="$2"
  local lora_type="$3"
  local fd="$4"
  local cd="$5"
  local tune_vision="$6"
  local tune_text="$7"
  local text_mode="$8"
  local use_recipe="$9"
  local seed="${10}"

  local run_name="${name}_seed${seed}"
  local run_dir="${OUT_DIR}/runs/${run_name}"
  local log="${OUT_DIR}/logs/${run_name}.log"
  mkdir -p "${run_dir}"

  local recipe_args=()
  if [[ "${use_recipe}" == "1" ]]; then
    recipe_args=(--use_lada_recipe_defaults)
  fi

  echo "[$(date '+%F %T')] start ${run_name} gpu=${gpu}" | tee -a "${OUT_DIR}/logs/launch.log"
  if [[ "${DRY_RUN}" == "1" ]]; then
    {
      printf 'DRY_RUN CUDA_VISIBLE_DEVICES=%q python main_incremental.py ' "${gpu}"
      printf '%q ' "${COMMON_ARGS[@]}"
      printf '%q ' \
        --lora_type "${lora_type}" \
        --init_mode "${lora_type}" \
        --fd_weight "${fd}" \
        --cd_weight "${cd}" \
        --tune_vision_encoder "${tune_vision}" \
        --tune_text_encoder "${tune_text}" \
        --text_classifier_mode "${text_mode}" \
        --seed "${seed}" \
        --experiment_name "${run_name}" \
        --output_dir "${run_dir}"
      if [[ "${#recipe_args[@]}" -gt 0 ]]; then
        printf '%q ' "${recipe_args[@]}"
      fi
      echo
    } > "${log}"
    echo "[$(date '+%F %T')] dry-run ${run_name}" | tee -a "${OUT_DIR}/logs/launch.log"
    return 0
  fi

  CUDA_VISIBLE_DEVICES="${gpu}" python main_incremental.py \
    "${COMMON_ARGS[@]}" \
    --lora_type "${lora_type}" \
    --init_mode "${lora_type}" \
    --fd_weight "${fd}" \
    --cd_weight "${cd}" \
    --tune_vision_encoder "${tune_vision}" \
    --tune_text_encoder "${tune_text}" \
    --text_classifier_mode "${text_mode}" \
    --seed "${seed}" \
    --experiment_name "${run_name}" \
    --output_dir "${run_dir}" \
    ${recipe_args+"${recipe_args[@]}"} \
    > "${log}" 2>&1
  echo "[$(date '+%F %T')] done ${run_name}" | tee -a "${OUT_DIR}/logs/launch.log"
}

write_manifest

PIDS=()
job_count=0
STATUS=0
for seed in ${SEEDS}; do
  for idx in "${!CONFIG_NAMES[@]}"; do
    if ! should_run_config "${CONFIG_NAMES[$idx]}"; then
      continue
    fi
    gpu="${GPU_LIST[$((job_count % ${#GPU_LIST[@]}))]}"
    run_one \
      "${gpu}" \
      "${CONFIG_NAMES[$idx]}" \
      "${CONFIG_LORA[$idx]}" \
      "${CONFIG_FD[$idx]}" \
      "${CONFIG_CD[$idx]}" \
      "${CONFIG_TUNE_VISION[$idx]}" \
      "${CONFIG_TUNE_TEXT[$idx]}" \
      "${CONFIG_TEXT_MODE[$idx]}" \
      "${CONFIG_RECIPE[$idx]}" \
      "${seed}" &
    PIDS+=("$!")
    job_count=$((job_count + 1))
    if [[ "${#PIDS[@]}" -ge "${PARALLEL_JOBS}" ]]; then
      for pid in "${PIDS[@]}"; do
        if ! wait "${pid}"; then
          STATUS=1
        fi
      done
      PIDS=()
    fi
  done
done

for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    STATUS=1
  fi
done

if [[ "${STATUS}" -ne 0 ]]; then
  echo "[$(date '+%F %T')] at least one ablation job failed" | tee -a "${OUT_DIR}/logs/launch.log" >&2
  exit "${STATUS}"
fi

python scripts/summarize_incremental_metrics.py \
  "${OUT_DIR}/runs" \
  --output_csv "${OUT_DIR}/incremental_summary.csv" \
  --output_markdown "${OUT_DIR}/incremental_summary.md" \
  --aggregate_csv "${OUT_DIR}/incremental_aggregate.csv" \
  --aggregate_markdown "${OUT_DIR}/incremental_aggregate.md" \
  --expected_tasks "${TASKS}" \
  | tee "${OUT_DIR}/logs/incremental_summary.log"
