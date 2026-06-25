#!/usr/bin/env bash
set -euo pipefail

if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" && -z "${CONDA_DEFAULT_ENV:-}" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-raoxuan}"
fi

ROOT="${ROOT:-/data1/open_datasets/X-TAIL}"
OUT_DIR="${OUT_DIR:-experiments/visual_tuning_anchor_sweep_$(date +%Y%m%d_%H%M%S)}"
TASKS="${TASKS:-aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397}"
SEED="${SEED:-42}"
GPUS="${GPUS:-0 1 5}"
PARALLEL_JOBS="${PARALLEL_JOBS:-3}"
BATCH_SIZE="${BATCH_SIZE:-64}"
ITERATIONS="${ITERATIONS:-800}"
NUM_CENTERS="${NUM_CENTERS:-4}"
ARTIFACT_NUM_CENTERS="${ARTIFACT_NUM_CENTERS:-1,4}"
ALPHAS="${ALPHAS:-0,0.05,0.1,0.2,0.5,1.0}"
VARIANTS="${VARIANTS:-sc:m=1:ft=0,mc4:m=4:ft=0,mc4ft200:m=4:ft=200:lr=0.01:source=gmm_sample}"
EVAL_SHOTS_PER_CLASS="${EVAL_SHOTS_PER_CLASS:-0}"
EVAL_SUBSET_SEED="${EVAL_SUBSET_SEED:-0}"
RGDA_EVAL_CHUNK_SIZE="${RGDA_EVAL_CHUNK_SIZE:-512}"
CONFIG_FILTER="${CONFIG_FILTER:-}"
DRY_RUN="${DRY_RUN:-0}"

export CLIP_USE_SAFETENSORS="${CLIP_USE_SAFETENSORS:-0}"
export CLIP_LOCAL_FILES_ONLY="${CLIP_LOCAL_FILES_ONLY:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

CONFIG_NAMES=(
  text_only_anchor
  vanilla_vision_anchor
  nsp_fd_cd_vision_anchor
)
CONFIG_LORA=(lora_vanilla lora_vanilla lora_nsp)
CONFIG_FD=(0.0 0.0 1.0)
CONFIG_CD=(0.0 0.0 1.0)
CONFIG_TUNE_VISION=(false true true)
CONFIG_TUNE_TEXT=(true true true)

mkdir -p "${OUT_DIR}/logs" "${OUT_DIR}/runs" "${OUT_DIR}/async" "${OUT_DIR}/sweep"

read -r -a GPU_LIST <<< "${GPUS}"
if [[ "${#GPU_LIST[@]}" -lt 1 ]]; then
  echo "GPUS must contain at least one GPU id" >&2
  exit 1
fi

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

write_manifest() {
  {
    echo "created_at: $(date '+%F %T %Z')"
    echo "hostname: $(hostname)"
    echo "pwd: $(pwd)"
    echo "python: $(command -v python)"
    echo "conda_env: ${CONDA_DEFAULT_ENV:-}"
    echo "root: ${ROOT}"
    echo "out_dir: ${OUT_DIR}"
    echo "tasks: ${TASKS}"
    echo "seed: ${SEED}"
    echo "gpus: ${GPUS}"
    echo "parallel_jobs: ${PARALLEL_JOBS}"
    echo "batch_size: ${BATCH_SIZE}"
    echo "iterations: ${ITERATIONS}"
    echo "num_centers: ${NUM_CENTERS}"
    echo "artifact_num_centers: ${ARTIFACT_NUM_CENTERS}"
    echo "alphas: ${ALPHAS}"
    echo "variants: ${VARIANTS}"
    echo "eval_shots_per_class: ${EVAL_SHOTS_PER_CLASS}"
    echo "eval_subset_seed: ${EVAL_SUBSET_SEED}"
    echo "rgda_eval_chunk_size: ${RGDA_EVAL_CHUNK_SIZE}"
    echo "config_filter: ${CONFIG_FILTER}"
    echo "dry_run: ${DRY_RUN}"
    echo "CLIP_USE_SAFETENSORS: ${CLIP_USE_SAFETENSORS}"
    echo "CLIP_LOCAL_FILES_ONLY: ${CLIP_LOCAL_FILES_ONLY}"
    echo
    echo "[git]"
    git rev-parse HEAD 2>/dev/null || true
    git status --short 2>/dev/null || true
    echo
    echo "[configs]"
    for idx in "${!CONFIG_NAMES[@]}"; do
      printf '%s lora=%s fd=%s cd=%s tune_vision=%s tune_text=%s text_mode=lada_hybrid\n' \
        "${CONFIG_NAMES[$idx]}" \
        "${CONFIG_LORA[$idx]}" \
        "${CONFIG_FD[$idx]}" \
        "${CONFIG_CD[$idx]}" \
        "${CONFIG_TUNE_VISION[$idx]}" \
        "${CONFIG_TUNE_TEXT[$idx]}"
    done
    echo
    echo "[failure_policy]"
    echo "Stop interpreting results if any config has missing artifacts, missing sweep JSONs, NaN loss, or class-count mismatch."
    echo "Do not retry the same failing config more than twice without code changes."
    echo "If the server cannot be reached for three consecutive checks, open EasyConnect; if it still fails, stop and mark the run blocked."
  } > "${OUT_DIR}/manifest.txt"
}

run_one() {
  local gpu="$1"
  local name="$2"
  local lora_type="$3"
  local fd="$4"
  local cd="$5"
  local tune_vision="$6"
  local tune_text="$7"

  local run_name="${name}_seed${SEED}"
  local run_dir="${OUT_DIR}/runs/${run_name}"
  local async_dir="${OUT_DIR}/async/${run_name}"
  local sweep_dir="${OUT_DIR}/sweep/${run_name}"
  local train_log="${OUT_DIR}/logs/${run_name}_train.log"
  local sweep_log="${OUT_DIR}/logs/${run_name}_sweep.log"

  mkdir -p "${run_dir}" "${async_dir}" "${sweep_dir}"

  local train_cmd=(
    python main_incremental.py
    --root "${ROOT}"
    --dataset_sequence ${TASKS}
    --num_shots 16
    --batch_size "${BATCH_SIZE}"
    --iterations "${ITERATIONS}"
    --eval_max_samples 0
    --num_centers "${NUM_CENTERS}"
    --artifact_num_centers "${ARTIFACT_NUM_CENTERS}"
    --alpha 0.5
    --lora_type "${lora_type}"
    --init_mode "${lora_type}"
    --fd_weight "${fd}"
    --cd_weight "${cd}"
    --tune_vision_encoder "${tune_vision}"
    --tune_text_encoder "${tune_text}"
    --text_classifier_mode lada_hybrid
    --seed "${SEED}"
    --experiment_name "${run_name}"
    --output_dir "${run_dir}"
    --async_eval_dir "${async_dir}"
    --save_step_artifacts
    --skip_inline_eval
  )

  local sweep_cmd=(
    python scripts/evaluate_incremental_rgda_sweep_artifacts.py
    --async_eval_dir "${async_dir}"
    --output_dir "${sweep_dir}"
    --experiment_name "${run_name}"
    --device cuda:0
    --variants "${VARIANTS}"
    --alphas "${ALPHAS}"
    --eval_shots_per_class "${EVAL_SHOTS_PER_CLASS}"
    --eval_subset_seed "${EVAL_SUBSET_SEED}"
    --rgda_eval_chunk_size "${RGDA_EVAL_CHUNK_SIZE}"
  )

  {
    echo "[$(date '+%F %T')] start training ${run_name} gpu=${gpu}"
    printf 'CUDA_VISIBLE_DEVICES=%q ' "${gpu}"
    printf '%q ' "${train_cmd[@]}"
    echo
  } | tee -a "${OUT_DIR}/logs/launch.log"

  if [[ "${DRY_RUN}" == "1" ]]; then
    {
      echo "DRY_RUN training command:"
      printf 'CUDA_VISIBLE_DEVICES=%q ' "${gpu}"
      printf '%q ' "${train_cmd[@]}"
      echo
      echo "DRY_RUN sweep command:"
      printf 'CUDA_VISIBLE_DEVICES=%q ' "${gpu}"
      printf '%q ' "${sweep_cmd[@]}"
      echo
    } > "${OUT_DIR}/logs/${run_name}.log"
    echo "[$(date '+%F %T')] dry-run ${run_name}" | tee -a "${OUT_DIR}/logs/launch.log"
    return 0
  fi

  CUDA_VISIBLE_DEVICES="${gpu}" "${train_cmd[@]}" > "${train_log}" 2>&1
  echo "[$(date '+%F %T')] done training ${run_name}" | tee -a "${OUT_DIR}/logs/launch.log"

  {
    echo "[$(date '+%F %T')] start rgda sweep ${run_name} gpu=${gpu}"
    printf 'CUDA_VISIBLE_DEVICES=%q ' "${gpu}"
    printf '%q ' "${sweep_cmd[@]}"
    echo
  } | tee -a "${OUT_DIR}/logs/launch.log"

  CUDA_VISIBLE_DEVICES="${gpu}" "${sweep_cmd[@]}" > "${sweep_log}" 2>&1
  echo "[$(date '+%F %T')] done rgda sweep ${run_name}" | tee -a "${OUT_DIR}/logs/launch.log"
}

write_manifest

declare -a PIDS=()
job_count=0
STATUS=0
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
    "${CONFIG_TUNE_TEXT[$idx]}" &
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

if [[ "${#PIDS[@]}" -gt 0 ]]; then
  for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
      STATUS=1
    fi
  done
fi

if [[ "${STATUS}" -ne 0 ]]; then
  echo "[$(date '+%F %T')] one or more configs failed" | tee -a "${OUT_DIR}/logs/launch.log"
  exit "${STATUS}"
fi

echo "[$(date '+%F %T')] all visual tuning anchor configs completed" | tee -a "${OUT_DIR}/logs/launch.log"
