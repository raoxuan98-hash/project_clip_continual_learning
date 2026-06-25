#!/usr/bin/env bash
set -euo pipefail

if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" && -z "${CONDA_DEFAULT_ENV:-}" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-raoxuan}"
fi

ROOT="${ROOT:-/data1/open_datasets/X-TAIL}"
OUT_DIR="${OUT_DIR:-experiments/text_only_rgda_sweep_$(date +%Y%m%d_%H%M%S)}"
TASKS="${TASKS:-aircraft caltech101 dtd eurosat}"
SEED="${SEED:-42}"
GPU="${GPU:-3}"
BATCH_SIZE="${BATCH_SIZE:-64}"
ITERATIONS="${ITERATIONS:-800}"
ALPHAS="${ALPHAS:-0,0.05,0.1,0.2,0.5,1.0}"
NUM_CENTERS="${NUM_CENTERS:-4}"
ARTIFACT_NUM_CENTERS="${ARTIFACT_NUM_CENTERS:-1,4}"
if [[ -z "${VARIANTS:-}" ]]; then
  VARIANTS="sc:m=1:ft=0,mc4:m=4:ft=0,mc4ft200:m=4:ft=200:lr=0.01:source=gmm_sample"
fi
DRY_RUN="${DRY_RUN:-0}"

export CLIP_USE_SAFETENSORS="${CLIP_USE_SAFETENSORS:-0}"
export CLIP_LOCAL_FILES_ONLY="${CLIP_LOCAL_FILES_ONLY:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

RUN_NAME="text_only_no_fd_cd_lada_hybrid_seed${SEED}"
ASYNC_DIR="${OUT_DIR}/async/${RUN_NAME}"
RUN_DIR="${OUT_DIR}/runs/${RUN_NAME}"
SWEEP_DIR="${OUT_DIR}/sweep/${RUN_NAME}"

mkdir -p "${OUT_DIR}/logs" "${RUN_DIR}" "${SWEEP_DIR}" "${ASYNC_DIR}"

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
    echo "gpu: ${GPU}"
    echo "batch_size: ${BATCH_SIZE}"
    echo "iterations: ${ITERATIONS}"
    echo "num_centers: ${NUM_CENTERS}"
    echo "artifact_num_centers: ${ARTIFACT_NUM_CENTERS}"
    echo "alphas: ${ALPHAS}"
    echo "variants: ${VARIANTS}"
    echo "dry_run: ${DRY_RUN}"
    echo "CLIP_USE_SAFETENSORS: ${CLIP_USE_SAFETENSORS}"
    echo "CLIP_LOCAL_FILES_ONLY: ${CLIP_LOCAL_FILES_ONLY}"
    echo
    echo "[git]"
    git rev-parse HEAD 2>/dev/null || true
    git status --short 2>/dev/null || true
  } > "${OUT_DIR}/manifest.txt"
}

write_manifest

TRAIN_CMD=(
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
  --lora_type lora_vanilla
  --init_mode lora_vanilla
  --fd_weight 0.0
  --cd_weight 0.0
  --tune_vision_encoder false
  --tune_text_encoder true
  --text_classifier_mode lada_hybrid
  --seed "${SEED}"
  --experiment_name "${RUN_NAME}"
  --output_dir "${RUN_DIR}"
  --async_eval_dir "${ASYNC_DIR}"
  --save_step_artifacts
  --skip_inline_eval
)

SWEEP_CMD=(
  python scripts/evaluate_incremental_rgda_sweep_artifacts.py
  --async_eval_dir "${ASYNC_DIR}"
  --output_dir "${SWEEP_DIR}"
  --experiment_name "${RUN_NAME}"
  --device cuda:0
  --variants "${VARIANTS}"
  --alphas "${ALPHAS}"
)

{
  echo "[$(date '+%F %T')] start training ${RUN_NAME} gpu=${GPU}"
  printf 'CUDA_VISIBLE_DEVICES=%q ' "${GPU}"
  printf '%q ' "${TRAIN_CMD[@]}"
  echo
} | tee -a "${OUT_DIR}/logs/launch.log"

if [[ "${DRY_RUN}" == "1" ]]; then
  {
    echo "DRY_RUN training command:"
    printf 'CUDA_VISIBLE_DEVICES=%q ' "${GPU}"
    printf '%q ' "${TRAIN_CMD[@]}"
    echo
    echo "DRY_RUN sweep command:"
    printf 'CUDA_VISIBLE_DEVICES=%q ' "${GPU}"
    printf '%q ' "${SWEEP_CMD[@]}"
    echo
  } > "${OUT_DIR}/logs/${RUN_NAME}.log"
  exit 0
fi

CUDA_VISIBLE_DEVICES="${GPU}" "${TRAIN_CMD[@]}" > "${OUT_DIR}/logs/${RUN_NAME}_train.log" 2>&1
echo "[$(date '+%F %T')] done training ${RUN_NAME}" | tee -a "${OUT_DIR}/logs/launch.log"

{
  echo "[$(date '+%F %T')] start rgda sweep ${RUN_NAME} gpu=${GPU}"
  printf 'CUDA_VISIBLE_DEVICES=%q ' "${GPU}"
  printf '%q ' "${SWEEP_CMD[@]}"
  echo
} | tee -a "${OUT_DIR}/logs/launch.log"

CUDA_VISIBLE_DEVICES="${GPU}" "${SWEEP_CMD[@]}" > "${OUT_DIR}/logs/${RUN_NAME}_sweep.log" 2>&1
echo "[$(date '+%F %T')] done rgda sweep ${RUN_NAME}" | tee -a "${OUT_DIR}/logs/launch.log"
