#!/usr/bin/env bash
set -euo pipefail

if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" && -z "${CONDA_DEFAULT_ENV:-}" ]]; then
  # Convenience for remote nohup launches. If an environment is already active,
  # leave it untouched.
  # shellcheck disable=SC1091
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-raoxuan}"
fi

GPU="${GPU:-0}"
ROOT="${ROOT:-/data1/open_datasets/X-TAIL}"
SEEDS="${SEEDS:-42 43 44}"
OUT_DIR="${OUT_DIR:-experiments/joint_classifier_replay}"
RUN_MEAN_ABLATION="${RUN_MEAN_ABLATION:-1}"
CLASSIFIER_FEATURE_TRANSFORMS="${CLASSIFIER_FEATURE_TRANSFORMS:-test}"
REPLAY_MODES="${REPLAY_MODES:-real gmm_raw gmm_sphere gmm_raw_mean}"
ALPHA_SENSITIVITY="${ALPHA_SENSITIVITY:-0}"
ENSEMBLE_ALPHA="${ENSEMBLE_ALPHA:-0.5}"
LADA_ALPHA="${LADA_ALPHA:-1.0}"
DRY_RUN="${DRY_RUN:-0}"

export CLIP_USE_SAFETENSORS="${CLIP_USE_SAFETENSORS:-0}"
export CLIP_LOCAL_FILES_ONLY="${CLIP_LOCAL_FILES_ONLY:-1}"

COMMON_ARGS=(
  --id_datasets ALL
  --ood_datasets
  --root "${ROOT}"
  --gpu "${GPU}"
  --iterations 0
  --num_shots 16
  --batch_size 64
  --enable_lada
  --lada_k 16
  --lada_beta 1.0
  --lada_train_iter 200
  --lada_train_lr 0.01
  --num_centers 4
  --rgda_rank 32
  --rgda_train_iter 200
  --rgda_train_lr 0.01
  --alpha "${ENSEMBLE_ALPHA}"
  --lada_alpha "${LADA_ALPHA}"
  --gmm_k 4
  --gaussian_samples_per_class 16
  --output_dir "${OUT_DIR}"
)

if [[ "${ALPHA_SENSITIVITY}" == "1" ]]; then
  COMMON_ARGS+=(--alpha_sensitivity)
fi

mkdir -p "${OUT_DIR}/logs"

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
    echo "gpu: ${GPU}"
    echo "seeds: ${SEEDS}"
    echo "classifier_feature_transforms: ${CLASSIFIER_FEATURE_TRANSFORMS}"
    echo "replay_modes: ${REPLAY_MODES}"
    echo "run_mean_ablation: ${RUN_MEAN_ABLATION}"
    echo "alpha_sensitivity: ${ALPHA_SENSITIVITY}"
    echo "ensemble_alpha: ${ENSEMBLE_ALPHA}"
    echo "lada_alpha: ${LADA_ALPHA}"
    echo "dry_run: ${DRY_RUN}"
    echo "CLIP_USE_SAFETENSORS: ${CLIP_USE_SAFETENSORS}"
    echo "CLIP_LOCAL_FILES_ONLY: ${CLIP_LOCAL_FILES_ONLY}"
    echo
    echo "[git]"
    git rev-parse HEAD 2>/dev/null || true
    git status --short 2>/dev/null || true
    echo
    echo "[common_args]"
    printf '%q ' "${COMMON_ARGS[@]}"
    echo
  } > "${manifest}"
}

write_manifest

run_one() {
  local name="$1"
  local seed="$2"
  local feature_transform="$3"
  shift 3
  local experiment="${feature_transform}_${name}"
  local log="${OUT_DIR}/logs/${experiment}_seed${seed}.log"
  local result="${OUT_DIR}/${experiment}_seed${seed}.json"
  if [[ -f "${result}" ]]; then
    echo "[$(date '+%F %T')] skip ${experiment} seed=${seed}; result exists: ${result}" | tee -a "${OUT_DIR}/logs/launch.log"
    return
  fi
  echo "[$(date '+%F %T')] start ${experiment} seed=${seed}" | tee -a "${OUT_DIR}/logs/launch.log"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY_RUN python main_joint.py ' > "${log}"
    printf '%q ' "${COMMON_ARGS[@]}" >> "${log}"
    printf '%q ' \
      --seed "${seed}" \
      --classifier_feature_transform "${feature_transform}" \
      --experiment_name "${experiment}" \
      "$@" \
      >> "${log}"
    echo >> "${log}"
    echo "[$(date '+%F %T')] dry-run ${experiment} seed=${seed}" | tee -a "${OUT_DIR}/logs/launch.log"
    return
  fi
  python main_joint.py \
    "${COMMON_ARGS[@]}" \
    --seed "${seed}" \
    --classifier_feature_transform "${feature_transform}" \
    --experiment_name "${experiment}" \
    "$@" \
    > "${log}" 2>&1
  echo "[$(date '+%F %T')] done ${experiment} seed=${seed}" | tee -a "${OUT_DIR}/logs/launch.log"
}

for feature_transform in ${CLASSIFIER_FEATURE_TRANSFORMS}; do
  for seed in ${SEEDS}; do
    for replay_mode in ${REPLAY_MODES}; do
      case "${replay_mode}" in
        real)
          run_one real "${seed}" "${feature_transform}"
          ;;
        gmm_raw)
          run_one gmm_raw "${seed}" "${feature_transform}" \
            --use_gaussian_features --gmm_fit_space raw --gmm_sample_mode sample
          ;;
        gmm_sphere)
          run_one gmm_sphere "${seed}" "${feature_transform}" \
            --use_gaussian_features --gmm_fit_space sphere --gmm_sample_mode sample
          ;;
        gmm_raw_mean)
          if [[ "${RUN_MEAN_ABLATION}" == "1" ]]; then
            run_one gmm_raw_mean "${seed}" "${feature_transform}" \
              --use_gaussian_features --gmm_fit_space raw --gmm_sample_mode mean
          fi
          ;;
        *)
          echo "Unknown replay mode: ${replay_mode}" >&2
          exit 1
          ;;
      esac
    done
  done
done

python scripts/summarize_joint_classifier_replay.py \
  --input_dir "${OUT_DIR}" \
  --output_csv "${OUT_DIR}/summary.csv" \
  --output_markdown "${OUT_DIR}/summary.md" \
  --expected_seeds "${SEEDS}" \
  | tee "${OUT_DIR}/logs/summary.log"
