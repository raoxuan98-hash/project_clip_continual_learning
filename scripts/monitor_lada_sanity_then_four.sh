#!/usr/bin/env bash
set -euo pipefail

SANITY_PID="${SANITY_PID:?SANITY_PID is required}"
SANITY_DIR="${SANITY_DIR:-experiments/lada_recipe_text_rgda_ablation_sanity_20260617}"
FOUR_DIR="${FOUR_DIR:-experiments/lada_recipe_text_rgda_ablation_four_20260617}"
ROOT="${ROOT:-/data1/open_datasets/X-TAIL}"
TASKS="${TASKS:-aircraft caltech101 dtd eurosat}"
SEEDS="${SEEDS:-42}"
GPUS="${GPUS:-3}"
PARALLEL_JOBS="${PARALLEL_JOBS:-1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
ITERATIONS="${ITERATIONS:-800}"
CONDA_ENV="${CONDA_ENV:-raoxuan}"
POLL_INTERVAL="${POLL_INTERVAL:-60}"
LOG="${LOG:-${SANITY_DIR}/four_task_monitor_robust.log}"
FAILURE_LOG="${FAILURE_LOG:-${SANITY_DIR}/runner.log}"

CONFIGS=(
  base_current_seed42
  recipe_current_seed42
  no_fd_cd_current_seed42
  no_fd_cd_lada_hybrid_seed42
  vanilla_no_fd_cd_current_seed42
  vanilla_no_fd_cd_lada_hybrid_seed42
  text_only_no_fd_cd_lada_hybrid_seed42
)
SUFFIXES=(zs rgda ens)

log() {
  echo "[$(date '+%F_%T')] $*"
}

result_exists() {
  local run_name="$1"
  local suffix="$2"
  [[ -f "${SANITY_DIR}/runs/${run_name}/${run_name}_${suffix}.json" ]] || \
    [[ -f "${SANITY_DIR}/runs/${run_name}/${run_name}_${suffix}_results.json" ]]
}

all_sanity_results_exist() {
  local run_name suffix
  for run_name in "${CONFIGS[@]}"; do
    for suffix in "${SUFFIXES[@]}"; do
      if ! result_exists "${run_name}" "${suffix}"; then
        return 1
      fi
    done
  done
  return 0
}

summarize_sanity_if_needed() {
  if [[ -f "${SANITY_DIR}/incremental_aggregate.md" ]]; then
    return 0
  fi
  if ! all_sanity_results_exist; then
    log "sanity aggregate missing and not all result files exist"
    return 1
  fi
  log "sanity aggregate missing; rebuilding summary from complete result files"
  python scripts/summarize_incremental_metrics.py \
    "${SANITY_DIR}/runs" \
    --output_csv "${SANITY_DIR}/incremental_summary.csv" \
    --output_markdown "${SANITY_DIR}/incremental_summary.md" \
    --aggregate_csv "${SANITY_DIR}/incremental_aggregate.csv" \
    --aggregate_markdown "${SANITY_DIR}/incremental_aggregate.md" \
    --expected_tasks "aircraft caltech101"
}

four_already_started() {
  if [[ -f "${FOUR_DIR}/manifest.txt" ]]; then
    return 0
  fi
  if [[ -d "${FOUR_DIR}/runs" ]] && find "${FOUR_DIR}/runs" -mindepth 1 -maxdepth 2 -print -quit | grep -q .; then
    return 0
  fi
  pgrep -af "main_incremental.py|run_lada_recipe_text_rgda_ablation.sh|OUT_DIR=${FOUR_DIR}" | \
    grep -F "${FOUR_DIR}" | \
    grep -Ev "pgrep -af|grep -F|monitor_lada_sanity_then_four" >/dev/null
}

main() {
  mkdir -p "$(dirname "${LOG}")"
  {
    log "robust monitor start sanity_pid=${SANITY_PID}"
    if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" && -z "${CONDA_DEFAULT_ENV:-}" ]]; then
      # shellcheck disable=SC1091
      source "${HOME}/miniconda3/etc/profile.d/conda.sh"
      conda activate "${CONDA_ENV}"
    fi

    while kill -0 "${SANITY_PID}" 2>/dev/null; do
      sleep "${POLL_INTERVAL}"
    done
    log "sanity process ended"

    if [[ -n "${FAILURE_LOG}" ]] && grep -qi "at least one ablation job failed" "${FAILURE_LOG}" 2>/dev/null; then
      log "sanity runner reported failure in ${FAILURE_LOG}; not launching four-task"
      exit 0
    fi

    if ! summarize_sanity_if_needed; then
      log "sanity did not pass summary gate; not launching four-task"
      exit 0
    fi

    if four_already_started; then
      log "four-task appears already started; not launching duplicate"
      exit 0
    fi

    log "launching four-task pilot into ${FOUR_DIR}"
    ROOT="${ROOT}" \
    OUT_DIR="${FOUR_DIR}" \
    TASKS="${TASKS}" \
    SEEDS="${SEEDS}" \
    GPUS="${GPUS}" \
    PARALLEL_JOBS="${PARALLEL_JOBS}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    ITERATIONS="${ITERATIONS}" \
    bash scripts/run_lada_recipe_text_rgda_ablation.sh
  } >> "${LOG}" 2>&1
}

main "$@"
