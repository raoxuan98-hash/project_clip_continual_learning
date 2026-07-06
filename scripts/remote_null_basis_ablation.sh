#!/usr/bin/env bash
# Upload local project to remote server and launch Phase 1 null/basis ablation.
#
# Environment (override as needed):
#   REMOTE            remote host, default raoxuan@10.20.34.30
#   REMOTE_DIR        remote project path, default /home/raoxuan/projects/project_clip_continual_learning
#   LOCAL_DIR         local project path, default current directory
#   OUT_DIR           experiment output dir on remote, default experiments/null_basis_ablation_phase1
#   MASTER_LOG        master log on remote, default ${OUT_DIR}/logs/remote_master.log
#
# Usage:
#   bash scripts/remote_null_basis_ablation.sh sync
#   bash scripts/remote_null_basis_ablation.sh launch
#   bash scripts/remote_null_basis_ablation.sh status

set -euo pipefail

REMOTE="${REMOTE:-raoxuan@10.20.34.30}"
REMOTE_DIR="${REMOTE_DIR:-/home/raoxuan/projects/project_clip_continual_learning}"
LOCAL_DIR="${LOCAL_DIR:-$(pwd)}"
OUT_DIR="${OUT_DIR:-experiments/null_basis_ablation_phase1}"
MASTER_LOG="${MASTER_LOG:-${OUT_DIR}/logs/remote_master.log}"

SSH_OPTS=(
  -o ConnectTimeout=8
  -o BatchMode=yes
)

remote_run() {
  ssh "${SSH_OPTS[@]}" "${REMOTE}" "$@"
}

usage() {
  cat <<EOF
Usage: $0 {sync|launch|status}

Environment:
  REMOTE=${REMOTE}
  REMOTE_DIR=${REMOTE_DIR}
  LOCAL_DIR=${LOCAL_DIR}
  OUT_DIR=${OUT_DIR}
  MASTER_LOG=${MASTER_LOG}
EOF
}

sync() {
  echo "[sync] ${LOCAL_DIR} -> ${REMOTE}:${REMOTE_DIR}"
  rsync -avz \
    --exclude='.git' \
    --exclude='.venv' \
    --exclude='venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='experiments' \
    --exclude='cache' \
    --exclude='data' \
    --exclude='figures' \
    --exclude='*.egg-info' \
    --exclude='.history' \
    --exclude='.DS_Store' \
    "${LOCAL_DIR}/" "${REMOTE}:${REMOTE_DIR}/"
  echo "[sync] done"
}

launch() {
  remote_run "
    cd '${REMOTE_DIR}' &&
    bash -n scripts/run_null_basis_phase1.sh &&
    OUT_DIR='${OUT_DIR}' nohup bash scripts/run_null_basis_phase1.sh > '${MASTER_LOG}' 2>&1 &
    sleep 3
    pgrep -af run_null_basis_phase1 || true
  "
  echo "[launch] master launched; remote log: ${REMOTE}:${REMOTE_DIR}/${MASTER_LOG}"
}

status() {
  remote_run "
    cd '${REMOTE_DIR}' &&
    echo '--- running processes ---' &&
    pgrep -af run_null_basis_phase1 || true &&
    echo '--- result files ---' &&
    find '${OUT_DIR}' -maxdepth 1 -name '*_results.json' -printf '%f\n' 2>/dev/null | sort || true &&
    echo '--- recent master log ---' &&
    tail -40 '${MASTER_LOG}' 2>/dev/null || true
  "
}

case "${1:-}" in
  sync)
    sync
    ;;
  launch)
    launch
    ;;
  status)
    status
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
