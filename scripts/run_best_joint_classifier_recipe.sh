#!/usr/bin/env bash
set -euo pipefail

# Reproduce the current best inference-side classifier recipe:
# LR-RGDA + multi-centroid + classifier fine-tuning + deterministic
# classifier features + GMM component-mean replay + low-weight ZS fusion.
#
# This wrapper intentionally leaves main_joint.py defaults unchanged and
# delegates the actual run/summary logic to run_joint_classifier_replay.sh.

export OUT_DIR="${OUT_DIR:-experiments/joint_classifier_best_recipe}"
export REPLAY_MODES="${REPLAY_MODES:-gmm_raw_mean}"
export CLASSIFIER_FEATURE_TRANSFORMS="${CLASSIFIER_FEATURE_TRANSFORMS:-test}"
export RUN_MEAN_ABLATION="${RUN_MEAN_ABLATION:-1}"
export ALPHA_SENSITIVITY="${ALPHA_SENSITIVITY:-0}"
export ENSEMBLE_ALPHA="${ENSEMBLE_ALPHA:-0.05}"
export LADA_ALPHA="${LADA_ALPHA:-0.05}"

bash scripts/run_joint_classifier_replay.sh "$@"
