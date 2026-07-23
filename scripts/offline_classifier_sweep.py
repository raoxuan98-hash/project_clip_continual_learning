#!/usr/bin/env python3
"""Offline LR-RGDA / LADA classifier sweep over Wave F step artifacts.

This script performs a read-only sweep over saved step artifacts:
  - Rebuilds the CLIP model from `model_state_dict`.
  - Rebuilds the inline LR-RGDA classifier from `lr_rgda.state_dict` (sanity A).
  - Rebuilds an unfitted LR-RGDA classifier from `rgda_stats_by_m[4]` (sanity B).
  - Refits LR-RGDA for num_centers in {1,2,4,8} and rgda_train_iter in {50,100,200,400,800}.
  - Rebuilds the LADA classifier from re-extracted training features (always),
    plus the artifact LADA state_dict when the run had LADA enabled.
  - Evaluates Zero-shot, LR-RGDA, Ensemble, LADA, and LADA+ZS on ALL tasks
    (seen + unseen) so the accuracy matrix is fully populated and the
    Transfer / Average / Last metrics (and (T+L)/2) are exact.

Usage:
    python scripts/offline_classifier_sweep.py \
        --artifact_dir experiments/paper_formal/WaveF_offline/async_e6/artifacts \
        --output_dir experiments/paper_formal/WaveF_offline/classifier_sweep \
        --device cuda:1 \
        --num_centers_list 1 2 4 8 \
        --rgda_train_iter_list 50 100 200 400 800 \
        --lada_alpha 0.05
"""

import argparse
import json
import logging
import os
import sys
import time
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main_incremental import (
    _build_text_classifier,
    _compute_dataset_balanced_global_cov,
    _fit_spherical_gmm_memory,
    _sample_spherical_gmm_memory,
)
from src.classifiers.gaussian_statistics import build_multi_center_stats_dict
from src.classifiers.lr_rgda_classifier import LRRGDAClassifier
from src.lada.lada_classifier import LADAClassifier
from src.utils.continual_metrics import ContinualLearningMetrics
from src.utils.data import get_transforms, get_xtail_trainloader
from src.utils.feature_extractor import extract_features
from src.utils.main_utils import combine_ensemble_logits

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline LR-RGDA / LADA classifier sweep over step artifacts"
    )
    parser.add_argument(
        "--artifact_dir",
        type=Path,
        required=True,
        help="Directory containing step_*/artifact.pt files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to write sweep JSON outputs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for evaluation (e.g. cuda:1). Defaults to cuda if available.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU index; used only if --device is not set.",
    )
    parser.add_argument(
        "--num_centers_list",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="List of num_centers (M) values to sweep.",
    )
    parser.add_argument(
        "--rgda_train_iter_list",
        type=int,
        nargs="+",
        default=[50, 100, 200, 400, 800],
        help="List of LR-RGDA fit iterations to sweep.",
    )
    parser.add_argument(
        "--lada_alpha",
        type=float,
        default=0.05,
        help="Ensemble weight for LADA+ZS.",
    )
    parser.add_argument(
        "--samples_per_class",
        type=int,
        default=16,
        help="Pseudo-samples per class for GMM replay fitting.",
    )
    parser.add_argument(
        "--rgda_fit_lr",
        type=float,
        default=0.01,
        help="Learning rate for LR-RGDA classifier-only fine-tuning.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=None,
        help="Batch size for evaluation feature extraction.",
    )
    parser.add_argument(
        "--rgda_eval_chunk_size",
        type=int,
        default=512,
        help="Chunk size for LR-RGDA forward during evaluation.",
    )
    parser.add_argument(
        "--skip_sanity_fit",
        action="store_true",
        help="Skip the M=4/iter=200 refit sanity comparison (not recommended).",
    )
    return parser.parse_args()


def torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def artifact_paths(artifact_dir):
    paths = sorted((artifact_dir).glob("step_*/artifact.pt"))
    if not paths:
        raise SystemExit(f"No artifacts found under {artifact_dir}")
    loaded = []
    for path in paths:
        artifact = torch_load(path, map_location="cpu")
        loaded.append((int(artifact["step_index"]), path, artifact))
    loaded.sort(key=lambda item: item[0])
    return loaded


def artifact_args(artifact, device):
    args = Namespace(**artifact["args"])
    args.device = device
    if not hasattr(args, "gpu"):
        args.gpu = 0
    # Provide sensible defaults for older artifacts that may lack these keys.
    if not hasattr(args, "rgda_rank"):
        args.rgda_rank = 32
    if not hasattr(args, "rgda_alpha1"):
        args.rgda_alpha1 = 0.2
    if not hasattr(args, "rgda_alpha2"):
        args.rgda_alpha2 = 2.0
    if not hasattr(args, "rgda_alpha3"):
        args.rgda_alpha3 = 0.5
    if not hasattr(args, "ensemble_normalize"):
        args.ensemble_normalize = "maxshift"
    if not hasattr(args, "alpha"):
        args.alpha = 0.5
    if not hasattr(args, "num_shots"):
        args.num_shots = 16
    if not hasattr(args, "classifier_feature_transform"):
        args.classifier_feature_transform = "test"
    if not hasattr(args, "gmm_fit_space"):
        args.gmm_fit_space = "raw"
    if not hasattr(args, "gmm_k"):
        args.gmm_k = 4
    if not hasattr(args, "rgda_fit_source"):
        args.rgda_fit_source = "gmm_sample"
    if not hasattr(args, "gaussian_samples_per_class"):
        args.gaussian_samples_per_class = 16
    if not hasattr(args, "lada_k"):
        args.lada_k = 16
    if not hasattr(args, "lada_beta"):
        args.lada_beta = 1.0
    if not hasattr(args, "lada_score_mode"):
        args.lada_score_mode = "exp_sum"
    if not hasattr(args, "num_workers"):
        args.num_workers = 4
    return args


def rebuild_model(artifact, device):
    """Rebuild adapted CLIP model and frozen reference model from artifact."""
    from src.models.clip import get_clip_model

    args = artifact_args(artifact, device)
    model, processor = get_clip_model(args, train_mode="lora")
    missing, unexpected = model.load_state_dict(artifact["model_state_dict"], strict=False)
    if missing:
        logging.warning("Missing keys loading model_state_dict: %s", missing)
    if unexpected:
        logging.warning("Unexpected keys loading model_state_dict: %s", unexpected)
    model.to(device)
    model.eval()

    frozen_model = None
    if getattr(args, "text_classifier_mode", "current") == "lada_hybrid":
        frozen_model, _ = get_clip_model(args, train_mode="frozen")
        frozen_model.to(device)
        frozen_model.eval()

    return args, model, processor, frozen_model


def _move_stats_dict_to(stats_dict, device="cpu"):
    """Move GaussianStatistics containers to requested device/dtype."""
    for stat in stats_dict.values():
        stat.mean = stat.mean.detach().to(device)
        stat.cov = stat.cov.detach().to(device)
        if getattr(stat, "L", None) is not None:
            stat.L = stat.L.detach().to(device)
    return stats_dict


def _move_center_means_to(center_means, device="cpu"):
    if center_means is None:
        return None
    return {int(cid): centers.detach().to(device) for cid, centers in center_means.items()}


def _get_rgda_stats_entry(artifact, requested_m):
    stats_by_m = artifact.get("rgda_stats_by_m")
    if not stats_by_m:
        return None
    for key in (requested_m, str(requested_m)):
        if key in stats_by_m:
            return stats_by_m[key]
    return None


def extract_training_features_for_step(
    artifact, run_args, model, device, up_to_step=None
):
    """Extract normalized training features for tasks 1..up_to_step using current model.

    Returns:
        all_features: Tensor[N, D] (normalized)
        all_labels: Tensor[N] (global label space)
        history_class_names: list of list of class names per dataset
        per_task_features: dict {task_name: (features, labels)}
    """
    task_names = list(artifact["task_names"])
    dataset_label_offsets = artifact["dataset_label_offsets"]
    if up_to_step is None:
        up_to_step = int(artifact["step_index"])

    history_class_names = []
    all_features = []
    all_labels = []
    per_task = {}

    for step_idx in range(up_to_step + 1):
        d_name = task_names[step_idx]
        train_transform, test_transform = get_transforms(d_name)
        tr_loader, tr4update, _, c_names = get_xtail_trainloader(
            root=run_args.root,
            dataset_name=d_name,
            transform_train=train_transform,
            transform_test=test_transform,
            num_shots=run_args.num_shots,
            batch_size=run_args.batch_size,
            num_workers=run_args.num_workers,
        )
        feature_loader = tr_loader if run_args.classifier_feature_transform == "train" else tr4update
        raw_features, labels = extract_features(model, feature_loader, device, normalize=False)
        features = F.normalize(raw_features.float(), dim=-1)
        global_labels = labels.long() + int(dataset_label_offsets[d_name])

        all_features.append(features.detach().cpu())
        all_labels.append(global_labels.detach().cpu())
        history_class_names.append(list(c_names))
        per_task[d_name] = (features.detach().cpu(), global_labels.detach().cpu())

    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    return all_features, all_labels, history_class_names, per_task


def _build_rgda_from_compact_stats(artifact, m, device, run_args):
    """Build LR-RGDA classifier from rgda_stats_by_m if available."""
    entry = _get_rgda_stats_entry(artifact, m)
    if entry is None:
        return None, None, None

    stats_dict = dict(entry["global_stats_dict"])
    center_means = entry.get("global_center_means")
    global_cov = entry.get("dataset_balanced_global_cov")

    stats_dict = _move_stats_dict_to(stats_dict, device="cpu")
    center_means = _move_center_means_to(center_means, device="cpu")
    if global_cov is not None:
        global_cov = global_cov.detach().cpu()

    if m == 1:
        center_means = None

    classifier = LRRGDAClassifier(
        stats_dict=stats_dict,
        device=device,
        rank=run_args.rgda_rank,
        qda_reg_alpha1=run_args.rgda_alpha1,
        qda_reg_alpha2=run_args.rgda_alpha2,
        qda_reg_alpha3=run_args.rgda_alpha3,
        temperature=1.0,
        M=m,
        center_means=center_means,
        global_cov=global_cov,
    )
    return classifier, stats_dict, center_means


def _build_rgda_from_reextracted_features(
    artifact, m, device, run_args, model, all_features, all_labels, history_class_names
):
    """Rebuild LR-RGDA stats by re-extracting training features with current model."""
    stats_dict, center_means = build_multi_center_stats_dict(
        all_features, all_labels, M=m
    )
    if m == 1:
        center_means = None
    global_cov = _compute_dataset_balanced_global_cov(stats_dict, history_class_names)

    classifier = LRRGDAClassifier(
        stats_dict=stats_dict,
        device=device,
        rank=run_args.rgda_rank,
        qda_reg_alpha1=run_args.rgda_alpha1,
        qda_reg_alpha2=run_args.rgda_alpha2,
        qda_reg_alpha3=run_args.rgda_alpha3,
        temperature=1.0,
        M=m,
        center_means=center_means,
        global_cov=global_cov,
    )
    return classifier, stats_dict, center_means, global_cov


def _fit_rgda_with_gmm_sample(classifier, gmm_memory, samples_per_class, device, iterations, lr):
    """Fit LR-RGDA classifier with GMM-sample pseudo-features."""
    if gmm_memory is None or not gmm_memory:
        logging.warning("GMM memory unavailable; skipping LR-RGDA fit.")
        return False
    if iterations <= 0:
        return False
    pseudo_features, pseudo_labels = _sample_spherical_gmm_memory(
        gmm_memory, samples_per_class, "gmm_sample", device
    )
    classifier.fit(
        pseudo_features,
        pseudo_labels,
        iterations=iterations,
        lr=lr,
        verbose=False,
    )
    return True


def build_rgda_classifier(
    artifact,
    run_args,
    m,
    iterations,
    device,
    model,
    all_features,
    all_labels,
    history_class_names,
    samples_per_class,
    fit_lr,
):
    """Build LR-RGDA classifier for a given (M, iterations) combination."""
    compact = _build_rgda_from_compact_stats(artifact, m, device, run_args)
    if compact[0] is not None:
        classifier, stats_dict, center_means = compact
    else:
        if all_features is None:
            raise ValueError(
                f"Artifact has no compact stats for M={m} and no re-extracted features were provided."
            )
        classifier, stats_dict, center_means, global_cov = _build_rgda_from_reextracted_features(
            artifact, m, device, run_args, model, all_features, all_labels, history_class_names
        )

    if iterations > 0:
        _fit_rgda_with_gmm_sample(
            classifier,
            artifact.get("gmm_memory"),
            samples_per_class,
            device,
            iterations,
            fit_lr,
        )
    classifier.classifier.eval()
    return classifier


def build_inline_rgda_classifier(artifact, device):
    """Build LR-RGDA classifier directly from saved lr_rgda.state_dict."""
    config = artifact["lr_rgda"]["config"]
    classifier = LRRGDAClassifier(
        stats_dict=artifact["global_stats_dict"],
        device=device,
        rank=config["rank"],
        qda_reg_alpha1=config["qda_reg_alpha1"],
        qda_reg_alpha2=config["qda_reg_alpha2"],
        qda_reg_alpha3=config["qda_reg_alpha3"],
        temperature=config.get("temperature", 1.0),
        M=config.get("M", 1),
        center_means=artifact.get("global_center_means"),
        global_cov=artifact.get("dataset_balanced_global_cov"),
    )
    classifier.classifier.load_state_dict(artifact["lr_rgda"]["state_dict"])
    classifier.classifier.to(device)
    classifier.classifier.eval()
    return classifier


def build_lada_from_state_dict(artifact, device):
    """Build LADA classifier from artifact state dict."""
    lada_entry = artifact.get("lada")
    if lada_entry is None:
        return None
    classifier = LADAClassifier(
        feature_dim=lada_entry["feature_dim"],
        beta=lada_entry.get("beta", 1.0),
        score_mode=lada_entry.get("score_mode", "exp_sum"),
    )
    classifier.load_state_dict(lada_entry["state_dict"])
    classifier.to(device)
    classifier.eval()
    return classifier


def build_lada_from_data(all_features, all_labels, feature_dim, device):
    """Rebuild LADA classifier from accumulated training features."""
    classifier = LADAClassifier(
        feature_dim=feature_dim,
        beta=1.0,
        score_mode="exp_sum",
    )
    features = all_features.to(device)
    labels = all_labels.to(device)
    classifier.build_from_data(features, labels, k=16)
    classifier.to(device)
    classifier.eval()
    return classifier


def _rgda_forward_chunked(classifier, features, chunk_size):
    chunk_size = int(chunk_size or 0)
    if chunk_size <= 0 or features.shape[0] <= chunk_size:
        return classifier.forward(features)
    logits = []
    for start in range(0, features.shape[0], chunk_size):
        end = min(start + chunk_size, features.shape[0])
        logits.append(classifier.forward(features[start:end]))
    return torch.cat(logits, dim=0)


def _mean_accuracy_difference(scores_a, scores_b):
    """Compute mean absolute difference between two per-task accuracy dicts."""
    keys = set(scores_a.keys()) & set(scores_b.keys())
    if not keys:
        return float("nan")
    diffs = [abs(scores_a[k] - scores_b[k]) for k in keys]
    return sum(diffs) / len(diffs)


def evaluate_step(
    artifact,
    run_args,
    model,
    processor,
    frozen_model,
    device,
    num_centers_list,
    rgda_train_iter_list,
    lada_alpha,
    samples_per_class,
    rgda_fit_lr,
    eval_batch_size,
    rgda_eval_chunk_size,
    skip_sanity_fit,
):
    """Evaluate one step artifact across all classifier variants.

    Returns:
        step_index, task_name, per_method_scores dict, sanity dict
    """
    task_names = list(artifact["task_names"])
    current_step = int(artifact["step_index"])
    current_num_classes = int(artifact["current_num_classes"])
    dataset_label_offsets = artifact["dataset_label_offsets"]

    # Build text classifier (lada_hybrid or current).
    zeroshot_classifier = _build_text_classifier(
        run_args,
        model,
        processor,
        frozen_model,
        artifact["global_class_names"],
        artifact.get("cached_seen_text_features"),
        device,
    )

    # Pre-extract training features for LADA rebuild and for M values not in artifact.
    # We always re-extract once per step; it is needed for LADA-from-data and for M=2,8.
    t0 = time.time()
    all_train_features, all_train_labels, history_class_names, per_task_train = (
        extract_training_features_for_step(artifact, run_args, model, device)
    )
    logging.info(
        "Re-extracted training features for %d seen tasks in %.1fs",
        current_step + 1,
        time.time() - t0,
    )

    # Build LADA classifiers. The from-data rebuild is always possible because
    # training features are re-extracted above; the artifact state_dict (when
    # LADA was enabled during the run) is kept as an additional reference.
    lada_state = build_lada_from_state_dict(artifact, device)
    lada_rebuilt = build_lada_from_data(
        all_train_features, all_train_labels, all_train_features.shape[1], device
    )

    # Build inline LR-RGDA (saved state_dict) for sanity reference.
    inline_config = artifact["lr_rgda"]["config"]
    inline_m = int(inline_config.get("M", 1))
    inline_iter = int(getattr(run_args, "rgda_train_iter", 0))
    inline_classifier = build_inline_rgda_classifier(artifact, device)

    # Build unfitted LR-RGDA from rgda_stats_by_m[inline_m] for sanity comparison.
    unfitted_classifier, _, _ = _build_rgda_from_compact_stats(
        artifact, inline_m, device, run_args
    )
    if unfitted_classifier is None:
        logging.warning(
            "No compact stats for inline M=%d; skipping A/B sanity check.", inline_m
        )

    # Determine which M values need refit (not present in compact artifact stats).
    available_ms = set()
    stats_by_m = artifact.get("rgda_stats_by_m") or {}
    for key in stats_by_m:
        try:
            available_ms.add(int(key))
        except ValueError:
            pass

    # Sweep over (M, iter) combinations. Always rebuild/refit so that the
    # M=4/iter=200 offline refit can be compared against the inline state_dict
    # version (acceptance criterion: per-cell difference <= 0.5%).
    sweep_classifiers = {}
    for m in num_centers_list:
        for iters in rgda_train_iter_list:
            key = f"rgda_m{m}_iter{iters}"
            classifier = build_rgda_classifier(
                artifact,
                run_args,
                m,
                iters,
                device,
                model,
                all_train_features,
                all_train_labels,
                history_class_names,
                samples_per_class,
                rgda_fit_lr,
            )
            sweep_classifiers[key] = (classifier, m, iters)

    # Prepare method score trackers.
    method_scores = {}

    # Evaluate every method on ALL tasks (seen + unseen) in a single pass.
    # Supervised heads (LR-RGDA / LADA) only cover seen classes, so they score
    # 0 on unseen tasks by construction; ZS-based scores stay meaningful on
    # unseen tasks, which is what makes the Transfer column well-defined.
    for step_idx in range(len(task_names)):
        d_name = task_names[step_idx]
        eval_bs = eval_batch_size if eval_batch_size is not None else run_args.batch_size
        _, test_transform = get_transforms(d_name)
        _, _, te_loader, c_names = get_xtail_trainloader(
            root=run_args.root,
            dataset_name=d_name,
            transform_train=None,
            transform_test=test_transform,
            num_shots=run_args.num_shots,
            batch_size=eval_bs,
            num_workers=run_args.num_workers,
        )
        features, labels = extract_features(model, te_loader, device)
        features = F.normalize(features.float(), dim=-1).to(device)
        labels = (labels + dataset_label_offsets[d_name]).long().to(device)

        with torch.no_grad():
            # Zero-shot.
            zs_logits = features @ zeroshot_classifier
            zs_preds = zs_logits.argmax(dim=1)
            zs_acc = zs_preds.eq(labels).float().mean().item()
            method_scores.setdefault("zero_shot", {})[d_name] = zs_acc

            # Inline LR-RGDA sanity reference.
            inline_logits = _rgda_forward_chunked(
                inline_classifier, features, rgda_eval_chunk_size
            )
            inline_acc = inline_logits.argmax(dim=1).eq(labels).float().mean().item()
            method_scores.setdefault(f"inline_m{inline_m}_iter{inline_iter}", {})[d_name] = inline_acc

            # Ensemble with inline LR-RGDA.
            ens_logits = combine_ensemble_logits(
                zs_logits,
                inline_logits,
                current_num_classes,
                run_args.alpha,
                run_args.ensemble_normalize,
            )
            ens_acc = ens_logits.argmax(dim=1).eq(labels).float().mean().item()
            method_scores.setdefault(f"inline_ens_alpha{run_args.alpha}", {})[d_name] = ens_acc

            # Unfitted sanity classifier.
            if unfitted_classifier is not None:
                unfitted_logits = _rgda_forward_chunked(
                    unfitted_classifier, features, rgda_eval_chunk_size
                )
                unfitted_acc = unfitted_logits.argmax(dim=1).eq(labels).float().mean().item()
                method_scores.setdefault(f"unfitted_m{inline_m}", {})[d_name] = unfitted_acc

            # LADA from artifact state_dict (only when the run had LADA enabled).
            if lada_state is not None:
                lada_state_logits = lada_state(features)
                lada_state_acc = lada_state_logits.argmax(dim=1).eq(labels).float().mean().item()
                method_scores.setdefault("lada_state", {})[d_name] = lada_state_acc

                lada_state_zs_logits = combine_ensemble_logits(
                    zs_logits,
                    lada_state_logits,
                    current_num_classes,
                    lada_alpha,
                    run_args.ensemble_normalize,
                )
                lada_state_zs_acc = lada_state_zs_logits.argmax(dim=1).eq(labels).float().mean().item()
                method_scores.setdefault(f"lada_state_zs_alpha{lada_alpha}", {})[d_name] = lada_state_zs_acc

            # LADA rebuilt from re-extracted training features (always available).
            lada_logits = lada_rebuilt(features)
            lada_acc = lada_logits.argmax(dim=1).eq(labels).float().mean().item()
            method_scores.setdefault("lada", {})[d_name] = lada_acc

            lada_zs_logits = combine_ensemble_logits(
                zs_logits,
                lada_logits,
                current_num_classes,
                lada_alpha,
                run_args.ensemble_normalize,
            )
            lada_zs_acc = lada_zs_logits.argmax(dim=1).eq(labels).float().mean().item()
            method_scores.setdefault(f"lada_zs_alpha{lada_alpha}", {})[d_name] = lada_zs_acc

            # Sweep classifiers + their ZS ensembles.
            for key, (classifier, m, iters) in sweep_classifiers.items():
                rgda_logits = _rgda_forward_chunked(
                    classifier, features, rgda_eval_chunk_size
                )
                rgda_acc = rgda_logits.argmax(dim=1).eq(labels).float().mean().item()
                method_scores.setdefault(key, {})[d_name] = rgda_acc

                sweep_ens_logits = combine_ensemble_logits(
                    zs_logits,
                    rgda_logits,
                    current_num_classes,
                    run_args.alpha,
                    run_args.ensemble_normalize,
                )
                sweep_ens_acc = sweep_ens_logits.argmax(dim=1).eq(labels).float().mean().item()
                method_scores.setdefault(f"ens_m{m}_iter{iters}_alpha{run_args.alpha}", {})[d_name] = sweep_ens_acc

    # Sanity checks (compared on seen tasks only, where both sides are
    # meaningful; unseen cells are 0=0 and would dilute the difference).
    seen_task_names = set(task_names[: current_step + 1])

    def _seen_only(scores):
        return {k: v for k, v in scores.items() if k in seen_task_names}

    sanity = {}
    if unfitted_classifier is not None:
        sanity["inline_vs_unfitted_diff"] = _mean_accuracy_difference(
            _seen_only(method_scores[f"inline_m{inline_m}_iter{inline_iter}"]),
            _seen_only(method_scores[f"unfitted_m{inline_m}"]),
        )
    if lada_state is not None:
        sanity["lada_state_vs_rebuilt_diff"] = _mean_accuracy_difference(
            _seen_only(method_scores["lada_state"]),
            _seen_only(method_scores["lada"]),
        )

    # Compare M=4/iter=200 refit vs inline (acceptance criterion).
    if not skip_sanity_fit and inline_m == 4 and inline_iter == 200:
        refit_key = "rgda_m4_iter200"
        if refit_key in method_scores:
            sanity["inline_vs_refit_m4_iter200_diff"] = _mean_accuracy_difference(
                _seen_only(method_scores[f"inline_m{inline_m}_iter{inline_iter}"]),
                _seen_only(method_scores[refit_key]),
            )

    return current_step, artifact["task_name"], method_scores, sanity


def compute_all_method_metrics(method_history, task_names):
    """Compute Transfer / Average / Last for all methods from full history.

    Args:
        method_history: {method_name: [per_dataset_row_step0, ...]} where each
            row maps dataset name -> accuracy (fraction) at that step. Rows
            cover all tasks (seen + unseen), so the accuracy matrix is fully
            populated and Transfer is well-defined.
    """
    results = {}
    for method_name, rows in method_history.items():
        tracker = ContinualLearningMetrics(task_names)
        for step_idx, per_dataset in enumerate(rows):
            tracker.update(step_idx, per_dataset)
        summary = tracker.get_summary()
        results[method_name] = {
            "transfer": summary["transfer"],
            "average": summary["average"],
            "last": summary["last"],
            "tl_mean": (summary["transfer"] + summary["last"]) / 2.0,
            "per_dataset": rows[-1] if rows else {},
        }
    return results


def _format_sanity(sanity):
    parts = []
    for k, v in sanity.items():
        if v is None or (isinstance(v, float) and v != v):
            parts.append(f"{k}=N/A")
        else:
            parts.append(f"{k}={v*100:.2f}%")
    return " | ".join(parts)


def main():
    args = parse_args()
    if args.device is None:
        if args.gpu is not None and torch.cuda.is_available():
            args.device = f"cuda:{args.gpu}"
        else:
            args.device = "cuda" if torch.cuda.is_available() else "cpu"

    logging.info("Using device: %s", args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = artifact_paths(args.artifact_dir)
    logging.info("Found %d artifacts to sweep.", len(artifacts))

    task_names = list(artifacts[0][2]["task_names"])
    summary_rows = []
    method_history = {}

    # Resume support: load per-step results that already exist in output_dir
    # so a rerun only executes the missing steps (e.g. after a crash).
    completed_steps = set()
    for prev_path in sorted(args.output_dir.glob("step_*_sweep.json")):
        try:
            with prev_path.open("r", encoding="utf-8") as handle:
                prev = json.load(handle)
            prev_step = int(prev["step_index"])
        except (ValueError, KeyError, json.JSONDecodeError):
            continue
        completed_steps.add(prev_step)
        for method_name, metrics in prev["method_metrics"].items():
            per_dataset = metrics.get("per_dataset")
            if per_dataset is not None:
                method_history.setdefault(method_name, []).append(per_dataset)
        summary_rows.append(
            {
                "step_index": prev_step,
                "task_name": prev["task_name"],
                "method_metrics": prev["method_metrics"],
                "sanity": prev.get("sanity", {}),
            }
        )
    if completed_steps:
        logging.info(
            "Resuming: %d steps already completed (%s); skipping them.",
            len(completed_steps),
            sorted(completed_steps),
        )

    for step_index, artifact_path, artifact in artifacts:
        if step_index in completed_steps:
            continue
        logging.info(
            "=== Sweeping step %d (%s): %s ===",
            step_index + 1,
            artifact["task_name"],
            artifact_path,
        )
        t0 = time.time()
        run_args, model, processor, frozen_model = rebuild_model(artifact, args.device)

        step_idx, task_name, method_scores, sanity = evaluate_step(
            artifact,
            run_args,
            model,
            processor,
            frozen_model,
            args.device,
            args.num_centers_list,
            args.rgda_train_iter_list,
            args.lada_alpha,
            args.samples_per_class,
            args.rgda_fit_lr,
            args.eval_batch_size,
            args.rgda_eval_chunk_size,
            args.skip_sanity_fit,
        )

        for method_name, per_dataset in method_scores.items():
            method_history.setdefault(method_name, []).append(per_dataset)
        method_metrics = compute_all_method_metrics(method_history, task_names)

        # Print key numbers.
        print("\n" + "-" * 102)
        print(
            f"Step {step_idx + 1:02d} ({task_name}) | Sanity: {_format_sanity(sanity)}"
        )
        print("-" * 102)
        header = f"{'Method':<35} {'Transfer':>10} {'Average':>10} {'Last':>10} {'(T+L)/2':>10}"
        print(header)
        print("-" * 102)
        for method_name in sorted(method_metrics.keys()):
            metrics = method_metrics[method_name]
            print(
                f"{method_name:<35} "
                f"{metrics['transfer']:10.2f} {metrics['average']:10.2f} "
                f"{metrics['last']:10.2f} {metrics['tl_mean']:10.2f}"
            )
        print("-" * 102)
        logging.info("Step %d sweep finished in %.1fs", step_index + 1, time.time() - t0)

        # Write per-step JSON.
        step_json = {
            "schema_version": 1,
            "step_index": step_idx,
            "task_name": task_name,
            "artifact_path": str(artifact_path.resolve()),
            "device": args.device,
            "num_centers_list": args.num_centers_list,
            "rgda_train_iter_list": args.rgda_train_iter_list,
            "lada_alpha": args.lada_alpha,
            "samples_per_class": args.samples_per_class,
            "rgda_fit_lr": args.rgda_fit_lr,
            "ensemble_alpha": run_args.alpha,
            "ensemble_normalize": run_args.ensemble_normalize,
            "sanity": {k: (None if (isinstance(v, float) and v != v) else v) for k, v in sanity.items()},
            "method_metrics": method_metrics,
        }
        step_out_path = (
            args.output_dir / f"step_{step_idx + 1:02d}_{task_name}_sweep.json"
        )
        tmp_path = step_out_path.with_suffix(step_out_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(step_json, handle, indent=2)
        os.replace(tmp_path, step_out_path)
        logging.info("Wrote per-step results to %s", step_out_path)

        summary_rows.append(
            {
                "step_index": step_idx,
                "task_name": task_name,
                "method_metrics": method_metrics,
                "sanity": sanity,
            }
        )

        # Clean up GPU memory.
        del model, processor, frozen_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Build and write summary JSON.
    summary = {
        "schema_version": 1,
        "artifact_dir": str(args.artifact_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "device": args.device,
        "num_centers_list": args.num_centers_list,
        "rgda_train_iter_list": args.rgda_train_iter_list,
        "lada_alpha": args.lada_alpha,
        "task_names": task_names,
        "steps": summary_rows,
    }

    # Reorganize summary by (M, iter) across steps.
    sweep_summary = {}
    for row in summary_rows:
        step_idx = row["step_index"]
        for method_name, metrics in row["method_metrics"].items():
            sweep_summary.setdefault(method_name, []).append(
                {
                    "step_index": step_idx,
                    "task_name": row["task_name"],
                    "transfer": metrics["transfer"],
                    "average": metrics["average"],
                    "last": metrics["last"],
                    "tl_mean": metrics["tl_mean"],
                }
            )
    summary["sweep_summary_by_method"] = sweep_summary

    summary_path = args.output_dir / "sweep_summary.json"
    tmp_path = summary_path.with_suffix(summary_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    os.replace(tmp_path, summary_path)
    logging.info("Wrote sweep summary to %s", summary_path)
    print(f"\nSweep complete. Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
