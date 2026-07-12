#!/usr/bin/env python3
"""Evaluate multiple LR-RGDA variants and alpha values from step artifacts."""

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main_incremental import _build_text_classifier
from src.classifiers.gaussian_statistics import build_multi_center_stats_dict
from src.classifiers.lr_rgda_classifier import LRRGDAClassifier
from src.lada.lada_classifier import LADAClassifier
from src.utils.continual_metrics import ContinualLearningMetrics
from src.utils.feature_extractor import extract_features
from src.utils.main_utils import combine_ensemble_logits, get_zeroshot_classifier
from src.utils.data import get_transforms, get_xtail_trainloader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--async_eval_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--experiment_name", default="rgda_sweep")
    parser.add_argument("--device", default=None)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument(
        "--variants",
        default="sc:m=1:ft=0,mc4:m=4:ft=0,mc4ft200:m=4:ft=200:lr=0.01",
        help=(
            "Comma-separated LR-RGDA variants. Format: "
            "name:m=<centers>:ft=<fit_iters>:lr=<fit_lr>:source=<fit_source>. "
            "fit_source is optional and can be none, center_replay, gmm_mean, or gmm_sample."
        ),
    )
    parser.add_argument(
        "--alphas",
        default="0,0.05,0.1,0.2,0.5,1.0",
        help="Comma-separated ensemble alpha values for RGDA+ZS.",
    )
    parser.add_argument(
        "--lada_alphas",
        default="0,0.05,0.1,0.2,0.5,1.0",
        help="Comma-separated ensemble alpha values for LADA+ZS.",
    )
    parser.add_argument(
        "--ensemble_normalize",
        default="maxshift",
        choices=["zscore", "maxshift", "prob", "raw"],
        help="Normalization used before fusing zero-shot logits with LR-RGDA logits.",
    )
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument(
        "--eval_shots_per_class",
        type=int,
        default=0,
        help=(
            "Fast evaluation mode. If > 0, evaluate each dataset on at most this "
            "many test samples per class. The default 0 keeps the full test set."
        ),
    )
    parser.add_argument(
        "--eval_subset_seed",
        type=int,
        default=0,
        help=(
            "Seed for deterministic per-class fast-eval subsampling. The same "
            "dataset uses the same subset across all incremental steps."
        ),
    )
    parser.add_argument(
        "--rgda_eval_chunk_size",
        type=int,
        default=512,
        help=(
            "Chunk size for LR-RGDA forward during evaluation. Keep this modest "
            "because full-test LR-RGDA logits can exceed GPU memory."
        ),
    )
    parser.add_argument(
        "--poll_interval",
        type=float,
        default=0.0,
        help=(
            "If > 0, run as a long-polling worker: wait for new artifacts, "
            "evaluate them incrementally, and rewrite results after each step. "
            "Useful for async evaluation alongside training."
        ),
    )
    return parser.parse_args()


def torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def parse_alphas(raw):
    return [float(item) for item in raw.replace(" ", "").split(",") if item]


def alpha_tag(alpha):
    return f"a{alpha:.2f}".replace(".", "p")


def parse_variants(raw):
    variants = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        variant = {"name": parts[0], "m": 1, "ft": 0, "lr": 0.01, "source": None}
        for part in parts[1:]:
            if "=" not in part:
                raise ValueError(f"Invalid variant field: {part}")
            key, value = part.split("=", 1)
            if key == "m":
                variant["m"] = int(value)
            elif key == "ft":
                variant["ft"] = int(value)
            elif key == "lr":
                variant["lr"] = float(value)
            elif key in ("source", "fit_source", "fs"):
                if value not in ("none", "center_replay", "gmm_mean", "gmm_sample"):
                    raise ValueError(f"Unsupported fit source for variant {parts[0]}: {value}")
                variant["source"] = value
            else:
                raise ValueError(f"Unknown variant key: {key}")
        if not variant["name"]:
            raise ValueError(f"Variant name is empty in {item}")
        variants.append(variant)
    if not variants:
        raise ValueError("--variants produced no variants")
    return variants


def artifact_paths(async_eval_dir):
    paths = sorted((async_eval_dir / "artifacts").glob("step_*/artifact.pt"))
    if not paths:
        raise SystemExit(f"No artifacts found under {async_eval_dir / 'artifacts'}")
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
    return args


def rebuild_model(artifact, device):
    from src.models.clip import get_clip_model

    args = artifact_args(artifact, device)
    model, processor = get_clip_model(args, train_mode="lora")
    model.load_state_dict(artifact["model_state_dict"])
    model.to(device)
    model.eval()

    frozen_model = None
    if getattr(args, "text_classifier_mode", "current") == "lada_hybrid":
        frozen_model, _ = get_clip_model(args, train_mode="frozen")
        frozen_model.to(device)
        frozen_model.eval()

    return args, model, processor, frozen_model


def dataset_balanced_cov(stats_dict, task_names, dataset_offsets, dataset_counts, current_num_classes):
    covs = []
    for task_name in task_names:
        offset = int(dataset_offsets[task_name])
        count = int(dataset_counts[task_name])
        if offset + count > current_num_classes:
            continue
        cov = sum(stats_dict[cid].cov for cid in range(offset, offset + count)) / count
        covs.append(cov)
    if not covs:
        return None
    return sum(covs) / len(covs)


def _move_stats_dict_to_cpu(stats_dict):
    for stat in stats_dict.values():
        stat.mean = stat.mean.detach().cpu().float()
        stat.cov = stat.cov.detach().cpu().float()
        if getattr(stat, "L", None) is not None:
            stat.L = stat.L.detach().cpu().float()
    return stats_dict


def _move_center_means_to_cpu(center_means):
    if center_means is None:
        return None
    return {int(cid): centers.detach().cpu().float() for cid, centers in center_means.items()}


def _infer_artifact_num_centers(artifact, stats_dict, center_means):
    lr_rgda_config = artifact.get("lr_rgda", {}).get("config", {})
    if "M" in lr_rgda_config:
        return int(lr_rgda_config["M"])
    if center_means:
        first_centers = next(iter(center_means.values()))
        return int(first_centers.shape[0])
    artifact_args_dict = artifact.get("args", {})
    if "num_centers" in artifact_args_dict:
        return int(artifact_args_dict["num_centers"])
    return 1


def _artifact_has_compact_stats(artifact):
    return (
        artifact.get("rgda_stats_by_m") is not None
        or (
            artifact.get("global_stats_dict") is not None
            and "dataset_balanced_global_cov" in artifact
        )
    )


def _get_rgda_stats_entry(artifact, requested_m):
    stats_by_m = artifact.get("rgda_stats_by_m")
    if not stats_by_m:
        return None
    for key in (requested_m, str(requested_m)):
        if key in stats_by_m:
            return stats_by_m[key]
    return None


def _load_compact_rgda_stats(artifact, variant):
    requested_m = int(variant["m"])
    stats_entry = _get_rgda_stats_entry(artifact, requested_m)
    if stats_entry is not None:
        stats_dict = _move_stats_dict_to_cpu(dict(stats_entry["global_stats_dict"]))
        center_means = _move_center_means_to_cpu(stats_entry.get("global_center_means"))
        global_cov = stats_entry.get("dataset_balanced_global_cov")
        if global_cov is not None:
            global_cov = global_cov.detach().cpu().float()
        if requested_m == 1:
            center_means = None
        return stats_dict, center_means, global_cov

    stats_dict = _move_stats_dict_to_cpu(dict(artifact["global_stats_dict"]))
    center_means = _move_center_means_to_cpu(artifact.get("global_center_means"))
    global_cov = artifact.get("dataset_balanced_global_cov")
    if global_cov is not None:
        global_cov = global_cov.detach().cpu().float()

    artifact_m = _infer_artifact_num_centers(artifact, stats_dict, center_means)
    if requested_m != artifact_m:
        raise ValueError(
            f"Variant {variant['name']} requests m={requested_m}, but this artifact "
            f"contains compact LR-RGDA stats for m={artifact_m}. New strict artifacts "
            "do not contain classifier_features and has no matching rgda_stats_by_m "
            "entry, so different center counts cannot be reclustered at evaluation "
            "time. Rerun training with --artifact_num_centers including the desired "
            "value, or use an old artifact that still has classifier_features."
        )

    if requested_m == 1:
        center_means = None

    return stats_dict, center_means, global_cov


def _load_legacy_feature_stats(artifact, variant, device):
    features = artifact.get("classifier_features")
    labels = artifact.get("classifier_labels")
    if features is None or labels is None:
        raise ValueError(
            f"Artifact for variant {variant['name']} has neither compact stats nor "
            "legacy classifier_features/classifier_labels."
        )

    fit_features = F.normalize(features.float(), dim=-1).to(device)
    fit_labels = labels.long().to(device)
    fit_features_cpu = fit_features.detach().cpu()
    fit_labels_cpu = fit_labels.detach().cpu()

    stats_dict, center_means = build_multi_center_stats_dict(
        fit_features_cpu, fit_labels_cpu, M=variant["m"]
    )
    global_cov = dataset_balanced_cov(
        stats_dict,
        artifact["task_names"],
        artifact["dataset_label_offsets"],
        artifact["dataset_class_counts"],
        artifact["current_num_classes"],
    )
    if global_cov is not None:
        global_cov = global_cov.detach().cpu().float()

    return stats_dict, center_means, global_cov, fit_features, fit_labels


def _build_center_replay_features(stats_dict, center_means, samples_per_class, device):
    pseudo_features = []
    pseudo_labels = []
    for cid in sorted(stats_dict):
        if center_means is not None and int(cid) in center_means:
            centers = center_means[int(cid)].to(device).float()
        else:
            centers = stats_dict[cid].mean.unsqueeze(0).to(device).float()

        n_centers = int(centers.shape[0])
        per_center = max(1, int(samples_per_class) // n_centers)
        remainder = int(samples_per_class) - per_center * n_centers
        for center_idx in range(n_centers):
            count = per_center + (1 if center_idx < remainder else 0)
            if count <= 0:
                continue
            samples = centers[center_idx].unsqueeze(0).repeat(count, 1)
            pseudo_features.append(F.normalize(samples, dim=-1))
            pseudo_labels.append(
                torch.full((count,), int(cid), dtype=torch.long, device=device)
            )

    if not pseudo_features:
        raise ValueError("Center replay requested but compact stats are empty")
    return torch.cat(pseudo_features, dim=0), torch.cat(pseudo_labels, dim=0)


def _sample_spherical_gmm_memory(gmm_memory, samples_per_class, mode, device):
    if not gmm_memory:
        raise ValueError("GMM replay requested but no gmm_memory is available in artifact")

    pseudo_features = []
    pseudo_labels = []
    for cid in sorted(gmm_memory):
        entry = gmm_memory[cid]
        means = entry["means"].to(device).float()
        covariances = entry["covariances"].to(device).float()
        weights = entry["weights"].to(device).float()
        n_components = int(means.shape[0])

        total = int(samples_per_class)
        counts = torch.clamp((weights * total).round().long(), min=1)
        diff = total - int(counts.sum().item())
        if diff != 0:
            counts[int(weights.argmax().item())] += diff
        counts = torch.clamp(counts, min=1)

        for comp_idx in range(n_components):
            count = int(counts[comp_idx].item())
            mean = means[comp_idx]
            if mode == "gmm_sample":
                std = torch.sqrt(torch.clamp(covariances[comp_idx], min=1e-8))
                samples = mean.unsqueeze(0) + torch.randn(
                    count, mean.shape[0], device=device
                ) * std
            elif mode == "gmm_mean":
                samples = mean.unsqueeze(0).repeat(count, 1)
            else:
                raise ValueError(f"Unsupported GMM replay mode: {mode}")
            pseudo_features.append(F.normalize(samples, dim=-1))
            pseudo_labels.append(
                torch.full((count,), int(cid), dtype=torch.long, device=device)
            )

    return torch.cat(pseudo_features, dim=0), torch.cat(pseudo_labels, dim=0)


def _resolve_fit_source(artifact, run_args, variant):
    if variant.get("source") is not None:
        return variant["source"]
    artifact_source = getattr(run_args, "rgda_fit_source", None)
    if artifact_source in ("none", "center_replay", "gmm_mean", "gmm_sample"):
        return artifact_source
    if artifact.get("gmm_memory"):
        return "gmm_mean"
    return "center_replay"


def _build_fit_data_from_compact_artifact(artifact, stats_dict, center_means, run_args, variant, device):
    fit_source = _resolve_fit_source(artifact, run_args, variant)
    if fit_source == "none":
        return None, None, fit_source

    samples_per_class = int(getattr(run_args, "gaussian_samples_per_class", 16))
    if fit_source == "center_replay":
        fit_features, fit_labels = _build_center_replay_features(
            stats_dict, center_means, samples_per_class, device
        )
    elif fit_source in ("gmm_mean", "gmm_sample"):
        fit_features, fit_labels = _sample_spherical_gmm_memory(
            artifact.get("gmm_memory"), samples_per_class, fit_source, device
        )
    else:
        raise ValueError(f"Unsupported fit source: {fit_source}")

    return fit_features, fit_labels, fit_source


def build_rgda_classifier(artifact, run_args, variant, device):
    fit_features = None
    fit_labels = None
    if _artifact_has_compact_stats(artifact):
        stats_dict, center_means, global_cov = _load_compact_rgda_stats(artifact, variant)
    else:
        stats_dict, center_means, global_cov, fit_features, fit_labels = _load_legacy_feature_stats(
            artifact, variant, device
        )

    classifier = LRRGDAClassifier(
        stats_dict=stats_dict,
        device=device,
        rank=getattr(run_args, "rgda_rank", 32),
        qda_reg_alpha1=getattr(run_args, "rgda_alpha1", 0.2),
        qda_reg_alpha2=getattr(run_args, "rgda_alpha2", 2.0),
        qda_reg_alpha3=getattr(run_args, "rgda_alpha3", 0.5),
        temperature=1.0,
        M=variant["m"],
        center_means=center_means,
        global_cov=global_cov,
    )
    if variant["ft"] > 0:
        if _artifact_has_compact_stats(artifact):
            fit_features, fit_labels, fit_source = _build_fit_data_from_compact_artifact(
                artifact, stats_dict, center_means, run_args, variant, device
            )
            if fit_features is None:
                print(
                    f"Skipping classifier fine-tune for variant {variant['name']} "
                    f"because fit_source={fit_source}",
                    flush=True,
                )
        else:
            fit_source = "legacy_classifier_features"
            print(
                f"WARNING: variant {variant['name']} is using legacy "
                "classifier_features/classifier_labels for fine-tuning because compact "
                "stats are not present in this artifact.",
                flush=True,
            )

        if fit_features is not None:
            print(
                f"Fine-tuning variant {variant['name']} for {variant['ft']} iters "
                f"on {fit_features.shape[0]} pseudo-features (source={fit_source})",
                flush=True,
            )
            fit_features = fit_features.to(device)
            fit_labels = fit_labels.to(device)
            classifier.fit(
                fit_features,
                fit_labels,
                iterations=variant["ft"],
                lr=variant["lr"],
                verbose=True,
            )
    classifier.classifier.eval()
    return classifier


def build_lada_classifier(artifact, device):
    lada_entry = artifact.get("lada")
    if lada_entry is None:
        return None
    classifier = LADAClassifier(
        feature_dim=lada_entry["feature_dim"],
        beta=lada_entry["beta"],
        score_mode=lada_entry["score_mode"],
    )
    classifier.load_state_dict(lada_entry["state_dict"])
    classifier.to(device)
    classifier.eval()
    return classifier


def _get_dataset_label_without_transform(dataset, index):
    if isinstance(dataset, Subset):
        return _get_dataset_label_without_transform(dataset.dataset, dataset.indices[index])

    data_source = getattr(dataset, "data_source", None)
    if data_source is not None:
        return int(data_source[index].label)

    targets = getattr(dataset, "targets", None)
    if targets is not None:
        return int(targets[index])

    labels = getattr(dataset, "labels", None)
    if labels is not None:
        return int(labels[index])

    _, label = dataset[index]
    return int(label)


def _stable_subset_seed(base_seed, dataset_name, label):
    raw = f"{int(base_seed)}:{dataset_name}:{int(label)}".encode("utf-8")
    digest = hashlib.sha256(raw).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _limit_loader_shots_per_class(loader, shots_per_class, dataset_name, subset_seed=0):
    if shots_per_class is None or int(shots_per_class) <= 0:
        return loader

    shots_per_class = int(shots_per_class)
    dataset = loader.dataset
    by_label = defaultdict(list)
    for index in range(len(dataset)):
        label = _get_dataset_label_without_transform(dataset, index)
        by_label[label].append(index)

    selected_indices = []
    for label in sorted(by_label):
        label_indices = list(by_label[label])
        rng = random.Random(_stable_subset_seed(subset_seed, dataset_name, label))
        rng.shuffle(label_indices)
        selected_indices.extend(sorted(label_indices[:shots_per_class]))

    if len(selected_indices) >= len(dataset):
        print(
            f"[FastEval] {dataset_name}: full test set retained "
            f"({len(dataset)} samples; <= {shots_per_class}/class).",
            flush=True,
        )
        return loader

    limited_dataset = Subset(dataset, selected_indices)
    print(
        f"[FastEval] {dataset_name}: using {len(limited_dataset)}/{len(dataset)} "
        f"test samples ({shots_per_class}/class cap, {len(by_label)} classes).",
        flush=True,
    )

    return DataLoader(
        limited_dataset,
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        drop_last=False,
        collate_fn=loader.collate_fn,
    )


def _rgda_forward_chunked(classifier, features, chunk_size):
    chunk_size = int(chunk_size or 0)
    if chunk_size <= 0 or features.shape[0] <= chunk_size:
        return classifier.forward(features)

    logits = []
    for start in range(0, features.shape[0], chunk_size):
        end = min(start + chunk_size, features.shape[0])
        logits.append(classifier.forward(features[start:end]))
    return torch.cat(logits, dim=0)


def evaluate_step(
    artifact,
    device,
    variants,
    alphas,
    lada_alphas=None,
    ensemble_normalize="maxshift",
    eval_batch_size_override=None,
    eval_shots_per_class=0,
    eval_subset_seed=0,
    rgda_eval_chunk_size=512,
):
    run_args, model, processor, frozen_model = rebuild_model(artifact, device)
    if run_args.eval_max_samples != 0:
        print(
            "WARNING: eval_max_samples is ignored by this sweep evaluator; "
            "utils_data test loader is used as-is.",
            flush=True,
        )

    task_names = artifact["task_names"]
    zeroshot_classifier = _build_text_classifier(
        run_args,
        model,
        processor,
        frozen_model,
        artifact["global_class_names"],
        artifact.get("cached_seen_text_features"),
        device,
    )

    rgda_classifiers = {
        variant["name"]: build_rgda_classifier(artifact, run_args, variant, device)
        for variant in variants
    }
    lada_classifier = build_lada_classifier(artifact, device)
    lada_alphas = lada_alphas or []

    step_scores = {"zero_shot": {}}
    for variant in variants:
        step_scores[f"rgda_{variant['name']}"] = {}
        for alpha in alphas:
            step_scores[f"ens_{variant['name']}_{alpha_tag(alpha)}"] = {}
    if lada_classifier is not None:
        step_scores["lada"] = {}
        for alpha in lada_alphas:
            step_scores[f"lada_zs_{alpha_tag(alpha)}"] = {}

    eval_batch_size = eval_batch_size_override or getattr(run_args, "batch_size", 64)
    for d_name in task_names:
        _, test_transform = get_transforms(d_name)
        _, _, te_loader, c_names = get_xtail_trainloader(
            root=run_args.root,
            dataset_name=d_name,
            transform_train=None,
            transform_test=test_transform,
            num_shots=run_args.num_shots,
            batch_size=eval_batch_size,
        )
        te_loader = _limit_loader_shots_per_class(
            te_loader, eval_shots_per_class, d_name, eval_subset_seed
        )
        features, labels = extract_features(model, te_loader, device)
        features = F.normalize(features.float(), dim=-1).to(device)
        labels = (labels + artifact["dataset_label_offsets"][d_name]).long().to(device)

        with torch.no_grad():
            zs_logits = features @ zeroshot_classifier
            zs_acc = zs_logits.argmax(dim=1).eq(labels).float().mean().item()
            step_scores["zero_shot"][d_name] = zs_acc

            for variant in variants:
                variant_name = variant["name"]
                rgda_logits = _rgda_forward_chunked(
                    rgda_classifiers[variant_name],
                    features,
                    rgda_eval_chunk_size,
                )
                rgda_acc = rgda_logits.argmax(dim=1).eq(labels).float().mean().item()
                step_scores[f"rgda_{variant_name}"][d_name] = rgda_acc

                for alpha in alphas:
                    current_num_classes = int(artifact["current_num_classes"])
                    ensemble_logits = combine_ensemble_logits(
                        zs_logits,
                        rgda_logits,
                        current_num_classes,
                        alpha,
                        ensemble_normalize,
                    )
                    ens_acc = ensemble_logits.argmax(dim=1).eq(labels).float().mean().item()
                    step_scores[f"ens_{variant_name}_{alpha_tag(alpha)}"][d_name] = ens_acc

            if lada_classifier is not None:
                lada_logits = lada_classifier(features)
                lada_acc = lada_logits.argmax(dim=1).eq(labels).float().mean().item()
                step_scores["lada"][d_name] = lada_acc
                for alpha in lada_alphas:
                    current_num_classes = int(artifact["current_num_classes"])
                    lada_zs_logits = combine_ensemble_logits(
                        zs_logits,
                        lada_logits,
                        current_num_classes,
                        alpha,
                        ensemble_normalize,
                    )
                    lada_zs_acc = lada_zs_logits.argmax(dim=1).eq(labels).float().mean().item()
                    step_scores[f"lada_zs_{alpha_tag(alpha)}"][d_name] = lada_zs_acc

    return step_scores


def result_payload(
    tracker,
    task_names,
    method_name,
    run_args,
    ensemble_normalize="maxshift",
    eval_shots_per_class=0,
    eval_subset_seed=0,
    rgda_eval_chunk_size=512,
):
    summary = tracker.get_summary()
    return {
        "args": {
            "task_sequence": task_names,
            "seed": getattr(run_args, "seed", ""),
            "method": method_name,
            "eval_max_samples": getattr(run_args, "eval_max_samples", ""),
            "eval_shots_per_class": int(eval_shots_per_class or 0),
            "eval_subset_seed": int(eval_subset_seed or 0),
            "rgda_eval_chunk_size": int(rgda_eval_chunk_size or 0),
            "ensemble_normalize": ensemble_normalize,
            "num_shots": getattr(run_args, "num_shots", ""),
        },
        "accuracy_matrix": tracker.get_accuracy_matrix().tolist(),
        "metrics": {
            "transfer": summary["transfer"],
            "average": summary["average"],
            "last": summary["last"],
        },
        "per_task_metrics": tracker.calculate_per_task_metrics(),
    }


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp_path, path)


def write_summary(output_dir, rows):
    csv_path = output_dir / "rgda_sweep_summary.csv"
    md_path = output_dir / "rgda_sweep_summary.md"
    headers = ["method", "transfer", "average", "last", "path"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("| method | Transfer | Average | Last | path |\n")
        handle.write("|---|---:|---:|---:|---|\n")
        for row in rows:
            handle.write(
                f"| {row['method']} | {row['transfer']:.2f} | "
                f"{row['average']:.2f} | {row['last']:.2f} | {row['path']} |\n"
            )


def _save_results(args, trackers, method_names, task_names, run_args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for method_name in method_names:
        payload = result_payload(
            trackers[method_name],
            task_names,
            method_name,
            run_args,
            args.ensemble_normalize,
            args.eval_shots_per_class,
            args.eval_subset_seed,
            args.rgda_eval_chunk_size,
        )
        out_path = args.output_dir / f"{args.experiment_name}_{method_name}_results.json"
        write_json(out_path, payload)
        rows.append(
            {
                "method": method_name,
                "transfer": payload["metrics"]["transfer"],
                "average": payload["metrics"]["average"],
                "last": payload["metrics"]["last"],
                "path": str(out_path),
            }
        )
        print(
            f"{method_name}: Transfer={payload['metrics']['transfer']:.2f} "
            f"Average={payload['metrics']['average']:.2f} "
            f"Last={payload['metrics']['last']:.2f}",
            flush=True,
        )
    write_summary(args.output_dir, rows)


def main():
    args = parse_args()
    if args.device is None:
        if args.gpu is not None and torch.cuda.is_available():
            args.device = f"cuda:{args.gpu}"
        else:
            args.device = "cuda" if torch.cuda.is_available() else "cpu"

    variants = parse_variants(args.variants)
    alphas = parse_alphas(args.alphas)
    lada_alphas = parse_alphas(args.lada_alphas)

    method_names = ["zero_shot"]
    for variant in variants:
        method_names.append(f"rgda_{variant['name']}")
        for alpha in alphas:
            method_names.append(f"ens_{variant['name']}_{alpha_tag(alpha)}")
    for alpha in lada_alphas:
        method_names.append(f"lada_zs_{alpha_tag(alpha)}")
    method_names.append("lada")

    if args.poll_interval > 0:
        print(
            f"Running as polling worker (interval={args.poll_interval}s). "
            f"Waiting for artifacts in {args.async_eval_dir}...",
            flush=True,
        )
        trackers = None
        task_names = None
        run_args = None
        evaluated_steps = set()
        while True:
            loaded = artifact_paths(args.async_eval_dir)
            new_artifacts = [
                item for item in loaded if int(item[0]) not in evaluated_steps
            ]
            if not new_artifacts:
                time.sleep(args.poll_interval)
                continue

            if trackers is None:
                task_names = list(new_artifacts[0][2]["task_names"])
                trackers = {
                    name: ContinualLearningMetrics(task_names) for name in method_names
                }
                run_args = artifact_args(new_artifacts[0][2], args.device)

            for step_index, artifact_path, artifact in new_artifacts:
                print(f"Evaluating step {step_index + 1}: {artifact_path}", flush=True)
                step_scores = evaluate_step(
                    artifact,
                    args.device,
                    variants,
                    alphas,
                    lada_alphas,
                    args.ensemble_normalize,
                    args.eval_batch_size,
                    args.eval_shots_per_class,
                    args.eval_subset_seed,
                    args.rgda_eval_chunk_size,
                )
                for method_name, scores in step_scores.items():
                    trackers[method_name].update(step_index, scores)
                evaluated_steps.add(step_index)

            _save_results(args, trackers, method_names, task_names, run_args)
            print(
                f"Saved incremental results for {len(evaluated_steps)} step(s). "
                f"Waiting for more artifacts...",
                flush=True,
            )
            time.sleep(args.poll_interval)
    else:
        loaded_artifacts = artifact_paths(args.async_eval_dir)
        task_names = list(loaded_artifacts[0][2]["task_names"])
        trackers = {name: ContinualLearningMetrics(task_names) for name in method_names}
        run_args = artifact_args(loaded_artifacts[0][2], args.device)

        for step_index, artifact_path, artifact in loaded_artifacts:
            print(f"Evaluating step {step_index + 1}: {artifact_path}", flush=True)
            step_scores = evaluate_step(
                artifact,
                args.device,
                variants,
                alphas,
                lada_alphas,
                args.ensemble_normalize,
                args.eval_batch_size,
                args.eval_shots_per_class,
                args.eval_subset_seed,
                args.rgda_eval_chunk_size,
            )
            for method_name, scores in step_scores.items():
                trackers[method_name].update(step_index, scores)

        _save_results(args, trackers, method_names, task_names, run_args)


if __name__ == "__main__":
    main()
