"""
增量微调 (Incremental Fine-tuning) 入口脚本

功能：
1. 按 dataset_sequence 逐任务增量训练
2. 每个任务进行 LoRA-NSP 训练 → 零空间投影 → 特征提取 → LR-RGDA 分类器构建
3. 自动评估所有任务，记录 LADA Transfer / Average / Last 指标矩阵
4. 保存结果 JSON

用法示例：
    python main_incremental.py \\
        --dataset_sequence aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397 \\
        --num_shots 16 --batch_size 32 --iterations 800 \\
        --lora_type lora_nsp --alpha 0.05
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import torch.nn.functional as F
import argparse
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset

from src.trainers.lora_nsp_trainer import LoRANSPTrainer
from src.classifiers.lr_rgda_classifier import LRRGDAClassifier
from src.classifiers.gaussian_statistics import build_multi_center_stats_dict
from src.utils.reference_loader import load_reference_dataset
from src.utils.main_utils import (
    fix_random_seed,
    get_zeroshot_classifier,
    evaluate_dataset,
)
from src.utils.continual_metrics import ContinualLearningMetrics
from utils_data import get_xtail_trainloader, get_xtail_classnames, get_transforms


LADA_16SHOT_EPOCHS = {
    "aircraft": 40,
    "caltech101": 10,
    "dtd": 30,
    "eurosat": 100,
    "flowers": 30,
    "food101": 5,
    "mnist": 200,
    "oxford_pets": 10,
    "stanford_cars": 30,
    "sun397": 10,
}


def _parse_epoch_overrides(spec):
    if not spec:
        return {}
    result = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                f"Invalid --dataset_epoch_overrides item '{item}', expected dataset=epochs"
            )
        name, value = item.split("=", 1)
        name = name.strip()
        try:
            epochs = int(value)
        except ValueError as exc:
            raise ValueError(f"Invalid epoch count for {name}: {value}") from exc
        if epochs <= 0:
            raise ValueError(f"Epoch count must be positive for {name}, got {epochs}")
        result[name] = epochs
    return result


def _parse_int_list(spec):
    if not spec:
        return []
    result = []
    for item in str(spec).replace(" ", "").split(","):
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError(f"Expected positive integer in list, got {value}")
        result.append(value)
    return result


def _resolve_task_train_iterations(args, task_datasets, steps_per_epoch):
    if args.train_budget_mode == "uniform":
        return int(args.iterations), None
    if args.train_budget_mode != "lada_epochs":
        raise ValueError(f"Unsupported train_budget_mode: {args.train_budget_mode}")

    epoch_map = dict(LADA_16SHOT_EPOCHS)
    epoch_map.update(_parse_epoch_overrides(args.dataset_epoch_overrides))
    missing = [d for d in task_datasets if d not in epoch_map]
    if missing:
        raise ValueError(
            "Missing LADA epoch schedule for datasets: "
            + ", ".join(missing)
            + ". Use --dataset_epoch_overrides dataset=epochs,..."
        )
    epochs = max(epoch_map[d] for d in task_datasets)
    return int(epochs * steps_per_epoch), epochs


@torch.no_grad()
def _encode_text_classifier_columns(model, processor, class_names, device):
    classifier = get_zeroshot_classifier(model, processor, class_names, device)
    return classifier.t().detach()


def _build_text_classifier(args, model, processor, frozen_model, global_class_names,
                           cached_seen_text_features, device):
    if args.text_classifier_mode == "current":
        return get_zeroshot_classifier(model, processor, global_class_names, device)
    if args.text_classifier_mode != "lada_hybrid":
        raise ValueError(f"Unsupported text_classifier_mode: {args.text_classifier_mode}")

    frozen_classifier = get_zeroshot_classifier(frozen_model, processor, global_class_names, device)
    if cached_seen_text_features is None or cached_seen_text_features.numel() == 0:
        return frozen_classifier
    seen_count = cached_seen_text_features.shape[0]
    hybrid = frozen_classifier.clone()
    hybrid[:, :seen_count] = cached_seen_text_features.to(device).t()
    hybrid = hybrid / hybrid.norm(dim=0, keepdim=True)
    return hybrid


def _fit_spherical_gmm_memory(features, labels, k):
    """Fit compact per-class spherical GMM memory from currently accessible features."""
    from sklearn.mixture import GaussianMixture

    features_np = features.detach().cpu().float().numpy()
    labels_cpu = labels.detach().cpu().long()
    memory = {}
    for cid in sorted(labels_cpu.unique().tolist()):
        class_features = features_np[labels_cpu.numpy() == cid]
        actual_k = max(1, min(int(k), len(class_features)))
        gmm = GaussianMixture(
            n_components=actual_k,
            covariance_type="spherical",
            random_state=42 + int(cid),
            reg_covar=1e-6,
        )
        gmm.fit(class_features)
        memory[int(cid)] = {
            "means": torch.from_numpy(gmm.means_).float(),
            "covariances": torch.from_numpy(gmm.covariances_).float(),
            "weights": torch.from_numpy(gmm.weights_).float(),
        }
    return memory


def _sample_spherical_gmm_memory(gmm_memory, samples_per_class, mode, device):
    """Sample or repeat pseudo-features from compact spherical GMM memory."""
    if not gmm_memory:
        raise ValueError("GMM replay requested but no GMM memory is available")
    pseudo_features = []
    pseudo_labels = []
    for cid in sorted(gmm_memory):
        entry = gmm_memory[cid]
        means = entry["means"].to(device)
        covariances = entry["covariances"].to(device)
        weights = entry["weights"].to(device)
        n_components = means.shape[0]

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
            samples = F.normalize(samples, dim=-1)
            pseudo_features.append(samples)
            pseudo_labels.append(
                torch.full((count,), int(cid), dtype=torch.long, device=device)
            )

    return torch.cat(pseudo_features, dim=0), torch.cat(pseudo_labels, dim=0)


def _build_center_replay_features(stats_dict, center_means, samples_per_class, device):
    """Build compact center replay pseudo-features for classifier-only fine-tuning."""
    pseudo_features = []
    pseudo_labels = []
    for cid in sorted(stats_dict):
        if center_means is not None and cid in center_means:
            centers = center_means[cid].to(device)
        else:
            centers = stats_dict[cid].mean.unsqueeze(0).to(device)

        n_centers = centers.shape[0]
        per_center = max(1, int(samples_per_class) // n_centers)
        remainder = int(samples_per_class) - per_center * n_centers
        for center_idx in range(n_centers):
            count = per_center + (1 if center_idx < remainder else 0)
            samples = centers[center_idx].unsqueeze(0).repeat(count, 1)
            samples = F.normalize(samples, dim=-1)
            pseudo_features.append(samples)
            pseudo_labels.append(
                torch.full((count,), int(cid), dtype=torch.long, device=device)
            )

    return torch.cat(pseudo_features, dim=0), torch.cat(pseudo_labels, dim=0)


def _compute_dataset_balanced_global_cov(stats_dict, history_class_names):
    offset = 0
    per_dataset_covs = []
    for hcn in history_class_names:
        n_classes = len(hcn)
        ds_cov = sum(stats_dict[cid].cov for cid in range(offset, offset + n_classes)) / n_classes
        per_dataset_covs.append(ds_cov)
        offset += n_classes
    return sum(per_dataset_covs) / len(per_dataset_covs)


def _print_lada_metrics(name, tracker, task_names):
    """Print LADA-style Transfer/Average/Last metrics from a full KxK matrix."""
    print("\n" + "-" * 110)
    print(f"[{name} LADA Metrics]")
    header_str = " | ".join([f"{h[:8]:<8}" for h in task_names])
    print(f"Metric     | {header_str} | [Average]")
    print("-" * 110)

    per_task = tracker.calculate_per_task_metrics()
    transfer = [per_task[t]["transfer"] for t in task_names]
    average = [per_task[t]["average"] for t in task_names]
    last = [per_task[t]["last"] for t in task_names]
    summary = tracker.get_summary()

    print(
        "Transfer  | "
        + " | ".join(f"{x:8.1f}" if i > 0 else f"{'N/A':>8}" for i, x in enumerate(transfer))
        + f" | [{summary['transfer']:.1f}]"
    )
    print(
        "Average   | "
        + " | ".join(f"{x:8.1f}" for x in average)
        + f" | [{summary['average']:.1f}]"
    )
    print(
        "Last      | "
        + " | ".join(f"{x:8.1f}" for x in last)
        + f" | [{summary['last']:.1f}]"
    )
    print("-" * 110)


def _cpu_state_dict(state_dict):
    return {k: v.detach().cpu() for k, v in state_dict.items()}


def _serializable_args(args):
    return {
        k: str(v) if not isinstance(v, (int, float, bool, list, dict, type(None))) else v
        for k, v in vars(args).items()
    }


def _atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp_path, path)


def _atomic_save_step_artifact(
    args,
    step_index,
    task_name,
    model,
    cached_seen_text_features,
    lr_rgda_classifier,
    global_stats_dict,
    global_center_means,
    dataset_balanced_global_cov,
    task_names,
    global_class_names,
    dataset_label_offsets,
    dataset_class_counts,
    current_num_classes,
    gmm_memory=None,
    rgda_stats_by_m=None,
):
    """Save an immutable post-task snapshot and publish an eval-ready job."""
    if not args.async_eval_dir:
        raise ValueError("--async_eval_dir is required when saving step artifacts")

    base_dir = Path(args.async_eval_dir)
    artifact_root = base_dir / "artifacts"
    queue_dir = base_dir / "queue"
    artifact_root.mkdir(parents=True, exist_ok=True)
    queue_dir.mkdir(parents=True, exist_ok=True)

    step_name = f"step_{step_index + 1:02d}_{task_name}"
    tmp_dir = artifact_root / f".{step_name}.tmp"
    final_dir = artifact_root / step_name
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    if final_dir.exists():
        shutil.rmtree(final_dir)
    tmp_dir.mkdir(parents=True)

    artifact_path = tmp_dir / "artifact.pt"
    artifact = {
        "schema_version": 1,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "step_index": step_index,
        "task_name": task_name,
        "task_names": task_names,
        "global_class_names": global_class_names,
        "dataset_label_offsets": dataset_label_offsets,
        "dataset_class_counts": dataset_class_counts,
        "current_num_classes": current_num_classes,
        "gmm_memory": gmm_memory,
        "args": _serializable_args(args),
        "model_state_dict": _cpu_state_dict(model.state_dict()),
        "cached_seen_text_features": (
            cached_seen_text_features.detach().cpu()
            if cached_seen_text_features is not None
            else None
        ),
        "global_stats_dict": global_stats_dict,
        "global_center_means": global_center_means,
        "dataset_balanced_global_cov": dataset_balanced_global_cov.detach().cpu(),
        "rgda_stats_by_m": rgda_stats_by_m,
        "lr_rgda": {
            "state_dict": _cpu_state_dict(lr_rgda_classifier.classifier.state_dict()),
            "num_classes": lr_rgda_classifier.num_classes,
            "class_ids": lr_rgda_classifier.class_ids,
            "config": {
                "rank": args.rgda_rank,
                "qda_reg_alpha1": args.rgda_alpha1,
                "qda_reg_alpha2": args.rgda_alpha2,
                "qda_reg_alpha3": args.rgda_alpha3,
                "temperature": 1.0,
                "M": args.num_centers,
            },
        },
    }
    torch.save(artifact, artifact_path)

    os.replace(tmp_dir, final_dir)
    ready_payload = {
        "schema_version": 1,
        "step_index": step_index,
        "task_name": task_name,
        "artifact_path": str((final_dir / "artifact.pt").resolve()),
    }
    _atomic_write_json(queue_dir / f"{step_name}.ready.json", ready_payload)
    logging.info("Saved async eval artifact: %s", final_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Incremental Fine-tuning for CLIP Continual Learning")

    # 数据集相关参数
    parser.add_argument("--id_datasets", type=str, nargs='+',
                        default=["aircraft", "caltech101", "dtd", "eurosat", "flowers",
                                 "food101", "mnist", "oxford_pets", "stanford_cars", "sun397"],
                        help="List of all ID datasets (for reference).")
    parser.add_argument("--root", type=str, default="/data1/open_datasets/X-TAIL",
                        help="Root directory of the dataset.")
    parser.add_argument("--num_shots", type=int, default=16,
                        help="Number of shots for few-shot learning.")
    parser.add_argument("--full_shot", action="store_true", default=False,
                        help="Use full dataset instead of few-shot (overrides --num_shots).")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training and testing.")
    parser.add_argument("--eval_max_samples", type=int, default=0,
                        help="Maximum number of test samples per dataset (0=full split).")

    # 增量学习特定参数
    parser.add_argument("--dataset_sequence", type=str, nargs='+',
                        default=["aircraft", "caltech101", "dtd", "eurosat", "flowers",
                                 "food101", "mnist", "oxford_pets", "stanford_cars", "sun397"],
                        help="Sequence of datasets, one task per dataset. "
                             "Each element is a single dataset name.")

    # 训练基础参数
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU index to use (e.g., 0 for cuda:0). Sets CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--device", type=str,
                        default=None,
                        help="Device to use for training. Overrides --gpu if set.")
    parser.add_argument("--iterations", type=int, default=800,
                        help="Number of training iterations per task.")
    parser.add_argument("--train_budget_mode", type=str, default="uniform",
                        choices=["uniform", "lada_epochs"],
                        help="Training budget mode. uniform uses --iterations for every task; "
                             "lada_epochs uses the official LADA 16-shot per-dataset epoch schedule.")
    parser.add_argument("--dataset_epoch_overrides", type=str, default=None,
                        help="Comma-separated per-dataset epoch overrides, e.g. aircraft=40,eurosat=100.")
    parser.add_argument("--use_lada_recipe_defaults", action="store_true", default=False,
                        help="Opt in to LADA-style recipe defaults: batch_size=64, lr=1e-3, "
                             "weight_decay=5e-4, scheduler=onecycle, train_budget_mode=lada_epochs.")

    # 优化器参数
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=3e-5,
                        help="Weight decay for optimizer.")
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "onecycle"],
                        help="Per-step learning-rate scheduler.")

    # LoRA 相关参数
    parser.add_argument("--lora_rank", type=int, default=4,
                        help="Rank for LoRA adaptation.")
    parser.add_argument("--lora_alpha", type=float, default=None,
                        help="LoRA alpha scaling factor. Defaults to lora_rank when None.")
    parser.add_argument("--lora_dropout", type=float, default=0.0,
                        help="LoRA dropout rate (only effective for lora_vanilla).")
    parser.add_argument("--lora_type", type=str, default="lora_nsp",
                        choices=["lora_vanilla", "lora_sgp", "lora_nsp"],
                        help="Type of LoRA adaptation (for backward compat).")
    parser.add_argument("--init_mode", type=str, default="lora_nsp",
                        choices=["lora_nsp", "lora_vanilla",
                                 "proj_sigma_tail", "proj_sigma_middle",
                                 "weight_svd_tail", "weight_svd_middle"],
                        help="""Adapter initialization mode.
    lora_nsp:        (default) runtime P + random A/B init (current LoRA-NSP)
    lora_vanilla:    standard LoRA, no P, no structured init
    proj_sigma_tail: Σ's smallest eigenvectors → project W → init A/B
    proj_sigma_middle: Σ's middle eigenvectors → project W → init A/B
    weight_svd_tail: W's smallest singular components → init A/B
    weight_svd_middle: W's middle singular components → init A/B""")
    parser.add_argument("--nsp_eps", type=float, default=0.05,
                        help="Epsilon parameter for NSP.")
    parser.add_argument("--nsp_weight", type=float, default=0.02,
                        help="Weight parameter for NSP.")
    parser.add_argument("--weight_temp", type=float, default=1.0,
                        help="Temperature parameter for weight.")
    parser.add_argument("--weight_kind", type=str, default="log1p")
    parser.add_argument("--weight_p", type=float, default=1.0,
                        help="P parameter for weight function.")

    # 参考数据集参数
    parser.add_argument("--reference_dataset", type=str, default="flickr8k",
                        help="Reference dataset for training.")
    parser.add_argument("--reference_batch_size", type=int, default=32,
                        help="Batch size for reference dataset.")
    parser.add_argument("--num_workers", type=int, default=6,
                        help="Number of workers for data loading.")

    # 损失函数权重参数
    parser.add_argument("--fd_weight", type=float, default=1.0,
                        help="Weight for feature distillation loss (0=disabled).")
    parser.add_argument("--cd_weight", type=float, default=1.0,
                        help="Weight for cross-modal distillation loss (0=disabled).")
    parser.add_argument("--aux_weight", type=float, default=1.0,
                        help="Weight for auxiliary linear classifier loss (0=disabled). "
                             "Adds a linear head on features during training to improve "
                             "feature separability for downstream LR-RGDA.")
    parser.add_argument("--sce_a", type=float, default=0.5,
                        help="Weight for CE in symmetric cross-entropy loss.")
    parser.add_argument("--sce_b", type=float, default=0.5,
                        help="Weight for RCE in symmetric cross-entropy loss.")

    # 分类器参数
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Weight for LR-RGDA classifier in ensemble (paper: 0.05).")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for zero-shot classifier.")
    parser.add_argument("--adaptive_ensemble", action='store_true', default=False,
                        help="Use per-sample adaptive alpha based on classifier confidence.")
    parser.add_argument("--ensemble_normalize", type=str, default="maxshift",
                        choices=["zscore", "maxshift", "prob", "raw"],
                        help="Normalization used before fusing zero-shot logits with "
                             "the ID-only LR-RGDA classifier.")
    parser.add_argument("--classifier_feature_transform", type=str, default="test",
                        choices=["train", "test"],
                        help="Transform for LR-RGDA classifier feature extraction: "
                             "'train'=augmented, 'test'=deterministic.")

    # LR-RGDA 构建参数
    parser.add_argument("--rgda_rank", type=int, default=32,
                        help="Rank for LR-RGDA low-rank decomposition.")
    parser.add_argument("--rgda_alpha1", type=float, default=0.2,
                        help="qda_reg_alpha1 for LR-RGDA.")
    parser.add_argument("--rgda_alpha2", type=float, default=2.0,
                        help="qda_reg_alpha2 for LR-RGDA.")
    parser.add_argument("--rgda_alpha3", type=float, default=0.5,
                        help="qda_reg_alpha3 for LR-RGDA.")
    parser.add_argument("--rgda_train_iter", type=int, default=0,
                        help="LR-RGDA classifier fine-tuning iterations (0=analytical only).")
    parser.add_argument("--rgda_train_lr", type=float, default=0.01,
                        help="LR-RGDA classifier fine-tuning learning rate.")
    parser.add_argument("--rgda_fit_source", type=str, default="gmm_sample",
                        choices=["none", "center_replay", "gmm_mean", "gmm_sample"],
                        help="Pseudo-feature source for LR-RGDA classifier-only fine-tuning. "
                             "This does not replace analytical LR-RGDA mean/cov construction. "
                             "Used only when --rgda_train_iter > 0.")

    # 文本编码器 LoRA 参数
    parser.add_argument("--tune_text_encoder", type=lambda x: x.lower() == 'true', default=True,
                        help="是否同时微调文本编码器（默认 True）。设为 False 则仅微调图像编码器。")
    parser.add_argument("--text_adapter_type", type=str, default="matched",
                        choices=["matched", "lada_adaptformer"],
                        help="文本侧可训练模块。matched=沿用 lora_type 对应的 text LoRA；"
                             "lada_adaptformer=使用官方 LADA 风格的 text AdaptFormer。")
    parser.add_argument("--text_adapter_dim", type=int, default=16,
                        help="LADA text AdaptFormer bottleneck dimension.")
    parser.add_argument("--text_adapter_scale", type=float, default=0.1,
                        help="LADA text AdaptFormer residual scale.")
    parser.add_argument("--text_lora_rank", type=int, default=4,
                        help="文本编码器 LoRA rank（默认 4，与 image lora_rank 一致）。")
    parser.add_argument("--max_zs_classes", type=int, default=128,
                        help="Zero-shot 分类器最大类数上限（用于联合训练时的随机采样）。")
    parser.add_argument("--tune_vision_encoder", type=lambda x: x.lower() == 'true', default=True,
                        help="是否微调视觉编码器（默认 True）。设为 False 则仅微调文本编码器（text-only模式）。")
    parser.add_argument("--text_tuning_schedule", type=str, default=None,
                        choices=["always", "never", "freeze_after", "low_lr_after"],
                        help="文本编码器微调调度。None=根据tune_text_encoder自动选择。")
    parser.add_argument("--text_schedule_switch_task", type=int, default=1,
                        help="切换到降低文本LR的任务编号（1-indexed）。")
    parser.add_argument("--text_lr_scale_after_task", type=float, default=0.2,
                        help="switch_task之后文本LR的缩放因子（low_lr_after模式下）。")
    parser.add_argument("--num_centers", type=int, default=1,
                        help="Number of k-means centers per class for multi-center LR-RGDA. 1=single-center.")
    parser.add_argument("--artifact_num_centers", type=str, default="",
                        help="Comma-separated center counts whose compact LR-RGDA stats are saved in "
                             "step artifacts for offline classifier sweeps. The active --num_centers "
                             "is always included. Example: 1,4.")
    parser.add_argument("--text_classifier_mode", type=str, default="current",
                        choices=["current", "lada_hybrid"],
                        help="Text classifier used at evaluation. current encodes all classes with the "
                             "current adapted text encoder. lada_hybrid uses cached tuned text prototypes "
                             "for seen classes and frozen CLIP zero-shot features for unseen classes.")

    # LADA 分类器参数（用于与 LR-RGDA 对比）
    parser.add_argument("--enable_lada", action="store_true", default=False,
                        help="Enable LADA classifier for comparison.")
    parser.add_argument("--lada_k", type=int, default=16,
                        help="LADA prototypes per class.")
    parser.add_argument("--lada_beta", type=float, default=1.0,
                        help="LADA affinity sharpness.")
    parser.add_argument("--lada_alpha", type=float, default=0.05,
                        help="LADA+ZS ensemble weight.")
    parser.add_argument("--lada_train_iter", type=int, default=0,
                        help="LADA classifier fine-tuning iterations (0=analytical).")
    parser.add_argument("--lada_train_lr", type=float, default=0.01,
                        help="LADA classifier fine-tuning learning rate.")

    # GMM 统计回放参数
    parser.add_argument("--use_gaussian_features", action="store_true", default=False,
                        help="Use GMM pseudo-features for classifier replay.")
    parser.add_argument("--gmm_k", type=int, default=4,
                        help="GMM components per class for statistical replay.")
    parser.add_argument("--gmm_sample_mode", type=str, default="mean",
                        choices=["mean", "sample"],
                        help="GMM replay mode: component means or stochastic samples.")
    parser.add_argument("--gmm_fit_space", type=str, default="raw",
                        choices=["raw", "sphere"],
                        help="Space for GMM fitting: raw features or L2-normalized sphere.")
    parser.add_argument("--gaussian_samples_per_class", type=int, default=16,
                        help="Number of pseudo-samples per class for GMM replay.")

    # 输出参数
    parser.add_argument("--output_dir", type=str, default="experiments",
                        help="Output directory for result JSON files.")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Stable experiment label for result file naming and summarizer.")
    parser.add_argument("--async_eval_dir", type=str, default=None,
                        help="Directory for async eval artifacts, queue files, and results.")
    parser.add_argument("--save_step_artifacts", action="store_true", default=False,
                        help="Save a post-task model/classifier artifact for external evaluation.")
    parser.add_argument("--skip_inline_eval", action="store_true", default=False,
                        help="Skip in-process evaluation. Requires --save_step_artifacts.")

    args = parser.parse_args()
    # 将 dataset_sequence 转换为嵌套列表格式 [[d1], [d2], ...]
    args.dataset_sequence = [[d] for d in args.dataset_sequence]

    if args.use_lada_recipe_defaults:
        args.batch_size = 64
        args.lr = 1e-3
        args.weight_decay = 5e-4
        args.scheduler = "onecycle"
        args.train_budget_mode = "lada_epochs"

    # 解析设备：优先 --device，否则用 --gpu 指定 cuda:N
    if args.device is None:
        args.device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    if args.skip_inline_eval and not args.save_step_artifacts:
        parser.error("--skip_inline_eval requires --save_step_artifacts")
    if args.save_step_artifacts and not args.async_eval_dir:
        parser.error("--save_step_artifacts requires --async_eval_dir")

    return args


def main(args):
    if args.seed is not None:
        fix_random_seed(args.seed)

    # ========== 1. 初始化 ==========
    logging.info("\n=== Initializing Incremental Learning ===")
    trainer = LoRANSPTrainer(args)
    model = trainer.model
    processor = trainer.processor

    # 解析文本编码器调度的默认值
    if args.text_tuning_schedule is None:
        args.text_tuning_schedule = "always" if args.tune_text_encoder else "never"

    # 验证文本调度参数
    if args.text_schedule_switch_task < 1:
        raise ValueError("--text_schedule_switch_task must be >= 1")
    if args.text_lr_scale_after_task < 0:
        raise ValueError("--text_lr_scale_after_task must be >= 0")

    # 非 never 调度时强制启用文本 LoRA（否则调度无意义）
    if args.text_tuning_schedule == "never":
        args.tune_text_encoder = False
    else:
        args.tune_text_encoder = True

    use_distillation = args.fd_weight > 0 or args.cd_weight > 0
    reference_loader = load_reference_dataset(args, trainer.model_pretrain,
                                              processor, args.device) if use_distillation else None

    # 预收集所有数据集的类名，用于全局 ZS 分类器（text LoRA 每次 merge 后更新）
    global_class_names = []
    task_names = [d[0] for d in args.dataset_sequence]
    dataset_label_offsets = {}
    dataset_class_counts = {}
    next_label_offset = 0
    for task_datasets in args.dataset_sequence:
        for d_name in task_datasets:
            c_names = get_xtail_classnames(args.root, d_name, args.num_shots)
            dataset_label_offsets[d_name] = next_label_offset
            dataset_class_counts[d_name] = len(c_names)
            next_label_offset += len(c_names)
            global_class_names.extend(c_names)

    # ========== 2. 增量学习循环 ==========
    history_class_names = []  # 记录所有已学类名列表的列表
    cached_seen_text_features = None
    artifact_num_centers = sorted(set([args.num_centers] + _parse_int_list(args.artifact_num_centers)))
    global_rgda_stats_by_m = {m: {} for m in artifact_num_centers}
    global_rgda_centers_by_m = {m: None for m in artifact_num_centers}
    dataset_balanced_cov_by_m = {}
    global_stats_dict = {}
    global_center_means = None  # 多中心 LR-RGDA 的类中心映射，用局部变量替代 main._global_center_means
    global_gmm_memory = {}

    metrics_zs = ContinualLearningMetrics(task_names)
    metrics_rgda = ContinualLearningMetrics(task_names)
    metrics_ens = ContinualLearningMetrics(task_names)
    acc_matrix_lada = [] if args.enable_lada else None
    acc_matrix_lada_zs = [] if args.enable_lada else None

    for i, task_datasets in enumerate(args.dataset_sequence):
        print(f"\n" + "=" * 50)
        print(f"=== Task {i+1}: {task_datasets} ===")
        print("=" * 50)

        # --- 2a. 准备训练数据 ---
        train_loaders = []
        task_class_names = []
        for d_name in task_datasets:
            train_transform, test_transform = get_transforms(d_name)
            num_shots = None if args.full_shot else args.num_shots
            tr_loader, _, _, c_names = get_xtail_trainloader(
                root=args.root, dataset_name=d_name,
                transform_train=train_transform, transform_test=test_transform,
                num_shots=num_shots, batch_size=args.batch_size
            )
            train_loaders.append(tr_loader)
            task_class_names.extend(c_names)

        merged_dataset = ConcatDataset([loader.dataset for loader in train_loaders])
        # 用于提取协方差的 loader 不打乱顺序
        cov_loader = DataLoader(merged_dataset, batch_size=args.batch_size, shuffle=False)
        merged_loader = DataLoader(merged_dataset, batch_size=args.batch_size, shuffle=True)
        task_train_iterations, task_epochs = _resolve_task_train_iterations(
            args, task_datasets, len(merged_loader))
        if task_epochs is None:
            logging.info(
                "Task %d training budget: %d iterations (uniform)",
                i + 1, task_train_iterations,
            )
        else:
            logging.info(
                "Task %d training budget: %d epochs x %d steps/epoch = %d iterations",
                i + 1, task_epochs, len(merged_loader), task_train_iterations,
            )

        # --- 2b. 初始化适配器（非标准 LoRA 时在训练前初始化） ---
        if args.init_mode not in ["lora_nsp", "lora_vanilla"]:
            logging.info(f"\n=== Initializing adapters ({args.init_mode}) ===")
            if "proj_sigma" in args.init_mode and args.init_mode != "lora_nsp":
                window = "tail" if "tail" in args.init_mode else "middle"
                if trainer.covariance_history:
                    trainer.model.vision_model.initialize_adapters_from_covariance(
                        trainer.covariance_history, window=window)
                else:
                    logging.info("No covariance history yet, using default random init for Task 1")
            elif "weight_svd" in args.init_mode:
                window = "tail" if "tail" in args.init_mode else "middle"
                trainer.model.vision_model.initialize_adapters_from_weight_svd(window=window)

        # --- 2c. 训练模型 ---
        # 文本编码器调度：根据当前任务决定 text_lr
        task_number = i + 1
        schedule = args.text_tuning_schedule
        text_lr = args.lr  # 默认
        if schedule == "never":
            text_lr = 0.0
        elif schedule in ("freeze_after", "low_lr_after"):
            if task_number > args.text_schedule_switch_task:
                if schedule == "freeze_after":
                    text_lr = 0.0
                else:
                    text_lr = args.lr * args.text_lr_scale_after_task
        train_text_this_task = text_lr > 0
        model = trainer.train(merged_loader, task_class_names, reference_loader,
                              aux_weight=args.aux_weight,
                              train_text_encoder=train_text_this_task,
                              text_lr=text_lr,
                              iterations=task_train_iterations)

        # --- 2d. 任务后处理：合入 + 协方差累积 ---
        if args.init_mode == "lora_nsp":
            print("\n=== Applying Null-Space Projection (NSP) ===")
            # 图像编码器 NSP
            text_covariances = None
            if trainer.has_vision_lora:
                covariances = trainer.extract_layer_covariances(cov_loader)
            # 文本编码器 NSP（仅当本轮实际训练了文本编码器）
            if trainer.has_text_lora and train_text_this_task:
                text_covariances = trainer.extract_text_covariances(task_class_names)
            trainer.finalize_task_for_incremental()
            if trainer.has_vision_lora:
                trainer.update_covariance_history(covariances)
            if trainer.has_text_lora and train_text_this_task:
                trainer.update_text_covariance_history(text_covariances)
        elif "proj_sigma" in args.init_mode:
            print("\n=== Proj-Σ: Extracting covariances + merging ===")
            if trainer.has_vision_lora:
                covariances = trainer.extract_layer_covariances(cov_loader)
                trainer.update_covariance_history(covariances, update_projection=False)
            trainer.finalize_task_for_incremental()
        else:
            print(f"\n=== Merging LoRA Weights (init_mode={args.init_mode}) ===")
            trainer.finalize_task_for_incremental()

        if args.text_classifier_mode == "lada_hybrid":
            task_text_features = _encode_text_classifier_columns(
                model, processor, task_class_names, args.device)
            cached_seen_text_features = (
                task_text_features.detach()
                if cached_seen_text_features is None
                else torch.cat([cached_seen_text_features, task_text_features.detach()], dim=0)
            )
            logging.info(
                "Cached LADA-style seen text prototypes: %s",
                tuple(cached_seen_text_features.shape),
            )
            if args.text_adapter_type == "lada_adaptformer" and hasattr(model.text_model, "reset_adapters"):
                model.text_model.reset_adapters()
                logging.info("Reset LADA text AdaptFormer after caching current task prototypes.")

        # --- 2e. 提取特征并构建统计字典 ---
        task_features = []
        task_labels = []
        task_gmm_features = []
        task_gmm_labels = []
        label_offset = sum(len(c_names) for c_names in history_class_names)

        for d_name in task_datasets:
            train_transform, test_transform = get_transforms(d_name)
            tr_loader, tr4update, _, c_names = get_xtail_trainloader(
                root=args.root, dataset_name=d_name,
                transform_train=train_transform, transform_test=test_transform,
                num_shots=num_shots, batch_size=args.batch_size
            )
            # 根据 --classifier_feature_transform 选择特征提取的 transform
            # 'test' = 确定性变换（publication protocol），'train' = 随机增强
            feature_loader = tr_loader if args.classifier_feature_transform == "train" else tr4update
            from src.utils.feature_extractor import extract_features
            raw_features, labels = extract_features(
                model, feature_loader, args.device, normalize=False)
            features = F.normalize(raw_features, dim=-1)
            global_labels = labels + label_offset
            task_features.append(features)
            task_labels.append(global_labels)
            gmm_features = raw_features if args.gmm_fit_space == "raw" else features
            task_gmm_features.append(gmm_features)
            task_gmm_labels.append(global_labels)

        task_features = torch.cat(task_features)
        task_labels = torch.cat(task_labels)
        task_gmm_features = torch.cat(task_gmm_features)
        task_gmm_labels = torch.cat(task_gmm_labels)

        # 构建一套或多套 compact LR-RGDA stats。主 M 用于 inline classifier；
        # 其它 M 只写入 artifact，供离线 classifier sweep 使用。
        for m in artifact_num_centers:
            task_stats_dict_m, center_means_m = build_multi_center_stats_dict(
                task_features, task_labels, M=m)
            global_rgda_stats_by_m[m].update(task_stats_dict_m)
            if center_means_m is not None:
                if global_rgda_centers_by_m[m] is None:
                    global_rgda_centers_by_m[m] = {}
                # center_means 的 key 已是全局类ID（来自 task_labels + label_offset）
                global_rgda_centers_by_m[m].update(center_means_m)

        global_stats_dict = global_rgda_stats_by_m[args.num_centers]
        global_center_means = global_rgda_centers_by_m[args.num_centers]

        task_gmm_memory = _fit_spherical_gmm_memory(
            task_gmm_features, task_gmm_labels, k=args.gmm_k)
        global_gmm_memory.update(task_gmm_memory)
        logging.info(
            "[GMM Memory] Updated spherical GMM memory: %d seen classes, "
            "k=%d, fit_space=%s",
            len(global_gmm_memory), args.gmm_k, args.gmm_fit_space,
        )

        history_class_names.append(task_class_names)

        # --- 2f. 构建分类器 ---
        # 计算数据集等权的全局协方差
        dataset_balanced_cov_by_m = {
            m: _compute_dataset_balanced_global_cov(stats_dict_m, history_class_names)
            for m, stats_dict_m in global_rgda_stats_by_m.items()
        }
        dataset_balanced_global_cov = dataset_balanced_cov_by_m[args.num_centers]

        lr_rgda_classifier = LRRGDAClassifier(
            stats_dict=global_stats_dict,
            device=args.device,
            rank=args.rgda_rank,
            M=args.num_centers,
            center_means=global_center_means,
            qda_reg_alpha1=args.rgda_alpha1,
            qda_reg_alpha2=args.rgda_alpha2,
            qda_reg_alpha3=args.rgda_alpha3,
            temperature=1.0,
            global_cov=dataset_balanced_global_cov,
        )
        # 分类器微调（如果启用）：LR-RGDA 仍由正常 mean/cov 构建；
        # 这里只使用 compact replay pseudo-features 微调 classifier 参数。
        if args.rgda_train_iter > 0:
            if args.rgda_fit_source == "none":
                logging.info(
                    "[Fit] --rgda_train_iter=%d but --rgda_fit_source=none; "
                    "skipping classifier fine-tuning.",
                    args.rgda_train_iter,
                )
                replay_features = replay_labels = None
            elif args.rgda_fit_source == "center_replay":
                replay_features, replay_labels = _build_center_replay_features(
                    global_stats_dict,
                    global_center_means,
                    args.gaussian_samples_per_class,
                    args.device,
                )
            elif args.rgda_fit_source in ("gmm_mean", "gmm_sample"):
                replay_features, replay_labels = _sample_spherical_gmm_memory(
                    global_gmm_memory,
                    args.gaussian_samples_per_class,
                    args.rgda_fit_source,
                    args.device,
                )
            else:
                raise ValueError(f"Unsupported rgda_fit_source: {args.rgda_fit_source}")

            if replay_features is not None:
                logging.info(
                    "[Fit] Fine-tuning LR-RGDA for %d iters on %d compact replay "
                    "pseudo-features (fit_source=%s, samples_per_class=%d)...",
                    args.rgda_train_iter,
                    replay_features.shape[0],
                    args.rgda_fit_source,
                    args.gaussian_samples_per_class,
                )
                lr_rgda_classifier.fit(
                    replay_features, replay_labels,
                    iterations=args.rgda_train_iter,
                    lr=args.rgda_train_lr,
                )

        current_num_classes = sum(len(c_names) for c_names in history_class_names)
        if args.save_step_artifacts:
            rgda_stats_by_m = {
                int(m): {
                    "global_stats_dict": global_rgda_stats_by_m[m],
                    "global_center_means": global_rgda_centers_by_m[m],
                    "dataset_balanced_global_cov": dataset_balanced_cov_by_m[m].detach().cpu(),
                }
                for m in artifact_num_centers
            }
            _atomic_save_step_artifact(
                args=args,
                step_index=i,
                task_name=task_datasets[0],
                model=model,
                cached_seen_text_features=cached_seen_text_features,
                lr_rgda_classifier=lr_rgda_classifier,
                global_stats_dict=global_stats_dict,
                global_center_means=global_center_means,
                dataset_balanced_global_cov=dataset_balanced_global_cov,
                task_names=task_names,
                global_class_names=global_class_names,
                dataset_label_offsets=dataset_label_offsets,
                dataset_class_counts=dataset_class_counts,
                current_num_classes=current_num_classes,
                gmm_memory=global_gmm_memory,
                rgda_stats_by_m=rgda_stats_by_m,
            )

        if args.skip_inline_eval:
            continue

        zeroshot_classifier = _build_text_classifier(
            args, model, processor, trainer.model_pretrain, global_class_names,
            cached_seen_text_features, args.device)

        # --- 2g. 按 LADA 协议评估所有任务 ---
        print("\n=== Evaluating Task ===")
        step_accs_zs, step_accs_rgda, step_accs_ens = {}, {}, {}

        for j in range(len(args.dataset_sequence)):
            eval_datasets = args.dataset_sequence[j]
            d_name = eval_datasets[0]  # 每个 Task 只有一个数据集
            eval_label_offset = dataset_label_offsets[d_name]

            zs_acc, rgda_acc, ens_acc, lada_acc, lada_zs_acc, c_len, _ = evaluate_dataset(
                args, d_name, model, zeroshot_classifier, lr_rgda_classifier,
                current_num_classes, eval_label_offset
            )
            expected_c_len = dataset_class_counts[d_name]
            if c_len != expected_c_len:
                raise ValueError(
                    f"Class count mismatch for {d_name}: eval={c_len}, expected={expected_c_len}"
                )

            print(f"[Tested on Task {j+1}: {d_name:<10s}] -> "
                  f"Zero-shot: {zs_acc:5.1f}% | LR-RGDA: {rgda_acc:5.1f}% | "
                  f"Ensemble: {ens_acc:5.1f}%")

            step_accs_zs[d_name] = zs_acc / 100.0
            step_accs_rgda[d_name] = rgda_acc / 100.0
            step_accs_ens[d_name] = ens_acc / 100.0

        metrics_zs.update(i, step_accs_zs)
        metrics_rgda.update(i, step_accs_rgda)
        metrics_ens.update(i, step_accs_ens)

    if args.skip_inline_eval:
        logging.info(
            "Skipped inline evaluation. Use scripts/evaluate_incremental_artifacts.py "
            "and scripts/merge_incremental_async_results.py with --async_eval_dir=%s",
            args.async_eval_dir,
        )
        return

    # ========== 3. 打印最终结果 ==========
    print("\n=== Training and Evaluation Completed ===")
    print("\n" + "=" * 80)
    print("Final Results Mapping to Paper Tables")
    print("=" * 80)

    _print_lada_metrics("Zero-shot Baseline", metrics_zs, task_names)
    _print_lada_metrics("LR-RGDA Only", metrics_rgda, task_names)
    _print_lada_metrics(f"Ours Ensemble (alpha={args.alpha})", metrics_ens, task_names)

    # ========== 4. 保存结果 JSON ==========
    # 格式与 summarize_incremental_metrics.py 兼容：完整 KxK matrix，数值为 [0, 1] fraction。
    zs_stats = metrics_zs.get_summary()
    rgda_stats = metrics_rgda.get_summary()
    ens_stats = metrics_ens.get_summary()

    output_dir = args.output_dir or "experiments"
    os.makedirs(output_dir, exist_ok=True)
    stem = args.experiment_name or f"incremental_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    file_name = f"{stem}.json"
    file_name_zs = f"{stem}_zs_results.json"
    file_name_rgda = f"{stem}_rgda_results.json"
    file_name_ens = f"{stem}_ens_results.json"
    save_path = os.path.join(output_dir, file_name)

    # Per-classifier result files (summarizer-compatible)
    for suffix, tracker, stats, method_name in [
        ("_zs_results", metrics_zs, zs_stats, "zero_shot"),
        ("_rgda_results", metrics_rgda, rgda_stats, "lr_rgda"),
        ("_ens_results", metrics_ens, ens_stats, "ensemble"),
    ]:
        per_file = os.path.join(output_dir, f"{stem}{suffix}.json")
        per_result = {
            "args": {
                "task_sequence": task_names,
                "seed": args.seed,
                "method": method_name,
                "eval_max_samples": args.eval_max_samples,
                "num_shots": args.num_shots,
            },
            "accuracy_matrix": tracker.get_accuracy_matrix().tolist(),
            "metrics": {
                "transfer": stats["transfer"],
                "average": stats["average"],
                "last": stats["last"],
            },
            "per_task_metrics": tracker.calculate_per_task_metrics(),
        }
        with open(per_file, 'w', encoding='utf-8') as f:
            json.dump(per_result, f, indent=2)

    # 汇总 JSON（兼容旧格式 + 新增信息）
    save_results = {
        "mode": "Incremental Learning",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_order": task_names,
        "arguments": {k: str(v) if not isinstance(v, (int, float, bool, list, dict, type(None))) else v
                      for k, v in vars(args).items()},
        "args": {
            "task_sequence": task_names,
            "seed": args.seed,
            "method": args.lora_type,
            "eval_max_samples": args.eval_max_samples,
            "num_shots": args.num_shots,
        },
        "accuracy_matrix": metrics_ens.get_accuracy_matrix().tolist(),
        "metrics": {
            "zero_shot": zs_stats,
            "lr_rgda": rgda_stats,
            "ours_ensemble": ens_stats,
        },
        "summary_metrics": {
            "zero_shot": zs_stats,
            "lr_rgda": rgda_stats,
            "ours_ensemble": ens_stats,
        },
        "per_file_results": {
            "zero_shot": os.path.basename(file_name_zs),
            "lr_rgda": os.path.basename(file_name_rgda),
            "ensemble": os.path.basename(file_name_ens),
        },
    }

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)

    logging.info(f"\n增量学习结果已保存至: {save_path}")


if __name__ == "__main__":
    command_line_args = parse_args()
    main(command_line_args)
