"""
集成分类器 Alpha 失调诊断脚本 v2

实验 1：6 种 RGDA 配置的 alpha sweep（加权混合）
  A/B: 分析版（1/4-center）
  C/D: 真实特征微调（1/4-center）
  E/F: GMM 伪特征微调（1/4-center）
实验 2：Logit 分布统计
实验 3：两种集成方式对比（加权混合 vs 加法+指数变换）

指标：per-dataset 平均准确率（10 个数据集等权平均）

用法：
    python debug_ensemble_alpha.py --id_datasets ALL --gpu 0
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import torch
import torch.nn.functional as F
import argparse
import logging
import numpy as np
from sklearn.mixture import GaussianMixture

from src.models.clip import get_clip_model
from src.classifiers.lr_rgda_classifier import LRRGDAClassifier
from src.classifiers.gaussian_statistics import build_multi_center_stats_dict
from src.utils.feature_extractor import extract_features
from src.utils.main_utils import fix_random_seed, get_zeroshot_classifier
from src.utils.data import get_xtail_trainloader, get_xtail_classnames, get_transforms

ALL_XTAIL_DATASETS = [
    "aircraft", "caltech101", "dtd", "eurosat", "flowers",
    "food101", "mnist", "oxford_pets", "stanford_cars", "sun397"
]


def parse_args():
    parser = argparse.ArgumentParser(description="Ensemble Alpha Diagnosis v2")
    parser.add_argument("--id_datasets", type=str, nargs='+', default=ALL_XTAIL_DATASETS)
    parser.add_argument("--root", type=str, default="/data1/open_datasets/X-TAIL")
    parser.add_argument("--num_shots", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--rgda_rank", type=int, default=32)
    parser.add_argument("--rgda_alpha1", type=float, default=0.2)
    parser.add_argument("--rgda_alpha2", type=float, default=2.0)
    parser.add_argument("--rgda_alpha3", type=float, default=0.5)
    parser.add_argument("--rgda_train_iter", type=int, default=200)
    parser.add_argument("--rgda_train_lr", type=float, default=0.01)
    parser.add_argument("--n_alpha", type=int, default=41)
    parser.add_argument("--beta_values", type=float, nargs='+', default=[0.5, 1.0, 2.0, 5.0])
    parser.add_argument("--gmm_k", type=int, default=4)
    parser.add_argument("--gaussian_samples_per_class", type=int, default=16)
    args = parser.parse_args()

    if len(args.id_datasets) == 1 and args.id_datasets[0].upper() == 'ALL':
        args.id_datasets = ALL_XTAIL_DATASETS

    if args.device is None:
        args.device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    return args


def build_rgda_classifier(train_feats, train_labels, args, num_centers, do_fit):
    stats_dict, center_means = build_multi_center_stats_dict(
        train_feats, train_labels, M=num_centers
    )

    per_dataset_covs = []
    offset = 0
    for d_name in args.id_datasets:
        c_names = get_xtail_classnames(args.root, d_name, args.num_shots)
        n_classes = len(c_names)
        ds_cov = sum(stats_dict[cid].cov for cid in range(offset, offset + n_classes)) / n_classes
        per_dataset_covs.append(ds_cov)
        offset += n_classes
    global_cov = sum(per_dataset_covs) / len(per_dataset_covs)

    classifier = LRRGDAClassifier(
        stats_dict=stats_dict, device=args.device,
        rank=args.rgda_rank,
        qda_reg_alpha1=args.rgda_alpha1,
        qda_reg_alpha2=args.rgda_alpha2,
        qda_reg_alpha3=args.rgda_alpha3,
        temperature=1.0,
        M=num_centers, center_means=center_means,
        global_cov=global_cov,
    )

    if do_fit:
        classifier.fit(train_feats.to(args.device), train_labels.to(args.device),
                       iterations=args.rgda_train_iter, lr=args.rgda_train_lr,
                       verbose=False)

    return classifier


def sample_gmm_features(raw_features, labels, n_classes, gmm_k, samples_per_class, device):
    features_np = raw_features.cpu().numpy()
    labels_np = labels.cpu().numpy()
    syn_feats = []
    syn_labels = []

    for cid in range(n_classes):
        mask = labels_np == cid
        feats_c = features_np[mask]

        actual_k = min(gmm_k, len(feats_c)) if len(feats_c) > 0 else 1
        if len(feats_c) == 0:
            continue

        gmm = GaussianMixture(n_components=actual_k, covariance_type='spherical',
                              random_state=42, reg_covar=1e-6)
        gmm.fit(feats_c)

        total = samples_per_class
        n_per_comp = np.maximum(1, (gmm.weights_ * total).round().astype(int))
        diff = total - n_per_comp.sum()
        n_per_comp[np.argmax(gmm.weights_)] += diff

        for comp in range(actual_k):
            mean = torch.from_numpy(gmm.means_[comp]).float()
            var = gmm.covariances_[comp]
            n = max(1, int(n_per_comp[comp]))
            noise = torch.randn(n, mean.shape[0]) * np.sqrt(max(var, 1e-8))
            samples = mean.unsqueeze(0) + noise
            samples = samples / samples.norm(dim=-1, keepdim=True)
            syn_feats.append(samples.to(device))
            syn_labels.append(torch.full((samples.shape[0],), cid, device=device))

    return torch.cat(syn_feats), torch.cat(syn_labels)


def compute_logit_stats(logits, name):
    logits_norm = logits - logits.max(dim=-1, keepdim=True).values
    probs = F.softmax(logits_norm, dim=-1)
    entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean().item()

    top2, _ = logits_norm.topk(2, dim=-1)
    margin = (top2[:, 0] - top2[:, 1]).mean().item()

    std = logits_norm.std(dim=-1).mean().item()
    max_val = logits.max(dim=-1).values.mean().item()
    min_val = logits.min(dim=-1).values.mean().item()
    spread = max_val - min_val

    return {
        "name": name,
        "entropy": entropy,
        "margin": margin,
        "std": std,
        "spread": spread,
    }


def per_dataset_acc(logits, labels, dataset_slices, num_id_classes, alpha, method="weighted", beta=1.0):
    zs_norm = logits[:, :num_id_classes] - logits[:, :num_id_classes].max(dim=-1, keepdim=True).values
    rgda_norm = logits[:, num_id_classes:] - logits[:, num_id_classes:].max(dim=-1, keepdim=True).values

    accs = []
    for start, end in dataset_slices:
        zs = zs_norm[start:end]
        rgda = rgda_norm[start:end]
        lbl = labels[start:end]

        if method == "weighted":
            ens = zs * (1 - alpha) + alpha * rgda
        elif method == "additive_exp":
            rgda_t = torch.exp(-beta * (1 - rgda))
            ens = zs + alpha * rgda_t
        else:
            raise ValueError(f"Unknown method: {method}")

        acc = ens.argmax(dim=1).eq(lbl).float().mean().item() * 100
        accs.append(acc)
    return sum(accs) / len(accs)


def alpha_sweep_per_dataset(combined_logits, test_labels, dataset_slices,
                            num_id_classes, alphas, method="weighted", beta=1.0):
    results = []
    for a in alphas:
        acc = per_dataset_acc(combined_logits, test_labels, dataset_slices,
                              num_id_classes, a, method, beta)
        results.append((round(a, 4), round(acc, 2)))
    return results


def _extract_features_for_branch(model, args, all_class_names, dataset_slices, use_test_transform):
    """提取训练特征（单分支）+ 测试特征。返回 (train_feats_norm_gpu, train_labels_gpu, raw_train_feats_cpu, test_feats_norm_gpu, test_labels_gpu, zs_test_logits_gpu)"""
    all_train_feats, all_train_labels = [], []
    all_test_feats, all_test_labels = [], []
    all_cn = []
    ds_slices = []
    offset = 0
    test_offset = 0

    for d_name in args.id_datasets:
        train_transform, test_transform = get_transforms(d_name)
        tr_transform = test_transform if use_test_transform else train_transform
        tr_loader, _, te_loader, c_names = get_xtail_trainloader(
            root=args.root, dataset_name=d_name,
            transform_train=tr_transform, transform_test=test_transform,
            num_shots=args.num_shots, batch_size=args.batch_size
        )
        tr_feats, tr_lbls = extract_features(model, tr_loader, args.device)
        te_feats, te_lbls = extract_features(model, te_loader, args.device)
        all_train_feats.append(tr_feats)
        all_train_labels.append(tr_lbls + offset)
        all_test_feats.append(te_feats)
        all_test_labels.append(te_lbls + offset)
        ds_slices.append((test_offset, test_offset + te_feats.shape[0]))
        all_cn.extend(c_names)
        offset += len(c_names)
        test_offset += te_feats.shape[0]
        logging.info(f"  {d_name}: train={tr_feats.shape[0]}, test={te_feats.shape[0]}, classes={len(c_names)}")

    train_feats_raw = torch.cat(all_train_feats)
    train_labels_cpu = torch.cat(all_train_labels)
    test_feats = torch.cat(all_test_feats)
    test_labels = torch.cat(all_test_labels)
    num_id_classes = len(all_cn)

    train_feats_norm = (train_feats_raw / train_feats_raw.norm(dim=-1, keepdim=True)).to(args.device)
    test_feats_norm = (test_feats / test_feats.norm(dim=-1, keepdim=True)).to(args.device)
    train_labels_gpu = train_labels_cpu.to(args.device)
    test_labels_gpu = test_labels.to(args.device)

    all_class_names[:] = all_cn
    dataset_slices[:] = ds_slices

    return (train_feats_norm, train_labels_gpu, train_feats_raw, train_labels_cpu,
            test_feats_norm, test_labels_gpu, num_id_classes)


def _build_and_eval_classifiers(args, branch_label, train_feats_norm, train_labels_gpu,
                                 train_feats_raw, train_labels_cpu,
                                 test_feats_norm, test_labels_gpu,
                                 zs_test_logits, zs_cls, num_id_classes,
                                 dataset_slices, zs_per_ds, zs_avg, configs, alphas):
    """构建 6 种 RGDA 配置，逐数据集 forward，返回 results dict。"""
    results = {}

    logging.info("\nBuilding analytical classifier (shared for A & B)...")
    analytical_classifier = build_rgda_classifier(
        train_feats_norm.cpu(), train_labels_gpu.cpu(), args, num_centers=1, do_fit=False
    )

    logging.info("Building B: 4-center analytical...")
    stats_dict_b, center_means_b = build_multi_center_stats_dict(
        train_feats_norm.cpu(), train_labels_gpu.cpu(), M=4
    )
    per_dataset_covs_b = []
    off = 0
    for d_name in args.id_datasets:
        c_names = get_xtail_classnames(args.root, d_name, args.num_shots)
        n_c = len(c_names)
        ds_cov = sum(stats_dict_b[cid].cov for cid in range(off, off + n_c)) / n_c
        per_dataset_covs_b.append(ds_cov)
        off += n_c
    global_cov_b = sum(per_dataset_covs_b) / len(per_dataset_covs_b)
    classifier_b = LRRGDAClassifier(
        stats_dict=stats_dict_b, device=args.device,
        rank=args.rgda_rank,
        qda_reg_alpha1=args.rgda_alpha1,
        qda_reg_alpha2=args.rgda_alpha2,
        qda_reg_alpha3=args.rgda_alpha3,
        temperature=1.0,
        M=4, center_means=center_means_b,
        global_cov=global_cov_b,
    )

    logging.info("Sampling GMM pseudo features...")
    raw_train_feats_norm = train_feats_raw / train_feats_raw.norm(dim=-1, keepdim=True)
    gmm_feats, gmm_labels = sample_gmm_features(
        raw_train_feats_norm, train_labels_cpu, num_id_classes,
        args.gmm_k, args.gaussian_samples_per_class, args.device
    )
    logging.info(f"GMM sampled: {gmm_feats.shape[0]} features for {num_id_classes} classes")

    for cfg_name, n_centers, do_fit, use_gmm in configs:
        logging.info(f"\n{'='*60}")
        logging.info(f"[{branch_label}] Config: {cfg_name}")
        logging.info(f"  centers={n_centers}, fit={do_fit}, gmm={use_gmm}")

        if cfg_name.startswith("A"):
            classifier = analytical_classifier
        elif cfg_name.startswith("B"):
            classifier = classifier_b
        else:
            fit_feats = gmm_feats if use_gmm else train_feats_norm
            fit_labels_use = gmm_labels if use_gmm else train_labels_gpu
            classifier = build_rgda_classifier(
                fit_feats.cpu(), fit_labels_use.cpu(), args, n_centers, do_fit
            )

        # 逐数据集 forward（避免 OOM）
        rgda_logits_list = []
        rgda_per_ds = []
        with torch.no_grad():
            for start, end in dataset_slices:
                batch = test_feats_norm[start:end]
                rgda_l = classifier.forward(batch)
                rgda_l_norm = rgda_l - rgda_l.max(dim=-1, keepdim=True).values
                acc = rgda_l_norm.argmax(dim=1).eq(test_labels_gpu[start:end]).float().mean().item() * 100
                rgda_per_ds.append(acc)
                rgda_logits_list.append(rgda_l.cpu())
                del rgda_l, rgda_l_norm
                torch.cuda.empty_cache()

        rgda_test_logits = torch.cat(rgda_logits_list).to(args.device)
        rgda_avg = sum(rgda_per_ds) / len(rgda_per_ds)

        combined_logits = torch.cat([zs_test_logits, rgda_test_logits], dim=1)

        sweep_w = alpha_sweep_per_dataset(
            combined_logits, test_labels_gpu, dataset_slices,
            num_id_classes, alphas, method="weighted"
        )
        best_w = max(sweep_w, key=lambda x: x[1])

        sweep_ae = {}
        best_ae = {}
        for beta in args.beta_values:
            sw = alpha_sweep_per_dataset(
                combined_logits, test_labels_gpu, dataset_slices,
                num_id_classes, alphas, method="additive_exp", beta=beta
            )
            sweep_ae[beta] = sw
            best_ae[beta] = max(sw, key=lambda x: x[1])

        rgda_stats = compute_logit_stats(rgda_test_logits, "RGDA")

        results[cfg_name] = {
            "rgda_avg": rgda_avg,
            "rgda_per_ds": rgda_per_ds,
            "weighted": {"sweep": sweep_w, "best": best_w},
            "additive_exp": {"sweep": sweep_ae, "best": best_ae},
            "rgda_stats": rgda_stats,
        }

        del rgda_logits_list, rgda_test_logits, combined_logits
        torch.cuda.empty_cache()

    return results


def _print_all_tables(branch_label, configs, results, zs_avg, zs_per_ds, zs_test_logits, args):
    """输出实验 1-3 + 详细表格"""

    def print_section(tag):
        if branch_label:
            print(f"\n{'=' * 120}")
            print(f"=== {branch_label} — {tag} ===")
            print(f"{'=' * 120}")

    # --- 实验 1 ---
    print_section("实验 1")
    print(f"\n{'Config':<25s} | {'RGDA':>7s} | {'Best α':>8s} | {'Best Ens':>8s} | {'α=0.5':>8s} | {'ZS avg':>7s}")
    print("-" * 120)
    for cfg_name, _, _, _ in configs:
        r = results[cfg_name]
        best_a, best_acc = r["weighted"]["best"]
        sweep_dict = dict(r["weighted"]["sweep"])
        closest_05 = min(sweep_dict.keys(), key=lambda x: abs(x - 0.5))
        acc_05 = sweep_dict[closest_05]
        print(f"{cfg_name:<25s} | {r['rgda_avg']:>5.1f}%  | {best_a:>7.3f}  | {best_acc:>6.1f}%  | {acc_05:>6.1f}%  | {zs_avg:>5.1f}%")

    # --- 实验 2 ---
    print_section("实验 2")
    print(f"\n{'Config':<25s} | {'Classifier':>10s} | {'Entropy':>8s} | {'Margin':>8s} | {'Std':>8s} | {'Spread':>8s}")
    print("-" * 120)
    print(f"{'ZS':<25s} | {'ZS':>10s} | {compute_logit_stats(zs_test_logits, 'ZS')['entropy']:>8.3f} | "
          f"{compute_logit_stats(zs_test_logits, 'ZS')['margin']:>8.4f} | "
          f"{compute_logit_stats(zs_test_logits, 'ZS')['std']:>8.4f} | "
          f"{compute_logit_stats(zs_test_logits, 'ZS')['spread']:>8.2f}")
    for cfg_name, _, _, _ in configs:
        s = results[cfg_name]["rgda_stats"]
        print(f"{cfg_name:<25s} | {s['name']:>10s} | {s['entropy']:>8.3f} | {s['margin']:>8.4f} | {s['std']:>8.4f} | {s['spread']:>8.2f}")

    # --- 实验 3 ---
    print_section("实验 3")
    print(f"\n{'Config':<25s} | {'Method':<25s} | {'Best α':>8s} | {'Best Acc':>8s} | {'α=0.5':>8s}")
    print("-" * 120)
    for cfg_name, _, _, _ in configs:
        r = results[cfg_name]
        best_a, best_acc = r["weighted"]["best"]
        sweep_dict = dict(r["weighted"]["sweep"])
        closest_05 = min(sweep_dict.keys(), key=lambda x: abs(x - 0.5))
        acc_05 = sweep_dict[closest_05]
        print(f"{cfg_name:<25s} | {'加权混合':<25s} | {best_a:>7.3f}  | {best_acc:>6.1f}%  | {acc_05:>6.1f}%")

        for beta in args.beta_values:
            ba, bac = r["additive_exp"]["best"][beta]
            sd = dict(r["additive_exp"]["sweep"][beta])
            c05 = min(sd.keys(), key=lambda x: abs(x - 0.5))
            a05 = sd[c05]
            print(f"{cfg_name:<25s} | {'加法+exp(β='+str(beta)+')':<25s} | {ba:>7.3f}  | {bac:>6.1f}%  | {a05:>6.1f}%")
        print("-" * 120)

    # --- 详细 Alpha Sweep ---
    print_section("详细 Alpha Sweep")
    header = f"\n{'Alpha':>8s}"
    for cfg_name, _, _, _ in configs:
        header += f" | {cfg_name:>18s}"
    print(header)
    print("-" * 120)

    key_alphas = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                  0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    for a in key_alphas:
        row = f"{a:>8.2f}"
        for cfg_name, _, _, _ in configs:
            sweep_dict = dict(results[cfg_name]["weighted"]["sweep"])
            closest_a = min(sweep_dict.keys(), key=lambda x: abs(x - a))
            acc = sweep_dict[closest_a]
            row += f" | {acc:>16.1f}%"
        print(row)

    # --- Per-dataset 详细表 ---
    print_section("Per-dataset 详细准确率")
    header2 = f"\n{'Dataset':<15s} | {'ZS':>7s}"
    for cfg_name, _, _, _ in configs:
        header2 += f" | {cfg_name:>14s}"
    print(header2)
    print("-" * 120)
    for i, d_name in enumerate(args.id_datasets):
        row = f"{d_name:<15s} | {zs_per_ds[i]:>5.1f}%"
        for cfg_name, _, _, _ in configs:
            rgda_d = results[cfg_name]["rgda_per_ds"][i]
            row += f" | {rgda_d:>12.1f}%"
        print(row)
    print("-" * 120)
    row_avg = f"{'Average':<15s} | {zs_avg:>5.1f}%"
    for cfg_name, _, _, _ in configs:
        row_avg += f" | {results[cfg_name]['rgda_avg']:>12.1f}%"
    print(row_avg)

    # --- 最佳加法+指数变换 ---
    print_section("最佳加法+指数变换配置")
    for cfg_name, _, _, _ in configs:
        r = results[cfg_name]
        best_beta = None
        best_overall = -1
        for beta in args.beta_values:
            _, acc = r["additive_exp"]["best"][beta]
            if acc > best_overall:
                best_overall = acc
                best_beta = beta
        ba, bac = r["additive_exp"]["best"][best_beta]
        print(f"{cfg_name:<25s} | β={best_beta:.1f}, α={ba:.3f}, Acc={bac:.1f}%")


def main(args):
    fix_random_seed(args.seed)
    logging.info(f"Device: {args.device}")

    logging.info("Loading CLIP model (frozen)...")
    dummy_args = argparse.Namespace(
        lora_type='lora_vanilla', lora_rank=4,
        tune_vision_encoder=False, tune_text_encoder=False
    )
    model, processor = get_clip_model(dummy_args, train_mode='frozen')
    model = model.to(args.device)
    model.eval()

    all_class_names = []
    dataset_slices = []

    configs = [
        ("A: 1c analytical",       1, False, False),
        ("B: 4c analytical",       4, False, False),
        ("C: 1c fit(real)",        1, True,  False),
        ("D: 4c fit(real)",        4, True,  False),
        ("E: 1c fit(GMM)",         1, True,  True),
        ("F: 4c fit(GMM)",         4, True,  True),
    ]
    alphas = torch.linspace(0, 1.0, args.n_alpha).tolist()

    branches = [
        ("train_transform", False),
        ("test_transform", True),
    ]

    zs_classifier = None
    zs_test_logits = None
    zs_per_ds = None
    zs_avg = None
    test_feats_shared = None
    test_labels_shared = None
    num_id_classes = None

    all_branch_results = {}

    for branch_label, use_test_transform in branches:
        logging.info(f"\n{'#'*60}")
        logging.info(f"Branch: {branch_label}")
        logging.info(f"{'#'*60}")

        (train_feats_norm, train_labels_gpu, train_feats_raw, train_labels_cpu,
         test_feats_norm, test_labels_gpu, ncid) = _extract_features_for_branch(
            model, args, all_class_names, dataset_slices, use_test_transform
        )
        num_id_classes = ncid

        if zs_classifier is None:
            zs_classifier = get_zeroshot_classifier(model, processor, all_class_names, args.device)
            test_feats_shared = test_feats_norm
            test_labels_shared = test_labels_gpu

            with torch.no_grad():
                zs_test_logits = test_feats_norm @ zs_classifier

            zs_per_ds = []
            for start, end in dataset_slices:
                zs_l = zs_test_logits[start:end]
                zs_l_norm = zs_l - zs_l.max(dim=-1, keepdim=True).values
                acc = zs_l_norm.argmax(dim=1).eq(test_labels_gpu[start:end]).float().mean().item() * 100
                zs_per_ds.append(acc)
            zs_avg = sum(zs_per_ds) / len(zs_per_ds)
            logging.info(f"Zero-shot per-dataset avg: {zs_avg:.1f}%")
        else:
            test_feats_norm = test_feats_shared
            test_labels_gpu = test_labels_shared

        branch_results = _build_and_eval_classifiers(
            args, branch_label, train_feats_norm, train_labels_gpu,
            train_feats_raw, train_labels_cpu,
            test_feats_norm, test_labels_gpu, zs_test_logits, zs_classifier,
            num_id_classes, dataset_slices, zs_per_ds, zs_avg, configs, alphas
        )
        all_branch_results[branch_label] = branch_results

        _print_all_tables(branch_label, configs, branch_results, zs_avg, zs_per_ds, zs_test_logits, args)

        del train_feats_norm, train_labels_gpu, train_feats_raw, train_labels_cpu
        torch.cuda.empty_cache()

    # ========== 最终对比总结 ==========
    print("\n\n" + "=" * 120)
    print("=== 双分支对比总结 ===")
    print("=" * 120)
    print(f"\n{'Config':<25s} | {'RGDA(train_t)':>13s} | {'RGDA(test_t)':>13s} | {'Diff':>8s}")
    print("-" * 75)
    train_r = all_branch_results.get("train_transform", {})
    test_r = all_branch_results.get("test_transform", {})
    for cfg_name, _, _, _ in configs:
        tv = train_r.get(cfg_name, {}).get("rgda_avg", 0)
        sv = test_r.get(cfg_name, {}).get("rgda_avg", 0)
        print(f"{cfg_name:<25s} | {tv:>11.1f}%  | {sv:>12.1f}%  | {sv-tv:>+7.1f}%")


if __name__ == "__main__":
    args = parse_args()
    main(args)
