"""
LADA 分类器评估脚本 v3

实验 1：8 种 LADA 配置的 alpha sweep（加权混合 ZS+LADA）
  A: k=1 分析版     B: k=4 分析版     C: k=16 分析版
  D: k=1 微调       E: k=4 微调       F: k=16 微调
  G: k=4 GMM采样    H: k=16 GMM采样
实验 2：Logit 分布统计
实验 3：两种集成方式对比（加权混合 vs 加法+指数变换）
实验 4：train_transform vs test_transform 双分支对比
实验 5：ID/OOD 分半评估

用法：
    python debug_lada_eval.py --id_datasets ALL --gpu 0                     # 默认（全部 ID）
    python debug_lada_eval.py --id_datasets ALL --gpu 0 --id_ratio 0.5      # 半ID评估
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import logging
import numpy as np
from sklearn.mixture import GaussianMixture

from src.models.clip import get_clip_model
from src.lada import LADAClassifier
from src.utils.feature_extractor import extract_features
from src.utils.main_utils import fix_random_seed, get_zeroshot_classifier
from src.utils.data import get_xtail_trainloader, get_xtail_classnames, get_transforms

from tqdm import tqdm


def _extract_raw_features(model, dataloader, device):
    """提取未归一化的 CLIP 特征（绕过 extract_features 的 L2 归一化）"""
    model.eval()
    features, labels_list = [], []
    for images, lbls in tqdm(dataloader, desc="Extracting raw features"):
        images = images.to(device)
        with torch.no_grad():
            vision_outputs = model.vision_model(images)
            if hasattr(vision_outputs, 'pooler_output') and vision_outputs.pooler_output is not None:
                pooled = vision_outputs.pooler_output
            else:
                pooled = vision_outputs[1]
            feats = model.visual_projection(pooled)
        features.append(feats.cpu())
        labels_list.append(lbls.cpu())
    return torch.cat(features), torch.cat(labels_list)

ALL_XTAIL_DATASETS = [
    "aircraft", "caltech101", "dtd", "eurosat", "flowers",
    "food101", "mnist", "oxford_pets", "stanford_cars", "sun397"
]


def parse_args():
    parser = argparse.ArgumentParser(description="LADA Evaluation v2")
    parser.add_argument("--id_datasets", type=str, nargs='+', default=ALL_XTAIL_DATASETS)
    parser.add_argument("--root", type=str, default="/data1/open_datasets/X-TAIL")
    parser.add_argument("--num_shots", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--lada_beta", type=float, default=1.0)
    parser.add_argument("--lada_train_iter", type=int, default=200)
    parser.add_argument("--lada_train_lr", type=float, default=0.01)
    parser.add_argument("--n_alpha", type=int, default=41)
    parser.add_argument("--beta_values", type=float, nargs='+', default=[0.5, 1.0, 2.0, 5.0])
    parser.add_argument("--id_ratio", type=float, default=1.0,
                        help="ID 数据集比例（1.0=全部ID, 0.5=前5个ID后5个OOD）")
    parser.add_argument("--gmm_k", type=int, default=4,
                        help="GMM 初始化分量数（配置 G 用 k=4, H 用 k=16，均用此值作为 GMM 分量数）")
    args = parser.parse_args()

    if len(args.id_datasets) == 1 and args.id_datasets[0].upper() == 'ALL':
        args.id_datasets = ALL_XTAIL_DATASETS

    n = len(args.id_datasets)
    n_id = max(1, int(n * args.id_ratio))
    args.id_list = args.id_datasets[:n_id]
    args.ood_list = args.id_datasets[n_id:] if n_id < n else []
    logging.info(f"ID={args.id_list}, OOD={args.ood_list}")

    if args.device is None:
        args.device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    return args


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
    return {"name": name, "entropy": entropy, "margin": margin, "std": std, "spread": spread}


def build_lada_classifier(features, labels, args, k, do_fit):
    feature_dim = features.shape[1]
    classifier = LADAClassifier(feature_dim=feature_dim, beta=args.lada_beta)
    classifier.build_from_data(features.to(args.device), labels.to(args.device), k=k, label_offset=0)
    classifier.to(args.device)
    if do_fit and args.lada_train_iter > 0:
        classifier.fit(features.to(args.device), labels.to(args.device),
                       iterations=args.lada_train_iter, lr=args.lada_train_lr, verbose=False)
    return classifier


def build_lada_from_gmm_samples(raw_feats, norm_feats, labels, args, k, do_fit):
    """
    用 GMM 采样伪特征构建 LADA 分类器

    流程：
    1. 在原始（未归一化）特征上每类拟合 spherical GMM(k)
    2. 从 GMM 采样大量伪特征（每类 16*k 个）
    3. L2 归一化伪特征
    4. 用 GMM 分量均值直接初始化 LADA 的 curr_lada_features
    5. 在伪特征上梯度微调
    """
    device = args.device
    raw_np = raw_feats.cpu().numpy()
    labels_np = labels.cpu().numpy()
    feature_dim = raw_feats.shape[1]

    unique_labels = sorted(set(int(l) for l in labels_np))
    num_classes = len(unique_labels)

    SAMPLE_MULTIPLIER = 4  # 每 GMM 分量采样 4 个伪特征（16→4 避免 OOM）

    all_gmm_centers, all_gmm_labels = [], []
    all_syn_feats, all_syn_labels = [], []

    for lbl in unique_labels:
        mask = labels_np == lbl
        raw_c = raw_np[mask]
        actual_k = min(k, len(raw_c))

        gmm = GaussianMixture(n_components=actual_k, covariance_type='spherical', random_state=42)
        gmm.fit(raw_c)

        # 收集 GMM 分量均值作为 LADA 初始中心
        all_gmm_centers.append(gmm.means_)
        all_gmm_labels.append(np.full(actual_k, lbl, dtype=np.int64))

        # 每分量采样 SAMPLE_MULTIPLIER 个伪特征
        samples_per_comp = max(1, int(SAMPLE_MULTIPLIER * len(raw_c) / actual_k))
        comp_samples, _ = gmm.sample(samples_per_comp * actual_k)
        all_syn_feats.append(comp_samples)
        all_syn_labels.append(np.full(samples_per_comp * actual_k, lbl, dtype=np.int64))

    # GMM 均值 → LADA 初始中心
    centers_array = np.concatenate(all_gmm_centers, axis=0)
    center_labels_array = np.concatenate(all_gmm_labels)
    centers_t = torch.from_numpy(centers_array).float()
    centers_t = centers_t / centers_t.norm(dim=-1, keepdim=True)

    # 采样伪特征
    syn_array = np.concatenate(all_syn_feats, axis=0)
    syn_labels_array = np.concatenate(all_syn_labels)
    syn_t = torch.from_numpy(syn_array).float()
    syn_t = syn_t / syn_t.norm(dim=-1, keepdim=True)  # L2 归一化
    syn_t = syn_t.to(device)
    syn_labels_t = torch.from_numpy(syn_labels_array).long().to(device)

    logging.info(f"GMM sampled: {syn_t.shape[0]} pseudo features for {num_classes} classes "
                 f"(k={k}, {SAMPLE_MULTIPLIER}× per component)")

    classifier = LADAClassifier(feature_dim=feature_dim, beta=args.lada_beta)
    classifier.curr_lada_features = nn.Parameter(centers_t.t().to(device))

    curr_clf = F.one_hot(torch.from_numpy(center_labels_array).long(), num_classes=num_classes).float().to(device)
    classifier.curr_classifier = curr_clf
    classifier.register_buffer('joint_classifier', curr_clf)
    classifier.num_curr_classes = num_classes
    classifier.to(device)

    if do_fit and args.lada_train_iter > 0:
        classifier.fit(syn_t, syn_labels_t,
                       iterations=args.lada_train_iter, lr=args.lada_train_lr, verbose=False)

    return classifier


def _extract_features_for_branch(model, args, all_id_names, all_ood_names, dataset_slices,
                                  ood_slices, use_test_transform):
    all_train_feats, all_train_labels, all_train_raw = [], [], []
    all_test_feats, all_test_labels = [], []
    ood_test_feats, ood_test_labels = [], []
    id_cn, ood_cn = [], []
    ds_slices, od_slices = [], []
    offset, test_offset = 0, 0
    ood_offset = 0
    ood_test_offset = 0

    for d_name in args.id_list + args.ood_list:
        train_transform, test_transform = get_transforms(d_name)
        tr_transform = test_transform if use_test_transform else train_transform
        tr_loader, _, te_loader, c_names = get_xtail_trainloader(
            root=args.root, dataset_name=d_name,
            transform_train=tr_transform, transform_test=test_transform,
            num_shots=args.num_shots, batch_size=args.batch_size
        )

        is_id = d_name in args.id_list

        if is_id:
            tr_feats, tr_lbls = extract_features(model, tr_loader, args.device)
            all_train_feats.append(tr_feats)
            all_train_labels.append(tr_lbls + offset)
            # 提取原始（未归一化）特征用于 GMM
            tr_raw, _ = _extract_raw_features(model, tr_loader, args.device)
            all_train_raw.append(tr_raw)
            offset += len(c_names)
            id_cn.extend(c_names)

        te_feats, te_lbls = extract_features(model, te_loader, args.device)

        if is_id:
            all_test_feats.append(te_feats)
            all_test_labels.append(te_lbls + (offset - len(c_names)))
            ds_slices.append((test_offset, test_offset + te_feats.shape[0]))
            test_offset += te_feats.shape[0]
        else:
            ood_test_feats.append(te_feats)
            ood_test_labels.append(te_lbls)
            od_slices.append((ood_test_offset, ood_test_offset + te_feats.shape[0]))
            ood_test_offset += te_feats.shape[0]
            ood_cn.extend(c_names)
            ood_offset += len(c_names)

        logging.info(f"  {d_name}: {'ID' if is_id else 'OOD'}, train={tr_feats.shape[0] if is_id else 'N/A'}, "
                     f"test={te_feats.shape[0]}, classes={len(c_names)}")

    train_feats = torch.cat(all_train_feats)
    train_labels_gpu = torch.cat(all_train_labels).to(args.device)
    train_raw_feats = torch.cat(all_train_raw).to(args.device) if all_train_raw else torch.empty(0)
    id_test_feats = torch.cat(all_test_feats) if all_test_feats else torch.empty(0)
    id_test_labels = torch.cat(all_test_labels).to(args.device) if all_test_labels else torch.empty(0, dtype=torch.long)

    train_feats_norm = (train_feats / train_feats.norm(dim=-1, keepdim=True)).to(args.device)
    id_test_feats_norm = (id_test_feats / id_test_feats.norm(dim=-1, keepdim=True)).to(args.device)

    ood_test_feats_cat = torch.cat(ood_test_feats) if ood_test_feats else torch.empty(0)
    ood_test_labels_cat = torch.cat(ood_test_labels).to(args.device) if ood_test_labels else torch.empty(0, dtype=torch.long)
    ood_test_feats_norm = (ood_test_feats_cat / ood_test_feats_cat.norm(dim=-1, keepdim=True)).to(args.device) if ood_test_feats else None

    all_id_names[:] = id_cn
    all_ood_names[:] = ood_cn
    dataset_slices[:] = ds_slices
    ood_slices[:] = od_slices
    num_id_classes = len(id_cn)

    return (train_feats_norm, train_labels_gpu, train_raw_feats,
            id_test_feats_norm, id_test_labels, num_id_classes,
            ood_test_feats_norm, ood_test_labels_cat)


def per_dataset_acc_zs_lada(zs_logits, lada_logits, labels, dataset_slices,
                             num_all_classes, alpha, method="weighted", beta=1.0):
    """zs_logits: [N, C_all], lada_logits: [N, C_id]. Ensemble with lada padded to C_all."""
    n_id = lada_logits.shape[1]
    zs_norm = zs_logits - zs_logits.max(dim=-1, keepdim=True).values
    lada_padded = torch.zeros_like(zs_logits)
    lada_padded[:, :n_id] = lada_logits - lada_logits.max(dim=-1, keepdim=True).values

    accs = []
    for start, end in dataset_slices:
        zs = zs_norm[start:end]
        lada = lada_padded[start:end]
        lbl = labels[start:end]
        if method == "weighted":
            ens = zs * (1 - alpha) + alpha * lada
        elif method == "additive_exp":
            lada_t = torch.exp(-beta * (1 - lada))
            ens = zs + alpha * lada_t
        else:
            raise ValueError(f"Unknown method: {method}")
        acc = ens.argmax(dim=1).eq(lbl).float().mean().item() * 100
        accs.append(acc)
    return sum(accs) / len(accs)


def alpha_sweep_lada(zs_logits, lada_logits, labels, dataset_slices,
                      num_all_classes, alphas, method="weighted", beta=1.0):
    results = []
    for a in alphas:
        acc = per_dataset_acc_zs_lada(zs_logits, lada_logits, labels, dataset_slices,
                                       num_all_classes, a, method, beta)
        results.append((round(a, 4), round(acc, 2)))
    return results


def _eval_and_sweep(args, classifier, zs_id_logits, zs_all_logits, test_feats_norm, test_labels, dataset_slices,
                     num_id, num_all, alphas, is_id):
    lada_logits_list = []
    lada_per_ds = []
    with torch.no_grad():
        for start, end in dataset_slices:
            batch = test_feats_norm[start:end]
            ll = classifier(batch)
            ll_norm = ll - ll.max(dim=-1, keepdim=True).values
            acc = ll_norm.argmax(dim=1).eq(test_labels[start:end]).float().mean().item() * 100
            lada_per_ds.append(acc)
            lada_logits_list.append(ll.cpu())
            del ll, ll_norm
            torch.cuda.empty_cache()

    lada_test_logits = torch.cat(lada_logits_list).to(args.device)
    lada_avg = sum(lada_per_ds) / len(lada_per_ds)

    zs_use = zs_id_logits if is_id else zs_all_logits
    combined_logits = torch.cat([zs_use, lada_test_logits], dim=1)
    n_all = num_all if not is_id else num_id

    sweep_w = alpha_sweep_lada(zs_use, lada_test_logits, test_labels, dataset_slices,
                                  n_all, alphas, method="weighted") if is_id else \
              alpha_sweep_lada(zs_use, lada_test_logits, test_labels, dataset_slices,
                                  n_all, alphas, method="weighted")
    best_w = max(sweep_w, key=lambda x: x[1])

    sweep_ae, best_ae = {}, {}
    for beta_val in args.beta_values:
        sw = alpha_sweep_lada(zs_use, lada_test_logits, test_labels, dataset_slices,
                                n_all, alphas, method="additive_exp", beta=beta_val)
        sweep_ae[beta_val] = sw
        best_ae[beta_val] = max(sw, key=lambda x: x[1])

    lada_stats = compute_logit_stats(lada_test_logits, "LADA")

    result = {
        "lada_avg": lada_avg,
        "lada_per_ds": lada_per_ds,
        "weighted": {"sweep": sweep_w, "best": best_w},
        "additive_exp": {"sweep": sweep_ae, "best": best_ae},
        "lada_stats": lada_stats,
    }
    return result, lada_test_logits


def _print_all_tables(branch_label, configs, results, zs_avg, zs_per_ds, zs_test_logits, args):
    def print_section(tag):
        print(f"\n{'=' * 120}")
        print(f"=== {branch_label} — {tag} ===")
        print(f"{'=' * 120}")

    print_section("实验 1")
    print(f"\n{'Config':<25s} | {'LADA':>7s} | {'Best α':>8s} | {'Best Ens':>8s} | {'α=0.5':>8s} | {'ZS avg':>7s}")
    print("-" * 120)
    for cfg_name, _, _, _ in configs:
        r = results[cfg_name]
        best_a, best_acc = r["weighted"]["best"]
        sweep_dict = dict(r["weighted"]["sweep"])
        closest_05 = min(sweep_dict.keys(), key=lambda x: abs(x - 0.5))
        acc_05 = sweep_dict[closest_05]
        print(f"{cfg_name:<25s} | {r['lada_avg']:>5.1f}%  | {best_a:>7.3f}  | {best_acc:>6.1f}%  | {acc_05:>6.1f}%  | {zs_avg:>5.1f}%")

    print_section("实验 2")
    print(f"\n{'Config':<25s} | {'Classifier':>10s} | {'Entropy':>8s} | {'Margin':>8s} | {'Std':>8s} | {'Spread':>8s}")
    print("-" * 120)
    zs_stats = compute_logit_stats(zs_test_logits, 'ZS')
    print(f"{'ZS':<25s} | {'ZS':>10s} | {zs_stats['entropy']:>8.3f} | {zs_stats['margin']:>8.4f} | {zs_stats['std']:>8.4f} | {zs_stats['spread']:>8.2f}")
    for cfg_name, _, _, _ in configs:
        s = results[cfg_name]["lada_stats"]
        print(f"{cfg_name:<25s} | {s['name']:>10s} | {s['entropy']:>8.3f} | {s['margin']:>8.4f} | {s['std']:>8.4f} | {s['spread']:>8.2f}")

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
        for beta_val in args.beta_values:
            ba, bac = r["additive_exp"]["best"][beta_val]
            sd = dict(r["additive_exp"]["sweep"][beta_val])
            c05 = min(sd.keys(), key=lambda x: abs(x - 0.5))
            a05 = sd[c05]
            print(f"{cfg_name:<25s} | {'加法+exp(β='+str(beta_val)+')':<25s} | {ba:>7.3f}  | {bac:>6.1f}%  | {a05:>6.1f}%")
        print("-" * 120)

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

    print_section("Per-dataset 详细准确率")
    id_list = getattr(args, 'id_list', args.id_datasets)
    header2 = f"\n{'Dataset':<15s} | {'ZS':>7s}"
    for cfg_name, _, _, _ in configs:
        header2 += f" | {cfg_name:>14s}"
    print(header2)
    print("-" * 120)
    for i, d_name in enumerate(id_list):
        row = f"{d_name:<15s} | {zs_per_ds[i]:>5.1f}%"
        for cfg_name, _, _, _ in configs:
            ld = results[cfg_name]["lada_per_ds"][i]
            row += f" | {ld:>12.1f}%"
        print(row)
    print("-" * 120)
    row_avg = f"{'Average':<15s} | {zs_avg:>5.1f}%"
    for cfg_name, _, _, _ in configs:
        row_avg += f" | {results[cfg_name]['lada_avg']:>12.1f}%"
    print(row_avg)


def _print_ood_table(branch_label, configs, id_results, ood_results, ood_zs_avg, ood_zs_per_ds, args):
    """打印 ID/OOD 干扰测试表"""
    print(f"\n{'=' * 120}")
    print(f"=== {branch_label} — ID/OOD 干扰测试 ===")
    print(f"{'=' * 120}")
    print(f"\n{'Config':<25s} | {'ID LADA':>8s} | {'ID Ens(best)':>12s} | "
          f"{'OOD ZS':>8s} | {'OOD Ens(best)':>12s} | {'Δ OOD':>8s}")
    print("-" * 85)

    for cfg_name, _, _, _ in configs:
        ir = id_results[cfg_name]
        oo = ood_results[cfg_name]
        id_best_a, id_best_acc = ir["weighted"]["best"]
        ood_best_a, ood_best_acc = oo["weighted"]["best"]
        delta = ood_best_acc - ood_zs_avg
        print(f"{cfg_name:<25s} | {ir['lada_avg']:>6.1f}% | {id_best_acc:>10.1f}% | "
              f"{ood_zs_avg:>6.1f}% | {ood_best_acc:>10.1f}% | {delta:>+7.1f}%")

    # Per-dataset OOD 详细
    print(f"\n{'Dataset':<15s} | {'ZS':>7s}", end="")
    for cfg_name, _, _, _ in configs:
        print(f" | {cfg_name:>14s}", end="")
    print()
    print("-" * 120)
    ood_list = getattr(args, 'ood_list', [])
    for i, d_name in enumerate(ood_list):
        row = f"{d_name:<15s} | {ood_zs_per_ds[i]:>5.1f}%"
        for cfg_name, _, _, _ in configs:
            ld = ood_results[cfg_name]["lada_per_ds"][i]
            row += f" | {ld:>12.1f}%"
        print(row)
    print("-" * 120)
    row_avg = f"{'Average':<15s} | {ood_zs_avg:>5.1f}%"
    for cfg_name, _, _, _ in configs:
        row_avg += f" | {ood_results[cfg_name]['lada_avg']:>12.1f}%"
    print(row_avg)


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

    all_id_names, all_ood_names = [], []
    ds_slices, od_slices = [], []
    has_ood = len(args.ood_list) > 0

    configs = [
        ("A: k=1 analytical",       1,  False, False),
        ("B: k=4 analytical",       4,  False, False),
        ("C: k=16 analytical",      16, False, False),
        ("D: k=1 fit",              1,  True,  False),
        ("E: k=4 fit",              4,  True,  False),
        ("F: k=16 fit",             16, True,  False),
        ("G: k=4 GMM-sample",       4,  True,  True),
        ("H: k=16 GMM-sample",      16, True,  True),
    ]
    alphas = torch.linspace(0, 1.0, args.n_alpha).tolist()

    branches = [
        ("train_transform", False),
        ("test_transform", True),
    ]

    zs_classifier = None
    zs_all_logits = None
    zs_id_logits = None
    zs_ood_logits = None
    zs_per_ds = None
    zs_avg = None
    zs_ood_per_ds = None
    zs_ood_avg = None
    all_branch_results = {}
    all_ood_results = {}

    for branch_label, use_test_transform in branches:
        logging.info(f"\n{'#'*60}")
        logging.info(f"Branch: {branch_label}")
        logging.info(f"{'#'*60}")

        (train_feats_norm, train_labels_gpu, train_raw_feats,
         id_test_feats_norm, id_test_labels, num_id,
         ood_test_feats_norm, ood_test_labels) = _extract_features_for_branch(
            model, args, all_id_names, all_ood_names, ds_slices, od_slices, use_test_transform
        )

        if zs_classifier is None:
            all_class_names = all_id_names + all_ood_names
            zs_classifier = get_zeroshot_classifier(model, processor, all_class_names, args.device)
            num_all = len(all_class_names)

            with torch.no_grad():
                zs_all_logits = id_test_feats_norm @ zs_classifier

            zs_per_ds = []
            for start, end in ds_slices:
                zs_l = zs_all_logits[start:end]
                acc = (zs_l - zs_l.max(dim=-1, keepdim=True).values).argmax(dim=1).eq(
                    id_test_labels[start:end]).float().mean().item() * 100
                zs_per_ds.append(acc)
            zs_avg = sum(zs_per_ds) / len(zs_per_ds)

            zs_id_logits = zs_all_logits

            if has_ood:
                with torch.no_grad():
                    zs_ood_logits_t = ood_test_feats_norm @ zs_classifier
                zs_ood_per_ds = []
                for start, end in od_slices:
                    zs_l = zs_ood_logits_t[start:end]
                    acc = (zs_l - zs_l.max(dim=-1, keepdim=True).values).argmax(dim=1).eq(
                        ood_test_labels[start:end]).float().mean().item() * 100
                    zs_ood_per_ds.append(acc)
                zs_ood_avg = sum(zs_ood_per_ds) / len(zs_ood_per_ds)
                zs_all_logits = zs_ood_logits_t
                logging.info(f"Zero-shot ID avg: {zs_avg:.1f}%, OOD avg: {zs_ood_avg:.1f}%")
            else:
                zs_ood_avg = 0
                zs_ood_per_ds = []
                logging.info(f"Zero-shot per-dataset avg: {zs_avg:.1f}%")

            id_test_shared = id_test_feats_norm
            id_label_shared = id_test_labels
            ood_test_shared = ood_test_feats_norm
            ood_label_shared = ood_test_labels
        else:
            id_test_feats_norm = id_test_shared
            id_test_labels = id_label_shared
            ood_test_feats_norm = ood_test_shared
            ood_test_labels = ood_label_shared

        # ====== 构建 + 评估所有 8 配置（ID 部分） ======
        branch_results = {}
        branch_ood = {}

        for cfg_name, k, do_fit, use_gmm in configs:
            logging.info(f"[{branch_label}] {cfg_name}: k={k}, fit={do_fit}, gmm={use_gmm}")
            if use_gmm:
                classifier = build_lada_from_gmm_samples(
                    train_raw_feats, train_feats_norm, train_labels_gpu, args, k, do_fit
                )
            else:
                classifier = build_lada_classifier(train_feats_norm, train_labels_gpu, args, k, do_fit)

            r_id, _ = _eval_and_sweep(args, classifier, zs_id_logits, zs_all_logits,
                                        id_test_feats_norm, id_test_labels, ds_slices,
                                        num_id, num_all, alphas, is_id=True)
            branch_results[cfg_name] = r_id

            if has_ood:
                r_ood, _ = _eval_and_sweep(args, classifier, zs_id_logits, zs_all_logits,
                                             ood_test_feats_norm, ood_test_labels, od_slices,
                                             num_id, num_all, alphas, is_id=False)
                branch_ood[cfg_name] = r_ood

            del classifier
            torch.cuda.empty_cache()

        all_branch_results[branch_label] = branch_results
        if has_ood:
            all_ood_results[branch_label] = branch_ood

        _print_all_tables(branch_label, configs, branch_results, zs_avg, zs_per_ds, zs_all_logits, args)
        if has_ood:
            _print_ood_table(branch_label, configs, branch_results, branch_ood, zs_ood_avg, zs_ood_per_ds, args)

        del train_feats_norm, train_labels_gpu
        torch.cuda.empty_cache()

    # ========== 最终对比总结 ==========
    print("\n\n" + "=" * 120)
    print("=== 双分支对比总结 ===")
    print("=" * 120)
    print(f"\n{'Config':<25s} | {'LADA(train_t)':>13s} | {'LADA(test_t)':>13s} | {'Diff':>8s}")
    print("-" * 75)
    train_r = all_branch_results.get("train_transform", {})
    test_r = all_branch_results.get("test_transform", {})
    for cfg_name, _, _, _ in configs:
        tv = train_r.get(cfg_name, {}).get("lada_avg", 0)
        sv = test_r.get(cfg_name, {}).get("lada_avg", 0)
        print(f"{cfg_name:<25s} | {tv:>11.1f}%  | {sv:>12.1f}%  | {sv-tv:>+7.1f}%")

    if has_ood:
        print("\n\n" + "=" * 120)
        print("=== OOD 干扰总结（test_transform 分支）===")
        print("=" * 120)
        tood = all_ood_results.get("test_transform", {})
        ttest = all_branch_results.get("test_transform", {})
        print(f"\n{'Config':<25s} | {'ID LADA':>8s} | {'OOD ZS':>8s} | {'OOD Ens(best)':>13s} | {'Δ OOD':>8s}")
        print("-" * 80)
        for cfg_name, _, _, _ in configs:
            id_la = ttest.get(cfg_name, {}).get("lada_avg", 0)
            oo = tood.get(cfg_name, {})
            oo_best_a, oo_best_acc = oo.get("weighted", {}).get("best", (0, 0))
            delta = oo_best_acc - zs_ood_avg
            print(f"{cfg_name:<25s} | {id_la:>6.1f}% | {zs_ood_avg:>6.1f}% | {oo_best_acc:>11.1f}% | {delta:>+7.1f}%")


if __name__ == "__main__":
    arguments = parse_args()
    main(arguments)
