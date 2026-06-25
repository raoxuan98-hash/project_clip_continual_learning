"""
共享工具函数：被 main_joint.py 和 main_incremental.py 共同使用

包含：
- fix_random_seed: 设置随机种子
- get_zeroshot_classifier: 构建零样本分类器权重
- evaluate_dataset: 在单个数据集上评估 ZS / RGDA / Ensemble 准确率
- get_full_stats: 从准确率矩阵计算 Transfer/Average/Last 指标
- print_paper_metrics: 打印论文风格的指标报告
"""

import os
import random
import numpy as np
import torch
import torch.nn.functional as F


def fix_random_seed(seed=42):
    """设置随机种子以确保结果可复现"""
    print(f"Setting fixed seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_zeroshot_classifier(model, processor, class_names, device):
    """构建零样本分类器（批量编码，支持 text LoRA）"""
    templates = [lambda x: f"a photo of a {x}."]
    all_texts = []
    class_text_counts = []
    for classname in class_names:
        classname = classname.replace('_', ' ')
        texts = [template(classname) for template in templates]
        all_texts.extend(texts)
        class_text_counts.append(len(texts))

    with torch.no_grad():
        text_inputs = processor(text=all_texts, return_tensors="pt", padding=True, truncation=True)
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        text_outputs = model.text_model(**text_inputs)
        if hasattr(text_outputs, 'pooler_output') and text_outputs.pooler_output is not None:
            pooled = text_outputs.pooler_output
        elif hasattr(text_outputs, 'last_hidden_state'):
            pooled = text_outputs.last_hidden_state[:, -1, :]
        else:
            pooled = text_outputs[1] if isinstance(text_outputs, tuple) else text_outputs
        all_embeddings = model.text_projection(pooled)
        all_embeddings = all_embeddings / all_embeddings.norm(dim=-1, keepdim=True)

    zeroshot_weights = []
    start = 0
    for count in class_text_counts:
        class_embedding = all_embeddings[start:start + count].mean(dim=0)
        class_embedding = class_embedding / class_embedding.norm()
        zeroshot_weights.append(class_embedding)
        start += count

    return torch.stack(zeroshot_weights, dim=1).to(device)


def _normalize_for_ensemble(logits, mode):
    if mode == "maxshift":
        return logits - logits.max(dim=-1, keepdim=True).values
    if mode == "zscore":
        centered = logits - logits.mean(dim=-1, keepdim=True)
        return centered / logits.std(dim=-1, keepdim=True).clamp_min(1e-6)
    if mode == "prob":
        return F.softmax(logits, dim=-1)
    if mode == "raw":
        return logits
    raise ValueError(
        f"Unsupported ensemble_normalize={mode!r}. "
        "Expected one of: zscore, maxshift, prob, raw."
    )


def combine_ensemble_logits(zs_logits, id_logits, current_num_classes, alpha, mode="zscore"):
    """Fuse global zero-shot logits with an ID-only classifier."""
    zs_scores = _normalize_for_ensemble(zs_logits, mode)
    id_scores = _normalize_for_ensemble(id_logits, mode)
    ensemble_logits = zs_scores * (1.0 - alpha)
    ensemble_logits[:, :current_num_classes] += alpha * id_scores
    return ensemble_logits


def evaluate_dataset(args, d_name, model, zeroshot_classifier, lr_rgda_classifier,
                     current_num_classes, eval_label_offset,
                     alpha_sensitivity=False, n_alpha_samples=21,
                     lada_classifier=None, lada_alpha=1.0):
    """
    在单个数据集上评估 ZS / RGDA / Ensemble 准确率

    Args:
        args: 全局参数
        d_name: 数据集名称
        model: CLIP 模型
        zeroshot_classifier: 零样本分类器权重 [D, num_classes]
        lr_rgda_classifier: LR-RGDA 分类器实例
        current_num_classes: 当前 ID 类别总数（用于 ensemble 时只在 ID 区域叠加 RGDA）
        eval_label_offset: 评估时的标签偏移量
        alpha_sensitivity: 是否对 alpha 做敏感性分析（枚举多个 alpha 值）
        n_alpha_samples: 敏感性分析时 alpha 的采样点数（默认 21，即 0, 0.05, ..., 1.0）
        lada_classifier: LADA 分类器实例（可选，传入后额外计算 LADA / LADA+ZS 指标）
        lada_alpha: LADA+ZS 集成中 LADA 的权重（默认 1.0）

    Returns:
        (zs_acc, rgda_acc, ens_acc, lada_acc, lada_zs_acc, num_classes_in_dataset, sensitivity_list)
        当 lada_classifier 为 None 时，lada_acc 和 lada_zs_acc 为 None
    """
    from utils_data import get_xtail_trainloader, get_transforms
    from src.utils.feature_extractor import extract_features

    _, test_transform = get_transforms(d_name)
    _, _, te_loader, c_names = get_xtail_trainloader(
        root=args.root, dataset_name=d_name,
        transform_train=None, transform_test=test_transform,
        num_shots=args.num_shots, batch_size=args.batch_size)

    features, labels = extract_features(model, te_loader, args.device)

    # 确保特征严格进行 L2 归一化
    features = features / features.norm(dim=-1, keepdim=True)
    features = features.to(args.device)
    labels = (labels + eval_label_offset).to(args.device)

    with torch.no_grad():
        # Zero-shot 预测（不乘 logit_scale，与 debug_classifier_router.py 一致）
        ensemble_mode = getattr(args, "ensemble_normalize", "maxshift")
        zs_logits = features @ zeroshot_classifier
        zs_logits_norm = _normalize_for_ensemble(zs_logits, "maxshift")
        zs_preds = zs_logits.argmax(dim=1)
        zs_acc = zs_preds.eq(labels).float().mean().item() * 100

        # 2. 纯 LR-RGDA 预测
        rgda_logits = lr_rgda_classifier.forward(features)
        rgda_logits_norm = _normalize_for_ensemble(rgda_logits, "maxshift")
        rgda_preds = rgda_logits.argmax(dim=1)
        rgda_acc = rgda_preds.eq(labels).float().mean().item() * 100

        # 3. Ensemble 预测（固定 α 或自适应）
        use_adaptive = getattr(args, 'adaptive_ensemble', False)
        if use_adaptive:
            zs_scores = _normalize_for_ensemble(zs_logits, ensemble_mode)
            rgda_scores = _normalize_for_ensemble(rgda_logits, ensemble_mode)
            zs_probs = F.softmax(zs_scores, dim=-1)
            rgda_probs = F.softmax(rgda_scores, dim=-1)
            zs_conf = zs_probs.max(dim=-1).values        # [B]
            rgda_conf = rgda_probs.max(dim=-1).values    # [B]
            alpha_sample = torch.sigmoid(rgda_conf - zs_conf).unsqueeze(-1)  # [B, 1]
            ensemble_logits = (1 - alpha_sample) * zs_scores
            ensemble_logits[:, :current_num_classes] += alpha_sample * rgda_scores
        else:
            ensemble_logits = combine_ensemble_logits(
                zs_logits, rgda_logits, current_num_classes, args.alpha, ensemble_mode
            )
        ens_preds = ensemble_logits.argmax(dim=1)
        ens_acc = ens_preds.eq(labels).float().mean().item() * 100

        # LADA 和 LADA+ZS 预测（可选）
        lada_acc, lada_zs_acc = None, None
        if lada_classifier is not None:
            lada_logits = lada_classifier(features)
            lada_logits_norm = lada_logits - lada_logits.max(dim=-1, keepdim=True).values
            lada_preds = lada_logits_norm.argmax(dim=1)
            lada_acc = lada_preds.eq(labels).float().mean().item() * 100

            lada_zs_logits = (1 - lada_alpha) * zs_logits_norm + lada_alpha * lada_logits_norm
            lada_zs_preds = lada_zs_logits.argmax(dim=1)
            lada_zs_acc = lada_zs_preds.eq(labels).float().mean().item() * 100

        # 4. Alpha 敏感性分析（可选，只在固定 α 模式下有意义）
        sensitivity_list = None
        if alpha_sensitivity and not use_adaptive:
            lr_sensitivity = []
            for alpha in torch.linspace(0, 1.0, n_alpha_samples):
                ens_logits = combine_ensemble_logits(
                    zs_logits, rgda_logits, current_num_classes, alpha, ensemble_mode
                )
                ens_preds = ens_logits.argmax(dim=1)
                ens_acc_alpha = ens_preds.eq(labels).float().mean().item() * 100
                lr_sensitivity.append((round(alpha.item(), 3), round(ens_acc_alpha, 2)))
            if lada_classifier is not None:
                lada_sensitivity = []
                for alpha in torch.linspace(0, 1.0, n_alpha_samples):
                    lada_zs_logits = (1 - alpha) * zs_logits_norm + alpha * lada_logits_norm
                    lada_zs_preds = lada_zs_logits.argmax(dim=1)
                    lada_acc_alpha = lada_zs_preds.eq(labels).float().mean().item() * 100
                    lada_sensitivity.append((round(alpha.item(), 3), round(lada_acc_alpha, 2)))
                sensitivity_list = {
                    "lr": lr_sensitivity,
                    "lada": lada_sensitivity,
                }
            else:
                sensitivity_list = lr_sensitivity

    return zs_acc, rgda_acc, ens_acc, lada_acc, lada_zs_acc, len(c_names), sensitivity_list


def batch_evaluate_datasets(
    dataset_names, model, processor, zeroshot_classifier, lr_rgda_classifier,
    num_id_classes, args=None, root=None, num_shots=16, batch_size=32, device='cuda',
    alpha=0.5, alpha_sensitivity=False, n_alpha_samples=21,
    lada_classifier=None, lada_alpha=1.0
):
    """
    批处理评估：先收集所有数据集的 feature 和 label 拼到一起，再统一评估。
    参考 debug_classifier_router.py 的评估逻辑。

    Args:
        dataset_names: 数据集名称列表
        model: CLIP 模型
        processor: CLIP processor
        zeroshot_classifier: 零样本分类器权重 [D, num_classes]
        lr_rgda_classifier: LR-RGDA 分类器实例
        num_id_classes: ID 类别总数
        args: 可选，全局参数对象（与下面独立参数二选一）
        root: 数据根目录
        num_shots: few-shot 数
        batch_size: batch size
        device: 设备
        alpha: 集成权重
        alpha_sensitivity: 是否做 α 敏感性分析
        n_alpha_samples: α 采样点数
        lada_classifier: 可选 LADA 分类器，用于同步扫描 LADA+ZS 系数
        lada_alpha: LADA+ZS 固定集成权重

    Returns:
        {
            "per_dataset": {d_name: {"zs": zs, "rgda": rgda, "ens": ens}, ...},
            "overall": {"zs": zs, "rgda": rgda, "ens": ens},
            "sensitivity": [(alpha, overall_acc), ...] | None
        }
    """
    # 解析参数（兼容 args 对象和独立传参）
    use_adaptive = False
    if args is not None:
        root = args.root
        num_shots = args.num_shots
        batch_size = args.batch_size
        device = args.device
        alpha = args.alpha
        lada_alpha = getattr(args, 'lada_alpha', lada_alpha)
        use_adaptive = getattr(args, 'adaptive_ensemble', False)

    from utils_data import get_xtail_trainloader, get_transforms
    from src.utils.feature_extractor import extract_features

    # 收集所有数据集的 features 和 labels
    all_features, all_labels = [], []
    per_dataset = {}
    offset = 0

    for d_name in dataset_names:
        _, test_transform = get_transforms(d_name)
        _, _, te_loader, c_names = get_xtail_trainloader(
            root=root, dataset_name=d_name,
            transform_train=None, transform_test=test_transform,
            num_shots=num_shots, batch_size=batch_size
        )
        features, labels = extract_features(model, te_loader, device)
        features = features / features.norm(dim=-1, keepdim=True)

        all_features.append(features.cpu())
        all_labels.append((labels + offset).cpu())
        offset += len(c_names)

    all_features = torch.cat(all_features)
    all_labels = torch.cat(all_labels)

    with torch.no_grad():
        total = all_labels.numel()
        eval_chunk_size = int(getattr(args, "alpha_sweep_batch_size", 512) or 512)
        alpha_values = (
            torch.linspace(0, 1.0, n_alpha_samples, device=device)
            if alpha_sensitivity and not use_adaptive
            else None
        )
        zs_correct = 0
        rgda_correct = 0
        ens_correct = 0
        lada_overall = None
        lada_zs_overall = None
        lada_sensitivity = None
        lada_correct = 0
        lada_zs_correct = 0
        sensitivity_correct = (
            [0 for _ in range(n_alpha_samples)]
            if alpha_values is not None
            else None
        )
        lada_sensitivity_correct = (
            [0 for _ in range(n_alpha_samples)]
            if alpha_values is not None and lada_classifier is not None
            else None
        )

        for start in range(0, total, eval_chunk_size):
            end = min(start + eval_chunk_size, total)
            features_chunk = all_features[start:end].to(device)
            labels_chunk = all_labels[start:end].to(device)

            # Zero-shot logits（不乘 logit_scale，与 debug_classifier_router.py 一致）
            ensemble_mode = getattr(args, "ensemble_normalize", "maxshift")
            zs_logits = features_chunk @ zeroshot_classifier
            zs_logits_norm = _normalize_for_ensemble(zs_logits, "maxshift")
            zs_preds = zs_logits.argmax(dim=1)
            zs_correct += int(zs_preds.eq(labels_chunk).sum().item())

            # LR-RGDA logits. Keep this chunked: full-test LR-RGDA logits can exceed GPU memory.
            rgda_logits = lr_rgda_classifier.forward(features_chunk)
            rgda_logits_norm = _normalize_for_ensemble(rgda_logits, "maxshift")
            rgda_preds = rgda_logits.argmax(dim=1)
            rgda_correct += int(rgda_preds.eq(labels_chunk).sum().item())

            if use_adaptive:
                zs_scores = _normalize_for_ensemble(zs_logits, ensemble_mode)
                rgda_scores = _normalize_for_ensemble(rgda_logits, ensemble_mode)
                zs_probs = F.softmax(zs_scores, dim=-1)
                rgda_probs = F.softmax(rgda_scores, dim=-1)
                zs_conf = zs_probs.max(dim=-1).values
                rgda_conf = rgda_probs.max(dim=-1).values
                alpha_sample = torch.sigmoid(rgda_conf - zs_conf).unsqueeze(-1)
                ensemble_logits = (1 - alpha_sample) * zs_scores
                ensemble_logits[:, :num_id_classes] += alpha_sample * rgda_scores
            else:
                ensemble_logits = combine_ensemble_logits(
                    zs_logits, rgda_logits, num_id_classes, alpha, ensemble_mode
                )
            ens_preds = ensemble_logits.argmax(dim=1)
            ens_correct += int(ens_preds.eq(labels_chunk).sum().item())

            lada_logits_norm = None
            if lada_classifier is not None:
                lada_logits = lada_classifier(features_chunk)
                lada_logits_norm = lada_logits - lada_logits.max(dim=-1, keepdim=True).values
                lada_preds = lada_logits_norm.argmax(dim=1)
                lada_correct += int(lada_preds.eq(labels_chunk).sum().item())

                lada_zs_logits = (1 - lada_alpha) * zs_logits_norm + lada_alpha * lada_logits_norm
                lada_zs_preds = lada_zs_logits.argmax(dim=1)
                lada_zs_correct += int(lada_zs_preds.eq(labels_chunk).sum().item())

            if alpha_values is not None:
                for idx, a in enumerate(alpha_values):
                    ens_logits = combine_ensemble_logits(
                        zs_logits, rgda_logits, num_id_classes, a, ensemble_mode
                    )
                    ens_preds = ens_logits.argmax(dim=1)
                    sensitivity_correct[idx] += int(ens_preds.eq(labels_chunk).sum().item())

                    if lada_logits_norm is not None:
                        lada_zs_logits = (1 - a) * zs_logits_norm + a * lada_logits_norm
                        lada_zs_preds = lada_zs_logits.argmax(dim=1)
                        lada_sensitivity_correct[idx] += int(
                            lada_zs_preds.eq(labels_chunk).sum().item()
                        )

        zs_overall = zs_correct / total * 100
        rgda_overall = rgda_correct / total * 100
        ens_overall = ens_correct / total * 100

        if lada_classifier is not None:
            lada_overall = lada_correct / total * 100
            lada_zs_overall = lada_zs_correct / total * 100

        sensitivity = None
        if alpha_values is not None:
            sensitivity = [
                (round(float(a.item()), 3), round(count / total * 100, 2))
                for a, count in zip(alpha_values, sensitivity_correct)
            ]
            if lada_sensitivity_correct is not None:
                lada_sensitivity = []
                for a, count in zip(alpha_values, lada_sensitivity_correct):
                    lada_sensitivity.append(
                        (round(float(a.item()), 3), round(count / total * 100, 2))
                    )

    return {
        "overall": {
            "zs": zs_overall,
            "rgda": rgda_overall,
            "ens": ens_overall,
            "lada": lada_overall,
            "lada_zs": lada_zs_overall,
        },
        "sensitivity": sensitivity,
        "lada_sensitivity": lada_sensitivity,
    }


def get_full_stats(matrix):
    """
    从准确率矩阵计算 Transfer/Average/Last 指标（LADA 论文定义）

    Args:
        matrix: 二维列表
                - 单行 (联合微调): [[acc1, acc2, ..., accN]]
                - 多行 (增量微调): [[...], [...], ..., [...]] (T x N)

    Returns:
        dict 包含 raw_matrix, transfer, transfer_total_avg,
              average_per_task, average_total_avg, last, last_total_avg
    """
    num_rows = len(matrix)
    if num_rows == 1:
        data_row = matrix[0]
        return {
            "raw_matrix": matrix,
            "transfer": data_row,
            "transfer_total_avg": sum(data_row) / len(data_row),
            "average_per_task": data_row,
            "average_total_avg": sum(data_row) / len(data_row),
            "last": data_row,
            "last_total_avg": sum(data_row) / len(data_row)
        }
    else:
        K = num_rows
        # Transfer_k = mean accuracy on task k before task k is learned.
        # In a full LADA matrix, row j is "after training task j" and column k
        # is "evaluated on task k", so Transfer uses the upper triangle.
        trans = []
        for k in range(K):
            if k == 0:
                trans.append(0.0)  # placeholder for display
            else:
                trans.append(sum(matrix[j][k] for j in range(k)) / k)
        # Transfer = mean of Transfer_k for k=2..K (K-1 values)
        transfer_values = [trans[k] for k in range(1, K)]
        transfer_total_avg = sum(transfer_values) / len(transfer_values)

        # Average_k = mean of column k across all training steps.
        avgs = [sum(matrix[j][k] for j in range(K)) / K for k in range(K)]
        average_total_avg = sum(avgs) / K

        # Last_k = accuracy on task k after all K tasks trained
        lasts = [matrix[K-1][k] for k in range(K)]
        last_total_avg = sum(lasts) / K

        return {
            "raw_matrix": matrix,
            "transfer": trans,
            "transfer_total_avg": transfer_total_avg,
            "average_per_task": avgs,
            "average_total_avg": average_total_avg,
            "last": lasts,
            "last_total_avg": last_total_avg
        }


def print_paper_metrics(matrix, name, headers):
    """打印论文风格的指标报告"""
    stats = get_full_stats(matrix)
    num_cols = len(matrix[0])

    print(f"\n" + "-" * 110)
    print(f"[{name} 指标报告]")
    # 自动截取对应的表头
    header_str = " | ".join([f"{h[:8]:<8}" for h in headers[:num_cols]])
    print(f"指标类型   | " + header_str + " | [平均总分]")
    print("-" * 110)

    print(f"Transfer  | " + " | ".join([f"{x:8.1f}" for x in stats['transfer']])
          + f" | [{stats['transfer_total_avg']:.1f}]")
    print(f"Average   | " + " | ".join([f"{x:8.1f}" for x in stats['average_per_task']])
          + f" | [{stats['average_total_avg']:.1f}]")
    print(f"Last      | " + " | ".join([f"{x:8.1f}" for x in stats['last']])
          + f" | [{stats['last_total_avg']:.1f}]")
    print("-" * 110)
