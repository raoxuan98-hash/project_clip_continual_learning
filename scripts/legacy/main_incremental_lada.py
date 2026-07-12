"""
LADA 增量学习入口脚本

功能：
1. 按 dataset_sequence 逐任务增量训练
2. 每个任务：LoRA-NSP 训练 + LADA 构建 + DPT 更新
3. 评估：ZS / LADA / LADA+ZS / LR-RGDA / LR-RGDA+ZS（独立对比）
4. 保存结果 JSON

用法示例：
    python main_incremental_lada.py \
        --dataset_sequence aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397 \
        --num_shots 16 --iterations 800 \
        --lora_type lora_nsp --lada_k 16 --lada_beta 1.0 --lada_alpha 1.0 \
        --enable_dpt --prototype_k 4
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import argparse
import json
import logging
from datetime import datetime
from torch.utils.data import DataLoader, ConcatDataset

from src.lada.lada_trainer import LADATrainer
from src.classifiers.lr_rgda_classifier import LRRGDAClassifier
from src.classifiers.gaussian_statistics import build_stats_dict_from_features
from src.utils.reference_loader import load_reference_dataset
from src.utils.main_utils import (
    fix_random_seed,
    get_zeroshot_classifier,
    get_full_stats,
    print_paper_metrics,
)
from src.utils.feature_extractor import extract_features
from src.utils.data import get_xtail_trainloader, get_transforms


def parse_args():
    parser = argparse.ArgumentParser(description="LADA Incremental Learning for CLIP")

    parser.add_argument("--id_datasets", type=str, nargs='+',
                        default=["aircraft", "caltech101", "dtd", "eurosat", "flowers",
                                 "food101", "mnist", "oxford_pets", "stanford_cars", "sun397"])
    parser.add_argument("--root", type=str, default="/data1/open_datasets/X-TAIL")
    parser.add_argument("--num_shots", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_num_per_test_dataset", type=int, default=1000)

    parser.add_argument("--dataset_sequence", type=str, nargs='+',
                        default=["aircraft", "caltech101", "dtd", "eurosat", "flowers",
                                 "food101", "mnist", "oxford_pets", "stanford_cars", "sun397"])

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--iterations", type=int, default=800)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=3e-5)

    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_type", type=str, default="lora_nsp",
                        choices=["lora_vanilla", "lora_sgp", "lora_nsp"])
    parser.add_argument("--init_mode", type=str, default="lora_nsp",
                        choices=["lora_nsp", "lora_vanilla",
                                 "proj_sigma_tail", "proj_sigma_middle",
                                 "weight_svd_tail", "weight_svd_middle"])
    parser.add_argument("--nsp_eps", type=float, default=0.05)
    parser.add_argument("--nsp_weight", type=float, default=0.02)
    parser.add_argument("--weight_temp", type=float, default=1.0)
    parser.add_argument("--weight_kind", type=str, default="log1p")
    parser.add_argument("--weight_p", type=float, default=1.0)

    parser.add_argument("--reference_dataset", type=str, default="flickr8k")
    parser.add_argument("--reference_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=6)

    parser.add_argument("--fd_weight", type=float, default=0.0,
                        help="LADA 模式不使用 FD，保持为 0")
    parser.add_argument("--cd_weight", type=float, default=0.0,
                        help="LADA 模式不使用 CD，保持为 0")
    parser.add_argument("--aux_weight", type=float, default=0.0,
                        help="LADA 模式不使用辅助头，保持为 0")
    parser.add_argument("--sce_a", type=float, default=0.5)
    parser.add_argument("--sce_b", type=float, default=0.5)

    parser.add_argument("--alpha", type=float, default=0.5,
                        help="LR-RGDA 集成权重（独立对比用）")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--adaptive_ensemble", action='store_true', default=False)

    parser.add_argument("--rgda_rank", type=int, default=32)
    parser.add_argument("--rgda_alpha1", type=float, default=0.2)
    parser.add_argument("--rgda_alpha2", type=float, default=2.0)
    parser.add_argument("--rgda_alpha3", type=float, default=0.5)

    parser.add_argument("--tune_text_encoder", type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument("--text_lora_rank", type=int, default=4)
    parser.add_argument("--tune_vision_encoder", type=lambda x: x.lower() == 'true', default=True)

    parser.add_argument("--lada_k", type=int, default=16,
                        help="每类 k-means 聚类中心数")
    parser.add_argument("--lada_beta", type=float, default=1.0,
                        help="指数亲和变换锐度参数")
    parser.add_argument("--lada_alpha", type=float, default=1.0,
                        help="LADA logits 在总 logits 中的权重")

    parser.add_argument("--enable_dpt", action='store_true', default=False,
                        help="启用 Distribution-Preserved Training")
    parser.add_argument("--prototype_k", type=int, default=4,
                        help="DPT 每类 GMM 分量数")
    parser.add_argument("--image_prototypes_weight_coef", type=float, default=64.0,
                        help="DPT 原型损失权重系数")
    parser.add_argument("--lada_official_mode", action='store_true', default=False,
                        help="对齐官方 LADA：冻结视觉编码器、确定性原型、训练时直接相加 logits")
    parser.add_argument("--lada_replay_mode", type=str, default=None,
                        choices=["none", "mean", "dpt"],
                        help="旧类回放消融：none / GMM均值 / 官方DPT噪声增强")
    parser.add_argument("--dpt_feature_normalize", type=lambda x: x.lower() == 'true',
                        default=None,
                        help="GMM 拟合前是否 L2 归一化；官方 LADA 使用 False")

    args = parser.parse_args()
    args.dataset_sequence = [[d] for d in args.dataset_sequence]

    if args.lada_replay_mode is None:
        args.lada_replay_mode = "dpt" if args.enable_dpt else "none"
    args.enable_dpt = args.lada_replay_mode != "none"

    if args.lada_official_mode:
        args.tune_vision_encoder = False
        args.tune_text_encoder = True
        args.lora_type = "lora_vanilla"
        args.init_mode = "lora_vanilla"
        args.dpt_feature_normalize = False
    elif args.dpt_feature_normalize is None:
        args.dpt_feature_normalize = False

    if args.device is None:
        args.device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    return args


def evaluate_dataset_lada(args, d_name, model, zeroshot_classifier, lada_classifier,
                          lr_rgda_classifier, current_num_classes, eval_label_offset,
                          logit_scale_val, trainer=None):
    """
    在单个数据集上评估 ZS / LADA / LADA+ZS / LR-RGDA / LR-RGDA+ZS

    所有 logits 覆盖全部已见类（C_total），直接 argmax 比较全局标签。
    """
    _, test_transform = get_transforms(d_name)
    _, _, te_loader, c_names = get_xtail_trainloader(
        root=args.root, dataset_name=d_name,
        transform_train=None, transform_test=test_transform,
        num_shots=args.num_shots, batch_size=args.batch_size)

    if trainer is not None:
        features, labels = trainer._extract_features_manual(te_loader)
    else:
        features, labels = extract_features(model, te_loader, args.device)
        features = features / features.norm(dim=-1, keepdim=True)
    features = features.to(args.device)
    labels = (labels + eval_label_offset).to(args.device)

    with torch.no_grad():
        zs_logits = features @ zeroshot_classifier
        zs_preds = zs_logits.argmax(dim=1)
        zs_acc = zs_preds.eq(labels).float().mean().item() * 100

        lada_logits = lada_classifier(features)
        lada_preds = lada_logits.argmax(dim=1)
        lada_acc = lada_preds.eq(labels).float().mean().item() * 100

        lada_classes = lada_logits.shape[1]
        mask = (zs_preds < lada_classes).float().unsqueeze(1)
        lada_ens_logits = zs_logits + mask * args.lada_alpha * lada_logits
        lada_ens_preds = lada_ens_logits.argmax(dim=1)
        lada_ens_acc = lada_ens_preds.eq(labels).float().mean().item() * 100

        rgda_logits = lr_rgda_classifier.forward(features)
        rgda_preds = rgda_logits.argmax(dim=1)
        rgda_acc = rgda_preds.eq(labels).float().mean().item() * 100

        ens_logits = zs_logits * (1 - args.alpha)
        ens_logits[:, :current_num_classes] += args.alpha * rgda_logits
        rgda_ens_preds = ens_logits.argmax(dim=1)
        rgda_ens_acc = rgda_ens_preds.eq(labels).float().mean().item() * 100

    return zs_acc, lada_acc, lada_ens_acc, rgda_acc, rgda_ens_acc, len(c_names)


def main(args):
    if args.seed is not None:
        fix_random_seed(args.seed)

    logging.info("\n=== Initializing LADA Incremental Learning ===")
    trainer = LADATrainer(args)
    model = trainer.model
    processor = trainer.processor

    if args.lada_official_mode:
        reference_loader = None
    else:
        reference_loader = load_reference_dataset(args, trainer.model_pretrain,
                                                   processor, args.device)

    global_class_names = []
    for task_datasets in args.dataset_sequence:
        for d_name in task_datasets:
            _, _, _, c_names = get_xtail_trainloader(
                root=args.root, dataset_name=d_name,
                transform_train=None, transform_test=None,
                num_shots=args.num_shots, batch_size=args.batch_size
            )
            global_class_names.extend(c_names)

    history_class_names = []
    global_stats_dict = {}

    acc_matrix_zs = []
    acc_matrix_lada = []
    acc_matrix_lada_ens = []
    acc_matrix_rgda = []
    acc_matrix_rgda_ens = []

    for i, task_datasets in enumerate(args.dataset_sequence):
        print(f"\n" + "=" * 50)
        print(f"=== Task {i+1}: {task_datasets} ===")
        print("=" * 50)

        train_loaders = []
        update_loaders = []
        task_class_names = []
        for d_name in task_datasets:
            train_transform, test_transform = get_transforms(d_name)
            tr_loader, update_loader, _, c_names = get_xtail_trainloader(
                root=args.root, dataset_name=d_name,
                transform_train=train_transform, transform_test=test_transform,
                num_shots=args.num_shots, batch_size=args.batch_size
            )
            train_loaders.append(tr_loader)
            update_loaders.append(update_loader)
            task_class_names.extend(c_names)

        merged_dataset = ConcatDataset([loader.dataset for loader in train_loaders])
        update_dataset = ConcatDataset([loader.dataset for loader in update_loaders])
        cov_loader = DataLoader(merged_dataset, batch_size=args.batch_size, shuffle=False)
        merged_loader = DataLoader(merged_dataset, batch_size=args.batch_size, shuffle=True)
        prototype_loader = DataLoader(
            update_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True)

        label_offset = sum(len(c_names) for c_names in history_class_names)
        current_all_class_names = []
        for hcn in history_class_names:
            current_all_class_names.extend(hcn)
        current_all_class_names.extend(task_class_names)

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

        model = trainer.train(merged_loader, task_class_names, reference_loader,
                              aux_weight=args.aux_weight,
                              label_offset=label_offset,
                              all_class_names=current_all_class_names,
                              prototype_loader=(
                                  prototype_loader if args.lada_official_mode
                                  else merged_loader))

        if args.init_mode == "lora_nsp":
            print("\n=== Applying Null-Space Projection (NSP) ===")
            if trainer.has_vision_lora:
                covariances = trainer.extract_layer_covariances(cov_loader)
            if trainer.has_text_lora:
                text_covariances = trainer.extract_text_covariances(task_class_names)
            trainer.finalize_task_for_incremental()
            if trainer.has_vision_lora:
                trainer.update_covariance_history(covariances)
            if trainer.has_text_lora:
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

        print("\n=== Finalizing LADA Task ===")
        trainer.finalize_lada_task(
            prototype_loader if args.lada_official_mode else cov_loader,
            task_class_names, label_offset)

        task_features = []
        task_labels = []

        for d_name in task_datasets:
            train_transform, test_transform = get_transforms(d_name)
            tr_loader, update_loader, _, c_names = get_xtail_trainloader(
                root=args.root, dataset_name=d_name,
                transform_train=train_transform, transform_test=test_transform,
                num_shots=args.num_shots, batch_size=args.batch_size
            )
            feature_loader = update_loader if args.lada_official_mode else tr_loader
            features, labels = trainer._extract_features_manual(feature_loader)
            task_features.append(features)
            task_labels.append(labels + label_offset)

        task_features = torch.cat(task_features)
        task_labels = torch.cat(task_labels)

        task_stats_dict = build_stats_dict_from_features(task_features, task_labels)
        global_stats_dict.update(task_stats_dict)
        history_class_names.append(task_class_names)

        offset = 0
        per_dataset_covs = []
        for hcn in history_class_names:
            n_classes = len(hcn)
            ds_cov = sum(global_stats_dict[cid].cov for cid in range(offset, offset + n_classes)) / n_classes
            per_dataset_covs.append(ds_cov)
            offset += n_classes

        dataset_balanced_global_cov = sum(per_dataset_covs) / len(per_dataset_covs)

        lr_rgda_classifier = LRRGDAClassifier(
            stats_dict=global_stats_dict,
            device=args.device,
            rank=args.rgda_rank,
            qda_reg_alpha1=args.rgda_alpha1,
            qda_reg_alpha2=args.rgda_alpha2,
            qda_reg_alpha3=args.rgda_alpha3,
            temperature=1.0,
            global_cov=dataset_balanced_global_cov,
        )

        flat_class_names = [name for sublist in history_class_names for name in sublist]
        current_num_classes = len(flat_class_names)
        zeroshot_classifier = get_zeroshot_classifier(model, processor,
                                                       flat_class_names, args.device)

        logit_scale_val = model.logit_scale.detach().exp().item()

        print("\n=== Evaluating Task ===")
        step_accs_zs, step_accs_lada, step_accs_lada_ens = [], [], []
        step_accs_rgda, step_accs_rgda_ens = [], []

        eval_label_offset = 0
        for j in range(i + 1):
            eval_datasets = args.dataset_sequence[j]
            d_name = eval_datasets[0]

            zs_acc, lada_acc, lada_ens_acc, rgda_acc, rgda_ens_acc, c_len = \
                evaluate_dataset_lada(
                    args, d_name, model, zeroshot_classifier, trainer.lada_classifier,
                    lr_rgda_classifier, current_num_classes, eval_label_offset,
                    logit_scale_val, trainer=trainer
                )
            eval_label_offset += c_len

            print(f"[Tested on Task {j+1}: {d_name:<10s}] -> "
                  f"ZS: {zs_acc:5.1f}% | LADA: {lada_acc:5.1f}% | "
                  f"LADA+ZS: {lada_ens_acc:5.1f}% | "
                  f"RGDA: {rgda_acc:5.1f}% | RGDA+ZS: {rgda_ens_acc:5.1f}%")

            step_accs_zs.append(zs_acc)
            step_accs_lada.append(lada_acc)
            step_accs_lada_ens.append(lada_ens_acc)
            step_accs_rgda.append(rgda_acc)
            step_accs_rgda_ens.append(rgda_ens_acc)

        acc_matrix_zs.append(step_accs_zs)
        acc_matrix_lada.append(step_accs_lada)
        acc_matrix_lada_ens.append(step_accs_lada_ens)
        acc_matrix_rgda.append(step_accs_rgda)
        acc_matrix_rgda_ens.append(step_accs_rgda_ens)

    print("\n=== Training and Evaluation Completed ===")
    print("\n" + "=" * 80)
    print("Final Results")
    print("=" * 80)

    task_names = [d[0] for d in args.dataset_sequence]

    print_paper_metrics(acc_matrix_zs, "Zero-shot Baseline", task_names)
    print_paper_metrics(acc_matrix_lada, "LADA Only", task_names)
    print_paper_metrics(acc_matrix_lada_ens, f"LADA + ZS Ensemble (alpha={args.lada_alpha})", task_names)
    print_paper_metrics(acc_matrix_rgda, "LR-RGDA Only (independent)", task_names)
    print_paper_metrics(acc_matrix_rgda_ens, f"LR-RGDA + ZS Ensemble (alpha={args.alpha})", task_names)

    save_results = {
        "mode": "LADA Incremental Learning",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_order": task_names,
        "arguments": vars(args),
        "metrics": {
            "zero_shot": get_full_stats(acc_matrix_zs),
            "lada": get_full_stats(acc_matrix_lada),
            "lada_ensemble": get_full_stats(acc_matrix_lada_ens),
            "lr_rgda": get_full_stats(acc_matrix_rgda),
            "lr_rgda_ensemble": get_full_stats(acc_matrix_rgda_ens),
        }
    }

    save_dir = "experiments"
    os.makedirs(save_dir, exist_ok=True)
    file_name = f"lada_incremental_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_path = os.path.join(save_dir, file_name)

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(save_results, f, indent=4, ensure_ascii=False)

    logging.info(f"\nLADA 增量学习结果已保存至: {save_path}")


if __name__ == "__main__":
    command_line_args = parse_args()
    main(command_line_args)
