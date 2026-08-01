#!/usr/bin/env python3
"""Frozen CLIP 零样本分类行（Wave F 的 Frozen 行）。

对 X-TAIL 10 个数据集，用冻结预训练 CLIP + 每数据集局部零样本分类器
（与主实验相同的模板/类别名/测试 loader 构建路径）评估 ZS 准确率，
输出 JSON 供汇总表使用。

用法（服务器）：
    CLIP_LOCAL_FILES_ONLY=1 CLIP_MODEL_NAME=/mnt/raoxuan/models/clip-vit-base-patch16 \
    python scripts/eval_frozen_zeroshot.py \
        --root /data1/open_datasets/X-TAIL \
        --output experiments/paper_formal/WaveF_offline/frozen_zeroshot.json
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from src.models.clip import get_clip_model
from src.utils.main_utils import get_zeroshot_classifier
from src.utils.data import get_xtail_trainloader, get_xtail_classnames, get_transforms
from src.utils.feature_extractor import extract_features

DATASETS = ["aircraft", "caltech101", "dtd", "eurosat", "flowers",
            "food101", "mnist", "oxford_pets", "stanford_cars", "sun397"]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data1/open_datasets/X-TAIL")
    ap.add_argument("--eval-batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--output", type=Path,
                    default=Path("experiments/paper_formal/WaveF_offline/frozen_zeroshot.json"))
    ap.add_argument("--resize-mode", default="legacy_square",
                    choices=["legacy_square", "preserve_aspect"],
                    help="评估图像预处理协议；preserve_aspect=短边等比缩放+center crop（新协议）")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = get_clip_model(SimpleNamespace(), train_mode="frozen")
    model.to(device).eval()

    results = {}
    for d_name in DATASETS:
        class_names = get_xtail_classnames(args.root, d_name, num_shots=16)
        _, test_transform = get_transforms(d_name, test_resize_mode=args.resize_mode)
        _, _, te_loader, c_names = get_xtail_trainloader(
            root=args.root, dataset_name=d_name,
            transform_train=None, transform_test=test_transform,
            num_shots=16, batch_size=args.eval_batch_size,
            num_workers=args.num_workers)
        assert len(c_names) == len(class_names), (d_name, len(c_names), len(class_names))
        classifier = get_zeroshot_classifier(model, processor, class_names, device)
        features, labels = extract_features(model, te_loader, device, normalize=True)
        logits = features @ classifier
        pred = logits.argmax(dim=-1)
        acc = (pred == labels).float().mean().item() * 100.0
        results[d_name] = {"zs_acc": acc, "num_classes": len(class_names),
                           "num_test": int(labels.numel())}
        print(f"{d_name:<15s} ZS={acc:5.2f}%  (C={len(class_names)}, N={labels.numel()})", flush=True)

    results["_mean"] = {"zs_acc": sum(r["zs_acc"] for r in results.values()) / len(DATASETS)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"written -> {args.output}")


if __name__ == "__main__":
    main()
