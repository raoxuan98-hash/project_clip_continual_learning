#!/usr/bin/env python3
"""E6 融合系数 alpha 离线扫描（features 层面）。

训练 run（--save_step_artifacts --async_eval_dir ...）每任务落一个 artifact。
本脚本离线重建每个 step 的模型 / LR-RGDA / ZS 分类器，对全部 10 个数据集
（与 K×K 矩阵行语义一致：每行 = 训练完该任务后在所有列上的集成准确率）
分别提取 test 与 val（= train_loader4updating，同一 16-shot 训练集 +
test transform，本协议没有真 val）特征各一次，然后在 logits 层面对 alpha
网格重复调用 combine_ensemble_logits，避免重复提特征。

选择协议：global alpha* = 最终 step（所有已处理 step 中 step_index 最大者）
10 个数据集 val 集成准确率均值的 argmax；并列取较小 alpha（升序扫描、严格
大于才更新）。oracle 表（最终 step 每 alpha 的 test 均值）仅供分析，不参与
选择。

T/A/L 口径与 scripts/summarize_incremental_metrics.py 的 computed_metrics
逐条一致（见文件末尾 _computed_metrics 的注释）。

用法（服务器）：
    python scripts/eval_alpha_scan_artifacts.py \
        --async_eval_dir experiments/paper_formal/WaveF_offline/async_e6 \
        --gpu 1 --output experiments/paper_formal/WaveF_offline/alpha_scan.json
"""

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

import torch

# sys.path 自举：脚本目录（scripts/）供 import evaluate_incremental_artifacts，
# 项目根供 import src/* 与 main_incremental（eval_frozen_zeroshot.py 曾缺这步踩坑）。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_incremental_artifacts import (  # noqa: E402
    atomic_write_json,
    rebuild_model_and_classifier,
    torch_load,
)
from main_incremental import _build_text_classifier  # noqa: E402
from src.utils.data import get_transforms, get_xtail_trainloader  # noqa: E402
from src.utils.feature_extractor import extract_features  # noqa: E402
from src.utils.main_utils import combine_ensemble_logits  # noqa: E402

DEFAULT_ALPHAS = "0,0.01,0.02,0.05,0.10,0.20,0.50,1.0"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline alpha scan over async step artifacts (E6).")
    parser.add_argument("--async_eval_dir", type=Path,
                        default=Path("experiments/paper_formal/WaveF_offline/async_e6"))
    parser.add_argument("--alphas", type=str, default=DEFAULT_ALPHAS,
                        help="Comma/space separated alpha grid.")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--output", type=Path,
                        default=Path("experiments/paper_formal/WaveF_offline/alpha_scan.json"))
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--root", type=str, default="/data1/open_datasets/X-TAIL")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--max_steps", type=int, default=None,
                        help=argparse.SUPPRESS)  # 隐藏参数：只处理前 N 个 artifact，供 smoke
    return parser.parse_args()


def parse_alphas(raw):
    alphas = sorted({float(x) for x in str(raw).replace(",", " ").split() if x})
    if not alphas:
        raise ValueError("empty alpha grid")
    return alphas


def find_artifacts(async_eval_dir, max_steps=None):
    """返回按 step_index 排序的 [(step_index, path)]（不常驻加载 artifact 本体）。"""
    paths = sorted(Path(async_eval_dir).glob("**/artifact.pt"))
    if not paths:
        raise FileNotFoundError(f"no artifact.pt found under {async_eval_dir}")
    entries = []
    for path in paths:
        artifact = torch_load(path, map_location="cpu")
        entries.append((int(artifact["step_index"]), path))
        del artifact
    entries.sort(key=lambda item: item[0])
    if max_steps is not None:
        entries = entries[: int(max_steps)]
    return entries


@torch.inference_mode()
def _logits_for_split(model, lr_rgda, zeroshot_classifier, loader, device):
    """单 split 只提一次特征，返回 (zs_logits, rgda_logits, labels)（均在 device 上）。"""
    features, labels = extract_features(model, loader, device, normalize=True)
    features = features / features.norm(dim=-1, keepdim=True)
    features = features.to(device)
    zs_logits = features @ zeroshot_classifier
    rgda_logits = lr_rgda.forward(features)
    return zs_logits, rgda_logits, labels.to(device)


def evaluate_step(artifact, alphas, args):
    """对单个 artifact：10 数据集 × (test, val) × len(alphas) 的集成准确率。"""
    run_args, model, processor, frozen_model, lr_rgda = rebuild_model_and_classifier(
        artifact, args.device)

    task_names = artifact["task_names"]
    global_class_names = artifact["global_class_names"]
    dataset_label_offsets = artifact["dataset_label_offsets"]
    dataset_class_counts = artifact["dataset_class_counts"]
    current_num_classes = artifact["current_num_classes"]
    ensemble_mode = getattr(run_args, "ensemble_normalize", "maxshift")

    zeroshot_classifier = _build_text_classifier(
        run_args, model, processor, frozen_model, global_class_names,
        artifact.get("cached_seen_text_features"), args.device)

    num_shots = getattr(run_args, "num_shots", 16)
    model_name = getattr(run_args, "model_name", None)

    per_dataset = {}
    for task_idx, d_name in enumerate(task_names):
        _, test_transform = get_transforms(d_name, model_name=model_name)
        # 与 evaluate_dataset 相同的 loader 构建路径；val = train_loader4updating
        # （同一 16-shot 训练集 + test transform + 不打乱，本协议无真 val）。
        _, val_loader, test_loader, c_names = get_xtail_trainloader(
            root=args.root, dataset_name=d_name,
            transform_train=None, transform_test=test_transform,
            num_shots=num_shots, batch_size=args.eval_batch_size,
            num_workers=args.num_workers)
        if len(c_names) != dataset_class_counts[d_name]:
            raise ValueError(
                f"Class count mismatch for {d_name}: "
                f"{len(c_names)} vs {dataset_class_counts[d_name]}")
        label_offset = dataset_label_offsets[d_name]

        split_logits = {}
        for split, loader in (("val", val_loader), ("test", test_loader)):
            zs_logits, rgda_logits, labels = _logits_for_split(
                model, lr_rgda, zeroshot_classifier, loader, args.device)
            labels = labels + label_offset
            zs_acc = zs_logits.argmax(dim=1).eq(labels).float().mean().item() * 100
            rgda_acc = rgda_logits.argmax(dim=1).eq(labels).float().mean().item() * 100
            alpha_accs = {}
            for alpha in alphas:
                ens_logits = combine_ensemble_logits(
                    zs_logits, rgda_logits, current_num_classes, alpha, ensemble_mode)
                acc = ens_logits.argmax(dim=1).eq(labels).float().mean().item() * 100
                alpha_accs[f"{alpha:g}"] = round(acc, 4)
            split_logits[split] = {
                "zs_acc": round(zs_acc, 4),
                "rgda_acc": round(rgda_acc, 4),
                "ensemble_by_alpha": alpha_accs,
                "num_samples": int(labels.numel()),
            }
        per_dataset[d_name] = split_logits
        print(
            f"  [step {artifact['step_index'] + 1:02d} col {task_idx + 1:02d}] {d_name}: "
            + " ".join(f"a={a:g} test={split_logits['test']['ensemble_by_alpha'][f'{a:g}']:.2f}"
                       for a in alphas),
            flush=True,
        )

    del model, frozen_model, lr_rgda, zeroshot_classifier
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "step_index": artifact["step_index"],
        "task_name": artifact["task_name"],
        "current_num_classes": current_num_classes,
        "ensemble_mode": ensemble_mode,
        "per_dataset": per_dataset,
    }


def _alpha_mean(step_entry, split, alpha, task_names):
    """某 step 在 10 个数据集上某 split 集成准确率的均值（数据集等权）。"""
    return mean(
        step_entry["per_dataset"][d][split]["ensemble_by_alpha"][f"{alpha:g}"]
        for d in task_names
    )


# T/A/L 口径与 scripts/summarize_incremental_metrics.py::computed_metrics 逐条一致：
#   Transfer：对每列 task_idx∈[1,K)，取 matrix[step][task_idx]（step<task_idx，
#             即训练该任务之前的行，上三角）的均值（忽略 <0 的未评估哨兵），
#             再对列取均值 ×100；
#   Average ：每列对全部 K 行取均值（忽略 <0），再对列取均值 ×100；
#   Last    ：末行（忽略 <0）简单均值 ×100。
def _computed_metrics(matrix):
    if not matrix:
        return {"transfer": None, "average": None, "last": None}
    k = len(matrix)
    transfer_values = []
    for task_idx in range(1, k):
        values = [
            matrix[step_idx][task_idx]
            for step_idx in range(task_idx)
            if matrix[step_idx][task_idx] >= 0
        ]
        if values:
            transfer_values.append(mean(values) * 100)
    average_values = []
    for task_idx in range(k):
        values = [
            matrix[step_idx][task_idx]
            for step_idx in range(k)
            if matrix[step_idx][task_idx] >= 0
        ]
        if values:
            average_values.append(mean(values) * 100)
    last_values = [
        matrix[k - 1][task_idx] * 100
        for task_idx in range(k)
        if matrix[k - 1][task_idx] >= 0
    ]
    return {
        "transfer": mean(transfer_values) if transfer_values else None,
        "average": mean(average_values) if average_values else None,
        "last": mean(last_values) if last_values else None,
    }


def finalize_output(output, alphas, task_names):
    """由 output["steps"] 里所有 step 重算选择协议 / K×K 矩阵 / T/A/L / oracle。"""
    step_keys = sorted(output["steps"], key=lambda s: int(s))
    if not step_keys:
        return output
    final_key = step_keys[-1]
    final_entry = output["steps"][final_key]

    # 选择协议：最终 step 的 val 均值 argmax；升序扫描、严格大于才更新 => 并列取较小 alpha。
    val_curve = {f"{a:g}": _alpha_mean(final_entry, "val", a, task_names) for a in alphas}
    best_alpha, best_val = None, None
    for alpha in alphas:
        value = val_curve[f"{alpha:g}"]
        if best_val is None or value > best_val:
            best_alpha, best_val = alpha, value

    # 用 alpha* 重算的完整 K×K 集成矩阵（fraction，0-1；行=step，列=数据集）。
    matrix = []
    for key in step_keys:
        entry = output["steps"][key]
        matrix.append([
            entry["per_dataset"][d]["test"]["ensemble_by_alpha"][f"{best_alpha:g}"] / 100.0
            for d in task_names
        ])
    metrics = _computed_metrics(matrix)

    # oracle 表：最终 step 每 alpha 的 test 均值（仅供分析，不参与选择）。
    oracle_curve = {f"{a:g}": _alpha_mean(final_entry, "test", a, task_names) for a in alphas}

    output["selection"] = {
        "protocol": ("global alpha* = argmax over the final processed step of the "
                     "mean val-split ensemble accuracy across all datasets; ties go to "
                     "the smaller alpha (ascending scan, strict-greater update)."),
        "selection_step_index": int(final_key),
        "selected_alpha": best_alpha,
        "final_step_val_mean_by_alpha": val_curve,
        "final_step_val_mean": best_val,
    }
    output["ensemble_matrix"] = {
        "alpha": best_alpha,
        "task_names": task_names,
        "accuracy_matrix": matrix,
        "metrics": metrics,
    }
    output["oracle"] = {
        "note": "oracle only: final-step test-split mean per alpha; NOT used for selection",
        "step_index": int(final_key),
        "final_step_test_mean_by_alpha": oracle_curve,
        "oracle_alpha": max(alphas, key=lambda a: oracle_curve[f"{a:g}"]),
    }
    return output


def main():
    args = parse_args()
    alphas = parse_alphas(args.alphas)
    if args.gpu is not None and torch.cuda.is_available():
        args.device = f"cuda:{args.gpu}"
    else:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    artifacts = find_artifacts(args.async_eval_dir, args.max_steps)
    first = torch_load(artifacts[0][1], map_location="cpu")
    task_names = first["task_names"]
    del first
    print(f"found {len(artifacts)} artifact(s) under {args.async_eval_dir} "
          f"(cols={len(task_names)}, alphas={alphas}, device={args.device})", flush=True)

    output = None
    if args.output.exists() and not args.overwrite:
        with args.output.open() as handle:
            output = json.load(handle)
        print(f"resume from {args.output} (steps present: "
              f"{sorted(output.get('steps', {}), key=lambda s: int(s))})", flush=True)
    if output is None:
        output = {
            "schema_version": 1,
            "kind": "e6_alpha_scan",
            "async_eval_dir": str(args.async_eval_dir.resolve()),
            "alphas": alphas,
            "eval_batch_size": args.eval_batch_size,
            "root": args.root,
            "val_split": ("train_loader4updating: same 16-shot train set with test "
                          "transform, unshuffled (protocol has no real val split)"),
            "steps": {},
        }

    for step_index, path in artifacts:
        key = str(step_index)
        if key in output["steps"] and not args.overwrite:
            print(f"skip step {step_index + 1:02d} (already in {args.output})", flush=True)
            continue
        print(f"[step {step_index + 1:02d}] rebuilding from {path}", flush=True)
        artifact = torch_load(path, map_location="cpu")
        output["steps"][key] = evaluate_step(artifact, alphas, args)
        output["steps"][key]["artifact_path"] = str(Path(path).resolve())
        del artifact
        finalize_output(output, alphas, task_names)
        atomic_write_json(args.output, output)
        print(f"[step {step_index + 1:02d}] done -> {args.output}", flush=True)

    finalize_output(output, alphas, task_names)
    atomic_write_json(args.output, output)
    sel = output["selection"]
    met = output["ensemble_matrix"]["metrics"]
    print(f"selected alpha = {sel['selected_alpha']:g} "
          f"(final step {int(sel['selection_step_index']) + 1} val mean = "
          f"{sel['final_step_val_mean']:.2f}%)", flush=True)
    def _fmt(value):
        return "n/a" if value is None else f"{value:.2f}"
    print(f"KxK ensemble @alpha*: Transfer={_fmt(met['transfer'])} "
          f"Average={_fmt(met['average'])} Last={_fmt(met['last'])}", flush=True)
    print(f"oracle (analysis only): {output['oracle']['final_step_test_mean_by_alpha']}",
          flush=True)


if __name__ == "__main__":
    main()
