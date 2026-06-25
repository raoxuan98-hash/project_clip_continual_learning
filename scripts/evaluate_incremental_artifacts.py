#!/usr/bin/env python3
"""Evaluate async incremental-learning step artifacts.

The training process writes immutable artifacts plus queue/*.ready.json files.
This worker claims ready jobs, evaluates each artifact on all tasks, and writes
one result JSON per step under eval_results/.
"""

import argparse
import json
import os
import sys
import time
from argparse import Namespace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.classifiers.lr_rgda_classifier import LRRGDAClassifier
from src.utils.main_utils import evaluate_dataset, get_zeroshot_classifier
from main_incremental import _build_text_classifier
from src.utils.retrieval_eval import (
    evaluate_retrieval_model,
    load_retrieval_dataset,
    parse_recall_ks,
    parse_retrieval_roots,
    retrieval_payload,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--async_eval_dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--poll_interval", type=float, default=30.0)
    parser.add_argument("--once", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--enable_retrieval_eval", action="store_true", default=False)
    parser.add_argument(
        "--retrieval_datasets",
        default="",
        help="Comma-separated retrieval datasets: coco_val2014,flickr30k_cn,flickr8k.",
    )
    parser.add_argument(
        "--retrieval_roots",
        default="",
        help="Comma-separated dataset=/path entries, e.g. coco_val2014=/data/coco.",
    )
    parser.add_argument("--retrieval_output_dir", type=Path, default=None)
    parser.add_argument("--retrieval_max_images", type=int, default=0)
    parser.add_argument("--retrieval_batch_size", type=int, default=128)
    parser.add_argument("--retrieval_text_batch_size", type=int, default=256)
    parser.add_argument("--retrieval_num_workers", type=int, default=4)
    parser.add_argument("--retrieval_similarity_chunk_size", type=int, default=512)
    parser.add_argument("--retrieval_recall_ks", default="1,5,10")
    parser.add_argument("--include_frozen_retrieval_baseline", action="store_true", default=False)
    return parser.parse_args()


def torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def claim_job(ready_path):
    running_path = ready_path.with_name(ready_path.name.replace(".ready.json", ".running.json"))
    try:
        os.replace(ready_path, running_path)
    except FileNotFoundError:
        return None
    return running_path


def load_json(path):
    with Path(path).open() as handle:
        return json.load(handle)


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp_path, path)


def parse_retrieval_dataset_names(raw):
    return [item for item in str(raw).replace(",", " ").split() if item]


def artifact_args(artifact, device):
    args = Namespace(**artifact["args"])
    args.device = device
    if not hasattr(args, "gpu"):
        args.gpu = 0
    return args


def rebuild_model_and_classifier(artifact, device):
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

    config = artifact["lr_rgda"]["config"]
    lr_rgda = LRRGDAClassifier(
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
    lr_rgda.classifier.load_state_dict(artifact["lr_rgda"]["state_dict"])
    lr_rgda.classifier.to(device)
    lr_rgda.classifier.eval()

    return args, model, processor, frozen_model, lr_rgda


def rebuild_frozen_model(artifact, device):
    from src.models.clip import get_clip_model

    args = artifact_args(artifact, device)
    model, processor = get_clip_model(args, train_mode="frozen")
    model.to(device)
    model.eval()
    return args, model, processor


def _retrieval_output_dir(args):
    return args.retrieval_output_dir or (args.async_eval_dir / "retrieval_results")


def _retrieval_eval_args(args):
    return {
        "max_images": args.retrieval_max_images,
        "batch_size": args.retrieval_batch_size,
        "text_batch_size": args.retrieval_text_batch_size,
        "num_workers": args.retrieval_num_workers,
        "recall_ks": parse_recall_ks(args.retrieval_recall_ks),
        "similarity_chunk_size": args.retrieval_similarity_chunk_size,
    }


def _evaluate_retrieval_dataset(model, processor, dataset, args):
    eval_args = _retrieval_eval_args(args)
    return evaluate_retrieval_model(
        model,
        processor,
        dataset,
        args.device,
        batch_size=eval_args["batch_size"],
        text_batch_size=eval_args["text_batch_size"],
        num_workers=eval_args["num_workers"],
        recall_ks=eval_args["recall_ks"],
        similarity_chunk_size=eval_args["similarity_chunk_size"],
    )


def _write_retrieval_json(path, payload):
    atomic_write_json(path, payload)


def _evaluate_frozen_retrieval_if_needed(artifact, dataset_name, dataset, args):
    out_dir = _retrieval_output_dir(args)
    out_path = out_dir / f"frozen_clip_{dataset_name}.json"
    if out_path.exists() and not args.overwrite:
        print(f"reuse frozen retrieval baseline: {out_path}", flush=True)
        return

    _, model, processor = rebuild_frozen_model(artifact, args.device)
    metrics = _evaluate_retrieval_dataset(model, processor, dataset, args)
    payload = retrieval_payload(
        "frozen_clip",
        dataset,
        metrics,
        args=_retrieval_eval_args(args),
    )
    _write_retrieval_json(out_path, payload)
    print(
        f"frozen_clip {dataset_name}: "
        f"I2T R@1={metrics['i2t'].get('r@1', 0.0):.2f} "
        f"T2I R@1={metrics['t2i'].get('r@1', 0.0):.2f}",
        flush=True,
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_retrieval_artifact(artifact, artifact_path, model, processor, args):
    if not args.enable_retrieval_eval:
        return

    dataset_names = parse_retrieval_dataset_names(args.retrieval_datasets)
    if not dataset_names:
        raise ValueError("--enable_retrieval_eval requires --retrieval_datasets")
    roots = parse_retrieval_roots(args.retrieval_roots)
    missing_roots = [name for name in dataset_names if name not in roots]
    if missing_roots:
        raise ValueError(
            "--retrieval_roots is missing entries for: " + ", ".join(missing_roots)
        )

    out_dir = _retrieval_output_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    step = int(artifact["step_index"])
    task = artifact["task_name"]
    method = f"step_{step + 1:02d}_{task}"

    for dataset_name in dataset_names:
        dataset = load_retrieval_dataset(
            dataset_name,
            roots[dataset_name],
            max_images=args.retrieval_max_images,
        )
        if args.include_frozen_retrieval_baseline:
            _evaluate_frozen_retrieval_if_needed(artifact, dataset_name, dataset, args)

        out_path = out_dir / f"{method}_{dataset_name}.json"
        if out_path.exists() and not args.overwrite:
            print(f"skip existing retrieval result: {out_path}", flush=True)
            continue

        metrics = _evaluate_retrieval_dataset(model, processor, dataset, args)
        payload = retrieval_payload(
            method,
            dataset,
            metrics,
            args=_retrieval_eval_args(args),
            artifact_path=artifact_path,
            step_index=step,
            task=task,
        )
        _write_retrieval_json(out_path, payload)
        print(
            f"{method} {dataset_name}: "
            f"I2T R@1={metrics['i2t'].get('r@1', 0.0):.2f} "
            f"T2I R@1={metrics['t2i'].get('r@1', 0.0):.2f}",
            flush=True,
        )


def evaluate_artifact(artifact_path, output_path, args):
    artifact = torch_load(artifact_path, map_location="cpu")
    run_args, model, processor, frozen_model, lr_rgda = rebuild_model_and_classifier(
        artifact, args.device
    )

    task_names = artifact["task_names"]
    global_class_names = artifact["global_class_names"]
    dataset_label_offsets = artifact["dataset_label_offsets"]
    dataset_class_counts = artifact["dataset_class_counts"]
    current_num_classes = artifact["current_num_classes"]

    cached_seen_text_features = artifact.get("cached_seen_text_features")
    zeroshot_classifier = _build_text_classifier(
        run_args,
        model,
        processor,
        frozen_model,
        global_class_names,
        cached_seen_text_features,
        args.device,
    )

    per_dataset = {}
    for task_idx, d_name in enumerate(task_names):
        zs_acc, rgda_acc, ens_acc, lada_acc, lada_zs_acc, c_len, _ = evaluate_dataset(
            run_args,
            d_name,
            model,
            zeroshot_classifier,
            lr_rgda,
            current_num_classes,
            dataset_label_offsets[d_name],
        )
        expected_c_len = dataset_class_counts[d_name]
        if c_len != expected_c_len:
            raise ValueError(
                f"Class count mismatch for {d_name}: eval={c_len}, expected={expected_c_len}"
            )
        per_dataset[d_name] = {
            "zero_shot": zs_acc / 100.0,
            "lr_rgda": rgda_acc / 100.0,
            "ensemble": ens_acc / 100.0,
        }
        print(
            f"[step {artifact['step_index'] + 1:02d} task {task_idx + 1:02d}] "
            f"{d_name}: ZS={zs_acc:.2f} RGDA={rgda_acc:.2f} ENS={ens_acc:.2f}",
            flush=True,
        )

    result = {
        "schema_version": 1,
        "step_index": artifact["step_index"],
        "task_name": artifact["task_name"],
        "task_names": task_names,
        "artifact_path": str(Path(artifact_path).resolve()),
        "current_num_classes": current_num_classes,
        "per_dataset": per_dataset,
    }
    atomic_write_json(output_path, result)
    evaluate_retrieval_artifact(artifact, artifact_path, model, processor, args)


def evaluate_retrieval_only(artifact_path, args):
    artifact = torch_load(artifact_path, map_location="cpu")
    _, model, processor, _, _ = rebuild_model_and_classifier(artifact, args.device)
    evaluate_retrieval_artifact(artifact, artifact_path, model, processor, args)


def process_one(job_path, args):
    running_path = claim_job(job_path)
    if running_path is None:
        return False

    payload = load_json(running_path)
    artifact_path = Path(payload["artifact_path"])
    result_dir = args.async_eval_dir / "eval_results"
    result_path = result_dir / f"step_{payload['step_index'] + 1:02d}_{payload['task_name']}_results.json"

    if result_path.exists() and not args.overwrite:
        try:
            if args.enable_retrieval_eval:
                evaluate_retrieval_only(artifact_path, args)
        except Exception:
            failed_path = running_path.with_name(running_path.name.replace(".running.json", ".failed.json"))
            os.replace(running_path, failed_path)
            raise
        done_path = running_path.with_name(running_path.name.replace(".running.json", ".done.json"))
        os.replace(running_path, done_path)
        print(f"skip existing result: {result_path}", flush=True)
        return True

    try:
        evaluate_artifact(artifact_path, result_path, args)
    except Exception:
        failed_path = running_path.with_name(running_path.name.replace(".running.json", ".failed.json"))
        os.replace(running_path, failed_path)
        raise

    done_path = running_path.with_name(running_path.name.replace(".running.json", ".done.json"))
    os.replace(running_path, done_path)
    print(f"wrote {result_path}", flush=True)
    return True


def main():
    args = parse_args()
    if args.device is None:
        if args.gpu is not None and torch.cuda.is_available():
            args.device = f"cuda:{args.gpu}"
        else:
            args.device = "cuda" if torch.cuda.is_available() else "cpu"

    queue_dir = args.async_eval_dir / "queue"
    queue_dir.mkdir(parents=True, exist_ok=True)

    while True:
        ready_jobs = sorted(queue_dir.glob("*.ready.json"))
        processed = False
        for job_path in ready_jobs:
            processed = process_one(job_path, args) or processed
        if args.once:
            break
        if not processed:
            time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
