#!/usr/bin/env python3
"""Evaluate image-text retrieval for incremental CLIP artifacts."""

import argparse
import csv
import json
import sys
from argparse import Namespace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_incremental_rgda_sweep_artifacts import artifact_paths, torch_load
from src.models.clip import get_clip_model
from src.utils.retrieval_eval import (
    evaluate_retrieval_model,
    flatten_retrieval_row,
    load_retrieval_dataset,
    parse_recall_ks,
    retrieval_payload,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--async_eval_dir", type=Path, default=None)
    parser.add_argument("--artifact_path", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--experiment_name", default="retrieval_eval")
    parser.add_argument("--device", default=None)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument(
        "--dataset",
        default="flickr8k",
        choices=["coco_val2014", "mscoco_2014_5k", "flickr30k_hf", "flickr30k_cn", "flickr8k"],
    )
    parser.add_argument("--retrieval_root", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--text_batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--similarity_chunk_size", type=int, default=512)
    parser.add_argument("--recall_ks", default="1,5,10")
    parser.add_argument(
        "--include_frozen_baseline",
        action="store_true",
        help="Also evaluate the frozen CLIP model built from the first artifact args.",
    )
    return parser.parse_args()


def artifact_args(artifact, device):
    args = Namespace(**artifact["args"])
    args.device = device
    if not hasattr(args, "gpu"):
        args.gpu = 0
    return args


def load_artifacts(async_eval_dir, artifact_path):
    if artifact_path is not None:
        artifact = torch_load(artifact_path, map_location="cpu")
        return [(int(artifact["step_index"]), artifact_path, artifact)]
    if async_eval_dir is None:
        raise SystemExit("Either --async_eval_dir or --artifact_path is required")
    return artifact_paths(async_eval_dir)


def rebuild_artifact_model(artifact, device):
    args = artifact_args(artifact, device)
    model, processor = get_clip_model(args, train_mode="lora")
    model.load_state_dict(artifact["model_state_dict"])
    model.to(device)
    model.eval()
    return args, model, processor


def rebuild_frozen_model(artifact, device):
    args = artifact_args(artifact, device)
    model, processor = get_clip_model(args, train_mode="frozen")
    model.to(device)
    model.eval()
    return args, model, processor


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_summary(output_dir, rows):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "retrieval_summary.csv"
    md_path = output_dir / "retrieval_summary.md"
    headers = [
        "method",
        "step",
        "task",
        "dataset",
        "num_images",
        "num_captions",
        "i2t_r@1",
        "i2t_r@5",
        "i2t_r@10",
        "t2i_r@1",
        "t2i_r@5",
        "t2i_r@10",
        "path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("| method | step | task | dataset | I2T R@1 | I2T R@5 | I2T R@10 | T2I R@1 | T2I R@5 | T2I R@10 |\n")
        handle.write("|---|---:|---|---|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                f"| {row['method']} | {row['step']} | {row['task']} | {row['dataset']} | "
                f"{row['i2t_r@1']:.2f} | {row['i2t_r@5']:.2f} | {row['i2t_r@10']:.2f} | "
                f"{row['t2i_r@1']:.2f} | {row['t2i_r@5']:.2f} | {row['t2i_r@10']:.2f} |\n"
            )


def evaluate_one(model, processor, dataset, args, ks):
    return evaluate_retrieval_model(
        model,
        processor,
        dataset,
        args.device,
        batch_size=args.batch_size,
        text_batch_size=args.text_batch_size,
        num_workers=args.num_workers,
        recall_ks=ks,
        similarity_chunk_size=args.similarity_chunk_size,
    )


def main():
    args = parse_args()
    if args.device is None:
        if args.gpu is not None and torch.cuda.is_available():
            args.device = f"cuda:{args.gpu}"
        else:
            args.device = "cuda" if torch.cuda.is_available() else "cpu"

    ks = parse_recall_ks(args.recall_ks)
    artifacts = load_artifacts(args.async_eval_dir, args.artifact_path)
    dataset = load_retrieval_dataset(args.dataset, args.retrieval_root, args.max_images)

    rows = []
    if args.include_frozen_baseline:
        out_path = args.output_dir / f"{args.experiment_name}_frozen_clip_{args.dataset}.json"
        if out_path.exists():
            payload = read_json(out_path)
            metrics = payload["metrics"]
            print(f"Reusing frozen CLIP baseline from {out_path}", flush=True)
        else:
            _, _, artifact = artifacts[0]
            _, model, processor = rebuild_frozen_model(artifact, args.device)
            metrics = evaluate_one(model, processor, dataset, args, ks)
            payload = retrieval_payload(
                "frozen_clip",
                dataset,
                metrics,
                args={
                    "max_images": args.max_images,
                    "batch_size": args.batch_size,
                    "text_batch_size": args.text_batch_size,
                    "recall_ks": ks,
                },
            )
            write_json(out_path, payload)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        rows.append(flatten_retrieval_row("frozen_clip", -1, "frozen", args.dataset, metrics, out_path))

    for step_index, artifact_path, artifact in artifacts:
        run_args, model, processor = rebuild_artifact_model(artifact, args.device)
        task_names = artifact.get("task_names", [])
        task = task_names[int(step_index)] if int(step_index) < len(task_names) else f"step_{step_index + 1}"
        method = f"step_{int(step_index) + 1:02d}_{task}"
        metrics = evaluate_one(model, processor, dataset, args, ks)
        out_path = args.output_dir / f"{args.experiment_name}_{method}_{args.dataset}.json"
        payload = retrieval_payload(
            method,
            dataset,
            metrics,
            args={
                "max_images": args.max_images,
                "batch_size": args.batch_size,
                "text_batch_size": args.text_batch_size,
                "recall_ks": ks,
                "source_seed": getattr(run_args, "seed", None),
            },
            artifact_path=artifact_path,
            step_index=step_index,
            task=task,
        )
        write_json(out_path, payload)
        rows.append(flatten_retrieval_row(method, int(step_index) + 1, task, args.dataset, metrics, out_path))
        print(f"{method}: I2T R@1={metrics['i2t'].get('r@1', 0.0):.2f} "
              f"T2I R@1={metrics['t2i'].get('r@1', 0.0):.2f}", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_summary(args.output_dir, rows)
    print(f"Wrote retrieval summary to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
