#!/usr/bin/env python3
"""Merge async incremental eval rows into LADA-style result JSON files."""

import argparse
import csv
import json
import sys
from argparse import Namespace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.continual_metrics import ContinualLearningMetrics


CLASSIFIERS = (
    ("zero_shot", "_zs"),
    ("lr_rgda", "_rgda"),
    ("ensemble", "_ens"),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--async_eval_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--experiment_name", default=None)
    parser.add_argument("--expected_tasks", default="")
    parser.add_argument("--allow_missing", action="store_true", default=False)
    parser.add_argument("--include_retrieval", action="store_true", default=False)
    parser.add_argument("--retrieval_results_dir", type=Path, default=None)
    return parser.parse_args()


def torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def parse_list(raw):
    return [item for item in raw.replace(",", " ").split() if item]


def load_json(path):
    with Path(path).open() as handle:
        return json.load(handle)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def flatten_retrieval_payload(path, payload):
    metrics = payload["metrics"]
    return {
        "method": payload["method"],
        "step": int(payload.get("step_index", -1)) + 1 if "step_index" in payload else -1,
        "task": payload.get("task", "frozen"),
        "dataset": payload["dataset"],
        "num_images": int(metrics["num_images"]),
        "num_captions": int(metrics["num_captions"]),
        "i2t_r@1": float(metrics["i2t"].get("r@1", 0.0)),
        "i2t_r@5": float(metrics["i2t"].get("r@5", 0.0)),
        "i2t_r@10": float(metrics["i2t"].get("r@10", 0.0)),
        "t2i_r@1": float(metrics["t2i"].get("r@1", 0.0)),
        "t2i_r@5": float(metrics["t2i"].get("r@5", 0.0)),
        "t2i_r@10": float(metrics["t2i"].get("r@10", 0.0)),
        "path": str(path),
    }


def retrieval_sort_key(row):
    is_frozen = 0 if row["method"] == "frozen_clip" else 1
    return (row["dataset"], is_frozen, row["step"], row["method"])


def write_retrieval_summaries(rows, output_dir, stem):
    rows = sorted(rows, key=retrieval_sort_key)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{stem}_retrieval_summary.csv"
    md_path = output_dir / f"{stem}_retrieval_summary.md"
    json_path = output_dir / f"{stem}_retrieval_summary.json"

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

    key_metrics = {}
    for dataset in sorted({row["dataset"] for row in rows}):
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        frozen_rows = [row for row in dataset_rows if row["method"] == "frozen_clip"]
        step_rows = [row for row in dataset_rows if row["method"] != "frozen_clip"]
        latest = max(step_rows, key=lambda row: row["step"]) if step_rows else None
        key_metrics[dataset] = {}
        if frozen_rows:
            frozen = frozen_rows[0]
            key_metrics[dataset]["frozen_clip"] = {
                "i2t_r@1": frozen["i2t_r@1"],
                "t2i_r@1": frozen["t2i_r@1"],
            }
        if latest is not None:
            key_metrics[dataset]["latest_step"] = {
                "method": latest["method"],
                "step": latest["step"],
                "i2t_r@1": latest["i2t_r@1"],
                "t2i_r@1": latest["t2i_r@1"],
            }

    payload = {
        "schema_version": 1,
        "rows": rows,
        "key_metrics": key_metrics,
    }
    write_json(json_path, payload)
    return {
        "summary_json": str(json_path),
        "summary_csv": str(csv_path),
        "summary_markdown": str(md_path),
        "datasets": sorted({row["dataset"] for row in rows}),
        "key_metrics": key_metrics,
    }


def merge_retrieval_results(results_dir, output_dir, stem):
    if not results_dir.exists():
        raise SystemExit(f"retrieval results dir not found: {results_dir}")
    result_paths = sorted(results_dir.glob("*.json"))
    if not result_paths:
        raise SystemExit(f"no retrieval result JSON files found in {results_dir}")
    rows = [flatten_retrieval_payload(path, load_json(path)) for path in result_paths]
    return write_retrieval_summaries(rows, output_dir, stem)


def build_trackers(results, task_names):
    trackers = {name: ContinualLearningMetrics(task_names) for name, _ in CLASSIFIERS}
    for result in results:
        step_idx = int(result["step_index"])
        per_dataset = result["per_dataset"]
        for classifier_name, _ in CLASSIFIERS:
            row = {
                task: float(per_dataset[task][classifier_name])
                for task in task_names
            }
            trackers[classifier_name].update(step_idx, row)
    return trackers


def artifact_args(first_result):
    artifact = torch_load(first_result["artifact_path"], map_location="cpu")
    return Namespace(**artifact.get("args", {}))


def result_payload(tracker, task_names, args_data, method_name):
    summary = tracker.get_summary()
    return {
        "args": {
            "task_sequence": task_names,
            "seed": getattr(args_data, "seed", ""),
            "method": method_name,
            "eval_max_samples": getattr(args_data, "eval_max_samples", ""),
            "num_shots": getattr(args_data, "num_shots", ""),
        },
        "accuracy_matrix": tracker.get_accuracy_matrix().tolist(),
        "metrics": {
            "transfer": summary["transfer"],
            "average": summary["average"],
            "last": summary["last"],
        },
        "per_task_metrics": tracker.calculate_per_task_metrics(),
    }


def main():
    args = parse_args()
    result_dir = args.async_eval_dir / "eval_results"
    result_paths = sorted(result_dir.glob("step_*_results.json"))
    if not result_paths:
        raise SystemExit(f"no async eval results found in {result_dir}")

    results = [load_json(path) for path in result_paths]
    task_names = list(results[0]["task_names"])
    expected_tasks = parse_list(args.expected_tasks)
    if expected_tasks and task_names != expected_tasks:
        raise SystemExit(
            "task order mismatch: "
            f"results={task_names} expected={expected_tasks}"
        )

    seen_steps = {int(result["step_index"]) for result in results}
    expected_steps = set(range(len(task_names)))
    missing_steps = sorted(expected_steps - seen_steps)
    extra_steps = sorted(seen_steps - expected_steps)
    if missing_steps or extra_steps:
        message = f"missing_steps={missing_steps} extra_steps={extra_steps}"
        if not args.allow_missing:
            raise SystemExit(message)
        print(f"WARNING: {message}")

    # Keep one row per step if duplicate result files exist.
    latest_by_step = {}
    for result in results:
        latest_by_step[int(result["step_index"])] = result
    ordered_results = [latest_by_step[i] for i in sorted(latest_by_step)]

    trackers = build_trackers(ordered_results, task_names)
    run_args = artifact_args(ordered_results[0])
    output_dir = args.output_dir or (args.async_eval_dir / "merged")
    stem = args.experiment_name or getattr(run_args, "experiment_name", None) or "async_incremental"

    merged = {
        "schema_version": 1,
        "task_names": task_names,
        "source_async_eval_dir": str(args.async_eval_dir),
        "results": {},
    }

    for classifier_name, suffix in CLASSIFIERS:
        payload = result_payload(trackers[classifier_name], task_names, run_args, classifier_name)
        out_path = output_dir / f"{stem}{suffix}_results.json"
        write_json(out_path, payload)
        merged["results"][classifier_name] = {
            "path": str(out_path),
            "metrics": payload["metrics"],
        }
        print(
            f"{classifier_name}: Transfer={payload['metrics']['transfer']:.2f} "
            f"Average={payload['metrics']['average']:.2f} "
            f"Last={payload['metrics']['last']:.2f}"
        )

    if args.include_retrieval:
        retrieval_dir = args.retrieval_results_dir or (args.async_eval_dir / "retrieval_results")
        merged["retrieval"] = merge_retrieval_results(retrieval_dir, output_dir, stem)
        print(
            "retrieval: wrote "
            f"{merged['retrieval']['summary_markdown']}",
            flush=True,
        )

    write_json(output_dir / f"{stem}_merged.json", merged)


if __name__ == "__main__":
    main()
