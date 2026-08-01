#!/usr/bin/env python3
"""Convert B0 frozen-zero-shot outputs to PA_main standard JSON format.

Reads:
  - <name>_zeroshot_preserve_aspect.json  (per-dataset zs_acc)
  - <name>_retrieval_baseline.json        (dict of dataset -> metrics)

Writes:
  - <output-dir>/<output-name>_zs_results.json
  - <output-dir>/<output-name>_ens_results.json
  - <output-dir>/<output-name>_retrieval.json

The frozen model sees no incremental training, so every row of the
KxK accuracy matrix is the same vector of per-dataset ZS accuracies.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.utils.continual_metrics import ContinualLearningMetrics

DEFAULT_TASKS = [
    "aircraft", "caltech101", "dtd", "eurosat", "flowers",
    "food101", "mnist", "oxford_pets", "stanford_cars", "sun397",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zeroshot-json", required=True)
    ap.add_argument("--retrieval-json", required=True)
    ap.add_argument("--output-name", default="PA__b0_preserve_aspect")
    ap.add_argument("--output-dir", default="experiments/paper_formal/PA_main")
    ap.add_argument("--task-sequence", nargs="+", default=DEFAULT_TASKS)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load zero-shot per-dataset accuracies
    with open(args.zeroshot_json) as f:
        zs = json.load(f)

    task_names = list(args.task_sequence)
    K = len(task_names)
    accs = []
    for t in task_names:
        if t not in zs:
            raise KeyError(f"task {t} not found in {args.zeroshot_json}")
        accs.append(zs[t]["zs_acc"] / 100.0)

    # Build KxK accuracy matrix: every row identical (frozen model)
    acc_matrix = np.tile(np.array(accs, dtype=float), (K, 1))

    metrics_tracker = ContinualLearningMetrics(task_names)
    metrics_tracker.accuracy_matrix = acc_matrix

    payload = {
        "args": {
            "task_sequence": task_names,
            "method": "frozen_zero_shot",
            "num_shots": 16,
            "resize_mode": "preserve_aspect",
        },
        "accuracy_matrix": acc_matrix.tolist(),
        "metrics": metrics_tracker.get_summary(),
        "per_task_metrics": metrics_tracker.calculate_per_task_metrics(),
    }

    for suffix in ("_zs_results", "_ens_results"):
        out_path = out_dir / f"{args.output_name}{suffix}.json"
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[convert] wrote {out_path}")

    # Convert retrieval baseline dict to the list format expected by aggregate_pa_results.py
    with open(args.retrieval_json) as f:
        ret = json.load(f)

    ret_list = []
    for ds_name, metrics in ret.items():
        entry = {"step": 0, "task": "frozen_b0", "dataset": ds_name}
        # Accept both @ and _ style keys
        for k, v in metrics.items():
            entry[k] = v
        ret_list.append(entry)

    ret_out = out_dir / f"{args.output_name}_retrieval.json"
    ret_out.write_text(json.dumps(ret_list, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[convert] wrote {ret_out}")


if __name__ == "__main__":
    main()
