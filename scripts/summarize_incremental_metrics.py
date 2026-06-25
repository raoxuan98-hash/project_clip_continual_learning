#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


METRIC_ALIASES = {
    "transfer": ("metrics", "transfer"),
    "average": ("metrics", "average"),
    "last": ("metrics", "last"),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Result JSON files or directories containing *_results.json files.",
    )
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--output_markdown", type=Path, required=True)
    parser.add_argument("--aggregate_csv", type=Path)
    parser.add_argument("--aggregate_markdown", type=Path)
    parser.add_argument(
        "--expected_tasks",
        default="",
        help="Optional space/comma-separated task names expected in each result.",
    )
    return parser.parse_args()


def parse_list(raw):
    return [item for item in raw.replace(",", " ").split() if item]


def result_paths(inputs):
    paths = {}
    for item in inputs:
        if item.is_dir():
            for path in sorted(item.rglob("*_results.json")):
                paths[path.resolve()] = path
            # Compatibility with older main_incremental.py outputs generated
            # before per-classifier files used the *_results.json suffix.
            for pattern in ("*_zs.json", "*_rgda.json", "*_ens.json"):
                for path in sorted(item.rglob(pattern)):
                    paths[path.resolve()] = path
        else:
            paths[item.resolve()] = item
    return list(paths.values())


def nested_get(data, keys):
    current = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def metric_value(data, name):
    value = nested_get(data, METRIC_ALIASES[name])
    if value is not None:
        return float(value)
    summary = data.get("summary")
    if isinstance(summary, dict) and name in summary:
        return float(summary[name])
    return None


def task_names(data):
    if "args" in data and isinstance(data["args"], dict) and "task_sequence" in data["args"]:
        return list(data["args"]["task_sequence"])
    if "task_names" in data:
        return list(data["task_names"])
    per_task = data.get("per_task_metrics")
    if isinstance(per_task, dict):
        return list(per_task.keys())
    return []


def validate_matrix(matrix, k):
    if not isinstance(matrix, list):
        return "missing accuracy_matrix"
    if len(matrix) != k:
        return f"accuracy_matrix rows={len(matrix)}, expected {k}"
    for idx, row in enumerate(matrix, start=1):
        if not isinstance(row, list):
            return f"accuracy_matrix row {idx} is not a list"
        if len(row) != k:
            return f"accuracy_matrix row {idx} cols={len(row)}, expected {k}"
    return ""


def task_order_warning(tasks, expected_tasks):
    if not expected_tasks or not tasks or tasks == expected_tasks:
        return ""
    if set(tasks) == set(expected_tasks):
        return "task_sequence order differs from expected order"
    return ""


def computed_metrics(matrix):
    if not isinstance(matrix, list) or not matrix:
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


def metric_delta(stored, computed):
    if stored is None or computed is None:
        return ""
    return f"{stored - computed:+.4f}"


def format_score(value):
    return "" if value is None else f"{value:.2f}"


def parse_float(value):
    if value == "":
        return None
    return float(value)


def format_mean_std(values):
    if not values:
        return ""
    if len(values) == 1:
        return f"{values[0]:.2f}"
    return f"{mean(values):.2f} +/- {stdev(values):.2f}"


def write_table(rows, headers, output_csv, output_markdown):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "| " + " | ".join(headers) + " |",
        "|---" * len(headers) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[h] for h in headers) + " |")
    output_markdown.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def aggregate_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)

    aggregate = []
    for method in sorted(grouped):
        method_rows = grouped[method]
        seeds = sorted({row["seed"] for row in method_rows if row["seed"]})
        missing_tasks = sorted(
            {
                task
                for row in method_rows
                for task in row["missing expected tasks"].split(",")
                if task
            }
        )
        warnings = sorted({row["matrix warning"] for row in method_rows if row["matrix warning"]})
        order_warnings = sorted(
            {row["task order warning"] for row in method_rows if row["task order warning"]}
        )
        out = {
            "method": method,
            "n": str(len(method_rows)),
            "seeds": ",".join(seeds),
            "K": ",".join(sorted({row["K"] for row in method_rows})),
            "missing expected tasks": ",".join(missing_tasks),
            "task order warning": "; ".join(order_warnings),
            "Transfer": format_mean_std(
                [value for value in (parse_float(row["Transfer"]) for row in method_rows) if value is not None]
            ),
            "Average": format_mean_std(
                [value for value in (parse_float(row["Average"]) for row in method_rows) if value is not None]
            ),
            "Last": format_mean_std(
                [value for value in (parse_float(row["Last"]) for row in method_rows) if value is not None]
            ),
            "Forgetting": format_mean_std(
                [value for value in (parse_float(row["Forgetting"]) for row in method_rows) if value is not None]
            ),
            "matrix warning": "; ".join(warnings),
        }
        aggregate.append(out)
    return aggregate


def main():
    args = parse_args()
    expected_tasks = parse_list(args.expected_tasks)
    rows = []

    for path in result_paths(args.inputs):
        with path.open() as handle:
            data = json.load(handle)
        tasks = task_names(data)
        k = len(tasks)
        args_data = data.get("args", {}) if isinstance(data.get("args"), dict) else {}
        method = (
            args_data.get("experiment_name")
            or args_data.get("method_label")
            or args_data.get("method")
            or args_data.get("lora_type")
            or path.stem.replace("_results", "")
        )
        seed = args_data.get("seed", "")
        eval_max_samples = args_data.get("eval_max_samples", "")
        matrix_warning = validate_matrix(data.get("accuracy_matrix"), k)
        recomputed = computed_metrics(data.get("accuracy_matrix")) if not matrix_warning else {}
        stored_transfer = metric_value(data, "transfer")
        stored_average = metric_value(data, "average")
        stored_last = metric_value(data, "last")
        missing_tasks = sorted(set(expected_tasks) - set(tasks)) if expected_tasks else []
        order_warning = task_order_warning(tasks, expected_tasks)
        row = {
            "path": str(path),
            "method": str(method),
            "seed": str(seed),
            "K": str(k),
            "eval max samples": str(eval_max_samples),
            "missing expected tasks": ",".join(missing_tasks),
            "task order warning": order_warning,
            "Transfer": format_score(stored_transfer),
            "Average": format_score(stored_average),
            "Last": format_score(stored_last),
            "Forgetting": format_score(data.get("forgetting_rate")),
            "computed Transfer": format_score(recomputed.get("transfer")),
            "computed Average": format_score(recomputed.get("average")),
            "computed Last": format_score(recomputed.get("last")),
            "Transfer stored-computed": metric_delta(stored_transfer, recomputed.get("transfer")),
            "Average stored-computed": metric_delta(stored_average, recomputed.get("average")),
            "Last stored-computed": metric_delta(stored_last, recomputed.get("last")),
            "matrix warning": matrix_warning,
        }
        rows.append(row)

    headers = [
        "path",
        "method",
        "seed",
        "K",
        "eval max samples",
        "missing expected tasks",
        "task order warning",
        "Transfer",
        "Average",
        "Last",
        "Forgetting",
        "computed Transfer",
        "computed Average",
        "computed Last",
        "Transfer stored-computed",
        "Average stored-computed",
        "Last stored-computed",
        "matrix warning",
    ]
    write_table(rows, headers, args.output_csv, args.output_markdown)

    if args.aggregate_csv or args.aggregate_markdown:
        if not args.aggregate_csv or not args.aggregate_markdown:
            raise SystemExit("--aggregate_csv and --aggregate_markdown must be provided together")
        aggregate_headers = [
            "method",
            "n",
            "seeds",
            "K",
            "missing expected tasks",
            "task order warning",
            "Transfer",
            "Average",
            "Last",
            "Forgetting",
            "matrix warning",
        ]
        aggregate = aggregate_rows(rows)
        print()
        write_table(aggregate, aggregate_headers, args.aggregate_csv, args.aggregate_markdown)


if __name__ == "__main__":
    main()
