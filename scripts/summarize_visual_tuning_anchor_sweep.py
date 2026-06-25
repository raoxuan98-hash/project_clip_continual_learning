#!/usr/bin/env python3
"""Summarize visual-tuning anchor sweep results across configs."""

import argparse
import csv
import json
from pathlib import Path


CONFIGS = (
    "text_only_anchor_seed42",
    "vanilla_vision_anchor_seed42",
    "nsp_fd_cd_vision_anchor_seed42",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--output_prefix", default="visual_tuning_anchor_comparison")
    parser.add_argument("--allow_incomplete", action="store_true", default=False)
    return parser.parse_args()


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def metric(row, name):
    return float(row[name])


def load_config_rows(out_dir, config):
    path = out_dir / "sweep" / config / "rgda_sweep_summary.csv"
    if not path.exists():
        return None, path
    rows = read_csv(path)
    return {row["method"]: row for row in rows}, path


def method_sort_key(method):
    if method == "zero_shot":
        return (0, method)
    if method.startswith("rgda_"):
        return (1, method)
    if method.startswith("ens_"):
        return (2, method)
    return (3, method)


def best_ensemble(rows):
    candidates = [row for name, row in rows.items() if name.startswith("ens_")]
    if not candidates:
        return None
    return max(candidates, key=lambda row: metric(row, "average"))


def write_csv(path, rows, headers):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows, headers, title):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"# {title}\n\n")
        if not rows:
            handle.write("No rows available.\n")
            return
        handle.write("| " + " | ".join(headers) + " |\n")
        handle.write("|" + "|".join("---" for _ in headers) + "|\n")
        for row in rows:
            handle.write("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |\n")


def fmt(value):
    return f"{value:.2f}"


def main():
    args = parse_args()
    loaded = {}
    missing = {}
    for config in CONFIGS:
        rows, path = load_config_rows(args.out_dir, config)
        if rows is None:
            missing[config] = str(path)
        else:
            loaded[config] = rows

    status = {
        "out_dir": str(args.out_dir),
        "expected_configs": list(CONFIGS),
        "loaded_configs": sorted(loaded),
        "missing": missing,
    }

    report_dir = args.out_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / f"{args.output_prefix}_status.json").open("w", encoding="utf-8") as handle:
        json.dump(status, handle, indent=2)

    if missing and not args.allow_incomplete:
        raise SystemExit(
            "Missing sweep summaries: "
            + ", ".join(f"{cfg}={path}" for cfg, path in missing.items())
        )

    all_methods = sorted(
        {method for rows in loaded.values() for method in rows},
        key=method_sort_key,
    )

    wide_rows = []
    for method in all_methods:
        row = {"method": method}
        for config, rows in loaded.items():
            if method not in rows:
                continue
            source = rows[method]
            row[f"{config}_transfer"] = fmt(metric(source, "transfer"))
            row[f"{config}_average"] = fmt(metric(source, "average"))
            row[f"{config}_last"] = fmt(metric(source, "last"))
        wide_rows.append(row)

    wide_headers = ["method"]
    for config in CONFIGS:
        wide_headers.extend(
            [f"{config}_transfer", f"{config}_average", f"{config}_last"]
        )
    write_csv(report_dir / f"{args.output_prefix}_wide.csv", wide_rows, wide_headers)
    write_markdown(
        report_dir / f"{args.output_prefix}_wide.md",
        wide_rows,
        wide_headers,
        "Visual Tuning Anchor Sweep",
    )

    delta_rows = []
    comparisons = (
        ("vanilla_minus_text_only", "vanilla_vision_anchor_seed42", "text_only_anchor_seed42"),
        ("nsp_minus_text_only", "nsp_fd_cd_vision_anchor_seed42", "text_only_anchor_seed42"),
        ("nsp_minus_vanilla", "nsp_fd_cd_vision_anchor_seed42", "vanilla_vision_anchor_seed42"),
    )
    for label, left, right in comparisons:
        if left not in loaded or right not in loaded:
            continue
        common_methods = sorted(set(loaded[left]) & set(loaded[right]), key=method_sort_key)
        for method in common_methods:
            lrow = loaded[left][method]
            rrow = loaded[right][method]
            delta_rows.append(
                {
                    "comparison": label,
                    "method": method,
                    "delta_transfer": fmt(metric(lrow, "transfer") - metric(rrow, "transfer")),
                    "delta_average": fmt(metric(lrow, "average") - metric(rrow, "average")),
                    "delta_last": fmt(metric(lrow, "last") - metric(rrow, "last")),
                }
            )

    delta_headers = ["comparison", "method", "delta_transfer", "delta_average", "delta_last"]
    write_csv(report_dir / f"{args.output_prefix}_deltas.csv", delta_rows, delta_headers)
    write_markdown(
        report_dir / f"{args.output_prefix}_deltas.md",
        delta_rows,
        delta_headers,
        "Visual Tuning Anchor Deltas",
    )

    best_rows = []
    for config, rows in loaded.items():
        best = best_ensemble(rows)
        if best is None:
            continue
        best_rows.append(
            {
                "config": config,
                "method": best["method"],
                "transfer": fmt(metric(best, "transfer")),
                "average": fmt(metric(best, "average")),
                "last": fmt(metric(best, "last")),
            }
        )
    best_headers = ["config", "method", "transfer", "average", "last"]
    write_csv(report_dir / f"{args.output_prefix}_best_ensemble.csv", best_rows, best_headers)
    write_markdown(
        report_dir / f"{args.output_prefix}_best_ensemble.md",
        best_rows,
        best_headers,
        "Best Ensemble by Average",
    )

    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
