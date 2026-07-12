#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


METRICS = (
    ("zero_shot", "CLIP-ZS"),
    ("lr_rgda", "LR-RGDA"),
    ("lada", "LADA"),
    ("ours_ensemble", "LR-RGDA+ZS"),
    ("lada_zs", "LADA+ZS"),
)

PROTOCOL_FIELDS = (
    ("num_shots", "num shots"),
    ("classifier_feature_transform", "classifier feature transform"),
    ("enable_lada", "enable lada"),
    ("lada_k", "lada k"),
    ("num_centers", "num centers"),
    ("tune_vision_encoder", "tune vision encoder"),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--output_markdown", type=Path, required=True)
    parser.add_argument(
        "--expected_seeds",
        default="",
        help="Optional space/comma-separated seed ids expected for each experiment.",
    )
    return parser.parse_args()


def format_score(values):
    if len(values) == 1:
        return f"{values[0]:.2f}"
    return f"{mean(values):.2f} +/- {stdev(values):.2f}"


def format_min(values):
    return f"{min(values):.2f}" if values else ""


def format_all_positive(values):
    return "yes" if values and all(value > 0 for value in values) else "no"


def format_seed_values(seed_values):
    return ", ".join(f"{seed}:{value:.2f}" for seed, value in seed_values)


def format_seed_text_values(seed_values):
    return ", ".join(f"{seed}:{value}" for seed, value in seed_values)


def format_protocol_values(values):
    unique = sorted({str(value) for value in values if value != ""})
    return ",".join(unique)


def parse_seed_list(raw):
    return [seed for seed in raw.replace(",", " ").split() if seed]


def main():
    args = parse_args()
    grouped = defaultdict(lambda: defaultdict(list))
    seeds = defaultdict(set)
    seed_metric_values = defaultdict(lambda: defaultdict(list))
    source_files = defaultdict(list)
    protocol_values = defaultdict(lambda: defaultdict(list))
    lr_lada_deltas = defaultdict(list)
    ensemble_lada_deltas = defaultdict(list)
    expected_seeds = set(parse_seed_list(args.expected_seeds))

    for path in sorted(args.input_dir.glob("*_seed*.json")):
        with path.open() as handle:
            result = json.load(handle)
        experiment = result.get("experiment_name") or path.stem
        arguments = result.get("arguments", {}) if isinstance(result.get("arguments"), dict) else {}
        averages = result["metrics"]["id"]["average"]
        if "seed" in result:
            seed = str(result["seed"])
        else:
            seed_part = path.stem.rsplit("_seed", 1)
            seed = seed_part[1] if len(seed_part) == 2 else ""
        for key, _ in METRICS:
            if key in averages:
                value = float(averages[key])
                grouped[experiment][key].append(value)
                if seed:
                    seed_metric_values[experiment][key].append((seed, value))
        if seed:
            seeds[experiment].add(seed)
            source_files[experiment].append((seed, str(path)))
        for key, _ in PROTOCOL_FIELDS:
            protocol_values[experiment][key].append(arguments.get(key, ""))
        if "lr_rgda" in averages and "lada" in averages:
            lr_lada_deltas[experiment].append(
                (seed, float(averages["lr_rgda"]) - float(averages["lada"]))
            )
        if "ours_ensemble" in averages and "lada" in averages:
            ensemble_lada_deltas[experiment].append(
                (seed, float(averages["ours_ensemble"]) - float(averages["lada"]))
            )

    rows = []
    for experiment in sorted(grouped):
        present_seeds = seeds[experiment]
        missing_seeds = sorted(expected_seeds - present_seeds)
        row = {
            "experiment": experiment,
            "n": str(max((len(values) for values in grouped[experiment].values()), default=0)),
            "seeds": ",".join(sorted(present_seeds)),
            "missing expected seeds": ",".join(missing_seeds),
            "source files": format_seed_text_values(sorted(source_files[experiment])),
        }
        for key, label in PROTOCOL_FIELDS:
            row[label] = format_protocol_values(protocol_values[experiment].get(key, []))
        for key, label in METRICS:
            values = grouped[experiment].get(key, [])
            row[label] = format_score(values) if values else ""
            row[f"{label} by seed"] = (
                format_seed_values(sorted(seed_metric_values[experiment].get(key, [])))
                if seed_metric_values[experiment].get(key)
                else ""
            )
        if lr_lada_deltas[experiment]:
            delta_pairs = sorted(lr_lada_deltas[experiment])
            deltas = [value for _, value in delta_pairs]
            row["LR-RGDA - LADA"] = format_score(deltas)
            row["min LR-RGDA - LADA"] = format_min(deltas)
            row["all LR-RGDA > LADA"] = format_all_positive(deltas)
            row["LR-RGDA - LADA by seed"] = format_seed_values(delta_pairs)
        else:
            row["LR-RGDA - LADA"] = ""
            row["min LR-RGDA - LADA"] = ""
            row["all LR-RGDA > LADA"] = "no"
            row["LR-RGDA - LADA by seed"] = ""
        if ensemble_lada_deltas[experiment]:
            delta_pairs = sorted(ensemble_lada_deltas[experiment])
            deltas = [value for _, value in delta_pairs]
            row["LR-RGDA+ZS - LADA"] = format_score(deltas)
            row["min LR-RGDA+ZS - LADA"] = format_min(deltas)
            row["all LR-RGDA+ZS > LADA"] = format_all_positive(deltas)
            row["LR-RGDA+ZS - LADA by seed"] = format_seed_values(delta_pairs)
        else:
            row["LR-RGDA+ZS - LADA"] = ""
            row["min LR-RGDA+ZS - LADA"] = ""
            row["all LR-RGDA+ZS > LADA"] = "no"
            row["LR-RGDA+ZS - LADA by seed"] = ""
        rows.append(row)

    headers = [
        "experiment",
        "n",
        "seeds",
        "missing expected seeds",
        "source files",
        *[label for _, label in PROTOCOL_FIELDS],
        *[label for _, label in METRICS],
        *[f"{label} by seed" for _, label in METRICS],
        "LR-RGDA - LADA",
        "min LR-RGDA - LADA",
        "all LR-RGDA > LADA",
        "LR-RGDA - LADA by seed",
        "LR-RGDA+ZS - LADA",
        "min LR-RGDA+ZS - LADA",
        "all LR-RGDA+ZS > LADA",
        "LR-RGDA+ZS - LADA by seed",
    ]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "| Experiment | " + " | ".join(headers[1:]) + " |",
        "|---|" + "|".join("---:" for _ in headers[1:]) + "|",
    ]
    for row in rows:
        lines.append(
            "| " + row["experiment"] + " | "
            + " | ".join(row[h] for h in headers[1:]) + " |"
        )
    args.output_markdown.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
