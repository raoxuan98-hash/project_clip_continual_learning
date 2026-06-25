#!/usr/bin/env python3
"""Append LADA ablation summaries to chat-history when runs finish.

This script is intentionally passive: it does not launch or kill training jobs.
It waits for summary files produced by run_lada_recipe_text_rgda_ablation.sh and
records their contents once, guarded by stable markdown markers.
"""

import argparse
import subprocess
import time
from datetime import datetime
from pathlib import Path


CONFIG_NAMES = (
    "base_current",
    "recipe_current",
    "no_fd_cd_current",
    "no_fd_cd_lada_hybrid",
    "vanilla_no_fd_cd_current",
    "vanilla_no_fd_cd_lada_hybrid",
    "text_only_no_fd_cd_lada_hybrid",
)

CLASSIFIER_SUFFIXES = ("zs", "rgda", "ens")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity-dir", type=Path, required=True)
    parser.add_argument("--four-dir", type=Path, required=True)
    parser.add_argument("--chat-history", type=Path, required=True)
    parser.add_argument("--poll-interval", type=float, default=300.0)
    parser.add_argument("--conda-env", default="raoxuan")
    return parser.parse_args()


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_text(path):
    return path.read_text(encoding="utf-8", errors="replace")


def append_once(chat_history, marker, content):
    chat_history.parent.mkdir(parents=True, exist_ok=True)
    existing = read_text(chat_history) if chat_history.exists() else ""
    if marker in existing:
        return False
    with chat_history.open("a", encoding="utf-8") as handle:
        if existing and not existing.endswith("\n"):
            handle.write("\n")
        handle.write("\n")
        handle.write(marker + "\n\n")
        handle.write(content.rstrip() + "\n")
    return True


def run_summarizer(exp_dir, expected_tasks):
    runs_dir = exp_dir / "runs"
    if not runs_dir.exists():
        return
    cmd = [
        "python",
        "scripts/summarize_incremental_metrics.py",
        str(runs_dir),
        "--output_csv",
        str(exp_dir / "incremental_summary.csv"),
        "--output_markdown",
        str(exp_dir / "incremental_summary.md"),
        "--aggregate_csv",
        str(exp_dir / "incremental_aggregate.csv"),
        "--aggregate_markdown",
        str(exp_dir / "incremental_aggregate.md"),
        "--expected_tasks",
        expected_tasks,
    ]
    subprocess.run(cmd, check=False)


def result_file_exists(run_dir, run_name, suffix):
    candidates = [
        run_dir / f"{run_name}_{suffix}.json",
        run_dir / f"{run_name}_{suffix}_results.json",
    ]
    return any(path.exists() for path in candidates)


def experiment_complete(exp_dir):
    manifest = exp_dir / "manifest.txt"
    if not manifest.exists():
        return False

    missing = []
    for config in CONFIG_NAMES:
        run_name = f"{config}_seed42"
        run_dir = exp_dir / "runs" / run_name
        for suffix in CLASSIFIER_SUFFIXES:
            if not result_file_exists(run_dir, run_name, suffix):
                missing.append(f"{run_name}_{suffix}")
    if missing:
        return False

    return True


def build_section(title, exp_dir):
    aggregate = exp_dir / "incremental_aggregate.md"
    summary = exp_dir / "incremental_summary.md"
    launch = exp_dir / "logs" / "launch.log"
    monitor_paths = [
        exp_dir / "four_task_monitor_retry.log",
        exp_dir / "four_task_monitor_robust.log",
        exp_dir / "four_task_monitor_fixed.log",
    ]

    parts = [
        f"## {title}",
        "",
        f"Recorded at: `{now()}`",
        "",
        "Output directory:",
        "",
        "```text",
        str(exp_dir),
        "```",
        "",
    ]

    if launch.exists():
        launch_lines = read_text(launch).splitlines()[-30:]
        parts.extend([
            "Launcher tail:",
            "",
            "```text",
            *launch_lines,
            "```",
            "",
        ])

    for monitor in monitor_paths:
        if not monitor.exists():
            continue
        monitor_lines = read_text(monitor).splitlines()[-30:]
        parts.extend([
            f"Monitor tail ({monitor.name}):",
            "",
            "```text",
            *monitor_lines,
            "```",
            "",
        ])
        break

    if aggregate.exists():
        parts.extend([
            "Aggregate summary:",
            "",
            read_text(aggregate).rstrip(),
            "",
        ])

    if summary.exists():
        parts.extend([
            "Per-result summary:",
            "",
            read_text(summary).rstrip(),
            "",
        ])

    return "\n".join(parts)


def main():
    args = parse_args()
    stages = [
        {
            "name": "sanity",
            "dir": args.sanity_dir,
            "expected": "aircraft caltech101",
            "marker": "<!-- lada-ablation-sanity-final -->",
            "title": "11. LADA recipe/text-RGDA sanity final summary",
        },
        {
            "name": "four",
            "dir": args.four_dir,
            "expected": "aircraft caltech101 dtd eurosat",
            "marker": "<!-- lada-ablation-four-final -->",
            "title": "12. LADA recipe/text-RGDA four-task final summary",
        },
    ]

    pending = list(stages)
    while pending:
        next_pending = []
        for stage in pending:
            exp_dir = stage["dir"]
            aggregate = exp_dir / "incremental_aggregate.md"
            if not experiment_complete(exp_dir):
                next_pending.append(stage)
                continue
            if not aggregate.exists():
                run_summarizer(exp_dir, stage["expected"])
            if aggregate.exists():
                content = build_section(stage["title"], exp_dir)
                wrote = append_once(args.chat_history, stage["marker"], content)
                print(f"[{now()}] recorded {stage['name']}: wrote={wrote}", flush=True)
            else:
                next_pending.append(stage)
        pending = next_pending
        if pending:
            time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
