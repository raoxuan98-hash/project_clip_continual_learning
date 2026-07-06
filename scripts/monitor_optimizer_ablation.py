#!/usr/bin/env python3
import os, sys, json, time, subprocess, datetime

PROJECT_DIR = "/home/raoxuan/projects/project_clip_continual_learning"
OUT_DIR = os.path.join(PROJECT_DIR, "experiments", "optimizer_ablation")
STATUS_FILE = os.path.join(OUT_DIR, "monitor_status.json")
SUMMARY_JSON = os.path.join(OUT_DIR, "summary_optimizer_ablation.json")
SUMMARY_MD = os.path.join(OUT_DIR, "summary_optimizer_ablation.md")

VARIANTS = ["current_nsp", "hist_null_init_runtime"]
OPTIMIZERS = ["adamw", "adam", "sgd", "adagrad", "rmsprop"]


def result_exists(variant, opt, suffix):
    name = f"{variant}_{opt}_seed42_{suffix}_results.json"
    return os.path.exists(os.path.join(OUT_DIR, name))


def parse_last_acc(variant, opt, suffix):
    path = os.path.join(OUT_DIR, f"{variant}_{opt}_seed42_{suffix}_results.json")
    try:
        with open(path) as f:
            d = json.load(f)
        return float(d["metrics"]["last"])
    except Exception:
        return None


def count_results():
    total = 0
    detail = []
    for v in VARIANTS:
        for o in OPTIMIZERS:
            ens = result_exists(v, o, "ens")
            zs = result_exists(v, o, "zs")
            detail.append({"variant": v, "optimizer": o, "ens": ens, "zs": zs})
            total += ens + zs
    return total, detail


def pgrep_count(pattern):
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", pattern], stderr=subprocess.DEVNULL, text=True, timeout=10
        )
        return len([x for x in out.splitlines() if x.strip()])
    except subprocess.TimeoutExpired:
        return -1
    except subprocess.CalledProcessError:
        return 0


def write_summary():
    rows = []
    for v in VARIANTS:
        for o in OPTIMIZERS:
            zs = parse_last_acc(v, o, "zs")
            ens = parse_last_acc(v, o, "ens")
            rows.append({"variant": v, "optimizer": o, "zero_shot_last": zs, "ensemble_last": ens})
    best_zs = {}
    best_ens = {}
    for v in VARIANTS:
        vrows = [r for r in rows if r["variant"] == v]
        best_zs[v] = max(
            vrows,
            key=lambda r: r["zero_shot_last"] if r["zero_shot_last"] is not None else -1.0,
        )
        best_ens[v] = max(
            vrows,
            key=lambda r: r["ensemble_last"] if r["ensemble_last"] is not None else -1.0,
        )
    summary = {"rows": rows, "best_zero_shot": best_zs, "best_ensemble": best_ens}
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    lines = [
        "# Optimizer Ablation Summary (LoRA)",
        "",
        "| Variant | Optimizer | Zero-shot Last | Ensemble Last |",
        "|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['variant']} | {r['optimizer']} | "
            f"{r['zero_shot_last']:.4f} | {r['ensemble_last']:.4f} |"
        )
    lines.append("")
    for v in VARIANTS:
        lines.append(
            f"- **{v}** best zero-shot: `{best_zs[v]['optimizer']}` "
            f"({best_zs[v]['zero_shot_last']:.4f})"
        )
        lines.append(
            f"- **{v}** best ensemble: `{best_ens[v]['optimizer']}` "
            f"({best_ens[v]['ensemble_last']:.4f})"
        )
    with open(SUMMARY_MD, "w") as f:
        f.write("\n".join(lines))
    return summary


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    max_wait = 60 * 60 * 12  # 12 hours
    poll_interval = 30
    start = time.time()
    print(f"[{datetime.datetime.now().isoformat()}] Monitor started")
    sys.stdout.flush()
    while True:
        elapsed = time.time() - start
        total, detail = count_results()
        running_main = pgrep_count("main_incremental.py")
        running_pipes = pgrep_count("run_optimizer_ablation_pipeline")
        status = {
            "timestamp": datetime.datetime.now().isoformat(),
            "elapsed_seconds": elapsed,
            "result_count": total,
            "expected": len(VARIANTS) * len(OPTIMIZERS) * 2,
            "running_main_incremental": running_main,
            "running_pipelines": running_pipes,
            "details": detail,
        }
        with open(STATUS_FILE, "w") as f:
            json.dump(status, f, indent=2)
        print(
            f"[{status['timestamp']}] results={total}/{status['expected']} "
            f"main={running_main} pipes={running_pipes}"
        )
        sys.stdout.flush()

        if total == status["expected"] and running_main == 0:
            summary = write_summary()
            status["summary_written"] = True
            status["summary"] = summary
            with open(STATUS_FILE, "w") as f:
                json.dump(status, f, indent=2)
            print("All results collected. Summary written.")
            return 0

        if running_main == 0 and running_pipes == 0 and total < status["expected"]:
            status["error"] = "All pipelines/main processes finished but results incomplete."
            status["partial_summary"] = write_summary()
            with open(STATUS_FILE, "w") as f:
                json.dump(status, f, indent=2)
            print(status["error"], file=sys.stderr)
            return 1

        if elapsed > max_wait:
            status["error"] = "Timeout waiting for results."
            with open(STATUS_FILE, "w") as f:
                json.dump(status, f, indent=2)
            print(status["error"], file=sys.stderr)
            return 2

        time.sleep(poll_interval)


if __name__ == "__main__":
    sys.exit(main())
