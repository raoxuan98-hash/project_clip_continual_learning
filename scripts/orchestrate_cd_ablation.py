#!/usr/bin/env python3
"""
Orchestrate cd_divergence ablation + temperature tuning while avoiding GPU0.
"""
import json
import os
import subprocess
import sys
import time

PROJECT = "/home/raoxuan/projects/project_clip_continual_learning"
PYTHON = "/home/raoxuan/ENTER/envs/raoxuan/bin/python"
RESULT_DIR = os.path.join(PROJECT, "experiments", "optimizer_ablation")
LOG_DIR_DIV = os.path.join(RESULT_DIR, "logs_cd_divergence_ablation")
LOG_DIR_TEMP = os.path.join(RESULT_DIR, "logs_cd_temperature_tuning")
CHAT_DIR = os.path.join(PROJECT, "chat-history")

SEED = 42
DIVS = ["kl_forward", "kl_reverse", "js", "mse", "cosine", "l1"]
RUNNING = ["kl_reverse", "js", "mse", "kl_forward"]
REMAINING = ["kl_forward", "cosine", "l1"]
TEMPS = [1.0, 2.0, 4.0]
GPUS = [2, 3, 4]

COMMON_ARGS = [
    "--root", "/data1/open_datasets/X-TAIL",
    "--dataset_sequence", "aircraft", "caltech101", "dtd", "eurosat", "flowers", "oxford_pets",
    "--num_shots", "16",
    "--batch_size", "64",
    "--iterations", "800",
    "--lr", "1e-4",
    "--weight_decay", "3e-5",
    "--optimizer", "adamw",
    "--lora_type", "lora_nsp",
    "--lora_rank", "4",
    "--use_dora", "false",
    "--train_budget_mode", "uniform",
    "--projection_param_mode", "full",
    "--null_init_mode", "none",
    "--cd_weight", "2.0",
    "--aux_weight", "0.0",
    "--fd_weight", "1.0",
    "--seed", str(SEED),
    "--output_dir", "experiments/optimizer_ablation",
]


def now() -> str:
    return time.strftime("%H:%M:%S")


def exp_name(div: str, temp: float) -> str:
    return f"current_nsp_adamw_cd_divergence_{div}_temp{temp}_seed{SEED}"


def result_files(div: str, temp: float):
    name = exp_name(div, temp)
    return (
        os.path.join(RESULT_DIR, f"{name}_lada_results.json"),
        os.path.join(RESULT_DIR, f"{name}_zs_results.json"),
    )


def compute_transfer_from_zs(path: str) -> float:
    with open(path) as f:
        data = json.load(f)
    M = data["accuracy_matrix"]
    n = len(M)
    vals = [M[i][j] for j in range(1, n) for i in range(j)]
    return sum(vals) / len(vals) * 100 if vals else 0.0


def wait_for_results(items, timeout_hours: float = 6.0):
    deadline = time.time() + timeout_hours * 3600
    while time.time() < deadline:
        missing = []
        for div, temp in items:
            lada, zs = result_files(div, temp)
            if not os.path.exists(lada) or not os.path.exists(zs):
                missing.append(exp_name(div, temp))
        if not missing:
            print(f"[{now()}] All required results ready: {items}")
            return
        print(f"[{now()}] Missing {len(missing)} result files: {missing[:6]}... sleeping 60s")
        time.sleep(60)
    raise RuntimeError(f"Timeout waiting for results: {items}")


def launch_batch(items, log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    procs = []
    for div, temp, gpu in items:
        lada, zs = result_files(div, temp)
        if os.path.exists(lada) and os.path.exists(zs):
            print(f"[{now()}] Skip {exp_name(div, temp)} (results already exist)")
            continue
        name = exp_name(div, temp)
        log = os.path.join(log_dir, f"{name}.log")
        cmd = (
            f"cd {PROJECT} && CUDA_VISIBLE_DEVICES={gpu} nohup {PYTHON} main_incremental.py "
            + " ".join(COMMON_ARGS)
            + f" --cd_divergence {div} --cd_temperature {temp} --gpu 0 --experiment_name {name}"
            + f" > {log} 2>&1 &"
        )
        print(f"[{now()}] Launch {name} on physical GPU {gpu}")
        procs.append(subprocess.Popen(cmd, shell=True))
    for p in procs:
        p.wait()
    print(f"[{now()}] Batch launched ({len(items)} jobs)")


def select_best_divergence():
    scores = {}
    details = {}
    for div in DIVS:
        lada, zs = result_files(div, 2.0)
        transfer = compute_transfer_from_zs(zs)
        with open(lada) as f:
            metrics = json.load(f)["metrics"]
        scores[div] = transfer
        details[div] = {
            "transfer": transfer,
            "last": metrics["last"],
            "average": metrics["average"],
        }
    best = max(scores, key=scores.get)
    print("\nDivergence form comparison (temp=2.0):")
    for div in sorted(scores, key=scores.get, reverse=True):
        d = details[div]
        marker = " <-- best" if div == best else ""
        print(f"  {div:12s} Transfer={d[transfer]:6.2f} Last={d[last]:6.2f} Avg={d[average]:6.2f}{marker}")
    return best, scores, details


def collect_temperature_results(best_div: str):
    details = {}
    for temp in TEMPS:
        lada, zs = result_files(best_div, temp)
        transfer = compute_transfer_from_zs(zs)
        with open(lada) as f:
            metrics = json.load(f)["metrics"]
        details[temp] = {
            "transfer": transfer,
            "last": metrics["last"],
            "average": metrics["average"],
        }
    print(f"\nTemperature comparison for {best_div}:")
    for temp in sorted(details):
        d = details[temp]
        print(f"  temp={temp:<4} Transfer={d[transfer]:6.2f} Last={d[last]:6.2f} Avg={d[average]:6.2f}")
    best_temp = max(details, key=lambda t: details[t]["transfer"])
    return best_temp, details


def write_summary(div_scores, div_details, best_div, temp_details, best_temp):
    os.makedirs(CHAT_DIR, exist_ok=True)
    path = os.path.join(CHAT_DIR, "2026-07-02-cd-divergence-ablation-summary.md")
    lines = []
    lines.append("# CD Divergence Form & Temperature Ablation Summary\n\n")
    lines.append(f"- Seed: {SEED}\n")
    lines.append("- Base config: current_nsp + AdamW + lr=1e-4 + cd_weight=2.0 + aux_weight=0.0 + fd_weight=1.0 + batch_size=64 + iterations=800 + lora_rank=4\n\n")
    lines.append("## 1. Divergence form ablation (cd_temperature = 2.0)\n\n")
    lines.append("| cd_divergence | Transfer | Last | Average |\n")
    lines.append("|---------------|----------|------:|---------:|\n")
    for div in sorted(div_scores, key=div_scores.get, reverse=True):
        d = div_details[div]
        marker = " **best**" if div == best_div else ""
        lines.append(f"| {div}{marker} | {d[transfer]:.2f} | {d[last]:.2f} | {d[average]:.2f} |\n")
    lines.append(f"\n**Recommended divergence form:** `{best_div}` (highest Transfer = {div_scores[best_div]:.2f})\n\n")

    lines.append("## 2. Temperature tuning for best divergence form\n\n")
    lines.append(f"Best divergence form: `{best_div}`\n\n")
    lines.append("| cd_temperature | Transfer | Last | Average |\n")
    lines.append("|----------------|----------|------:|---------:|\n")
    for temp in sorted(temp_details):
        d = temp_details[temp]
        marker = " **best**" if temp == best_temp else ""
        lines.append(f"| {temp}{marker} | {d[transfer]:.2f} | {d[last]:.2f} | {d[average]:.2f} |\n")
    lines.append(f"\n**Recommended temperature:** `{best_temp}` (highest Transfer = {temp_details[best_temp][transfer]:.2f})\n\n")
    lines.append("## 3. Final recommendation\n\n")
    lines.append(f"- `cd_divergence = {best_div}`\n")
    lines.append(f"- `cd_temperature = {best_temp}`\n")
    lines.append(f"- Corresponding metrics: Transfer={temp_details[best_temp][transfer]:.2f}, Last={temp_details[best_temp][last]:.2f}, Average={temp_details[best_temp][average]:.2f}\n")

    with open(path, "w") as f:
        f.writelines(lines)
    print(f"\nSummary written to {path}")


def main():
    os.makedirs(LOG_DIR_DIV, exist_ok=True)
    print("=" * 60)
    print("Waiting for already-running wave: kl_reverse, js, mse")
    wait_for_results([(d, 2.0) for d in RUNNING], timeout_hours=6.0)

    print("=" * 60)
    print("Launching remaining divergence forms on GPUs 2/3/4")
    launch_batch([(div, 2.0, gpu) for div, gpu in zip(REMAINING, GPUS)], LOG_DIR_DIV)

    print("=" * 60)
    print("Waiting for all divergence ablation results")
    wait_for_results([(d, 2.0) for d in DIVS], timeout_hours=6.0)

    best_div, div_scores, div_details = select_best_divergence()

    print("=" * 60)
    print(f"Launching temperature tuning for best divergence: {best_div}")
    launch_batch([(best_div, temp, gpu) for temp, gpu in zip(TEMPS, GPUS)], LOG_DIR_TEMP)

    print("=" * 60)
    print("Waiting for temperature tuning results")
    wait_for_results([(best_div, t) for t in TEMPS], timeout_hours=6.0)

    best_temp, temp_details = collect_temperature_results(best_div)
    write_summary(div_scores, div_details, best_div, temp_details, best_temp)
    print("=" * 60)
    print("All done.")


if __name__ == "__main__":
    main()
