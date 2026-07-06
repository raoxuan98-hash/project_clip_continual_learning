#!/usr/bin/env python3
"""
Orchestrate temperature tuning -> iterations ablation -> scheduler ablation
for cross-modal distillation (kl_forward).
"""
import json
import os
import subprocess
import re
import sys
import time

PROJECT = "/home/raoxuan/projects/project_clip_continual_learning"
PYTHON = "/home/raoxuan/ENTER/envs/raoxuan/bin/python"
RESULT_DIR = os.path.join(PROJECT, "experiments", "optimizer_ablation")
LOG_DIR = os.path.join(RESULT_DIR, "logs_cd_iter_scheduler_ablation")
CHAT_DIR = os.path.join(PROJECT, "chat-history")

SEED = 42
BEST_DIV = "kl_forward"
TEMPS = [1.0, 2.0, 4.0]
ITERS = [400, 800, 1600]
SCHEDULERS = ["cosine", "linear", "cosine_with_warmup", "constant"]

COMMON_ARGS = [
    "--root", "/data1/open_datasets/X-TAIL",
    "--dataset_sequence", "aircraft", "caltech101", "dtd", "eurosat", "flowers", "oxford_pets",
    "--num_shots", "16",
    "--batch_size", "64",
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


def log(msg: str):
    print(f"[{now()}] {msg}", flush=True)


def result_path(name: str, suffix: str) -> str:
    return os.path.join(RESULT_DIR, f"{name}_{suffix}.json")


def load_metrics(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    return data["metrics"]


def wait_for_result_files(checklist: list, timeout_hours: float = 8.0):
    """checklist: list of (exp_name, suffixes)"""
    deadline = time.time() + timeout_hours * 3600
    while time.time() < deadline:
        missing = []
        for name, suffixes in checklist:
            for suffix in suffixes:
                path = result_path(name, suffix)
                if not os.path.exists(path):
                    missing.append(f"{name}_{suffix}")
        if not missing:
            log(f"All required results ready ({len(checklist)} experiments).")
            return
        log(f"Missing {len(missing)} result files: {missing[:8]}... sleeping 60s")
        time.sleep(60)
    raise RuntimeError(f"Timeout waiting for results. Still missing: {missing}")


def launch_experiment(name: str, extra_args: list, gpu: int):
    log_path = os.path.join(LOG_DIR, f"{name}.log")
    cmd = (
        f"cd {PROJECT} && CUDA_VISIBLE_DEVICES={gpu} nohup {PYTHON} main_incremental.py "
        + " ".join(COMMON_ARGS)
        + " " + " ".join(extra_args)
        + f" --gpu 0 --experiment_name {name}"
        + f" > {log_path} 2>&1 &"
    )
    log(f"Launch {name} on physical GPU {gpu}")
    subprocess.Popen(cmd, shell=True).wait()


def gpu_available(gpu_idx: int) -> bool:
    """Simple check: avoid GPU1/GPU5; treat GPU0/2/3/4 as usable if not reserved."""
    if gpu_idx in (1, 5):
        return False
    return True


def collect_temperature_results():
    rows = {}
    for temp in TEMPS:
        name = f"current_nsp_adamw_cd_divergence_{BEST_DIV}_temp{temp}_seed{SEED}"
        rows[temp] = {
            "zs": load_metrics(result_path(name, "zs_results")),
            "ens": load_metrics(result_path(name, "ens_results")),
        }
    return rows


def collect_iterations_results(best_temp: float):
    rows = {}
    for it in ITERS:
        name = f"current_nsp_adamw_cd_divergence_{BEST_DIV}_temp{best_temp}_iter{it}_seed{SEED}"
        rows[it] = {
            "zs": load_metrics(result_path(name, "zs_results")),
            "ens": load_metrics(result_path(name, "ens_results")),
        }
    return rows


def collect_scheduler_results(best_temp: float, best_iter: int):
    rows = {}
    for sched in SCHEDULERS:
        name = f"current_nsp_adamw_cd_divergence_{BEST_DIV}_temp{best_temp}_iter{best_iter}_sched{sched}_seed{SEED}"
        rows[sched] = {
            "zs": load_metrics(result_path(name, "zs_results")),
            "ens": load_metrics(result_path(name, "ens_results")),
        }
    return rows


def write_temperature_summary(rows: dict):
    os.makedirs(CHAT_DIR, exist_ok=True)
    path = os.path.join(CHAT_DIR, "2026-07-02-cd-divergence-ablation-summary.md")
    best_temp = max(rows, key=lambda t: rows[t]["zs"]["transfer"])
    with open(path, "w") as f:
        f.write("# CD Temperature Ablation Summary (kl_forward)\n\n")
        f.write("Base config: current_nsp + AdamW + lr=1e-4 + cd_weight=2.0 + aux_weight=0.0 + fd_weight=1.0 + batch_size=64 + iterations=800 + lora_rank=4 + cd_divergence=kl_forward\n\n")
        f.write("## Zero-shot classifier (LADA metrics)\n\n")
        f.write("| cd_temperature | Transfer | Last | Average |\n")
        f.write("|---------------|----------|------:|---------:|\n")
        for temp in sorted(rows):
            m = rows[temp]["zs"]
            marker = " **best**" if temp == best_temp else ""
            f.write(f"| {temp}{marker} | {m['transfer']:.2f} | {m['last']:.2f} | {m['average']:.2f} |\n")
        f.write(f"\nRecommended temperature (zero-shot Transfer): `{best_temp}`\n\n")

        f.write("## Ensemble classifier (LADA metrics)\n\n")
        f.write("| cd_temperature | Transfer | Last | Average |\n")
        f.write("|---------------|----------|------:|---------:|\n")
        for temp in sorted(rows):
            m = rows[temp]["ens"]
            f.write(f"| {temp} | {m['transfer']:.2f} | {m['last']:.2f} | {m['average']:.2f} |\n")
        f.write(f"\nBest ensemble Transfer at temperature: `{best_temp}`\n")
    log(f"Temperature summary written to {path}")
    return best_temp


def write_final_summary(temp_rows: dict, best_temp: float, iter_rows: dict, best_iter: int, sched_rows: dict, best_sched: str):
    os.makedirs(CHAT_DIR, exist_ok=True)
    path = os.path.join(CHAT_DIR, "2026-07-02-iterations-scheduler-ablation-summary.md")
    with open(path, "w") as f:
        f.write("# Iterations & Scheduler Ablation Summary\n\n")
        f.write(f"Base config: current_nsp + AdamW + lr=1e-4 + cd_weight=2.0 + aux_weight=0.0 + fd_weight=1.0 + batch_size=64 + lora_rank=4 + cd_divergence=kl_forward + cd_temperature={best_temp}\n\n")

        f.write("## 1. Temperature selection (reference)\n\n")
        f.write("| cd_temperature | ZS Transfer | ZS Last | ZS Average | Ens Transfer | Ens Last | Ens Average |\n")
        f.write("|---------------|-------------|--------:|-----------:|--------------|---------:|------------:|\n")
        for temp in sorted(temp_rows):
            zs = temp_rows[temp]["zs"]
            ens = temp_rows[temp]["ens"]
            f.write(f"| {temp} | {zs['transfer']:.2f} | {zs['last']:.2f} | {zs['average']:.2f} | {ens['transfer']:.2f} | {ens['last']:.2f} | {ens['average']:.2f} |\n")
        f.write(f"\nSelected temperature: `{best_temp}`\n\n")

        f.write("## 2. Iterations ablation\n\n")
        f.write("| iterations | ZS Transfer | ZS Last | ZS Average | Ens Transfer | Ens Last | Ens Average |\n")
        f.write("|-----------|-------------|--------:|-----------:|--------------|---------:|------------:|\n")
        for it in sorted(iter_rows):
            zs = iter_rows[it]["zs"]
            ens = iter_rows[it]["ens"]
            marker = " **best**" if it == best_iter else ""
            f.write(f"| {it}{marker} | {zs['transfer']:.2f} | {zs['last']:.2f} | {zs['average']:.2f} | {ens['transfer']:.2f} | {ens['last']:.2f} | {ens['average']:.2f} |\n")
        f.write(f"\nSelected iterations: `{best_iter}` (highest zero-shot Transfer)\n\n")

        f.write("## 3. Scheduler ablation\n\n")
        f.write("| scheduler | ZS Transfer | ZS Last | ZS Average | Ens Transfer | Ens Last | Ens Average |\n")
        f.write("|----------|-------------|--------:|-----------:|--------------|---------:|------------:|\n")
        for sched in ["cosine", "linear", "cosine_with_warmup", "constant"]:
            zs = sched_rows[sched]["zs"]
            ens = sched_rows[sched]["ens"]
            marker = " **best**" if sched == best_sched else ""
            f.write(f"| {sched}{marker} | {zs['transfer']:.2f} | {zs['last']:.2f} | {zs['average']:.2f} | {ens['transfer']:.2f} | {ens['last']:.2f} | {ens['average']:.2f} |\n")
        f.write(f"\nSelected scheduler: `{best_sched}` (highest zero-shot Transfer)\n\n")

        f.write("## 4. Final recommendation\n\n")
        f.write(f"- cd_divergence = `{BEST_DIV}`\n")
        f.write(f"- cd_temperature = `{best_temp}`\n")
        f.write(f"- iterations = `{best_iter}`\n")
        f.write(f"- scheduler = `{best_sched}`\n")
        best_metrics = sched_rows[best_sched]["zs"]
        f.write(f"- Zero-shot metrics: Transfer={best_metrics['transfer']:.2f}, Last={best_metrics['last']:.2f}, Average={best_metrics['average']:.2f}\n")
    log(f"Final summary written to {path}")



def get_free_gpus(min_count: int = 3, mem_thresh_mib: int = 500, util_thresh: int = 10):
    """Return indices of GPUs (excluding 1 and 5) with low memory/utilization."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            text=True,
        )
    except Exception as e:
        log(f"nvidia-smi failed: {e}")
        return []
    free = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        idx = int(parts[0])
        if idx in (1, 5):
            continue
        mem_used = int(parts[2])
        util = int(parts[4])
        if mem_used < mem_thresh_mib and util < util_thresh:
            free.append(idx)
    return free

def main():
    os.makedirs(LOG_DIR, exist_ok=True)

    # === Phase 1: wait for temperature tuning ===
    log("=" * 60)
    log("Phase 1: waiting for temperature tuning (1.0, 2.0, 4.0)")
    temp_checklist = []
    for temp in TEMPS:
        name = f"current_nsp_adamw_cd_divergence_{BEST_DIV}_temp{temp}_seed{SEED}"
        temp_checklist.append((name, ["zs_results", "ens_results"]))
    wait_for_result_files(temp_checklist, timeout_hours=8.0)

    temp_rows = collect_temperature_results()
    best_temp = write_temperature_summary(temp_rows)
    log(f"Best temperature selected: {best_temp}")

    # === Phase 2: iterations ablation ===
    log("=" * 60)
    log(f"Phase 2: launching iterations ablation (best_temp={best_temp})")
    # Use first 3 available GPUs among {0,2,3,4}, avoiding GPU1/GPU5
    available_gpus = get_free_gpus(min_count=3)
    if len(available_gpus) < 3:
        log(f"Only {len(available_gpus)} GPUs free, waiting for at least 3...")
        deadline = time.time() + 60 * 60  # wait up to 1 hour
        while len(available_gpus) < 3 and time.time() < deadline:
            time.sleep(60)
            available_gpus = get_free_gpus(min_count=3)
        if len(available_gpus) < 3:
            raise RuntimeError("Could not find 3 free GPUs for iterations ablation")
    iter_gpu_map = {400: available_gpus[0], 800: available_gpus[1], 1600: available_gpus[2]}
    for it, gpu in iter_gpu_map.items():
        name = f"current_nsp_adamw_cd_divergence_{BEST_DIV}_temp{best_temp}_iter{it}_seed{SEED}"
        if (os.path.exists(result_path(name, "zs_results")) and
            os.path.exists(result_path(name, "ens_results"))):
            log(f"Skip existing {name}")
            continue
        extra = [f"--iterations {it}", f"--cd_temperature {best_temp}", "--cd_divergence", BEST_DIV]
        launch_experiment(name, extra, gpu)
    time.sleep(5)

    iter_checklist = []
    for it in ITERS:
        name = f"current_nsp_adamw_cd_divergence_{BEST_DIV}_temp{best_temp}_iter{it}_seed{SEED}"
        iter_checklist.append((name, ["zs_results", "ens_results"]))
    wait_for_result_files(iter_checklist, timeout_hours=12.0)

    iter_rows = collect_iterations_results(best_temp)
    best_iter = max(iter_rows, key=lambda it: iter_rows[it]["zs"]["transfer"])
    log(f"Best iterations selected: {best_iter}")

    # === Phase 3: scheduler ablation ===
    log("=" * 60)
    log(f"Phase 3: launching scheduler ablation on GPUs 2/3/4 (best_temp={best_temp}, best_iter={best_iter})")
    sched_gpu_map = {"cosine": 2, "linear": 3, "cosine_with_warmup": 4, "constant": 2}
    # Launch first batch on 2/3/4, then constant after one finishes (simple sequential fallback)
    sched_order = ["cosine", "linear", "cosine_with_warmup", "constant"]
    gpu_pool = [2, 3, 4]
    running = []
    for sched in sched_order:
        name = f"current_nsp_adamw_cd_divergence_{BEST_DIV}_temp{best_temp}_iter{best_iter}_sched{sched}_seed{SEED}"
        if (os.path.exists(result_path(name, "zs_results")) and
            os.path.exists(result_path(name, "ens_results"))):
            log(f"Skip existing {name}")
            continue
        extra = [f"--iterations {best_iter}", f"--cd_temperature {best_temp}", "--cd_divergence", BEST_DIV, f"--scheduler {sched}"]
        # Round-robin on available GPUs
        gpu = gpu_pool[len(running) % len(gpu_pool)]
        launch_experiment(name, extra, gpu)
        running.append(sched)
    time.sleep(5)

    sched_checklist = []
    for sched in SCHEDULERS:
        name = f"current_nsp_adamw_cd_divergence_{BEST_DIV}_temp{best_temp}_iter{best_iter}_sched{sched}_seed{SEED}"
        sched_checklist.append((name, ["zs_results", "ens_results"]))
    wait_for_result_files(sched_checklist, timeout_hours=12.0)

    sched_rows = collect_scheduler_results(best_temp, best_iter)
    best_sched = max(sched_rows, key=lambda s: sched_rows[s]["zs"]["transfer"])
    log(f"Best scheduler selected: {best_sched}")

    # === Phase 4: final summary ===
    log("=" * 60)
    log("Phase 4: writing final summary")
    write_final_summary(temp_rows, best_temp, iter_rows, best_iter, sched_rows, best_sched)
    log("All done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"ERROR: {e}")
        sys.exit(1)
