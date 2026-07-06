#!/usr/bin/env python3
"""
在 cd_divergence 消融结束后，读取零样本结果矩阵计算 Transfer，
选出最佳 cd_divergence，然后对温度 {1.0, 2.0, 4.0} 做并行调优。
"""
import json
import os
import subprocess
import sys
import time

PROJECT = "/home/raoxuan/projects/project_clip_continual_learning"
PYTHON = "/home/raoxuan/ENTER/envs/raoxuan/bin/python"
RESULT_DIR = os.path.join(PROJECT, "experiments", "optimizer_ablation")
LOG_DIR = os.path.join(RESULT_DIR, "logs_cd_temperature_tuning")
DIVS = ["kl_forward", "kl_reverse", "js", "mse", "cosine", "l1"]
TEMPS = [1.0, 2.0, 4.0]
GPUS = [0, 2, 3]
SEED = 42

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


def compute_transfer_from_zs(path: str) -> float:
    with open(path) as f:
        data = json.load(f)
    M = data["accuracy_matrix"]
    n = len(M)
    vals = [M[i][j] for j in range(1, n) for i in range(j)]
    return sum(vals) / len(vals) * 100 if vals else 0.0


def wait_for_divergence_results(timeout_hours: float = 6.0):
    print("Waiting for cd_divergence ablation results...")
    deadline = time.time() + timeout_hours * 3600
    while time.time() < deadline:
        missing = []
        for div in DIVS:
            path = os.path.join(RESULT_DIR, f"current_nsp_adamw_cd_divergence_{div}_temp2.0_seed{SEED}_zs_results.json")
            if not os.path.exists(path):
                missing.append(div)
        if not missing:
            print("All divergence results are ready.")
            return
        print(f"Missing: {missing}; sleeping 60s")
        time.sleep(60)
    raise RuntimeError("Timeout waiting for divergence ablation results")


def select_best_divergence():
    scores = {}
    for div in DIVS:
        path = os.path.join(RESULT_DIR, f"current_nsp_adamw_cd_divergence_{div}_temp2.0_seed{SEED}_zs_results.json")
        scores[div] = compute_transfer_from_zs(path)
    best = max(scores, key=scores.get)
    print("Divergence Transfer scores:")
    for div, score in sorted(scores.items(), key=lambda x: -x[1]):
        marker = " <-- best" if div == best else ""
        print(f"  {div:12s}: {score:.2f}{marker}")
    return best, scores


def launch_temperature_tuning(best_div: str):
    os.makedirs(LOG_DIR, exist_ok=True)
    procs = []
    for temp, gpu in zip(TEMPS, GPUS):
        name = f"current_nsp_adamw_cd_divergence_{best_div}_temp{temp}_seed{SEED}"
        log = os.path.join(LOG_DIR, f"{name}.log")
        cmd = (
            f"cd {PROJECT} && CUDA_VISIBLE_DEVICES={gpu} nohup {PYTHON} main_incremental.py "
            + " ".join(COMMON_ARGS)
            + f" --cd_divergence {best_div} --cd_temperature {temp} --gpu 0 --experiment_name {name}"
            + f" > {log} 2>&1 &"
        )
        print(f"[Launch] {name} on GPU {gpu}")
        procs.append(subprocess.Popen(cmd, shell=True))
    for p in procs:
        p.wait()
    print("Temperature tuning jobs launched")


def main():
    wait_for_divergence_results()
    best_div, scores = select_best_divergence()
    launch_temperature_tuning(best_div)
    # Save selection for later summarization
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(os.path.join(LOG_DIR, "best_divergence.json"), "w") as f:
        json.dump({"best_divergence": best_div, "scores": scores}, f, indent=2)


if __name__ == "__main__":
    main()
