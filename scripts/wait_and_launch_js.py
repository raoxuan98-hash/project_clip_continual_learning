import os, time, subprocess, sys

RESULT_DIR = "experiments/optimizer_ablation"
LOG_DIR = os.path.join(RESULT_DIR, "logs_cd_divergence_ablation")
PYTHON = "/home/raoxuan/ENTER/envs/raoxuan/bin/python"
COMMON_ARGS = [
    "main_incremental.py",
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
    "--seed", "42",
    "--output_dir", RESULT_DIR,
    "--num_workers", "4",
    "--gpu", "0",
    "--cd_divergence", "js",
    "--cd_temperature", "2.0",
    "--experiment_name", "current_nsp_adamw_cd_divergence_js_temp2.0_seed42",
]

# Map from the divergences currently running to the GPU they occupy.
DIV_TO_GPU = {
    "kl_forward": "2",
    "cosine": "3",
    "l1": "4",
}

def result_file(div):
    return os.path.join(RESULT_DIR, f"current_nsp_adamw_cd_divergence_{div}_temp2.0_seed42_zs_results.json")

def log_file():
    return os.path.join(LOG_DIR, "current_nsp_adamw_cd_divergence_js_temp2.0_seed42.log")

def launch_on_gpu(gpu):
    log_path = log_file()
    os.makedirs(LOG_DIR, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["OMP_NUM_THREADS"] = "4"
    env["MKL_NUM_THREADS"] = "4"
    env["NUMEXPR_NUM_THREADS"] = "4"
    with open(log_path, "w") as f:
        subprocess.Popen(
            [PYTHON] + COMMON_ARGS,
            stdout=f, stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
    print(f"Launched js on GPU {gpu}; log {log_path}", flush=True)

def main():
    print("Waiting for kl_forward/cosine/l1 to finish and free a GPU...", flush=True)
    for _ in range(120):  # poll for up to 60 minutes
        for div, gpu in DIV_TO_GPU.items():
            if os.path.exists(result_file(div)):
                print(f"{div} result found; launching js on GPU {gpu}", flush=True)
                launch_on_gpu(gpu)
                return
        time.sleep(30)
    print("Timeout: no GPU freed within 60 minutes", file=sys.stderr, flush=True)
    sys.exit(1)

if __name__ == "__main__":
    main()
