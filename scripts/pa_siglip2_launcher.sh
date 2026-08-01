#!/bin/bash
# pa_siglip2_launcher.sh — PA 重跑计划 W5 SigLIP2 接力器
# 与 pa_wave_launcher.sh 共用 PA_main 输出与全局 ≤2 进程约束（our_running 统计所有 PA__ 进程）。
# 用法: bash scripts/pa_siglip2_launcher.sh
set -u
cd /home/raoxuan/projects/project_clip_continual_learning

export CLIP_LOCAL_FILES_ONLY=1
export CLIP_MODEL_NAME=/mnt/raoxuan/models/siglip2-base-patch16-224
PY=/home/raoxuan/ENTER/envs/raoxuan/bin/python
OUT=experiments/paper_formal/PA_main
LOGDIR=artifacts/launch_logs
MAX_RUNS=2
mkdir -p "$OUT" "$LOGDIR"

BASE="--root /data1/open_datasets/X-TAIL --dataset_sequence aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397 --num_shots 16 --batch_size 32 --eval_batch_size 128 --eval_resize_mode preserve_aspect --optimizer adamw --lr 1e-4 --weight_decay 3e-5 --scheduler cosine_with_warmup --warmup_ratio 0.1 --eta_min 0.0 --lora_rank 4 --lora_alpha 4 --lora_dropout 0.0 --lora_target_modules q_proj,k_proj,v_proj,out_proj,fc1,fc2 --projection_param_mode full --nsp_eps 0.20 --nsp_weight 0.02 --reference_dataset flickr30k_train_sub8k --reference_batch_size 32 --fd_weight 0.0 --cd_weight 2.0 --cd_divergence kl_forward --cd_temperature 4.0 --aux_weight 0.0 --tune_vision_encoder true --tune_text_encoder true --text_lora_rank 4 --text_tuning_schedule always --text_classifier_mode lada_hybrid --classifier_feature_transform test --rgda_rank 32 --rgda_alpha1 0.2 --rgda_alpha2 2.0 --rgda_alpha3 0.5 --num_centers 4 --rgda_train_iter 200 --rgda_train_lr 0.01 --rgda_fit_source gmm_sample --alpha 0.05 --ensemble_normalize maxshift --temperature 1.0 --enable_retrieval_eval --retrieval_datasets mscoco_2014_5k,flickr30k_hf --retrieval_roots mscoco_2014_5k=/mnt/raoxuan/open_datasets/mscoco_2014_5k_test_hf,flickr30k_hf=/mnt/raoxuan/open_datasets/flickr30k_hf --retrieval_batch_size 128 --retrieval_recall_ks 1,5,10 --retrieval_max_images 0 --no-alpha_sensitivity --use_dora false --iterations 600 --output_dir $OUT"

# 队列：name|extra_args。loranf 两条显式 --disable_lada（LADA 默认开启）；lada 两条不加。
QUEUE=(
  "PA__siglip2_loranf_i600__16shot__seed43|--lora_type lora_nsp --init_mode lora_nsp --null_init_mode none --disable_lada --seed 43"
  "PA__siglip2_loranf_i600__16shot__seed42|--lora_type lora_nsp --init_mode lora_nsp --null_init_mode none --disable_lada --seed 42"
  "PA__siglip2_lada_i600__16shot__seed43|--lora_type lora_nsp --init_mode lora_nsp --null_init_mode none --seed 43"
  "PA__siglip2_lada_i600__16shot__seed42|--lora_type lora_nsp --init_mode lora_nsp --null_init_mode none --seed 42"
)

# 只统计顶层主进程：DataLoader workers 继承父进程 cmdline，会被 pgrep 误匹配。
# 通过 PPid 过滤，只保留 ppid 不在匹配进程列表中的顶层进程。
our_running() {
  pgrep -af "experiment_name PA__" | grep -v 'pgrep -af' | awk '{print $1}' | while read pid; do
    ppid=$(awk '/PPid:/{print $2}' /proc/$pid/status 2>/dev/null)
    if ! pgrep -af "experiment_name PA__" | grep -v 'pgrep -af' | awk '{print $1}' | grep -q "^${ppid}$"; then
      echo $pid
    fi
  done | wc -l
}

is_running() {
  local name="$1"
  pgrep -af "experiment_name ${name}" | grep -v 'pgrep -af' | awk '{print $1}' | while read pid; do
    ppid=$(awk '/PPid:/{print $2}' /proc/$pid/status 2>/dev/null)
    if ! pgrep -af "experiment_name ${name}" | grep -v 'pgrep -af' | awk '{print $1}' | grep -q "^${ppid}$"; then
      echo $pid
    fi
  done | grep -q .
}
free_gpu() {
  local excl="${1:-}"
  for g in 0 1 2 3 4 5; do
    [ "$g" = "$excl" ] && continue
    if ! nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $g 2>/dev/null | grep -q .; then
      echo $g; return 0
    fi
  done
  return 1
}

n=$(our_running)
echo "[siglip2-launcher] our PA__ runs: $n / $MAX_RUNS"
[ "$n" -ge "$MAX_RUNS" ] && exit 0

last_gpu=""
for item in "${QUEUE[@]}"; do
  name="${item%%|*}"
  extra="${item#*|}"
  if [ -f "$OUT/${name}_ens_results.json" ]; then continue; fi
  if is_running "$name"; then continue; fi
  n=$(our_running)
  [ "$n" -ge "$MAX_RUNS" ] && break
  gpu=$(free_gpu "$last_gpu") || { echo "[siglip2-launcher] no free gpu"; break; }
  last_gpu=$gpu
  echo "[siglip2-launcher] start $name on gpu $gpu ($(date '+%F %T'))"
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY -u main_incremental.py $BASE $extra \
    --experiment_name "$name" > "$LOGDIR/${name}.log" 2>&1 &
  if wait_for_ps "$name"; then
    echo "[siglip2-launcher] $name confirmed in ps"
  else
    echo "[siglip2-launcher] WARNING $name not visible after 30s"
  fi
  sleep 5
done
echo "[siglip2-launcher] done ($(date '+%F %T'))"
