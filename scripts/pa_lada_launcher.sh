#!/bin/bash
# pa_lada_launcher.sh — PA 重跑计划：常规 CLIP + LADA 分类器 16-shot 复现
# 与主接力器共用 PA_main 输出与全局 ≤2 进程约束。
# LADA 运行不传递 --disable_lada，因此启用 LADA 分类器评估。
set -u
cd /home/raoxuan/projects/project_clip_continual_learning

export CLIP_LOCAL_FILES_ONLY=1
export CLIP_MODEL_NAME=/mnt/raoxuan/models/clip-vit-base-patch16
PY=/home/raoxuan/ENTER/envs/raoxuan/bin/python
OUT=experiments/paper_formal/PA_main
LOGDIR=artifacts/launch_logs
MAX_RUNS=2
mkdir -p "$OUT" "$LOGDIR"

BASE="--root /data1/open_datasets/X-TAIL --dataset_sequence aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397 --num_shots 16 --batch_size 32 --eval_batch_size 128 --eval_resize_mode preserve_aspect --optimizer adamw --lr 1e-4 --weight_decay 3e-5 --scheduler cosine_with_warmup --warmup_ratio 0.1 --eta_min 0.0 --lora_rank 4 --lora_alpha 4 --lora_dropout 0.0 --lora_target_modules q_proj,k_proj,v_proj,out_proj,fc1,fc2 --projection_param_mode full --nsp_eps 0.20 --nsp_weight 0.02 --reference_dataset flickr30k_train_sub8k --reference_batch_size 32 --fd_weight 0.0 --cd_weight 2.0 --cd_divergence kl_forward --cd_temperature 4.0 --aux_weight 0.0 --tune_vision_encoder true --tune_text_encoder true --text_lora_rank 4 --text_tuning_schedule always --text_classifier_mode lada_hybrid --classifier_feature_transform test --rgda_rank 32 --rgda_alpha1 0.2 --rgda_alpha2 2.0 --rgda_alpha3 0.5 --num_centers 4 --rgda_train_iter 200 --rgda_train_lr 0.01 --rgda_fit_source gmm_sample --alpha 0.05 --ensemble_normalize maxshift --temperature 1.0 --enable_retrieval_eval --retrieval_datasets mscoco_2014_5k,flickr30k_hf --retrieval_roots mscoco_2014_5k=/mnt/raoxuan/open_datasets/mscoco_2014_5k_test_hf,flickr30k_hf=/mnt/raoxuan/open_datasets/flickr30k_hf --retrieval_batch_size 128 --retrieval_recall_ks 1,5,10 --retrieval_max_images 0 --no-alpha_sensitivity --use_dora false --iterations 600 --output_dir $OUT"

QUEUE=(
  "PA__lada_i600__16shot__seed43|--lora_type lora_nsp --init_mode lora_nsp --null_init_mode none --seed 43"
  "PA__lada_i600__16shot__seed42|--lora_type lora_nsp --init_mode lora_nsp --null_init_mode none --seed 42"
)

our_running() { pgrep -af "main_incremental.py.*PA__" | grep -o 'PA__[A-Za-z0-9_]*' | sort -u | wc -l; }

free_gpu() {
  local used
  used=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
  for g in 0 1 2 3 4 5; do
    if ! nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $g 2>/dev/null | grep -q .; then
      echo $g; return 0
    fi
  done
  return 1
}

n=$(our_running)
echo "[lada-launcher] our PA__ runs: $n / $MAX_RUNS"
[ "$n" -ge "$MAX_RUNS" ] && exit 0

for item in "${QUEUE[@]}"; do
  name="${item%%|*}"
  extra="${item#*|}"
  if [ -f "$OUT/${name}_ens_results.json" ]; then continue; fi
  if pgrep -f "main_incremental.py.*${name}" > /dev/null; then continue; fi
  n=$(our_running)
  [ "$n" -ge "$MAX_RUNS" ] && break
  gpu=$(free_gpu) || { echo "[lada-launcher] no free gpu"; break; }
  echo "[lada-launcher] start $name on gpu $gpu ($(date '+%F %T'))"
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY -u main_incremental.py $BASE $extra \
    --experiment_name "$name" > "$LOGDIR/${name}.log" 2>&1 &
  sleep 5
done
echo "[lada-launcher] done ($(date '+%F %T'))"
