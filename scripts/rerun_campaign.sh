#!/bin/bash
# rerun_campaign.sh — 修复后正式重跑战役启动器（main_v3）
#
# 用法:
#   bash scripts/rerun_campaign.sh smoke <gpu>        # Wave 0 smoke（2 task, 200 iter）
#   bash scripts/rerun_campaign.sh waveA <gpu>        # Wave A: E1-16shot + E3（每 GPU 的份额串行）
#   bash scripts/rerun_campaign.sh waveB <gpu>
#   bash scripts/rerun_campaign.sh waveC <gpu>
#   bash scripts/rerun_campaign.sh waveD <gpu>
#   bash scripts/rerun_campaign.sh waveE <gpu>
#
# 设计: 冻结基座配置（fd=0, cd=2 双向, 见计划文档 §0），每个 run 显式传全部关键参数；
# 每 GPU 串行；已有 _ens_results.json 的 run 自动跳过（断点续跑）。
set -u
cd "$(dirname "$0")/.."

# 模型权重全部走本地 HF 缓存，避免多天无人值守运行期间的 hub 网络抖动
export CLIP_LOCAL_FILES_ONLY=1

PY=/home/raoxuan/ENTER/envs/raoxuan/bin/python
RETR_ROOTS="mscoco_2014_5k=/mnt/raoxuan/open_datasets/mscoco_2014_5k_test_hf,flickr30k_hf=/mnt/raoxuan/open_datasets/flickr30k_hf"

BASE_ARGS="--root /data1/open_datasets/X-TAIL \
--dataset_sequence aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397 \
--num_shots 16 --batch_size 32 --eval_batch_size 128 --iterations 800 --train_budget_mode uniform \
--optimizer adamw --lr 1e-4 --weight_decay 3e-5 --scheduler cosine_with_warmup --warmup_ratio 0.1 --eta_min 0.0 \
--lora_rank 4 --lora_alpha 4 --lora_dropout 0.0 \
--lora_target_modules q_proj,k_proj,v_proj,out_proj,fc1,fc2 \
--projection_param_mode full --nsp_eps 0.20 --nsp_weight 0.02 \
--reference_dataset flickr8k --reference_batch_size 32 \
--fd_weight 0.0 --cd_weight 2.0 --cd_divergence kl_forward --cd_temperature 4.0 --aux_weight 0.0 \
--tune_vision_encoder true --tune_text_encoder true --text_lora_rank 4 \
--text_tuning_schedule always --text_classifier_mode lada_hybrid \
--classifier_feature_transform test \
--rgda_rank 32 --rgda_alpha1 0.2 --rgda_alpha2 2.0 --rgda_alpha3 0.5 \
--num_centers 4 --rgda_train_iter 200 --rgda_train_lr 0.01 --rgda_fit_source gmm_sample \
--alpha 0.05 --ensemble_normalize maxshift --temperature 1.0 \
--enable_retrieval_eval --retrieval_datasets mscoco_2014_5k,flickr30k_hf \
--retrieval_roots $RETR_ROOTS --retrieval_batch_size 128 --retrieval_recall_ks 1,5,10 --retrieval_max_images 0 \
--disable_lada --no-alpha_sensitivity"

LORA_NF="--lora_type lora_nsp --init_mode lora_nsp --use_dora false --null_init_mode none"

run_one() {
  # run_one <gpu> <subdir> <name> [overrides...]
  local gpu=$1 subdir=$2 name=$3; shift 3
  local outdir=experiments/paper_formal/$subdir
  local log=$outdir/${name}.log
  mkdir -p "$outdir"
  if [ -f "$outdir/${name}_ens_results.json" ]; then
    echo "[skip] $name (ens results exist)"
    return 0
  fi
  echo "[run ] gpu=$gpu $name  ($(date '+%F %T'))"
  CUDA_VISIBLE_DEVICES=$gpu $PY -u main_incremental.py \
    $BASE_ARGS --output_dir "$outdir" --experiment_name "$name" "$@" \
    > "$log" 2>&1
  local rc=$?
  echo "[done] gpu=$gpu $name rc=$rc ($(date '+%F %T'))"
  return $rc
}

WAVE=${1:?usage: rerun_campaign.sh <wave> <gpu>}
GPU=${2:?usage: rerun_campaign.sh <wave> <gpu>}

case "$WAVE" in
smoke)
  SMOKE="--dataset_sequence aircraft caltech101 --iterations 200"
  run_one $GPU Wave0_smoke smoke__lora_nf__2task__seed43 $SMOKE $LORA_NF --seed 43
  run_one $GPU Wave0_smoke smoke__lora_null__2task__seed43 $SMOKE $LORA_NF --null_init_mode history_init_only --seed 43
  ;;

waveA)
  for seed in 42 43 44; do
    run_one $GPU WaveA_main waveA__lora_nf__16shot__seed${seed} $LORA_NF --seed $seed
  done
  for seed in 42 43 44; do
    run_one $GPU WaveA_main waveA__lora__16shot__seed${seed} \
      --lora_type lora_vanilla --init_mode lora_vanilla --use_dora false --null_init_mode none --seed $seed
  done
  for seed in 42 43 44; do
    run_one $GPU WaveA_main waveA__lora_null__16shot__seed${seed} $LORA_NF \
      --null_init_mode history_init_only --seed $seed
  done
  for seed in 42 43 44; do
    run_one $GPU WaveA_main waveA__gradproj__16shot__seed${seed} $LORA_NF \
      --use_gradient_projection --seed $seed
  done
  ;;

waveB)
  for seed in 42 43 44; do
    run_one $GPU WaveB_components waveB__C0_fd0_cd0__16shot__seed${seed} $LORA_NF \
      --fd_weight 0.0 --cd_weight 0.0 --seed $seed
    run_one $GPU WaveB_components waveB__C1_fd1_cd0__16shot__seed${seed} $LORA_NF \
      --fd_weight 1.0 --cd_weight 0.0 --seed $seed
    run_one $GPU WaveB_components waveB__C3_fd1_cd2__16shot__seed${seed} $LORA_NF \
      --fd_weight 1.0 --cd_weight 2.0 --seed $seed
  done
  ;;

waveC)
  for seed in 42 43 44; do
    run_one $GPU WaveC_fullshot waveC__lora_nf__fs__seed${seed} $LORA_NF --full_shot --seed $seed
    run_one $GPU WaveC_fullshot waveC__lora__fs__seed${seed} \
      --lora_type lora_vanilla --init_mode lora_vanilla --use_dora false --null_init_mode none \
      --full_shot --seed $seed
  done
  ;;

waveD)
  # seed 43 先行；默认点（cd2/t4/eps0.20/w0.02/all）= WaveA seed43 复用
  run_one $GPU WaveD_hparams waveD__cdw0p5__16shot__seed43 $LORA_NF --cd_weight 0.5 --seed 43
  run_one $GPU WaveD_hparams waveD__cdw1p0__16shot__seed43 $LORA_NF --cd_weight 1.0 --seed 43
  run_one $GPU WaveD_hparams waveD__cdw4p0__16shot__seed43 $LORA_NF --cd_weight 4.0 --seed 43
  run_one $GPU WaveD_hparams waveD__cdt1p0__16shot__seed43 $LORA_NF --cd_temperature 1.0 --seed 43
  run_one $GPU WaveD_hparams waveD__cdt2p0__16shot__seed43 $LORA_NF --cd_temperature 2.0 --seed 43
  run_one $GPU WaveD_hparams waveD__cdt8p0__16shot__seed43 $LORA_NF --cd_temperature 8.0 --seed 43
  run_one $GPU WaveD_hparams waveD__eps0p02__16shot__seed43 $LORA_NF --nsp_eps 0.02 --seed 43
  run_one $GPU WaveD_hparams waveD__eps0p05__16shot__seed43 $LORA_NF --nsp_eps 0.05 --seed 43
  run_one $GPU WaveD_hparams waveD__eps0p10__16shot__seed43 $LORA_NF --nsp_eps 0.10 --seed 43
  run_one $GPU WaveD_hparams waveD__nspw0p00__16shot__seed43 $LORA_NF --nsp_weight 0.0 --seed 43
  run_one $GPU WaveD_hparams waveD__nspw0p04__16shot__seed43 $LORA_NF --nsp_weight 0.04 --seed 43
  run_one $GPU WaveD_hparams waveD__nspw0p08__16shot__seed43 $LORA_NF --nsp_weight 0.08 --seed 43
  run_one $GPU WaveD_hparams waveD__nspw0p16__16shot__seed43 $LORA_NF --nsp_weight 0.16 --seed 43
  run_one $GPU WaveD_hparams waveD__layers_attn__16shot__seed43 $LORA_NF \
    --lora_target_modules q_proj,k_proj,v_proj,out_proj --seed 43
  run_one $GPU WaveD_hparams waveD__layers_ffn__16shot__seed43 $LORA_NF \
    --lora_target_modules fc1,fc2 --seed 43
  ;;

waveE)
  export CLIP_MODEL_NAME=google/siglip2-base-patch16-224
  for seed in 42 43 44; do
    run_one $GPU WaveE_siglip2 waveE__siglip2_lora_nf__16shot__seed${seed} $LORA_NF --seed $seed
  done
  ;;

*)
  echo "unknown wave: $WAVE"; exit 1 ;;
esac
echo "[wave $WAVE gpu $GPU] all queued runs finished ($(date '+%F %T'))"
