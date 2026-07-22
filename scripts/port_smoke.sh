#!/bin/bash
# SigLIP2 移植验证 smoke（一次性脚本，验证后可删）
# 用法: port_smoke.sh <clip|siglip2> <gpu>
set -u
cd "$(dirname "$0")/.."

BACKBONE=${1:?usage: port_smoke.sh <clip|siglip2> <gpu>}
GPU=${2:?usage: port_smoke.sh <clip|siglip2> <gpu>}

export CLIP_LOCAL_FILES_ONLY=1
case "$BACKBONE" in
  clip)    export CLIP_MODEL_NAME=/mnt/raoxuan/models/clip-vit-base-patch16 ;;
  siglip2) export CLIP_MODEL_NAME=/mnt/raoxuan/models/siglip2-base-patch16-224 ;;
  *) echo "unknown backbone $BACKBONE"; exit 1 ;;
esac

PY=/home/raoxuan/ENTER/envs/raoxuan/bin/python
RETR_ROOTS="mscoco_2014_5k=/mnt/raoxuan/open_datasets/mscoco_2014_5k_test_hf,flickr30k_hf=/mnt/raoxuan/open_datasets/flickr30k_hf"
OUTDIR=experiments/paper_formal/PortSmoke
mkdir -p "$OUTDIR"
NAME="port_smoke__${BACKBONE}__2task30iter"

CUDA_VISIBLE_DEVICES=$GPU $PY -u main_incremental.py \
  --root /data1/open_datasets/X-TAIL \
  --dataset_sequence aircraft caltech101 \
  --num_shots 16 --batch_size 32 --eval_batch_size 128 --iterations 30 --train_budget_mode uniform \
  --optimizer adamw --lr 1e-4 --weight_decay 3e-5 --scheduler cosine_with_warmup --warmup_ratio 0.1 --eta_min 0.0 \
  --lora_rank 4 --lora_alpha 4 --lora_dropout 0.0 \
  --lora_target_modules q_proj,k_proj,v_proj,out_proj,fc1,fc2 \
  --projection_param_mode full --nsp_eps 0.20 --nsp_weight 0.02 \
  --reference_dataset flickr30k_train_sub8k --reference_batch_size 32 \
  --fd_weight 0.0 --cd_weight 2.0 --cd_divergence kl_forward --cd_temperature 4.0 --aux_weight 0.0 \
  --tune_vision_encoder true --tune_text_encoder true --text_lora_rank 4 \
  --text_tuning_schedule always --text_classifier_mode lada_hybrid \
  --classifier_feature_transform test \
  --rgda_rank 32 --rgda_alpha1 0.2 --rgda_alpha2 2.0 --rgda_alpha3 0.5 \
  --num_centers 4 --rgda_train_iter 200 --rgda_train_lr 0.01 --rgda_fit_source gmm_sample \
  --alpha 0.05 --ensemble_normalize maxshift --temperature 1.0 \
  --enable_retrieval_eval --retrieval_datasets mscoco_2014_5k,flickr30k_hf \
  --retrieval_roots "$RETR_ROOTS" --retrieval_batch_size 128 --retrieval_recall_ks 1,5,10 --retrieval_max_images 200 \
  --disable_lada --no-alpha_sensitivity \
  --lora_type lora_nsp --init_mode lora_nsp --use_dora false --null_init_mode none \
  --seed 42 \
  --output_dir "$OUTDIR" --experiment_name "$NAME" \
  > "$OUTDIR/${NAME}.log" 2>&1
rc=$?
echo "[port_smoke] $BACKBONE gpu=$GPU rc=$rc ($(date '+%F %T'))"
exit $rc
