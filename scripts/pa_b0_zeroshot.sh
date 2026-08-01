#!/bin/bash
# pa_b0_zeroshot.sh — PA 新协议 B0（冻结零样本）评测：分类 + 检索
# 仅评测、单进程，等 GPU 槽位空闲时运行（约 20-30 分钟）。
# 用法: CUDA_VISIBLE_DEVICES=<gpu> bash scripts/pa_b0_zeroshot.sh
set -u
cd /home/raoxuan/projects/project_clip_continual_learning

export CLIP_LOCAL_FILES_ONLY=1
export CLIP_MODEL_NAME=/mnt/raoxuan/models/clip-vit-base-patch16
PY=/home/raoxuan/ENTER/envs/raoxuan/bin/python
OUT=experiments/paper_formal/PA_main
LOGDIR=artifacts/launch_logs
mkdir -p "$OUT" "$LOGDIR"

ZS_JSON="$OUT/PA__b0_zeroshot_preserve_aspect.json"
RET_JSON="$OUT/PA__b0_retrieval_baseline.json"

# 检查是否已有转换后的标准文件；有则跳过，避免重复跑
test -f "$OUT/PA__b0_preserve_aspect_ens_results.json" && { echo "[b0] already converted, skip"; exit 0; }

echo "[b0] zeroshot classification (preserve_aspect) $(date '+%F %T')"
$PY scripts/eval_frozen_zeroshot.py \
  --root /data1/open_datasets/X-TAIL \
  --resize-mode preserve_aspect \
  --output "$ZS_JSON" \
  > "$LOGDIR/PA__b0_zeroshot.log" 2>&1

echo "[b0] retrieval baseline $(date '+%F %T')"
$PY scripts/eval_frozen_retrieval_baseline.py \
  --datasets mscoco_2014_5k,flickr30k_hf \
  --output "$RET_JSON" \
  >> "$LOGDIR/PA__b0_zeroshot.log" 2>&1

echo "[b0] convert to PA_main standard format $(date '+%F %T')"
$PY scripts/convert_b0_to_pa_format.py \
  --zeroshot-json "$ZS_JSON" \
  --retrieval-json "$RET_JSON" \
  --output-name PA__b0_preserve_aspect \
  --output-dir "$OUT" \
  >> "$LOGDIR/PA__b0_zeroshot.log" 2>&1

echo "[b0] done $(date '+%F %T')"
