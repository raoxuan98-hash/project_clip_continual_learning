#!/bin/bash
# ============================================================
# 通过 HF mirror 下载检索数据集: COCO val2014 + Flickr30K
# ============================================================
set -euo pipefail

ROOT="${1:-/mnt/open_datasets}"
mkdir -p "$ROOT"

export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_DOWNLOAD_TIMEOUT=3600
PYTHON=$(which python 2>/dev/null || echo "/home/raoxuan/ENTER/envs/raoxuan/bin/python")

echo "=== Downloading COCO val2014 captions via HF mirror ==="
COCO_DIR="$ROOT/coco_val2014_hf"
if [ -d "$COCO_DIR/data" ] && ls "$COCO_DIR/data"/*.parquet >/dev/null 2>&1; then
    echo "COCO already exists at $COCO_DIR, skipping."
else
    rm -rf "$COCO_DIR"
    mkdir -p "$COCO_DIR"
    $PYTHON -c "
import os; os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from datasets import load_dataset
print('Downloading COCO val2014 from HF mirror...')
ds = load_dataset('HuggingFaceM4/COCO', split='validation', trust_remote_code=True)
ds.save_to_disk('$COCO_DIR')
print('COCO saved to $COCO_DIR')
"
fi

echo ""
echo "=== Downloading Flickr30K via HF mirror ==="
FLICKR_DIR="$ROOT/flickr30k_hf"
if [ -d "$FLICKR_DIR/data" ] && ls "$FLICKR_DIR/data"/*.parquet >/dev/null 2>&1; then
    echo "Flickr30K already exists at $FLICKR_DIR, skipping."
else
    rm -rf "$FLICKR_DIR"
    mkdir -p "$FLICKR_DIR"
    $PYTHON -c "
import os; os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from datasets import load_dataset
print('Downloading Flickr30K from HF mirror...')
ds = load_dataset('nlphuji/flickr30k', split='test', trust_remote_code=True)
ds.save_to_disk('$FLICKR_DIR')
print('Flickr30K saved to $FLICKR_DIR')
"
fi

echo ""
echo "=== Done ==="
echo "COCO:     $COCO_DIR"
echo "Flickr30K: $FLICKR_DIR"
