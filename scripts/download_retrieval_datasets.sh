#!/bin/bash
# ============================================================
# 下载多模态检索评估数据集: COCO val2014 + Flickr30K
# 保存到 --retrieval_root 指定的共享路径 (默认 /data1/open_datasets)
# ============================================================
set -euo pipefail

ROOT="${1:-/data1/open_datasets}"
mkdir -p "$ROOT"

echo "=== Downloading COCO val2014 ==="
COCO_DIR="$ROOT/coco_val2014"
if [ -d "$COCO_DIR/val2014" ] && [ -f "$COCO_DIR/annotations/captions_val2014.json" ]; then
    echo "COCO val2014 already exists at $COCO_DIR, skipping."
else
    echo "Step 1: Download COCO val2014 images (wget)"
    mkdir -p "$COCO_DIR"
    # COCO 2014 validation images
    wget -c http://images.cocodataset.org/zips/val2014.zip -O /tmp/val2014.zip
    unzip -qo /tmp/val2014.zip -d "$COCO_DIR"
    rm -f /tmp/val2014.zip

    echo "Step 2: Download COCO annotations"
    wget -c http://images.cocodataset.org/annotations/annotations_trainval2014.zip -O /tmp/annotations.zip
    mkdir -p "$COCO_DIR/annotations"
    unzip -qo /tmp/annotations.zip -d "$COCO_DIR/annotations"
    rm -f /tmp/annotations.zip
    echo "COCO val2014 ready at $COCO_DIR"
fi

echo ""
echo "=== Downloading Flickr30K (HF parquet) ==="
FLICKR_DIR="$ROOT/flickr30k_hf"
if [ -d "$FLICKR_DIR/data" ] && ls "$FLICKR_DIR/data"/*.parquet >/dev/null 2>&1; then
    echo "Flickr30K already exists at $FLICKR_DIR, skipping."
else
    echo "Downloading Flickr30K via HuggingFace datasets..."
    mkdir -p "$FLICKR_DIR"
    python -c "
from datasets import load_dataset
ds = load_dataset('nlphuji/flickr30k', split='test')
ds.save_to_disk('$FLICKR_DIR')
"
    # HF save_to_disk puts parquet files under data/
    if [ -d "$FLICKR_DIR/data" ]; then
        echo "Flickr30K ready at $FLICKR_DIR"
    else
        # Older datasets versions may save differently
        mv "$FLICKR_DIR"/*.parquet "$FLICKR_DIR/data/" 2>/dev/null || true
        mkdir -p "$FLICKR_DIR/data"
        echo "Flickr30K saved to $FLICKR_DIR (check structure)"
    fi
fi

echo ""
echo "=== Done ==="
echo "COCO val2014:  $COCO_DIR"
echo "Flickr30K:     $FLICKR_DIR"
echo ""
echo "Usage:"
echo "  python main_incremental.py ... --enable_retrieval_eval \\"
echo "    --retrieval_datasets coco_val2014,flickr30k_hf \\"
echo "    --retrieval_root $ROOT"
