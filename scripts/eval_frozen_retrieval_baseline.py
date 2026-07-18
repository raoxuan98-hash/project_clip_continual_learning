"""冻结 CLIP 检索基线评估（canonical split 验证脚本）。

用途：在 canonical Karpathy test split（MSCOCO 2014 5K test / Flickr30K 1K test）
上评估未做任何适配的冻结 CLIP，对齐公开报告数值，作为后续一切检索对比的锚点。

数据目录约定（与 src/utils/retrieval_eval.py 的 loader 一致）：
  mscoco_2014_5k: {root}/images_mscoco_2014_5k_test/ + {root}/test_5k_mscoco_2014.csv
  flickr30k_hf:   {root}/data/test-*.parquet（仅 Karpathy 1K test，无全量回退）

运行示例（服务器）：
  CUDA_VISIBLE_DEVICES=2 python scripts/eval_frozen_retrieval_baseline.py \
      --output experiments/retrieval_baseline/frozen_clip_retrieval.json
"""

import argparse
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.models.clip import get_clip_model
from src.utils.retrieval_eval import load_retrieval_dataset, evaluate_retrieval_model

DEFAULT_ROOTS = {
    "mscoco_2014_5k": "/mnt/raoxuan/open_datasets/mscoco_2014_5k_test_hf",
    "flickr30k_hf": "/mnt/raoxuan/open_datasets/flickr30k_hf",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default="mscoco_2014_5k,flickr30k_hf")
    parser.add_argument("--mscoco_root", type=str, default=DEFAULT_ROOTS["mscoco_2014_5k"])
    parser.add_argument("--flickr30k_root", type=str, default=DEFAULT_ROOTS["flickr30k_hf"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--text_batch_size", type=int, default=256)
    parser.add_argument("--recall_ks", type=str, default="1,5,10")
    parser.add_argument("--max_images", type=int, default=0, help=">0 时仅 smoke test（跳过 split 断言）")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    roots = {
        "mscoco_2014_5k": args.mscoco_root,
        "flickr30k_hf": args.flickr30k_root,
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = get_clip_model(SimpleNamespace(), train_mode="frozen")
    model = model.to(device).eval()

    results = {}
    for name in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        ds = load_retrieval_dataset(name, roots[name], max_images=args.max_images)
        print(f"[{name}] metadata: {ds.metadata()}", flush=True)
        metrics = evaluate_retrieval_model(
            model, processor, ds, device,
            batch_size=args.batch_size,
            text_batch_size=args.text_batch_size,
            recall_ks=tuple(int(k) for k in args.recall_ks.split(",")),
        )
        results[name] = metrics
        print(f"[{name}] {json.dumps(metrics, indent=2)}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
