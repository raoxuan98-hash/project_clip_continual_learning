#!/usr/bin/env python3
"""Frozen CLIP 跨模态检索行（Wave F）：canonical 检索副本上的零样本检索基线。

    CLIP_LOCAL_FILES_ONLY=1 CLIP_MODEL_NAME=/mnt/raoxuan/models/clip-vit-base-patch16 \
    python scripts/eval_frozen_retrieval.py \
        --output experiments/paper_formal/WaveF_offline/frozen_retrieval.json
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.clip import get_clip_model
from src.utils.retrieval_eval import (
    evaluate_retrieval_model,
    load_retrieval_dataset,
)

ROOTS = {
    "mscoco_2014_5k": "/mnt/raoxuan/open_datasets/mscoco_2014_5k_test_hf",
    "flickr30k_hf": "/mnt/raoxuan/open_datasets/flickr30k_hf",
}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path,
                    default=Path("experiments/paper_formal/WaveF_offline/frozen_retrieval.json"))
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = get_clip_model(SimpleNamespace(model_name=None), train_mode="frozen")
    model.to(device).eval()

    results = {}
    for name, root in ROOTS.items():
        dataset = load_retrieval_dataset(name, root, max_images=0)
        metrics = evaluate_retrieval_model(model, processor, dataset, device,
                                           batch_size=args.batch_size)
        results[name] = {
            "i2t_r@1": metrics["i2t"].get("r@1"),
            "t2i_r@1": metrics["t2i"].get("r@1"),
            "i2t_r@5": metrics["i2t"].get("r@5"),
            "t2i_r@5": metrics["t2i"].get("r@5"),
            "num_images": metrics["num_images"],
            "num_captions": metrics["num_captions"],
        }
        print(f"{name}: I2T R@1={results[name]['i2t_r@1']:.2f} "
              f"T2I R@1={results[name]['t2i_r@1']:.2f} "
              f"(N_img={metrics['num_images']})", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"written -> {args.output}")


if __name__ == "__main__":
    main()
