#!/usr/bin/env python
"""从 Flickr30K Karpathy train split 构建蒸馏参考集（flickr8k 目录格式）。

背景：原蒸馏参考 Flickr8K 几乎全是 Flickr30K 子集，与 Karpathy 1K test
存在 243/1000 的图像重合。改为从 train split 采样（split 层面与 test 不
相交，并显式断言零文件名重叠），检索评测继续使用完整 1K test。

输出格式（与 src.utils.data.Flickr8kDataset 兼容）：
    <out>/images/*.jpg
    <out>/captions.txt        # 表头 image,caption；每图 1 条（seeded 随机选）

用法（服务器）：
    python scripts/build_flickr30k_train_reference.py
"""
import argparse
import csv
import io
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/mnt/raoxuan/open_datasets/flickr30k_hf/data",
                    help="含 train-*.parquet / test-*.parquet 的目录")
    ap.add_argument("--out", default="/mnt/raoxuan/open_datasets/flickr30k_train_sub8k")
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = Path(args.src)
    train_paths = sorted(src.glob("train-*.parquet"))
    test_paths = sorted(src.glob("test-*.parquet"))
    if not train_paths or not test_paths:
        raise FileNotFoundError(f"train/test parquet not found under {src}")

    # 1) test 文件名集合（用于零重叠断言）
    test_fnames = set()
    for p in test_paths:
        df = pd.read_parquet(p, columns=["filename"])
        test_fnames.update(df["filename"].astype(str))
    print(f"test split filenames: {len(test_fnames)}")

    # 2) 第一遍：只读 filename，建立 (shard, row, filename) 索引，低内存
    entries = []  # (shard_path, row_idx, filename)
    for p in train_paths:
        df = pd.read_parquet(p, columns=["filename"])
        for row_idx, fname in enumerate(df["filename"].astype(str)):
            entries.append((str(p), row_idx, fname))
    print(f"train split images: {len(entries)}")

    overlap = {e[2] for e in entries} & test_fnames
    if overlap:
        raise AssertionError(f"train/test filename overlap: {len(overlap)} e.g. {sorted(overlap)[:5]}")
    print("✓ zero train/test filename overlap")

    # 3) seeded 采样 n 张
    rng = random.Random(args.seed)
    picked = rng.sample(entries, min(args.n, len(entries)))
    by_shard = defaultdict(list)
    for shard, row_idx, fname in picked:
        by_shard[shard].append((row_idx, fname))

    # 4) 第二遍：按 shard 读出选中行，写 JPG + captions.txt
    out = Path(args.out)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with open(out / "captions.txt", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "caption"])
        for shard, wanted in sorted(by_shard.items()):
            wanted.sort()
            row_ids = {r for r, _ in wanted}
            df = pd.read_parquet(shard, columns=["image", "filename", "original_alt_text", "alt_text"])
            df = df.iloc[sorted(row_ids)]
            for _, row in df.iterrows():
                fname = str(row["filename"])
                if not fname.endswith(".jpg"):
                    fname = fname + ".jpg"
                caps = row["original_alt_text"]
                if caps is None or len(caps) == 0:
                    caps = row["alt_text"]
                caps = [str(c).strip() for c in caps if str(c).strip()]
                if not caps:
                    continue
                img_obj = row["image"]
                data = img_obj.get("bytes") if isinstance(img_obj, dict) else None
                if not data:
                    continue
                img = Image.open(io.BytesIO(data)).convert("RGB")
                img.save(img_dir / fname, format="JPEG", quality=95)
                writer.writerow([fname, rng.choice(caps)])
                n_written += 1
    print(f"✓ written {n_written} images -> {img_dir}")
    print(f"✓ captions -> {out / 'captions.txt'}")

    # 5) 回读校验
    df = pd.read_csv(out / "captions.txt")
    disk_imgs = {p.name for p in img_dir.glob("*.jpg")}
    assert len(df) == len(disk_imgs) == n_written, (len(df), len(disk_imgs), n_written)
    assert set(df["image"].astype(str)) == disk_imgs
    assert not (disk_imgs & test_fnames), "output overlaps test split!"
    print(f"✓ verified: {n_written} images, captions aligned, no test overlap")


if __name__ == "__main__":
    main()
