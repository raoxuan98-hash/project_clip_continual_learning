# 2026-06-19 英文图文检索数据集下载与接入完成

## 背景

用户要求在新增图文检索评估后，下载英文 COCO/Flickr30K，并可能通过 HuggingFace mirror 下载。目标是让当前 retrieval evaluator 能直接评估英文 image-caption benchmark，而不是只依赖 Flickr8K 或 Flickr30k-CN。

## 存储位置

远端 `/mnt/open_datasets` 对当前用户不可写，因此实际下载位置为：

```text
/mnt/raoxuan/open_datasets
```

下载日志目录：

```text
/mnt/raoxuan/open_datasets/download_logs
```

## 已下载数据集

### Flickr30K 英文版

来源：

```text
HuggingFace mirror: AnyModal/flickr30k
HF_ENDPOINT=https://hf-mirror.com
```

路径：

```text
/mnt/raoxuan/open_datasets/flickr30k_hf
```

验证结果：

```text
size: 4.1G
parquet files: 11
rows: 31014
```

数据格式：

- `data/*.parquet`
- `image` 字段为 `{"bytes": ..., "path": ...}`
- `original_alt_text` 包含 5 条英文 caption
- `alt_text` 通常包含单条 caption

下载过程说明：

- 初始整包下载在最后一个 shard 上卡住。
- 终止原 HF 下载进程后，只补下缺失文件：

```text
data/train-00003-of-00009.parquet
```

- 最终无 `.incomplete` 文件残留。

### MSCOCO 2014 5K Retrieval Split

来源：

```text
HuggingFace mirror: nlphuji/mscoco_2014_5k_test_image_text_retrieval
HF_ENDPOINT=https://hf-mirror.com
```

路径：

```text
/mnt/raoxuan/open_datasets/mscoco_2014_5k_test_hf
```

验证结果：

```text
size: 1.6G
images: 5000
csv rows: 5000
```

关键文件：

```text
/mnt/raoxuan/open_datasets/mscoco_2014_5k_test_hf/test_5k_mscoco_2014.csv
/mnt/raoxuan/open_datasets/mscoco_2014_5k_test_hf/images_mscoco_2014_5k_test/
```

数据格式：

- CSV 每行对应一张图。
- `filename` 指向图片文件名。
- `raw` 保存 5 条英文 caption 的 list 字符串。
- 图片 zip 已解压到 `images_mscoco_2014_5k_test/`。

## 官方完整 COCO 下载尝试

曾尝试下载官方完整 COCO val2014：

```text
http://images.cocodataset.org/zips/val2014.zip
http://images.cocodataset.org/annotations/annotations_trainval2014.zip
```

问题：

- `/mnt/open_datasets` 不可写。
- snap 版本 `aria2c` 不能写 `/mnt/raoxuan`。
- `wget/curl` 断点续传官方 full val2014 时链路不稳定，停在 partial zip。

处理：

- 已删除 partial `coco_val2014` 目录，避免后续误用。
- 当前采用 MSCOCO 2014 5K retrieval split 作为 COCO 图文检索 benchmark；它更轻，也更贴近 image-text retrieval 常用评估。

## 代码接入

更新了：

```text
src/utils/retrieval_eval.py
scripts/evaluate_retrieval_artifacts.py
```

新增 retrieval dataset 名称：

```text
mscoco_2014_5k
flickr30k_hf
```

当前 `scripts/evaluate_retrieval_artifacts.py --dataset` 支持：

```text
coco_val2014
mscoco_2014_5k
flickr30k_hf
flickr30k_cn
flickr8k
```

## 验证

本地和远端均通过：

```bash
python -m py_compile \
  src/utils/retrieval_eval.py \
  scripts/evaluate_retrieval_artifacts.py \
  scripts/evaluate_incremental_artifacts.py \
  scripts/merge_incremental_async_results.py
```

远端 smoke：

```text
mscoco_2014_5k: 2 images, frozen + step artifact retrieval 通过
flickr30k_hf:   2 images, frozen + step artifact retrieval 通过
```

smoke 输出：

```text
experiments/retrieval_english_download_smoke/mscoco
experiments/retrieval_english_download_smoke/flickr30k
```

确认无遗留下载进程。

## 可用命令

手动评估 MSCOCO 5K：

```bash
python scripts/evaluate_retrieval_artifacts.py \
  --artifact_path <artifact.pt> \
  --output_dir experiments/retrieval_mscoco \
  --experiment_name <name> \
  --dataset mscoco_2014_5k \
  --retrieval_root /mnt/raoxuan/open_datasets/mscoco_2014_5k_test_hf \
  --include_frozen_baseline
```

手动评估 Flickr30K：

```bash
python scripts/evaluate_retrieval_artifacts.py \
  --artifact_path <artifact.pt> \
  --output_dir experiments/retrieval_flickr30k \
  --experiment_name <name> \
  --dataset flickr30k_hf \
  --retrieval_root /mnt/raoxuan/open_datasets/flickr30k_hf \
  --include_frozen_baseline
```

接入 async worker：

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/evaluate_incremental_artifacts.py \
  --async_eval_dir experiments/<exp>/async/<run_name> \
  --device cuda:0 \
  --enable_retrieval_eval \
  --retrieval_datasets mscoco_2014_5k,flickr30k_hf \
  --retrieval_roots mscoco_2014_5k=/mnt/raoxuan/open_datasets/mscoco_2014_5k_test_hf,flickr30k_hf=/mnt/raoxuan/open_datasets/flickr30k_hf \
  --include_frozen_retrieval_baseline
```

## 当前注意事项

- `flickr30k_hf` loader 会读取 parquet 中的 image bytes；全量评估会占用一定内存和 I/O。
- `mscoco_2014_5k` 是当前推荐的 COCO retrieval benchmark。
- `coco_val2014` loader 仍保留，但官方 full val2014 数据当前未准备好。
- 本地相关文件仍是 untracked 状态，未执行 git add/commit。
