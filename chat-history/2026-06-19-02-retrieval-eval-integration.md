# 2026-06-19 图文检索评估接入进展

## 背景

用户希望在提升零样本分类能力的同时，评估并维持 CLIP 的跨模态图文检索能力。此前阅读 C-CLIP 相关工作后，确认第一步应先把 retrieval evaluator 接入现有实验流程，而不是立即改训练目标。

## 设计决策

- 不在 `main_incremental.py` 的训练 inline eval 中执行 retrieval，避免拖慢训练。
- retrieval 接入现有训推分离流程：
  - 训练端继续保存 step artifact。
  - `scripts/evaluate_incremental_artifacts.py` 作为 artifact worker，同时可选执行分类评估和图文检索评估。
  - `scripts/merge_incremental_async_results.py` 合并分类结果时，可选引用 retrieval summary。
- retrieval 结果保持独立 summary：
  - 不把完整 retrieval 曲线塞进分类 merged JSON。
  - merged JSON 只保存 retrieval summary 路径和关键 R@1 指标。
- `frozen CLIP` 每个 retrieval dataset 只评估一次；已有 baseline JSON 时直接复用。
- 当前 retrieval 衡量 artifact 保存态 encoder 的图文对齐能力，不使用 LR-RGDA、LADA classifier、zero-shot class anchors 或 cached class text features。
- 对 LADA text-adapter reset 后保存的 artifact，retrieval 可能等同 frozen CLIP；这不表示分类侧缓存语义锚点没有变化。

## 已实现内容

新增共享模块：

- `src/utils/retrieval_eval.py`
  - 统一实现 I2T/T2I R@1/R@5/R@10。
  - 支持 chunked similarity，避免一次性构造大相似度矩阵。
  - 支持 `max_images` smoke。
  - 支持统一 JSON payload 和 summary row。
  - 支持 dataset loader：
    - `flickr8k`
    - `flickr30k_cn`
    - `coco_val2014`

重构手动脚本：

- `scripts/evaluate_retrieval_artifacts.py`
  - 保持原有手动 artifact 扫描能力。
  - 改为调用 `src/utils/retrieval_eval.py`，避免与 worker 产生两套指标实现。
  - 支持 `--include_frozen_baseline`，且已有 frozen baseline JSON 时复用。

接入 async worker：

- `scripts/evaluate_incremental_artifacts.py`
  - 新增参数：
    - `--enable_retrieval_eval`
    - `--retrieval_datasets`
    - `--retrieval_roots`
    - `--retrieval_output_dir`
    - `--retrieval_max_images`
    - `--retrieval_batch_size`
    - `--retrieval_text_batch_size`
    - `--retrieval_num_workers`
    - `--retrieval_similarity_chunk_size`
    - `--retrieval_recall_ks`
    - `--include_frozen_retrieval_baseline`
  - 分类 result 已存在但 retrieval 未跑时，可以只补 retrieval。
  - 补跑 retrieval 失败时，queue job 会标记为 `.failed.json`，不会停在 `.running.json`。

接入 merge：

- `scripts/merge_incremental_async_results.py`
  - 新增参数：
    - `--include_retrieval`
    - `--retrieval_results_dir`
  - 生成：
    - `<stem>_retrieval_summary.csv`
    - `<stem>_retrieval_summary.md`
    - `<stem>_retrieval_summary.json`
  - 在 `<stem>_merged.json` 中新增轻量字段 `retrieval`，记录 summary 路径、dataset 列表和 frozen/latest step 的关键 R@1。

## 验证结果

本地和远端均通过：

```bash
python -m py_compile \
  src/utils/retrieval_eval.py \
  scripts/evaluate_retrieval_artifacts.py \
  scripts/evaluate_incremental_artifacts.py \
  scripts/merge_incremental_async_results.py
```

远端 smoke 结果：

- 手动 Flickr8K smoke 通过：
  - 输出：`experiments/retrieval_integration_smoke`
  - 数据：4 images
  - 成功生成 frozen + step artifact retrieval JSON 和 summary。
- worker integration smoke 通过：
  - 输出：`experiments/retrieval_worker_integration_smoke`
  - 验证了“分类 result 已存在，只补 retrieval”的路径。
  - merge 成功生成 `worker_smoke_retrieval_summary.*`，并在 merged JSON 中加入 `retrieval` 字段。
- Flickr30k-CN loader smoke 通过：
  - 输出：`experiments/retrieval_flickr30k_cn_smoke`
  - 数据：2 images
  - 验证 TSV/jsonl/base64 格式读取正常。

## 重要限制

- 远端目前已确认可用：
  - `/mnt/open_datasets/flickr8k`
  - `/mnt/open_datasets/chinese-clip-eval/Flickr30k-CN`
- 标准英文 COCO/Flickr30K 数据路径尚未确认。
- `flickr30k_cn` 是中文 caption benchmark，对英文 CLIP 的结果需要单独解释，不能直接作为英文 Flickr30K 结果。
- `coco_val2014` loader 已实现，期望格式：

```text
<root>/val2014/
<root>/annotations/captions_val2014.json
```

## 推荐命令

Flickr8K retrieval worker：

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/evaluate_incremental_artifacts.py \
  --async_eval_dir experiments/<exp>/async/<run_name> \
  --device cuda:0 \
  --enable_retrieval_eval \
  --retrieval_datasets flickr8k \
  --retrieval_roots flickr8k=/mnt/open_datasets/flickr8k \
  --include_frozen_retrieval_baseline
```

COCO val2014 retrieval worker，数据准备好后：

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/evaluate_incremental_artifacts.py \
  --async_eval_dir experiments/<exp>/async/<run_name> \
  --device cuda:0 \
  --enable_retrieval_eval \
  --retrieval_datasets coco_val2014 \
  --retrieval_roots coco_val2014=/mnt/open_datasets/coco_val2014 \
  --include_frozen_retrieval_baseline
```

合并 retrieval summary：

```bash
python scripts/merge_incremental_async_results.py \
  --async_eval_dir experiments/<exp>/async/<run_name> \
  --output_dir experiments/<exp>/merged \
  --experiment_name <run_name> \
  --include_retrieval
```

## 下一步

- 下载英文 COCO/Flickr30K 到 `/mnt/open_datasets`。
- 优先准备标准 `coco_val2014`，因为当前 loader 已实现。
- 若下载标准 Flickr30K 英文版，需确认其 annotation 格式后补 loader；不要把 `flickr30k_cn` 当英文 Flickr30K 使用。

## 追加进展：英文 COCO/Flickr30K 下载与 loader 扩展

远端 `/mnt/open_datasets` 对当前用户不可写，因此实际下载位置改为：

```text
/mnt/raoxuan/open_datasets
```

已下载并验证：

- Flickr30K 英文版：
  - 来源：HuggingFace mirror，`AnyModal/flickr30k`
  - 路径：`/mnt/raoxuan/open_datasets/flickr30k_hf`
  - 格式：parquet shards
  - 验证：`11` 个 parquet 文件，`31014` rows
  - 字段：`image` 为 `{"bytes": ..., "path": ...}`，`original_alt_text` 包含 5 条英文 caption
- MSCOCO 2014 5K retrieval split：
  - 来源：HuggingFace mirror，`nlphuji/mscoco_2014_5k_test_image_text_retrieval`
  - 路径：`/mnt/raoxuan/open_datasets/mscoco_2014_5k_test_hf`
  - 格式：`test_5k_mscoco_2014.csv` + `images_mscoco_2014_5k_test.zip`
  - 已解压：`images_mscoco_2014_5k_test/`
  - 验证：`5000` images，`5000` CSV rows，每张图 5 条英文 caption

下载中遇到的情况：

- 曾尝试下载官方完整 COCO `val2014.zip` + `annotations_trainval2014.zip`。
- `/mnt/open_datasets` 无写权限，因此不能放到该目录。
- `aria2c` 是 snap 版本，不能写 `/mnt/raoxuan`。
- `wget/curl` 从官方 COCO 源下载 full val2014 不稳定，停在 partial zip；已删除 partial 目录，避免误用。
- 当前选择 MSCOCO 2014 5K retrieval split 作为 COCO 图文检索 benchmark，更轻且更符合 retrieval evaluation 常用设置。

同时扩展了 retrieval loader：

- `src/utils/retrieval_eval.py`
  - 新增 `mscoco_2014_5k`
  - 新增 `flickr30k_hf`
- `scripts/evaluate_retrieval_artifacts.py`
  - dataset choices 新增 `mscoco_2014_5k` 和 `flickr30k_hf`

验证：

```bash
python -m py_compile \
  src/utils/retrieval_eval.py \
  scripts/evaluate_retrieval_artifacts.py \
  scripts/evaluate_incremental_artifacts.py \
  scripts/merge_incremental_async_results.py
```

远端 smoke：

- `mscoco_2014_5k`，2 images，frozen + step artifact retrieval 通过。
- `flickr30k_hf`，2 images，frozen + step artifact retrieval 通过。

英文 retrieval 可用命令：

```bash
python scripts/evaluate_retrieval_artifacts.py \
  --artifact_path <artifact.pt> \
  --output_dir experiments/retrieval_mscoco \
  --experiment_name <name> \
  --dataset mscoco_2014_5k \
  --retrieval_root /mnt/raoxuan/open_datasets/mscoco_2014_5k_test_hf \
  --include_frozen_baseline
```

```bash
python scripts/evaluate_retrieval_artifacts.py \
  --artifact_path <artifact.pt> \
  --output_dir experiments/retrieval_flickr30k \
  --experiment_name <name> \
  --dataset flickr30k_hf \
  --retrieval_root /mnt/raoxuan/open_datasets/flickr30k_hf \
  --include_frozen_baseline
```
