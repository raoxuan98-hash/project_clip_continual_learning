# Async Incremental Evaluation Implementation

**日期**: 2026-06-16
**会话概况**: 用户提出训练完当前任务并微调分类器后立即保存，由另一个程序在另一张 GPU 上评估。本次实现了 `main_incremental.py` 的可选异步评估 artifact 保存、独立 worker 和结果合并脚本。

---

## 1. 关键讨论 / 决策

- 采用训推分离的文件队列方案，不在进程间共享正在变化的模型对象。
- 训练进程在每个 task 完成、LoRA finalize、LR-RGDA 构建和可选 fit 之后保存 immutable step artifact。
- 评估进程读取 `queue/*.ready.json`，加载 artifact 后在另一张 GPU 上评估全部任务。
- 合并脚本将每个 step 的评估结果合成为完整 KxK matrix，并用 LADA 口径输出 `zero_shot`、`lr_rgda`、`ensemble` 三套结果。
- 默认同步评估行为不变；只有显式传入异步参数时才启用训推分离。

## 2. 重要发现

- LR-RGDA fit 后的参数会回到 module buffer，因此可以保存底层 classifier `state_dict`，worker 重建同结构分类器后加载，不需要重复 fit。
- per-classifier 输出文件必须使用 `*_results.json` 后缀，否则 `scripts/summarize_incremental_metrics.py` 在目录模式下不会自动收集。
- worker 的 `--help` 已避免顶层导入 CLIP/transformers；实际评估仍需要远程训练环境中的 transformers 和本地 CLIP cache。

## 3. 新增用法

训练进程示例：

```bash
CUDA_VISIBLE_DEVICES=0 python main_incremental.py \
  --dataset_sequence aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397 \
  --num_shots 16 \
  --batch_size 32 \
  --iterations 800 \
  --lora_type lora_nsp \
  --text_tuning_schedule low_lr_after \
  --text_schedule_switch_task 1 \
  --text_lr_scale_after_task 0.2 \
  --num_centers 4 \
  --rgda_train_iter 200 \
  --gmm_sample_mode mean \
  --gaussian_samples_per_class 16 \
  --alpha 0.05 \
  --output_dir experiments/inc_async_demo \
  --experiment_name inc_async_demo \
  --async_eval_dir experiments/inc_async_demo/async_eval \
  --save_step_artifacts \
  --skip_inline_eval
```

评估 worker 示例：

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/evaluate_incremental_artifacts.py \
  --async_eval_dir experiments/inc_async_demo/async_eval \
  --device cuda:0
```

合并结果示例：

```bash
python scripts/merge_incremental_async_results.py \
  --async_eval_dir experiments/inc_async_demo/async_eval \
  --output_dir experiments/inc_async_demo/merged \
  --experiment_name inc_async_demo \
  --expected_tasks "aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397"

python scripts/summarize_incremental_metrics.py \
  experiments/inc_async_demo/merged \
  --output_csv experiments/inc_async_demo/incremental_summary.csv \
  --output_markdown experiments/inc_async_demo/incremental_summary.md \
  --aggregate_csv experiments/inc_async_demo/incremental_aggregate.csv \
  --aggregate_markdown experiments/inc_async_demo/incremental_aggregate.md \
  --expected_tasks "aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397"
```

## 4. 待办事项 / 遗留问题

- [ ] 在远程 GPU 环境做 2-task smoke test：同步评估与异步 worker+merge 的每个 cell 应一致。
- [ ] `--enable_lada` 仍未纳入异步 worker；当前异步路径只覆盖 `zero_shot`、`lr_rgda`、`ensemble`。
- [ ] 大规模运行会保存每步完整 model state，磁盘占用较大；如空间紧张，可后续加 artifact 清理或只保存最近 N 步。

## 5. 相关文件

- `main_incremental.py`: 新增 `--async_eval_dir`、`--save_step_artifacts`、`--skip_inline_eval`，保存 step artifact 和 ready job。
- `scripts/evaluate_incremental_artifacts.py`: 异步评估 worker。
- `scripts/merge_incremental_async_results.py`: 合并 step eval rows 为 LADA KxK result JSON。
- `scripts/summarize_incremental_metrics.py`: 可直接读取合并后的 `*_results.json`。
