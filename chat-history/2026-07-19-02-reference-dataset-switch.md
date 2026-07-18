# 蒸馏参考集切换为 Flickr30K train 子集 + Wave A 全量重启

**日期**: 2026-07-19
**会话概况**: 用户拍板蒸馏参考改用 Flickr30K train split 子集、检索评测恢复完整测试集；终止旧参考的全部在跑实验，完成代码修改、参考集构建与提交（`231deae`），计划文档追加 §7 修订记录。

---

## 1. 关键讨论 / 决策

- **蒸馏参考集**：Flickr8K → **Flickr30K Karpathy train split 子集**（8000 图、每图 1 条 caption、seed=42 采样）。原因：Flickr8K 几乎全是 Flickr30K 子集，与 1K test 有 243/1000 图像重合；train/test split 层面不相交。
- **检索评测**：始终使用完整测试集（MSCOCO 5K、Flickr30K 1K），废弃"剔除重叠"的 `flickr30k_hf_clean` 子集方案（不再需要）。
- **旧参考实验作废**：Wave A 在跑 runs（无任何完成 run）与 stray waveD run 全部终止、输出清除；Wave A 以新参考重新启动。
- **vanilla LoRA 崩溃修复**：`update_projection_matrices` 三处调用（`__init__` 协方差加载、两个协方差更新函数）加 `hasattr` 守卫，vanilla 走 `_no_op`。

## 2. 重要发现

- `/mnt/raoxuan/open_datasets/flickr30k_hf/data/` 已含完整 train split（9 个 parquet，29,000 图），无需重新下载。
- 构建脚本断言：train/test 零文件名重叠 ✓；输出 8000 图与 captions.txt 对齐 ✓。
- `reference_dataset` 只来自 `configs/base/default.yaml` 与启动器 CLI，configs/experiments 无覆盖，一处修改即全战役生效。
- 服务器 `pkill -f` 会匹配到发起命令自身的远程 shell，需用 `[h]` 字符类技巧规避。

## 3. 待办事项 / 遗留问题

- [ ] 新参考 smoke（lora_nf + vanilla 守卫路径）通过后重启 Wave A（6 GPU）
- [ ] v2 数字与本战役不可混比，汇总文档需注明参考集差异
- [ ] smoke 用的 `Flickr8kDataset` 类名与新数据集名不一致（仅命名问题，功能正常，输出日志"加载了 8000 个Flickr8K样本"易造成误读）

## 4. 相关文件

- `scripts/build_flickr30k_train_reference.py`：参考集构建脚本（含零重叠断言与回读校验）
- `src/utils/reference_loader.py`：`REFERENCE_ROOTS` 路径映射
- `src/utils/retrieval_eval.py`：已移除 `flickr30k_hf_clean` 分支
- `src/trainers/lora_nsp_trainer.py`：vanilla 守卫 ×3
- `docs/rerun_campaign_main_v3_2026-07-18.md` §7：修订记录
- 服务器参考集：`/mnt/raoxuan/open_datasets/flickr30k_train_sub8k/`
- commit：`231deae`（main_v3）
