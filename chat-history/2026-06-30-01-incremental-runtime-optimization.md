# main_incremental.py 运行时间分析与优化

**日期**: 2026-06-30
**会话概况**: 分析 `main_incremental.py` 执行 7-8 小时的原因，定位主要计算瓶颈，并实现低风险的运行时优化与性能剖析工具。

---

## 1. 关键讨论 / 决策

- **决策 1**: 在 `main_incremental.py` 中引入 `StageTimer`，对每个任务的数据加载、训练、NSP/merge、特征提取、分类器构建、评估等阶段计时，以便在服务器上 empirical 地验证瓶颈。
- **决策 2**: 新增 `--eval_batch_size` 参数，让评估阶段可以使用比训练更大的 batch size（如 128/256），显著加速测试集特征提取。
- **决策 3**: 将 `--text_tuning_schedule` 默认值从 `always` 改为 `freeze_after`，即仅在任务 1 训练文本端，之后冻结。
- **决策 4**: 将 `--text_classifier_mode` 默认值从 `current` 改为 `lada_hybrid`，确保零样本分类器构建符合持续学习协议：已见类用各任务当时的文本原型，未见类用预训练 CLIP。
- **决策 5**: 明确推荐 **async eval 路径** 作为 fastest path：训练时用 `--save_step_artifacts --async_eval_dir <dir> --skip_inline_eval`，评估用 `scripts/evaluate_incremental_artifacts.py` 异步执行。

---

## 2. 重要发现

### 2.1 主要瓶颈（静态分析）

| 阶段 | 耗时特征 | 优化手段 |
|------|---------|---------|
| **Inline 评估** | 10 个任务共需 1+2+...+10 = **55 次完整测试集特征提取**，是 O(K²) 开销 | 增大 `--eval_batch_size`；使用 async eval 解耦 |
| **LADA 分类器构建** | 默认启用，每任务把所有已见任务的训练特征拼起来重新 build/fit，随任务数增加而变慢 | 不需要对比时加 `--disable_lada` |
| **训练** | 10 × 800 iter，每步两次 forward（训练图 + 蒸馏参考图） | 属于必要开销，不易压缩 |
| **特征提取重复调用** | 每个任务为训练/协方差/分类器/GMM 分别构建 loader 并前向传播 | 已通过代码结构尽量复用，16-shot 数据量小，收益有限 |
| **Artifact 保存** | `--save_step_artifacts` 时每任务保存完整模型 state dict（~600MB） | 仅在需要 async eval 时开启 |

### 2.2 已实现的代码改动

- `src/utils/main_utils.py`:
  - `evaluate_dataset` 增加可选参数 `eval_batch_size=None`、`te_loader`、`c_names`、`keep_features_on_device`，无感兼容旧调用。
  - `evaluate_dataset` 与 `batch_evaluate_datasets` 改用 `torch.inference_mode()` 加速推理。
- `src/utils/feature_extractor.py`:
  - `extract_features` 增加 `keep_on_device` 参数，支持把特征保留在 GPU 上。
  - 使用 `torch.inference_mode()` 替代 `torch.no_grad()`。
- `main_incremental.py`:
  - 新增 `--eval_batch_size` 参数。
  - 新增 `--eval_keep_features_on_device` 参数。
  - 新增 `StageTimer` 类与全局 `_RUN_TIMER`，对 `init`、`data_load`、`train`、`nsp_and_merge`、`feature_extraction`、`build_classifiers`、`evaluate` 以及每个 task 的 total 时间进行累积计时。
  - 在每个 task 结束时与脚本末尾打印计时摘要。
  - 预构建所有测试 loader 并缓存，避免 100 次评估中重复构造 dataset。
  - 预计算 frozen CLIP 的零样本分类器并在每轮复用。
  - 把 `eval_batch_size`、`te_loader`、`keep_features_on_device` 传递给 `evaluate_dataset`。
  - 更新模块 docstring，加入加速建议。

---

## 3. 待办事项 / 遗留问题

- [x] 将本地修改提交并同步到远程服务器 `/home/raoxuan/projects/project_clip_continual_learning`（通过 rsync，GitHub push 因 VPN/网络限制失败）。
- [ ] 在服务器上实际跑一次带 `--eval_batch_size 128/256` 的实验，确认 timing summary 中 `evaluate` 阶段占比，验证优化效果。
- [ ] 如果评估仍占主要时间，进一步实现全量 async eval launcher。
- [ ] 考虑把 `src/experiments/run_continual_learning.py` 也加入 `eval_batch_size` 支持（当前用户主要关注 `main_incremental.py`）。

---

## 4. 相关文件

- `main_incremental.py`: 主要改动文件（计时、--eval_batch_size、text_tuning_schedule 默认改为 freeze_after、docstring）。
- `src/utils/main_utils.py`: `evaluate_dataset` 增加 `eval_batch_size` 参数。
- `scripts/evaluate_incremental_artifacts.py`: async eval 入口，推荐与 `--skip_inline_eval` 配合使用。
