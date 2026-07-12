# 同步训练 + 内联 alpha 扫描进展记录

**日期**：2026-07-05  
**记录人**：Kimi Code CLI  
**对应目标**：完成 `project_clip_continual_learning` 的同步训练 + 内联 alpha 扫描验证与完整 10-task LADA 评估。

## 已完成的工作

1. **代码改动（服务器 `/home/raoxuan/projects/project_clip_continual_learning/`）**
   - `main_incremental.py`：
     - 增加 `--alpha_sensitivity` CLI 参数。
     - inline eval 调用 `evaluate_dataset(..., alpha_sensitivity=args.alpha_sensitivity, n_alpha_samples=21)`。
     - 训练结束后为每个 alpha 构造 K×K 准确率矩阵，并打印 `[Alpha Sensitivity Sweep] Transfer / Average / Last per alpha` 表格。
   - `scripts/launch_lada_10task_eval.sh`：
     - 用 `--alpha_sensitivity` 替换原来的 `--save_step_artifacts --skip_inline_eval`。
     - 移除所有 `--async_eval_dir` 参数，避免生成大体积异步目录。
     - 启动时清理旧的 train.log 和 json。
   - `scripts/monitor_lada_10task_eval.sh`：简化为只等待训练结束，不再调用离线评估脚本。
   - 新增 `scripts/extract_alpha_sweep.py`：用于从 `combo*_train.log` 中提取 alpha 扫描表格并汇总。

2. **smoke test 启动**
   - 命令：2 个 task（aircraft, caltech101）、100 iter、`--alpha_sensitivity`。
   - PID：`916249`，日志：`experiments/alpha_sweep_smoke.log`。
   - 训练速度约 3.6 it/s；Task 1 和 Task 2 训练均正常完成。

3. **环境检查**
   - 磁盘 `/` 剩余约 213G，无 `*_async` 大目录。
   - GPU 0 被 smoke test 占用，其余 GPU 空闲。

## 当前阻塞

smoke test 在打印 alpha 扫描表格前崩溃：

```text
File "/home/raoxuan/projects/project_clip_continual_learning/main_incremental.py", line 1293, in main
    stats = _alpha_tracker_to_stats(tracker)
File "/home/raoxuan/projects/project_clip_continual_learning/main_incremental.py", line 1284, in _alpha_tracker_to_stats
    return get_full_stats(matrix)
NameError: name 'get_full_stats' is not defined
```

原因：`main_incremental.py` 的 alpha 扫描汇总代码调用了 `get_full_stats()`，但该函数在当前文件中未定义。它只在 `main.py` 中作为局部函数存在，且用户指出 `main.py` 的实现对准确度的处理不够准确，**不应参考 `main.py` 的准确度实现**。

因此下一步需要：
- 在 `main_incremental.py` 中正确定义（或从 `src/utils/continual_metrics.py` 导入）`get_full_stats`。
- 实现应基于 `ContinualLearningMetrics` 中已有的 LADA 公式（Transfer / Average / Last 分别按论文定义计算），并保持与 alpha 扫描代码中矩阵值范围一致（代码里存入的是 `acc / 100.0`）。

## 尚未完成

1. 修复 `get_full_stats` 后重新跑 smoke test，确认日志中出现 21 个 alpha 的 Transfer / Average / Last 表格。
2. smoke test 通过后启动 4 个 best combo 的完整 10-task 评估。
3. 4 个 combo 完成后提取并汇总 alpha 扫描结果。

## 相关文件

- `main_incremental.py`
- `src/utils/continual_metrics.py`
- `scripts/launch_lada_10task_eval.sh`
- `scripts/monitor_lada_10task_eval.sh`
- `scripts/extract_alpha_sweep.py`
- `experiments/alpha_sweep_smoke.log`
