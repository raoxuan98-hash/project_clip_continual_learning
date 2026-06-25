# Context Reading and Discussion Prep

**日期**: 2026-06-16
**会话概况**: 阅读 project_clip_continual_learning 的项目文档、AGENTS、chat-history 最新记录、实验状态文件和当前未提交代码，整理用于后续讨论的上下文。

---

## 1. 关键讨论 / 决策

- 本次未修改实验代码，只做上下文阅读和语法检查。
- 当前讨论应以后续 chat-history 和 publication gate 文件为准；早期 PROJECT_DOCUMENTATION.md / paper_draft_README.md 中关于 OOD routing 的表述已被后续记录修正，不应直接作为当前主张。
- 论文主线应区分训练端和推理端两层：训练端 LoRA-NSP + vision/text + FD/CD + text schedule；推理端 LR-RGDA + ZS under compact statistical replay。

## 2. 重要发现

- `chat-history/2026-06-14-64-text-schedule-current-progress-and-best-settings.md` 给出训练端当前最佳候选：`text_tuning_schedule=low_lr_after`，Task 1 文本 LR 为 `1e-4`，Task 2+ 为 `2e-5`。该结论目前只支持 pilot/stress-test 级别，尚不能作为 10-task/3-seed 论文级结论。
- `chat-history/2026-06-14-57-fixed-alpha-gmm-mean-results.md` 和 `experiments/README_publication_repro.md` 给出推理端当前最稳证据：GMM component mean replay + fixed alpha=0.05 下，LR-RGDA+ZS 优于 LADA/LADA+ZS；但 real 16-shot feature 边界下 LADA 更强。
- `chat-history/2026-06-16-02-num-centers-ablation-results.md` 显示当前增量集成最优配置是 `num_centers=4 + GMM mean replay + rgda_train_iter=200`，Ensemble 达到 Transfer 72.30 / Average 81.57 / Last 82.05，Last 距 LADA paper 83.3 仍差 1.25。
- `experiments/publication_gate_status.json` 当前为 PASS=5, PENDING=2。两个 pending 是训练端 LoRA-NSP ablation 和 strict incremental metrics。
- 本地工作区已有未提交修改：`main_incremental.py`, `src/utils/main_utils.py`，以及新增 docs/chat-history/script 文件。不要误以为这是干净基线。
- `main_incremental.py` 已集成训练端调度和推理端多中心/GMM replay/分类器微调；`--enable_lada` CLI 已存在，但按 6/16 记录完整 LADA 增量逻辑仍待接。

## 3. 待办事项 / 遗留问题

- [ ] 决定下一步优先跑 full 10-task/3-seed gate，还是先修/接 LADA 增量对比逻辑。
- [ ] 若要写论文，先同步删除或改写旧文档中 adaptive routing / OOD detection 相关过时表述。
- [ ] 完成 publication gates 中两个 pending 项：训练端 ablation summary 和 strict incremental summary/aggregate。
- [ ] 处理本地未提交变更：审查后提交，或明确哪些是临时实验改动。

## 4. 相关文件

- `AGENTS.md`: 项目工作约定和当前高层设定。
- `chat-history/2026-06-14-64-text-schedule-current-progress-and-best-settings.md`: 训练端文本调度总结。
- `chat-history/2026-06-14-57-fixed-alpha-gmm-mean-results.md`: 推理端 fixed-alpha GMM mean replay 结果。
- `chat-history/2026-06-16-01-main-incremental-integration.md`: `main_incremental.py` 集成记录。
- `chat-history/2026-06-16-02-num-centers-ablation-results.md`: 多中心/GMM/fit 增量结果。
- `experiments/README_publication_repro.md`: 当前论文证据边界和复现实验说明。
- `experiments/publication_gate_status.json`: 当前 publication gate 状态。
- `main_incremental.py`: 当前集成入口。
- `src/utils/main_utils.py`: LADA 指标计算修复处。
