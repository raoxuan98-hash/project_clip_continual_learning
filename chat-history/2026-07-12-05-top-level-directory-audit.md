# 顶层目录分类审计

**日期**: 2026-07-12
**会话概况**: 审计 `.agents`、`.arbor`、`.git`、`archive`、`configs`、`demos`、`docs`、`figures`、`logs` 和 `results`，区分正式项目资产与中间运行产物。

---

## 1. 关键分类

- 正式项目资产：`configs/`、`docs/`。
- 辅助但可保留的项目资产：`demos/`。
- 工具会话中间产物：`.arbor/`，约 23 MB、210 个文件。
- 实验运行产物：`logs/`，约 191 MB、97 个文件。
- 零散实验结果：`results/`，当前仅两个实验目录。
- 生成图表：`figures/`，当前为 alpha sensitivity 输出，不是正式论文图目录。
- 历史隔离产物：`archive/`，包含备份与 macOS 元数据。
- 空目录：`.agents/`、`.git/`；其中空 `.git/` 不能提供版本控制能力。

## 2. 引用审计

- 多个实验启动脚本直接向根目录 `logs/` 写入日志，移动该目录需要批量更新脚本。
- `docs/arbor_ablation_results.md` 直接引用 `.arbor/` 中的原始会话报告。
- `configs/` 被代码、文档和实验命令广泛引用，必须保留原位。
- 当前 `figures/` 文件未被正式论文草稿直接引用，更接近生成分析产物。
- `results/` 中内容规模很小，属于 Arbor/历史实验遗留，不是统一正式结果库。

## 3. 建议目标结构

- 保留顶层：`configs/`、`docs/`、`demos/`。
- 新建 `artifacts/`，下设 `logs/`、`results/`、`figures/`、`tool-runs/arbor/` 和 `archive/`。
- 正式论文图片统一进入 `paper_writing/paper-template/figures/`，不与实验生成图混放。
- 删除无价值的 macOS `._*` 元数据；备份代码先进入 `artifacts/archive/code-backups/`，待 Git 恢复后再决定删除。
- `.agents/` 若持续为空可删除；`.git/` 应恢复为有效仓库或删除空壳后重新克隆，不能作为普通中间目录处理。

## 4. 风险与实施边界

- 合并 `logs/` 会影响大量 shell 启动器的硬编码路径，需要一次性批量迁移并在 Linux 验证。
- 移动 `.arbor/` 会使旧工具会话无法原位恢复，应先确认只需要保留报告而不需要 resume。
- 实验日志和结果可能是论文证据，不能仅因体积大而删除。

## 5. 待办事项

- [ ] 确认是否采用统一 `artifacts/` 结构。
- [ ] 确认旧 Arbor session 是否还需要 resume。
- [ ] 确认 `logs/10task_reablation/` 是否已由服务器原始结果完整备份。
- [ ] 恢复有效 Git 仓库后再清除代码备份。
