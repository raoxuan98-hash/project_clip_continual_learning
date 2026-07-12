# 运行产物目录统一

**日期**: 2026-07-12
**会话概况**: 将分散在项目顶层的工具会话、日志、历史结果、生成图表和备份统一迁入 `artifacts/`，并同步修改脚本与文档引用。

---

## 1. 关键讨论 / 决策

- `configs/`、`docs/` 和 `demos/` 属于项目资产，保留在顶层。
- 可再生成或仅供本地追溯的产物统一放入 `artifacts/`。
- 正式论文图片应放在 `paper_writing/paper-template/figures/`，不与分析脚本生成图混放。
- 实验日志和结果不因整理而删除，以免破坏论文证据追溯。

## 2. 目录迁移

- `.arbor/` → `artifacts/tool-runs/arbor/`。
- `logs/` → `artifacts/logs/`。
- `results/` → `artifacts/results/`。
- `figures/` 中全部文件 → `artifacts/figures/`。
- `archive/` → `artifacts/archive/`。
- 删除 `artifacts/archive/os_metadata/` 中的 macOS `._*` 元数据。

## 3. 代码与文档更新

- 10-task 重消融和 NSP 启动脚本的根日志路径改为 `artifacts/logs/`。
- `docs/arbor_ablation_results.md` 的原始会话引用改为新路径。
- `.gitignore` 改为忽略 `artifacts/` 下的生成内容。
- 新增 `artifacts/README.md`，定义各子目录职责。
- 更新 `AGENTS.md` 和项目重组审计文档。

## 4. 验证结果

- 顶层 `.arbor/`、`archive/`、`logs/`、`results/` 已不存在。
- shell 脚本中没有直接写入根 `logs/` 的路径残留。
- 文档中没有仍指向旧 `.arbor/sessions/` 的有效引用。
- 原 `figures/` 内容已全部迁移，但空目录被外部进程占用，暂时无法删除。
- `.agents/` 与 `.git/` 删除后被工作区宿主自动创建为空目录，属于环境占位符，不应存放项目内容。

## 5. 待办事项 / 遗留问题

- [ ] 外部进程释放 `figures/` 后删除该空目录。
- [ ] 在远程 Linux 项目同步相同目录结构并验证修改后的启动脚本。
- [ ] 决定论文最终结果包是否从 `artifacts/results/` 提炼到单独可追溯目录。

## 6. 相关文件

- `artifacts/README.md`
- `.gitignore`
- `docs/project_reorganization_audit.md`
- `docs/arbor_ablation_results.md`
