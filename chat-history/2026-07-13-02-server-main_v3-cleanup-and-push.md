# 服务器 main_v3 分支清理与推送

**日期**: 2026-07-13
**会话概况**: 登录服务器完成 main_v3 分支推送，并将工作区中已暂存的清理改动（删除 Arbor 会话、实验中间产物、备份文件，整理 scripts/debug 与 scripts/legacy，补充 chat-history/meta-prompts）提交推送到 GitHub。

---

## 1. 关键讨论 / 决策

- **推送 main_v3 到 GitHub**: 使用服务器上的私钥通过 SSH 登录，执行 `git push -u origin main_v3`，成功将本地 main_v3 分支上传到远程。
- **清理并提交**: 服务器工作区已暂存大量删除/移动/新增文件，直接作为一个清理提交推送到 main_v3，避免分支继续携带 Arbor 会话、备份文件、旧结果图等本地不需要的内容。
- **保留服务器上的实验输出**: `experiments/`、`optimization/` 等目录仍被 `.gitignore` 排除，本次未从服务器文件系统中删除这些结果目录。

## 2. 重要发现

- 服务器工作区中 `.arbor/sessions/` 等目录曾被 Git 跟踪，导致 main_v3 分支体积变大；提交后已将其从版本库中移除。
- 当前 `.gitignore` 已覆盖 `experiments/`、`optimization/`、`artifacts/tool-runs/`、`.workbuddy/`、`*.pdf`、`*.aux`、`*.out` 等本地/中间产物。
- 检查确认：暂存后的跟踪文件中不再有被 `.gitignore` 忽略的文件。

## 3. 待办事项 / 遗留问题

- [ ] 在 GitHub 上确认 main_v3 最新提交内容是否符合预期。
- [ ] 如果后续仍需进一步清理分支（例如 coordinator/* 临时分支），可单独处理。

## 4. 相关文件

- `.gitignore`: 已更新忽略规则。
- `chat-history/`: 新增多条论文写作与工程对话记录。
- `scripts/debug/`、`scripts/legacy/`: 原根目录下的调试/遗留脚本已归集。
- `tests/`、`demos/`、`meta-prompts/`、`artifacts/README.md` 等：新增或整理。
