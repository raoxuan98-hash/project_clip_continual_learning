# 根目录代码收敛

**日期**: 2026-07-12
**会话概况**: 按用户要求将项目根目录的 Python 代码入口收敛为 `main_incremental.py` 和 `main_joint.py`，其余调试、测试、演示、旧入口及共享数据工具迁入职责明确的目录。

---

## 1. 关键讨论 / 决策

- 根目录只保留两个正式 Python 入口：`main_incremental.py` 与 `main_joint.py`。
- 有复现价值的旧代码不直接删除，而是迁入 `scripts/legacy/`。
- 手工诊断脚本迁入 `scripts/debug/`，研究测试迁入 `tests/`，notebook 迁入 `demos/`。
- `utils_data.py` 属于共享运行代码，迁入 `src/utils/data.py`，而不是归档。
- macOS 元数据、备份和根目录日志分别迁入 `archive/` 与 `logs/legacy_root/`。

## 2. 代码与路径变动

- `utils_data.py` → `src/utils/data.py`，所有项目内 Python 导入更新为 `src.utils.data`。
- `main.py`、`main_incremental_lada.py` → `scripts/legacy/`。
- 三个根目录兼容评估入口 → `scripts/legacy/`；规范实现仍位于 `scripts/`。
- `debug_*.py`、`quick_alpha_sweep.py` → `scripts/debug/`。
- `test_basis_variants.py`、`test_dora_init_fix.py` → `tests/`。
- `demo_ood.ipynb` → `demos/`。
- `scripts/launch_phase3_gpu2.sh` 和 `scripts/run_lada_dpt_ablation.sh` 已更新旧入口路径。
- 移动后的直接执行脚本增加项目根路径初始化，保持 `src` 导入能力。

## 3. 验证结果

- 根目录 `.py`/`.ipynb` 检查仅剩 `main_incremental.py` 和 `main_joint.py`。
- 项目内不存在 `from utils_data` 或 `import utils_data` 残留。
- 启动脚本和非历史文档中不存在旧根入口调用残留。
- 17 个受影响 Python 文件通过静态语法编译检查。
- 当前 Windows 环境没有 Bash，无法在本地执行 shell 语法检查。
- `src.utils.data` 运行时导入被本地缺少 `scenario_datasets/` 阻断；这是迁移前已有的外部/缺失项目依赖问题。

## 4. 待办事项 / 遗留问题

- [ ] 确认远程服务器项目包含 `scenario_datasets/`，或明确其安装/同步来源。
- [ ] 在 Linux 服务器运行两个修改过路径的 shell 启动器的语法检查。
- [ ] 恢复有效 Git 元数据并提交本次迁移。
- [ ] 继续审计 `src/models/trainer.py`、顶层 `models/` 与 `utils/` 的旧版重复关系。

## 5. 相关文件

- `src/utils/data.py`
- `scripts/README.md`
- `scripts/debug/README.md`
- `scripts/legacy/README.md`
- `tests/README.md`
- `demos/README.md`
- `docs/project_reorganization_audit.md`
