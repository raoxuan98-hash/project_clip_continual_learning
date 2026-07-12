# 第一阶段代码整理

**日期**: 2026-07-12
**会话概况**: 对项目入口层进行引用与哈希审计，将三个根目录重复工具收敛为兼容入口，并明确 `scripts/` 为唯一维护位置。

---

## 1. 关键讨论 / 决策

- 代码整理以实验可复现性为优先，不直接删除历史启动器或核心入口。
- 正式代码和启动脚本已经引用 `scripts/` 下的评估实现。
- 根目录的三个同名工具与 `scripts/` 版本字节级完全重复，因此保留旧路径兼容入口即可。
- `scripts/` 是实验启动、评估、汇总、审计和远程运行工具的规范位置。

## 2. 代码变动

- `evaluate_incremental_artifacts.py` 改为调用 `scripts.evaluate_incremental_artifacts.main`。
- `evaluate_incremental_rgda_sweep_artifacts.py` 改为调用 `scripts.evaluate_incremental_rgda_sweep_artifacts.main`。
- `merge_incremental_async_results.py` 改为调用 `scripts.merge_incremental_async_results.main`。
- 新增 `scripts/README.md`，说明可复用入口和脚本分类。
- `.gitignore` 新增 `*.bak*`，覆盖项目现存的多种备份后缀。

## 3. 验证结果

- 三个兼容入口通过 Python 静态语法编译检查。
- 三个规范实现均存在公开 `main()` 函数。
- 根目录重复实现由约 56 KB 减少为约 1 KB 兼容代码。
- 未改动核心算法、实验配置或 `scripts/` 下的规范实现。

## 4. 重要发现

- 根目录仍混有 `debug_*.py`、临时测试、日志、macOS `._*` 元数据和备份文件。
- `main.py`、`main_joint.py`、`main_incremental.py` 和 `main_incremental_lada.py` 的职责仍需进一步确认。
- `src/models/trainer.py`、根目录 `utils/` 与 `src/utils/` 可能属于旧版或重复实现，需要导入图审计。
- 本地 `.git` 是空目录，Git 无法识别工作树，因此本次变动尚不能提交。

## 5. 待办事项 / 遗留问题

- [ ] 建立正式入口到配置、脚本和输出的完整映射。
- [ ] 审计根目录调试与临时测试脚本的历史引用后再决定归档。
- [ ] 审计旧 trainer、重复 utils 和空 detectors/routing 包。
- [ ] 恢复有效 Git 元数据后提交本阶段变动。

## 6. 相关文件

- `scripts/README.md`
- `docs/project_reorganization_audit.md`
- `.gitignore`
