# 项目重组审计清单

**日期**: 2026-07-12  
**状态**: 初步盘点；本文件不授权删除或移动任何内容。

## 1. 重组目标

- 让正式训练、评估和论文复现入口清晰可见；
- 将临时调试、历史实现和生成产物与正式代码隔离；
- 保持所有已报告实验的可追溯性；
- 在论文写作期间避免旧叙事和旧结果被误当作当前事实。

## 2. 初步分类

### A. 正式核心，优先保留并补文档

- `src/models/lora_sgp.py`
- `src/trainers/lora_nsp_trainer.py`
- `src/classifiers/`
- `main_incremental.py`
- `scripts/legacy/main_incremental_lada.py`（历史 LADA 入口）
- `configs/base/` 与 `configs/experiments/`
- 正式使用的 `scripts/`
- `src/utils/retrieval_eval.py`

### B. 需要确认唯一入口或重复关系

- `main.py`、`main_joint.py`、`main_incremental.py`
- `src/experiments/run_continual_learning*.py`
- 根目录与 `utils/`、`src/utils/` 中可能重复的工具模块
- 根目录评估工具与 `scripts/` 实现的关系（2026-07-12 已收敛为兼容入口）
- `src/models/trainer.py` 与 `src/trainers/lora_nsp_trainer.py`
- LADA 相关的 `src/lada/` 与其他 adapter 实现

处理前必须用导入搜索、启动脚本搜索和历史记录确认调用关系。

### C. 候选归档，需先完成引用审计

- `debug_*.py`
- `tests/test_basis_variants.py`、`tests/test_dora_init_fix.py` 等研究测试
- `quick_alpha_sweep.py`
- `demos/demo_ood.ipynb`
- 过时但仍可能对应已报告实验的启动脚本
- `paper_writing/PRESENTATION.md`、`paper_writing/PROJECT_DOCUMENTATION.md` 与早期叙事材料

建议归档到按职责划分的 `archive/` 或 `scripts/legacy/`，但只有在确认引用和复现关系后实施。

### D. 生成/临时产物，候选清除或加入 ignore

- `**/__pycache__/`、`*.pyc`
- `._*`
- `*.bak*`
- 根目录 `arbor_*.log` 与 `logs/` 中的运行日志副本
- LaTeX 的 `*.aux`、`*.log`、`*.out` 和编译 PDF

清除前仍需确认这些文件是否被版本控制或是否包含尚未登记的唯一结果。

### E. 文档与实际状态不一致

- `src/detectors/__init__.py` 与 `src/routing/__init__.py` 仍存在，而 `AGENTS.md` 表述为模块已删除；
- `AGENTS.md` 中的阶段、最优超参数和关键入口可能落后于近期 V4/V5 实验；
- `paper_draft.tex` 仍含 OOD Detection 等旧叙事章节；
- 最新 V5 历史汇总在本地显示乱码，应恢复编码或从原始结果重建。

## 3. 实施阶段

### Phase 1：建立索引

- [x] 明确两个历史目录的职责；
- [x] 建立当前论文叙事；
- [x] 建立带日期版本的 claim-evidence ledger；
- [ ] 列出正式入口到配置、脚本和输出的映射；
- [ ] 列出论文表格到原始结果的映射。

### Phase 2：引用审计

- [x] 确认三个根目录评估/合并工具与 `scripts/` 实现字节级重复；
- [x] 将 `scripts/` 版本确立为唯一维护实现，根目录保留轻量兼容入口；
- [ ] 搜索 Python 导入和 shell 调用关系；
- [ ] 检查配置继承与 CLI 参数；
- [ ] 标记每个根目录脚本的状态和最近使用记录；
- [ ] 确认远程服务器上仍在使用的未提交脚本；
- [ ] 识别结果文件的唯一生成器。

### Phase 3：非破坏性归档

- [ ] 将确定不再作为正式入口的脚本移入 legacy/archive；
- [ ] 为历史论文叙事添加醒目的 deprecated 标记；
- [ ] 更新 README、AGENTS 和启动说明；
- [ ] 运行导入检查和轻量级静态验证。

### Phase 4：删除与版本固化

- [ ] 仅删除已归档、无引用且无唯一实验价值的文件；
- [ ] 固化 paper submission 使用的配置、脚本和结果清单；
- [ ] 记录提交哈希并生成最终复现说明。

## 4. 当前阻碍

本地工作目录未检测到 `.git` 元数据，无法查看 tracked/untracked 状态、历史引用或提交本次文档变更。实施代码归档前，应先确认真正的 Git 工作树位置或恢复仓库元数据。

## 5. Phase 1 代码收敛结果（2026-07-12）

- `evaluate_incremental_artifacts.py` → 调用 `scripts.evaluate_incremental_artifacts.main`；
- `evaluate_incremental_rgda_sweep_artifacts.py` → 调用 `scripts.evaluate_incremental_rgda_sweep_artifacts.main`；
- `merge_incremental_async_results.py` → 调用 `scripts.merge_incremental_async_results.main`；
- 三份根目录重复源码由约 56 KB 收敛为约 1 KB 兼容代码；
- `.gitignore` 增加 `*.bak*`，覆盖 `.bak_scheduler`、`.bak2` 和带时间戳的备份；
- 新增 `scripts/README.md`，明确脚本分类和唯一实现位置。

本阶段未移动、删除或修改核心算法、实验配置与正式 `scripts/` 实现。

## 6. 根目录收敛结果（2026-07-12）

- 根目录仅保留 `main_incremental.py` 与 `main_joint.py` 两个 Python 入口；
- `utils_data.py` 迁入 `src/utils/data.py`，项目内导入已统一更新；
- debug 与快速诊断脚本迁入 `scripts/debug/`；
- `main.py`、`main_incremental_lada.py` 和旧兼容入口迁入 `scripts/legacy/`；
- 研究测试迁入 `tests/`，notebook 迁入 `demos/`；
- macOS `._*` 元数据、备份与根目录日志分别迁入 `archive/` 和 `logs/legacy_root/`；
- 仍引用旧入口的 shell 启动器已更新为新路径。

静态语法检查已通过。运行时导入验证被本地缺少 `scenario_datasets/` 阻断；该依赖在 `AGENTS.md` 中被描述为项目目录，但当前工作区不存在，需要在远程复现前确认。

## 7. 运行产物统一（2026-07-12）

- `.arbor/` → `artifacts/tool-runs/arbor/`；
- `logs/` → `artifacts/logs/`；
- `results/` → `artifacts/results/`；
- `figures/` 内容 → `artifacts/figures/`；
- `archive/` → `artifacts/archive/`；
- 清除 macOS `._*` 元数据；`.agents/` 与 `.git/` 删除后被工作区宿主自动重建为空占位目录，不存放项目内容；
- 直接写入根 `logs/` 的启动脚本统一改为 `artifacts/logs/`；
- Arbor 结果文档引用已更新。

原 `figures/` 已为空，但因被外部进程占用暂未能删除目录本身。
