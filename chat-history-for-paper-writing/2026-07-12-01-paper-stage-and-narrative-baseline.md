# 论文阶段与叙事基线

**日期**: 2026-07-12
**会话概况**: 明确项目已进入“算法固定、论文启动、大规模实验实施”阶段，并区分工程历史、论文叙事历史与论文素材目录的职责。

---

## 1. 当前论文主叙事

- 研究目标：在保持 CLIP 原有跨模态检索能力的同时，实现 SOTA 级别的 CLIP 类增量持续学习。
- 训练端主创新：LoRA-NSP，需要建立清晰、可验证的理论基础。
- 训练端辅助设计：跨模态知识蒸馏，用于保持跨模态对齐与检索能力。
- 预测端创新：LR-RGDA 与零样本分类器组成的集成分类器。
- 旧版 `PRESENTATION.md`、`PROJECT_DOCUMENTATION.md` 与早期论文草稿只能作为历史材料，不能继续充当当前论文故事的权威来源。

## 2. 目录职责约定

- `chat-history/`：记录代码变动、实验过程、实验结果、调试结论和项目元构思。
- `chat-history-for-paper-writing/`：记录论文故事、贡献表述、理论叙事和随实验结果发生的叙述版本变化。
- `paper_writing/my_original_papers/`：作为作者语言、叙述和公式风格的样本库。
- `paper_writing/reference_papers/`：保存关键参考文献。
- `paper_writing/paper-template/`：保存本论文实际使用的 LaTeX 工程。
- `paper_writing/` 中的早期草稿与说明：作为历史输入，后续需要标明状态，避免与当前叙事混淆。

## 3. 项目整理原则

- 先建立目录说明、当前叙事基线与实验事实索引，再移动或删除中间文件。
- 任何脚本在归档前都需要检查是否被配置、启动脚本、文档或结果记录引用。
- 代码清理与论文内容整理分开进行，避免清理工作改变实验可复现性。
- 论文中的结论必须能追溯到配置、日志、结果文件和对应代码版本。

## 4. 当前观察到的问题

- 根目录混有主入口、调试脚本、评估脚本、备份文件和日志。
- 存在 `.bak*`、`._*`、`__pycache__`、LaTeX 辅助文件等生成或临时产物。
- `src/detectors/`、`src/routing/` 仍保留空壳，与既有文档“已删除”的描述不完全一致。
- `chat-history/` 存在同日重复序号和无两位序号的文件，后续检索需要索引而非立即批量重命名。
- 最新历史文件出现疑似编码乱码，需要从原始日志或正确编码源恢复，不能直接作为论文事实来源。
- 当前工作目录没有可用的 `.git` 元数据，暂时无法执行项目要求的提交与推送流程。

## 5. 本次已完成

- [x] 为两个历史目录分别补充 README，明确记录边界、命名规范和交叉引用规则。
- [x] 建立唯一的当前论文叙事文件，并保留叙事变更规则。
- [x] 建立实验 claim-to-evidence ledger，将论文主张映射到结果、配置、日志和代码版本。
- [x] 建立工程重组审计清单，但未移动或删除任何代码。
- [x] 在 `AGENTS.md` 中补充论文历史目录及双向记录规则。

## 6. 后续建议

- [ ] 对根目录、`src/`、`scripts/`、`docs/` 做引用审计后分批归档。
- [ ] 恢复 Git 元数据或确认真正的 Git 工作树位置，然后再提交整理改动。
- [ ] 从服务器原始结果建立最终主表，并逐项更新 claim-evidence ledger。
- [ ] 从实际 LoRA-NSP 代码重新构建理论定义、假设和定理。

## 7. 相关文件

- `AGENTS.md`：现有项目协作规则与旧版项目概览。
- `chat-history/README.md`：工程与实验历史记录规范。
- `chat-history-for-paper-writing/README.md`：论文历史记录规范。
- `chat-history-for-paper-writing/2026-07-12-02-current-narrative.md`：当前论文叙事。
- `chat-history-for-paper-writing/2026-07-12-03-claim-evidence-ledger.md`：论文主张与证据映射。
- `docs/project_reorganization_audit.md`：非破坏性的工程整理审计清单。
- `paper_writing/paper-template/paper_draft.tex`：当前 LaTeX 草稿，但仍含旧叙事章节。
- `paper_writing/paper_draft_README.md`：早期论文结构说明。
- `chat-history/2026-07-12-01-v5-completion-summary.md`：最新实验汇总，但本地文本疑似存在编码问题。
