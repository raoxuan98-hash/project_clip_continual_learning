# LADA 与 LR-RGDA 论文定位核验

**日期**: 2026-07-12  
**会话概况**: 阅读 LADA 原论文并核对本项目分类器实现，确定 LR-RGDA 与 CLIP 文本分类器集成应作为独立预测端创新，而非普通后处理模块。

---

## 1. 关键讨论 / 决策

- LADA 的强性能来自冻结视觉编码器、文本侧持续适配、label-specific visual memory 和推理融合的组合，不能只将其描述为视觉 adapter。
- 本文与 LADA 共享“监督视觉证据 + 文本语义先验”的高层预测原则，但监督分支不同：LADA 使用 prototype/memory affinity，本文使用 LR-RGDA 类条件分布判别。
- LR-RGDA + CLIP 文本分类器是完整方法实现强持续分类性能的重要预测组件，但引言不将其扩展为与 LoRA-NF 平行的第二条主线。
- LADA 的详细对比移至相关工作与实验；引言只简要说明监督视觉统计和文本语义先验的互补性。
- 该集成只作用于分类决策，不能解释图文检索保持；检索保持仍由训练端编码器及跨模态蒸馏支撑。
- 当前默认文本分支是 seen-tuned/unseen-frozen 的 LADA-style hybrid。在最终配置固定前，论文统一称其为 `CLIP text classifier`，不称为完全冻结的 zero-shot classifier。
- 后续引言讨论进一步固定机制责任：LoRA-NF 主要负责历史任务保持，跨模态检索保持主要来自知识蒸馏；摘要和引言用零空间启发的 (A) 初始化/约束概括相关 LoRA 方法，不点名 LoRA-Null。

## 2. 重要发现

- LADA 的视觉分支以每类聚类中心构成标签记忆，并对历史记忆冻结；其文本分支对已见类使用适配后的文本特征、对未见类使用 vanilla CLIP 文本特征。
- 当前 `main_incremental.py` 默认 `text_classifier_mode="lada_hybrid"`，与“所有类别均使用冻结零样本文本权重”的叙述不同。
- 当前 LR-RGDA 实现使用类别均值、正则化低秩协方差、全局协方差和多中心结构，并将其分数与 CLIP 文本分数软融合。

## 3. 待办事项 / 遗留问题

- [ ] 固定最终文本分支：pure frozen zero-shot 或 LADA-style hybrid。
- [ ] 在相同 encoder features 和 matched text branch 下比较 LADA memory、LR-RGDA 及各自 ensemble。
- [ ] 汇总 CLIP-text only、LR-RGDA only 和 ensemble 的多种子结果与融合敏感性。

## 4. 相关文件

- `paper_writing/reference_papers/25-ICML-LADA Scalable Label-Specific CLIP Adapter for Continual Learning.pdf`: LADA 原论文。
- `src/classifiers/lr_rgda_classifier.py`: LR-RGDA 与文本分数融合实现。
- `main_incremental.py`: 当前 hybrid text classifier 配置和已见/未见类别文本权重逻辑。
- `chat-history-for-paper-writing/2026-07-12-16-lada-and-lr-rgda-ensemble-positioning.md`: 论文叙事详细分析。
- `chat-history-for-paper-writing/2026-07-12-15-introduction-first-draft.md`: 已修订的引言初稿。
