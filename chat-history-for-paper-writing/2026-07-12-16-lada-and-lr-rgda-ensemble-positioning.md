# LADA 与 LR-RGDA/文本分类器集成的叙事定位

**日期**: 2026-07-12  
**状态**: 已用于修订引言初稿  
**会话概况**: 阅读 LADA 原论文的方法、推理流程、消融与附录文本适配细节，重新确定 LR-RGDA + CLIP 文本分类器在本文中的贡献地位。

**叙事层级更新**: 本文保留 LR-RGDA 集成作为重要预测组件，但不在引言中以 LADA 为动机展开第二条故事线。本文档中的详细 LADA 对比主要服务于 Related Work、Experiments 和公平性分析；引言仅简述监督视觉统计与文本语义先验的互补性。

---

## 1. LADA 实际使用的预测结构

LADA 不是单一视觉 adapter。其强性能来自文本侧、视觉侧和融合预测的组合。

### 1.1 文本侧

- 冻结 CLIP 图像编码器；
- 使用 AdaptFormer 对文本编码器进行参数高效微调；
- 当前任务的类别文本特征通过当前图像样本优化；
- 历史类别文本特征通过保留的图像原型参与训练，并在后续任务中缓存/冻结；
- 推理时，seen classes 使用已经适配的文本特征，unseen classes 使用 vanilla text encoder 产生的文本特征。

因此，LADA 的所谓文本/零样本分支并非对所有类别都使用完全冻结的 vanilla zero-shot classifier，而是 seen-tuned/unseen-frozen 的混合文本分类器。

### 1.2 视觉侧

- 每类通过图像特征聚类得到多个 label-specific memory units；
- memory units 与图像特征计算相似度，经过非负变换和聚合形成类别 logits；
- 历史 memory units 冻结，只更新当前任务部分；
- DPT 使用 GMM 描述历史类别原型分布并生成增强特征，帮助训练新任务 memory units。

### 1.3 融合

- 训练阶段将文本分支 logits 与 LADA visual-memory logits 相加后优化；
- 推理阶段先形成包含 seen adapted text features 和 unseen vanilla text features 的统一文本分类空间；
- 对 unseen prediction 使用文本结果；对 seen classes 线性融合 LADA logits 与相应文本 logits；
- LADA 的消融表显示 basic text fine-tuning 已经很强，而 LADA visual branch 与 DPT 继续提升 Transfer/Average/Last。

## 2. 我们与 LADA 的共同原则

两者都认识到，强 X-TAIL 分类不能只依赖单一分支：

- 文本/零样本分支提供语言语义和未来类别识别能力；
- 监督视觉分支利用已见类别数据提高细粒度判别；
- 融合在 forward transfer 与 seen-class performance 之间取得平衡。

因此，本文不应把 ensemble classifier 描述为普通工程技巧。它是与 LADA 正面对比所必需的预测端设计，也是达到强持续分类性能的重要组成。

## 3. 我们与 LADA 的关键差异

### 3.1 视觉表示学习

- LADA 冻结 CLIP 图像编码器，在其输出后增加 label-specific adapter；
- 本文通过 LoRA-NF 持续适配编码器表示，并以跨模态蒸馏约束图文关系。

### 3.2 监督分类分支

- LADA：label-specific prototype/memory affinity，近似类别原型相似度判别；
- 本文：LR-RGDA 类条件分布判别，显式利用均值、正则化低秩协方差、全局协方差与多中心结构。

核心区别不是“我们有监督、LADA 无监督”，因为二者都是监督视觉分支；区别是 prototype-affinity memory versus distributional discriminant modeling。

### 3.3 与文本分支的融合

- LADA：seen/unseen 逻辑加 seen-class 线性融合；
- 本文：CLIP text logits 提供全类别分数，LR-RGDA 为已建模类别提供监督分数，再在统一 logit 空间进行 soft fusion；
- 本文应强调无需把 LR-RGDA 解释为 OOD detector 或 hard router。

### 3.4 能力边界

- LR-RGDA + 文本分类器只改变分类决策，不能提高或保持图文检索；
- 图文检索保持必须归因于训练后的表示及跨模态蒸馏；
- ensemble 对论文主张的贡献是 SOTA 级持续/零样本分类，而不是跨模态能力保持。

## 4. 推荐的论文叙述

引言中应把预测端问题写成：

> Even with a well-preserved representation, continual classification must reconcile supervised visual evidence from seen classes with CLIP's language-derived prior over both seen and unseen classes. LADA demonstrates the importance of this principle through label-specific visual memory and adapted text classifiers. We pursue the same high-level objective through a different statistical route: LR-RGDA models class-conditional distributions in the adapted feature space, and its discriminant scores are fused with the CLIP text classifier.

对应的贡献表述：

> We introduce an ensemble classifier that combines LR-RGDA class-conditional visual statistics with the CLIP text classifier, providing a distributional alternative to LADA's label-specific visual-memory fusion.

## 5. 必须澄清的实现—叙事问题

当前代码 `text_classifier_mode=lada_hybrid` 默认使用：

- seen classes：缓存的适配文本特征；
- unseen classes：冻结 CLIP 的 vanilla 文本特征。

因此最终论文必须在以下两条路线中选择并保持一致：

1. **Matched LADA-style text branch**：承认使用 hybrid text classifier，把创新聚焦在 LoRA-NF 表示和 LR-RGDA 分布判别分支；
2. **Pure frozen zero-shot branch**：所有类别都用 frozen CLIP text features，更容易声称“zero-shot classifier”，但必须由对应主实验结果支持。

在该选择完成前，正文优先使用 `CLIP text classifier`，不要笼统称为 `frozen zero-shot classifier`。

## 6. 实验要求

- [ ] LADA basic text fine-tuning、LADA visual branch、LADA full method的同协议结果；
- [ ] CLIP text only、LR-RGDA only、LR-RGDA + text ensemble；
- [ ] LADA memory only、LADA + text，与 LR-RGDA 对应组合公平比较；
- [ ] pure frozen text 与 LADA-hybrid text 两种分支消融；
- [ ] 相同 encoder features、相同历史统计来源和相同 fusion normalization 下比较 LADA 与 LR-RGDA；
- [ ] 参数、存储和推理开销比较；
- [ ] 明确分类结果与图文检索结果的归因边界。

## 7. 相关材料

- `paper_writing/reference_papers/25-ICML-LADA Scalable Label-Specific CLIP Adapter for Continual Learning.pdf`
- `src/classifiers/lr_rgda_classifier.py`
- `src/classifiers/gaussian_classifier.py`
- `main_incremental.py`
- `chat-history-for-paper-writing/2026-07-12-15-introduction-first-draft.md`
