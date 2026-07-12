# 相关工作结构讨论

**日期**: 2026-07-12  
**状态**: 讨论基线，尚未形成正式正文  
**目标**: 固定相关工作的主题边界、文献组织方式以及本文定位。

---

## 1. 组织原则

相关工作不是按关键词罗列论文，而是建立三条最终汇聚到本文的研究线：视觉语言模型持续学习、知识保持型参数高效适配、结合预训练先验的增量预测。

每个小节采用相同结构：研究问题 → 代表性方法类别 → 与本文最接近的方法 → 尚未解决的缺口 → 本文局部定位。最后增加一段总体定位，但不重复引言贡献列表。

## 2. 推荐小节

### 2.1 Class-Incremental Learning with Pre-trained Vision-Language Models

需要覆盖：

- 传统类增量学习中的主要方法谱系：replay/rehearsal、knowledge distillation and regularization、parameter isolation/dynamic architectures，以及 prototype/classifier correction；
- 代表性传统方法只选取能建立问题演进的工作，例如 iCaRL/LUCIR/PODNet 类表示与蒸馏方法、EWC/LwF 类正则化方法、DER 类回放方法，以及 BiC/WA 等分类器偏置校正方法；最终名单按引用空间和实验相关性收缩；
- 传统 CIL 的核心目标是吸收新类别并维持旧类判别，同时处理旧类/新类数据不平衡和分类器偏置；
- 从传统 CIL 到预训练 VLM 持续学习的任务变化：除历史类别性能外，还需要保留开放词汇语义、零样本泛化和跨模态对齐；
- CLIP 持续学习中的分类适配与零样本/跨模态能力退化；
- X-TAIL 或本文实际采用的类增量协议；
- LADA、ZSCL、Mod-X、RAIL、C-CLIP、MG-CLIP 等直接相关方法；
- 对 LADA 不能只概括为 label-specific adapter：应准确说明其冻结视觉编码器、用 AdaptFormer 持续适配文本侧、以类别聚类中心构成 label-specific visual memory，并在推理时融合 visual-memory 与 seen-tuned/unseen-frozen 文本分数。

建议采用三段式：

1. **传统 CIL**：按方法机制概括旧知识保持和分类器偏置处理，不逐篇罗列；
2. **预训练模型带来的变化**：说明强预训练模型使 continual learning 从“旧类—新类平衡”扩展为“新知识学习—历史知识保持—预训练能力保持”；
3. **CLIP/VLM CIL**：介绍直接相关方法与协议，并指出跨模态能力成为额外评价维度。

本节重点回答：传统方法如何保持历史类别，预训练 VLM 为何增加了新的保持对象，以及现有 VLM 方法如何处理新知识学习和预训练能力保持。

本文定位：继承传统 CIL 的历史任务保持目标，同时将 CLIP 的跨模态能力视为独立的保持对象；LoRA-NF 主要服务于历史任务保持，跨模态蒸馏主要服务于图文关系保持。

### 2.2 Parameter-Efficient Adaptation and Knowledge Preservation

建议分成两层：

1. LoRA、adapter、prompt 等参数高效适配方法；
2. 正则化、蒸馏、梯度/子空间投影及零空间感知 LoRA 等知识保护方法。

需要重点讨论：

- 参数高效不自动等价于不遗忘；
- LoRA 限制更新秩，但不限制适配发生在哪些历史激活方向；
- 子空间或梯度投影方法在反向更新阶段保护旧知识的基本思想；
- LoRA-Null 如何通过预训练激活的近似零空间初始化 LoRA，以及 initialization-only 的作用边界；
- 模态内特征蒸馏与跨模态关系保持之间的区别。

本文定位：LoRA-NF 将历史激活谱诱导的近似零空间滤波器持续嵌入 LoRA 前向支路，把普通 (AB) 重参数化为 (PAB)。这不同于 post-backward gradient projection，也不同于 LoRA-Null 的 initialization-only bias。跨模态蒸馏进一步补充图文关系层面的保持。

### 2.3 Class-Incremental Prediction with Pre-trained Priors

需要覆盖：

- 原型、最近类均值和统计判别分类器在增量学习中的使用；
- LDA/QDA/GDA 与少样本条件下的协方差正则化；
- 低秩统计建模的计算与估计动机；
- CLIP 文本分类器以及监督分支与语言先验分支的融合；
- LADA 的标签记忆/文本双分支作为最直接对照：其监督视觉分支基于 prototype-memory affinity，而本文监督视觉分支基于正则化类别条件分布判别。

本文定位：LR-RGDA 从适配后的视觉空间提取类别均值、低秩正则协方差和多中心统计，CLIP 文本分支提供语言语义先验。二者融合共享 LADA 所揭示的“监督视觉证据 + 文本先验”原则，但以 distributional discriminant modeling 替代 label-specific memory affinity。相关工作必须把这是“同一高层原则下的不同分类器设计”讲清楚，不能暗示 LADA 没有监督视觉分支。

## 3. 不建议保留的旧结构

- 不单列宽泛的 Continual Learning Settings；必要任务定义放到方法预备知识；
- 不把 Inference-Time Adaptation 作为独立主线，除非相关方法确实与本文预测集成直接可比；
- 不再围绕 OOD detection、adaptive routing 或 statistical replay 展开；
- 不使用“现有工作都忽略……”之类难以证明的绝对表述。

## 4. 文献收集清单

- [ ] CLIP/VLM continual learning 的最新任务定义与综述性来源；
- [ ] 传统 CIL 每个方法族选择一至两个最有代表性且与本文问题相关的引用，避免形成历史综述；
- [x] LADA 与本文最接近部分的机制已核验；最终实验协议与数值仍需对齐；
- [ ] ZSCL、Mod-X、RAIL、C-CLIP、MG-CLIP 的问题、方法和评估目标；
- [ ] LoRA 及面向持续学习的 LoRA 变体；
- [ ] OGD/GPM/子空间投影类代表工作及其与本文投影对象的区别；
- [ ] LoRA-Null 的初始化公式、校准数据、训练后子空间漂移边界及可比实验；
- [ ] 图文对齐或关系蒸馏代表工作；
- [ ] 增量学习中的 GDA、协方差正则化和低秩统计分类器；
- [ ] 监督分类器与零样本 CLIP 融合的直接先例。

## 5. 待讨论问题

- [ ] 是否把跨模态蒸馏并入第二节，还是单列一个短小节；
- [ ] LR-RGDA 有无足够直接相关文献支撑独立第三节；
- [ ] 哪些工作与本文是实验基线，哪些只用于概念定位；
- [ ] 是否需要一张 related-work comparison table；
- [ ] 最终投稿版面允许相关工作占用多少篇幅。


