# 论文大纲结构讨论稿

**日期**: 2026-07-12  
**状态**: 待讨论，尚未固定为最终大纲  
**会话概况**: 围绕当前论文主线，建立从摘要、引言、相关工作、方法、实验到附录的论证型大纲。本文档只固定章节功能和叙事顺序，不撰写正式正文。

---

## 1. 全文中心论证

论文不应被叙述为三个松散模块的组合，而应围绕同一个矛盾展开：监督式持续适配提高增量类别判别能力，却会侵蚀 CLIP 的预训练跨模态知识。本文分别在训练参数空间、跨模态表示空间和预测空间处理这一矛盾：

1. LoRA-NF 将 (AB) 重参数化为 (PAB)，通过持续前向零空间过滤抑制历史高能量方向上的有效学习；
2. 跨模态知识蒸馏从表示层面辅助维持图文对齐；
3. LR-RGDA 与零样本分类器集成，在推理时结合增量监督知识和预训练语义先验。

全文最终应由同一组模型同时报告持续学习与图文检索结果，形成“适应—保持”联合证据，而不是分别展示两个无关指标。

## 2. 推荐正文结构

### Abstract

按“背景与矛盾—现有缺口—方法—关键结果—结论”组织。摘要中的 SOTA、保持检索能力等强表述，必须等主表与检索结果核验后再确定强度。

### 1. Introduction

建议采用六段式论证：

1. CLIP 的跨模态预训练能力及持续适配的现实需求；
2. 核心矛盾：新类监督适配与预训练跨模态能力保持之间的冲突；
3. 现有方法缺口：主要从回放、正则化、提示或适配器等单侧处理遗忘，缺少对更新方向、跨模态关系和预测决策的协同设计；
4. 训练端方案：以 LoRA-NF 的持续前向零空间过滤为主，跨模态蒸馏为辅助；
5. 推理端方案：解释 LR-RGDA 与零样本分支为何互补；
6. 贡献列表：问题视角、LoRA-NF/理论、辅助蒸馏、集成分类器及系统实验。

引言需要尽早说明本文保留的“能力”具体指图文检索，而不是含混的通用零样本能力。

### 2. Related Work

建议保留三个主题小节：

1. Continual Learning with Vision-Language Models：建立任务谱系，重点定位 CLIP 类增量持续学习、X-TAIL、LADA、ZSCL、Mod-X、RAIL、C-CLIP、MG-CLIP；
2. Parameter-Efficient Adaptation and Knowledge Preservation：LoRA、adapter/prompt、子空间或梯度投影方法、蒸馏与跨模态对齐保持；
3. Class-Incremental Prediction with Pre-trained Priors：原型/统计分类器、GDA/正则化协方差、零样本分类器融合。

不建议单列宽泛的 Continual Learning Settings 或 Inference-Time Adaptation 综述。每个小节末尾只做局部定位，最后用一小段统一说明本文如何连接三条研究线。

### 3. Method

建议顺序：

1. Problem Setup and Overview：类增量任务、CLIP 图像/文本编码器、评估目标，以及训练端—推理端总览；
2. LoRA-NF for Knowledge-Preserving Adaptation：标准 LoRA 缺口、历史激活谱、零空间滤波器、(PAB) 前向重参数化与跨任务更新流程；
3. Why LoRA-NF Reduces Historical Interference：正文保留历史响应上界、可逆 (P) 下表达集合一致性和谱相关优化几何，完整证明放附录；
4. Cross-Modal Knowledge Distillation：明确 FD/CD 各自对象、参考数据与总损失，并定位为 LoRA-NF 的辅助表示约束；
5. Ensemble Prediction with LR-RGDA and Zero-Shot CLIP：类别统计、低秩正则协方差、判别分数、零样本分数及融合/校准；
6. Training and Inference Procedure：用算法框或简短流程收束，并交代存储与计算开销。

当前叙事不再包含 Adaptive Routing、OOD Detection 或 Statistical Replay。

### 4. Experiments

实验章按研究问题组织，而不是按已有脚本组织：

1. Experimental Setup：协议、数据集/任务顺序、shot、基线、公平预算、指标、实现细节和统计方式；
2. Main Results on Class-Incremental Learning：回答完整方法是否达到强竞争力或 SOTA；
3. Preserving Cross-Modal Retrieval：回答持续学习后图文检索是否得到保持；
4. Adaptation–Preservation Trade-off：同一模型联合展示持续学习和检索，最好包含二维表或 Pareto 图；
5. Component Ablations：按 Standard LoRA → LoRA-NF → +FD → +CD → +Ensemble 的累积链条，并补 LoRA-Null/梯度投影对照；
6. Mechanism Analysis of LoRA-NF：strict/leaky/soft 过滤、谱方向更新能量、历史响应、表示漂移与理论预测的对应；
7. Analysis of Ensemble Prediction：ZS、LR-RGDA、Ensemble 三分支，融合系数、低秩维数、任务阶段稳定性；
8. Efficiency and Robustness：参数量、存储、训练/推理开销，多种子、任务顺序或 shot 敏感性。版面不足时移附录。

主实验与消融必须分别承担“证明有效”和“解释为何有效”的功能。

### 5. Conclusion and Limitations

先回到适应—保持矛盾，总结训练端和预测端如何共同解决；再给出受证据支持的主要结论；最后诚实说明零空间估计、参考图文数据、类别统计估计以及当前协议范围等限制。避免重复摘要式罗列全部模块。

## 3. 推荐附录结构

1. Reproducibility Details：数据划分、任务顺序、超参数、训练与硬件；
2. Full Theoretical Results and Proofs：LoRA-NF 的定义、假设、命题和完整证明；
3. LR-RGDA Derivation and Complexity：判别函数、低秩计算与存储复杂度；
4. Additional Experimental Results：逐数据集/逐任务结果、完整多种子统计；
5. Extended Ablations and Sensitivity：投影、蒸馏、rank、融合系数；
6. Retrieval Evaluation Details：检索数据、候选池、R@K 与评估时点；
7. Pseudocode and Implementation Notes；
8. Broader Impact / Limitations / Checklist：按投稿模板要求安排。

## 4. 待讨论的结构性决策

- [ ] 方法名称与整套框架名称是否需要区分；
- [ ] 正文核心理论结果能否从当前实现严格推出；
- [ ] 跨模态检索使用哪些数据集、指标和评估时点；
- [ ] 主文是否同时容纳 16-shot 与 full-shot，还是把 full-shot 移到附录；
- [ ] 是否有足够结果支撑独立的 Adaptation–Preservation Trade-off 小节；
- [ ] 效率分析放正文还是附录；
- [ ] 最终贡献列表中是否把跨模态蒸馏列为独立贡献，或仅列为整体训练机制的一部分。

## 5. 与 Claim–Evidence Ledger 的对应

- 主实验对应 C1；
- 检索保持对应 C2；
- LoRA-NF 消融对应 C3；
- 理论与机制分析对应 C4；
- 蒸馏消融对应 C5；
- 集成分析对应 C6；
- 适应—保持联合分析对应 C7。

只有当相应 claim 达到可用状态时，摘要、引言贡献和结论中的表述才能固定。

