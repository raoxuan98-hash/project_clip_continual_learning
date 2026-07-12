# 方法章节结构讨论

**日期**: 2026-07-12  
**状态**: 讨论基线，数学定义与理论尚需按代码核验  
**目标**: 固定方法章节的模块层级、数学叙述顺序和理论边界。

---

## 1. 方法章的中心逻辑

方法章应沿着“训练时保护—表示上维持—预测时融合”展开。LoRA-NF 是方法与理论主创新，跨模态蒸馏是辅助训练机制，LR-RGDA/CLIP-text 集成是完整分类系统的重要预测组件。

理论不建议脱离方法成为独立一级章节。核心结论紧随 LoRA-NF，完整证明进入附录。

## 2. 推荐结构

### 3.1 Problem Setup and Framework Overview

定义：

- 任务序列、类别集合与数据可用性；
- CLIP 图像编码器、文本编码器和文本分类器，并明确已见/未见类别分别使用何种文本权重；
- 类增量学习输出空间；
- 图文检索作为预训练跨模态能力的评估目标；
- 训练阶段与推理阶段的整体数据流。

本节需要一张总框架图，但不在此处展开所有损失和分类器公式。

### 3.2 LoRA-NF for Knowledge-Preserving Adaptation

建议叙述顺序：

1. 标准 LoRA 参数化 (AB)，以及“低秩不等于方向受保护”的缺口；
2. 历史层输入 activation second-moment matrix 的定义；
3. 从历史激活谱识别低能量近似零空间与高能量受保护方向；
4. strict、leaky 和 soft 三类滤波算子 (P)；
5. 前向重参数化 (AB\mapsto PAB)，以及 (X\rightarrow XP\rightarrow XPA\rightarrow XPAB)；
6. 跨任务统计更新、LoRA 合并与重初始化流程。

必须从实际 `src/models/lora_sgp.py` 和训练器重新核验：投影对象、左右乘方向、统计来源、更新频率、软权重函数和跨任务累积方式。

### 3.3 Theoretical Understanding of LoRA-NF

正文理论只承担四项任务：

1. 建立历史响应上界，把 \(\|XP\|\) 与 adapter 对历史任务的干扰联系起来；
2. 证明可逆泄漏式 (P) 下 LoRA-NF 与同 rank LoRA 的表达集合一致；
3. 推导有效参数更新中的 (P^2) 谱预条件，解释高能量方向学习速度为何受抑制；
4. 给出可由实验验证的预测，例如谱方向更新能量、历史响应和表示漂移。

需要严格区分：精确零空间保护、一阶干扰抑制和经验能力保持。未从代码与假设推出的结论不得称为 theorem guarantee。

### 3.4 Cross-Modal Knowledge Distillation

定义并区分：

- 监督分类损失；
- feature distillation 的教师、学生、输入和距离；
- cross-modal distillation/contrastive distillation 的图文对象；
- 参考图文数据的使用协议；
- 总训练目标及权重。

叙事上明确分工：LoRA-NF 通过前向零空间过滤塑造历史方向上的优化几何，负责历史任务保持；蒸馏显式约束图文表示行为，是跨模态检索保持的主要机制。不要把检索保持写成 LoRA-NF 理论的自然推论。

### 3.5 Distributional Ensemble Prediction with LR-RGDA and the CLIP Text Classifier

建议顺序：

1. 从决策层缺口切入：已见类别需要监督视觉判别，未见类别仍依赖 CLIP 的语言先验；
2. 增量类别的均值、全局/类别协方差与多中心统计；
3. 少样本条件下正则化 GDA 与低秩协方差近似的动机；
4. LR-RGDA 判别分数；
5. CLIP 文本分类分数，并明确 pure-frozen 或 seen-tuned/unseen-frozen hybrid 的最终定义；
6. 两类分数的归一化、校准与软融合，以及 LR-RGDA 分数只覆盖已建模类别的处理；
7. 类别或任务增长时统计量如何更新。

方法章专注定义 LR-RGDA 及融合公式。与 LADA 的 label-memory affinity 对比放在相关工作和实验，不在本节展开。

必须明确 LR-RGDA 使用的特征来自哪个模型阶段，以及训练端变化后统计量是否重算。不得将该模块写成 OOD detector 或 hard router，也不得暗示它影响图文检索。

### 3.6 Overall Algorithm and Complexity

用算法框总结每个任务的顺序：统计收集 → 投影构造 → LoRA-NF/蒸馏训练 → 权重处理 → 类别统计更新 → 集成预测。

正文只给关键参数量、存储与复杂度；完整 LR-RGDA 推导、复杂度证明和伪代码放附录。

## 3. 图表规划

- Figure 1：问题动机或适应—保持冲突；
- Figure 2：训练端与预测端整体框架；
- Figure 3：LoRA、梯度投影、LoRA-Null 与 LoRA-NF 的机制对照，可视版面决定是否保留；
- Algorithm 1：完整持续学习流程；
- 一个 notation table 可放附录或方法开头。

## 4. 附录承接内容

- 完整数学假设、命题与证明；
- 标准 LoRA 与 LoRA-NF 的有效更新推导；
- LR-RGDA 判别函数与低秩计算推导；
- 训练和预测伪代码；
- 复杂度与存储细节；
- 实现参数及额外说明。

## 5. 待讨论问题

- [ ] 主文以 leaky LoRA-NF 为默认定义，还是先统一定义 strict/leaky/soft；
- [x] 方法正文保留历史响应界、可逆泄漏式 (P) 下的表达集合等价，以及 (P^2) 谱优化几何三个正式结论；完整证明放附录；
- [ ] LoRA-NF 是否包含蒸馏，还是二者在命名上保持分离；
- [ ] 整体算法是否存在统一名称；
- [x] LR-RGDA 集成是完整分类方法的重要预测组件；方法章保留完整定义，但不与 LoRA-NF 建立平行主线；
- [ ] 最终采用 pure-frozen 还是 LADA-style hybrid CLIP text branch；
- [ ] 分数融合采用固定系数是否需要概率校准解释；
- [ ] 当前草稿中的旧定理与复杂度推导哪些能够保留。


