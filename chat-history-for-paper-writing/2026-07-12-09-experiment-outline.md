# 实验章节结构讨论

**日期**: 2026-07-12  
**状态**: 讨论基线，具体表格数字仍以 claim–evidence ledger 核验为准  
**目标**: 让每组实验对应一个明确研究问题，并规划主实验、消融、机制分析和附录结果。

---

## 1. 实验章需要回答的问题

1. 完整方法是否达到有竞争力或 SOTA 级别的 CLIP 类增量学习性能？
2. 持续学习后是否保持 CLIP 的图文检索能力？
3. LoRA-NF 是否在公平预算下优于标准 LoRA、梯度投影和 LoRA-Null-style 初始化？
4. 改进是否确实来自持续 (PAB) 前向过滤及其谱相关优化几何？
5. 跨模态蒸馏是否提供分类与检索上的辅助收益？
6. LR-RGDA 与 CLIP 文本分类器是否互补，并在控制相同文本分支后提供优于 LADA label-memory branch 的分布判别能力？
7. 完整方法是否形成更好的适应—保持联合平衡？

## 2. 推荐结构

### 4.1 Experimental Setup

应包含：

- 数据集、类别划分、任务顺序和 16-shot/full-shot 设置；
- 初始模型与参考图文数据；
- 对比方法及选择理由；
- Transfer、Average、Last 等持续学习指标的精确定义；
- 图文检索数据集、候选池和 R@K 指标；
- 随机种子、均值/方差和显著性处理；
- 公平性：训练预算、可训练参数、回放/外部数据和模型骨干；
- 实现细节只保留影响复现的关键项，其余放附录。

### 4.2 Main Results on Class-Incremental Learning

主表应与 LADA 等强基线使用可比协议。LADA 不应被当作单一 adapter 基线：比较时需明确其文本侧适配、label-specific visual memory、DPT 和推理融合是否完整启用。建议报告总体指标，并把逐数据集或逐任务结果放附录。

本节只回答完整方法是否有效，不在主表之后立即混入大量超参数分析。

对应 claim：C1。

### 4.3 Preserving Cross-Modal Retrieval

至少比较：

- 原始冻结 CLIP；
- 标准 LoRA；
- LoRA-NF；
- LoRA-NF + 蒸馏；
- 完整方法对应的训练后编码器。

需要明确评估发生在每个任务后还是最后任务后。若预测集成不参与检索，应明确完整方法的检索结果实际来自其训练端编码器。

对应 claim：C2、C5。

### 4.4 Adaptation–Preservation Trade-off

这是全文主线的联合证据。用同一模型同时展示持续学习指标和检索指标，形式可以是：

- 二维联合表；
- 以分类性能和检索保持率为坐标的散点/Pareto 图；
- 随任务推进的双指标曲线。

对应 claim：C7。

### 4.5 Component Ablations

建议累积链条：

1. Standard LoRA + ZS；
2. LoRA-NF + ZS；
3. LoRA-NF + FD；
4. LoRA-NF + FD + CD；
5. LoRA-NF + FD + CD + LR-RGDA Ensemble。

另做非累积拆分，避免无法区分 FD 与 CD 的独立作用。LoRA-NF 机制对照应至少包括标准 LoRA、gradient-projected LoRA、LoRA-Null-style initialization、strict/leaky/soft filtering。所有训练端消融同时报告分类和检索指标。

对应 claim：C3、C5。

### 4.6 Mechanism Analysis of LoRA-NF

候选分析：

- strict、leaky 与 soft 零空间滤波；
- 泄漏系数、零空间阈值或软权重超参数；
- LoRA rank；
- 有效更新在历史谱各方向上的能量；
- (\|XP\|)、历史 adapter 响应、表示漂移、图文相似度漂移或检索退化；
- 理论预测与观察结果的对应。

这一节承担“为什么有效”，不能仅重复准确率消融。

对应 claim：C4。

### 4.7 Analysis of Ensemble Prediction

必须分别报告：

- CLIP text classifier；
- LR-RGDA；
- Ensemble。

核心公平对照应包括：

- pure frozen CLIP text 与 seen-tuned/unseen-frozen hybrid text branch；
- LADA visual memory only、LADA + matched text branch；
- LR-RGDA only、LR-RGDA + 同一个 matched text branch；
- 在相同 encoder features、历史数据/统计来源及尽可能一致的分数归一化下比较 LADA memory 与 LR-RGDA。

进一步分析融合系数、LR-RGDA rank/正则参数、任务阶段和不同数据集上的稳定性。如果使用分数归一化或校准，也需要消融。该节需明确 ensemble 的收益属于分类决策，检索结果只由训练后编码器决定。

对应 claim：C6。

### 4.8 Efficiency and Robustness

候选内容：

- 可训练参数量、额外存储、训练开销和推理开销；
- 多随机种子；
- 不同任务顺序；
- 不同 shot 数；
- 关键超参数敏感性。

版面不足时，正文保留一张效率表和最关键的稳健性结论，其余移至附录。

## 3. 主文与附录分配

### 主文优先保留

- 持续学习主表；
- 跨模态检索主表；
- 适应—保持联合图/表；
- 核心组件消融；
- 一项最能验证 LoRA-NF 机制的分析；
- 三分支集成比较。

### 附录承接

- 逐数据集、逐任务和逐种子结果；
- 完整超参数敏感性；
- 额外任务顺序与 shot 设置；
- 完整效率与复杂度数据；
- 更多表示漂移和定性结果；
- 检索协议与样例。

## 4. 明确删除的旧实验线

- OOD detection performance；
- adaptive routing；
- 与当前方法无关的 statistical replay protocol；
- 不能追溯到当前代码和配置的占位结果。

## 5. 待讨论问题

- [ ] 主文以 16-shot 为主还是同时展示 full-shot；
- [ ] 检索选择哪些数据集以及在哪些任务节点评估；
- [ ] C7 是否有足够结果形成独立联合分析；
- [ ] LoRA-NF 的最佳机制指标是什么；
- [ ] baseline 是否全部能在严格相同协议下比较；
- [ ] 多种子和任务顺序实验的最低配置；
- [ ] 哪些表格必须等待服务器重跑后才能固定。


