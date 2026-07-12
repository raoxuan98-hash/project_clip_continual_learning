# 引言机制责任划分

**日期**: 2026-07-12  
**会话概况**: 固定引言的三个关键叙事决定：以方向性干扰提出问题、不在引言点名 LoRA-Null、将历史任务保持与跨模态检索保持分别归因于 LoRA-NF 和知识蒸馏。

---

## 1. 关键讨论 / 决策

1. 引言以持续适配中的方向性干扰作为理论切入口，并将历史任务遗忘与跨模态关系漂移区分为两种后果。
2. 摘要和引言不点名 LoRA-Null。相关方法统一概括为：利用低能量激活方向构成的近似零空间基初始化或约束 LoRA 的输入侧因子 (A)。
3. LoRA-NF 的直接机制主张是历史任务保持：持续过滤历史高能量激活方向，抑制这些方向上的有效学习。
4. LoRA-NF 不负责跨模态检索保持。跨模态知识蒸馏通过显式维持图文关系，承担检索保持的主要机制责任。
5. LR-RGDA/CLIP-text 集成只作用于最终分类决策，不影响跨模态检索。

## 2. 推荐论证链

1. 持续监督适配同时造成历史任务方向干扰和预训练图文关系漂移；
2. 标准 LoRA 只控制秩，不能控制适配发生的历史输入方向；
3. 梯度投影修改更新，零空间启发的 LoRA 方法初始化或约束 (A)，但均不同于在两个可训练因子之前持续过滤输入；
4. LoRA-NF 以 (PAB) 对历史任务方向进行持续前向过滤，主要保护历史任务；
5. 跨模态蒸馏显式维持图文关系，主要保护跨模态检索；
6. LR-RGDA/text 集成补足分类决策。

## 3. 实验责任

- 历史任务保持：LoRA、null-space-inspired (A) variants、gradient projection 与 LoRA-NF 的公平比较及机制指标；
- 跨模态检索保持：LoRA-NF 无蒸馏、单一蒸馏分量与完整蒸馏的直接消融；
- 分类决策：CLIP-text、LR-RGDA 与 ensemble 的拆分结果。

## 4. 相关文件

- `chat-history-for-paper-writing/2026-07-12-15-introduction-first-draft.md`
- `chat-history-for-paper-writing/2026-07-12-06-abstract-and-introduction-outline.md`
- `chat-history-for-paper-writing/2026-07-12-02-current-narrative.md`
- `chat-history-for-paper-writing/2026-07-12-03-claim-evidence-ledger.md`
- `chat-history-for-paper-writing/2026-07-12-08-method-outline.md`
