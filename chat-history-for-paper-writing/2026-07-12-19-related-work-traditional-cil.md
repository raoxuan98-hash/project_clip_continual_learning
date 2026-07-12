# Related Work 中传统类增量学习的保留方式

**日期**: 2026-07-12  
**会话概况**: 决定在 Related Work 第一节保留传统类增量学习方法，以建立从旧类保持到预训练 VLM 能力保持的问题演进。

---

## 1. 关键决策

- 第一节标题调整为 `Class-Incremental Learning with Pre-trained Vision-Language Models`，同时覆盖传统 CIL 和 CLIP/VLM CIL。
- 传统 CIL 按机制组织，而不是按论文逐篇罗列：
  1. replay/rehearsal；
  2. knowledge distillation and regularization；
  3. parameter isolation/dynamic architectures；
  4. prototype learning and classifier-bias correction。
- 传统 CIL 的作用是建立历史类别保持、旧新类不平衡和分类器偏置等基础问题。
- 随后强调 VLM 场景新增的保持对象：开放词汇语义、零样本泛化和图文对齐。
- LoRA-NF 对应历史任务保持；跨模态蒸馏对应 VLM 特有的跨模态能力保持。

## 2. 篇幅原则

- 传统 CIL 使用一个紧凑段落，约占第一小节的三分之一；
- 每个方法族选择一至两个代表性引用；
- 不写成完整 CIL 综述，不展开与本文无关的任务无关性、动态网络或大规模 replay 细节；
- 重点是完成从 traditional CIL 到 VLM continual learning 的概念过渡。

## 3. 候选代表方法

- Distillation/regularization: LwF, EWC, LUCIR, PODNet；
- Replay/rehearsal: iCaRL, DER；
- Classifier correction: BiC, WA；
- Parameter isolation/dynamic architectures: 只在需要完整方法谱系时保留一至两个代表工作。

最终引用名单需结合当前 BibTeX、版面和与 X-TAIL 协议的相关性核定。

## 4. 相关文件

- `chat-history-for-paper-writing/2026-07-12-07-related-work-outline.md`
- `chat-history-for-paper-writing/2026-07-12-17-introduction-mechanism-responsibility.md`
