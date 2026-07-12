# Related Work 中传统类增量学习范围

**日期**: 2026-07-12  
**会话概况**: 确定 Related Work 第一节同时覆盖传统类增量学习与预训练 VLM 持续学习，并固定二者之间的问题演进关系。

---

## 1. 关键讨论 / 决策

- 保留传统类增量学习方法，按 replay、蒸馏/正则化、参数隔离以及原型/分类器校正等机制概括。
- 传统 CIL 部分只占第一小节约三分之一，用于建立历史任务保持、旧新类不平衡及分类器偏置问题。
- 随后转向 VLM 场景新增的开放词汇、零样本和跨模态能力保持目标。
- 第一小节建议命名为 `Class-Incremental Learning with Pre-trained Vision-Language Models`。

## 2. 重要发现

- 传统 CIL 与本文并非割裂：LoRA-NF 延续历史任务保持目标；跨模态蒸馏处理 VLM 场景新增的图文关系保持。
- 代表方法应按相关性精选，避免将 Related Work 写成完整历史综述。

## 3. 待办事项 / 遗留问题

- [ ] 核对当前 BibTeX 是否已有 iCaRL、LwF、EWC、LUCIR、PODNet、DER、BiC 和 WA。
- [ ] 根据版面为每个方法族保留一至两个代表引用。

## 4. 相关文件

- `chat-history-for-paper-writing/2026-07-12-07-related-work-outline.md`
- `chat-history-for-paper-writing/2026-07-12-19-related-work-traditional-cil.md`
