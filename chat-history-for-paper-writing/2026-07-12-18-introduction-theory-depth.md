# 引言与方法正文的理论深度划分

**日期**: 2026-07-12  
**会话概况**: 确定 LoRA-NF 在摘要和引言中只保留机制直觉，正式理论结论进入方法正文，完整证明进入附录。

---

## 1. 关键决策

- 摘要和引言保留的直觉：LoRA-NF 在低秩因子之前持续执行泄漏式历史激活过滤，使适配偏离历史高能量方向，同时保留各方向的学习通道。
- 摘要和引言不展开：(P) 的可逆条件、同 rank LoRA 表达集合等价、有效因子更新中的 (P^2) 谱预条件以及优化器细节。
- 方法正文给出三个正式结论：历史响应/干扰界；可逆泄漏式 (P) 下的表达集合等价；(P^2) 诱导的谱相关优化几何。
- 附录承接完整假设、证明、优化器扩展讨论和额外推导。

## 2. 推荐引言措辞

> We use a leaky rather than strict filter, allowing the adapter to retain flexibility across directions while biasing learning away from those most strongly occupied by historical tasks.

该句只表达机制直觉，不在引言中预支 theorem-level claim。

## 3. 相关文件

- `chat-history-for-paper-writing/2026-07-12-15-introduction-first-draft.md`
- `chat-history-for-paper-writing/2026-07-12-06-abstract-and-introduction-outline.md`
- `chat-history-for-paper-writing/2026-07-12-08-method-outline.md`
