# 当前论文无预设 Venue 与篇幅限制

**日期**: 2026-07-13  
**会话概况**: 更正将 NeurIPS LaTeX 模板误认为投稿目标的推断，确认当前论文没有确定投稿 venue，也没有固定长度要求。

---

## 1. 关键决策

- `paper-template` 中的 NeurIPS 样式仅为当前排版模板，不代表投稿 NeurIPS。
- 当前论文不受 NeurIPS 九页限制，也不执行任何 venue-driven compression。
- 写作优先保证问题谱系、理论推导、机制解释和实验验证完整。
- 主文与附录的分工由叙事流畅性决定：主文保留正式结论和必要解释，附录承接过长但重要的证明与补充实验。
- LoRA-NF 的中心地位通过叙事结构和论证权重体现，而不是通过机械限制蒸馏或 LR-RGDA 的字数实现。

## 2. 对后续写作的影响

- Related Work 可以充分覆盖传统 CIL 与 VLM 持续学习谱系；
- Method 可以采用作者 TPAMI 论文中较完整的“问题—公式—命题—Remark”组织；
- Experiments 可以保留机制、稳健性、复杂度和边界条件分析；
- 仅删除信息重复、无证据形容词和与主线无关的内容。

## 3. 相关文件

- `meta-prompts/2026-07-13-01-作者论文写作风格.md`
- `chat-history-for-paper-writing/2026-07-13-01-author-writing-style-analysis.md`
