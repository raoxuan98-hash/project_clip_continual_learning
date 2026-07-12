# 作者论文写作风格分析

**日期**: 2026-07-13  
**会话概况**: 阅读作者自定义论文写作 skills，并结合 TPAMI 投稿论文的摘要、引言、相关工作、方法、理论、实验和结论，提炼当前论文的写作风格约束。

---

## 1. 操作状态

- **Operation Type**: Consistency Repair / Style Calibration
- **Evidence State**: Strong evidence，包含显式 skill 规则和完整实际论文样本
- **Impact Scope**: 全文写作风格，尤其是 Introduction、Related Work、Method 和 Experiments
- **Claim Changes**: 无；本次只校准表达与结构

## 2. 核心发现

- 作者偏好先拆解技术瓶颈，再按瓶颈顺序推出模块，而不是直接罗列贡献。
- 段落采用显式因果推进，常在段末自然引出下一个问题。
- 公式写作遵循“目的 → 公式 → 符号定义 → 结论解释”。
- 正式理论使用 Lemma/Proposition/Remark 分层，主文保留结论和证明思路，附录承接完整证明。
- Related Work 按方法机制分类，每类结尾给出局部限制或与本文的区别。
- 实验强调隔离变量、稳定区间和机制解释，而不仅报告最优数字。
- 当前项目没有确定投稿 venue 或篇幅限制，应充分保留 TPAMI 样本中的完整论证、理论解释和多层次实验分析。

## 3. 对当前论文的影响

- LoRA-NF 方法部分应从标准 LoRA 的方向性缺口自然推出 (PAB)，避免模板化 `Motivation/Design/Advantages` 标题堆叠。
- 正文理论沿用命题后解释其算法含义的风格，但完整证明进入附录。
- 蒸馏和 LR-RGDA 保持完整可复现定义；它们在叙事层级上低于 LoRA-NF，但不设置机械篇幅上限。

## 5. 后续更正

- `paper-template` 使用 NeurIPS 样式不代表论文将投稿 NeurIPS。
- 当前写作不存在页数约束，不执行 venue-driven compression。
- 主文与附录的内容划分依据可读性和论证层级，而不是页数限制。
- Related Work 应按 traditional CIL、VLM continual learning、knowledge-preserving adaptation 和 statistical prediction 的技术谱系组织。

## 4. 相关文件

- `meta-prompts/2026-07-13-01-作者论文写作风格.md`
- `paper_writing/my_original_papers/my_paper_submitted_to_tpami.pdf`
- `paper_writing/kimi/xuan-research-paper-writing-v2/SKILL.md`
- `paper_writing/kimi/xuan-research-paper-writing-v2/references/method.md`
