# LoRA-NSP 代码、LoRA-Null 与理论叙事核对

**日期**: 2026-07-12  
**状态**: 方法理解基线，理论命题仍需正式推导与验证  
**会话概况**: 阅读作者 meta-prompt、当前 LoRA-NSP 实现、历史零空间变体消融以及 LoRA-Null 原论文，核对 LoRA-NSP 的准确参数化、创新边界和可建立的理论路线。

---

## 1. 当前实现的准确公式

对列向量输入 (x\in\mathbb{R}^{d_{\mathrm{in}}})，当前代码使用：

\[
W = W_0 + \Delta W,\qquad \Delta W = BAP,
\]

其中：

\[
B\in\mathbb{R}^{d_{\mathrm{out}}\times r},\quad
A\in\mathbb{R}^{r\times d_{\mathrm{in}}},\quad
P\in\mathbb{R}^{d_{\mathrm{in}}\times d_{\mathrm{in}}}.
\]

因此 adapter 支路为：

\[
\Delta y=B A(Px).
\]

PyTorch 使用行向量批表示时，对应 `x P^T A^T B^T`。由于当前 (P) 对称，可直观写成 `x -> xP -> A^T -> B^T`。

作者 meta-prompt 中的 (PAB) 应在正式论文中修正为与维度一致的 (BAP)。如果希望把 (P) 写在最左侧，需要重新定义矩阵方向，不能混用。

“前向滤波”的精确定义是：只过滤低秩增量支路的输入，冻结主干支路 (W_0x) 仍接收原始输入。不是先过滤整个层输入再同时送入 (W_0) 和 adapter。

## 2. (P) 的真实来源与性质

代码通过 hook 收集每个 LoRA 线性层的输入激活，并计算：

\[
\Sigma_h=\frac{1}{N}X_h^\top X_h.
\]

这不是中心化 covariance，更准确的名称是 activation second-moment matrix 或 activation Gram matrix。

当前增量流程在一个任务训练完成后收集统计量，与历史统计做等权任务平均，再为后续任务构造 (P)。因此它主要保护已经学习的历史任务激活方向，并不直接等同于用预训练校准数据保护原始 CLIP 知识。

默认所谓 hard NSP 实际为：

\[
P=(1-\rho)UU^\top+\rho I,
\]

其中 (U) 由低特征值方向构成，默认 (\rho=0.02)。当 (\rho>0) 时，(P) 是满秩的泄漏式谱滤波器，而不是严格的正交投影；只有 (\rho=0) 时才是严格低秩投影。

## 3. 与传统梯度投影的差异

可保留的核心区别是：

- 梯度投影通常在常规前向计算后修改参数梯度；
- LoRA-NSP 把 (P) 放进模型计算图，adapter 始终在 (Px) 上学习；
- (P) 同时影响前向响应、反向梯度和后续优化器状态。

但不宜直接声称“事后梯度投影必然损失信息”。在线性模型、固定 (P) 和普通 SGD 的特定条件下，前向右投影和梯度右投影可能在一阶更新上等价。论文应把区别落在低秩因子化、持续的函数级约束以及优化器几何上。

## 4. 与 LoRA-Null 的准确差异

LoRA-Null 使用代表预训练知识的校准数据得到激活 (X_{\mathrm{pre}})，取其近似零空间 (U_{\mathrm{null}})，然后构造：

\[
BA_{\mathrm{init}}=W_0U_{\mathrm{null}}U_{\mathrm{null}}^\top.
\]

它再对该矩阵做 SVD 来初始化 (A,B)，并把冻结基座设为：

\[
W_0'=W_0-BA_{\mathrm{init}},
\]

从而保证初始化时总权重不变。之后仅正常优化 (A,B)，运行时不存在固定 (P)。LoRA-Null 论文证明的是初始化时 (A^\top) 的列空间落在 (U_{\mathrm{null}}) 中；自由训练后没有严格保证它始终留在该空间。

两者最准确的对比维度为：

| 维度 | LoRA-Null | LoRA-NSP |
|---|---|---|
| 子空间数据 | 预训练知识校准激活 | 历史增量任务激活统计 |
| 介入时机 | 初始化一次 | 每次 adapter 前向持续存在 |
| 参数形式 | 训练时为普通 (BA) | 训练时始终为 (BAP) |
| 基座处理 | 初始化时改成 residual weight | 主流程中冻结基座，任务后合并 delta |
| 约束强度 | 初始化偏置 | 严格投影或泄漏式持续谱偏置 |

因此本文创新不能写成“首次使用激活零空间做 LoRA”，而应强调持续的前向空间滤波与由此产生的优化几何。

## 5. 为什么 (BAP) 不能简单吸收到 (A)

令有效参数 (C=AP)。当 (P) 可逆时，(BAP=BC) 与标准 LoRA 具有相同函数表达集合，但训练动力学不相同。

在只观察 (A/C) 的普通梯度下降下：

\[
\nabla_A\mathcal{L}=\nabla_C\mathcal{L}P^\top,
\]

于是：

\[
C_{t+1}=C_t-\eta\nabla_C\mathcal{L}P^\top P.
\]

对称 (P) 下为右乘 (P^2)。所以即便函数类相同，LoRA-NSP 也在谱方向上改变了有效学习率，是一种由历史激活统计决定的各向异性优化几何。

对 AdamW，差异更强：AdamW 在 (A) 的坐标中维护逐元素一阶矩和二阶矩。一般稠密 (P) 会混合坐标，逐元素二阶矩预条件并不对这种重参数化保持不变；weight decay 映射到 (C) 空间后也不是普通的各向同性衰减。因此不能通过在符号上令 (C=AP) 就认为优化过程等价。

这一点适合成为 LoRA-NSP 理论的核心，而“AdamW 二阶矩导致差异”应先写成正式命题或受控实验后再升级为强结论。

## 6. 可建立的理论路线

### 6.1 历史响应上界

对历史激活 (X_h)：

\[
\|X_h\Delta W^\top\|_F
=\|X_hP A^\top B^\top\|_F
\leq \|X_hP\|_F\|A\|_2\|B\|_2.
\]

该结果直接把投影质量 (\|X_hP\|) 与历史任务上的 adapter 干扰联系起来。

### 6.2 严格投影与泄漏式投影的分层结论

- 当 (P=UU^\top) 且 (X_hP=0) 时，可得到 adapter 对历史激活的精确零响应；
- 当 (P=(1-\rho)UU^\top+\rho I) 时，只能得到由 (\rho) 与残余能量控制的近似上界；
- 当前主配置属于第二种，不能声称严格零干扰。

### 6.3 表达能力与优化偏置

- (P) 可逆时，LoRA-NSP 与同 rank LoRA 的表达集合相同；收益来自优化偏置而非额外容量；
- (P) 不可逆时，表达集合被限制在其像空间中，稳定性与可塑性产生显式权衡；
- 这能够自然解释 (nsp_weight)、(nsp_eps) 与性能之间的消融。

### 6.4 优化器非等价性

先证明 SGD 下的 (P^2) 谱预条件，再讨论 AdamW 对一般线性重参数化的不变性缺失。建议用一个二维解析例子或单层受控实验辅助说明，而不是直接声称普遍更优。

## 7. 参数效率表述

当前实现的可训练参数仍然是低秩 (A,B)，因此从 trainable-parameter count 看仍具有参数效率。但每层保存稠密、非训练的 (d\times d) 矩阵 (P)，并承担特征分解与稠密乘法成本，所以不宜把“标准 PEFT 效率”作为主要卖点。

更准确的表述是：

> LoRA-NSP is a low-rank adaptation method augmented with persistent activation-space spectral filtering.

同时需要说明，稠密 (P) 是当前实现形式而非不可消除的数学必然；当 (P=(1-\rho)UU^\top+\rho I) 时，可用 (U) 和标量 (\rho) 因子化存储与计算。因此论文若把稠密存储称为方法的固有缺点，会被审稿人质疑。

## 8. 对作者原始理解的结论

### 可以保留并强化

- LoRA-NSP 的最大特点是投影持续存在于 adapter 前向路径；
- 它不同于训练后再投影梯度，也不同于只做零空间初始化；
- (BAP) 与吸收 (P) 后的普通 LoRA 在优化动力学上不等价；
- AdamW 的逐元素矩估计是这种非等价的重要来源；
- 方法应被定位在子空间保护与低秩适配之间。

### 需要修正或降强度

- 公式应为 (BAP)，不是 (PAB)；
- 过滤的是 adapter 支路输入，不是完整主干输入；
- 当前 (P) 来自历史任务 second moment，不能自动声称直接保护预训练 CLIP 知识；
- 默认 (\rho=0.02) 时 (P) 是满秩谱滤波器，不是严格零空间投影；
- “梯度投影会损失信息”不是已证明结论；
- “全量微调无法做前向滤波”过强，更准确地说普通全量微调缺少天然独立的 residual adaptation branch；
- LoRA-Null 不是简单把 (A) 初始化为零空间基；
- 稠密 (P) 增加状态和计算，但不增加可训练参数，且可做因子化实现。

## 9. 实验证据的谨慎解读

早期短程/少任务实验中，`hist_null_init_only` 曾略优于 `current_nsp`；后续更完整的组合实验中，`current_nsp` 在 Average/Last 和整体平衡上更强，最终被选为主配置。`fixed_basis`、`core_basis` 和 init/runtime 变体总体没有稳定超过当前 (BAP) 形式。

因此可以说现有消融支持当前 runtime (BAP) 参数化是主候选，但“只有该形式有效”仍需限定任务数、训练 recipe、随机种子和公平子空间维度，最终以统一的 publication-scale 表格为准。

## 10. 相关文件

- `meta-prompts/2026-07-12-02-作者对Lora_nsp的理解.md`
- `src/models/lora_sgp.py`
- `src/trainers/lora_nsp_trainer.py`
- `main_incremental.py`
- `docs/lora_null_basis_ablation_plan.md`
- `docs/arbor_ablation_results.md`
- `paper_writing/deep-research-reports/lora-null-init/pdfs/2503.02659.pdf`
- `chat-history/2026-06-30-02-lora-nsp-basis-null-ablation-math-and-dora-bug.md`
- `chat-history/2026-07-02-ablation-results-final.md`
- `chat-history/2026-07-05-weekly-ablation-review.md`

## 11. 下一步讨论

- [ ] 确定论文中把 LoRA-NSP 定义为严格投影还是泄漏式谱滤波；
- [ ] 决定理论主线以历史响应上界还是优化几何为中心；
- [ ] 设计 LoRA、梯度投影、LoRA-Null 和 LoRA-NSP 的统一公平消融；
- [ ] 核验最终 10-task 多种子结果是否支持“runtime BAP 最优”；
- [ ] 决定是否把 (P) 因子化实现作为效率改进或附录讨论。
