# LoRA-NSP 右乘线性层记号约定

**日期**: 2026-07-12  
**状态**: 已确定，后续论文讨论与写作统一遵循  
**会话概况**: 为避免 PyTorch 权重存储方向与论文数学记号反复冲突，确定使用行向量输入和右乘线性层的统一约定。

---

## 1. 统一记号

设批量输入为：

\[
X\in\mathbb{R}^{n\times d_{\mathrm{in}}},
\]

线性层采用右乘形式：

\[
Y=XW_0,
\qquad
W_0\in\mathbb{R}^{d_{\mathrm{in}}\times d_{\mathrm{out}}}.
\]

标准 LoRA 为：

\[
\Delta W=AB,
\]

其中：

\[
A\in\mathbb{R}^{d_{\mathrm{in}}\times r},
\qquad
B\in\mathbb{R}^{r\times d_{\mathrm{out}}}.
\]

LoRA-NSP 为：

\[
\Delta W=PAB,
\qquad
P\in\mathbb{R}^{d_{\mathrm{in}}\times d_{\mathrm{in}}}.
\]

完整前向传播写作：

\[
Y=X(W_0+PAB)=XW_0+(XP)AB.
\]

因此 adapter 支路的计算顺序为：

\[
X\longrightarrow XP\longrightarrow XPA\longrightarrow XPAB.
\]

## 2. 与代码的对应

PyTorch `F.linear(x, weight)` 实际计算 (xW_{\mathrm{code}}^\top)，代码权重形状为 \([d_{\mathrm{out}},d_{\mathrm{in}}]\)。

因此代码中的：

```python
lora_delta = B @ A @ P
```

对应论文右乘记号中的：

\[
\Delta W_{\mathrm{paper}}=PAB.
\]

二者满足：

\[
\Delta W_{\mathrm{code}}
=
\Delta W_{\mathrm{paper}}^\top.
\]

这只是存储和记号约定的转置关系，不代表两种不同方法。

## 3. 后续写作规则

- 正文、附录、方法图和后续讨论一律使用 (XW) 右乘约定；
- 标准 LoRA 一律写成 (AB)；
- LoRA-NSP 一律写成 (PAB)；
- “前向滤波”一律解释为 LoRA 增量支路先计算 (XP)；
- 只有在解释代码实现时，才注明 PyTorch 存储形式为转置后的 `B @ A @ P`；
- 所有梯度、优化动力学和理论推导都必须按照该约定重新检查维度。

## 4. 对上一份讨论记录的解释

`2026-07-12-10-lora-nsp-code-and-lora-null-interpretation.md` 使用了与代码直接一致的列向量/左乘表达 (BAP)。其方法含义仍然有效，但后续引用其中公式时应统一转写到本文件规定的右乘形式 (PAB)。
