# LoRA-NSP 核心叙事与方法命名讨论

**日期**: 2026-07-12  
**状态**: 核心方法认知已基本确定；正式名称待最终选择  
**会话概况**: 综合作者 meta-prompt、代码核对、LoRA-Null 阅读和优化动力学讨论，形成 LoRA-NSP 后续论文写作的基础叙述，并比较候选方法名称。

---

## 1. 统一记号与方法定义

论文统一使用行向量输入和右乘线性层：

\[
Y=XW_0,
\qquad
\Delta W_{\mathrm{LoRA}}=AB.
\]

LoRA-NSP 定义为：

\[
\Delta W_{\mathrm{NSP}}=PAB,
\]

从而：

\[
Y=X(W_0+PAB)
=XW_0+(XP)AB.
\]

其中 (P) 由历史任务线性层输入激活的 second-moment/Gram matrix 的特征谱构造。它在 LoRA 增量支路中持续存在，使低秩适配器始终在经过历史谱滤波的输入 (XP) 上学习。

## 2. 论文核心方法叙述

LoRA-NSP 的主要创新不应表述为“又一种零空间投影”，而应表述为：

> 在不缩减标准 LoRA 表达集合的泄漏式设置下，利用历史激活谱构造持续的前向滤波重参数化，从而改变低秩适配器的优化几何，并抑制其在历史高能量方向上的有效学习速度。

这一叙述包含四个不可分割的部分：

1. **History-aware activation spectrum**：(P) 来自历史任务输入激活，而不是只来自当前层权重；
2. **Persistent forward filtering**：(P) 进入 adapter 的每次前向传播，而不是仅在初始化或梯度更新后使用；
3. **LoRA reparameterization**：低秩因子 (A,B) 在过滤后的坐标中学习；
4. **Optimization geometry**：谱滤波改变不同历史方向上的有效学习率，而不必依靠缩减同 rank LoRA 的表达集合。

## 3. 与标准 LoRA 的关系

令：

\[
C=PA.
\]

当 (P) 可逆时，(PAB=CB)，因此 LoRA-NSP 与同 rank 标准 LoRA 具有相同的函数表达集合。方法收益不是来自额外容量，而来自不同的优化路径。

普通梯度下降下：

\[
\nabla_A\mathcal L=P^\top\nabla_C\mathcal L,
\]

因此：

\[
C_{t+1}
=C_t-\eta PP^\top\nabla_C\mathcal L.
\]

对称 (P) 下为：

\[
C_{t+1}
=C_t-\eta P^2\nabla_C\mathcal L.
\]

这说明历史激活谱通过 (P^2) 对有效更新进行各向异性预条件。AdamW 的逐坐标一阶/二阶矩与 weight decay 进一步使该重参数化不等价于普通 LoRA。

## 4. 与传统零空间梯度投影的区别

传统做法通常先执行普通前向和反向计算，再对得到的梯度或参数更新进行零空间投影。LoRA-NSP 则把历史谱算子直接放入 adapter 的模型参数化：

\[
X\rightarrow XP\rightarrow XPA\rightarrow XPAB.
\]

因此 (P) 同时影响前向响应、反向梯度和优化器状态。论文中应强调“函数级持续滤波”与“更新后外部修正”的区别。

不宜在没有严格条件分析时声称梯度后投影必然丢失信息；更稳妥的结论是，两者在低秩双因子参数化和自适应优化器下具有不同优化动力学。

## 5. 与 LoRA-Null 的区别

LoRA-Null 使用预训练知识校准激活的近似零空间，只在初始化时构造：

\[
BA_{\mathrm{init}}
=W_0U_{\mathrm{null}}U_{\mathrm{null}}^\top,
\]

并调整 residual base weight，使初始化总权重保持不变。之后使用普通 (AB) 训练，不保留运行时投影。

LoRA-NSP 与其区别在于：

- LoRA-Null 是 activation-null **initialization**；
- LoRA-NSP 是 history-spectrum **persistent reparameterization**；
- LoRA-Null 在训练后允许适配器离开初始子空间；
- LoRA-NSP 在整个训练期间持续施加严格投影或泄漏式谱偏置。

因此论文不能声称首次把激活零空间用于 LoRA，而应强调首次或较早系统研究“持续前向历史谱滤波的低秩适配”。“首次”仍需在完整文献检索后确认。

## 6. 严格零空间与泄漏式谱滤波

当前主配置使用：

\[
P=(1-\rho)UU^\top+\rho I,
\qquad \rho>0.
\]

此时 (P) 满秩，不缩减 LoRA 表达集合，更准确地属于历史激活诱导的泄漏式谱滤波。只有 (\rho=0) 时才是严格的 null-space projection，并真正限制表达空间。

正式论文应把两种情况统一在同一框架中：

- strict projection：强调空间约束与精确/近似历史响应保护；
- leaky spectral filtering：强调保持表达能力，同时通过谱预条件塑造优化几何。

主方法若使用 (\rho>0)，论文核心理论应以第二种情况为主。

## 7. 参数效率边界

LoRA-NSP 的可训练参数仍为低秩 (A,B)，但当前实现额外保存每层稠密 (d\times d) 的非训练矩阵 (P)，并承担谱分解和前向滤波开销。因此不把“轻量 PEFT”作为核心卖点。

同时，稠密存储不是方法的数学必然。对于 (P=(1-\rho)UU^\top+\rho I)，可通过保存 (U,\rho) 进行因子化计算。论文应区分：

- trainable-parameter efficiency；
- optimizer-state cost；
- auxiliary projection-state cost；
- runtime computation cost。

## 8. 候选名称

### 8.1 首选：LoRA-NSP

建议保留现有简称，但将全称从容易被理解为严格投影的 **Null-Space Projection** 调整为：

> **LoRA-NSP: Null-Space Preconditioned Low-Rank Adaptation**

优点：

- 保留现有代码、实验和项目中的 LoRA-NSP 名称；
- `Preconditioned` 与 (P^2) 改变优化几何的理论结论吻合；
- 能覆盖严格投影与泄漏式满秩滤波；
- 与 LoRA-Null 的 initialization 定位形成明显区别。

风险：

- `preconditioning` 常被理解为直接修改优化器；正文必须明确它由前向重参数化诱导，而不是一个外置 optimizer；
- `null-space` 在 (\rho>0) 时是近似来源而非严格作用空间，需要明确 `null-space-inspired` 或 `leaky` 设置。

### 8.2 最直观：SF-LoRA

> **SF-LoRA: Spectrally Filtered Low-Rank Adaptation**

优点：准确、易懂、不过度承诺严格零空间，直接对应 (XP)。

缺点：没有显式表达历史任务、持续学习或优化几何；简称辨识度一般。

### 8.3 最强调机制：FF-LoRA

> **FF-LoRA: Forward-Filtered Low-Rank Adaptation**

优点：直接突出与梯度后投影、LoRA-Null 初始化的差异。

缺点：`FF` 容易被理解为 feed-forward；没有指出滤波器来自激活谱。

### 8.4 最数学化：NSR-LoRA

> **NSR-LoRA: Null-Space Reparameterized LoRA**

优点：准确突出 (PAB) 是持续重参数化，而非初始化。

缺点：名称不够自然；对泄漏式满秩 (P) 仍需限定“null-space-inspired”。

### 8.5 最完整但较长：HASR-LoRA

> **HASR-LoRA: History-Aware Spectral Reparameterized LoRA**

优点：同时体现历史统计、谱结构和重参数化。

缺点：简称拗口，不利于传播；没有保留已有 LoRA-NSP 品牌。

## 9. 命名建议

当前最推荐的双层表达是：

> **LoRA-NSP: Null-Space Preconditioned Low-Rank Adaptation via Persistent Forward Spectral Filtering**

其中：

- `LoRA-NSP` 是方法简称；
- `Null-Space Preconditioned` 是全称解释；
- `persistent forward spectral filtering` 是正文中反复使用的机制描述。

如果最终不希望使用容易引发严格零空间质疑的 `NSP`，则第二选择为：

> **SF-LoRA: Spectrally Filtered Low-Rank Adaptation**.

快速精确名称检索暂未发现上述完整名称被明显用作已有 LoRA 方法名，但这不是完整的相关工作查重，定名前仍需在 arXiv、Google Scholar 和代码平台做系统检索。

## 10. 后续论文中的推荐一句话定义

候选英文工作句：

> We introduce LoRA-NSP, a null-space preconditioned low-rank adaptation method that persistently filters the adapter input through a history-induced activation spectral operator, preserving LoRA's expressive set while reshaping its optimization geometry away from high-energy directions of previously learned tasks.

该句目前只作为故事和结构材料。正式正文需要根据理论成立范围，将 `preserving LoRA's expressive set` 限定在 (P) 可逆的泄漏式设置下。

## 11. 待决定事项

- [ ] 是否将 NSP 正式改释为 Null-Space Preconditioned；
- [ ] 标题和方法名使用 LoRA-NSP 还是 SF-LoRA；
- [ ] 是否在正文中把 strict 与 leaky 两个版本统一定义；
- [ ] 是否以 optimization geometry 作为理论主命题；
- [ ] 完成系统文献查重后再使用“首次”或“不同于所有现有方法”等表述。

