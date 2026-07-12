# 方法命名演化与当前结论

**日期**: 2026-07-12  
**状态**: 当前候选为 LoRA-NF，尚未最终锁定  
**会话概况**: 合并此前分散的 SHIFT、LoRA-SF、SpectraLoRA、LoRA-NSP、NSR-LoRA、LoRA-NSF 与 LoRA-NF 命名讨论，保留演化理由，并固定统一数学记号。

---

## 1. 命名必须表达什么

标准 LoRA 为：

\[
\Delta W=AB.
\]

本文方法为：

\[
\Delta W=PAB,
\qquad
X\Delta W=(XP)AB.
\]

其中 (P) 由历史任务的层输入激活 second-moment/Gram matrix 的特征谱构造，用于保留低能量、近似零空间方向，并抑制历史高能量方向。

方法名应尽量同时满足：

1. 保留 LoRA 身份，因为创新是 (AB\mapsto PAB)，不是脱离 LoRA 的新框架；
2. 保留 null space，因为它是知识保护原则，spectrum 只是构造工具；
3. 突出 filtering，因为 (P) 持续存在于 adapter 前向路径；
4. 不把 (P) 过度称为严格 projection，因为主配置是满秩泄漏式滤波；
5. 简短、自然，适合标题、表格、图例和消融名称。

## 2. 当前首选：LoRA-NF

> **LoRA-NF: Low-Rank Adaptation with Null-Space Filtering**

更完整的标题形式：

> **LoRA-NF: Continual Low-Rank Adaptation through Persistent Null-Space Filtering**

### 为什么当前优先选择 LoRA-NF

- `LoRA` 明确低秩适配本体；
- `NF` 直接指向 Null-Space Filtering；
- 比 LoRA-NSF 更简洁、更像自然的方法名；
- 比 LoRA-NSP 更忠实于 strict、leaky 和 soft 三类实现；
- 不把优化解释层面的 preconditioning 强塞进方法名；
- 可以与 LoRA、LoRA-Null 在表格中形成清晰对照。

### 必须规避的歧义

`NF` 也可能被理解为 normalizing flow、neural field 或 noise filtering。因此：

- 标题、摘要或正文首次出现时必须完整写出 **Null-Space Filtering**；
- 不单独使用含混的 `null filtering`；
- 方法图必须展示 (X\rightarrow XP\rightarrow XPA\rightarrow XPAB)；
- 明确被过滤的是 LoRA adapter branch 的输入，而不是整个主干输入。

## 3. 推荐的一句话定义

> We propose **LoRA-NF**, a low-rank adaptation method with persistent null-space filtering. LoRA-NF replaces the conventional update (AB) with (PAB), where a history-induced activation spectral operator filters the adapter input throughout training and reshapes its optimization geometry away from high-energy directions of previous tasks.

对应中文：

> LoRA-NF 是一种具有持续零空间滤波机制的低秩适配方法。它将标准 LoRA 更新 (AB) 重参数化为 (PAB)，利用历史激活谱诱导的算子在整个训练期间过滤增量支路输入，并抑制历史高能量方向上的有效学习速度。

## 4. strict、leaky 与 soft 版本

`Filtering` 可以统一覆盖三种形式：

### Strict LoRA-NF

\[
P=UU^\top.
\]

当 (U) 张成严格零空间时，(P) 是低秩正交投影，方法会限制 LoRA 的表达空间。

### Leaky LoRA-NF

\[
P=(1-\rho)UU^\top+\rho I,
\qquad \rho>0.
\]

这是当前主配置。(P) 满秩，因此在 (P) 可逆时不缩减同 rank LoRA 的表达集合，但会改变优化几何。

### Soft LoRA-NF

\[
P=V\operatorname{diag}(w(\lambda_i))V^\top.
\]

该版本使用连续谱权重，对不同历史激活方向施加不同过滤强度。

## 5. 命名演化及淘汰原因

### SHIFT / HILT / SILT / SPIRA

这些名字能够表达历史谱、前向过滤或优化几何，但过度偏离 LoRA，使读者误以为本文提出了独立的持续学习框架。因此不再作为主方法名。

### LoRA-SF / LoRA-HSF

Spectrally Filtered LoRA 能准确描述实现，但 spectrum 是构造工具，不是方法保护原则；名称丢失了 null-space 概念。

### SpectraLoRA / LoRA-SR

Spectral Reparameterization 能自然承接表达能力与优化几何理论，但仍然弱化了为什么选择低能量谱方向，因此不作为最终名称。

### LoRA-NSP

旧解释 Null-Space Projection 对当前满秩泄漏式 (P) 不够准确。改释为 Null-Space Preconditioned LoRA 可以成立，但 preconditioning 更适合作为理论解释，而不是方法第一识别位。

### NSR-LoRA

Null-Space Reparameterized LoRA 数学上准确，也能区别于 LoRA-Null 和梯度投影，但名称略硬，不如 LoRA-NF 自然。

### LoRA-NSF

Low-Rank Adaptation with Null-Space Filtering 最无歧义，但 `NSF` 较长，且不如 `NF` 简洁。可作为 LoRA-NF 的正式机制解释，但不必作为简称。

### NuLoRA / NullFilter-LoRA

NuLoRA 容易与 LoRA-Null 混淆；NullFilter-LoRA 更像代码模块名，均不采用。

## 6. 推荐术语层级

- 方法简称：**LoRA-NF**；
- 正式全称：**Low-Rank Adaptation with Null-Space Filtering**；
- 核心机制：**persistent forward null-space filtering**；
- 滤波器来源：**history-induced activation spectral operator**；
- 数学形式：(PAB)；
- 理论解释：**implicit spectral preconditioning** 或 **spectrum-dependent optimization geometry**；
- 与 LoRA-Null 的区别：persistent filtering versus initialization-only null-space bias；
- 与传统零空间方法的区别：forward filtering versus post-backward gradient projection。

## 7. 尚待决定

- [ ] 是否最终锁定 LoRA-NF；
- [ ] 完成 LoRA-NF 及其全称的系统文献和代码查重；
- [ ] 决定论文标题使用简短全称还是强调 continual adaptation；
- [ ] 决定是否在正文中正式命名 strict、leaky 和 soft 三个版本；
- [ ] 名称固定后再统一更新 meta-prompt、论文大纲、图表和代码展示名称；代码内部暂不必重命名。

