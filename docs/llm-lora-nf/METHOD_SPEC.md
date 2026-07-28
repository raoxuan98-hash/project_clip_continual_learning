# LLM LoRA-NF 方法规范

**版本**: 0.4
**日期**: 2026-07-28
**状态**: Phase -1 审计完成；当前隔离副本完整测试
`184 passed, 3 warnings`，正式数值证据仍须来自干净 commit 的 GPU 实验
**审计来源**:

- `meta-prompts/2026-07-12-02-作者对Lora_nsp的理解.md`
- `src/models/lora_sgp.py`
- `src/trainers/lora_nsp_trainer.py`
- `docs/lora_null_basis_ablation_plan.md`

---

## 1. 方法身份

LoRA-NF 是运行时、输入侧、持续生效的激活空间滤波 LoRA。它不是：

- LoRA 初始化方法；
- optimizer step 前的事后梯度投影；
- 训练开始前只执行一次的权重投影；
- LoRA-Null 或 MiLoRA 的重命名；
- 仅保留零空间 basis 参数化的简化 adapter。

本 LLM 分支使用标准 LoRA 因子作为 LoRA-NF 的可训练参数，不把 DoRA magnitude 分解混入本文方法。DoRA 作为独立基线。

## 2. 统一记号

论文使用样本为行向量的记号：

\[
X\in\mathbb{R}^{n\times d_{\mathrm{in}}},
\quad
W_0\in\mathbb{R}^{d_{\mathrm{in}}\times d_{\mathrm{out}}},
\quad
Y=XW_0.
\]

标准 LoRA：

\[
\Delta W=AB,
\quad
A\in\mathbb{R}^{d_{\mathrm{in}}\times r},
\quad
B\in\mathbb{R}^{r\times d_{\mathrm{out}}}.
\]

LoRA-NF：

\[
\Delta W_{\mathrm{NF}}=PAB,
\qquad
Y=XW_0+(XP)AB.
\]

PyTorch `nn.Linear` 将权重保存为
\(W_{\mathrm{pt}}\in\mathbb{R}^{d_{\mathrm{out}}\times d_{\mathrm{in}}}\)，
因此实现侧使用：

\[
\Delta W_{\mathrm{pt}}=B_{\mathrm{pt}}A_{\mathrm{pt}}P,
\]

\[
\operatorname{Linear}_{\mathrm{NF}}(x)
=xW_{\mathrm{pt}}^\top+xP A_{\mathrm{pt}}^\top B_{\mathrm{pt}}^\top.
\]

其中：

\[
A_{\mathrm{pt}}\in\mathbb{R}^{r\times d_{\mathrm{in}}},
\qquad
B_{\mathrm{pt}}\in\mathbb{R}^{d_{\mathrm{out}}\times r}.
\]

该方向与当前 CLIP `SGPBaseLoRA._compute_lora_delta()` 中的
`B @ A @ P` 完全一致。

## 3. 作用模块

主实验只包装 decoder self-attention 的：

- `q_proj`；
- `k_proj`；
- `v_proj`；
- `o_proj`。

LoRA-NF 不直接修改 \(QK^\top\)、softmax attention score 或 attention mask。
它修改产生 Q/K/V/O 的线性投影矩阵上的 LoRA 分支。

主实验不包装：

- `gate_proj`；
- `up_proj`；
- `down_proj`；
- embedding；
- LM head；
- LayerNorm/RMSNorm。

Track B 中所有基线使用相同的 attention-only 模块范围。Track A 保留官方实现所需的模块范围，只用于公开协议复现，不和 Track B 混表。

## 4. 激活二阶统计

对目标层输入、去除 padding token 后收集：

\[
M=\frac{1}{N}\sum_{i=1}^{N}x_i^\top x_i
=\frac{1}{N}X^\top X.
\]

这里的 \(M\) 是未中心化 activation second moment / Gram matrix，不称为 covariance。

统计规则：

- `q_proj/k_proj/v_proj` 在同一 decoder layer 共享同一输入，使用同一份统计和 filter；
- `o_proj` 的输入是 attention 聚合结果，单独统计；
- 只累计 attention mask 为 1 的非 padding token；
- instruction、system、user、assistant 和特殊 token 是否纳入统计由 manifest 固定，主协议纳入所有非 padding token；
- LoRA-NF 在模型所在设备以 FP32 累计，完成后一次性转移聚合矩阵到 CPU；
  特征分解使用 FP64 或经验证的稳定替代；
- 分解前拒绝 NaN/Inf；对称化与 ridge 后若出现超过谱尺度 `1e-6` 的负
  特征值则判为损坏/非 PSD 统计并停止，只把舍入误差量级的微小负值截为 0，
  不使用绝对值掩盖错误；
- 原始 token activation 不落盘，仅保存聚合统计、basis 和元数据。

## 5. Hard leaky null-space filter

对称分解：

\[
M=V\operatorname{diag}(\lambda_1,\ldots,\lambda_d)V^\top,
\quad
\lambda_1\le\cdots\le\lambda_d.
\]

按照当前 CLIP hard LoRA-NF 语义，先将特征值归一化到均值 1，再选择最小的 \(m\)，使：

\[
\frac{\sum_{i=1}^{m}\lambda_i}
{\sum_{i=1}^{d}\lambda_i}\ge\epsilon.
\]

低能量可塑子空间：

\[
U_{\mathrm{tail}}=V_{[:,1:m]}.
\]

泄漏系数记为 \(\rho\)，对应当前 CLIP 配置中的 `nsp_weight`。filter 为：

\[
P=(1-\rho)U_{\mathrm{tail}}U_{\mathrm{tail}}^\top+\rho I.
\]

主协议初始锚点：

- \(\epsilon=0.20\)；
- \(\rho=0.02\)；
- 消融时再改变，不能按数据集逐项调参。

### 5.1 阈值边界兼容性审计

公式中的 \(m\) 是满足累计能量条件的最小**维数**，因此必须包含首次跨过
\(\epsilon\) 的特征向量。实现使用 `searchsorted(...)+1` 作为切片终点。

现有 CLIP `build_projection()` 把首次跨过阈值的**索引**直接当作 Python
切片终点，因而会少包含一个向量。LLM 分支不复制该 off-by-one 行为，
而采用与论文公式一致的闭区间定义。兼容性测试同时保存 legacy 与 corrected
维数，后续若回修 CLIP 实现，必须单独重跑受影响的 CLIP 消融，不能把两种
定义的结果混表。

## 6. 精确低秩应用

令受保护高能量子空间为：

\[
U_{\mathrm{prot}}=V_{[:,m+1:d]}.
\]

由于 \(V\) 为正交基：

\[
U_{\mathrm{tail}}U_{\mathrm{tail}}^\top
=I-U_{\mathrm{prot}}U_{\mathrm{prot}}^\top.
\]

因此：

\[
P=I-(1-\rho)U_{\mathrm{prot}}U_{\mathrm{prot}}^\top.
\]

运行时不构造稠密 \(P\)，而计算：

\[
xP=x-(1-\rho)(xU_{\mathrm{prot}})U_{\mathrm{prot}}^\top.
\]

这对同一 \(U_{\mathrm{prot}}\)、\(\rho\)、dtype 和乘法次序定义下的 \(P\) 是代数等价实现。

需要分别报告：

1. factorized apply 与显式 dense \(P\) 的数值误差；
2. 特征分解或 randomized/truncated 求解相对精确 eigensolver 的子空间误差。

两类误差不得混为“factorization error”。

## 7. LoRA 分支与缩放

主实现：

\[
y=xW^\top
+s\cdot B(A(\operatorname{Dropout}(xP))),
\quad
s=\frac{\alpha}{r}.
\]

默认采用论文协议的 \(\alpha=r\)，因此 \(s=1\)，与当前 CLIP `BAP` 无额外缩放的语义一致。

若使用 LoRA dropout，dropout 位于 filter 之后、\(A\) 之前。eval/merge 时 dropout 关闭，等效权重为：

\[
W_{\mathrm{eff}}=W+sBAP.
\]

初始化遵循标准 LoRA：

- \(A\)：Kaiming uniform；
- \(B=0\)；
- 初始模型函数与 Instruct checkpoint 完全一致。

为与 PEFT 0.17.1 的默认 adapter autocast 行为一致，所有 native 方法的
可训练 \(A/B\) 固定为 FP32；冻结基座与 LoRA-NF filter basis 保持模型
精度。forward 在 filter 后将 adapter 输入转为 FP32，并将 adapter 输出转回
基座输出 dtype。否则 native BF16 adapter 与 PEFT FP32 adapter 会形成未声明
的优化精度差异。

Track B runner 在初始化后、训练前执行统一的可训练参数硬审计：

- 每个可训练参数名必须位于 `q_proj/k_proj/v_proj/o_proj` 之一；
- 四个目标必须全部实际出现；
- PEFT 显式启用 `autocast_adapter_dtype=True`，正式运行只接受 FP32
  adapter 参数；
- 报告封存可训练 tensor/element 数、按 dtype 计数和排序参数名
  SHA-256；
- trainer 返回的可训练参数数必须与训练前封存值完全相同。

因此 attention-only 与优化精度不只由 YAML 声明，还由每个正式运行的实际
模型参数状态证明。DoRA 的 magnitude 参数仍计入审计和单独参数量报告。

## 8. 单任务知识保持协议

第一轮 SFT 前，使用 256 条 NQ Open calibration 样本构建 reference second moment 和 \(U_{\mathrm{prot}}\)。

因此，LLM LoRA-NF 从第一个训练步开始就持续保护 Instruct checkpoint 的参考激活空间；它不同于当前 CLIP 持续学习中第一个任务常使用单位阵占位、任务后才形成历史 filter 的流程。

训练期间：

- \(W_0\) 冻结；
- \(U_{\mathrm{prot}}\)、\(\epsilon\)、\(\rho\) 固定；
- 只优化 attention-only LoRA \(A/B\)；
- 每个 forward 都执行 \(xP\)；
- 不对 optimizer gradient 再执行第二次投影。

LoRA-NF calibration 采用未中心化、未做逐样本幅值归一化的
\(X^\top X/N\)。LoRA-Null Track A 和 Track B 均保留官方源码的
`input / torch.max(input).abs()` 与 `/256` 累计规则。前者的分母是
“有符号全局最大值的绝对值”，不等于通常的 `input.abs().max()`。
二者是方法定义的一部分，不为了表面统一而互换。

Track B 的“统一受控”仅统一目标模块和训练预算，不抹除方法特定
calibration。其 LoRA-Null 使用官方 NQ Open raw character-span sampler、
seed 233、`256 × 2048`，只把 hook 范围限制为 `q/k/v/o`。其
`q/k/v` 同层共享数学相同的输入 moment，`o` 独立。run report 必须记录
sampling、normalization、样本数和序列长度。LoRA-NF 仍使用 Instruct chat
template、256 个问题样本、1024 token 上限和非 padding token 的全局
second moment；两者不得复用同一 calibration cache。

LoRA-Null 的 covariance 与 base weight 在分解前转为 FP32，并使用
`torch.linalg.svd`，对齐官方源码。共享输入组只执行一次 SVD，随后由同组
各投影复用同一尾部 basis；这消除 q/k/v 的重复分解但不改变矩阵或子空间。
该 FP32 约束仅属于 LoRA-Null；LoRA-NF 的 reference eigensolver 仍使用
FP64。

LoRA-Null 的逐样本 \(X^\top X/256\) 保留在 activation 所在设备累加，
与官方 hook 一致；每个聚合 moment 只在 calibration 完成后搬到 CPU。
这不改变逐样本归一化或加法顺序，并避免每条样本传输整张 \(d\times d\)
矩阵。

Track A 使用独立配置、注入器、checkpoint 和训练入口，包装
`q/k/v/o/gate/up/down`；它不能通过 `AdapterConfig` 放宽 Track B 的
attention-only 约束。q/k/v 与 gate/up 的输入完全相同，统一封装只存一份
对应 second moment，但每个目标层仍使用与官方重复 hook 数学相同的矩阵。

官方 adapter 构建阶段用 FP16 收集统计并构建稠密 residual。统一 Track A
封装保留 FP16 calibration arithmetic，但在 FP32 中安装分解，从而使
adapter-only checkpoint 能从原始 FP32 checkpoint 精确重建。该数值稳定化
必须在 Track A 报告中披露，并通过同 checkpoint/seed/evaluator 的差异门控，
不表述为官方代码的逐位复现。

对 LoRA-Null/MiLoRA 的函数保持重参数化，记初始化因子为
\(A_0,B_0\)。官方写法把 \(\alpha B_0A_0/r\) 从冻结 base weight 中减去，
再由当前 adapter 加回。实现使用严格数学等价的
\[
W_0+\frac{\alpha}{r}(BA-B_0A_0)
\]
形式，保持原始 \(W_0\) 不变。这样初始化时两个因子路径逐元素相同，
adapter 增量在 FP32 中精确为零，避免深层模型把两个独立 dense matmul 的
舍入误差累积到 logits。对 \(A,B\) 的训练梯度、merge 后的有效权重和
checkpoint 重放均与减 base weight 的写法相同；merge 与累计 factor stack
必须显式减去保存的 \(B_0A_0\)。

官方 MetaMathQA formatter 对 source 与 source+target 分别执行 512 token
右截断，再按截断后的 source 长度 mask labels。若 source 自身占满窗口，
整行 labels 均为 `-100`；官方 Trainer 仍将该行计入 shuffle、gradient
accumulation 与 scheduler，但 PyTorch mean cross entropy 会返回 NaN。
本实现不丢弃、不左截断这类 Track A 行，而是把它们作为显式零梯度
micro-batch 消费，并在训练报告中记录
`zero_supervision_micro_batches` 与 `supervised_tokens`。Track B 另行固定
`left_preserve_response`，保证统一比较中的每条样本都有回答监督。

## 9. 连续任务协议

设 reference calibration 为状态 0。完成任务 \(t\) 后：

1. 在任务 \(t\) 数据上采集当前模型的 attention 输入 second moment；
2. 将任务 LoRA delta merge 到当前模型；
3. 重置 LoRA \(A/B\)；
4. 以阶段等权方式更新历史 second moment：

   \[
   \bar M_t=\frac{c_{t-1}\bar M_{t-1}+M_t}{c_{t-1}+1};
   \]

5. 从 \(\bar M_t\) 重建受保护 basis；
6. 下一任务全程使用新的 filter。

reference calibration 计为第一个阶段，确保预训练/Instruct 能力不会在历史平均中消失。

必须在 TRACE 前做两种状态更新消融：

- `reference_fixed`：全序列固定 reference basis；
- `reference_plus_history`：按上述方式等权更新。

主协议根据预注册 pilot 选择一次，之后不按任务顺序切换。

## 10. 保存、加载与 merge

连续任务不保存每个阶段的模型 checkpoint。训练时每个任务的有效低秩更新
先合并到内存中的当前模型，再把其可重放因子追加到内存状态；整个序列结束
后只写一个累计 adapter 目录。LoRA-NF 将 \(AP_t\) 与 \(B_t\) 保存为该任务
的有效低秩分支，因此无需保存每个历史 filter basis 或稠密 delta；
LoRA-Null 额外保存其函数保持初始化的减项，DoRA 保存 PEFT adapter state，
均可从同一个原始 Instruct checkpoint 按任务顺序精确重放。

最终 checkpoint 必须包含：

- base model ID、revision 和 Instruct 标识；
- attention-only target module 清单；
- 每任务可重放的有效 LoRA \(A/B\)（或 DoRA state）、rank、alpha、dropout；
- 最终保护状态中每层 `qkv` 与 `o` 的 \(U_{\mathrm{prot}}\)；
- \(\epsilon\)、\(\rho\)、特征值摘要和选定维数；
- second-moment 阶段计数；
- manifest hash 和代码 commit。

禁止写入中间 step/epoch checkpoint、optimizer state 和 scheduler state。
任务×时间评测在每个任务合并后直接以内存模型完成，只保存预测、指标矩阵
和证据 manifest。累计 adapter 是一个最终 artifact，不是八个伪装成子目录
的模型 checkpoint。

merge 后必须满足 eval 模式下：

\[
\max |f_{\mathrm{unmerged}}(x)-f_{\mathrm{merged}}(x)|
\]

低于按 dtype 预注册的容差。

## 11. 必须通过的正确性测试

1. Dense \(P\) 与 factorized \(P\) 的 forward 等价；
2. Dense 与 factorized 的 \(A/B\) 梯度等价；
3. 一个 optimizer step 后参数等价；
4. filter 为单位阵时退化为标准 LoRA；
5. \(\rho=1\) 时退化为标准 LoRA；
6. \(x\) 位于受保护子空间时 adapter 输入响应按 \(\rho\) 缩放；
7. q/k/v 同层共享 basis，o 独立；
8. padding token 不进入 second moment；
9. merge/unmerge 等价；
10. save/load 后 logits 和 basis 元数据一致；
11. 只有 attention projection 的 LoRA/NF 参数 `requires_grad=True`；
12. CPU smoke 输出被强制标记且不能被正式聚合器读取。

FP32 单元测试目标容差 `atol=1e-5, rtol=1e-4`；BF16 GPU 容差在首次 smoke 后预注册，不根据方法优劣调整。

完整模型的初始化函数保持检查会累积跨层浮点舍入误差，因此另行预注册：
FP32 `atol=2e-4, rtol=2e-4`，BF16 `atol=2e-2, rtol=2e-2`。
这只用于检查初始化前后 logits 等价，不用于比较任务指标。

## 12. 与基线的责任边界

- LoRA：锁定 PEFT 0.17.1，相同 attention 模块、rank、alpha、dropout，
  标准初始化且无 filter。
- DoRA：相同 attention 模块和训练预算，额外 magnitude 参数单独报告。
- PiSSA/MiLoRA/CorDA/LoRA-Null：保留各自初始化/校准机制，但 Track B 统一 attention-only。
- LoRA-Null：初始化后不持续使用 \(P\)。
- LoRA-NF：标准 LoRA 初始化，训练全程持续使用 \(P\)。

上述区分必须由配置和结果元数据显式编码，不能仅依赖实验名称。
