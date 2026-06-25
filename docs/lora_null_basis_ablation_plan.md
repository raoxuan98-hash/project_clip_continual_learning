# LoRA-NSP Basis Parameterization and LoRA-Null-Style Initialization Ablation Plan

## 1. 目标

本文档固定本轮消融实验方案，服务于后续实现和服务器实验。实验要回答两个互补问题：

1. **参数化问题**：当前 LoRA-NSP/SGP 的运行时形式为
   \[
   \Delta W = B A P.
   \]
   其中 \(P\) 来自历史任务激活的低能量子空间。我们要测试是否可以把 \(A P\) 重参数化为显式的低秩 basis 形式，以减少参数、增强约束，或提升抗遗忘能力。

2. **初始化问题**：LoRA-Null 的核心思想是把 LoRA 初始化到激活零空间中：
   \[
   B A_{\text{init}} = W_0 U_{\text{null}} U_{\text{null}}^\top.
   \]
   我们要测试这种 LoRA-Null-style 初始化能否替代或增强当前的 runtime projection。

实验先聚焦 vision-side LoRA，以隔离投影和初始化机制本身。文本编码器、FD/CD、LADA-style text prototypes 等因素后续再加入。

## 2. 背景

### 2.1 当前 LoRA-NSP 的实际计算方向

代码中的线性层权重约定为：

\[
W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}.
\]

当前 `src/models/lora_sgp.py` 中的 LoRA 更新为：

\[
\Delta W = B A P,
\]

其中：

\[
B \in \mathbb{R}^{d_{\text{out}} \times r}, \quad
A \in \mathbb{R}^{r \times d_{\text{in}}}, \quad
P \in \mathbb{R}^{d_{\text{in}} \times d_{\text{in}}}.
\]

模型前向使用 PyTorch `F.linear(x, W)`，实际计算：

\[
\text{F.linear}(x, W) = x W^\top.
\]

因此：

\[
x(W_0+\Delta W)^\top
= x W_0^\top + x (BAP)^\top
= x W_0^\top + x P^\top A^\top B^\top.
\]

由于 \(P\) 是由特征向量构造的对称矩阵，\(P^\top=P\)，所以 adapter 分支从输入流角度看是：

\[
x \rightarrow xP \rightarrow A^\top \rightarrow B^\top.
\]

也就是说，当前 LoRA-NSP 的运行时含义是：**先用 \(P\) 过滤输入激活，再让 LoRA 作用于过滤后的变量**。

### 2.2 历史任务保护条件

设某一层历史任务输入激活矩阵为：

\[
X_h \in \mathbb{R}^{N \times d_{\text{in}}}.
\]

如果投影矩阵 \(P_h\) 满足：

\[
X_h P_h \approx 0,
\]

则对历史任务样本，adapter 输出近似为：

\[
X_h \Delta W^\top
= X_h P_h A^\top B^\top
\approx 0.
\]

这就是 runtime NSP 的核心保护机制：**只要 \(P_h\) 持续存在，无论 \(A,B\) 如何训练，历史激活上的 adapter 输出都会被抑制**。

### 2.3 \(P\) 的来源

当前代码通过 hook 抓取每个 LoRA 线性层的输入激活，累积：

\[
\Sigma_h = \frac{1}{N} X_h^\top X_h.
\]

严格说这里是 second moment / Gram matrix，不是中心化 covariance。由于我们关心的是线性层输入方向上的响应能量，而不是统计意义上的方差，使用 \(X^\top X/N\) 是合理的；论文表述中建议称为 **activation second moment** 或 **activation Gram matrix**。

对称特征分解：

\[
\Sigma_h = V \Lambda V^\top,
\]

其中特征值按升序排列：

\[
\lambda_1 \le \lambda_2 \le \cdots \le \lambda_d.
\]

tail/null 子空间取：

\[
U_{\text{tail}} = V[:, 1:k].
\]

hard NSP 投影为：

\[
P_h = U_{\text{tail}} U_{\text{tail}}^\top.
\]

soft SGP 投影为：

\[
P_h = V \operatorname{diag}(w(\lambda_i)) V^\top,
\]

其中 \(w(\lambda_i)\) 对大特征值方向给较小权重，对小特征值方向给较大权重。

## 3. LoRA-Null 和当前方法的关系

LoRA-Null 使用预训练知识校准集产生某一层输入激活：

\[
X_{\text{pre}} \in \mathbb{R}^{d_{\text{in}} \times M}.
\]

其 SVD 为：

\[
X_{\text{pre}} = U \Sigma V^\top.
\]

取最小奇异值对应方向：

\[
U_{\text{null}} = U[:, -r:].
\]

LoRA-Null 初始化：

\[
B A_{\text{init}} = W_0 U_{\text{null}} U_{\text{null}}^\top,
\]

并设置 residual weight：

\[
W_0' = W_0 - B A_{\text{init}}.
\]

因此初始化后合并权重不变：

\[
W_0' + B A_{\text{init}} = W_0.
\]

### 3.1 与当前 `proj_sigma_tail` 的一致性

如果当前项目使用：

\[
\Sigma = X^\top X,
\]

并对 \(\Sigma\) 做特征分解：

\[
\Sigma = V \Lambda V^\top,
\]

那么 \(V\) 与 \(X\) 的右奇异向量一致，\(\Lambda\) 是奇异值平方。由于当前线性层输入 \(x \in \mathbb{R}^{d_{\text{in}}}\)，我们需要的 basis 位于 input feature dimension 中，因此使用 \(\Sigma=X^\top X\) 的特征向量是正确的。

在这个意义上：

\[
B A_{\text{init}} = W_0 V_{\text{tail}} V_{\text{tail}}^\top
\]

与 LoRA-Null 的 activation-null projection 在数学核心上是一致的。

### 3.2 不一致之处

当前项目中的 history-null 初始化不应直接称为严格 LoRA-Null，原因如下：

1. **数据来源不同**  
   LoRA-Null 使用 pretrain/reference calibration activations，目标是保护预训练知识；history-null 使用已学任务 activations，目标是保护增量历史任务。

2. **约束时机不同**  
   LoRA-Null 是 init-only。训练时没有 runtime \(P\)，adapter 可能逐渐漂移出初始零空间。当前 LoRA-NSP 是 runtime constraint，每步 forward 都乘 \(P\)。

3. **保护对象不同**  
   strict LoRA-Null 保护 zero-shot/pretrained knowledge；history-null 保护 seen task knowledge。

因此本文档中把相关变体命名为：

\[
\text{LoRA-Null-style history-null initialization}.
\]

严格 LoRA-Null 版本需额外实现 reference/pretrain calibration activation collection。

## 4. 实验设计轴

本轮实验分成两个家族。

### 4.1 第一组：参数化消融

问题：

\[
BAP
\]

是否应该保留为当前 full LoRA 参数化，还是可以被更直接的 basis 参数化替代？

这一组测试：

1. 当前 runtime projection 是否过度冗余；
2. 显式低秩 basis 是否能减少参数并保持抗遗忘；
3. 固定 basis 和可训练 core 哪个更合适。

### 4.2 第二组：初始化消融

问题：

LoRA-Null-style 初始化

\[
B A_{\text{init}} = W_0 U U^\top
\]

是否能替代或增强 runtime NSP？

这一组测试：

1. init-only 是否足够防遗忘；
2. init + runtime projection 是否互补；
3. history-null 初始化和 current NSP 的相对贡献。

## 5. 方案 0：Current NSP Baseline

### 5.1 名称

```text
current_nsp
```

### 5.2 公式

\[
\Delta W = B A P_h.
\]

其中 \(P_h\) 来自历史任务 activation Gram matrix 的 tail/null 子空间。

### 5.3 前向等价形式

\[
x \Delta W^\top = x P_h A^\top B^\top.
\]

### 5.4 作用

历史任务上：

\[
X_h P_h \approx 0 \Rightarrow X_h \Delta W^\top \approx 0.
\]

### 5.5 实验角色

这是所有变体的主 baseline。不能改变其默认行为。

## 6. 方案 1：Fixed Basis Parameterization

### 6.1 名称

```text
basis_fixed_tail
```

### 6.2 公式

令：

\[
U_h \in \mathbb{R}^{d_{\text{in}} \times k}
\]

为历史 activation Gram matrix 的 tail basis。固定 \(U_h\)，只训练 \(B\)：

\[
\Delta W = B U_h^\top,
\]

其中：

\[
B \in \mathbb{R}^{d_{\text{out}} \times k}.
\]

### 6.3 前向形式

\[
x \Delta W^\top
= x U_h B^\top.
\]

也就是：

\[
x \rightarrow xU_h \rightarrow B^\top.
\]

### 6.4 与 current NSP 的关系

current NSP:

\[
\Delta W = B A U_h U_h^\top.
\]

令：

\[
\tilde{B} = B A U_h,
\]

则：

\[
\Delta W = \tilde{B} U_h^\top.
\]

因此，在 hard projection 且 \(P=U_hU_h^\top\) 的情况下，current NSP 的有效更新空间也落在 \(U_h^\top\) 上。Fixed Basis 直接参数化这个有效空间，但去掉了 \(A\) 的冗余自由度。

### 6.5 动机

1. 参数更少；
2. 约束更强；
3. 更直接体现 “LoRA 只作用于 \(xU_h\)” 的解释。

### 6.6 风险

如果 \(k=r\)，表达力可能太弱。尤其在 tail/null 子空间中，输入能量本来就小，新任务学习可能不足。

### 6.7 建议设置

第一轮：

\[
k = r.
\]

如果效果显示 forgetting 降低但 Last 明显下降，第二轮增加：

\[
k = 4r.
\]

## 7. 方案 2：Core Basis Parameterization

### 7.1 名称

```text
basis_core_tail
```

### 7.2 公式

固定历史 tail basis：

\[
U_h \in \mathbb{R}^{d_{\text{in}} \times k}.
\]

训练：

\[
B \in \mathbb{R}^{d_{\text{out}} \times r},
\quad
C \in \mathbb{R}^{r \times k}.
\]

更新为：

\[
\Delta W = B C U_h^\top.
\]

### 7.3 前向形式

\[
x \Delta W^\top
= x U_h C^\top B^\top.
\]

即：

\[
x \rightarrow xU_h \rightarrow C^\top \rightarrow B^\top.
\]

### 7.4 与 Fixed Basis 的关系

如果 \(k=r\) 且 \(C\) 满秩，则：

\[
B C U_h^\top = \tilde{B} U_h^\top.
\]

这与 Fixed Basis 等价。因此 core basis 的意义在于：

\[
k > r.
\]

即让 adapter 在更宽的保护子空间中学习一个低秩组合。

### 7.5 参数量

以 \(d_{\text{in}}=d_{\text{out}}=768, r=4, k=16\) 为例：

current LoRA:

\[
|A| + |B| = 4 \cdot 768 + 768 \cdot 4 = 6144.
\]

fixed basis:

\[
|B| = 768 \cdot 4 = 3072.
\]

core basis:

\[
|B| + |C| = 768 \cdot 4 + 4 \cdot 16 = 3136.
\]

core basis 只比 fixed basis 多很少参数，但显著增加子空间内的组合能力。

### 7.6 动机

Fixed Basis 可能太硬；Core Basis 是更合理的中间方案：

1. 仍然保证更新只发生在历史低能量子空间；
2. 比 fixed basis 有更强表达力；
3. 参数量接近 fixed basis，明显低于 current LoRA。

### 7.7 建议设置

第一轮：

\[
r=4, \quad k=16.
\]

命令参数建议：

```text
--projection_param_mode core_basis
--basis_rank 16
--basis_window tail
```

## 8. 方案 3：Soft Core Basis

### 8.1 名称

```text
basis_core_soft
```

### 8.2 公式

soft SGP 投影：

\[
P = V \operatorname{diag}(w_i) V^\top.
\]

取前 \(k\) 个低能量方向，并吸收权重：

\[
U_{\text{soft}} = V[:,1:k]\operatorname{diag}(\sqrt{w_1}, \ldots, \sqrt{w_k}).
\]

更新：

\[
\Delta W = B C U_{\text{soft}}^\top.
\]

### 8.3 动机

hard tail 可能过于保守。Soft Core Basis 保留 SGP 的连续谱权重，同时避免 full \(d \times d\) projection 的运行时成本和参数冗余。

### 8.4 优先级

不是第一轮核心实验。只有当 `basis_core_tail` 有潜力但 Last 不足时，再引入该方案。

## 9. 方案 4：History-Null Init Only

### 9.1 名称

```text
hist_null_init_only
```

### 9.2 公式

从历史 activation Gram matrix 中取 tail basis：

\[
U_h = V_h[:,1:r].
\]

构造：

\[
B A_{\text{init}} = W_0 U_h U_h^\top.
\]

设置：

\[
W_0' = W_0 - B A_{\text{init}}.
\]

初始化后：

\[
W_0' + B A_{\text{init}} = W_0.
\]

训练时不使用 runtime \(P\)：

\[
\Delta W = B A.
\]

### 9.3 与 LoRA-Null 的一致性

该方案与 LoRA-Null 的 algebra 一致：

\[
B A_{\text{init}} = W_0 U_{\text{null}} U_{\text{null}}^\top.
\]

但因为 \(U_h\) 来自 history task activations，不来自 pretrain/reference calibration set，所以这是：

```text
LoRA-Null-style history-null initialization
```

不是 strict LoRA-Null。

### 9.4 实验意义

如果该方案明显弱于 `current_nsp`，说明仅靠初始化不能维持子空间约束，runtime projection 是必要的。

如果该方案接近或优于 `current_nsp`，说明初始化空间本身贡献很大，后续可进一步做 strict LoRA-Null calibration 版本。

## 10. 方案 5：History-Null Init + Runtime NSP

### 10.1 名称

```text
hist_null_init_runtime
```

### 10.2 公式

初始化同方案 4：

\[
B A_{\text{init}} = W_0 U_h U_h^\top,
\quad
W_0' = W_0 - B A_{\text{init}}.
\]

训练时继续使用 runtime NSP：

\[
\Delta W = B A P_h.
\]

前向 adapter：

\[
x \Delta W^\top = x P_h A^\top B^\top.
\]

### 10.3 动机

该方案测试两个机制是否互补：

1. 初始化时，adapter 已位于 history-null space；
2. 训练过程中，runtime \(P_h\) 持续阻止 adapter 漂移到历史高能量方向。

### 10.4 预期

这是第二组中最值得关注的方案。如果它优于 `current_nsp`，说明：

```text
LoRA-Null-style initialization + runtime NSP
```

可以成为新的主方法候选。

## 11. 方案 6：Reference-Null Init + Runtime NSP

### 11.1 名称

```text
ref_null_init_runtime
```

### 11.2 公式

使用 reference/pretrain calibration activations：

\[
X_{\text{ref}}.
\]

得到：

\[
U_{\text{ref-null}}.
\]

初始化：

\[
B A_{\text{init}} = W_0 U_{\text{ref-null}} U_{\text{ref-null}}^\top.
\]

训练时使用 history projection：

\[
\Delta W = B A P_h.
\]

### 11.3 动机

这更接近 strict LoRA-Null：

1. \(U_{\text{ref-null}}\) 保护 CLIP 原始 zero-shot/pretrain 知识；
2. \(P_h\) 保护增量已学任务。

### 11.4 优先级

不进入第一轮。原因：

1. 需要额外实现 reference activation collection；
2. reference dataset 的选择会引入 confound；
3. 第一轮应该先验证 history-null 是否值得继续。

## 12. 第一轮实验矩阵

第一轮只跑 5 个配置：

| ID | 名称 | 家族 | 公式 | 目的 |
|---|---|---|---|---|
| A0 | `current_nsp` | baseline | \(\Delta W=BAP_h\) | 当前主基线 |
| A1 | `basis_fixed_tail` | 参数化 | \(\Delta W=BU_h^\top\) | 测试最强 basis 约束 |
| A2 | `basis_core_tail` | 参数化 | \(\Delta W=BCU_h^\top\) | 测试更合理的低秩 basis 参数化 |
| B1 | `hist_null_init_only` | 初始化 | \(BA_{\text{init}}=W_0U_hU_h^\top\), train \(BA\) | 测试 init-only 是否足够 |
| B2 | `hist_null_init_runtime` | 初始化+运行时 | init as B1, train \(BAP_h\) | 测试 init 与 runtime 是否互补 |

推荐第一轮设置：

```text
dataset_sequence: aircraft caltech101 dtd
seed: 42
num_shots: 16
iterations: 800
batch_size: 64
tune_vision_encoder: true
tune_text_encoder: false
fd_weight: 0
cd_weight: 0
aux_weight: 0
classifier_feature_transform: test
```

第一轮解释重点：

1. `basis_fixed_tail` 如果 Transfer 好但 Last 差，说明约束有效但表达力不足。
2. `basis_core_tail` 如果保留 Transfer 且 Last 明显好于 fixed，是第一组主候选。
3. `hist_null_init_only` 如果弱于 baseline，说明 runtime constraint 必要。
4. `hist_null_init_runtime` 如果优于 baseline，说明 LoRA-Null-style initialization 与 NSP 互补。

## 13. 第二轮实验矩阵

从第一轮选出 top 2，加 `current_nsp`，跑：

```text
dataset_sequence: aircraft caltech101 dtd eurosat
seeds: 42 43 44
iterations: 800
batch_size: 64
```

可做两个 setting：

1. **Mechanism setting**：vision-only, no FD/CD, no aux。
2. **Recipe setting**：加入当前论文主设置，如 text tuning schedule、FD/CD、LR-RGDA ensemble。

第二轮用于判断能否进入 full 10-task publication-scale 实验。

## 14. 实现接口建议

### 14.1 新增 CLI 参数

建议不要把所有语义塞入旧的 `init_mode`。新增：

```text
--projection_param_mode full|fixed_basis|core_basis
--basis_rank INT
--basis_window tail|middle
--null_init_mode none|history_init_only|history_init_runtime
```

含义：

```text
projection_param_mode=full
    当前 BAP 参数化。

projection_param_mode=fixed_basis
    使用 ΔW = B U^T。

projection_param_mode=core_basis
    使用 ΔW = B C U^T。

null_init_mode=history_init_only
    训练前做 history-null BA 初始化，训练时不使用 runtime P。

null_init_mode=history_init_runtime
    训练前做 history-null BA 初始化，训练时继续使用 runtime P。
```

### 14.2 模块状态

LoRA 模块需要新增：

```text
self.projection_param_mode
self.basis_U
self.basis_ready
self.C    # only for core_basis
```

### 14.3 delta 计算分支

```text
full:
    ΔW = B A P

fixed_basis:
    ΔW = B U^T

core_basis:
    ΔW = B C U^T
```

其中 `basis_ready=False` 时，应回退到 `full`，保证 Task 1 在没有 history covariance 时可训练。

### 14.4 合并逻辑

`merge_lora_weights()` 必须与 forward 使用同一套 delta 计算，否则训练和合并后行为不一致。

### 14.5 DoRA 注意事项

当前默认 `lora_class=SGPBaseDoRA`。DoRA 中需要能取到等价当前权重：

\[
W_{\text{current}} = \text{weight\_directions} \odot \text{magnitude}.
\]

如果初始化函数直接访问 `module.linear.weight`，对 DoRA 会失败或不一致。因此实现 history-null init 时必须为 LoRA/DoRA 分别提供：

```text
get_base_weight()
set_base_weight()
```

DoRA 的 residual update 应更新：

\[
\text{weight\_directions}, \text{magnitude}
\]

或先明确禁用 DoRA，用纯 LoRA 跑本轮机制消融。若要最小风险，第一轮建议显式用 `SGPBaseLoRA` 跑机制实验；若要与现有主线一致，则必须补齐 DoRA 的 weight setter。

## 15. 服务器执行计划

服务器：

```text
raoxuan@10.20.34.30
```

第一步必须检查：

```bash
ssh -o ConnectTimeout=10 -o BatchMode=yes raoxuan@10.20.34.30 "echo OK"
```

如果失败，直接退出，不做本地替代实验，不进入循环重试。

远端路径：

```text
/home/raoxuan/projects/project_clip_continual_learning
```

数据路径：

```text
/data1/open_datasets/X-TAIL
```

实现后先远端语法检查：

```bash
cd /home/raoxuan/projects/project_clip_continual_learning
python -m py_compile \
  src/models/lora_sgp.py \
  src/models/clip.py \
  src/trainers/lora_nsp_trainer.py \
  main_incremental.py
```

然后做 1-iteration smoke test：

```bash
python main_incremental.py \
  --dataset_sequence aircraft caltech101 \
  --root /data1/open_datasets/X-TAIL \
  --num_shots 16 \
  --iterations 1 \
  --batch_size 8 \
  --lora_type lora_nsp \
  --projection_param_mode core_basis \
  --basis_rank 16 \
  --basis_window tail \
  --null_init_mode none \
  --tune_text_encoder false \
  --fd_weight 0 \
  --cd_weight 0 \
  --aux_weight 0 \
  --output_dir experiments/null_basis_smoke
```

## 16. 输出与记录

每个实验必须记录：

```text
method name
projection_param_mode
basis_rank
basis_window
null_init_mode
lora_rank
tune_vision_encoder
tune_text_encoder
fd_weight
cd_weight
aux_weight
seed
dataset_sequence
git commit / git status
```

建议目录：

```text
experiments/null_basis_ablation_YYYYMMDD/
  manifest.txt
  logs/
    current_nsp_seed42.log
    basis_fixed_tail_seed42.log
    basis_core_tail_seed42.log
    hist_null_init_only_seed42.log
    hist_null_init_runtime_seed42.log
  *_results.json
  summary.csv
  summary.md
```

核心指标：

```text
Transfer
Average
Last
zero_shot
LR-RGDA
ensemble
```

额外建议记录：

```text
trainable parameter count
basis_ready ratio
effective basis rank per layer
||X_history U||_F / ||X_history||_F
```

其中：

\[
\frac{\|X_h U\|_F}{\|X_h\|_F}
\]

可以直接衡量选出的 basis 是否真的是历史低响应方向。

## 17. 判读标准

第一轮不是最终性能实验，而是机制筛选。

### 17.1 `basis_fixed_tail`

若：

```text
Transfer 上升，Last 下降
```

说明固定 basis 的稳定性有效，但塑性不足。

若：

```text
Transfer 和 Last 都下降
```

说明 tail basis 太弱或参数化过硬，应降低优先级。

### 17.2 `basis_core_tail`

若：

```text
Transfer 接近 fixed，Last 明显高于 fixed
```

说明 core basis 是第一组主候选。

若：

```text
接近 current_nsp，但参数更少
```

也有价值，可作为参数效率贡献。

### 17.3 `hist_null_init_only`

若弱于 `current_nsp`：

```text
runtime projection 必要
```

若接近或优于 `current_nsp`：

```text
初始化空间贡献很强
```

后续应实现 strict reference-null calibration。

### 17.4 `hist_null_init_runtime`

若优于 `current_nsp`：

```text
LoRA-Null-style initialization 与 runtime NSP 互补
```

这是最有论文价值的结果。

若不优于：

```text
初始化可能与 runtime projection 重复，或 BA_init 引入 residual mismatch / optimization issue
```

需检查 DoRA 实现、初始化 scale、以及 `W0' + BA_init = W0` 是否数值成立。

## 18. 推荐优先级

实现与实验优先级：

1. `current_nsp` baseline 复跑，确认协议正常。
2. `basis_fixed_tail`。
3. `basis_core_tail`。
4. `hist_null_init_only`。
5. `hist_null_init_runtime`。
6. 若 2 或 3 有趋势，再做 `basis_core_soft`。
7. 若 4 或 5 有趋势，再做 `ref_null_init_runtime`。

最终最值得进入论文主线的候选是：

```text
basis_core_tail
hist_null_init_runtime
```

前者代表参数化创新；后者代表 LoRA-Null-style initialization 与 continual runtime projection 的结合。
