# 论文实验执行计划（团队协作版）

**创建日期**：2026-07-12  
**最后更新**：2026-07-13  
**状态**：当前唯一有效的论文实验计划  
**适用对象**：实验开发、服务器运行、结果核验与论文写作协作者

---

## 实验计划概览

现阶段需要完成六组相互配合的论文实验。它们分别承担总体性能验证、组件归因、核心方法对比、超参数分析、跨模态知识保持分析和预测端分析，共同支撑论文“在持续学习新类别的同时保持 CLIP 跨模态能力”的中心结论。

第一组是 **LADA-style 主实验**。我们将在与 LADA 对齐的 X-TAIL 类增量协议下分别开展 16-shot 和 full-shot 实验，每种设置运行三个随机种子并报告均值与标准差。除持续分类的 Transfer、Average 和 Last 外，模型每完成一个增量任务都要立即进行一次图文检索评估。最终报告整个持续学习过程的 Retrieval Average，以及最后一个任务结束后的 Retrieval Last，从而同时衡量分类适应能力和跨模态能力保持。

第二组是 **整体组件消融**。该实验用于拆解完整系统中 LoRA-NF、特征蒸馏（FD）、跨模态蒸馏（CD）和集成分类器的实际贡献。FD 在已有内部实验中没有表现出稳定的类增量分类收益，但暂时保留，以进一步检查它是否有助于跨模态检索；如果最终在分类和检索上均无收益，则不写入最终方法和正文。

第三组是 **LoRA、LoRA-Null 与 LoRA-NF 的公平对比**。三种方法使用相同的知识蒸馏、LoRA rank、目标层、优化器、训练预算、任务顺序和随机种子。每种方法都报告零样本分类器和集成分类器的结果，并同步报告跨模态检索。该实验用于判断 LoRA-NF 的持续前向零空间过滤是否优于普通 LoRA，以及仅在初始化阶段利用零空间信息的 LoRA-Null。

第四组是 **LoRA-NF 超参数与应用位置消融**。重点研究 `nsp_eps`、`nsp_weight` 以及 LoRA-NF 所作用的网络层。`nsp_eps` 控制需要过滤的历史谱方向范围，`nsp_weight` 控制 Hard LoRA-NF 中历史高能量方向的泄漏强度；应用层实验比较 Attention、FFN 和 Attention + FFN。所有设置均报告零样本分类和跨模态检索，以分析不同过滤配置下的新任务适应—历史知识保持权衡。

第五组是 **跨模态蒸馏消融**。主要考察跨模态蒸馏强度和蒸馏温度的影响，并同时观察零样本分类与图文检索。该实验需要回答跨模态蒸馏是否确实延缓 CLIP 图文对齐退化，以及约束过弱或过强时会如何影响新任务学习。

第六组是 **集成分类器消融**。论文主要比较零样本分类器与“零样本分类器 + RGDA”的集成结果，并只消融二者的融合权重。RGDA 的多中心统计、拟合方式、低秩维数和分数归一化全部固定为当前候选配置，不再扩展新的消融维度。RGDA-only 结果会完整保存用于内部诊断，但由于它只覆盖已经学习的类别，不作为与零样本分类器同口径的主表方法展示。

这六组实验应共享统一的数据划分、任务顺序、评估器和结果格式。每次正式训练都要保存逐任务、逐数据集和逐随机种子的完整分类与检索结果；论文最终展示哪些表格，可以在所有结果完成后决定，但原始实验信息必须从一开始完整保留。

---

## 1. 我们要用实验回答什么

论文要证明的不是“LoRA-NF 分类精度更高”这一件事，而是：

> 在持续学习新类别的过程中，LoRA-NF 能获得强分类能力，同时更好地保持 CLIP 原有的图文检索能力。

因此，除纯预测端消融外，每个正式训练实验都必须同时评估：

1. **零样本分类**：反映训练后 CLIP 表示与文本类别语义的匹配能力；
2. **集成分类**：零样本分类器与 RGDA 的最终预测结果；
3. **跨模态检索**：反映 CLIP 原有图文对齐能力是否退化。

正式实验分成六个实验包：

| 编号 | 实验包 | 主要回答的问题 | 优先级 |
|---|---|---|---|
| E1 | LADA-style 主实验 | 完整方法总体是否足够强？ | 最高 |
| E2 | 整体组件消融 | LoRA-NF、FD、CD、Ensemble 各贡献多少？ | 最高 |
| E3 | LoRA 系列公平对比 | LoRA-NF 是否优于 LoRA 和 LoRA-Null？ | 最高 |
| E4 | LoRA-NF 超参数 | 过滤范围、泄漏强度和应用层如何影响结果？ | 高 |
| E5 | 跨模态蒸馏消融 | 蒸馏强度和温度如何影响分类与检索？ | 高 |
| E6 | 集成分类器消融 | 不同融合权重下，RGDA 与零样本分支如何互补？ | 高 |

## 2. 所有实验共同遵守的协议

### 2.1 数据与模型

| 项目 | 固定设置 |
|---|---|
| Backbone | `openai/clip-vit-base-patch16` |
| 持续学习数据 | X-TAIL 十数据集 |
| 主任务顺序 | aircraft → caltech101 → dtd → eurosat → flowers → food101 → mnist → oxford_pets → stanford_cars → sun397 |
| 数据规模 | 主实验分别运行 16-shot 和 full-shot；消融默认只运行 16-shot |
| 随机种子 | 正式主实验 `{42, 43, 44}`；普通超参数消融先用固定 seed，关键结论再补 3 seeds |
| 检索评估 | 每完成一个任务立即评估一次，而不是只评估最终模型 |

检索数据集尚未最终冻结。选择前必须确认它与跨模态蒸馏 reference data 没有数据泄漏。若蒸馏使用 Flickr30K，Flickr30K 不能直接作为唯一的无污染检索测试集。

### 2.2 当前候选训练配置

除被消融的变量外，其余参数固定为当前 V4/V5 候选配置：

```text
adapter                 = LoRA-NF
use_soft_projection     = false
lora_rank               = 4
target_modules          = q_proj,k_proj,v_proj,out_proj,fc1,fc2
nsp_eps                 = 0.20
nsp_weight              = 0.02
optimizer               = AdamW
learning_rate           = 1e-4
weight_decay            = 3e-5
batch_size              = 32
iterations              = 800
scheduler               = cosine_with_warmup
fd_weight               = 1.0
cd_weight               = 2.0
cd_temperature          = 4.0
aux_weight              = 0.0
text_tuning_schedule    = always
num_centers             = 4
rgda_train_iter         = 200
ensemble_normalization  = maxshift
```

这只是正式实验的候选冻结点。开始大规模运行前，需要由一名协作者逐项核对 CLI、trainer 和模型实际收到的值。

### 2.3 统一报告指标

分类结果分别报告：

- Zero-shot Transfer / Average / Last；
- Ensemble Transfer / Average / Last；
- 3 seeds 的 `mean ± std`。

RGDA-only 结果必须保存，但不进入论文主表。原因是 RGDA 只覆盖已经学习并建立统计量的类别，与覆盖完整类别集合的零样本分类不是同一评估口径。

检索在任务 (t) 结束后得到指标 (R_t)。论文报告：

\[
R_{\mathrm{Avg}}=\frac{1}{T}\sum_{t=1}^{T}R_t,
\qquad
R_{\mathrm{Last}}=R_T.
\]

每个 (R_t) 完整保存：

- Image-to-Text R@1/R@5/R@10；
- Text-to-Image R@1/R@5/R@10。

主文可以用双向 mean recall 压缩表格，但原始六项指标不能丢失。

### 2.4 可直接复用的公共命令

以下命令是所有 16-shot 正式训练的基础模板。协作者只修改 GPU、seed、实验名和对应实验包的覆盖参数。

```bash
python -u main_incremental.py \
  --root /data1/open_datasets/X-TAIL \
  --dataset_sequence aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397 \
  --num_shots 16 \
  --batch_size 32 \
  --eval_batch_size 128 \
  --iterations 800 \
  --train_budget_mode uniform \
  --optimizer adamw \
  --lr 1e-4 \
  --weight_decay 3e-5 \
  --scheduler cosine_with_warmup \
  --warmup_ratio 0.1 \
  --eta_min 0.0 \
  --lora_type lora_nsp \
  --init_mode lora_nsp \
  --use_dora false \
  --lora_rank 4 \
  --lora_alpha 4 \
  --lora_dropout 0.0 \
  --lora_target_modules q_proj,k_proj,v_proj,out_proj,fc1,fc2 \
  --projection_param_mode full \
  --null_init_mode none \
  --nsp_eps 0.20 \
  --nsp_weight 0.02 \
  --reference_dataset flickr8k \
  --reference_batch_size 32 \
  --fd_weight 1.0 \
  --cd_weight 2.0 \
  --cd_divergence kl_forward \
  --cd_temperature 4.0 \
  --aux_weight 0.0 \
  --tune_vision_encoder true \
  --tune_text_encoder true \
  --text_lora_rank 4 \
  --text_tuning_schedule always \
  --text_classifier_mode lada_hybrid \
  --classifier_feature_transform test \
  --rgda_rank 32 \
  --rgda_alpha1 0.2 \
  --rgda_alpha2 2.0 \
  --rgda_alpha3 0.5 \
  --num_centers 4 \
  --rgda_train_iter 200 \
  --rgda_train_lr 0.01 \
  --rgda_fit_source gmm_sample \
  --alpha 0.05 \
  --ensemble_normalize maxshift \
  --temperature 1.0 \
  --enable_retrieval_eval \
  --retrieval_datasets mscoco_2014_5k,flickr30k_hf \
  --retrieval_batch_size 128 \
  --retrieval_recall_ks 1,5,10 \
  --retrieval_max_images 0 \
  --disable_lada \
  --seed 43 \
  --gpu 0 \
  --output_dir experiments/paper_formal/E1 \
  --experiment_name E1__lora_nf__16shot__seed43
```

实施说明：

- `--lora_alpha 4` 显式写出，等价于当前 `None → rank` 的默认行为，避免默认值以后变化；
- `--retrieval_max_images 0` 表示使用完整检索测试集；
- 不需要同时传 `--device` 和 `--gpu`；正式 launcher 统一使用 `--gpu`；
- `--use_soft_projection` 是无值开关：Hard NSP 时不要传，Soft NSP 时才传；
- `--full_shot` 同样是无值开关，不能写成 `--full_shot true`；
- 主表运行外部 LADA 比较时移除 `--disable_lada`，并显式填写 LADA 参数；
- 检索数据集最终冻结后，应同步替换模板中的 `retrieval_datasets/roots`。

### 2.5 16-shot 与 full-shot 的唯一差别

16-shot 使用：

```bash
--num_shots 16
```

Full-shot 使用：

```bash
--full_shot
```

Full-shot 下仍可保留 `--num_shots 16` 作为 parser 默认占位，但数据加载必须由 `--full_shot` 覆盖。正式运行前要核验输出 JSON 同时记录 `full_shot=true`；当前单分类器 JSON 只写 `num_shots`，容易把 full-shot 误标为 16-shot，这一元数据问题必须先修复。

### 2.6 三种方法的精确参数覆盖

在公共命令基础上使用：

| 方法 | 必须覆盖的 CLI 参数 |
|---|---|
| Standard LoRA | `--lora_type lora_vanilla --init_mode lora_vanilla --null_init_mode none` |
| LoRA-Null | `--lora_type lora_nsp --init_mode lora_nsp --projection_param_mode full --null_init_mode history_init_only` |
| LoRA-NF | `--lora_type lora_nsp --init_mode lora_nsp --projection_param_mode full --null_init_mode none --nsp_eps 0.20 --nsp_weight 0.02` |

LoRA-Null 的当前代码路径会用历史零空间初始化 adapter，随后把运行时 (P) 设为单位阵。正式跑前必须用最小测试再次验证“初始函数等价”和“训练阶段不再使用 (P)”两点。

## 3. E1：LADA-style 主实验

### 3.1 目的

回答完整方法在 16-shot 和 full-shot 下是否达到强竞争力，并展示分类能力与跨模态检索保持。

### 3.2 实验矩阵

每种方法分别运行：

```text
2 个 shot 设置 × 3 个 seeds = 6 次完整持续学习
```

| 方法 | 16-shot | Full-shot | Zero-shot | Ensemble | Retrieval |
|---|:---:|:---:|:---:|:---:|:---:|
| Frozen CLIP | ✓ | ✓ | ✓ | — | ✓ |
| LADA | ✓ | ✓ | 按其正式协议 | 按其正式协议 | 若 encoder 可评估则 ✓ |
| 其他可可靠复现的强基线 | ✓ | ✓ | ✓ | 视方法而定 | ✓ |
| Standard LoRA + distillation | ✓ | ✓ | ✓ | ✓ | ✓ |
| LoRA-NF + distillation | ✓ | ✓ | ✓ | ✓ | ✓ |

LADA 必须使用完整公开配方，明确 DPT、文本侧调优、label-specific memory 和融合分支是否启用。原论文数字与本项目统一协议复现数字必须分开标注。

### 3.3 超参数与运行数量

内部主方法不单独调参，全部使用第 2.2 节配置。每个需要训练的方法：

| Shot | Seed | 额外参数 |
|---|---|---|
| 16-shot | 42/43/44 | `--num_shots 16` |
| Full-shot | 42/43/44 | `--full_shot` |

因此，每增加一个训练方法就是 `2 shot × 3 seeds = 6 runs`。当前两个必须重跑的内部训练方法（Standard LoRA、LoRA-NF）共 12 runs；Frozen CLIP 不训练。LADA 和其他外部基线按最终纳入的方法数各增加 6 runs。

Full-shot 暂时仍使用每任务 800 iterations，以保持方法间相同更新预算。如果最终决定按 LADA 官方 epoch recipe 比较，则所有相关方法必须统一切换到 `--train_budget_mode lada_epochs`，不能只给 LADA 更多训练预算。

### 3.4 评估时点

- 训练前：保存原始 CLIP 检索结果 (R_0)；
- 每个任务训练后：评估分类和检索；
- 第十个任务后：形成 Last；
- 十个训练阶段平均：形成 Average。

### 3.5 论文产出

- 主表 1：16-shot 分类；
- 主表 2：full-shot 分类；
- 主表 3：16-shot/full-shot 检索 Average 与 Last；
- 附录：逐任务、逐数据集、逐 seed 和完整双向 R@K。

## 4. E2：整体组件消融

### 4.1 目的

回答完整方法的收益来自哪个组件，并检查 FD 是否确实可以从最终方法中删除。

### 4.2 实验矩阵

统一使用 LoRA-NF、16-shot 和当前候选配置：

| 编号 | FD | CD | Ensemble | 解释 |
|---|:---:|:---:|:---:|---|
| C0 | ✗ | ✗ | ✗ | 纯 LoRA-NF |
| C1 | ✓ | ✗ | ✗ | FD 的独立贡献 |
| C2 | ✗ | ✓ | ✗ | CD 的独立贡献 |
| C3 | ✓ | ✓ | ✗ | 完整训练端 |
| C4 | ✓ | ✓ | ✓ | 当前完整系统 |

另外保留 `Standard LoRA + FD + CD`，用于判断蒸馏收益是否依赖 LoRA-NF。

精确覆盖参数：

| 编号 | CLI 覆盖 |
|---|---|
| C0 | `--fd_weight 0 --cd_weight 0` |
| C1 | `--fd_weight 1 --cd_weight 0` |
| C2 | `--fd_weight 0 --cd_weight 2 --cd_temperature 4` |
| C3/C4 | `--fd_weight 1 --cd_weight 2 --cd_temperature 4` |

C3 和 C4 是同一次 encoder 训练：C4 只是读取 C3 的 Ensemble 结果，不得重复训练。单 seed 需要 4 个 LoRA-NF 训练 runs；若整组补齐 3 seeds，则为 12 runs。Standard LoRA + FD + CD 直接复用 E3 的 Standard LoRA 正式 run。

### 4.3 FD 的处理原则

已有内部实验显示 FD 对类增量分类没有稳定收益，但尚未完整验证检索侧。因此：

- 正式实验先保留 FD；
- 如果分类和检索都没有收益，最终正文和方法定义可以删除 FD；
- 即使删除，完整实验结果仍保留在原始产物或附录中。

### 4.4 必须报告

- C0–C3：Zero-shot 分类 + Retrieval Average/Last；
- C4：再增加 Ensemble 分类；
- RGDA-only：只保存在内部结果中。

## 5. E3：LoRA、LoRA-Null、LoRA-NF 公平对比

### 5.1 目的

这是验证 LoRA-NF 主创新最重要的实验。它回答持续前向过滤是否优于普通低秩适配和仅在初始化时使用零空间。

### 5.2 对比方法

| 方法 | 形式 | 零空间信息何时生效 |
|---|---|---|
| LoRA | \(AB\) | 不使用 |
| LoRA-Null | 初始化后仍优化普通 \(AB\) | 仅初始化时 |
| LoRA-NF | \(PAB\) | 整个训练过程持续生效 |
| Gradient-projected LoRA（建议） | 普通前向，反向后投影梯度 | 参数更新时 |

前三项为必须项，梯度投影是强烈建议项，因为它直接对应论文对传统零空间投影的讨论。

### 5.3 公平性要求

所有方法必须使用完全相同的：

- rank 和目标层；
- FD/CD 配置；
- optimizer、lr、batch size 和 iterations；
- 数据、任务顺序和 seed；
- Zero-shot 和 Ensemble 评估器。

正式结果使用 3 seeds。旧 init-only/runtime 实验只能用于预估趋势，不能代替正式结果。

精确运行矩阵：

| 方法 | Seed | 训练次数 |
|---|---|---:|
| Standard LoRA + FD1 + CD2 | 42/43/44 | 3 |
| LoRA-Null + FD1 + CD2 | 42/43/44 | 3 |
| LoRA-NF + FD1 + CD2 | 42/43/44 | 3 |

最低共 9 runs。若加入 Gradient-projected LoRA，再增加 3 runs。Standard LoRA 与 LoRA-NF 的结果应与 E1 16-shot 主实验复用，不要因实验包名称不同重复训练。

### 5.4 必须报告

- Zero-shot Transfer/Average/Last；
- Ensemble Transfer/Average/Last；
- Retrieval Average/Last；
- 完整逐任务曲线。

## 6. E4：LoRA-NF 超参数消融

这一组默认使用 16-shot。每次只改变一个变量，其他参数保持不变。

### 6.1 `nsp_eps`：过滤范围

`nsp_eps` 决定哪些历史谱方向被划入需要抑制的高能量子空间。

```text
nsp_eps    = {0.02, 0.05, 0.10, 0.20}
nsp_weight = 0.02（固定）
hard NSP   = true
```

已有有效探索表明 `0.20` 是当前候选最优值，但正式实验需要增加跨模态检索。

对应实验名与覆盖项：

| 实验名后缀 | CLI 覆盖 |
|---|---|
| `eps0p02` | `--nsp_eps 0.02 --nsp_weight 0.02` |
| `eps0p05` | `--nsp_eps 0.05 --nsp_weight 0.02` |
| `eps0p10` | `--nsp_eps 0.10 --nsp_weight 0.02` |
| `eps0p20` | `--nsp_eps 0.20 --nsp_weight 0.02` |

### 6.2 `nsp_weight`：泄漏强度

Hard LoRA-NF 使用：

\[
P=(1-w)P_0+wI,
\qquad w=\texttt{nsp\_weight}.
\]

正式网格：

```text
nsp_weight = {0, 0.02, 0.04, 0.08, 0.16}
nsp_eps    = 0.20（固定）
hard NSP   = true
```

- `0`：严格投影；
- `0.02`：当前默认值；
- 其余取值：逐渐增加历史方向泄漏。

只有 `nsp_weight>0` 时，当前 (P) 满秩，才能使用“不缩减同 rank LoRA 表达集合”的理论表述。

**历史纠错**：旧 Phase 7a 在 Soft NSP 下扫描 `nsp_weight`，但 Soft 分支不读取这个参数。因此旧实验无效，不能用于论文，也不能说明 `nsp_weight` 不敏感。

对应实验名与覆盖项：

| 实验名后缀 | CLI 覆盖 |
|---|---|
| `weight0` | `--nsp_eps 0.20 --nsp_weight 0` |
| `weight0p02` | `--nsp_eps 0.20 --nsp_weight 0.02` |
| `weight0p04` | `--nsp_eps 0.20 --nsp_weight 0.04` |
| `weight0p08` | `--nsp_eps 0.20 --nsp_weight 0.08` |
| `weight0p16` | `--nsp_eps 0.20 --nsp_weight 0.16` |

### 6.3 LoRA-NF 应用层

| 设置 | 目标模块 | 用途 |
|---|---|---|
| Attention | q/k/v/out | 只过滤注意力适配 |
| FFN | fc1/fc2 | 只过滤前馈层适配 |
| Attention + FFN | q/k/v/out/fc1/fc2 | 当前完整设置 |

对应 CLI：

| 设置 | CLI 覆盖 |
|---|---|
| Attention | `--lora_target_modules q_proj,k_proj,v_proj,out_proj` |
| FFN | `--lora_target_modules fc1,fc2` |
| Attention + FFN | `--lora_target_modules q_proj,k_proj,v_proj,out_proj,fc1,fc2` |

Q/K/V/O 的更细拆分和 QKV 共享 (P) 结果放附录，不作为本轮正文核心。

### 6.4 必须报告

- Zero-shot Transfer/Average/Last；
- Retrieval Average/Last；
- Ensemble Average/Last 作为补充；
- 应用层消融额外报告 (P) 状态大小、训练时间和峰值显存。

E4 在单 seed 下共有 10 个不重复训练点：

- `nsp_eps`：4 runs；
- `nsp_weight`：除复用默认点外新增 4 runs；
- 应用层：除复用全层默认点外新增 2 runs。

先统一使用 seed 43。完成后对默认点和曲线中最关键的两个端点补 seeds 42/44，不要求一开始就把全部 10 点跑成三种子。

## 7. E5：跨模态蒸馏消融

### 7.1 目的

回答 CD 是否主要帮助保持图文对齐，以及约束过弱或过强时如何影响新任务学习。

### 7.2 蒸馏强度

固定 temperature 和其他参数，只改变 `cd_weight`。正式取值尚待最后确认，但必须包含：

- `0`：无 CD；
- 当前候选值 `2.0`；
- 至少一个更弱和一个更强的值。

建议候选：

```text
cd_weight = {0, 0.5, 1.0, 2.0, 4.0}
```

每个点固定：

```text
--fd_weight 0
--cd_temperature 4.0
--cd_divergence kl_forward
```

这里固定 `fd_weight=0` 是为了让曲线只反映 CD；组件表 C3 仍用于研究 FD/CD 交互。

### 7.3 蒸馏温度

固定 `cd_weight=2.0`，建议：

```text
cd_temperature = {1.0, 2.0, 4.0, 8.0}
```

每个点固定：

```text
--fd_weight 0
--cd_weight 2.0
--cd_divergence kl_forward
```

已有分类探索倾向 `4.0`，但正式选择必须同时查看检索结果。

### 7.4 必须报告

- Zero-shot Transfer/Average/Last；
- Retrieval Average/Last；
- Ensemble 作为补充；
- 每任务 CD loss 与完整检索曲线。

散度形式和 reference batch size 不作为正文重点，除非正式结果出现异常。

两个扫描共 8 个不重复 runs：5 个 strength 点 + 4 个 temperature 点 - 1 个公共点（weight 2、temperature 4）。先用 seed 43；最终默认点、无 CD 点和最有代表性的强约束点补 seeds 42/44。

## 8. E6：集成分类器消融

### 8.1 目的

回答监督类别统计是否能补充 CLIP 文本分类器，并验证提升不是由测试集调参得到的。

### 8.2 论文主要比较

| 预测方式 | 是否进入主文 |
|---|:---:|
| Zero-shot classifier | ✓ |
| Zero-shot + RGDA Ensemble | ✓ |
| RGDA-only | ✗，仅内部诊断 |

### 8.3 消融内容

E6 只改变 Ensemble weight，即 `--alpha`。其他预测端参数固定为：

```text
ensemble_normalize = maxshift
rgda_rank           = 32
rgda_alpha1         = 0.2
rgda_alpha2         = 2.0
rgda_alpha3         = 0.5
num_centers         = 4
rgda_train_iter     = 200
rgda_train_lr       = 0.01
rgda_fit_source     = gmm_sample
```

主结果必须固定一个由 validation protocol 决定的全局 alpha。对测试集逐任务或逐数据集选择的最佳 alpha 只能作为 oracle 分析，不能作为正式方法结果。

检索不受分类器影响，因此 E6 不需要重复运行检索；直接引用同一 encoder checkpoint 的检索结果。

正式融合权重网格：

| 变量 | 取值 |
|---|---|
| Ensemble alpha | `{0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0}` |

其中 `alpha=0` 是纯零样本分类器，`alpha=1` 是融合公式中的 RGDA 端点。E6 优先从同一 encoder checkpoint 保存的 logits/features 离线计算，不重复训练 encoder，也不重复进行跨模态检索。

## 9. 结果文件必须保存什么

### 9.1 当前 `main_incremental.py` 已经生成

设 `--experiment_name NAME --output_dir DIR`，当前代码会生成：

```text
DIR/NAME.json                 # 总配置、分类器汇总和文件索引
DIR/NAME_zs_results.json      # Zero-shot accuracy matrix 与 T/A/L
DIR/NAME_rgda_results.json    # RGDA-only 内部诊断
DIR/NAME_ens_results.json     # Ensemble accuracy matrix 与 T/A/L
DIR/NAME_retrieval.json       # 启用检索后，每任务、每检索数据集的双向 R@K
DIR/NAME_lada_results.json    # 仅启用 LADA 时
DIR/NAME_lada_zs_results.json # 仅启用 LADA 时
```

总 JSON 的 `arguments` 字段保存实际解析参数。任何汇总脚本必须优先读取它，而不是反推启动脚本。

### 9.2 Gate 0 需要补齐或确认

为了达到论文级可追溯性，每个 run 还应具备：

```text
run_manifest.json             # commit、数据 split、任务顺序、seed、开始/结束时间和状态
loss_history.json/csv         # 总损失、FD、CD 等；若当前代码未保存则补实现
resource_usage.json           # wall-clock、峰值显存、P/统计状态大小
checkpoints_or_artifacts/      # 每任务 adapter/classifier state 或可离线重建的 step artifact
stdout.log
```

若使用 `--save_step_artifacts`，应显式传 `--async_eval_dir`，并确认每个任务的 artifact 均存在。正式运行默认保留 inline evaluation；只有异步评估流水线通过自测后才使用 `--skip_inline_eval`。

推荐实验名：

```text
{experiment_pack}__{method}__{shots}shot__seed{seed}__{key_override}
```

例如：

```text
E4__lora_nf__16shot__seed43__nsp_weight0p08
```

禁止用 `test1`、`new_best`、`final_v2` 等无法追溯的名称。

### 9.3 目录与实验名示例

```text
experiments/paper_formal/
├── E1_main/
├── E2_components/
├── E3_adapters/
├── E4_lora_nf_hparams/
├── E5_distillation/
└── E6_ensemble/
```

同一配置重跑时不得覆盖原结果。应先检查目标 stem 是否已存在；需要重跑时在 manifest 中记录原因，并使用显式的 `__rerun1` 后缀。

## 10. 团队执行流程

### Gate 0：正式实验启动前

- [ ] 冻结 16-shot/full-shot 数据 split 与任务顺序；
- [ ] 确定检索数据集，并完成与 reference data 的去重审计；
- [ ] 核对所有 CLI 参数确实进入目标代码分支；
- [ ] 日志打印实际配置、任务顺序和数据规模；
- [ ] 改变 `nsp_eps/nsp_weight` 后，单元检查确认 (P) 确实变化；
- [ ] 固定 Ensemble alpha 的 validation 选择规则；
- [ ] 修复或确认 `--alpha_sensitivity` 的关闭方式；当前 help 提到 `--no-alpha_sensitivity`，但 parser 未注册该参数；
- [ ] Full-shot 分类结果 JSON 显式保存 `full_shot=true`，不能只显示 `num_shots=16`；
- [ ] 冻结结果 JSON schema；
- [ ] 用一个任务和一个 seed 完成端到端 dry run；
- [ ] 指标审计脚本通过。

### Wave 1：先跑出最小论文闭环

1. E1 的 16-shot：LoRA、LoRA-NF、完整方法，3 seeds；
2. 同步完成每任务检索；
3. E3：LoRA、LoRA-Null、LoRA-NF，3 seeds；
4. E2：FD/CD 组件消融。

这一波完成后，论文最核心的分类、检索和方法有效性就有初步闭环。

### Wave 2：主表与关键消融

1. LADA 和其他强基线；
2. E1 full-shot，3 seeds；
3. E4 的 `nsp_eps`、`nsp_weight` 和应用层；
4. E5 蒸馏强度和温度。

### Wave 3：预测端、机制和附录

1. E6 集成分类器；
2. 谱方向更新、历史表示漂移等机制指标；
3. 效率、额外任务顺序和其他稳健性结果；
4. 逐数据集、逐任务和逐 seed 附录表。

### 10.1 去重后的最低训练量

下表只计算 encoder 训练，不计算 Frozen CLIP 和离线 E6 classifier sweep：

| 实验包 | 最低训练 runs | 说明 |
|---|---:|---|
| E1 内部主方法 | 12 | LoRA/LoRA-NF × 16/full-shot × 3 seeds |
| E2 组件 | 12 | 4 个训练组件配置 × 3 seeds；其中完整 LoRA-NF 可复用 E1 |
| E3 adapter | 9 | 3 adapters × 3 seeds；LoRA/LoRA-NF 可复用 E1，实际新增主要是 LoRA-Null |
| E4 超参数 | 10 | 单 seed 去重后；关键端点再补种子 |
| E5 CD | 8 | 单 seed 去重后；关键端点再补种子 |
| E6 Ensemble alpha | 0 | 固定全部 RGDA 配置，只使用已保存 checkpoint/artifact 离线扫融合权重 |

这些数字不能直接相加：E1、E2、E3 之间存在 checkpoint 复用。实验登记表必须用完整配置哈希判断是否已有可复用 run，不能只看实验名称。

## 11. 单个实验的验收标准

一个 run 只有同时满足以下条件才算完成：

- [ ] 配置与计划一致；
- [ ] 所有十个任务均成功结束；
- [ ] 每个任务后的分类和检索结果齐全；
- [ ] 没有 NaN、OOM、静默参数失效或中途恢复污染；
- [ ] 原始日志和机器可读结果均存在；
- [ ] 聚合结果可以从原始结果重新计算；
- [ ] seed、split、任务顺序和代码版本可追溯；
- [ ] 自动审计通过。

只有通过验收的 3-seed 结果才能进入论文主表。旧实验若受 parser、参数未接线、检索 split 或 LR-RGDA 默认值影响，只能用于调试和趋势参考。

## 12. 协作分工表

开始执行时，在下表填写负责人和状态。状态统一使用：`待开始 / 开发中 / 运行中 / 待核验 / 已完成 / 阻塞`。

| 工作项 | 负责人 | 状态 | 结果路径/备注 |
|---|---|---|---|
| 检索数据与去重协议 | 待分配 | 待开始 | |
| 统一结果 schema | 待分配 | 待开始 | |
| E1 16-shot | 待分配 | 待开始 | |
| E1 full-shot | 待分配 | 待开始 | |
| E2 组件消融 | 待分配 | 待开始 | |
| E3 LoRA 系列对比 | 待分配 | 待开始 | |
| E4 `nsp_eps` | 待分配 | 待开始 | |
| E4 `nsp_weight` | 待分配 | 待开始 | |
| E4 应用层 | 待分配 | 待开始 | |
| E5 CD weight | 待分配 | 待开始 | |
| E5 CD temperature | 待分配 | 待开始 | |
| E6 Ensemble | 待分配 | 待开始 | |
| 结果审计与汇总 | 待分配 | 待开始 | |

## 13. 尚未冻结的事项

- [ ] 最终检索数据集及 reference data 去重方案；
- [ ] LADA 之外进入主表的强基线；
- [ ] `cd_weight` 与 temperature 的最终网格；
- [ ] Ensemble alpha 的 validation protocol；
- [ ] full-shot 的精确定义及与 LADA 的一致性；
- [ ] Gradient-projected LoRA 是否纳入正文核心对照；
- [ ] checkpoint 全量保存还是保存可重建 adapter/state；
- [ ] 服务器 GPU 预算与各实验包负责人。

这些事项未冻结前，可以开发评估与结果保存代码，但不应启动大规模正式实验。

## 14. 论文图表对应关系

| 论文内容 | 来源实验 |
|---|---|
| 16-shot/full-shot 持续学习主表 | E1 |
| 跨模态检索 Average/Last | E1 |
| 分类—检索联合曲线 | E1 + E3 |
| 组件消融 | E2 |
| LoRA/LoRA-Null/LoRA-NF | E3 |
| `nsp_eps/nsp_weight/层` | E4 |
| 蒸馏强度与温度 | E5 |
| Zero-shot 与 Ensemble | E6 |
| 完整逐任务、逐数据集、逐 seed 表 | 所有实验的原始结果 |

在正式结果完成前，不在摘要和引言中提前写“达到 SOTA”“严格保持检索能力”或“严格零干扰”。
