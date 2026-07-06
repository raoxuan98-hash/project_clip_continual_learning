# 10-task 重消融实验方案（修订版 v5）

**日期**: 2026-07-06 → 2026-07-07
**修订**: v5 — Pre-Wave A/Phase 1 完成并记录结果；新增 Phase 6–8 (Soft/Hard NSP, nsp_eps, eta_min)；修复 use_soft_projection/eta_min CLI 缺失；§13 重写；砍 Wave B/E。

---

## 目录

1. [背景与动机](#1-背景与动机)
    1.1 问题定义：视觉持续学习中的灾难性遗忘
    1.2 基础架构：CLIP + LoRA-NSP
    1.3 LoRA vs DoRA
    1.4 LADA 集成分类器
    1.5 指标定义
    1.6 6-task 阶段回顾
    1.7 为什么需要 10-task 重消融
    1.8 数据集与协议
    1.9 LADA 评估指标
2. [已知事实：6-task 消融全景](#2-已知事实6-task-消融全景)
3. [核心矛盾：6-task SOTA 迁移到 10-task 退化](#3-核心矛盾6-task-sota-迁移到-10-task-退化)
4. [实验目标与指标](#4-实验目标与指标)
5. [公共配置](#5-公共配置)
6. [Pre-Wave A：DoRA Anchor 复现](#6-pre-wave-adora-anchor-复现)
7. [Wave A：DoRA vs LoRA × 学习率](#7-wave-adora-vs-lora--学习率)
8. [Wave B：优化器对照（AdamW vs SGD）](#8-wave-b优化器对照adamw-vs-sgd)
9. [Wave C：Batch Size](#9-wave-cbatch-size)
10. [Wave D：CD 温度](#10-wave-dcd-温度)
11. [Wave E：aux_weight](#11-wave-eaux_weight)
12. [Wave F：文本端微调策略](#12-wave-f文本端微调策略)
13. [实验汇总表](#13-实验汇总表)
14. [执行顺序与依赖](#14-执行顺序与依赖)
15. [执行策略（修订）](#15-执行策略修订)
16. [成功标准](#16-成功标准)

---

## 1. 背景与动机

### 1.1 问题定义：视觉持续学习中的灾难性遗忘

持续学习（Continual Learning）的核心挑战是**灾难性遗忘**：模型在顺序学习新任务时，会覆盖之前学到的知识，导致在旧任务上精度骤降。我们的场景是 **X-TAIL 细粒度图像分类**——10 个视觉分类数据集依次到达，每个数据集仅提供 16 个标注样本（16-shot），模型需要在不存储原始数据、不回放样本的前提下，持续学习所有任务。

这本质上是 **replay-free class-incremental learning**：每个新任务引入未见过的类别，分类头随任务数线性增长。到第 10 个任务，模型需在 10 个数据集的联合标签空间上做推理。

### 1.2 基础架构：CLIP + LoRA-NSP

**Backbone**：OpenAI CLIP ViT-B/16（视觉编码器 12 层 Transformer，文本编码器 12 层 Transformer，共享特征空间维度 d=512）。CLIP 的双塔结构天然适合零样本分类——文本编码器将类别名编码为 text embedding，视觉编码器提取图像特征，余弦相似度最高的类别即为预测。

**LoRA-NSP（Null-Space Projection）** 是我们的核心技术创新。标准 LoRA（Low-Rank Adaptation）在预训练权重旁添加低秩增量 ΔW = BA，其中 B∈R^(d×r)、A∈R^(r×d)、r≪d。LoRA-NSP 的关键改进是**在 LoRA delta 上施加零空间投影**：

```
ΔW_nsp = B @ A @ P
```

其中 P 是投影矩阵，将 LoRA 学习到的更新从全局空间投影到前序任务零空间的正交补上。具体来说：

1. **协方差累积**：每个任务训练完成后，计算该任务在视觉编码器输出层的特征协方差矩阵 Σ_t。
2. **SVD 分解**：对累积协方差 Σ_global 做 SVD，取前 k 个主成分张成"已占据子空间"。
3. **投影**：新任务的 LoRA delta 被投影到已占据子空间的零空间（null space）上，避免干扰旧任务的特征方向。

> **注意**：LoRA-NSP 中的 NSP 机制独立于底层适配器类型——可在 LoRA 或 DoRA 上叠加。历史上 82.79 的配置是 **DoRA + NSP**（旧代码 `use_dora` 默认为 `true`）。

**变体**：我们维护两种 NSP 策略：
- `current_nsp`：仅用当前任务协方差做投影（本次默认）
- `hist_null_init_runtime`：用历史协方差做投影，初始化时机不同

**文本端策略**：`text_tuning_schedule` 控制文本编码器的微调程度。四种模式：
- `freeze_after`：Task 1 正常微调，之后冻结（text_lr = 0）
- `low_lr_after`：Task 1 后降为低学习率持续微调（text_lr = lr × 0.2）
- `always`：全程与视觉端同学习率微调
- `never`：全程冻结文本编码器

历史上 82.79 是在 `low_lr_after`（旧代码无此参数，行为最接近 `low_lr_after`）下产生的。本轮公共配置默认采用 `low_lr_after`。

### 1.3 LoRA vs DoRA

**DoRA（Weight-Decomposed Low-Rank Adaptation）** 将预训练权重 W 分解为 **magnitude（幅度）** 和 **direction（方向）** 两部分：

```
W_adapted = magnitude ⊙ (direction + ΔW_direction) / ||direction + ΔW_direction||
```

其中 magnitude 是独立学习的标量向量，direction 是 W 的归一化方向矩阵加上 LoRA 增量。这与标准 LoRA 的 W_adapted = W + BA 不同——DoRA 声称通过解耦幅度和方向，在学习模式上更接近 full fine-tuning。

在 6-task 消融中，DoRA 未表现出明确优势。但我们有一个关键对照点：6 月 18 日的旧代码在 10-task 上跑 DoRA 曾达到 **Ens Last 82.79**（详见 §3.2）。

### 1.4 LADA 集成分类器

我们使用两种分类器评估模型：

**Zero-shot（ZS）分类器**：直接用 CLIP 文本编码器生成的类别 embedding 与视觉特征做余弦相似度匹配。不依赖任何训练样本的统计信息。这是衡量模型零样本泛化能力的最纯粹指标。

**集成分类器（Ensemble）**：将 Zero-shot 分类器与 LR-RGDA（低秩岭回归判别分析）分类器加权融合：

```
logits_ens = (1-α) × logits_zs + α × logits_lrrgda
```

LR-RGDA 在每个任务训练后，利用已见类别的少量样本（16-shot）拟合 GMM（高斯混合模型）特征分布，再通过岭回归学习分类边界。α 在 [0, 1] 上做 21 点 sweep，取验证集最优值。我们报告最优 α 下的精度。

此外还计算 **LADA** 分类器（基于 k-means 聚类中心的最近邻），作为补充参考但不作为主要指标。

### 1.5 指标定义

所有指标遵循 LADA 论文 Appendix A 的严格定义，从 `*_zs_results.json` / `*_ens_results.json` 的 `metrics` 字段读取，×100 为百分比。**Average 不等于 (Transfer + Last) / 2**——它是每个任务在所有训练步骤上的平均准确率再求均值。

| 指标 | 计算方式 | 含义 |
|---|---|---|
| Transfer | 学习任务 t 后，在前 t−1 个任务上的平均准确率，再对 t=2..T 取均值。Task 1 无前序任务，Transfer 为 N/A，不计入均值 | 衡量知识迁移与遗忘——值越高说明模型在新任务上学习时保留了旧知识 |
| Average | 每个任务在所有训练步骤上的平均准确率，再对全部 T 个任务取均值 | 综合衡量整个学习轨迹的精度，对训练过程中的波动不敏感 |
| Last | 最终模型（训练完第 T 个任务后）在每个任务上的准确率均值 | 衡量最终交付模型的性能 |

我们对 **Zero-shot 分类器**和**集成分类器**分别计算这三个指标，得到完整六指标。**ZS Average 是第一优先级指标**——它直接反映增量学习后的零样本泛化能力，不受分类器超参数影响。

### 1.6 6-task 阶段回顾

过去一周（6/30–7/5），我们在 **6-task**（aircraft → caltech101 → dtd → eurosat → flowers → oxford_pets，16-shot，seed=42）上完成了全面的单变量消融，目标是收窄超参数空间。实验记录在 `chat-history/2026-07-05-weekly-ablation-review.md`，关键路线如下：

1. **优化器**：AdamW/Adam/RMSprop 为第一梯队（ZS Average 差异 <0.1pp），SGD 显著落后（ZS Average 65.0 vs 68.6）。选定 AdamW。
2. **学习率**：AdamW lr=1e-4 为 Transfer/Average/Last 最佳平衡点。3e-4 的 Last 更高但 Transfer 下降 2.6pp。
3. **SGD 学习率**：lr 越大 Last 越高但 Transfer 越低。lr=5e-3 的 Ens Last 达 82.24，但 ZS Transfer 仅 57.44。整体不如 AdamW。
4. **Batch Size**：越大越好——bs=128 > bs=64 > bs=32，ZS Average 从 67.78 → 68.59 → 69.06。但 Transfer 轻微下降。
5. **CD 温度**：temp=4.0 在 Transfer/Average 上均最优。
6. **蒸馏权重**：cd_weight=2.0 最优，fd_weight 在 {0,1,2} 内几乎不敏感（<0.1pp），aux_weight 不敏感。
7. **CD 散度形式**：kl_forward 最优。
8. **Iterations × Scheduler**：iter=800 + cosine_with_warmup 最优；constant scheduler 明显劣。
9. **层消融**：q/k/v/out/ffn 全微调最优。
10. **Rank**：rank 越大 Last/Average 越高；rank=4 为效率平衡点。
11. **QKV 投影**：共享 P 优于独立 P。

最终组合成 4 个配置并验证（详见 §2.7），**C1 Balanced**（current_nsp + AdamW 1e-4 + bs128 + cwu + iter800）以 ZS Average=69.609、Ens Average=70.653、Ens Last=81.405 成为 6-task 最佳配置。

### 1.7 为什么需要 10-task 重消融

有三个原因：

**① 任务数敏感**。6-task → 10-task 追加 4 个数据集（food101、mnist、stanford_cars、sun397），总类别数从 ~260 增至 ~460。更长的任务序列意味着：(a) 旧任务遗忘累积效应更强；(b) 更多 NSP 投影步骤，投影误差累积；(c) 分类头规模的扩大可能改变 lr 和 bs 的最优区间。6-task 上搜出的 bs=128、temp=4.0 等结论不能假设在 10-task 下自动成立。

**② DoRA vs LoRA 悬案**。6-task 消融中 LoRA 系统优于 DoRA，但历史上的 DoRA anchor（6/18 旧代码，详见 §3.2）在 10-task 上曾达到 Ens Last **82.79**，比当前 LoRA 10-task 结果（80.0）高出 **2.8pp**。这可能意味着：(a) 旧代码的文本端行为与当前不同（无 text_tuning_schedule 参数）；(b) DoRA 的解耦幅度/方向机制在更长任务序列下确实更好；(c) 其他代码差异（scheduler 默认值、weight_decay 等）。最坏的情况是我们在 6-task 上提前淘汰了 DoRA，错过了 10-task 的真正最优 backbone。

**③ seed 迁移**。6-task 全部使用 seed=42。本次使用 seed=43，需要重新建立基准线。单种子虽有效率优势，但若恰好对某配置不友好，选出的 winner 可能是噪声。我们加了平局规则来缓解。

### 1.8 数据集与协议

**10 个数据集**（X-TAIL benchmark，按训练顺序）：

| # | 数据集 | 类别数 | 测试集大小 | 特点 |
|---|---|---|---|---|
| 1 | aircraft | 100 | 3,333 | 细粒度飞机型号 |
| 2 | caltech101 | 101 | 2,465 | 通用物体 |
| 3 | dtd | 47 | 1,692 | 纹理描述 |
| 4 | eurosat | 10 | 5,400 | 卫星遥感 |
| 5 | flowers | 102 | 2,465 | 花卉细粒度 |
| 6 | food101 | 101 | 30,300 | 食物分类（大测试集） |
| 7 | mnist | 10 | 10,000 | 手写数字（灰度→RGB） |
| 8 | oxford_pets | 37 | 3,669 | 宠物品种 |
| 9 | stanford_cars | 196 | 8,041 | 汽车型号（最大类别数） |
| 10 | sun397 | 397 | 21,750 | 场景识别（最大测试集） |

**训练协议**：
- 每个任务仅 16 个标注样本（16-shot），无数据回放（replay-free）
- 每个任务训练 800 iterations，batch_size=64（待消融验证），scheduler=cosine_with_warmup
- 每完成一个任务的训练，立即 inline 评估全部已见任务
- 学习率 1e-4（AdamW），weight_decay=3e-5
- NSP 投影在每任务训练后执行（projection_param_mode=full，null_init_mode=none）

**评估协议**：
- 每个检查点评估 Zero-shot、LR-RGDA、LADA、Ensemble（α sweep）四种分类器
- Ensemble α 在 [0, 1] 上 21 等分搜索，取每个任务最优 α 的精度
- 输出 `*_zs_results.json`（Zero-shot/LR-RGDA/LADA/Ensemble 完整 metrics）和 `*_ens_results.json`（仅 Ensemble metrics + 逐任务 α）

### 1.9 LADA 评估指标

除 ZS 和 Ensemble 外，我们还计算 **LADA 分类器** 的精度作为补充参考。LADA（Locally Adaptive Discriminant Analysis）基于 k-means 聚类中心做最近邻分类：

**LADA 分类器**：对每个已见类别的少量样本提取视觉特征，用 k-means（k=16）聚类构建多中心原型，推理时用最近邻投票。不依赖 Zero-shot 文本 embedding。

**LADA+ZS 融合**：与 Ensemble 类似，在 LADA 和 ZS 之间做 α 加权：
```
logits_lada_zs = α × logits_lada + (1-α) × logits_zs
```

LADA 和 LADA+ZS 的 Transfer/Average/Last 按 §1.5 同一定义计算，输出到 `*_zs_results.json` 中。

**指标优先级**：我们的结果表优先报告 ZS 和 Ensemble 六指标。LADA 指标作为辅助参考，仅在主要指标平局或异常时用于交叉验证。

---

## 服务器信息

| 项目 | 内容 |
|---|---|
| 地址 | `raoxuan@10.20.34.30` |
| SSH | `ssh raoxuan@10.20.34.30`（需先连接 EasyConnect VPN） |
| 项目路径 | `/home/raoxuan/projects/project_clip_continual_learning` |
| 数据集路径 | `/data1/open_datasets/X-TAIL` |
| Conda 环境 | `raoxuan`（source `~/miniconda3/etc/profile.d/conda.sh`） |
| HF 镜像 | `export HF_ENDPOINT=https://hf-mirror.com`（服务器无外网） |
| GPU | 6× RTX 4090 24GB（GPU 0-2,4 可用，GPU 3,5 被 vLLM 占用） |
| 日志目录 | `logs/10task_reablation/` |
| 结果目录 | `experiments/10task_reablation/` |

**非交互式执行**：
```bash
ssh raoxuan@10.20.34.30 \
  'source ~/miniconda3/etc/profile.d/conda.sh && \
   cd /home/raoxuan/projects/project_clip_continual_learning && \
   conda run -n raoxuan python main_incremental.py ...'
```

**文件传输**（本地无法连接 GitHub，服务器可以。代码通过 rsync 上传，git 操作在服务器端执行）：
```bash
# 本地上传代码
rsync -avz --exclude='.git/' ./ raoxuan@10.20.34.30:projects/project_clip_continual_learning/

# 下载结果
scp -r raoxuan@10.20.34.30:projects/project_clip_continual_learning/experiments/10task_reablation/ ./experiments/
```

---

## 2. 已知事实：6-task 消融全景

以下数据均为 **6-task、16-shot、seed=42、lora_nsp + rank=4 + q/k/v/out/ffn**，完整六指标格式。数据来源：`chat-history/2026-07-05-weekly-ablation-review.md`。

### 2.1 优化器消融（LoRA, lr=1e-4, bs=64）

| variant | optimizer | ZS Transfer | ZS Average | ZS Last | Ens Transfer | Ens Average | Ens Last |
|---|---|---|---|---|---|---|---|
| current_nsp | Adagrad | 60.067 | 67.618 | 76.266 | 60.103 | 69.076 | 79.644 |
| current_nsp | Adam | 60.871 | 68.586 | 77.538 | 60.914 | 69.852 | 80.298 |
| current_nsp | **AdamW** | 60.886 | 68.590 | 77.545 | 60.924 | 69.846 | 80.246 |
| current_nsp | RMSprop | 60.870 | 68.637 | 77.578 | 60.887 | 69.888 | 80.363 |
| current_nsp | SGD (1e-3) | 60.884 | 65.022 | 71.250 | 60.922 | 66.722 | 74.427 |

**结论**：AdamW/Adam/RMSprop 为第一梯队，差异 <0.1pp。SGD 显著落后。最终选定 AdamW。

### 2.2 AdamW 学习率消融（LoRA）

| lr | ZS Transfer | ZS Average | ZS Last | Ens Transfer | Ens Average | Ens Last |
|---|---|---|---|---|---|---|
| 5e-5 | 61.754 | 67.235 | 74.688 | 61.812 | 68.771 | 77.861 |
| **1e-4** | 60.886 | 68.590 | 77.545 | 60.924 | 69.846 | 80.246 |
| 3e-4 | 58.250 | 68.746 | 79.563 | 58.294 | 69.795 | 81.874 |
| 6e-4 | 57.975 | 68.268 | 78.536 | 58.030 | 69.644 | 81.468 |

**结论**：1e-4 是 Transfer/Average/Last 的最佳平衡点。

### 2.3 SGD 学习率消融（LoRA）

| lr | ZS Transfer | ZS Average | ZS Last | Ens Transfer | Ens Average | Ens Last |
|---|---|---|---|---|---|---|
| 1e-3 | 60.884 | 65.022 | 71.250 | 60.922 | 66.722 | 74.427 |
| 2e-3 | 60.744 | 67.174 | 74.492 | 60.820 | 68.883 | 78.013 |
| 3e-3 | 59.386 | 68.225 | 77.352 | 59.467 | 69.757 | 80.988 |
| 4e-3 | 58.159 | 68.536 | 79.018 | 58.257 | 69.925 | 81.995 |
| 5e-3 | 57.440 | 68.530 | 79.000 | 57.590 | 69.990 | 82.243 |
| 5e-2 | 61.354 | 26.387 | 0.174 | 61.354 | 38.187 | 70.950 |

**结论**：SGD lr 越大 Last 越高但 Transfer 越低；5e-2 摧毁 ZS。整体 SGD 不如 AdamW。

### 2.4 Batch Size 消融（LoRA, AdamW 1e-4）

| bs | ZS Transfer | ZS Average | ZS Last | Ens Transfer | Ens Average | Ens Last |
|---|---|---|---|---|---|---|
| 32 | 61.662 | 67.782 | 75.797 | 61.682 | 69.315 | 79.022 |
| 64 | 60.886 | 68.590 | 77.545 | 60.924 | 69.846 | 80.246 |
| 128 | 60.131 | 69.061 | 78.689 | 60.178 | 70.139 | 81.007 |

**结论**：bs 越大 Average/Last 越高。但需在 10-task 下验证。

### 2.5 CD 温度消融（kl_forward）

| temp | ZS Transfer | ZS Average | ZS Last | Ens Transfer | Ens Average | Ens Last |
|---|---|---|---|---|---|---|
| 1.0 | 60.304 | 68.827 | 77.934 | 60.344 | 70.001 | 80.561 |
| 2.0 | 60.267 | 68.983 | 78.344 | 60.303 | 70.086 | 80.718 |
| 4.0 | 60.309 | 69.046 | 78.404 | 60.367 | 70.137 | 80.704 |

**结论**：temp=4.0 在 ZS/Ens Transfer 上均最优。

### 2.6 蒸馏权重消融

| 配置 | ZS Transfer | ZS Average | ZS Last | Ens Transfer | Ens Average | Ens Last |
|---|---|---|---|---|---|---|
| cd=0 | 60.513 | 68.016 | 77.090 | 60.543 | 69.409 | 80.274 |
| cd=1 | 60.886 | 68.590 | 77.545 | 60.924 | 69.846 | 80.246 |
| cd=2 | 60.960 | 68.715 | 77.671 | 61.006 | 69.950 | 80.273 |
| fd=0 | 60.915 | 68.613 | 77.569 | 60.970 | 69.865 | 80.322 |
| fd=1 | 60.886 | 68.590 | 77.545 | 60.924 | 69.846 | 80.246 |
| fd=2 | 60.864 | 68.587 | 77.533 | 60.899 | 69.847 | 80.180 |
| aux=0 | 60.433 | 68.976 | 78.146 | 60.483 | 70.031 | 80.442 |
| aux=1 | 60.886 | 68.590 | 77.545 | 60.924 | 69.846 | 80.246 |
| aux=2 | 61.087 | 68.120 | 76.500 | 61.131 | 69.456 | 79.778 |

**结论**：cd/fd 在 {0,1,2} 内差异 <0.2pp；aux 不敏感。默认 cd=2.0, fd=1.0, aux=0.0。

### 2.7 4 Combo 验证（6-task, seed=42）

| Combo | 配置摘要 | ZS Transfer | ZS Average | ZS Last | Ens Transfer | Ens Average | Ens Last |
|---|---|---|---|---|---|---|---|
| C1 Balanced | current_nsp + AdamW 1e-4 + bs128 + cwu + iter800 | 59.962 | **69.609** | 79.181 | 60.041 | **70.653** | 81.405 |
| C2 Last Max | current_nsp + AdamW 3e-4 + bs128 + linear + iter1600 | 57.254 | 69.040 | 80.025 | 57.320 | 69.850 | 81.685 |
| C3 Transfer | hist_runtime + AdamW 1e-4 + bs128 + cosine + iter800 | 60.399 | 69.249 | 78.154 | 60.482 | 70.435 | 80.709 |
| C4 Init-Only | hist_init_only + AdamW 1e-4 + bs128 + cwu + iter800 | 59.927 | 67.798 | 75.219 | 59.981 | 69.059 | 78.016 |

**C1 为 6-task 最佳配置**。

---

## 3. 核心矛盾：6-task SOTA 迁移到 10-task 退化

### 3.1 迁移结果

把上述 C1（当前最优）推到 10-task（seed=42，追加 food101/mnist/stanford_cars/sun397）：

| 场景 | ZS Transfer | ZS Average | ZS Last | Ens Transfer | Ens Average | Ens Last |
|---|---|---|---|---|---|---|
| 6-task C1 (seed=42) | 59.962 | 69.609 | 79.181 | 60.041 | 70.653 | 81.405 |
| **10-task C1 (seed=42)** | 58.4 | **68.4** | 76.5 | 58.5 | **69.6** | 80.0 |
| 退化 (Δ) | −1.6 | −**1.2** | −2.7 | −1.5 | −**1.1** | −1.4 |

退化幅度：ZS Average 降 **1.2pp**，Ens Average 降 **1.1pp**，Ens Last 降 **1.4pp**。Transfer 也降了 1.6pp——新任务加入后前序任务的灾难性遗忘加重。

### 3.2 核心悬案：DoRA anchor 的 82.79

**6/18 旧代码**（`nsp_fd_cd_vision_anchor_seed42`）在 10-task 上曾达到：

| 配置 | ZS Transfer | ZS Average | ZS Last | Ens Transfer | Ens Average | Ens Last |
|---|---|---|---|---|---|---|
| DoRA+NSP anchor（6/18, 10-task, seed=42） | 59.85 | 68.54 | 77.72 | 60.42 | 70.38 | **82.79** |
| C1 LoRA+NSP（当前代码, 10-task, seed=42） | 58.4 | 68.4 | 76.5 | 58.5 | 69.6 | **80.0** |
| 差值 | −1.45 | −0.14 | −1.22 | −1.9 | −0.8 | −**2.8** |

> **重要澄清**：旧代码 `use_dora` 默认为 `true`，因此 82.79 是 **DoRA + NSP**（不是 LoRA + NSP）。同时旧代码的 `ensemble_normalize` 为 `maxshift`——当前代码已将默认值统一为 `maxshift`（见 `main_incremental.py:596`），因此以下所有讨论均在 **maxshift 口径**下进行。

**已知差异追溯**（确认版）：

| 维度 | 6/18 旧代码 | 当前代码 (7/6) |
|---|---|---|
| use_dora | **true (DoRA)** | false (LoRA) |
| text_tuning_schedule | **不存在该参数，实际行为 ≈ low_lr_after** | freeze_after（默认） |
| scheduler | cosine | cosine_with_warmup |
| cd_weight | 1.0 | 2.0 |
| lora_target_modules | q/k/v/out/ffn | q/k/v/out/ffn |
| seed | 42 | 42 (4-combo) / 43 (本次) |
| ensemble_normalize | maxshift | maxshift（已统一） |

最可疑的差异是 **use_dora** 和 **text_tuning_schedule**。旧代码的实际行为最接近 `low_lr_after`（文本端在 Task 1 后以降低的学习率持续微调），而当前代码默认为 `freeze_after`（Task 1 后完全冻结文本端）。此外，旧代码的 scheduler=cosine（无 warmup）和 cd_weight=1.0 也可能造成微小差异。

可能的解释：

- **(a) DoRA 在更长任务序列下优于 LoRA**：6-task 上差异不显著，但 4 个额外任务的累积效应放大了 DoRA 的优势。DoRA 的解耦学习可能提供了更好的跨任务泛化。
- **(b) 文本端持续微调是关键**：`low_lr_after` 或 `always` 让文本 embedding 随任务适配，在 10 个不同领域间建立更好的对齐。`freeze_after` 可能过早冻结了文本端。
- **(c) 其他差异**：scheduler（cosine vs cosine_with_warmup）、cd_weight（1.0 vs 2.0）等微小差异的累积效应。
- **(d) seed 方差**：旧代码仅 seed=42，当前 C1 也是 seed=42。若种子间方差较大，单点比较不可靠。

**Pre-Wave A 的设计正是为了区分 (a)、(b) 和 (d)**：在 seed=42 同一种子下，全交叉测试 DoRA/LoRA × freeze_after/low_lr_after/always（6 个实验），直接测量 backone 主效应、text_schedule 主效应、及二者交互。

---

## 4. 实验目标与指标

### 4.1 目标

1. **Pre-Wave A**：在当前代码（seed=43）下复现 DoRA anchor，确认与历史 82.79 的口径差异。
2. **Wave A**：在 10-task 上重建 DoRA/LoRA × lr 的精度曲面。
3. **Wave B–F**：在最优 backbone 上消融 optimizer、bs、temperature、aux_weight、text_tuning_schedule。
4. 输出完整的六指标结果表，明确 10-task 最优配置。

### 4.2 指标定义

所有实验同时报告 **6 个指标**（按 LADA 论文 Appendix A 口径，×100 为百分比）：

| 分类器 | Transfer | Average | Last |
|---|---|---|---|
| Zero-shot (ZS) | ZS Transfer | **ZS Average** ★ | ZS Last |
| Ensemble (Ens) | Ens Transfer | Ens Average | Ens Last |

**优先级**：
- **第一指标：ZS Average** ← 这是零样本分类器的核心指标，直接反映模型增量学习后的零样本泛化能力
- 第二指标：Ens Average（集成分类器，带 alpha sweep 最优 alpha）
- 第三指标：Ens Last
- 参考指标：ZS Last、ZS Transfer、Ens Transfer（Task 1 的 Transfer 为 N/A，不计入均值）

### 4.3 任务序列与协议

```
aircraft → caltech101 → dtd → eurosat → flowers → food101 → mnist → oxford_pets → stanford_cars → sun397
```

- 数据集：X-TAIL，16-shot
- 种子：seed=43（区别于 6-task 的 seed=42）
- 评估：每个任务训练完成后，inline 评估全部已见任务的 ZS/LR-RGDA/Ensemble/LADA 精度
- Ensemble alpha sweep：21 点 (0.00–1.00)，记录最优 alpha 对应精度

---

## 5. 公共配置

所有实验统一的基础设置。各 Wave 的实验仅在标明的参数上偏离公共值。

```text
# 任务与数据
dataset_sequence     = aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397
root                 = /data1/open_datasets/X-TAIL
num_shots            = 16
seed                 = 43

# 训练
iterations           = 800
batch_size           = 64                          # Wave C 中消融 {32, 64, 128}
scheduler            = cosine_with_warmup           # Pre-Wave A 用 cosine
eval_batch_size      = 128
num_workers          = 2

# LoRA-NSP
lora_type            = lora_nsp
use_dora             = false                       # Wave A 中消融 DoRA/LoRA
lora_rank            = 4
lora_target_modules  = q_proj,k_proj,v_proj,out_proj,fc1,fc2
projection_param_mode = full
null_init_mode       = none

# 优化器
optimizer            = adamw                        # Wave B 中对比 SGD
lr                   = 1e-4                         # Wave A 中消融 {5e-5, 1e-4, 3e-4}
weight_decay         = 3e-5

# 蒸馏
cd_weight            = 2.0
fd_weight            = 1.0
aux_weight           = 0.0                          # Wave E 中消融 {0.0, 1.0}
cd_divergence        = kl_forward
cd_temperature       = 2.0                          # Wave D 中消融 {1.0, 2.0, 4.0}

# 文本端
text_tuning_schedule       = low_lr_after           # 对齐历史 82.79 配置；Wave F 中验证
text_schedule_switch_task  = 1
text_lr_scale_after_task   = 0.2

# 输出
output_dir           = /home/raoxuan/projects/project_clip_continual_learning/experiments/10task_reablation
```

---

## 6. Pre-Wave A：DoRA vs LoRA × Text Schedule 全交叉

### 6.1 目标

在 seed=42 下，全交叉测试 backbone（LoRA / DoRA）× text_tuning_schedule（freeze_after / low_lr_after / always），一次性回答：

1. **DoRA vs LoRA 主效应**：在 3 种 text schedule 取均值或逐对比较，DoRA 是否有稳定增益？
2. **最优 text schedule**：`low_lr_after`（历史配置）是否确实优于 `freeze_after`（当前默认）和 `always`？
3. **交互效应**：DoRA 的优势是否依赖于某种特定的 text schedule？例如 DoRA 可能需要文本端持续微调（`low_lr_after` 或 `always`）才能体现 magnitude/direction 解耦的优势。
4. **82.79 复现**：DoRA + low_lr_after + seed=42 应最接近历史配置，验证是否能在当前代码 + maxshift 下复现。

### 6.2 实验表

| # | 实验名 | use_dora | text_tuning_schedule |
|---|---|---|---|
| 1 | `lora_freeze` | false | freeze_after |
| 2 | `lora_lowlr` | false | low_lr_after |
| 3 | `lora_always` | false | always |
| 4 | `dora_freeze` | true | freeze_after |
| 5 | `dora_lowlr` | true | low_lr_after |
| 6 | `dora_always` | true | always |

```text
# 公共配置（对齐历史 82.79）
lora_type              = lora_nsp
lora_rank              = 4
lora_target_modules    = q_proj,k_proj,v_proj,out_proj,fc1,fc2
projection_param_mode  = full
null_init_mode         = none
optimizer              = adamw
lr                     = 1e-4
weight_decay           = 3e-5
batch_size             = 64
iterations             = 800
scheduler              = cosine                  # 匹配 6/18（无 warmup）
cd_weight              = 1.0                     # 匹配 6/18
fd_weight              = 1.0
aux_weight             = 0.0
cd_divergence          = kl_forward
cd_temperature         = 2.0
seed                   = 42
ensemble_normalize     = maxshift
```

### 6.3 分析框架

拿到 6 个实验的 ZS / Ens 六指标后：

- **DoRA 主效应**：对每个 text schedule 配对比较（dora_freeze vs lora_freeze, dora_lowlr vs lora_lowlr, dora_always vs lora_always），看 DoRA - LoRA 差值的方向和大小。
- **Text schedule 主效应**：在每个 backbone 内比较 freeze_after / low_lr_after / always，以 ZS Average 为第一排序键。
- **交互**：如果 DoRA - LoRA 的差值在 `always` 下明显大于 `freeze_after` 下，说明 DoRA 确实依赖文本端持续微调来释放潜力。
- **82.79 复现**：`dora_lowlr` 的 Ens Last 应接近 82.79。若显著偏离（< 81.0），排查 scheduler（cosine vs cosine_with_warmup）或代码版本差异。

### 6.4 门控与决策规则

**DoRA vs LoRA 选择**：
- 若 DoRA 在 ≥2/3 的 text schedule 配对上 ZS Average 领先 > 0.5pp → 后续 Waves 使用 DoRA
- 若 LoRA 在 ≥2/3 的配对上领先或持平 → 后续 Waves 使用 LoRA
- 若存在明显交互（如 DoRA 仅在 `always` 下领先）→ 后续 Waves 可能需要保留两个 backbone 分支

**Text schedule 选择**：
- 以两个 backbone 的平均 ZS Average 排序，选出最优 text schedule 作为后续 Waves 的公共配置
- 若 `low_lr_after` 与最优差距 < 0.3pp，优先保留 `low_lr_after`（历史连续性）

**82.79 复现门控**：
- 若 `dora_lowlr` 的 Ens Last ≥ 81.5：口径基本对齐，继续
- 若 < 81.0：暂停，检查 scheduler 实际生效值、cd_weight、代码版本差异

**种子方差评估**（非阻塞）：
- 在 winner 配置（确定 backbone + text schedule 后）上补跑 seed=43，与 Pre-Wave A 的 seed=42 结果对比，估计种子标准差 σ_seed
- 若 σ_seed > 0.5pp（ZS Average），后续 Waves 的平局阈值设为 1.0pp；否则保持 0.3pp

### 6.5 执行

6 个实验在 4 张 GPU 上分两轮：
- 第一轮：exp 1-4 并行（GPU 0,1,2,4）
- 第二轮：exp 5-6（GPU 0,1），同时 GPU 2,4 可用于种子方差对照

预计总耗时：~2 小时（每实验 ~55 分钟，两轮串行）。

---

## 7. Wave A：DoRA vs LoRA × 学习率

### 7.1 目标

在 Pre-Wave A 选定的最优 text_tuning_schedule 下，重建 DoRA 和 LoRA 的精度曲面，找出最优 (use_dora, lr) 组合。这是整个消融最关键的一步——后续所有 Wave 都在此基础上进行。

> **注意**：Pre-Wave A 已全交叉测试了 backbone × text_schedule。Wave A 的 text_tuning_schedule 固定为 Pre-Wave A 选出的最优值（预期为 `low_lr_after`）。

### 7.2 实验表

| # | 实验名 | use_dora | lr | bs | temp | optimizer |
|---|---|---|---|---|---|---|
| 1 | dora_lr5e5 | true | 5e-5 | 64 | 2.0 | adamw |
| 2 | dora_lr1e4 | true | 1e-4 | 64 | 2.0 | adamw |
| 3 | dora_lr3e4 | true | 3e-4 | 64 | 2.0 | adamw |
| 4 | lora_lr5e5 | false | 5e-5 | 64 | 2.0 | adamw |
| 5 | lora_lr1e4 | false | 1e-4 | 64 | 2.0 | adamw |
| 6 | lora_lr3e4 | false | 3e-4 | 64 | 2.0 | adamw |

**复用**：若 Pre-Wave A 的 `dora_lowlr` 与 `dora_lr1e4` 配置一致（use_dora=true, lr=1e-4, bs=64, temp=2.0），可直接复用（scheduler 从 cosine 改为 cosine_with_warmup，若精度差异可接受则跳过重跑）。

**新增训练数**：5（若 dora_lr1e4 复用）或 6（全部重跑）。

### 7.3 选择规则

以 **ZS Average** 为第一排序键选择最优 (use_dora, lr) 组合。该组合成为 "winner"，后续 Wave B–F 均固定此配置。

**平局规则**：若 top-2 的 ZS Average 差距 < 0.3pp → 看 Ens Average；Ens Average 也 < 0.3pp → 优先选 LoRA（参数更少、训练更快）。

---

## 8. Wave B：优化器对照（AdamW vs RMSprop vs SGD）

### 8.1 目标

在 Wave A 选出的最优 backbone 上，对比三种优化器。6-task 消融中 AdamW/Adam/RMSprop 为第一梯队（差异 <0.1pp），SGD 即使最优 lr 下 ZS Transfer 也大幅落后。10-task 下重新验证。

学习率尺度：AdamW 和 RMSprop 使用相同量级（1e-4），SGD 高一个数量级（5e-3，6-task 最优值）。

### 8.2 实验表

| # | 实验名 | optimizer | lr |
|---|---|---|---|
| 7 | winner_adamw1e4 | adamw | 1e-4 |
| 8 | winner_rmsprop1e4 | rmsprop | 1e-4 |
| 9 | winner_sgd5e3 | sgd | 5e-3 |

`winner_adamw1e4` 与 Wave A winner 为同一配置，可直接复用。

### 8.3 回退规则

- 若 SGD 的 ZS Average 落后最优优化器 > 1pp：后续不再扩展 SGD。
- 若 RMSprop 与 AdamW 差距 < 0.3pp：两者均可，优先 AdamW（6-task 更稳定）。
- 若 SGD 意外接近或超越 AdamW（<0.5pp）：在 Wave C/D/E 中同时测试两者。

---

## 9. Wave C：Batch Size

### 9.1 目标

验证 bs 在 10-task 下是否仍像 6-task 一样"越大越好"（bs=128 > bs=64 > bs=32）。更长的任务序列意味着更多 total iterations，bs 对泛化的影响可能改变。

### 9.2 实验表

| # | 实验名 | bs |
|---|---|---|
| 9 | winner_bs32 | 32 |
| 10 | winner_bs64 | 64 |
| 11 | winner_bs128 | 128 |

bs=64 复用 Wave A winner。

---

## 10. Wave D：CD 温度

### 10.1 目标

确认 6-task 选出的 temp=4.0 在 10-task 下是否仍最优。温度控制知识蒸馏的软标签平滑程度——更长任务序列中，旧任务的知识可能更需要高温（更软的标签）来保留。

### 10.2 实验表

| # | 实验名 | cd_temperature |
|---|---|---|
| 12 | winner_temp1p0 | 1.0 |
| 13 | winner_temp2p0 | 2.0 |
| 14 | winner_temp4p0 | 4.0 |

temp=2.0 复用 Wave A winner。

---

## 11. Wave E：aux_weight

### 11.1 目标

在最优配置上验证 aux_weight=1.0 是否有额外增益。6-task 上 aux_weight 不敏感（<0.2pp），但 10-task 下分类头更多，辅助损失的效果可能不同。

### 11.2 实验表

| # | 实验名 | aux_weight |
|---|---|---|
| 15 | winner_aux0 | 0.0 |
| 16 | winner_aux1 | 1.0 |

aux=0 复用 Wave A winner。

---

## 12. Wave F：文本端微调策略（验证）

### 12.1 目标

Pre-Wave A 已在历史配置（cosine, cd=1.0, bs=64）下全交叉测试了 text_schedule。Wave F 在 **最优 backbone + 完整调优超参数**（来自 Waves B–E）下重新验证 text_schedule 的选择，确保 Pre-Wave A 的结论在调优后仍然成立。

### 12.2 四种策略

| 策略 | text_tuning_schedule | 含义 |
|---|---|---|
| freeze_after | freeze_after | Task 1 正常训练文本端，之后冻结 |
| low_lr_after | low_lr_after | Task 1 后降为 0.2× lr 继续微调（预期最优） |
| always | always | 全程微调文本端（与视觉端同 lr） |
| never | never | 全程冻结文本端（仅微调视觉端，下界） |

### 12.3 实验表

| # | 实验名 | text_tuning_schedule |
|---|---|---|
| 17 | winner_text_freeze | freeze_after |
| 18 | winner_text_lowlr | low_lr_after |
| 19 | winner_text_always | always |
| 20 | winner_text_never | never |

`winner_text_lowlr` 复用 Wave A winner（当前公共配置已设为 `low_lr_after`）。

**若 Pre-Wave A 所选最优 ≠ low_lr_after**：将 Pre-Wave A 选出的最优值设为复用项，`low_lr_after` 降为对照。

---

## 13. 实验汇总表

### 已完成

| # | Phase | 实验名 | 状态 | 关键结果 |
|---|---|---|---|---|
| 1–6 | Pre-A | lora_freeze/low_lr/always, dora_freeze/low_lr/always | ✅ | LoRA > DoRA, always 最优 |
| 7 | P1 | lora_always_s43 (种子方差) | ✅ | σ_seed≈0.5pp |
| 8–10 | P1 | wa_lora_lr{5e5,1e4,3e4} | ✅ | lr=1e-4 winner, Ens L=83.65 |

### 待执行

| # | Phase | 实验名 | 关键变动 vs 公共配置 | 实验数 |
|---|---|---|---|---|
| 11–13 | P2 (Wave C) | winner_bs{32,64,128} | bs={32,64,128} | 3 |
| 14–16 | P3 (Wave D) | winner_temp{1.0,2.0,4.0} | cd_temperature={1.0,2.0,4.0} | 3 |
| 17–20 | P4 (Wave F) | winner_text_{freeze,lowlr,always,never} | text_tuning_schedule 全覆盖 | 4 |
| 21–22 | P6 | {hard,soft}_nsp | use_soft_projection={false,true} | 2 |
| 23–27 | P7 (hard) | nsp_eps_{0.02,0.05,0.08,0.12,0.20} | nsp_eps sweep | 5 |
| | P7 (soft) | nsp_weight_{0.005,0.01,0.02,0.05} | nsp_weight sweep (若 Phase 6 soft 胜) | 4 |
| 28–31 | P8 | eta_min_{0,1e-7,1e-6,1e-5} | eta_min sweep | 4 |

**公共配置** (Phase 2–8): LoRA+NSP, text=always, cosine_with_warmup, cd=2.0, fd=1.0, aux=0.0, num_centers=4, rgda_train_iter=200, bs=64, iter=800, lr=1e-4, seed=43, maxshift

**已砍**: Wave B (optimizer), Wave E (aux_weight) — 6-task 差异 <0.1pp。

**新增实验数**: 3+3+4+2+5+4 = **21** (hard NSP) 或 **20** (soft NSP)

---

## 14. 执行顺序与依赖

```
Pre-Wave A (backbone × text_sched 全交叉, 6 个实验, 2 轮)
    │  └─ 确定 backbone (DoRA/LoRA) + text_schedule (freeze/low_lr/always)
    │     同时评估 82.79 复现、种子方差（winner 补测 seed=43）
    ▼
Wave A (DoRA/LoRA × lr, 6 个实验, 4 GPU 并行)
    │  text_schedule = Pre-A 选出的最优值
    │
    ├── 选出 winner = 最优 (use_dora, lr)
    │   平局规则: ZS Average <0.3pp → Ens Average; 仍<0.3pp → 选 LoRA
    │   （若 Pre-A 已得出种子标准差 σ > 0.5pp，平局阈值上调至 1.0pp）
    │
    ├──→ Wave B (optimizer, 3 个实验) ── 与 C/D/E 并行
    │         ├── winner_adamw1e4 (复用)
    │         ├── winner_rmsprop1e4
    │         └── winner_sgd5e3
    │
    ├──→ Wave C (bs, 2 个新增 + bs64 复用) ── 与 B/D/E 并行
    │         ├── winner_bs32
    │         └── winner_bs128
    │
    ├──→ Wave D (temp, 2 个新增 + temp2 复用) ── 与 B/C/E 并行
    │         ├── winner_temp1p0
    │         └── winner_temp4p0
    │
    └──→ Wave E (aux, 1 个新增 + aux0 复用) ── 与 B/C/D 并行
              └── winner_aux1
                  │
                  ▼
              Wave F (text 验证, 3 个新增 + low_lr 复用)
                    ├── winner_text_freeze
                    ├── winner_text_always
                    └── winner_text_never
```

**阶段 1**：Pre-Wave A（6 个实验，4 GPU 分两轮：4 + 2，~2h）
**阶段 1.5**：种子方差对照（winner 配置 × seed=43，1 个实验，与阶段 2 可并行，~1h）
**阶段 2**：Wave A（5 个新增 + 1 复用，6 个分配到 4 GPU，~1.5h）
**阶段 3**：Wave B + C + D + E 并行（2+2+2+1 = 7 个新增，4 GPU，~2h）**← 每 GPU 1 模型，不重叠**
**阶段 4**：Wave F（3 个新增，4 GPU 并行，~1h）

---

## 15. 执行策略

### 15.1 已知问题与修复

| 问题 | 根因 | 修复 |
|---|---|---|
| 多实验 OOM | `run_queue` 用 `&` 在单 GPU 上并发启动多个实验 | GPU 内严格串行：`launch; wait $!; launch; wait $!` |
| HF 429 限流 | 多个实验同时 HEAD 请求 clip-vit 模型缓存 | 每 GPU 第一个实验间隔 ≥15s；Pre-A 第一轮 4 个实验预热缓存 |
| 僵尸进程残留 | OOM 后子进程未被清理 | 每阶段启动前 `pkill -f main_incremental.py` 清理 |

### 15.2 可用资源

- 服务器：10.20.34.30
- GPU 0, 1, 2, 4：空闲
- GPU 3, 5：Alan vLLM 占用，不可用
- 可用 GPU 数：**4 张**
- 单实验预计耗时：**45–55 分钟**（10-task × 800 iter）

### 15.3 关键约束

**每张 GPU 同时最多运行 1 个 main_incremental.py**。在阶段 3（Wave B+C+D+E 并行）中，7 个实验分布在 4 张 GPU 上意味着其中 3 张 GPU 需要跑 2 轮。脚本必须实现严格的 GPU 队列：

```bash
# 单 GPU 队列（伪代码）
launch_on_gpu exp1 GPU0 && wait $!   # 串行：等 exp1 跑完
launch_on_gpu exp2 GPU0 && wait $!   # 再跑 exp2
```

不同 GPU 之间可真正并行。

### 15.4 启动流程

```bash
# 0. 【代码版本锚点】在服务器上提交当前代码状态（见 §15.7）
ssh raoxuan@10.20.34.30 \
  'source ~/miniconda3/etc/profile.d/conda.sh && \
   cd /home/raoxuan/projects/project_clip_continual_learning && \
   git add -A && git commit -m "pre-wave-a: 10task reablation start"'

# 1. 清理残留进程
pkill -f main_incremental.py 2>/dev/null; sleep 2

# 2. 阶段 1：Pre-Wave A，6 个实验，4 GPU 分两轮
# 第一轮：lora_freeze, lora_lowlr, lora_always, dora_freeze (GPU 0,1,2,4)
launch_one "lora_freeze"  --use_dora false --text_tuning_schedule freeze_after  ... GPU 0 &
launch_one "lora_lowlr"   --use_dora false --text_tuning_schedule low_lr_after ... GPU 1 &
launch_one "lora_always"  --use_dora false --text_tuning_schedule always      ... GPU 2 &
launch_one "dora_freeze"  --use_dora true  --text_tuning_schedule freeze_after ... GPU 4 &
wait  # 等待第一轮完成

# 第二轮：dora_lowlr, dora_always (GPU 0,1)，GPU 2,4 空闲可用于种子方差
launch_one "dora_lowlr"   --use_dora true --text_tuning_schedule low_lr_after ... GPU 0 &
launch_one "dora_always"  --use_dora true --text_tuning_schedule always      ... GPU 1 &
wait

# 3. 分析 Pre-Wave A 结果，确定 backbone + text_schedule
# 4. 在 winner 上补测 seed=43（可与阶段 2 并行）
# 5. 【代码版本锚点】提交 Pre-A 分析结论

# 6. 阶段 2：Wave A，6 个实验，4 GPU
# ...（同上模式）

# 7. 【代码版本锚点】提交 Wave A winner 结论

# 8. 阶段 3：Wave B+C+D+E 并行
# 9. 阶段 4：Wave F
```

### 15.5 实验完成验证

每个实验完成后必须检查：
1. 进程正常退出（exit code 0）
2. `experiments/10task_reablation/{exp_name}_zs_results.json` 存在且非空
3. `experiments/10task_reablation/{exp_name}_ens_results.json` 存在且非空

缺任一条件标记为 FAILED，记录到日志。

### 15.6 日志清理

重启前，将上一轮的崩溃日志归档：
```bash
mkdir -p logs/10task_reablation/archive_round1
mv logs/10task_reablation/*.log logs/10task_reablation/archive_round1/ 2>/dev/null
```
避免新旧日志混淆。

### 15.7 代码版本管理

本地无法连接 GitHub，但**服务器可以**。工作流：本地改代码 → rsync 到服务器 → 服务器端 git commit + git push。

**关键节点 git commit**（在服务器上执行，每个阶段完成后 push）：

| 节点 | commit message | 时机 |
|------|---------------|------|
| 消融启动 | `pre-wave-a: 10task reablation start` | Pre-Wave A 启动前 |
| Pre-A 完成 | `pre-wave-a: results analyzed, winner={dora/lora}+{text_sched}` | 进入 Wave A 前 |
| Wave A 完成 | `wave-a: winner={config}, lr={value}` | 进入 Wave B–E 前 |
| Waves B–E 完成 | `waves-b-c-d-e: all results` | 进入 Wave F 前 |
| 全部完成 | `10task-reablation: final results` | 汇总后 |

**结果追溯**：
- 每个实验结果 JSON 中写入 `git_commit` 字段（在 `main_incremental.py` 中通过 `subprocess.check_output(['git', 'rev-parse', 'HEAD'])` 获取）
- 结果目录 `experiments/10task_reablation/` 下保存 `git_log.txt`（`git log --oneline -20` 的输出）

**同步命令**（本地→服务器）：
```bash
rsync -avz --exclude='.git/' --exclude='experiments/' --exclude='logs/' \
  --exclude='__pycache__/' --exclude='*.pyc' \
  ./ raoxuan@10.20.34.30:projects/project_clip_continual_learning/
```

**服务器端提交+推送**：
```bash
ssh raoxuan@10.20.34.30 \
  'cd /home/raoxuan/projects/project_clip_continual_learning && \
   git add -A && git commit -m "pre-wave-a: 10task reablation start" && git push'
```

**注意**：`.git/` 目录不同步（rsync exclude），服务器上的 git 仓库是独立的。每次 rsync 后需要在服务器上手动提交。

---

## 16. 成功标准

1. **Pre-Wave A**：6 个实验全部完成。明确：
   - DoRA vs LoRA 主效应及交互（是否依赖 text_schedule）
   - 最优 text_schedule（以 ZS Average 为第一排序键）
   - 82.79（DoRA + low_lr_after + seed=42）在当前代码 + maxshift 下能否复现
   - 种子标准差（winner 补测 seed=43 后）
2. **Wave A**：明确 10-task 下最优 (use_dora, lr)，选出 winner。平局阈值根据 Pre-A 的种子标准差校准。
3. **Wave B–F**：依次明确 optimizer、bs、temp、aux 的最优值，并验证 text_schedule 在调优后的稳定性。
4. **代码版本**：每个关键节点在服务器上 git commit，结果 JSON 含 commit hash。
5. **汇总**：所有实验的完整六指标结果整理到 `chat-history/2026-07-06-10task-reablation-results.md`，按 ZS Average 降序排列。

**门控**：
- 若 `dora_lowlr`（最接近历史 82.79 的配置）的 Ens Last < 81.0：暂停，排查 scheduler 实际生效值、cd_weight、代码版本差异
- 若 6 个实验中 ZS Average 全 < 67.0：代码或协议可能存在系统性问题，暂停排查
- 若 winner 配置的种子标准差 σ > 1.0pp（ZS Average）：该指标可能不适合作为单一排序键，考虑提升 Ens Average 权重

---

## 17. 执行 Checklist

| # | Phase | 内容 | 实验数 | 状态 | 结果 |
|---|---|---|---|---|---|
| 1 | Pre-Wave A | backbone × text_schedule 全交叉 | 6 | ✅ 完成 | LoRA > DoRA, always 最优 |
| 2 | 代码修复 | parser cd_temperature/cd_divergence/scheduler | — | ✅ 完成 | commit 710ee34 |
| 3 | Phase 1 | 种子方差 + Wave A lr sweep | 4 | ✅ 完成 | lr=1e-4 winner, σ_seed≈0.5pp, Ens L=83.65 |
| 4 | Phase 2 | Wave C (batch size) | 3 | ⏳ 待启动 | — |
| 5 | Phase 3 | Wave D (CD 温度) | 3 | ⬜ 未开始 | — |
| 6 | Phase 4 | Wave F (text schedule 验证) | 4 | ⬜ 未开始 | — |
| 7 | Phase 5 | 汇总 (Waves C/D/F) | — | ⬜ 未开始 | — |
| 8 | Phase 6 | Soft vs Hard NSP | 2 | ⬜ 未开始 | — |
| 9 | Phase 7 | NSP 超参数 (nsp_eps 或 nsp_weight) | 4–5 | ⬜ 未开始 | — |
| 10 | Phase 8 | eta_min 退火搜索 | 4 | ⬜ 未开始 | ✅ 代码已修复 |
| 11 | Phase 9 | 最终汇总 (全部) | — | ⬜ 未开始 | — |

**已砍**: Wave B (optimizer, 6-task 差异 <0.1pp), Wave E (aux_weight, 6-task 不敏感)

**Phase 1 结果** (7/6 深夜):

| 实验 | lr | ZS A | Ens A | Ens L |
|------|:---:|:---:|:---:|:---:|
| wa_lora_lr5e5 | 5e-5 | 69.42 | 71.59 | 83.58 |
| **wa_lora_lr1e4** | **1e-4** | **69.50** | 71.19 | **83.65** |
| wa_lora_lr3e4 | 3e-4 | 69.00 | 70.62 | 83.10 |

mc4ft200 下 Ens Last **83.65** 已超过历史 82.79。

---

## 19. Phase 6: Soft vs Hard NSP

### 19.1 背景

当前默认使用 **hard projection**：`use_soft_projection=False, nsp_weight=0.02`。Hard 模式下只保留前 m 个主成分（m 由 nsp_eps 自适应），将其余方向完全释放给 LoRA 更新。Soft 模式下所有权重按特征值加权：`w_i = 1/(1 + β·log(1 + λ_i^p))`，所有方向都有部分约束。

Soft NSP 的理论优势：不硬性截断子空间，对特征值估计噪声更鲁棒。但 6-task 消融中 soft 从未被系统测试。

### 19.2 目标

在 Waves C/D/F 确定的最佳配置上，对比 hard NSP vs soft NSP：

| # | 实验名 | use_soft_projection | nsp_weight | nsp_eps |
|---|---|---|---|---|
| 1 | hard_nsp_baseline | false | 0.02 | 0.05 |
| 2 | soft_nsp | true | 0.02 | 0.05 |

**公共配置**：LoRA+NSP, text=always, winner_lr, winner_bs, winner_temp, winner_aux, cosine_with_warmup, cd=2.0, mc4ft200, seed=43。

**决策**：ZS Average 排序。若 soft 领先 >0.3pp → 切到 soft，Phase 7 中扫 soft 的 nsp_weight；否则保持 hard，Phase 7 扫 hard 的 nsp_eps。

---

## 20. Phase 7: NSP 超参数消融

### 20.1 目标

在 Phase 6 选出的 NSP 模式下，消融关键超参数。

### 20.2 场景 A: Hard NSP 胜出 → 扫 nsp_eps

`nsp_eps` 控制保留子空间维度 m。eps 越小 → m 越大 → 约束越强（更多方向被保护）。

| # | nsp_eps | 预期 m (d=768) |
|---|:---:|:---:|
| 1 | 0.02 | ~200-400 |
| 2 | 0.05 | ~100-200 (当前默认) |
| 3 | 0.08 | ~50-150 |
| 4 | 0.12 | ~30-100 |
| 5 | 0.20 | ~10-80 |

5 个实验，并行于 4 GPU（两轮：4+1）。

### 20.3 场景 B: Soft NSP 胜出 → 扫 nsp_weight

`nsp_weight` (β) 控制软投影强度。越大 → 约束越强。

| # | nsp_weight |
|---|:---:|
| 1 | 0.005 |
| 2 | 0.01 |
| 3 | 0.02 (当前默认) |
| 4 | 0.05 |

4 个实验，1 轮 4 GPU。

### 20.4 决策

ZS Average 排序，选出最优 nsp_eps 或 nsp_weight。

---

## 21. Phase 8: 学习率退火 eta_min 搜索

### 21.1 背景

当前 `cosine_with_warmup` 的 LR 退火到底为 0（`eta_min = 0.0`）。这意味着训练后期 lr → 0，模型完全停止学习。小量级的 eta_min（如 1e-6）可保持微量学习，可能在长序列持续学习中防止极端遗忘。

### 21.2 所需代码改动

1. **`main_incremental.py`**: 新增 `--eta_min` CLI 参数（float, default=0.0）
2. **`src/trainers/lora_nsp_trainer.py`**: `cosine_with_warmup` LambdaLR 中注入 eta_min_ratio：
   ```python
   def cosine_with_warmup_lr(step):
       if step < warmup_steps:
           return step / max(1, warmup_steps)
       progress = (step - warmup_steps) / max(1, train_iterations - warmup_steps)
       return eta_min_ratio + (1.0 - eta_min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
   ```
   同时 CosineAnnealingLR 中传入 `eta_min` 替代硬编码的 `0.0`。

### 21.3 实验

在最佳 NSP 配置上扫 eta_min：

| # | eta_min |
|---|:---:|
| 1 | 0.0 (当前默认) |
| 2 | 1e-7 |
| 3 | 1e-6 |
| 4 | 1e-5 |

4 个实验，1 轮 4 GPU。eta_min=1e-4 不测——lr 自身才 1e-4，eta_min=1e-4 等于不退火。

### 21.4 决策

ZS Average 排序。若最优 eta_min ≠ 0.0 → 更新公共配置。若全部差异 <0.3pp → 保持 eta_min=0.0。

---

## 22. 最终汇总 (Phase 9)

完整实验表 + 最终推荐配置 + σ_seed + 与 82.79 对标 + NSP/EtaMin 发现。

输出: `chat-history/2026-07-07-reablation-final-results.md`

---

## 18. VPN 自动恢复流程 (Loop 模式)

Loop 执行时，若 SSH 连接失败，按以下流程处理：

```
1. ssh -o ConnectTimeout=10 raoxuan@10.20.34.30 'echo ok'

2. if 成功 → 进入实验检查流程

3. if 失败 (exit code 255):
   a. ping -n 1 10.20.34.30        (Windows: -n; Linux/Mac: -c 1)
   b. if ping 成功:
      # 网络通但 VPN 未连 → 自动打开 VPN 客户端
      - start "" "C:\Program Files\EasyConnect\EasyConnect.exe"   (Windows)
      - open -a "EasyConnect"                                       (macOS)
      - easyconnect &                                               (Linux)
      - 等待 60s
      - goto 1
   c. if ping 失败:
      # 网络不通 → 无法自动恢复
      - 汇报 "Network unreachable, waiting for recovery"
      - return (等待下次 loop)
```

**对应伪代码**:
```python
def connect_ssh():
    for attempt in range(2):
        result = run(f"ssh -o ConnectTimeout=10 {HOST} 'echo ok'")
        if result.success:
            return True
        if attempt == 0:
            ping_result = run(f"ping -n 1 {IP}")
            if ping_result.success:
                log("VPN disconnected, launching EasyConnect...")
                launch_vpn()
                sleep(60)
            else:
                log("Network unreachable")
                return False
    return False
```

---

*修订记录*：
- v1 (7/6 12:17)：初始方案，Ens Average 为第一指标
- v2 (7/6 15:00)：ZS Average 改为第一指标；补充完整六指标框架与详细动机；新增 Pre-Wave A 门控规则
- v3 (7/6 15:45)：Pre-A 新增 always 变体澄清文本端混淆；Wave B 新增 RMSprop、SGD lr 改为 5e-3；B 与 C/D/E 改为真正并行；新增平局规则与执行约束
- v3.1 (7/6 16:15)：大幅扩展 §1–§2，补充 CLIP+LoRA-NSP 架构细节、DoRA 原理、LADA 分类器、指标定义、6-task 完整消融路线图、数据集协议、DoRA anchor 差异追溯表
- v3.2 (7/6 17:00)：新增 §1.9 LADA 评估指标；新增服务器连接信息
- **v4 (7/6 晚间)**：
  - **关键修正**：§3.2 纠正了历史 82.79 的归因（是 DoRA+NSP，不是 LoRA+NSP）；澄清了 `text_tuning_schedule` 参数在旧代码中不存在，行为接近 `low_lr_after`；统一 `ensemble_normalize = maxshift` 口径
  - **Pre-Wave A 重构**：从 2 个实验（DoRA + freeze/always）扩展为 6 个实验（LoRA/DoRA × freeze/low_lr/always），全交叉测试 backbone × text_schedule
  - **§5 公共配置**：`text_tuning_schedule` 从 `freeze_after` 改为 `low_lr_after`（对齐历史 82.79 配置）
  - **Wave F 角色变更**：从"发现最优 text schedule"变为"在调优后验证 Pre-A 结论"
  - **新增 §15.7 代码版本管理**：关键节点 git commit + 结果 JSON 嵌入 commit hash
  - **§13 汇总表**：更新为 23 个实验（新增 21–22），明确每列取值的条件依赖
  - **种子方差**：Pre-A 确定 winner 后补测 seed=43，校准后续平局阈值
- v2 (7/6 15:00)：ZS Average 改为第一指标；补充完整六指标框架与详细动机；新增 Pre-Wave A 门控规则
- v3 (7/6 15:45)：Pre-A 新增 always 变体澄清文本端混淆；Wave B 新增 RMSprop、SGD lr 改为 5e-3；B 与 C/D/E 改为真正并行；新增平局规则与执行约束
- v3.1 (7/6 16:15)：大幅扩展 §1–§2，补充 CLIP+LoRA-NSP 架构细节、DoRA 原理、LADA 分类器、指标定义、6-task 完整消融路线图、数据集协议、DoRA anchor 差异追溯表
- v3.2 (7/6 17:00)：新增 §1.9 LADA 评估指标（LADA/LADA+ZS 分类器说明）；新增服务器连接信息（地址/SSH/路径/Conda/GPU 分配/文件传输）
