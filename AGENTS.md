# AGENTS.md — AI Agent 指南

本文件是为 AI 助手（如 Kimi Code CLI）阅读的说明文档。当你被要求在 `project_clip_continual_learning/` 中工作时，请先阅读此文件。

---

## 1. 核心职责：保存对话记录

**每次与本项目的用户进行有意义的沟通/协作后，必须将对话重点保存到 `chat-history/` 目录。**

论文写作相关协作另行保存到 `chat-history-for-paper-writing/`。两类目录的边界如下：

- `chat-history/`：代码、配置、实验、调试、结果和工程决策；
- `chat-history-for-paper-writing/`：论文故事、贡献表述、理论路线、章节组织和叙事版本变化。

若一次会话同时产生工程事实与论文叙事决策，应分别记录并相互链接。详细规范见两个目录各自的 `README.md`。

`meta-prompts/` 不属于对话历史。它保存从多次协作中提炼出的、跨会话长期有效的项目全局认知与 AI 元提示，例如项目定位、目录语义、核心方法关系和稳定的工作原则。较大规模任务开始前，应先阅读 `meta-prompts/README.md` 及相关全局认知文件；具体变更过程仍记录在对应 history 目录。

### 为什么要做？
- 跨对话的记忆持久化——AI 没有长期记忆，依赖文件系统
- 让后续进入此项目的 AI agent 能快速了解项目进展
- 记录设计决策、实验结论、待办事项，形成可追溯的研究日志

### 保存格式

文件命名：`chat-history/{YYYY-MM-DD}-{seq}-{topic}.md`

> **seq** 为两位序号，同一天的多条记录按时间递增。

```
chat-history/
├── 2026-05-23-01-project-overview.md        # 项目首次阅读记录
├── 2026-05-23-02-agents-md-creation.md      # AGENTS.md 创建记录
├── 2026-05-23-03-paper-writing-and-lada-baseline.md  # 论文写作与LADA基线
├── 2026-05-23-04-git-workflow-and-gitignore.md      # Git提交策略与.gitignore清理
├── 2026-05-23-05-server-workflow-and-repo-rename.md # 服务器工作流与仓库重命名
├── 2026-05-23-06-chat-history-naming-convention.md # 文件命名规范统一
└── {YYYY-MM-DD}-{seq}-{topic}.md             # 你创建的新记录
```

内容模板：

```markdown
# {简短标题}

**日期**: {YYYY-MM-DD}
**会话概况**: {1-2句话描述本次会话做了什么}

---

## 1. 关键讨论 / 决策

- {决策1}: {内容}
- {决策2}: {内容}

## 2. 重要发现

- {发现1}
- {发现2}

## 3. 待办事项 / 遗留问题

- [ ] {待办1}
- [ ] {待办2}

## 4. 相关文件

- {文件路径}: {说明}
```

### 什么时候必须写？
- 设计/架构决策被确定时
- 实验方案达成共识时
- 代码发生重大修改时
- 用户指明"记住这个"或"记录一下"时
- 每次对话结束时（如果产生了有意义的产出）

---

## 2. 论文写作（paper_writing/）

`paper_writing/` 是本论文的写作目录，包含 LaTeX 模板、论文草稿、参考论文库等。

### 目录结构

```
paper_writing/
├── paper-template/                       # LaTeX 模板 + 论文草稿
│   ├── neurips_2026.sty                  # NeurIPS 2026 样式文件
│   ├── template.tex                      # 空白模板
│   ├── paper_draft.tex                   # 当前论文草稿（主文件）
│   ├── notation_prompt.md                # 符号约定说明
│   └── paper_draft.pdf                   # 编译后的 PDF
├── paper_draft_README.md                 # 论文草稿说明（创新点、结构、参考）
├── reference_papers/                     # 参考论文库
│   ├── 25-ICML-LADA*.pdf                 # LADA — 本篇论文的核心基线
│   ├── 23-CVPR-ZSCL*.pdf                 # ZSCL: Preventing Zero-Shot Transfer Degradation
│   ├── 23-ICML-Mod-X*.pdf                # Mod-X: Off-Diagonal Information
│   ├── 24-NIPS-RAIL*.pdf                 # RAIL: Cross-domain Discriminability
│   ├── 25-ICCV-MG-CLIP*.pdf             # MG-CLIP: Mind the Gap
│   └── 25-ICLR-C-CLIP*.pdf              # C-CLIP: Multimodal Continual Learning
├── my_original_papers/                   # 作者自己的推导笔记
│   ├── LR-RGDA_原始推导.pdf              # LR-RGDA 数学推导
│   └── lora_nsp_原始推导.pdf             # LoRA-NSP 数学推导
├── kimi/                                 # Kimi paper-writing skills（v1 & v2）
│   ├── xuan-research-paper-writing/      # v1: 论文写作 AI skill
│   └── xuan-research-paper-writing-v2/   # v2: 改进版论文写作 AI skill
└── 讨论/                                 # 过往讨论记录
```

### 核心基线：LADA（25-ICML）

本篇论文的主要基线是 **LADA: Scalable Label-Specific CLIP Adapter for Continual Learning**（25-ICML）。

- **论文全称**：LADA: Scalable Label-Specific CLIP Adapter for Continual Learning
- **收录**：ICML 2025
- **核心思路**：在冻结的 CLIP 图像编码器后附加轻量级、标签特定的记忆单元，通过聚合任务无关知识生成判别性特征
- **与本论文的关系**：
  - 同样解决 CLIP 持续学习中的灾难性遗忘问题
  - 本论文的 LADA 评估指标（Transfer / Average / Last）沿用自 LADA 论文
  - 本论文的 X-TAIL 实验设置（10 个数据集、16-shot）与 LADA 一致
  - 本论文的核心对比对象：在 LADA 的评估协议下，用 LoRA-NSP + LR-RGDA 集成分类器达到更优的 ID/OOD 平衡

### 论文创新主张

本论文提出**训练端 + 推理端协同优化**框架：

| 端 | 创新 | 与 LADA 的区别 |
|----|------|---------------|
| **训练端** | LoRA-NSP（零空间投影 + 复合蒸馏） | LADA 仅用特征蒸馏，本文增加了跨模态蒸馏；LoRA-NSP 用零空间保护预训练知识，而 LADA 依赖标签特定记忆单元 |
| **推理端** | LR-RGDA 集成分类器（低秩正则化高斯判别分析 + 零样本集成） | LADA 同样有推理端融合（视觉/文本双分支加权），但 LADA 用标签记忆单元作为分类权重；本文用 LR-RGDA（基于类别统计分布的高斯建模）作为监督分类器，与零样本分类器集成 |

### 论文写作相关文件

| 文件 | 作用 |
|------|------|
| `paper-template/paper_draft.tex` | 主论文 LaTeX 源文件 |
| `paper_draft_README.md` | 论文结构和创新点说明 |
| `kimi/xuan-research-paper-writing-v2/SKILL.md` | Kimi 论文写作 AI skill（控制 AI 写作风格和流程） |
| `paper-template/notation_prompt.md` | 符号约定，写作时统一数学符号 |

### 写作注意事项

- 论文模板用的是 **NeurIPS 2026** 格式（`neurips_2026.sty`）
- 当前草稿中的实验数据是**占位符**，需根据实际实验结果替换
- 方法图（Figure 2/3）尚未添加，需要生成
- 参考论文 PDF 在 `reference_papers/` 目录，对应的 AI 笔记在 `.workbuddy/paper_refs/`

---

## 3. 开发与运行环境

### 架构：本地编辑 → 服务器运行

> **本项目仅在本地保存代码，实际运行在远程 GPU 服务器上。**
> AI agent 每次需要执行代码/运行实验时，必须按以下流程操作。

#### 服务器信息

| 项目 | 内容 |
|------|------|
| 服务器地址 | `raoxuan@10.20.34.30` |
| 远程项目路径 | `/home/raoxuan/projects/project_clip_continual_learning` |
| GitHub 仓库 | `https://github.com/raoxuan98-hash/clip_ood`（即将更名为 `project_clip_continual_learning`） |
| 数据路径 | `/home/raoxuan/projects/data/X-TAIL/` |

#### 标准工作流

```
本地（Mac）                        服务器（Linux + GPU）
───────                           ────────────────────
1. 修改代码
2. git add + git commit + git push
                                     3. ssh 登录
                                     4. cd /home/raoxuan/projects/project_clip_continual_learning
                                     5. git pull
                                     6. 运行实验 / 执行代码
                                     7. 实验结果保存在 experiments/ 目录
8. 如需查看结果，scp 或直接在服务器上查看
```

#### AI agent 执行代码/实验的操作步骤

当需要运行代码或实验时，必须执行：

```bash
# 第一步：确保本地修改已提交并推送
git add -A && git commit -m "[说明] ..." && git push

# 第二步：SSH 到服务器拉取最新代码
ssh raoxuan@10.20.34.30 "
  cd /home/raoxuan/projects/clip_ood &&
  git pull &&
  <执行命令>
"
```

#### 注意事项

- **不要在本地运行训练/实验代码**——本地 Mac 没有 GPU，训练脚本会失败
- **提交前确保代码可运行**——推送后再远程调试成本较高
- **服务器上的 `experiments/` 和 `optimization/` 目录被 `.gitignore` 排除**，不会被 `git pull` 覆盖
- 远程服务器已安装所需的 Python 环境和依赖（PyTorch, CUDA, transformers 等）
- 如需在服务器上安装新依赖，通过 SSH 执行 `pip install`

### GitHub 仓库重命名

当前远程仓库名为 `clip_ood`，将重命名为 `project_clip_continual_learning` 以与本地目录名保持一致。
重命名后需更新本地 remote URL：

```bash
git remote set-url origin https://github.com/raoxuan98-hash/project_clip_continual_learning.git
```

---

## 4. 项目快速概览

> 下面是为 AI agent 提供的高层上下文，避免每次重新探索。

### 研究问题
CLIP 模型在持续学习中的灾难性遗忘。微调会破坏跨模态对齐结构 → 零样本 OOD 泛化能力下降。

### 两大创新（训练端 + 推理端协同）

| # | 创新 | 核心文件 | 一句话 |
|---|------|---------|--------|
| 1 | **LoRA-NSP**（训练端） | `src/models/lora_sgp.py`, `src/trainers/lora_nsp_trainer.py` | LoRA 更新投影到预训练权重的零空间 + 蒸馏损失 |
| 2 | **LR-RGDA 集成分类器**（推理端） | `src/classifiers/lr_rgda_classifier.py`, `src/classifiers/gaussian_classifier.py` | 低秩正则化高斯判别分析 + 零样本集成 |

### 实验基线（4种）

| 基线 | 训练 | 分类器 | 对应配置文件 |
|------|------|--------|-------------|
| **B0** | 无 | 零样本 | `configs/experiments/incremental_b0_zeroshot.yaml` |
| **B2** | 标准 LoRA | 零样本 | `configs/experiments/incremental_b2_lora_zs.yaml` |
| **B3** | LoRA-NSP | 零样本 | `configs/experiments/incremental_b3_lora_nsp_zs.yaml` |
| **B4** | LoRA-NSP | LR-RGDA + 零样本集成（完整方法） | `configs/experiments/incremental_b4_lora_nsp_full.yaml` |

### 数据集
- **10 个 X-TAIL 数据集**：aircraft, caltech101, dtd, eurosat, flowers, food101, mnist, oxford_pets, stanford_cars, sun397
- **参考数据集**：Flickr8K（蒸馏用）
- **基础模型**：`openai/clip-vit-base-patch16`
- **16-shot 训练**

### 关键超参数（经网格搜索优化的）

| 参数 | 最优值 | 说明 |
|------|--------|------|
| LR-RGDA α₁ / α₂ / α₃ | 0.3 / 2.0 / 0.5 | 协方差正则化权重 |
| Ensemble α | 0.5 | LR-RGDA 的融合权重 |
| LoRA rank | 4 (训练) / 32 (LR-RGDA) | |
| cov_momentum（已废弃） | 0.9 | 改为等权平均，`update_covariance_history` 不再使用 |
| 蒸馏权重 FD / CD | 1.0 / 1.0 | |

### 完成状态

| 类别 | 状态 |
|------|------|
| 核心算法 | ✅ ~100% |
| 实验脚本 + 配置 | ✅ ~95% |
| Phase 1 实验（推理端） | ✅ 完成 |
| Phase 3-4 实验（训练端） | ⚠️ ~60%（框架就位，部分未跑完） |
| 超参数优化 | ✅ ~100% |
| OOD 检测模块 | ✅ 已删除（`src/detectors/`, `src/routing/`, 相关脚本） |
| 代码清理 | ⚠️ ~60%（有旧版遗留代码） |
| 单元测试 | ⚠️ ~20% |
| 文档 | ✅ ~95%（有 PROJECT_DOCUMENTATION.md + PRESENTATION.md） |

---

## 5. 代码组织速查

```
project_clip_continual_learning/
├── AGENTS.md                          # ← 本文件
├── chat-history/                      # 对话记录（必须更新）
├── chat-history-for-paper-writing/    # 论文叙事与写作演化记录
├── meta-prompts/                      # 跨会话稳定的项目全局认知
├── paper_writing/                     # 论文写作（LaTeX模板、草稿、参考论文）
├── main_incremental.py                # 正式持续学习入口
├── main_joint.py                      # 联合训练/分类器研究入口
├── src/
│   ├── classifiers/                   # LR-RGDA, LDA, QDA 分类器
│   ├── trainers/                      # LoRA-NSP 训练器
│   ├── models/                        # CLIP 封装, LoRA 变体, 蒸馏损失
│   ├── experiments/                   # 实验入口脚本
│   └── utils/                         # Config, 指标, 特征提取, 评估
│   (OOD 检测模块 src/detectors/ 和 src/routing/ 已删除)
├── configs/
│   ├── base/default.yaml              # 基础配置（所有实验继承）
│   └── experiments/*.yaml             # 实验特定配置
├── scripts/                           # 正式工具、启动器、debug 与 legacy 入口
├── tests/                             # 研究型实现检查
├── demos/                             # 探索性 notebook
├── artifacts/                         # 本地日志、结果、生成图和工具会话（不入库）
├── scenario_datasets/                 # 数据集加载器
├── experiments/                       # 实验结果输出
├── optimization/                      # 超参数优化结果
└── paper_writing/paper-template/figures/ # 论文正式图片
```

### 关键实验入口

| 文件 | 功能 |
|------|------|
| `src/experiments/run_continual_learning.py` | 持续学习主实验（B0/B2/B3） |
| `src/experiments/run_continual_learning_routing.py` | 带集成分类器的持续学习（B4） |
| `src/experiments/run_continual_learning_routing_v2.py` | v2 版本 |
| `src/experiments/generate_paper_tables.py` | 论文表格生成 |

---

## 6. 工作守则

### Git 提交策略

**每次重要的修改后必须提交并推送至 GitHub。**

这是因为 AI agent 没有长期记忆，文件系统是唯一可靠的记忆载体。不及时提交 = 丢失修改。

#### 什么时候必须提交？
- 创建或修改了核心代码文件（`src/` 下的任何改动）
- 修改了配置（`configs/`）
- 修改了实验脚本（`scripts/`）
- 更新了 `AGENTS.md` 或 `chat-history/`
- 添加了新的文件
- 完成了某个实验/任务

#### 提交规范

```bash
# 查看当前变更状态
git status

# 添加所有改动的代码文件（注意：不要添加不应跟踪的文件）
git add -A

# 提交并附上清晰的提交信息
git commit -m "[主要改动] 简要描述改动内容"

# 推送到 GitHub
git push
```

#### 提交信息格式

```
[改动类型] 具体描述
```

改动类型建议用：`[代码]`, `[配置]`, `[实验]`, `[文档]`, `[修复]` 等。

**不要在本地积累大量未提交的修改后再一次性提交。** 多个小提交比一个巨型提交更易于回溯。

#### .gitignore 说明

已配置了以下排除规则，`git add -A` 时不会包含：

| 目录/文件 | 原因 |
|-----------|------|
| `experiments/`, `optimization/` | 实验输出、超参搜索结果（已用 `git rm --cached` 解除跟踪） |
| `paper_writing/reference_papers/` | 他人论文 PDF + 截图，有版权问题 |
| `paper_writing/my_original_papers/` | 个人推导笔记 PDF（本地保留） |
| `paper_writing/讨论/` | 历史讨论记录 |
| `paper_writing/.workbuddy/` | 本地 AI 工具配置 |
| `paper_writing/generated-images/` | AI 生成图 |
| `*.pdf` | 任何编译生成的 PDF |
| `.workbuddy/` | 任何 .workbuddy 目录 |
| `*.pt`, `*.pth`, `*.pkl` | 模型权重和序列化文件 |
| `__pycache__/`, `*.pyc` | Python 缓存 |

### 修改代码前
- 先读 `AGENTS.md` 和最近一次的 `chat-history/` 记录，了解当前上下文
- 如果计划改动涉及多个文件或架构调整，先和用户讨论方案

### 修改代码后
- 如果修改了本 `AGENTS.md` 中提到的架构/参数/结构，同步更新
- 记录到 `chat-history/` 中
- **提交并推送至 GitHub**（见上方的 Git 提交策略）

### 运行实验

**牢记：代码在本地的 Mac 上，执行在远程 GPU 服务器上。**

标准流程：

```bash
# 1. 本地：提交并推送
cd /path/to/project_clip_continual_learning
git add -A && git commit -m "[实验] ..." && git push

# 2. SSH 到服务器拉取并运行
ssh raoxuan@10.20.34.30 "
  cd /home/raoxuan/projects/clip_ood &&
  git pull &&
  python scripts/run_cached_experiment.py --config ...
"
```

- 配置文件在 `configs/experiments/`，用 YAML 覆盖继承体系
- 数据路径（服务器上）：`/home/raoxuan/projects/data/X-TAIL/`
- 使用 `scripts/` 下的 shell 脚本批量运行实验
- 实验结果在服务器上的 `experiments/` 目录，不会被 `.gitignore` 误删除

---

## 7. 重要约定

- **`src/classifiers/` 中的 `lr_rgda_classifier.py` 和 `gaussian_classifier.py`**：前者是高层封装（继承自 `da_classifier_builder.py` 构建器），后者是底层 nn.Module。注意区分。
- **`src/models/trainer.py`** 是旧版/遗留代码。新的训练逻辑在 `src/trainers/lora_nsp_trainer.py`。
- **`AUROC` vs `AUROC`**：代码中两者混用，注意不要引入拼写不一致。
