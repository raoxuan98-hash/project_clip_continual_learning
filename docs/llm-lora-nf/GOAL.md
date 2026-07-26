# 目标草案：LoRA-NF 的 LLM 可发表级扩展

**版本**：v5
**状态**：已批准，执行中
**批准时间**：2026-07-25
**批准依据**：用户明确回复“开始执行”
**约束增补**：2026-07-26 用户要求限制 checkpoint 数量和存储负担

---

## 1. 总目标

自主完成 LoRA-NF 从 CLIP 到 0.6B--3B 大语言模型微调的系统性扩展，建立达到公开发表论文水准、可复现且公平的实验基准，验证完整 LoRA-NF 在新任务适配、预训练知识保持和连续学习抗遗忘方面的有效性。

最终交付独立代码、测试、配置、远程实验、强基线复现、机制与效率消融、论文级表格和图、复现文档及有证据约束的论文结论。目标成功的标准是形成可信、可复现的结论，不以 LoRA-NF 必须胜过所有基线为前提。

## 2. 完整 LoRA-NF 的方法要求

LLM 实验必须使用完整 LoRA-NF，包括：

1. 使用校准数据采集各 self-attention 目标投影层的输入激活；
2. 计算激活二阶统计；
3. 构造受保护信号子空间和低能量/零空间；
4. 构造包含泄漏系数的 null-space filter；
5. 在整个训练过程中持续过滤 self-attention 投影矩阵上的 LoRA 分支；
6. 在连续任务之间维护或更新保护状态；
7. 对滤波强度、子空间维度、校准规模和状态更新策略进行消融。

可以采用与稠密滤波矩阵 \(P\) 数学等价的低秩因子化实现，但必须：

- 给出等价推导；
- 通过稠密实现与因子化实现的数值等价测试；
- 验证 forward、梯度和参数更新的一致性；
- 明确报告因子化实现的显存、存储和吞吐收益。

不得将 LoRA-NF 简化为：

- 仅修改 LoRA 初始化；
- 训练开始前的一次性投影；
- 只在部分训练步生效而不说明的滤波；
- 与当前 LoRA-NF 数学定义不一致的近似方法。

执行前必须先只读审计当前 CLIP LoRA-NF 的数学定义和实现语义，形成 LLM 实现规范，防止方法漂移。

### 2.1 LoRA-NF 的作用位置

主实验中的 LoRA-NF 只作用于 Transformer self-attention 的四类线性投影矩阵：

- `q_proj`；
- `k_proj`；
- `v_proj`；
- `o_proj`。

LoRA-NF 不直接修改 softmax 后的 attention score，也不在主实验中作用于 MLP 的 `gate_proj`、`up_proj` 或 `down_proj`。

采用 Hugging Face `nn.Linear` 的行向量输入约定时，待审计确认的目标数学语义为：

\[
y=xW^\top+xP A^\top B^\top,
\qquad
\Delta W_{\mathrm{NF}}=BAP,
\]

其中 \(P\) 为对称的激活空间滤波器。也就是说，LoRA-NF 在 attention 投影矩阵的 LoRA 分支输入侧持续施加滤波，而不是只过滤一次初始化。

正式实现前必须产出 `docs/llm-lora-nf/METHOD_SPEC.md`，并依据当前 CLIP 实现逐项固定：

- 公式、张量形状和左右乘方向；
- filter 在 forward、梯度和 merge/unmerge 中的等价语义；
- `q/k/v` 是否共享同层输入校准基；
- `o_proj` 的独立输入校准基；
- 子空间选择、泄漏系数和数值精度；
- 连续任务间 adapter 与保护状态的更新方式。

低秩实现的“严格等价”仅指：对于同一选定子空间、同一 \(P\) 和同一数值精度，因子化应用与显式稠密 \(P\) 等价。如果子空间由截断 SVD、randomized SVD 或近似统计获得，必须单独报告子空间近似误差，不得宣称与完整协方差滤波完全等价。

## 3. 与 CLIP 代码的隔离

LLM 实现放在新的独立顶层目录：

```text
llm_lora_nf/
├── src/                 # LLM 模型、adapter、滤波与训练实现
├── configs/             # LLM 专属配置
├── scripts/             # LLM 训练、评测和调度入口
├── tests/               # LLM 专属测试
├── evaluation/          # benchmark 适配与指标
├── README.md            # 最小运行与复现入口
└── experiments/         # 服务器实验产物，不纳入 Git
```

强制边界：

- 不把 LLM 实现写入现有 CLIP 的 `src/`、`configs/`、`scripts/` 或 `tests/`。
- 可以只读现有 CLIP 实现，用于审计和对齐数学语义。
- 原则上不修改当前 CLIP 的训练、评测和配置行为。
- 如需复用逻辑，在 `llm_lora_nf/` 内建立独立模块或明确接口，不让 CLIP 代码反向依赖 LLM 模块。
- 根目录只允许进行必要的 `.gitignore`、文档、入口索引和项目记录更新。
- 所有 LLM 开发决策和实验过程写入 `docs/llm-lora-nf/records/`。

`docs/llm-lora-nf/` 是唯一的目标、方法规范和开发记录来源；代码目录不再建立第二套 `docs/`，避免文档漂移。

### 3.1 Git 上传边界

本开发线新增提交只上传复现 LoRA-NF 所需的代码资产：

- `llm_lora_nf/src/`；
- `llm_lora_nf/configs/`；
- `llm_lora_nf/scripts/`；
- `llm_lora_nf/tests/`；
- `llm_lora_nf/evaluation/`；
- 最小 `README.md`、依赖锁定文件和许可证/来源说明；
- 必要的 `.gitignore` 与小型非数据测试 fixture。

以下内容必须保留在本地或服务器，但不得上传到 Git：

- `chat-history/`；
- `chat-history-for-paper-writing/`；
- `paper_writing/`；
- `docs/llm-lora-nf/records/`；
- 模型缓存、ModelScope/Hugging Face 缓存；
- 模型权重、checkpoint、optimizer state；
- 数据集、预处理缓存和生成样本；
- 实验结果、日志、预测文件、图表中间产物；
- 大型二进制文件和归档。

目标获批后，先审计真实 Git 根目录与已跟踪文件，再更新 `.gitignore`。至少覆盖以下模式或等价目录：

```gitignore
chat-history/
chat-history-for-paper-writing/
paper_writing/
docs/llm-lora-nf/records/

model_cache/
models_cache/
hf_cache/
modelscope_cache/
.cache/

llm_lora_nf/data/
llm_lora_nf/datasets/
llm_lora_nf/checkpoints/
llm_lora_nf/experiments/
llm_lora_nf/artifacts/

*.safetensors
*.bin
*.gguf
*.onnx
*.ckpt
*.pt
*.pth
*.pkl
*.pickle
*.tar
*.tar.gz
*.zip
```

`.gitignore` 不会自动停止跟踪已经提交的文件。若上述目录已经被跟踪，获批后应先展示精确清单，再仅从 Git index 移除、保留本地文件；不得删除本地 history 或 `paper_writing/` 内容。

Git 还必须增加上传前检查：默认拒绝新增超过 10 MiB 的文件，除非它是经过审阅的小型必要代码 fixture。不得使用 Git LFS 上传模型、数据或实验产物。
上传前检查必须使用 allowlist，只接受本节列出的 LLM 代码资产、五个
`docs/llm-lora-nf/` 顶层规范文档和 `.gitignore`；未知小文件也必须默认
拒绝，不能只依赖扩展名和已知 cache 路径黑名单。

“仅上传模型代码”指本开发线的新提交范围，不授权删除远端仓库中既有的 CLIP 代码。

## 4. 服务器指向与禁止本地测试

唯一执行服务器：

```text
SSH: raoxuan@10.20.34.30
CLIP 活代码仓库:
/home/raoxuan/projects/project_clip_continual_learning

隔离 LLM Git worktree:
/home/raoxuan/projects/project_clip_continual_learning_llm

LLM 工作目录:
/home/raoxuan/projects/project_clip_continual_learning_llm/llm_lora_nf

数据根目录:
/home/raoxuan/projects/data/
```

项目文档中仍存在遗留路径 `/home/raoxuan/projects/clip_ood`。首次执行前必须通过只读 SSH 检查实际仓库位置，确定唯一规范路径并记录；不得静默创建第二份仓库。

执行规则：

- 本地仅用于代码编辑、文档编写、静态审阅和 Git 管理。
- 本地禁止运行模型加载、数据预处理、依赖运行检查、单元测试、smoke test、训练、推理和 benchmark。
- 所有 Python 测试、模型加载和实验均在远程服务器执行。
- 每次远程运行前，先提交并推送对应代码，再在服务器拉取准确 commit。
- 每次实验记录 commit SHA、模型 checkpoint、数据版本、配置、随机种子、环境和输出路径。
- 正式 export 必须把训练软件环境与 GPU 型号/计算能力/显存身份内嵌进
  manifest；多 seed 汇总和 LoRA-NF-vs-baseline 配对比较只接受完全一致的
  训练环境身份，不把环境漂移混入方法差异；
- 单 run 汇总、seed-level bootstrap 和 sample-level paired bootstrap
  本身也是正式派生证据：必须记录并校验统计脚本所在的 clean commit 与
  软件环境，且与输入 evaluation 的 commit/environment 相同；
- 不清理或覆盖服务器已有模型、数据、共享缓存和其他实验结果；唯一例外是
  4.3 节定义、由本流水线刚生成并通过完整性校验的临时 merged 输出。

### 4.1 GPU 使用与空闲资源约束

服务器 GPU 调度必须遵守：

- 单次任务或本开发线并发任务合计至多使用 3 张 GPU。
- 本项目在每次任务准入时必须主动保留至少 1 张 GPU，不得由本项目把当时的空闲 GPU 全部占满。其他用户之后启动任务不属于本项目可控制范围。
- 启动任何 GPU 任务前，使用 `nvidia-smi` 检查 GPU 数量、显存、利用率和现有进程，不假定固定 GPU 编号。
- 实际允许占用的 GPU 数为：

  \[
  N_{\mathrm{use}}=\max(0,\min(3,N_{\mathrm{idle}}-1)),
  \]

  其中 \(N_{\mathrm{idle}}\) 是启动时真正空闲且显存满足要求的 GPU 数量。
- 若仅有 1 张空闲 GPU，为满足至少保留 1 张空闲 GPU 的要求，本项目不得占用该 GPU。
- 使用 `CUDA_VISIBLE_DEVICES` 显式限定已确认可用的 GPU；不得抢占、终止或干扰服务器上的其他任务。
- 多个实验不得分别启动后合计超过上述限制。实验驱动必须使用本项目共享资源锁或等价的串行/限流机制；获得锁后、真正启动进程前必须再次运行 `nvidia-smi`，避免检查与启动之间的竞态。
- 长时间任务启动后仍需记录 GPU 编号、开始时间、预计资源和输出路径，便于恢复和审计。

### 4.2 无可用 GPU 时的 CPU 链路验证

如果按上述规则计算后 `N_use=0`，不得在本地运行，也不得等待时无记录地占用资源。可以在远程服务器使用 CPU 跑通最小链路，包括：

- 配置解析；
- 模型或极小测试模型的加载路径；
- adapter 注入；
- 极小合成数据或极小真实样本的数据流；
- 单个或极少训练步的 forward、backward、保存和恢复；
- 评测入口与结果文件生成。

CPU 链路验证的限制：

- 只用于发现接口、依赖、形状、设备和保存恢复问题；
- 使用最小样本、最短序列和极少训练步，避免无意义的长时间 CPU 训练；
- 必须在输出和记录中标注 `cpu_smoke_only`；
- 其 loss、速度和任务指标不得作为实验结果；
- 不进入正式统计、论文表格、图或方法结论；
- 正式结果必须在满足 GPU 空闲约束后重新运行。

### 4.3 Checkpoint 与存储负担约束

实验流水线必须采用“最小充分证据”保留策略，不得为每个训练步、epoch
或评测阶段永久复制完整模型：

- 每个“模型 × 方法 × seed”只保留 **1 个最终 adapter checkpoint**；
- 禁止保存中间 step/epoch checkpoint、optimizer state 和 scheduler state；
- 同一模型的基础 Instruct checkpoint 按精确 revision 只保留 1 份共享快照，
  所有方法和 seeds 复用，不重复下载或复制；
- merged Hugging Face checkpoint 只是评测所需的可重建临时产物，不作为
  长期保存证据；
- 任一时刻至多存在 1 个临时 merged checkpoint；评测原始输出写完、
  通过目录级 SHA-256 完整性封存并完成单次汇总后，流水线必须删除该
  merged checkpoint；
- 每次导出前必须扫描专用 merged root：只允许当前精确目标占用唯一槽位；
  若发现其他非空 merged/临时产物，停止并要求审计，不得猜测性批量删除；
- 删除 merged checkpoint 前，评测清单必须内嵌其 export manifest、
  基础模型快照哈希、adapter/training commit 与配置哈希，使后续多 seed
  聚合和样本级 bootstrap 不依赖已删除的稠密模型目录；
- 删除前还必须重新解析单 run summary，确认它精确绑定 adapted/base raw
  evaluation 的 manifest、目录完整性、数据、task definitions、evaluator
  seeds 和 request-cache identity；仅凭聚合子进程退出码为零不够；
- 长期保留的正式证据为：共享基础模型快照、每次训练的最终 adapter、
  完整性清单、原始 lm-eval 输出/样本日志、汇总和环境记录；
- calibration cache 使用内容寻址并跨相同模型/协议复用，不为不同 seed
  重复保存相同统计；
- 正式 LoRA-NF、LoRA-Null 和 CorDA 运行必须使用独立于单 run 输出的共享
  calibration cache；CorDA 首次生成的 per-run eigens/covariance 文件仅
  作为临时中介，在共享 artifact 写入并通过 SHA-256 复核后精确删除；
- lm-eval request cache 使用“基础模型快照 × evaluator/task/data/protocol”
  内容寻址共享；merged 模型统一从共享 base snapshot 加载 tokenizer，
  不把同一请求缓存复制进每个方法/seed 的 raw evaluation 目录；
- request cache identity 同时绑定 Python、关键包版本、完整 installed
  distributions hash、CUDA 与 cuDNN 版本，禁止跨软件环境反序列化和复用
  旧任务实例；
- 训练 seed（42/43/44）只标识训练重复；lm-eval 的 Python/NumPy/Torch/
  few-shot seeds 明确固定为 `0/1234/1234/1234`，不得随训练 seed 漂移，
  从而让所有方法和训练 seeds 使用完全相同的确定性 prompt/request，
  并安全复用一份共享 request cache；
- CPU smoke 即使带 `--limit` 也不得启用 lm-eval request cache，因为
  lm-eval 会忽略 limit 构造整个任务缓存；CPU 链路只保留极小的内容寻址
  identity marker，不生成全量请求缓存；
- CPU smoke、失败运行和诊断临时文件在问题解决并记录后应清理，不进入
  正式结果保留集；
- 所有 checkpoint 和缓存仍须位于 Git ignore 范围，绝不上传 Git。

正式配置和矩阵启动器必须硬校验上述策略；偏离策略的运行不得进入正式
结果。删除动作只允许作用于启动器刚创建、已验证可由“基础模型 + 最终
adapter”重建的精确 merged 输出目录，禁止清理共享模型、adapter、原始
评测证据或其他实验目录。

## 5. 模型与数据获取

服务器不能直接连接 Hugging Face，禁止依赖默认的 `huggingface.co` 下载。

获取优先级：

1. ModelScope；
2. `hf-mirror.com`；
3. 服务器已有且版本可验证的本地缓存。

对需要 Hugging Face 工具链的资源，显式配置：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

模型和数据准备完成后，正式训练和评测使用离线模式：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

每个资源必须记录：

- 规范名称和来源；
- ModelScope ID、镜像 ID 或服务器路径；
- revision/commit；
- tokenizer 和 chat template；
- 权重精度；
- 文件完整性检查结果。

正式训练前必须产出 `docs/llm-lora-nf/MODEL_DATA_EVAL_MANIFEST.md`，固定模型 revision、数据 revision/split、镜像来源、本地路径、chat template、最大长度、生成参数、答案抽取规则和 evaluator commit。

其他约束：

- Qwen 模型优先从 ModelScope 获取。
- Llama 模型优先检查服务器现有缓存，其次使用 HF-Mirror。
- 不因镜像缺失而静默替换模型、数据或版本。
- gated 模型因许可或认证不可用时，报告具体阻塞，不使用非等价模型冒充。
- 权重和数据只保存在服务器缓存或数据目录，不提交到 Git。
- token、账号和认证信息不得写入配置、日志或开发记录。

## 6. 模型范围

### 6.1 低成本开发与验证模型

- `Qwen/Qwen3-0.6B`（官方 instruction-tuned/post-trained checkpoint，不使用 `-Base`）

这是 Qwen 官方发布的最小 Qwen3 dense 模型，可视为约 0.7B 级模型。它用于：

- 完整 LoRA-NF 的远程 smoke test；
- 稠密—因子化等价性验证；
- 训练和评测闭环检查；
- 部分低成本消融；
- 及早发现显存和吞吐问题。

受控评测默认设置 `enable_thinking=False`，避免 Qwen3 的 thinking 模式成为额外变量；若后续研究 thinking 模式，必须作为独立扩展实验。

Qwen3-0.6B 可以进入补充结果，但不能独自支撑“在 1B--3B LLM 上有效”的主要论文结论。

官方来源：

- <https://qwenlm.github.io/blog/qwen3/>
- <https://huggingface.co/Qwen/Qwen3-0.6B>

### 6.2 正式证据模型

- `Qwen/Qwen2.5-1.5B-Instruct`
- `Qwen/Qwen2.5-3B-Instruct`
- `meta-llama/Llama-3.2-1B-Instruct`
- `meta-llama/Llama-3.2-3B-Instruct`

核心主表优先使用 Llama-3.2-3B 和 Qwen2.5-1.5B；其余模型在通过复现和实现门控后扩展。

所有主实验只使用 Instruct checkpoint。若某篇基线论文使用 Base checkpoint，本项目不将 Instruct 结果表述为对其绝对数值的严格复现，而表述为“公开协议对齐的 Instruct checkpoint 复现/重实现”。

## 7. 对比方法与公平性

完整基线集合：

- LoRA
- DoRA
- PiSSA
- MiLoRA
- CorDA
- LoRA-Null
- 完整 LoRA-NF

评测拆分为两个不可混淆的轨道：

### 7.1 Track A：公开协议复现

- 优先使用作者官方代码和公开配置；
- 保留每种方法必要的初始化、校准数据和方法特定预处理；
- 固定到经过记录的官方 commit；
- 如果原论文不是 Instruct checkpoint，只复现其数据、训练和评测协议，不宣称绝对数值严格复现；
- 在同一 Instruct checkpoint、同一 seed 和同一 evaluator 下，官方实现与本项目统一封装的差异原则上不超过 0.5 个绝对分数；超过时必须诊断并记录。

### 7.2 Track B：统一受控比较

- 所有方法仅适配 `q_proj/k_proj/v_proj/o_proj`；
- 统一 Instruct checkpoint、数据、prompt/chat template、训练预算和 evaluator；
- 主表报告同 rank 比较；
- 补充近似同可训练参数量比较；
- 方法必需的校准仍可保留，但校准样本、时间、显存和存储必须单独报告。

通用公平性要求：

- 统一基础模型和 checkpoint 类型；
- 统一目标线性模块；
- 统一数据版本、样本数量和拆分；
- 统一 prompt、chat template 和 response-only loss 规则；
- 统一优化器、训练步数、batch budget 和评测版本；
- 校准、SVD 和统计构建成本单独报告；
- 保留原始指标、均值、标准差和随机种子结果。
- pilot 和链路门控使用 1 个 seed；正式主表固定 3 个 seeds：`42/43/44`。
- 正式结果报告均值、标准差，并在样本级指标允许时补充 bootstrap 95% 置信区间。

QLoRA 或其他量化训练不作为第一阶段主比较，因为量化会引入额外的谱和数值变量；如资源允许，可作为效率扩展。

## 8. 实验路线

### Phase -1：不可跳过的执行前置

- 只读审计当前 CLIP LoRA-NF，产出 `METHOD_SPEC.md`。
- 确认本地真实 Git 根目录、remote 和分支。当前工作目录的 `.git/` 为空，禁止直接 `git init`。
- 只读核实服务器规范路径与遗留路径，确定唯一工作树。
- 审计现有 Git 跟踪清单，准备 `.gitignore` 与仅移除 index、不删除本地文件的精确方案。
- 产出 `MODEL_DATA_EVAL_MANIFEST.md`。
- 估算每种模型、rank、attention-only adapter 和 LoRA-NF basis 的显存与磁盘需求。
- 以上前置完成前不创建正式训练任务。

### Phase 0：远程正确性门控

- 模型：Qwen3-0.6B。
- 方法：LoRA、LoRA-Null、完整 LoRA-NF。
- 完成模块注入、激活统计、滤波、训练、保存、加载和评测闭环。
- 验证稠密 \(P\) 与低秩因子化实现的 forward、梯度和参数更新等价性。
- 如果没有符合空闲约束的 GPU，先用远程 CPU 完成最小链路验证，但不采集正式结果。

### Phase 1：公开协议对齐的 Instruct 复现

- 模型：Llama-3.2-3B。
- 训练：MetaMathQA 前 100k 样本。
- 新任务评测：GSM8K、MATH。
- 知识保持：TriviaQA、NQ Open、WebQuestions。
- 首先复现 LoRA 和 LoRA-Null，再加入完整 LoRA-NF。
- 使用 Track A 检查官方实现与统一封装在相同 Instruct checkpoint 上的一致性。
- 原论文数值只作为上下文；若 checkpoint 类型不同，不把绝对数值差异作为失败。
- 官方实现与统一封装差异超过 0.5 个绝对分数时，不进入完整强基线矩阵，先完成诊断。

### Phase 2：跨模型家族正式比较

- 模型：Llama-3.2-3B、Qwen2.5-1.5B。
- 方法：完整基线集合。
- 正式结果固定使用 `42/43/44` 三个随机种子。
- 扩展 Qwen2.5-3B 和 Llama-3.2-1B，用于规模趋势验证。

### Phase 3：任务扩展

- 数学：MetaMathQA → GSM8K、MATH。
- 代码：CodeFeedback → HumanEval、MBPP。
- 指令微调：公开指令数据 → MT-Bench、IFEval 等。
- 世界知识与通用能力保持：TriviaQA、NQ Open、WebQuestions、语言模型损失或基座模型漂移指标。

### Phase 4：连续学习

- 使用 TRACE 顺序任务流。
- 测量任务×时间矩阵、final average、forgetting、BWT。
- 测量通用能力、指令遵循和安全性变化。
- 至少检查官方任务顺序和一个替代顺序。
- 明确连续任务间 adapter 与保护状态的更新协议。

### Phase 5：机制与效率

- rank；
- 泄漏系数；
- 子空间维度或能量阈值；
- 校准样本数量和来源；
- 固定或动态保护状态；
- 每层独立基与合法共享基；
- 更新响应能量和正交性漂移；
- 峰值显存、tokens/s、预处理时间、额外存储。

## 9. 执行原则

- 使用阶段门控控制实验规模；早期负面结果不能未经诊断就停止，但可信负面结论也是有效交付。
- 对关键异常或负面结果，最多进行两次针对性诊断复跑和一次预注册的小型超参数检查。仍不改善时保留结果、分析原因并继续完成结论，不进行无限调参。
- 不运行无意义的完整笛卡尔积；由低成本结果决定后续资源分配。
- 正式结论必须包含跨模型家族证据和连续学习证据。
- 在证据充分前，不提前把当前 CLIP 论文重写为跨基础模型论文。
- CLIP 专用的 cross-modal distillation 和 LR-RGDA 不包装成 LLM 通用组件。

## 10. 完成交付标准

目标只有在以下交付完成后才算结束：

- 独立且经过服务器测试的 `llm_lora_nf/` 实现；
- 完整 LoRA-NF 数学与实现规范；
- attention-only `q/k/v/o` 作用位置及 Instruct checkpoint 规范；
- 稠密—低秩滤波的数值等价性测试；
- GPU 资源限流和至少保留一张空闲 GPU 的调度保护；
- 统一强基线训练和评测框架；
- LoRA-Null 公开协议复现报告；
- 至少两个模型家族的正式多种子结果；
- TRACE 连续学习结果；
- 机制、超参数和效率消融；
- 可恢复的服务器实验驱动、配置、日志和环境记录；
- 每次运行仅一个最终 adapter、临时 merged checkpoint 自动回收的低存储
  保留策略；
- 完整的 `MODEL_DATA_EVAL_MANIFEST.md`；
- 仅上传代码的 Git ignore、index 审计和 10 MiB 大文件保护；
- 论文级表格、图、统计分析和结论；
- 根据证据决定 LLM 内容进入主文、扩展章节或附录；
- 更新 `docs/llm-lora-nf/records/`、项目 history 和论文写作记录；
- 完成必要的 Git 提交和推送。

## 11. 审批状态与变更记录

- v4 已由用户在 2026-07-25 以“开始执行”明确批准；
- v5 的 checkpoint/存储约束由用户在 2026-07-26 明确提出并即时纳入
  已批准目标；
- v6 的 GPU 上限由用户在 2026-07-27 从 2 张调整为 3 张；“每次准入
  至少留出 1 张真正空闲 GPU”和 CPU 仅用于链路验证的约束保持不变；
- 当前状态为执行中，不再适用草案阶段的禁止执行规则；
- 后续若用户新增约束，先更新本目标与对应开发记录，再在不扩大其他授权
  范围的前提下继续执行。
