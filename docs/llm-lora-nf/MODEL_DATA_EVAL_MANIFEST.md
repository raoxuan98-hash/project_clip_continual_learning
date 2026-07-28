# LLM LoRA-NF 模型、数据与评测清单

**版本**: 0.6
**日期**: 2026-07-26
**状态**: Phase -1 资源、数据、基线和 evaluator 已锁定；GPU 环境已验证，正式训练等待空闲设备

---

## 1. 运行位置

```text
SSH: raoxuan@10.20.34.30
CLIP 活代码仓库:
/home/raoxuan/projects/project_clip_continual_learning

隔离 LLM worktree:
/home/raoxuan/projects/project_clip_continual_learning_llm

LLM 代码:
/home/raoxuan/projects/project_clip_continual_learning_llm/llm_lora_nf

数据根:
/home/raoxuan/projects/data/
```

采用 sibling Git worktree 是为了不触碰服务器当前 `average-push` 分支上的未提交 CLIP 改动。

LLM Git 分支：`llm-lora-nf`；已推送提交：

- `2b24ddd`：隔离核心与测试；
- `00bc1b9`：训练链路与基线初始化；
- `4b73851`：统一 SFT 协议与七方法链路。

当前批次包含 checkpoint v2、merge/eval、矩阵启动器、
Track A、多种子统计、正式训练语义修正、目录级完整性链与低存储
checkpoint 保留策略；代码任务提交 `006805ed` 的服务器锁定环境完整测试
为 `102 passed`，正式运行前
仍须提交并取得干净 SHA。

## 2. 网络与缓存

服务器禁止直连 `huggingface.co`。

优先级：

1. 服务器已有且 revision 可验证的缓存；
2. ModelScope；
3. `HF_ENDPOINT=https://hf-mirror.com`。

下载完成后正式运行：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

模型、数据和 cache 均不进入 Git。

## 3. 模型

| 角色 | 规范 checkpoint | 类型 | 主用途 | 首选来源 | revision | 服务器路径 |
|---|---|---|---|---|---|---|
| smoke | `Qwen/Qwen3-0.6B` | post-trained / Instruct | CPU/GPU 链路、低成本消融 | ModelScope | `master`, updated `1753546353` | `/home/raoxuan/projects/data/llm_lora_nf/model_cache/Qwen/Qwen3-0___6B` |
| core | `Qwen/Qwen2.5-1.5B-Instruct` | Instruct | 跨家族主表 | ModelScope | `master`, updated `1740586647`；文件 revision 见 snapshot manifest | `/home/raoxuan/projects/data/llm_lora_nf/model_cache/Qwen/Qwen2___5-1___5B-Instruct` |
| scale | `Qwen/Qwen2.5-3B-Instruct` | Instruct | 规模趋势 | ModelScope | 待核验 | 待核验 |
| scale | `meta-llama/Llama-3.2-1B-Instruct` | Instruct | 规模趋势 | 服务器缓存/HF-Mirror | 待核验 | 待核验 |
| core | `meta-llama/Llama-3.2-3B-Instruct` | Instruct | LoRA-Null 协议、主表 | ModelScope `LLM-Research/Llama-3.2-3B-Instruct` | `e443548a5da3c59ed14484f4bf4a3c61cccd7cab` | `/home/raoxuan/projects/data/llm_lora_nf/model_cache/LLM-Research/Llama-3___2-3B-Instruct` |

Qwen3 受控运行固定 `enable_thinking=False`。

Qwen3-0.6B snapshot 已核验：

- ModelScope license：Apache-2.0；
- `config.json` SHA256：
  `660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd`；
- `tokenizer_config.json` SHA256：
  `d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101`；
- `model.safetensors` 大小：`1,503,300,328` bytes；
- 本地 snapshot 清单保存在模型 cache 内，不进入 Git。

Llama-3.2-3B-Instruct 已核验：

- ModelScope license：Llama 3.2；
- 两个权重分片大小：`4,965,799,096` 与 `1,459,729,952` bytes；
- `config.json` SHA256：`39fb36…`；
- `tokenizer_config.json` SHA256：`9823dc…`；
- Transformers 离线 CPU 加载为 `LlamaForCausalLM`，28 层；
- tokenizer 含 chat template，pad/eos 均为 `128009`。

Qwen2.5-1.5B-Instruct 已完成 ModelScope snapshot 和离线加载；license 为
Apache-2.0。其逐文件 revision/SHA 保存在模型目录的
`local_snapshot_manifest.json`，不得用 `master` 字样替代正式运行时的
逐文件清单校验。

## 4. attention-only 目标模块

Track B 所有方法统一：

```text
q_proj
k_proj
v_proj
o_proj
```

不训练 MLP、embedding、norm 或 LM head。

## 5. 正式数学/知识保持协议

### 5.1 训练

- 数据：MetaMathQA；
- 样本：按 LoRA-Null 官方顺序取前 100,000 条；
- checkpoint：Instruct；
- loss：只在 assistant response token 上计算；
- 数值门控：每个 optimizer step 前要求 accumulated loss 与 gradient norm
  有限，保存前要求全部 trainable adapter tensor 有限；任何失败均禁止
  生成正式 checkpoint；
- 初始化证据：除固定 `assert_close` 门控外，每个 run 还封存 initialized
  logits 相对同一 base logits 的 max/mean absolute error、RMSE、reference
  RMS 和 relative RMSE；数学与代码评测入口会重新验证这些量均为有限值；
- Track B 训练编码固定 `left_preserve_response`：先渲染完整 Instruct
  chat，再从左侧裁到 512 token，以优先保留 assistant response；正式配置
  门禁拒绝回退到 tokenizer 默认右截断；
- epoch：1；
- optimizer：AdamW；
- learning rate：`2e-5`；
- global batch size：`128`；
- scheduler：cosine；
- warmup ratio：`0.03`；
- LoRA rank：`128`；
- LoRA alpha：`128`；
- LoRA dropout：`0.0`；
- seeds：pilot `42`；正式 `42/43/44`；
- precision：BF16 主训练，不使用量化作为主表。

### 5.2 LoRA-NF calibration

- 数据：NQ Open；
- 样本：固定 256 条；
- max sequence length：1024；
- 使用 Instruct chat template；
- 纳入所有非 padding token；
- 统计：FP32 second moment；
- eigensolver：FP64 dense reference；大模型可用经误差验证的低秩求解；
- `epsilon=0.20`；
- `rho=0.02`。

### 5.2.1 Track B LoRA-Null calibration

Track B 只统一 attention-only 目标范围与训练预算，不把 LoRA-Null 的
方法特定 calibration 替换成 LoRA-NF calibration：

- 数据：NQ Open train；
- sampler：官方 raw character spans；
- seed：`233`；
- 样本与长度：`256 × 2048`；
- batch size：`1`；
- normalization：`input / torch.max(input).abs()`；
- moment：每个样本累计 \(X^\top X/256\)；
- 分解：FP32 `torch.linalg.svd` 左奇异向量的最低能量 rank-\(r\) 尾部；
- hook/adapter 范围：只限 `q/k/v/o`；
- initialization 后训练期间不再使用 runtime filter。

正式 runner 对样本数和长度做硬门控，并把实际 sampler、长度和 calibration
耗时写入 `run_report.json`。CPU smoke 最多使用 64 token，仅验证链路，
不具备正式结果资格。

### 5.3 新任务能力

- `gsm8k_cot`；
- `minerva_math500`（MATH-500）。

### 5.4 知识保持

- TriviaQA；
- NQ Open；
- WebQuestions/WebQS；
- 原始 exact match；
- 相对未微调 Instruct checkpoint 的保持率；
- 三项平均与 LoRA-Null 式 geometric mean。

统一 evaluator：`lm-evaluation-harness` v0.4.12，commit
`6d642546f4688648fced259eb3302efd36ece5af`。配置固定在
`llm_lora_nf/configs/evaluation/track_b_math_knowledge.yaml`：

- `apply_chat_template=true`；
- `fewshot_as_multiturn=true`；
- 训练 seed 与评测 seed 解耦；所有训练重复统一使用 lm-eval
  `Python/NumPy/Torch/few-shot = 0/1234/1234/1234`；
- Qwen3 `enable_thinking=false`；
- primary merged checkpoint 使用 FP32；
- generation batch size 固定为 `8`，不使用会在 Llama-3.2-3B 上保守
  退化为 batch 1 的 `auto`；GSM8K 与 Minerva-MATH 的服务器探针均已
  验证该批大小在 24 GiB GPU 上可运行；
- formal run 禁止 task override、`limit` 和 CPU 输出。

数据 revision：

| 数据 | revision | split / fingerprint |
|---|---|---|
| GSM8K | `740312add88f781978c0658806c59bc2815b9866` | train 7473 / test 1319 |
| MATH-500 | `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be` | test 500 |
| TriviaQA | `0f7faf33a3908546c6fd5b73a660e0f8ff173c2f` | train 138384 / validation 17944 / test 17210 |
| NQ Open | `5dd9790a83002ad084ddeb7c420dc716852c6f28` | train 87925 / validation 3610 |
| WebQuestions | `0e473cbe21d1e91ec18da343644498be6a3f5454` | train 3778 / test 2032 |

完整 fingerprint 和离线准备状态保存在服务器
`eval_hf_cache/math_knowledge_prepare_manifest.json`。

### 5.5 LoRA-Null 官方协议审计

官方仓库：`HungerPWAY/LoRA-Null`，审计 commit
`1e6808abb81fe10e50b8172c40ac9a8ab4f11e83`。

- adapter calibration seed：`233`；
- NQ Open calibration：256 条；
- 官方 covariance hook 实际执行
  `input / torch.max(input).abs()`，即除以“有符号全局最大值的绝对值”，
  **不等于**常见的 `input.abs().max()`；随后累计 \(X^\top X/256\)；
- 数学训练：MetaMathQA 前 100,000 条、1 epoch、global batch 128、
  LR `2e-5`、cosine、warmup `0.03`、weight decay `0`、BF16/TF32；
- 对照 Transformers `4.47.0` commit
  `5d7739f15a6e50de416977fe2cc9cb516d67edda` 的 Trainer 语义，每 epoch
  执行 781 个完整 optimizer groups，实际消费 99,968 条，尾部 32 条
  shuffled 样本不进入更新；
- 官方 512 token 右截断产生的全 `-100` labels 行不删除、不重排；
  本实现按原 accumulation/scheduler 位置把它们作为显式零梯度
  micro-batch 消费，并把数量和监督 token 总数写入训练报告。正式启动前
  用 `audit_track_a_supervision.py` 对前 100,000 条逐行封存统计；
- AdamW 固定 betas `0.9/0.999`、epsilon `1e-8`、梯度裁剪 `1.0`；
- rank/alpha/dropout：`128/128/0.0`；
- dropout 为 0 时，官方 LoRA 对照包装
  `q/k/v/o/gate/up/down` 全 7 个线性模块；
- 官方 LoRA-Null 将参考 second moment 的最低能量 rank-\(r\) 子空间投影
  \(W U_{\rm low}U_{\rm low}^{\top}\) 分解为可训练双因子，并从冻结残差
  中减去同一分量，使初始化函数严格等于原 checkpoint；
- 官方脚本使用 Base Llama-2。本文 Track A 改为同家族 Instruct
  checkpoint 时，明确称为“协议对齐的 Instruct 重实现”，不冒充原论文数值。

Track A 独立入口为 `scripts/run_track_a_sft.py`，使用：

- 官方七模块范围；
- NQ Open raw character-span sampler，`256 × 2048`；
- 官方 Alpaca-style MetaMathQA prompt，而不是 checkpoint chat template；
- LoRA 使用 PEFT 官方七模块对照；
- LoRA-Null 使用官方 SVD 左奇异向量尾部子空间；
- 为避免 adapter-only checkpoint 无法重建 FP16 稠密残差的量化误差，
  统一封装在 FP16 calibration 后转 FP32 安装分解；这一数值稳定化必须
  与原官方实现做 Track A 差异审计，不能称为逐位复现。

官方 `step2.sh` 同时运行 v1 与 `train_model_freeze_a.py` v2；正式知识/数学
评测脚本指向 v1（双因子均训练），因此主 Track A 使用 v1。v2 源码存在
重复字符串条件，实际冻结的因子与文件名语义不一致，只作为代码审计事实，
不进入主表。

Track B 将同一 sampler、normalization、moment 和 SVD 初始化机制应用到
`q/k/v/o`，并与其他方法统一 `rank/alpha/dropout`。它不再复用 LoRA-NF
的 chat-template second moment。

## 6. 代码任务

### 6.1 训练数据

- 数据：`fxmeng/pissa-dataset` 的 `python/train.json`，即 PiSSA 官方整理的
  Python-only CodeFeedback 子集；
- revision：`d4746ceca8314940af8a61333bc2d395d9e259c9`；
- 原始来源：`m-a-p/CodeFeedback-Filtered-Instruction` revision
  `a08c213a9748c66c15d0225814be80a2e77adf4a`；
- 获取：`HF_ENDPOINT=https://hf-mirror.com`；
- 文件大小：`244222081` bytes；
- SHA-256：
  `2fc75475ecb65aa9fa7a0f7135e4c0b8c59cfd52ef27a9ea0040ef143b891a82`；
- 使用全部 `104848` 条 Python 样本，不从多语言原始集做未记录的前缀截取；
- 公开文件仅索引 `25233` 的 output 为空字符串。该行保留并锁定，使用
  Instruct chat template 时只监督 assistant 终止标记；不得静默过滤后仍
  声称使用完整公开子集；
- 1 epoch、global batch `128`，每 epoch 为 `819` 个完整 optimizer
  groups，实际消费 `104832` 条，尾部 `16` 条 shuffled 样本丢弃；
- 其余训练预算、attention-only 目标模块、rank/alpha/dropout、
  calibration 和 seeds 与数学 Track B 完全相同。

服务器数据 manifest：

```text
/home/raoxuan/projects/data/llm_lora_nf/codefeedback_python_manifest.json
SHA-256: 35eaabc524ee6b04b3323b63bfe0cb03146b2ec578632d94f5ff5eff8688020b
```

### 6.2 EvalPlus 评测

- evaluator：`evalplus/evalplus` v0.3.1，commit
  `e5d0ed0bab96280b60b637ec7f15b5e4841b0cb2`；
- HumanEval+：official release `v0.1.10`，164 tasks；
- MBPP+：official release `v0.2.0`，378 tasks；
- 生成：Instruct chat template、HF backend、greedy、temperature `0`、
  batch size `1`、每题 `1` 个 solution、`max_new_tokens=768`、
  FP32 merged checkpoint、eager attention；
- 主指标同时报告原始 base tests 和 base+extra tests：
  HumanEval/HumanEval+/MBPP/MBPP+ pass@1；
- 不生成 pass@10/pass@100 样本，避免把生成与存储成本扩大 10--100 倍；
- 代码生成与代码执行分离。生成阶段只使用一张合规 GPU；执行阶段在 CPU
  Bubblewrap 中运行，启用 `--unshare-all`、清空继承环境、禁网、只读
  evaluator/Python/data 挂载和每进程 4 GiB 内存上限；
- timeout 使用 EvalPlus 默认语义
  `max(1 second, ground-truth time × 4)`；`test_details=false` 只做
  fail-fast，不改变 pass/fail；
- 空或不可编译的模型输出计为失败，不把整次 run 判为流水线故障。
- 每个正式 adapted run 的导出、生成和沙箱报告显式绑定训练 seed、训练
  model ID、最终 adapter integrity、training/evaluation commit 和协议
  hash；三 seed 汇总只接受精确 `{42,43,44}`，并计算 mean、sample
  standard deviation、seed-level bootstrap 95% CI、LoRA-NF 对各基线的
  同 seed 配对差值及 adapted 相对共享 base 的变化。

HF-Mirror 的 `evalplus/humanevalplus` 与 `evalplus/mbppplus` 固定 revision
已核验，但其自动转换格式只有合并后的 `test` 代码，不含 official
evaluator 必需的 `base_input`/`plus_input`，因此明确拒绝作为正式 override。
正式数据使用 EvalPlus 官方 GitHub release；这不涉及直连
`huggingface.co`，压缩与解压内容均锁定：

| 数据 | compressed SHA-256 | JSONL SHA-256 | evaluator MD5 |
|---|---|---|---|
| HumanEvalPlus v0.1.10 | `272720b90ac375502c8ed23cd791c2a93dfb22a911641a494da74a426c09f101` | `42526ec0e7d5f3ee0b06d6ced98f8c8bae3d76519151bfb3d36f79010645bd7f` | `fe585eb4df8c88d844eeb463ea4d0302` |
| MbppPlus v0.2.0 | `af43697e8791c4c149bdfd6b489d8b5412507551ac20e28a439f650b8225db63` | `b54e762755248ca411b523c917fa9f93c07b5ff2966bf60b3917b853926a3dad` | `ee43ecabebf20deef4bb776a405ac5b1` |

服务器 EvalPlus data manifest：

```text
/home/raoxuan/projects/data/llm_lora_nf/evalplus_release_cache/evalplus_data_manifest.json
SHA-256: 970ce043160f5d13fd9385c9286d0dda7e62a6cb9362bcc71e9897366250c307
```

## 7. 指令任务

- 训练：WizardLM-Evol-Instruct 前 100,000 条；
- 评测：MT-Bench；
- 补充：IFEval；
- judge model、judge prompt 和版本必须固定；
- 若外部 judge API 不可用，不使用不可比的临时替代结果进入主表。

## 8. 连续学习

- benchmark：TRACE 官方任务流；
- checkpoint：Instruct；
- 主比较：LoRA、DoRA、LoRA-Null、LoRA-NF；
- 任务顺序：官方顺序 + 一个预注册替代顺序；
- seeds：`42/43/44`；
- 指标：任务×时间矩阵、final average、forgetting、BWT、通用能力、指令遵循和安全性变化。

### 8.1 TRACE evaluator 与顺序锁

- 官方仓库：`BeyonderXX/TRACE`；
- evaluator/code commit：
  `462e39f616134f4f819efeb3baea8638c03c7db4`；
- 官方顺序：
  `C-STANCE → FOMC → MeetingBank → Py150 → ScienceQA → NumGLUE-cm → NumGLUE-ds → 20Minuten`；
- 预注册替代顺序：
  `NumGLUE-cm → NumGLUE-ds → FOMC → 20Minuten → C-STANCE → Py150 → MeetingBank → ScienceQA`；
- LoRA 训练 epochs 按任务依次为 `5/3/7/5/3/5/5/7`；
- 官方 `scripts/train_lora.sh` 固定 `max_prompt_len=1024`、
  `max_ans_len=512`；`utils/data/data_collator.py` 将二者相加为 1536
  token 的 combined limit，并在 tokenizer `truncation_side=left` 下
  截断 `prompt+answer`；
- 主任务指标依次为 accuracy、accuracy、ROUGE-L、edit/fuzzy
  similarity、accuracy、accuracy、accuracy、SARI；
- Py150 的官方 similarity 为 `0--100`，进入 OP/BWT 前除以 100；
  SARI 同样归一化到 `0--1`。

TRACE 原文的 \(BWT_t\) 分母为当前已见任务数 \(t\)，而不是旧任务数
\(t-1\)。实现保留该定义，同时额外报告标准 final forgetting：
对每个旧任务计算“学习后至最终的历史最大分数减最终分数”，再在
\(T-1\) 个旧任务上平均。两者不得混称。

### 8.2 数据资格与当前可用 pilot

官方处理数据的 Google Drive 文件 ID 为
`1S0SmU0WEw5okW_XvP2Ns0URflNzZq6sV`，但服务器与本地当前均无法连接
Google Drive。不得把分别下载的相似原始数据集拼接后宣称为 TRACE。

当前已核验的链路/状态消融数据为 TreeLoRA 官方仓库提交的
`LLM-CL-Benchmark_500.tar.xz`：

- repository：`https://github.com/ZinYY/TreeLoRA`；
- commit：`1c7260c42b34e1961283797c742f08b9c3842501`；
- Git blob：`3b1dde98a9658fcf27162d30a4495cbbdcaebf0a`；
- archive SHA-256：
  `956caf12b59add0c7d961cf8ecbad0307e1abca8db7de8873c37d92dd709e9c2`；
- 八个任务均有 500 条训练样本；通常为 100 条 eval + 100 条 test，
  `NumGLUE-cm` 为 41 + 81。

因此该归档固定标记为 `treelora_500_pilot`，只允许链路、状态更新消融
和预算估算，绝不进入正式论文主表。论文正文概括为 5,000 train /
2,000 test，但公开发布档的 eval/test 行数按任务不同；二级复现仓库对
官方压缩包的统计例如 FOMC test 496、MeetingBank test 692、
NumGLUE-cm test 81。因此不得用统一 2,000 的推测覆盖实际发布文件。
正式 `paper_5k` 必须同时满足：

1. 每任务恰好 5,000 条 train；
2. eval/test 非空，且每任务行数与审核后的官方发布档 manifest 精确一致；
3. 八任务 24 个 JSON 文件都有外部审阅后锁定的精确 SHA-256；
4. 来源 manifest 能证明为官方 Drive 文件或内容等价镜像；
5. prompt/answer schema、顺序与 evaluator 输入逐项通过审计。

仅通过 5,000 train、非空 split 和 schema 检查仍标记
`formal_eligible=false`，不能启动正式结果收集。旧版统一要求每任务
2,000 test 的代码门禁已于 2026-07-27 移除，以免错误拒绝官方发布档；
严格性由待锁定的逐任务行数与文件哈希 manifest 提供。

### 8.3 连续状态与单最终 adapter

- 所有主比较方法采用相同的 task merge/reset 受控协议；
- native LoRA-NF 将每任务 \(AP_t,B_t\) 作为低秩重放因子；
- LoRA-Null 同时保存初始化减项，确保从原始 checkpoint 重放的函数相同；
- DoRA 保存每任务 PEFT state 并按顺序 merge/replay；
- 任务阶段只在内存保留这些分支，不写中间模型；
- 序列结束只写一个累计 adapter 目录、一个任务×时间指标/预测证据集；
- 最终 adapter 保存最后的 LoRA-NF 保护 basis、phase count 和谱摘要，
  不重复保存八套历史 basis 或稠密 delta。

### 8.4 连续训练与评测实现

- TRACE split 原始顺序不变地转换为 Instruct chat template；
- Instruct 重实现固定
  `trace_official_combined_left_v1`：先构造完整 chat-template
  `prompt+answer`，再从左侧裁到 `1024+512=1536` token；response-only
  label mask 使用裁剪前的真实 assistant 边界，绝不把右截断造成的
  “无监督回答”误判成需要提高上下文上限；
- 每个训练 stage 在报告中封存 source/full/response token 分布，以及
  1536 token 下 prompt 左裁、完整回答保留和回答部分裁剪的行数；
- 训练 batch 与官方实现一致使用左 padding；生成同样采用 1024 token
  prompt 左截断和最多 512 个新 token；
- 生成固定 greedy、`do_sample=false`、`num_beams=1`；
- 每个 stage 只评测所有已见任务，形成严格下三角任务×时间矩阵；
- prediction artifact 使用确定性 gzip，并绑定 commit、模型 revision、
  config/data hash、seed、任务顺序和生成参数；
- MeetingBank 固定 `rouge==1.0.1`，20Minuten SARI 固定
  `huggingface/evaluate@a7dd338386a4fae9a1767e05eb9ef9479513d9e8`；
- TRACE 原 SARI 包装的尾逗号 tuple 问题按数值意图确定性修正并记录；
- runner 在最终累计 adapter 写入后执行目录完整性和结构重载审计；
- 非正式 smoke 可在上述审计成功后删除 adapter，正式结果禁止删除。

### 8.5 TRACE 状态更新预注册选择

在干净提交 `74a34e9b28690d9e775f506f158e53db038a0819` 上，使用
`treelora_500_pilot`、seed 42、官方顺序、每任务完整 500 train、前 20
test、最多 128 个生成 token 和 64 条 NQ calibration，完成
`reference_fixed` 与 `reference_plus_history` 的八任务状态消融：

- fixed：final average `0.3159303`、TRACE-BWT `-0.0432352`、
  forgetting `0.0649116`、wall time `2185.85s`；
- history：final average `0.2999786`、TRACE-BWT `-0.0515040`、
  forgetting `0.0802903`、wall time `2692.54s`；
- history-minus-fixed：average `-0.0159517`、BWT `-0.0082689`、
  wall ratio `1.2318032`；
- 预注册门限中平均分和 BWT 均失败，耗时门限通过，最终选择
  `reference_fixed`。

`state_selection.json` SHA-256 为
`903310fbc4ad36cac31c26b75eacf1e8163d6bbef8df8d718c94d37f3cf58abd`。
两份 run 的 adapter 均在完整性/重载审计后删除，整个证据目录约 1.6 MB。
该消融硬标 `formal_result_eligible=false`，仅确定后续 LoRA-NF 状态规则。

### 8.6 TRACE-500 四方法受控 pilot

预注册 spec `trace_qwen3_0p6b_method_pilot.yaml` 在干净提交
`de681acfe3cf4dc3d7efff119696979900ffc13f` 上运行。四方法共享
comparison hash
`8376f1604a62ae0ebcde57d6a828000849c74bb82194fc117f4397d3fbd2993d`，
均使用 seed 42、官方顺序、完整 500 train、前 20 test、128 new tokens
和同一 Instruct checkpoint：

| 方法 | Final average | TRACE-BWT | Standard forgetting | Wall seconds |
|---|---:|---:|---:|---:|
| LoRA | 0.32967 | -0.04645 | 0.08880 | 1740.32 |
| DoRA | 0.33469 | -0.04351 | 0.07829 | 2345.82 |
| LoRA-Null | **0.36517** | **-0.03148** | 0.06455 | 2284.54 |
| LoRA-NF | 0.31378 | -0.03816 | **0.05919** | 2201.27 |

LoRA-NF 的 forgetting 最低，但 final average 比 LoRA 低 `0.01589`，
比 LoRA-Null 低 `0.05139`。当前强滤波设置表现为更强保留、更弱适应，
不能据此主张总体优于基线。按目标的“关键负面结果最多一次预注册小型
超参数检查”原则，下一步只检查少量 leakage/energy 设置；仍不改善则
保留可信负面结论，不进行无限调参。

四方法 summary SHA-256 为
`acc1c94e854a38e9b3315f6a8d771b8de39188500fa36b18bcde40e32f021f73`；
所有 adapter 均在完整性/重放审计后删除，证据目录约 3.1 MB。
结果硬标 `formal_result_eligible=false`。

### 8.7 TRACE-500 LoRA-NF 滤波强度消融

按 8.6 的负面结果，只执行一次提交
`ac34edf39045821aed194e0b6ed6a897056bb7f0` 上预注册的五设置检查。沿用
同一 Qwen3-0.6B Instruct checkpoint、seed 42、官方顺序、完整 500
train、前 20 test、128 new tokens、64 NQ calibration 和
`reference_fixed` 状态：

| 设置（energy/leakage） | Final average | TRACE-BWT | Standard forgetting | Wall seconds |
|---|---:|---:|---:|---:|
| `e20_r02`（0.20/0.02） | 0.31378 | -0.03816 | 0.05919 | 2201.27 |
| `e20_r05`（0.20/0.05） | **0.32262** | **-0.02671** | **0.04482** | 2238.69 |
| `e20_r10`（0.20/0.10） | 0.30259 | -0.05540 | 0.08595 | 2208.45 |
| `e20_r20`（0.20/0.20） | 0.30716 | -0.04637 | 0.06014 | 2196.63 |
| `e10_r10`（0.10/0.10） | 0.30033 | -0.05832 | 0.07490 | 2200.15 |

选择门限由 LoRA reference 确定：final average 至少 `0.3246693`、
forgetting 至多 `0.0787988`、TRACE-BWT 至少 `-0.0514489`。
`e20_r05` 的 forgetting/BWT 通过，但 final average 差 `0.0020521`；
其余设置也未三门全过。因此 `passed_settings=[]`，严格按预注册
`default_on_failed_gate` 冻结 `e20_r02`，不事后改选严格支配它但未过
适应性门限的 `e20_r05`，也不继续搜索超参数。

机器生成的 `filter_selection.json` SHA-256 为
`8e0e378e66268c564cd38dad4f2d0d3fe6472cb7582b5e1847ca6d94b8c67667`。
五个设置的 adapter 均已删除，无 optimizer/scheduler/权重残留；本次
新增证据共约 3.1 MB。完整回归为 `150 passed, 3 warnings`。该检查只
具备 `treelora_500_pilot` 资格，不进入正式论文结果。

## 9. 两条评测轨道

### Track A：公开协议对齐

- 使用官方代码、配置和方法特定预处理；
- checkpoint 固定为 Instruct；
- 若原论文使用 Base，只称“协议对齐的 Instruct 重实现”；
- 同 checkpoint/seed/evaluator 下，官方实现与统一封装目标差异不超过 0.5 个绝对分数。

### Track B：统一受控

- attention-only；
- 同 checkpoint、rank、alpha、dropout、数据、预算和 evaluator；
- 方法特定 calibration 允许存在，但单独报告成本；
- 同 rank 主表 + 近似同参数量补充表。
- 每个正式 run 在训练前封存实际可训练参数作用域、FP32 dtype element/
  tensor 计数和排序参数名 SHA-256；作用域外参数、缺失任一 `q/k/v/o`
  目标或非 FP32 adapter 均在训练前失败。

## 10. 基线来源锁定

| 方法 | 主要实现来源 | 审计 revision | Track B 实现 |
|---|---|---|---|
| LoRA | PEFT 0.17.1 | `53c25fe4fdd7c6aa4b40db0560815ea570d32303` | attention-only，PEFT 标准初始化 |
| DoRA | `NVlabs/DoRA` / PEFT | `7e2f10abbe8efe212c8fca1d983ae1d04ef13a18` | `use_dora=True` |
| LoRA-Null | `HungerPWAY/LoRA-Null` | `1e6808abb81fe10e50b8172c40ac9a8ab4f11e83` | 官方 raw-span/signed-max/SVD 初始化，attention-only |
| PiSSA | `MuLabPKU/PiSSA` | `a6e4c9c1d8bc1b73c0fd2f524be8e00e26043395` | `pissa_niter_16`，attention-only |
| MiLoRA | `sufenlp/MiLoRA` | `c3c94693b26c800a96dba84a1fe92d7384b7c28d` | 当前模型设备上 FP32 完整 SVD 的小奇异值初始化，attention-only |
| CorDA | PEFT 0.17.1 (`53c25fe…`) + CorDA 官方协议 | `baffb03ac090f23305e5fb586a2d3c16df7f12db` | KPM raw-span/signed-max FP32 preprocess 后 `init_lora_weights="corda"` |
| LoRA-NF | 本项目 | `llm-lora-nf` 分支 | 标准 LoRA 初始化 + 全程运行时 filter |

PEFT 0.17.1 的 CorDA covariance hook 只兼容 calibration batch size 1：
它仅对 batch 维执行一次 `squeeze(0)` 后调用二维 `Tensor.t()`。Track B KPM
因此使用 CorDA 官方 NQ raw character spans、seed 233、`256 × 2048`、
signed-global-max normalization 和 FP32 covariance。其额外耗时和峰值内存
单独报告；其他方法不继承这一实现限制。

## 11. 资源与结果资格

- 本开发线最多使用 3 张 GPU；
- 每次准入主动留下至少 1 张 GPU；
- 无符合条件 GPU 时只运行远程 CPU smoke；
- CPU 输出标记 `cpu_smoke_only`；
- CPU 指标不得进入正式聚合；
- 正式运行保存 commit、配置 hash、模型/data revision、seed、设备和日志路径。
- export manifest 额外绑定训练时的完整软件分发 hash 与 GPU
  name/capability/memory；多 seed 和跨方法配对拒绝混合训练环境；

### 11.1 Checkpoint 与证据保留

- 同一精确 revision 的基础模型只保存一份共享快照；
- 每个“模型 × 方法 × seed”仅保存一个最终 adapter checkpoint；
- 不保存中间 step/epoch、optimizer 或 scheduler state；
- calibration cache 按内容寻址跨 seeds 复用；
- Track B 标准 LoRA 直接使用锁定的 PEFT 实现；native wrapper 只承担
  LoRA-NF、LoRA-Null 与 MiLoRA 所需的运行时滤波/分解语义；
- PEFT 路径显式设置 `autocast_adapter_dtype=True`，与 native FP32
  adapter 相同；runner 以实际参数审计而不是实现默认值作为正式证据；
- CorDA 首次校准的 per-run eigens/covariance 临时文件在共享 artifact
  cache 写入且复核 SHA-256 后精确回收，不随每个 seed 重复长期保留；
- lm-eval request cache 按基础 snapshot、evaluator/task/data/protocol
  内容寻址跨方法和 seeds 复用，不重复进入每个 raw run；
- CPU smoke 禁用 lm-eval request cache，避免 harness 在 `--limit 1` 时
  仍构建完整任务缓存；仅封存一份小型 cache identity marker；
- 基础模型正式评测由评测矩阵启动器自动创建并跨 stage 复用，地址绑定
  model snapshot、评测协议、数据 manifest 和源代码 commit；同一精确
  身份只保存一份 raw base evaluation；
- FP32 merged checkpoint 仅为 evaluator 临时输入，任一时刻至多一个；
- 导出前扫描 merged root，发现当前目标以外的任何非空产物即停止审计，
  不自动清理来源不明的目录；
- `run_lm_eval.py` 将 export manifest 内嵌进 `evaluation_run.json`，并对
  原始评测目录逐文件 SHA-256 封存；
- 单次汇总成功后，矩阵启动器只删除该精确、已验证可重建的 merged 目录；
- 长期保留共享基础模型、最终 adapters、raw lm-eval/sample logs、完整性
  manifest、汇总和环境记录。

### 11.2 评测数据实物完整性

- `prepare_eval_data.py` 只允许 `https://hf-mirror.com`，逐项解析锁定的
  dataset revision；
- 除 dataset/config/revision、split rows 和 fingerprint 外，还记录实际
  Arrow/cache 文件相对路径、字节数和 SHA-256；
- base 与 adapted 的每次正式离线评测启动前都重新核验这些 cache 文件；
- evaluator 内置任务默认不传 revision，因此运行器从锁定的 v0.4.12
  checkout 复制所需 YAML/utility，注入 `dataset_kwargs.revision`，再以
  `--include_path` 覆盖同名任务；派生文件随 raw output 一起封存；
- 汇总时还要检查 lm-eval 结果内的 resolved task config 确实报告同一
  dataset path/config/revision，不能只相信启动命令；
- `evaluation_run.json` 记录 dataset manifest SHA-256，base/adapted、
  多 seed 和方法配对若数据实物哈希不同则拒绝聚合。

服务器 Phase -1 核验：

- 正式项目环境：
  `/home/raoxuan/projects/project_clip_continual_learning_llm/llm_lora_nf/.venv_gpu`；
- Python `3.10.20`；
- PyTorch `2.5.1+cu124`；
- Transformers `4.57.3`；
- Datasets `3.6.0`；
- PEFT `0.17.1`；
- Accelerate `1.10.1`；
- 复用服务器 `lora_nsp_llm` 的 CUDA/PyTorch，项目环境不重复安装
  PyTorch；RTX 4090 CUDA 张量与反向传播已通过；
- `nvidia-smi` 和 Python NVML 查询均已恢复。当前 6 张卡各有约
  16--17 GB 显存占用，按资源约束暂不启动新的正式训练；
- 提交 `006805ed39193d12406d242874f44e75c45b278c` 的源码在上述项目环境
  完整测试为 `102 passed`，另有一个不影响运行的
  `pynvml` 包名弃用警告。

2026-07-25 的远程 CPU 链路已通过：

- 模型类：`Qwen3ForCausalLM`；
- 28 层、112 个 attention projection；
- LoRA-NF trainable parameters：`2,293,760`（rank 8）；
- 逐 forward filter、1 optimizer step、save/load logits 等价均通过；
- 确定性生成结果：`smoke-ok`；
- 该运行标记为 `cpu_smoke_only`，不得进入论文结果。

2026-07-26 在干净提交
`d817858bc9d125b689e7bbdb244d56c280cfed48` 上重新完成同一闭环：

- 运行前复核模型目录完整性，snapshot manifest SHA-256 为
  `744cdbbd72f0bc2df371ce77f620c1225c7d21a1679f3aeb7225800a78f97b30`；
- `source_dirty=false`，LoRA-NF module 数量 `112`，可训练参数
  `2,293,760`；
- 一步优化、checkpoint 完整性、保存/重载 logits 等价和确定性生成
  `smoke-ok` 均通过；
- 报告位于外部数据根的
  `cpu_smoke_only/d817858-qwen3-0p6b-lora-nf/smoke_report.json`，
  SHA-256 为
  `60573de96281ccc7e74016e2b6c2af809381a306c042481ba34dfdc6b0fc9319`；
- 约 11.7 MB 的可重建 CPU smoke adapter 已在验证后精确删除，目录只
  保留约 16 KB 报告；
- 报告硬标记 `formal_result_eligible=false`，不进入论文表格。

2026-07-26 在干净提交
`2cdd550075eda333a98b6f5292c50cdcac173a41` 上完成 TRACE GPU
chain-smoke：

- 数据固定为 `treelora_500_pilot`，仅截取首任务 2 train / 1 test，
  输出标记 `gpu_chain_smoke_only`；
- 使用 GPU 0，资源门禁同时记录 GPU 5 保持空闲；
- Qwen3-0.6B 完整 q/k/v/o LoRA-NF 可训练参数 `2,293,760`；
- 完成 NQ Instruct 二阶矩校准、一步训练、greedy 生成、TRACE 指标、
  累计 adapter 完整性和结构重载；
- 峰值显存 `1,838,169,600` bytes，adapter 审计后自动删除；
- canonical 报告 SHA-256：
  `63774d290062fbdd8064d464d100268da54dcbed74b9bd151156c984e525de7f`；
- `formal_result_eligible=false`，单样本 0 分仅验证链路，不作为性能证据。

另已完成：

- Qwen3-0.6B 七种方法各 1 个真实数据 optimizer step；
- Qwen2.5-1.5B-Instruct LoRA-NF 真实数据 1-step smoke；
- native checkpoint v2 的 q/k/v 真共享 filter basis；
- FP32 native merge 与标准 HF checkpoint 导出；
- lm-eval NQ Open `limit=1` CPU 闭环；
- Track A、统计、训练语义、完整性、checkpoint-retention 和派生证据
  来源测试均已纳入当前回归测试。

## 12. 待核验项

- [x] 服务器 Git worktree 和 LLM 分支；
- [x] 服务器 Python/CUDA/NVML 状态；
- [x] ModelScope 可用性；
- [x] HF-Mirror 可用性；
- [ ] 五个模型全部缓存；当前已完成 Qwen3-0.6B、Qwen2.5-1.5B、Llama-3.2-3B；
- [x] LoRA-Null 官方代码 commit 与 dropout；
- [x] PiSSA、MiLoRA 官方实现 commit；
- [x] CorDA、DoRA 官方实现 commit；
- [x] MetaMathQA、NQ Open 和数学/知识评测数据 revision；
- [x] 数学/知识 lm-eval commit；
- [x] EvalPlus evaluator commit 与 HumanEval+/MBPP+ 数据；
- [ ] FastChat/IFEval evaluator commit；
- [x] TRACE evaluator commit 与指标兼容层；
- [x] 当前三个缓存模型的 chat template、EOS/pad 和最大长度链路；
- [x] 静态 adapter/filter/moment 估算脚本；
- [ ] GPU 资源满足准入后的实测峰值与吞吐。
