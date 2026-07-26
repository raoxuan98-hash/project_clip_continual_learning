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
checkpoint 保留策略；服务器锁定环境完整测试为 `89 passed`，正式运行前
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

- 训练：CodeFeedback 前 100,000 条；
- 评测：HumanEval、MBPP；
- evaluator：官方/BigCode evaluator 的固定 commit；
- execution sandbox、pass@k、temperature 和 sample 数必须锁定后再运行。

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

- 本开发线最多使用 2 张 GPU；
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
- 当前源码在上述项目环境完整测试为 `89 passed`，另有一个不影响运行的
  `pynvml` 包名弃用警告。

2026-07-25 的远程 CPU 链路已通过：

- 模型类：`Qwen3ForCausalLM`；
- 28 层、112 个 attention projection；
- LoRA-NF trainable parameters：`2,293,760`（rank 8）；
- 逐 forward filter、1 optimizer step、save/load logits 等价均通过；
- 确定性生成结果：`smoke-ok`；
- 该运行标记为 `cpu_smoke_only`，不得进入论文结果。

另已完成：

- Qwen3-0.6B 七种方法各 1 个真实数据 optimizer step；
- Qwen2.5-1.5B-Instruct LoRA-NF 真实数据 1-step smoke；
- native checkpoint v2 的 q/k/v 真共享 filter basis；
- FP32 native merge 与标准 HF checkpoint 导出；
- lm-eval NQ Open `limit=1` CPU 闭环；
- Track A、统计、训练语义、完整性、checkpoint-retention 和派生证据
  来源测试均已纳入当前 `89 passed`。

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
- [ ] bigcode/FastChat/TRACE evaluator commit；
- [x] 当前三个缓存模型的 chat template、EOS/pad 和最大长度链路；
- [x] 静态 adapter/filter/moment 估算脚本；
- [ ] GPU 修复后的实测峰值与吞吐。
