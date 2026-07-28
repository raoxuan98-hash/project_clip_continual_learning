# LLM LoRA-NF 服务器执行 Runbook

**版本**: 0.1
**日期**: 2026-07-26
**适用范围**: `/home/raoxuan/projects/project_clip_continual_learning_llm`

本文件只描述远程服务器执行。不得在本地 Mac 运行测试、模型加载、训练
或评测。所有正式命令必须在 `llm-lora-nf` 分支的干净 commit 上执行。

## 1. 固定目录变量

进入服务器的隔离 worktree 后设置：

```bash
cd /home/raoxuan/projects/project_clip_continual_learning_llm/llm_lora_nf

export PYTHONPATH="$PWD/src"
export LLM_DATA_ROOT=/home/raoxuan/projects/data/llm_lora_nf
export TRAIN_ROOT="$LLM_DATA_ROOT/formal_training"
export MERGED_ROOT="$LLM_DATA_ROOT/ephemeral_merged"
export EVAL_ROOT="$LLM_DATA_ROOT/formal_evaluation"
export CODE_MERGED_ROOT="$LLM_DATA_ROOT/code_ephemeral_merged"
export CODE_GENERATION_ROOT="$LLM_DATA_ROOT/code_generation"
export CODE_EXECUTION_ROOT="$LLM_DATA_ROOT/code_execution"
export REQUEST_CACHE_ROOT="$LLM_DATA_ROOT/request_cache"
export EVAL_HF_HOME="$LLM_DATA_ROOT/eval_hf_cache"
export EVAL_MANIFEST="$EVAL_HF_HOME/math_knowledge_prepare_manifest.json"
export EVALUATOR_PATH="$LLM_DATA_ROOT/evaluators/lm-evaluation-harness-v0.4.12"
export EVALPLUS_PATH="$LLM_DATA_ROOT/evaluators/evalplus-v0.3.1"
export EVALPLUS_DATA_ROOT="$LLM_DATA_ROOT/evalplus_release_cache"
export EVALPLUS_MANIFEST="$EVALPLUS_DATA_ROOT/evalplus_data_manifest.json"
export CODE_TRAIN_MANIFEST="$LLM_DATA_ROOT/codefeedback_python_manifest.json"
export TRACE_PILOT_ROOT="$PWD/data/trace_mirror_audit/extracted/LLM-CL-Benchmark_500"
export TRACE_OUTPUT_ROOT="$LLM_DATA_ROOT/trace_pilot"
export NQ_PARQUET="$LLM_DATA_ROOT/dataset_cache/opencompass/nq_open/nq_open/train-00000-of-00001.parquet"
export QWEN3_0P6B="$LLM_DATA_ROOT/model_cache/Qwen/Qwen3-0___6B"
export LLAMA3P2_3B="$LLM_DATA_ROOT/model_cache/LLM-Research/Llama-3___2-3B-Instruct"
export METAMATH_JSON="$LLM_DATA_ROOT/hf_mirror_cache/datasets--meta-math--MetaMathQA/snapshots/aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18/MetaMathQA-395K.json"
```

`EVALUATOR_PATH` 必须指向 clean commit
`6d642546f4688648fced259eb3302efd36ece5af`。训练与测试 Python 固定为：

```bash
export LLM_PY="$PWD/.venv_gpu/bin/python"
```

该轻量项目环境复用
`/home/raoxuan/ENTER/envs/lora_nsp_llm` 中已验证可用的
PyTorch `2.5.1+cu124`，项目层锁定 Transformers `4.57.3`、Datasets
`3.6.0`、PEFT `0.17.1` 和 Accelerate `1.10.1`。不要使用旧
`.venv` 的 PyTorch `2.8.0+cu128`，也不要在项目环境中重复安装
PyTorch。ModelScope 下载使用独立下载环境；正式训练保持离线。

## 2. 每批代码的准入

在任何模型操作前：

```bash
git status --short
git rev-parse HEAD
```

其中 `git status --short` 必须为空。`check_git_payload.py` 检查的是 staged
上传边界，应在提交前、文件已精确 staged 时执行；clean 服务器 worktree
没有 staged 文件，不能用它替代上传前审计。正式 runner 会再次检查整个
`llm_lora_nf/` 是否干净。payload guard 使用 allowlist：未知路径即使很小
也拒绝，history、paper、cache、模型/数据和大文件不能通过。

同步后必须先运行：

```bash
PYTHONPATH=src "$LLM_PY" -m pytest -q
```

服务器的 `llm_lora_nf/` 目录同时包含被 Git 忽略的虚拟环境、数据审计副本
和运行时文件，因此源码同步必须使用明确文件白名单：

- 禁止对整个 `llm_lora_nf/` 使用带 `--delete` 的 `rsync`；
- 禁止覆盖正在被 launcher 或 runner 读取的源文件；
- 先用 `rsync -naci` 对同一白名单 dry-run，确认只出现预期源码文件；
- 正式传输不带 `--delete`，保留相对目录，再用远程 `git diff --check`、
  `git status --short` 和 payload guard 审计；
- 当前 Track A 完成前不进行任何正式树同步。其后的并发批次只同步
  `configs/paper/math_base.yaml`、`math_track_a_llama3p2_3b.yaml`，
  `scripts/{launch_math_matrix,run_sft,run_track_a_sft}.py`，
  `src/llm_lora_nf/{baselines,inject,protocol_validation,resource_guard,training}.py`
  及对应六个测试文件；文档单独按文件同步。

本分支还从原 CLIP 历史继承了 406 个已 tracked 的 history/paper 文件。
它们的路径清单 SHA-256 为
`89fc45f7b39d8a53cf25321aa19c26a2a4391efb427ce95c8ac1ede0b782b375`，
index stage 清单 SHA-256 为
`bbc329e6cab0c68cf978d495275ef098fd509f7f749567e7d4e5bbe9c2302251`
，总字节数 `11,029,722`。Track A 训练与评测封存前禁止改变 HEAD；
封存后先复核上述 count/hash，再只对
`chat-history/`、`chat-history-for-paper-writing/` 和 `paper_writing/`
执行 `git rm -r --cached`。必须确认本地文件仍存在、staged 内容只有
index 删除和本批白名单代码，再提交；不得对这些目录执行文件系统删除。

2026-07-28 的 detached `ef69d35` 演练结果：操作前 tracked 406、操作后
tracked 0、staged deletion 406、全部 staged change 406、工作树缺失文件
0；staged 路径 SHA-256 为
`89fc45f7b39d8a53cf25321aa19c26a2a4391efb427ce95c8ac1ede0b782b375`。
三个目录均命中 `.gitignore`。临时 worktree 已回收，正式工作树保持 clean
`ef69d3543f5b6500eeb229610665de62181f9a3a`。正式 index 清理仍只能在
Track A 训练与评测证据封存后执行。

旧 `ef69d35` 正式树在 Track A LoRA 完成时的结果为 `157 passed`；
LoRA-Null 暴露函数保持参数化的 FP32 舍入缺陷后，等价
\(BA-B_0A_0\) 修复、评测 parser 依赖、并发 launcher、可观测性和
固定 generation batch 8 已在 `/tmp` 隔离副本完成
`184 passed, 3 warnings`。该版本还在
optimizer step 前硬拒绝非有限 accumulated loss/gradient norm，并在保存
checkpoint 前逐项验证全部 trainable adapter 参数有限。旧 base 的
batch-auto partial evidence 已封存；候选通过 payload guard 并提交后，
必须在同一新 commit 上重跑 Track A LoRA、LoRA-Null 与 base。不得把
旧训练/评测产物拼入新 commit 的正式比较。

## 3. 模型与数据

模型只允许 ModelScope、HF-Mirror 或已核验的服务器缓存。正式训练使用
matrix 中已有路径，且每个目录必须包含：

```text
local_snapshot_manifest.json
artifact_integrity.json
```

若新版评测数据 manifest 尚不存在，通过 HF-Mirror 一次性准备：

```bash
python scripts/prepare_eval_data.py \
  --config configs/evaluation/track_b_math_knowledge.yaml \
  --hf-home "$EVAL_HF_HOME" \
  --output-manifest "$EVAL_MANIFEST"
```

若目标 manifest 或临时 manifest 已存在，脚本会停止审计，不覆盖。正式
评测阶段全程启用 offline mode，并在每次 run 前复核实际 cache 文件的
字节数和 SHA-256。

代码任务数据准备：

```bash
"$LLM_PY" scripts/prepare_code_training_data.py \
  --cache-dir "$LLM_DATA_ROOT/hf_mirror_cache" \
  --output-manifest "$CODE_TRAIN_MANIFEST"

"$LLM_PY" scripts/prepare_evalplus_data.py \
  --config configs/evaluation/track_b_code_evalplus.yaml \
  --output-root "$EVALPLUS_DATA_ROOT" \
  --output-manifest "$EVALPLUS_MANIFEST" \
  --reuse-verified
```

`EVALPLUS_PATH` 必须是 clean detached commit
`e5d0ed0bab96280b60b637ec7f15b5e4841b0cb2`。依赖按
`requirements-code-evaluation.txt` 安装，项目环境不得安装第二份 PyTorch。
HF-Mirror 的自动转换 EvalPlus parquet 不是 official base/plus input 格式，
不得替换上述 release 数据。

## 4. 无 GPU 时的唯一允许动作

先检查准入：

```bash
python scripts/check_resources.py --requested 1
```

若无法在至少留空一张 GPU 的前提下获得设备，只允许显式
`--cpu-smoke-fallback` 的小链路。CPU smoke：

- 不能进入正式聚合；
- 不能作为论文结果；
- lm-eval request cache 被强制关闭；
- 不得用 matrix launcher 假装完成正式 stage；
- 输出必须放在独立的 `cpu_smoke_only` 目录，问题记录后再做精确审计。

## 5. 正式 stage 顺序

每个 training launcher 由项目级 training-pipeline lock 阻止两个矩阵进程
并发写同一批输出。launcher 可用 `--max-parallel 1..3` 启动多个互不相同的
单 GPU run；父进程在整个 batch 生命周期持有全局 GPU 锁，先按全局空闲
状态保留至少一张卡，再给每个子进程设置唯一物理
`CUDA_VISIBLE_DEVICES`。子进程跳过重复加锁但会再次执行准入；独立 runner
仍被父进程的全局锁排除。若空闲卡不足，
当前 batch 自动缩小，绝不通过复用同一卡满足并发数。单 run 仍只写一个
最终 adapter。

正式 GPU 数据流水固定 `dataloader_num_workers=4`、
`dataloader_prefetch_factor=2` 和 persistent workers；CPU smoke 强制
`num_workers=0`。服务器测试证明 worker 0/2 的 seed 42 sampler 顺序完全
一致，并用真实 Llama tokenizer + MetaMathQA 通过了双 worker formatter/
prefetch 链路。

数学/知识 lm-eval 的 FP32 generation batch size 固定为 `8`。不得改回
`auto`：2026-07-28 的 Llama-3.2-3B 实测中，`auto` 退化为 batch 1，
23,373 请求约 3.9 秒/请求；batch 8 在 GSM8K 32 题和 Minerva-MATH
16 题探针中均无 OOM，分别约 1.06 和 1.49 秒/请求。探针带 `limit`，
只用于吞吐与显存准入，不作为论文结果。

### 5.1 Track A gate

首次启动某个 Track A 模型/数据组合前，先在 CPU 上执行纯 tokenizer
监督覆盖审计；它不加载模型权重、不产生实验性能：

```bash
"$LLM_PY" scripts/audit_track_a_supervision.py \
  --config configs/paper/math_track_a_llama3p2_3b.yaml \
  --model-path "$LLAMA3P2_3B" \
  --metamath-json "$METAMATH_JSON" \
  --output "$LLM_DATA_ROOT/protocol_audits/track_a_llama3p2_3b_supervision.json"
```

审计必须绑定 clean commit、model snapshot integrity、MetaMathQA SHA-256，
并报告全截断行的精确索引。Track A 保留官方 512 token 右截断；全 `-100`
labels 行仍消费原 accumulation/scheduler 位置，但显式按零梯度处理。
Track B 则由正式配置门禁固定 `left_preserve_response`，两者不能混用。

```bash
python scripts/launch_math_matrix.py \
  --matrix configs/paper/math_matrix.yaml \
  --stage track_a_gate \
  --output-root "$TRAIN_ROOT" \
  --max-parallel 2
```

随后评测：

```bash
python scripts/launch_math_evaluation.py \
  --matrix configs/paper/math_matrix.yaml \
  --stage track_a_gate \
  --training-output-root "$TRAIN_ROOT" \
  --merged-output-root "$MERGED_ROOT" \
  --evaluation-output-root "$EVAL_ROOT" \
  --evaluation-config configs/evaluation/track_b_math_knowledge.yaml \
  --evaluator-path "$EVALUATOR_PATH" \
  --hf-home "$EVAL_HF_HOME" \
  --dataset-manifest "$EVAL_MANIFEST" \
  --request-cache-root "$REQUEST_CACHE_ROOT"
```

### 5.2 Track B protocol gate

只有 Track A 的 LoRA/LoRA-Null 对齐审计通过后：

```bash
python scripts/launch_math_matrix.py \
  --matrix configs/paper/math_matrix.yaml \
  --stage protocol_gate \
  --output-root "$TRAIN_ROOT" \
  --max-parallel 3

python scripts/launch_math_evaluation.py \
  --matrix configs/paper/math_matrix.yaml \
  --stage protocol_gate \
  --training-output-root "$TRAIN_ROOT" \
  --merged-output-root "$MERGED_ROOT" \
  --evaluation-output-root "$EVAL_ROOT" \
  --evaluation-config configs/evaluation/track_b_math_knowledge.yaml \
  --evaluator-path "$EVALUATOR_PATH" \
  --hf-home "$EVAL_HF_HOME" \
  --dataset-manifest "$EVAL_MANIFEST" \
  --request-cache-root "$REQUEST_CACHE_ROOT"
```

### 5.3 主矩阵

只有 protocol gate 达到预注册门槛后，才执行 `main`，命令与上节相同，
把两处 `protocol_gate` 改为 `main`；训练 launcher 继续使用
`--max-parallel 3`。

### 5.4 代码任务训练与分离评测

代码训练沿用同一串行 launcher：

```bash
"$LLM_PY" scripts/launch_sft_matrix.py \
  --matrix configs/paper/code_matrix.yaml \
  --stage protocol_gate \
  --output-root "$TRAIN_ROOT/code" \
  --max-parallel 3
```

正式评测使用可恢复的低存储 launcher。它逐个执行 FP32 导出、GPU 生成和
Bubblewrap CPU 执行，封存合格结果后立即删除当前临时 merged checkpoint：

```bash
"$LLM_PY" scripts/launch_code_evaluation.py \
  --matrix configs/paper/code_matrix.yaml \
  --stage protocol_gate \
  --training-output-root "$TRAIN_ROOT/code" \
  --merged-output-root "$CODE_MERGED_ROOT" \
  --generation-output-root "$CODE_GENERATION_ROOT" \
  --execution-output-root "$CODE_EXECUTION_ROOT" \
  --evaluation-config configs/evaluation/track_b_code_evalplus.yaml \
  --evaluator-path "$EVALPLUS_PATH" \
  --dataset-manifest "$EVALPLUS_MANIFEST"
```

内部仍严格把生成与执行分开，不允许在模型进程中运行生成代码。执行阶段
硬要求服务器可用的 Bubblewrap，清空继承环境并禁用网络。只有
`code_execution_run.json` 标记 formal、raw result 完整性封存成功后，才
允许按第 6 节规则精确删除该 run 的临时 merged checkpoint。

## 6. 自动低存储行为

评测启动器自动执行：

1. 验证本 stage 的全部最终 adapter 和当前 source commit；
2. 按模型 snapshot、协议、数据 manifest、source commit 创建或复用唯一
   base evaluation；
3. 按 base snapshot 和评测协议创建或复用共享 request cache；
4. CorDA 首次校准写入共享内容寻址 cache 后，复核 artifact hash 并精确
   回收单 run 目录中的 eigens/covariance 临时文件；
5. 扫描并锁定唯一 merged 槽位；
6. 一次只导出一个 FP32 merged checkpoint；
7. 完成 raw lm-eval、目录完整性封存和单 run summary，并反向核验 summary
   精确绑定 adapted/base 两份完整性证据；
8. 只删除刚完成且可由 base + adapter 重建的精确 merged 目录。

任何 incomplete output、未封存 cache、外来 merged 文件或 commit 不一致
都会停止。不得用批量删除、猜测性清理或覆盖参数绕过。

### 6.1 TRACE 连续链

当前已核验 TRACE-500 只能运行 pilot 或 chain smoke，不能进入论文主表。
最小链路在资源检查后使用一张合格 GPU；没有可准入 GPU 时才回退 CPU：

训练预处理固定为官方锁定提交的 combined-left 语义：
`max_prompt_len=1024`、`max_ans_len=512`、总长 1536，先构造 Instruct
chat-template 的 prompt+answer，再从左侧裁剪并保留 response-only
labels。不得改回 tokenizer 默认右截断，也不得因为 MeetingBank 原始
prompt 可超过 48k token 就擅自提高所有方法的上下文预算。

```bash
"$LLM_PY" scripts/run_trace_continual.py \
  --config configs/continual/trace_qwen3_0p6b_pilot_lora_nf_fixed.yaml \
  --model-path "$QWEN3_0P6B" \
  --trace-data-root "$TRACE_PILOT_ROOT" \
  --nq-parquet "$NQ_PARQUET" \
  --output-dir "$TRACE_OUTPUT_ROOT/chain_smoke/seed42" \
  --max-tasks 1 \
  --train-first-n 2 \
  --test-first-n 1 \
  --max-steps 1 \
  --max-new-tokens 8 \
  --calibration-samples 2 \
  --cpu-smoke-fallback \
  --delete-checkpoint-after-smoke
```

输出始终标记 `formal_result_eligible=false`。smoke adapter 仅在完整性与
累计分支结构审计通过后删除，预测、指标、报告和哈希证据保留。正式
`paper_5k` 必须先增加已审阅的 24 文件精确来源 manifest；在此之前
runner 会主动拒绝 `gpu_formal`。

提交 `74a34e9` 的预注册状态消融已选择 `reference_fixed`；后续主比较
使用 `trace_qwen3_0p6b_pilot_lora_nf_fixed.yaml`，不得改用 history。
选择证据位于外部数据根：

```text
trace_pilot/state_ablation/74a34e9-qwen3-0p6b-seed42/state_selection.json
```

其 SHA-256 为
`903310fbc4ad36cac31c26b75eacf1e8163d6bbef8df8d718c94d37f3cf58abd`。
该文件只决定状态规则，不具备正式结果资格。

四方法 TRACE-500 受控 pilot 的预注册 spec 为
`configs/continual/trace_qwen3_0p6b_method_pilot.yaml`。它固定 seed 42、
官方顺序、完整 500 条任务训练集、前 20 条测试、128 个生成 token、
64 条 calibration 和审计后删除 adapter。最多并行 3 个单卡 run，且
始终至少保留 1 张真正空闲 GPU。该矩阵仅是
`gpu_chain_smoke_only`，不得进入论文主表。

提交 `de681ac` 的四方法 pilot 已完成；统一 summary 位于：

```text
trace_pilot/method_pilot/de681ac-qwen3-0p6b-seed42/method_summary.json
```

summary SHA-256 为
`acc1c94e854a38e9b3315f6a8d771b8de39188500fa36b18bcde40e32f021f73`。
四方法 final average 排序为
`LoRA-Null > DoRA > LoRA > LoRA-NF`；LoRA-NF 的 standard forgetting
最低。该结果只触发一次预注册的滤波强度诊断，不改变正式结果门禁。

滤波诊断 spec 为
`configs/continual/trace_qwen3_0p6b_filter_ablation.yaml`。提交
`ac34edf` 上的四个新候选已与既有 `e20_r02`、LoRA reference 一并由
`scripts/aggregate_trace_filter_ablation.py` 审计。选择报告位于：

```text
trace_pilot/filter_ablation/ac34edf-qwen3-0p6b-seed42/filter_selection.json
```

报告 SHA-256 为
`8e0e378e66268c564cd38dad4f2d0d3fe6472cb7582b5e1847ca6d94b8c67667`。
五个设置均未通过三项预注册门限，最终按规则冻结 `e20_r02` 并停止继续
调参。所有 smoke adapter 已删除；新增目录约 3.1 MB，仅保留报告、
压缩预测和完整性 manifest。

任何 `--max-tasks`、行数、步数、生成长度或 calibration 截断都会把
GPU 输出硬标为 `gpu_chain_smoke_only`，即使服务器有可用 GPU，也不能
作为完整 pilot 或正式结果。

## 7. 多 seed 汇总

主矩阵全部单 run summary 完成后，将每个
`evaluation_summary.json` 以重复 `--summary` 参数传入：

```bash
python scripts/aggregate_seed_results.py \
  --summary /path/to/seed42/evaluation_summary.json \
  --summary /path/to/seed43/evaluation_summary.json \
  --summary /path/to/seed44/evaluation_summary.json \
  --expected-seeds 42,43,44 \
  --reference-method lora_nf \
  --output "$EVAL_ROOT/main/seed_aggregate.json"
```

实际主表需传入同模型、同 track 的全部方法和 seeds，不只示例中的三项。
输出必须位于 raw evaluation 目录之外。

代码任务使用独立汇总器，`--run-dir` 传入同一模型全部方法的 3-seed
`code_execution` 目录；`--base-run-dir` 传入 launcher 生成的唯一共享
base execution：

```bash
"$LLM_PY" scripts/aggregate_evalplus_seed_results.py \
  --run-dir "$CODE_EXECUTION_ROOT/main/llama3p2_3b/lora/seed_42" \
  --run-dir "$CODE_EXECUTION_ROOT/main/llama3p2_3b/lora/seed_43" \
  --run-dir "$CODE_EXECUTION_ROOT/main/llama3p2_3b/lora/seed_44" \
  --base-run-dir /exact/path/from/code_evaluation_launcher_manifest.json \
  --expected-seeds 42,43,44 \
  --reference-method lora_nf \
  --output "$EVAL_ROOT/code_main_seed_aggregate.json"
```

上例只展示一个方法的参数形状；正式命令必须追加该模型七种方法的全部
21 个 `--run-dir`。汇总器重新核验 generation/execution 两层目录完整性、
训练 seed、最终 adapter 哈希、临时 merged 导出身份、评测数据/evaluator、
源码 commit 与软件环境，并报告 mean、sample standard deviation、
seed-level bootstrap 95% CI、LoRA-NF 相对各基线的同 seed 配对差值，以及
每种方法相对共享 base 的变化。

样本级配对 bootstrap 对每个“LoRA-NF vs baseline、同模型、同训练 seed”
分别运行：

```bash
python scripts/bootstrap_paired_samples.py \
  --config configs/evaluation/track_b_math_knowledge.yaml \
  --reference-run-dir /path/to/lora_nf/raw_run \
  --comparison-run-dir /path/to/baseline/raw_run \
  --bootstrap-samples 10000 \
  --output /path/outside/raw/bootstrap.json
```

seed-level 区间与 sample-level 区间是两种不同证据，论文中必须分开报告。
两个统计入口都会复核当前 `llm_lora_nf/` 是生成 evaluation 的同一 clean
commit，并要求完整软件环境指纹一致；环境或源码有变化时必须在新 commit
下重跑相应评测，不能只重新汇总旧证据。

## 8. 记录与提交

每完成一次测试、CPU smoke、gate 或正式 stage，都更新：

- `docs/llm-lora-nf/records/` 的对应记录；
- `chat-history/` 工程摘要；
- 若改变论文 claim/叙事，再更新 `chat-history-for-paper-writing/`。

历史、论文目录、模型、数据、request/calibration cache、checkpoint 和
raw 输出不得进入 Git。只提交经 payload guard 检查的 LLM 源码、配置、
测试和必要实现说明。
