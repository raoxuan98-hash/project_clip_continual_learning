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
export REQUEST_CACHE_ROOT="$LLM_DATA_ROOT/request_cache"
export EVAL_HF_HOME="$LLM_DATA_ROOT/eval_hf_cache"
export EVAL_MANIFEST="$EVAL_HF_HOME/math_knowledge_prepare_manifest_v2.json"
export EVALUATOR_PATH=/path/to/clean/lm-evaluation-harness-v0.4.12
export EVALPLUS_PATH="$LLM_DATA_ROOT/evaluators/evalplus-v0.3.1"
export EVALPLUS_DATA_ROOT="$LLM_DATA_ROOT/evalplus_release_cache"
export EVALPLUS_MANIFEST="$EVALPLUS_DATA_ROOT/evalplus_data_manifest.json"
export CODE_TRAIN_MANIFEST="$LLM_DATA_ROOT/codefeedback_python_manifest.json"
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

提交 `006805ed39193d12406d242874f44e75c45b278c` 的结果为
`102 passed`。只有当前 commit 的完整结果可以写入新记录；不得沿用旧
测试计数。

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

每个 training launcher 都串行运行，并由项目级 training-pipeline lock
阻止两个矩阵进程并发写同一批输出；每个任务内重新执行 GPU 准入，单 run
使用一张 GPU，始终至少留空一张。

### 5.1 Track A gate

```bash
python scripts/launch_math_matrix.py \
  --matrix configs/paper/math_matrix.yaml \
  --stage track_a_gate \
  --output-root "$TRAIN_ROOT"
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
  --output-root "$TRAIN_ROOT"

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
把两处 `protocol_gate` 改为 `main`。

### 5.4 代码任务训练与分离评测

代码训练沿用同一串行 launcher：

```bash
"$LLM_PY" scripts/launch_sft_matrix.py \
  --matrix configs/paper/code_matrix.yaml \
  --stage protocol_gate \
  --output-root "$TRAIN_ROOT/code"
```

单 run EvalPlus 必须分成生成与执行，不允许在模型进程中直接运行生成代码：

```bash
"$LLM_PY" scripts/run_evalplus_codegen.py \
  --config configs/evaluation/track_b_code_evalplus.yaml \
  --evaluator-path "$EVALPLUS_PATH" \
  --model-path /path/to/base-or-ephemeral-fp32-merged \
  --model-role merged \
  --model-manifest /path/to/merged_export_manifest.json \
  --dataset-manifest "$EVALPLUS_MANIFEST" \
  --output-dir "$EVAL_ROOT/code_generation/model/method/seed"

"$LLM_PY" scripts/run_evalplus_sandbox.py \
  --config configs/evaluation/track_b_code_evalplus.yaml \
  --evaluator-path "$EVALPLUS_PATH" \
  --dataset-manifest "$EVALPLUS_MANIFEST" \
  --generation-dir "$EVAL_ROOT/code_generation/model/method/seed" \
  --output-dir "$EVAL_ROOT/code_execution/model/method/seed"
```

第二步硬要求服务器可用的 Bubblewrap，清空继承环境并禁用网络。只有
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
