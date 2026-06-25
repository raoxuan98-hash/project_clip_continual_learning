# LADA recipe 迁移与 text-RGDA 消融目标记录

**日期**: 2026-06-17
**目标**: 评估 LADA 官方训练 recipe 迁移到当前 `main_incremental.py` 是否带来收益，并测试在关闭 CD/FD 或关闭 LoRA-NSP 后，使用 LADA-style 文本侧机制结合 LR-RGDA 分类器是否提升。

---

## 1. 科学问题

本轮目标拆成两个问题：

1. LADA 的训练微操是否能提升我们的框架：
   - per-dataset training budget；
   - `OneCycleLR`；
   - LADA 官方风格 `batch_size=64, lr=1e-3, weight_decay=5e-4`。

2. 在不依赖 CD/FD 或 LoRA-NSP 时，文本侧 tuning 是否仍能结合 LR-RGDA 产生收益：
   - `fd_weight=0, cd_weight=0`；
   - `lora_type=lora_vanilla` 作为 no LoRA-NSP 对照；
   - `tune_vision_encoder=false, tune_text_encoder=true` 作为 text-only 对照；
   - 评估仍使用 LR-RGDA / zero-shot / ensemble，不把官方 LADA classifier 混入主结论。

---

## 2. 已实现代码开关

### 2.1 LADA recipe 训练预算

在 `main_incremental.py` 中新增：

```text
--train_budget_mode uniform|lada_epochs
--dataset_epoch_overrides aircraft=40,eurosat=100,...
--use_lada_recipe_defaults
```

`lada_epochs` 使用官方 LADA 16-shot epoch schedule：

| dataset | epochs |
|---|---:|
| aircraft | 40 |
| caltech101 | 10 |
| dtd | 30 |
| eurosat | 100 |
| flowers | 30 |
| food101 | 5 |
| mnist | 200 |
| oxford_pets | 10 |
| stanford_cars | 30 |
| sun397 | 10 |

每个任务实际训练步数为：

```text
task_iterations = official_epochs[dataset] * len(train_loader)
```

`--use_lada_recipe_defaults` 是显式 opt-in，会覆盖为：

```text
batch_size = 64
lr = 1e-3
weight_decay = 5e-4
scheduler = onecycle
train_budget_mode = lada_epochs
```

默认行为不变：

```text
train_budget_mode = uniform
scheduler = cosine
iterations = 800
```

### 2.2 OneCycleLR

在 `src/trainers/lora_nsp_trainer.py` 中新增：

```text
--scheduler cosine|onecycle
```

训练循环现在接受每任务独立的 `iterations`，并用这个值设置 scheduler 的 `total_steps` 或 `T_max`。

### 2.3 LADA-style text classifier

在 `main_incremental.py` 中新增：

```text
--text_classifier_mode current|lada_hybrid
```

语义：

```text
current:
  沿用当前实现。每个评估 step 都用当前 adapted text encoder 编码全部 global_class_names。

lada_hybrid:
  seen classes: 使用任务结束时缓存的 tuned text prototypes；
  unseen classes: 使用 frozen CLIP zero-shot text features；
  然后将这个 hybrid text classifier 与 LR-RGDA 做原有 ensemble。
```

重要 caveat：

```text
当前实现对“文本侧微调”的参数化仍是项目已有的 text LoRA，
不是官方 LADA 的 text-side AdaptFormer。

本轮实现精确迁移的是 LADA 的 text prototype sharing / evaluation mechanism：
seen cached tuned text prototypes + unseen frozen CLIP zero-shot features。
```

如果这条路径有收益，下一步再考虑是否需要实现官方 AdaptFormer 结构。

### 2.4 异步评估一致性

`scripts/evaluate_incremental_artifacts.py` 已同步支持 `lada_hybrid`：

- artifact 中保存 `cached_seen_text_features`；
- worker 重建 frozen CLIP；
- seen/unseen text classifier 的拼接逻辑与 inline eval 一致。

---

## 3. 新增 launcher

新增：

```text
scripts/run_lada_recipe_text_rgda_ablation.sh
```

该脚本使用当前入口：

```text
python main_incremental.py
--dataset_sequence
--lora_type
```

不再使用旧的 stale 入口：

```text
src/experiments/run_continual_learning.py
--task_sequence
--method
```

默认两任务 sanity：

```text
TASKS="aircraft caltech101"
SEEDS="42"
BATCH_SIZE=64
ITERATIONS=800
```

可通过环境变量切换四任务：

```bash
TASKS="aircraft caltech101 dtd eurosat" bash scripts/run_lada_recipe_text_rgda_ablation.sh
```

---

## 4. 当前消融矩阵

两任务 sanity 当前配置：

| config | lora_type | FD/CD | vision tuning | text tuning | text classifier | LADA recipe |
|---|---|---|---|---|---|---|
| `base_current` | `lora_nsp` | on | yes | yes | current | no |
| `recipe_current` | `lora_nsp` | on | yes | yes | current | yes |
| `no_fd_cd_current` | `lora_nsp` | off | yes | yes | current | no |
| `no_fd_cd_lada_hybrid` | `lora_nsp` | off | yes | yes | LADA hybrid | no |
| `vanilla_no_fd_cd_current` | `lora_vanilla` | off | yes | yes | current | no |
| `vanilla_no_fd_cd_lada_hybrid` | `lora_vanilla` | off | yes | yes | LADA hybrid | no |
| `text_only_no_fd_cd_lada_hybrid` | `lora_vanilla` | off | no | yes | LADA hybrid | no |

比较逻辑：

1. `recipe_current - base_current`：
   - 估计 LADA training recipe 对当前 strongest path 是否有收益。

2. `no_fd_cd_lada_hybrid - no_fd_cd_current`：
   - 在没有 CD/FD 时，LADA-style text prototype sharing 是否有收益。

3. `vanilla_no_fd_cd_lada_hybrid - vanilla_no_fd_cd_current`：
   - 在没有 LoRA-NSP 且没有 CD/FD 时，LADA-style text prototype sharing 是否有收益。

4. `text_only_no_fd_cd_lada_hybrid`：
   - 接近“只靠文本侧 tuning + frozen image features + LR-RGDA”的对照。

---

## 5. 回撤与停止规则

代码回撤原则：

- 新功能全部由 opt-in flags 控制；
- 默认 `main_incremental.py` 行为保持 `uniform + cosine + current text classifier`；
- 新 launcher 独立，不覆盖旧脚本；
- 训练结果写入独立目录。

长跑停止规则：

- 两任务 sanity 任一配置出现参数错误、NaN loss、全零 logits、矩阵缺列，先停止四任务扩展；
- 同一 bug 连续出现 3 次，不继续盲跑；
- 单个配置如果 30 分钟无训练 step 日志更新，检查 GPU/进程；
- 如果四任务 pilot 中关键对照比 baseline 低超过 5-8 points，除非有诊断价值，不扩展到 10-task；
- 四任务结果必须先过 summarizer 的 task order / matrix consistency 检查，再解读指标。

---

## 6. 验证记录

本地验证：

```bash
python -m py_compile main_incremental.py src/trainers/lora_nsp_trainer.py scripts/evaluate_incremental_artifacts.py
bash -n scripts/run_lada_recipe_text_rgda_ablation.sh
DRY_RUN=1 OUT_DIR=/tmp/lada_recipe_text_rgda_dryrun TASKS="aircraft caltech101" \
  PARALLEL_JOBS=2 GPUS="0 1" bash scripts/run_lada_recipe_text_rgda_ablation.sh
```

远端验证：

```bash
python -m py_compile main_incremental.py src/trainers/lora_nsp_trainer.py scripts/evaluate_incremental_artifacts.py
bash -n scripts/run_lada_recipe_text_rgda_ablation.sh
python main_incremental.py --help | grep -E 'use_lada_recipe_defaults|train_budget_mode|scheduler|text_classifier_mode|dataset_epoch_overrides'
DRY_RUN=1 OUT_DIR=/tmp/lada_recipe_text_rgda_dryrun_remote TASKS="aircraft caltech101" \
  PARALLEL_JOBS=2 GPUS="0 1" bash scripts/run_lada_recipe_text_rgda_ablation.sh
```

以上均已通过。

---

## 7. 当前远端运行状态

远端项目：

```text
/home/raoxuan/projects/project_clip_continual_learning
```

输出目录：

```text
experiments/lada_recipe_text_rgda_ablation_sanity_20260617
```

启动命令逻辑：

```bash
ROOT=/data1/open_datasets/X-TAIL
OUT_DIR=experiments/lada_recipe_text_rgda_ablation_sanity_20260617
TASKS="aircraft caltech101"
SEEDS="42"
GPUS="3"
PARALLEL_JOBS=1
BATCH_SIZE=64
ITERATIONS=800
bash scripts/run_lada_recipe_text_rgda_ablation.sh
```

当前状态：

```text
runner PID: 1599868
current config: base_current_seed42
GPU: 3
```

日志：

```text
experiments/lada_recipe_text_rgda_ablation_sanity_20260617/runner.log
experiments/lada_recipe_text_rgda_ablation_sanity_20260617/logs/base_current_seed42.log
```

首次检查时已经进入训练循环，`Iter[050/800]` 正常输出，无 NaN 或数据加载卡死。

---

## 8. 明确未完成事项

1. 两任务 sanity 仍在运行中，尚未得到结果矩阵。
2. 四任务 pilot 尚未启动，需要等两任务 sanity 至少跑通关键配置后再启动。
3. 当前 “LADA-style text tuning” 仍使用项目已有 text LoRA 参数化；尚未实现官方 AdaptFormer。
4. 若后续结果显示 hybrid text classifier 有明显收益，再决定是否实现官方 AdaptFormer 以进一步贴近 LADA。

---

## 9. 2026-06-17 01:50 更新

### 9.1 修复 monitor 变量展开问题

第一次放置的四任务自动 monitor 有 shell 变量展开问题：

```text
sanity_pid=
```

原因是通过本地 `ssh "... '${SANITY_PID}' ..."` 组合命令时，变量在本地 shell 层被提前展开。

已在远端改为写入显式脚本：

```text
experiments/lada_recipe_text_rgda_ablation_sanity_20260617/four_task_monitor_fixed.sh
experiments/lada_recipe_text_rgda_ablation_sanity_20260617/four_task_monitor_fixed.log
```

修正版 monitor 等待：

```text
SANITY_PID=1599868
```

并在 sanity 完成后检查：

```text
experiments/lada_recipe_text_rgda_ablation_sanity_20260617/incremental_aggregate.md
```

存在且 runner 未报告 ablation failure 时，才启动四任务：

```text
experiments/lada_recipe_text_rgda_ablation_four_20260617
```

### 9.2 修复结果文件命名与 summarizer 兼容问题

发现 `main_incremental.py` 定义了：

```text
file_name_zs = f"{stem}_zs_results.json"
file_name_rgda = f"{stem}_rgda_results.json"
file_name_ens = f"{stem}_ens_results.json"
```

但实际保存 per-classifier 文件时仍使用：

```text
f"{stem}_zs.json"
f"{stem}_rgda.json"
f"{stem}_ens.json"
```

而 `scripts/summarize_incremental_metrics.py` 原先只读取：

```text
*_results.json
```

这会导致 launcher 最后 summary 为空。

已修复：

1. `main_incremental.py` 后续保存：

```text
*_zs_results.json
*_rgda_results.json
*_ens_results.json
```

2. `scripts/summarize_incremental_metrics.py` 递归读取嵌套目录，并兼容旧命名：

```text
*_results.json
*_zs.json
*_rgda.json
*_ens.json
```

远端已验证：用已完成的 `base_current_seed42` legacy 结果可以正常汇总。

### 9.3 已完成的 sanity 子结果

`base_current_seed42` 已完成。

结果文件：

```text
experiments/lada_recipe_text_rgda_ablation_sanity_20260617/runs/base_current_seed42/base_current_seed42_zs.json
experiments/lada_recipe_text_rgda_ablation_sanity_20260617/runs/base_current_seed42/base_current_seed42_rgda.json
experiments/lada_recipe_text_rgda_ablation_sanity_20260617/runs/base_current_seed42/base_current_seed42_ens.json
```

两任务指标：

| method | Transfer | Average | Last |
|---|---:|---:|---:|
| zero_shot | 82.88 | 66.28 | 66.81 |
| lr_rgda | 0.00 | 48.52 | 71.43 |
| ensemble | 82.88 | 67.33 | 68.11 |

注意：这是两任务 sanity 的第一个配置，只用于确认路径和提供 baseline，不作为最终结论。

### 9.4 当前运行状态

当前 sanity 正在跑：

```text
recipe_current_seed42
```

已确认它使用 LADA recipe：

```text
Task 1 training budget: 40 epochs x 25 steps/epoch = 1000 iterations
text_lr = 0.001
```

这说明 `--use_lada_recipe_defaults` 已经生效：

```text
batch_size=64
lr=1e-3
weight_decay=5e-4
scheduler=onecycle
train_budget_mode=lada_epochs
```

---

## 10. 2026-06-17 02:30 更新

### 10.1 两任务 sanity 的 base vs recipe 局部结果

`recipe_current_seed42` 已完成，并且 launcher 已继续进入：

```text
no_fd_cd_current_seed42
```

两任务 sanity 中已经完成的两个配置如下：

| config | classifier | Transfer | Average | Last |
|---|---|---:|---:|---:|
| `base_current_seed42` | zero-shot | 82.88 | 66.28 | 66.81 |
| `base_current_seed42` | LR-RGDA | 0.00 | 48.52 | 71.43 |
| `base_current_seed42` | ensemble | 82.88 | 67.33 | 68.11 |
| `recipe_current_seed42` | zero-shot | 82.64 | 69.57 | 70.51 |
| `recipe_current_seed42` | LR-RGDA | 0.00 | 51.30 | 74.21 |
| `recipe_current_seed42` | ensemble | 82.64 | 70.42 | 71.79 |

`recipe_current - base_current` 的两任务差值：

| classifier | Transfer delta | Average delta | Last delta |
|---|---:|---:|---:|
| zero-shot | -0.24 | +3.29 | +3.70 |
| LR-RGDA | +0.00 | +2.78 | +2.78 |
| ensemble | -0.24 | +3.09 | +3.68 |

解释注意：

```text
这是两任务 sanity，只能说明代码路径和 recipe 对照没有明显坏掉。
它不能外推到四任务或十任务，也不能直接说明 LADA recipe 一定提升我们的最终框架。
```

### 10.2 LADA recipe 生效性检查

日志确认 per-dataset budget 生效：

```text
Aircraft: 40 epochs x 25 steps/epoch = 1000 iterations
Caltech101: 10 epochs x 25 steps/epoch = 250 iterations
```

`recipe_current_seed42` 的 wall time 约为：

```text
01:41:01 start
02:27:25 done
```

其中 LoRA-NSP covariance extraction 明显占时：

```text
Aircraft post-task covariance extraction: 25 batches, about 19.6 s/batch
Caltech101 post-task covariance extraction: 25 batches, about 17.7 s/batch
```

这说明如果继续扩展四任务，耗时主要瓶颈不只是训练 iter 数，也包括 NSP 后处理。

### 10.3 汇总脚本检查

已手动对当前完成结果运行：

```bash
python scripts/summarize_incremental_metrics.py \
  experiments/lada_recipe_text_rgda_ablation_sanity_20260617/runs \
  --expected_tasks "aircraft caltech101" ...
```

检查结果：

```text
matrix warning 为空；
Transfer/Average/Last stored-computed delta 均为 +0.0000。
```

注意：`aggregate` 表按 classifier 聚合，不按 config 聚合；因此比较 `base_current` vs `recipe_current` 时必须看 per-file/per-config 行，而不能直接看 aggregate 均值。

### 10.4 2026-06-17 02:40 monitor 修正

发现原四任务 monitor 的 launch gate 只检查：

```text
sanity PID ended
incremental_aggregate.md exists
runner.log 没有 "at least one ablation job failed"
```

这个逻辑有一个竞态风险：如果 sanity 所有 JSON 已经完成，但 aggregate 尚未生成或需要补跑 summarizer，monitor 可能直接判定 sanity 未通过，从而不启动四任务。

已新增并启用更稳健的 monitor：

```text
scripts/monitor_lada_sanity_then_four.sh
```

新逻辑：

```text
1. 等待 SANITY_PID=1599868 结束；
2. 如果 runner.log 报 failure，则不启动四任务；
3. 如果 aggregate 缺失，但 7 个 config x 3 个 classifier 结果文件齐全，则补跑 summarizer；
4. 如果四任务目录或进程已存在，则不重复启动；
5. 否则启动四任务 pilot。
```

远端当前 monitor 状态：

```text
result watcher PID: 1850712
robust four-task monitor PID: 1857085
old four_task_monitor_fixed PID: stopped
```

当前 sanity 运行状态：

```text
current config: no_fd_cd_current_seed42
progress at 2026-06-17 02:39 CST: about 541/800 iterations
```

### 10.5 2026-06-17 02:58 运行节奏澄清

`no_fd_cd_current_seed42` 不是 LADA recipe 配置，因此仍使用默认：

```text
train_budget_mode=uniform
iterations=800
```

这意味着两任务 sanity 中：

```text
Task 1: 800 iterations
Task 2: 800 iterations
```

而不是像 `recipe_current_seed42` 那样使用：

```text
Aircraft: 1000 iterations
Caltech101: 250 iterations
```

因此 02:50 后看到的 `Iter[550/800]` 是 Task 2 的训练进度，不代表整个 config 已接近结束。后续还需要：

```text
Task 2 remaining training
Task 2 post-task NSP covariance extraction
two-task evaluation
result JSON writing
```

### 10.6 2026-06-17 03:00 远端连接状态

03:00 左右继续检查时，远端仍在：

```text
current config: no_fd_cd_current_seed42
```

日志显示该配置已经进入 Task 2 末尾，并在 02:59:39 完成：

```text
Iter[800/800] | Loss: 1.3180 | Acc: 89.65% | AuxCE: 0.4890 | AuxAcc: 95.34% | FD: 0.0000 | CD: 0.0000
```

随后进入 Task 2 post-task NSP covariance extraction：

```text
=== Extracting Image Encoder Covariances ===
=== Applying Null-Space Projection (NSP) ===
Collecting image features: 2/25 ...
```

之后尝试继续查询远端时，SSH 连接层开始超时：

```text
ssh: connect to host 10.20.34.30 port 22: Operation timed out
```

解释：

```text
这是监控通道的连接超时，不是训练失败证据。
训练 runner、robust four-task monitor 和 result watcher 都是在远端独立进程中运行；
连接恢复后需要继续检查 no_fd_cd_current 是否写出 JSON，以及是否已进入 no_fd_cd_lada_hybrid。
```

---

## 11. 2026-06-17 09:55 更新

### 11.1 远端恢复与 sanity 新结果

09:50 左右 SSH 已恢复。sanity runner 继续执行并完成了：

```text
no_fd_cd_current_seed42
no_fd_cd_lada_hybrid_seed42
```

随后三个 vanilla/text-only 配置启动即失败：

```text
vanilla_no_fd_cd_current_seed42
vanilla_no_fd_cd_lada_hybrid_seed42
text_only_no_fd_cd_lada_hybrid_seed42
```

失败原因：

```text
TypeError: unsupported operand type(s) for /: 'NoneType' and 'int'
```

具体位置：

```text
src/models/lora_baseline.py: self.scaling = lora_alpha / r
```

根因：

```text
main_incremental.py 的 --lora_alpha 默认是 None，并且 help 文本说明 None 应默认 lora_rank；
但 src/models/clip.py 的 lora_vanilla 分支使用 getattr(args, "lora_alpha", rank)，
当属性存在但值为 None 时没有兜底，导致 VanillaLoRALinear 收到 lora_alpha=None。
```

### 11.2 修复

已修复：

```text
src/models/clip.py
src/models/lora_baseline.py
```

修复逻辑：

```python
alpha = getattr(args, "lora_alpha", None)
if alpha is None:
    alpha = rank
```

并为 launcher 增加：

```text
CONFIG_FILTER
```

用于只补跑指定配置，避免覆盖已完成结果。示例：

```bash
CONFIG_FILTER="vanilla_no_fd_cd_current vanilla_no_fd_cd_lada_hybrid text_only_no_fd_cd_lada_hybrid" \
  bash scripts/run_lada_recipe_text_rgda_ablation.sh
```

远端验证：

```text
python -m py_compile src/models/clip.py src/models/lora_baseline.py
bash -n scripts/run_lada_recipe_text_rgda_ablation.sh
DRY_RUN=1 CONFIG_FILTER=... bash scripts/run_lada_recipe_text_rgda_ablation.sh
```

均已通过。

### 11.3 已完成四个 sanity 配置的结果

| config | classifier | Transfer | Average | Last |
|---|---|---:|---:|---:|
| `base_current_seed42` | zero-shot | 82.88 | 66.28 | 66.81 |
| `base_current_seed42` | LR-RGDA | 0.00 | 48.52 | 71.43 |
| `base_current_seed42` | ensemble | 82.88 | 67.33 | 68.11 |
| `recipe_current_seed42` | zero-shot | 82.64 | 69.57 | 70.51 |
| `recipe_current_seed42` | LR-RGDA | 0.00 | 51.30 | 74.21 |
| `recipe_current_seed42` | ensemble | 82.64 | 70.42 | 71.79 |
| `no_fd_cd_current_seed42` | zero-shot | 81.26 | 65.87 | 66.68 |
| `no_fd_cd_current_seed42` | LR-RGDA | 0.00 | 48.65 | 71.61 |
| `no_fd_cd_current_seed42` | ensemble | 81.30 | 66.92 | 68.00 |
| `no_fd_cd_lada_hybrid_seed42` | zero-shot | 83.00 | 66.87 | 68.20 |
| `no_fd_cd_lada_hybrid_seed42` | LR-RGDA | 0.00 | 48.65 | 71.61 |
| `no_fd_cd_lada_hybrid_seed42` | ensemble | 83.00 | 67.74 | 69.20 |

两任务 sanity 的局部观察：

```text
recipe_current vs base_current:
  ensemble Average +3.09, Last +3.68

no_fd_cd_lada_hybrid vs no_fd_cd_current:
  zero-shot Average +1.00, Last +1.52
  LR-RGDA unchanged
  ensemble Average +0.82, Last +1.20
```

解释：

```text
关闭 CD/FD 后，LADA-style hybrid text classifier 的收益来自文本分类器侧；
LR-RGDA 指标完全一致，符合预期，因为 hybrid 只改变 zero-shot/text classifier 侧。
这仍然只是两任务 sanity，不足以作为最终结论。
```

### 11.4 补跑失败配置

已在远端补跑失败的三个配置：

```text
OUT_DIR=experiments/lada_recipe_text_rgda_ablation_sanity_20260617
CONFIG_FILTER="vanilla_no_fd_cd_current vanilla_no_fd_cd_lada_hybrid text_only_no_fd_cd_lada_hybrid"
TASKS="aircraft caltech101"
SEEDS="42"
GPUS="3"
PARALLEL_JOBS=1
```

补跑状态：

```text
retry launcher PID: 2580441
current config: vanilla_no_fd_cd_current_seed42
```

已确认 `vanilla_no_fd_cd_current_seed42` 不再因 `lora_alpha=None` 失败，已经进入训练循环。

### 11.5 四任务接续 monitor

原 robust monitor 已因为首次 sanity runner 报 failure 而退出。为补跑流程启动了新的接续 monitor：

```text
monitor PID: 2585990
SANITY_PID=2580441
FAILURE_LOG=experiments/lada_recipe_text_rgda_ablation_sanity_20260617/retry_failed_configs.log
LOG=experiments/lada_recipe_text_rgda_ablation_sanity_20260617/four_task_monitor_retry.log
```

逻辑：

```text
等待补跑结束；
确认 7 个 config x 3 个 classifier 结果齐全；
必要时补跑 summarizer；
如果四任务目录尚未存在，则启动 four-task pilot。
```

### 11.6 result watcher 修正

补跑流程会保留旧 `logs/launch.log` 中的历史失败记录：

```text
at least one ablation job failed
```

因此 result watcher 如果继续把旧 `launch.log` 中的 failure 作为 final summary gate，会在补跑成功后仍然拒绝记录 final sanity。

已修正：

```text
scripts/monitor_lada_ablation_status.py
```

新完成判据：

```text
7 个 config x 3 个 classifier 结果文件齐全即可视为 experiment complete。
失败状态由缺失结果文件体现，不再读取旧 launch.log 的历史 failure。
```

同时记录段优先读取：

```text
four_task_monitor_retry.log
four_task_monitor_robust.log
four_task_monitor_fixed.log
```

远端已重启 watcher：

```text
result watcher PID: 2600960
```

<!-- lada-ablation-sanity-final -->

## 11. LADA recipe/text-RGDA sanity final summary

Recorded at: `2026-06-17 11:06:01`

Output directory:

```text
experiments/lada_recipe_text_rgda_ablation_sanity_20260617
```

Launcher tail:

```text
[2026-06-17 00:52:26] start base_current_seed42 gpu=3
[2026-06-17 01:41:01] done base_current_seed42
[2026-06-17 01:41:01] start recipe_current_seed42 gpu=3
[2026-06-17 02:27:25] done recipe_current_seed42
[2026-06-17 02:27:25] start no_fd_cd_current_seed42 gpu=3
[2026-06-17 03:08:03] done no_fd_cd_current_seed42
[2026-06-17 03:08:03] start no_fd_cd_lada_hybrid_seed42 gpu=3
[2026-06-17 03:48:11] done no_fd_cd_lada_hybrid_seed42
[2026-06-17 03:48:11] start vanilla_no_fd_cd_current_seed42 gpu=3
[2026-06-17 03:48:17] start vanilla_no_fd_cd_lada_hybrid_seed42 gpu=3
[2026-06-17 03:48:22] start text_only_no_fd_cd_lada_hybrid_seed42 gpu=3
[2026-06-17 03:48:27] at least one ablation job failed
[2026-06-17 09:54:00] start vanilla_no_fd_cd_current_seed42 gpu=3
[2026-06-17 10:20:07] done vanilla_no_fd_cd_current_seed42
[2026-06-17 10:20:07] start vanilla_no_fd_cd_lada_hybrid_seed42 gpu=3
[2026-06-17 10:45:26] done vanilla_no_fd_cd_lada_hybrid_seed42
[2026-06-17 10:45:26] start text_only_no_fd_cd_lada_hybrid_seed42 gpu=3
[2026-06-17 11:05:47] done text_only_no_fd_cd_lada_hybrid_seed42
```

Monitor tail (four_task_monitor_retry.log):

```text
[2026-06-17_09:55:40] robust monitor start sanity_pid=2580441
```

Aggregate summary:

| method | n | seeds | K | missing expected tasks | task order warning | Transfer | Average | Last | Forgetting | matrix warning |
|---|---|---|---|---|---|---|---|---|---|---|
| ensemble | 7 | 42 | 2 |  |  | 83.37 +/- 2.71 | 67.21 +/- 1.64 | 67.87 +/- 2.21 |  |  |
| lr_rgda | 7 | 42 | 2 |  |  | 0.00 +/- 0.00 | 47.81 +/- 2.72 | 70.22 +/- 2.93 |  |  |
| zero_shot | 7 | 42 | 2 |  |  | 83.33 +/- 2.63 | 66.27 +/- 1.67 | 66.60 +/- 2.32 |  |  |

Per-result summary:

| path | method | seed | K | eval max samples | missing expected tasks | task order warning | Transfer | Average | Last | Forgetting | computed Transfer | computed Average | computed Last | Transfer stored-computed | Average stored-computed | Last stored-computed | matrix warning |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| experiments/lada_recipe_text_rgda_ablation_sanity_20260617/runs/no_fd_cd_current_seed42/no_fd_cd_current_seed42_ens_results.json | ensemble | 42 | 2 | 0 |  |  | 81.30 | 66.92 | 68.00 |  | 81.30 | 66.92 | 68.00 | +0.0000 | +0.0000 | +0.0000 |  |
| experiments/lada_recipe_text_rgda_ablation_sanity_20260617/runs/no_fd_cd_current_seed42/no_fd_cd_current_seed42_rgda_results.json | lr_rgda | 42 | 2 | 0 |  |  | 0.00 | 48.65 | 71.61 |  | 0.00 | 48.65 | 71.61 | +0.0000 | +0.0000 | +0.0000 |  |
| experiments/lada_recipe_text_rgda_ablation_sanity_20260617/runs/no_fd_cd_current_seed42/no_fd_cd_current_seed42_zs_results.json | zero_shot | 42 | 2 | 0 |  |  | 81.26 | 65.87 | 66.68 |  | 81.26 | 65.87 | 66.68 | +0.0000 | +0.0000 | +0.0000 |  |
| experiments/lada_recipe_text_rgda_ablation_sanity_20260617/runs/no_fd_cd_lada_hybrid_seed42/no_fd_cd_lada_hybrid_seed42_ens_results.json | ensemble | 42 | 2 | 0 |  |  | 83.00 | 67.74 | 69.20 |  | 83.00 | 67.74 | 69.20 | +0.0000 | +0.0000 | +0.0000 |  |
| experiments/lada_recipe_text_rgda_ablation_sanity_20260617/runs/no_fd_cd_lada_hybrid_seed42/no_fd_cd_lada_hybrid_seed42_rgda_results.json | lr_rgda | 42 | 2 | 0 |  |  | 0.00 | 48.65 | 71.61 |  | 0.00 | 48.65 | 71.61 | +0.0000 | +0.0000 | +0.0000 |  |
| experiments/lada_recipe_text_rgda_ablation_sanity_20260617/runs/no_fd_cd_lada_hybrid_seed42/no_fd_cd_lada_hybrid_seed42_zs_results.json | zero_shot | 42 | 2 | 0 |  |  | 83.00 | 66.87 | 68.20 |  | 83.00 | 66.87 | 68.20 | +0.0000 | +0.0000 | +0.0000 |  |
| experiments/lada_recipe_text_rgda_ablation_sanity_20260617/runs/text_only_no_fd_cd_lada_hybrid_seed42/text_only_no_fd_cd_lada_hybrid_seed42_ens_results.json | ensemble | 42 | 2 | 0 |  |  | 89.33 | 65.32 | 66.27 |  | 89.33 | 65.32 | 66.27 | +0.0000 | +0.0000 | +0.0000 |  |
| experiments/lada_recipe_text_rgda_ablation_sanity_20260617/runs/text_only_no_fd_cd_lada_hybrid_seed42/text_only_no_fd_cd_lada_hybrid_seed42_rgda_results.json | lr_rgda | 42 | 2 | 0 |  |  | 0.00 | 42.32 | 65.08 |  | 0.00 | 42.32 | 65.08 | +0.0000 | +0.0000 | +0.0000 |  |
| experiments/lada_recipe_text_rgda_ablation_sanity_20260617/runs/text_only_no_fd_cd_lada_hybrid_seed42/text_only_no_fd_cd_lada_hybrid_seed42_zs_results.json | zero_shot | 42 | 2 | 0 |  |  | 89.09 | 64.63 | 65.41 |  | 89.09 | 64.63 | 65.41 | +0.0000 | +0.0000 | +0.0000 |  |
| experiments/lada_recipe_text_rgda_ablation_sanity_20260617/runs/vanilla_no_fd_cd_current_seed42/vanilla_no_fd_cd_current_seed42_ens_results.json | ensemble | 42 | 2 | 0 |  |  | 81.62 | 65.89 | 65.00 |  | 81.62 | 65.89 | 65.00 | +0.0000 | +0.0000 | +0.0000 |  |
| experiments/lada_recipe_text_rgda_ablation_sanity_20260617/runs/vanilla_no_fd_cd_current_seed42/vanilla_no_fd_cd_current_seed42_rgda_results.json | lr_rgda | 42 | 2 | 0 |  |  | 0.00 | 47.62 | 68.81 |  | 0.00 | 47.62 | 68.81 | +0.0000 | +0.0000 | +0.0000 |  |
| experiments/lada_recipe_text_rgda_ablation_sanity_20260617/runs/vanilla_no_fd_cd_current_seed42/vanilla_no_fd_cd_current_seed42_zs_results.json | zero_shot | 42 | 2 | 0 |  |  | 81.62 | 64.69 | 63.21 |  | 81.62 | 64.69 | 63.21 | +0.0000 | +0.0000 | +0.0000 |  |
| experiments/lada_recipe_text_rgda_ablation_sanity_20260617/runs/vanilla_no_fd_cd_lada_hybrid_seed42/vanilla_no_fd_cd_lada_hybrid_seed42_ens_results.json | ensemble | 42 | 2 | 0 |  |  | 82.84 | 66.83 | 66.73 |  | 82.84 | 66.83 | 66.73 | +0.0000 | +0.0000 | +0.0000 |  |
| experiments/lada_recipe_text_rgda_ablation_sanity_20260617/runs/vanilla_no_fd_cd_lada_hybrid_seed42/vanilla_no_fd_cd_lada_hybrid_seed42_rgda_results.json | lr_rgda | 42 | 2 | 0 |  |  | 0.00 | 47.62 | 68.81 |  | 0.00 | 47.62 | 68.81 | +0.0000 | +0.0000 | +0.0000 |  |
| experiments/lada_recipe_text_rgda_ablation_sanity_20260617/runs/vanilla_no_fd_cd_lada_hybrid_seed42/vanilla_no_fd_cd_lada_hybrid_seed42_zs_results.json | zero_shot | 42 | 2 | 0 |  |  | 82.84 | 65.96 | 65.41 |  | 82.84 | 65.96 | 65.41 | +0.0000 | +0.0000 | +0.0000 |  |
| experiments/lada_recipe_text_rgda_ablation_sanity_20260617/runs/base_current_seed42/base_current_seed42_zs.json | zero_shot | 42 | 2 | 0 |  |  | 82.88 | 66.28 | 66.81 |  | 82.88 | 66.28 | 66.81 | +0.0000 | +0.0000 | +0.0000 |  |
| experiments/lada_recipe_text_rgda_ablation_sanity_20260617/runs/recipe_current_seed42/recipe_current_seed42_zs.json | zero_shot | 42 | 2 | 0 |  |  | 82.64 | 69.57 | 70.51 |  | 82.64 | 69.57 | 70.51 | +0.0000 | +0.0000 | +0.0000 |  |
| experiments/lada_recipe_text_rgda_ablation_sanity_20260617/runs/base_current_seed42/base_current_seed42_rgda.json | lr_rgda | 42 | 2 | 0 |  |  | 0.00 | 48.52 | 71.43 |  | 0.00 | 48.52 | 71.43 | +0.0000 | +0.0000 | +0.0000 |  |
| experiments/lada_recipe_text_rgda_ablation_sanity_20260617/runs/recipe_current_seed42/recipe_current_seed42_rgda.json | lr_rgda | 42 | 2 | 0 |  |  | 0.00 | 51.30 | 74.21 |  | 0.00 | 51.30 | 74.21 | +0.0000 | +0.0000 | +0.0000 |  |
| experiments/lada_recipe_text_rgda_ablation_sanity_20260617/runs/base_current_seed42/base_current_seed42_ens.json | ensemble | 42 | 2 | 0 |  |  | 82.88 | 67.33 | 68.11 |  | 82.88 | 67.33 | 68.11 | +0.0000 | +0.0000 | +0.0000 |  |
| experiments/lada_recipe_text_rgda_ablation_sanity_20260617/runs/recipe_current_seed42/recipe_current_seed42_ens.json | ensemble | 42 | 2 | 0 |  |  | 82.64 | 70.42 | 71.79 |  | 82.64 | 70.42 | 71.79 | +0.0000 | +0.0000 | +0.0000 |  |

### 11.1 2026-06-17 11:43 四任务 pilot 启动与监控脚本修正

远端 SSH 已恢复。sanity 补跑已经完整结束，result watcher 已把 sanity final summary 写入本文件。

四任务 pilot 没有由 retry monitor 自动启动，原因是 `scripts/monitor_lada_sanity_then_four.sh` 中 `four_already_started()` 的进程匹配过宽：

```text
pgrep -af "OUT_DIR=${FOUR_DIR}|${FOUR_DIR}|run_lada_recipe_text_rgda_ablation" | grep -q "${FOUR_DIR}"
```

该判断可能把 monitor/grep 自身命令或无关 launcher 匹配成 four-task 已启动，导致空目录也被误判为已启动。

已修正为：

```text
1. manifest.txt 存在则认为已启动；
2. runs/ 下已有实际文件/目录则认为已启动；
3. 只在真实 main_incremental.py / run_lada_recipe_text_rgda_ablation.sh / OUT_DIR=... 进程包含 FOUR_DIR 时认为已启动；
4. 排除 pgrep/grep/monitor_lada_sanity_then_four 自身。
```

修正后本地和远端均通过：

```text
bash -n scripts/monitor_lada_sanity_then_four.sh
```

四任务 pilot 已手动启动：

```bash
OUT_DIR=experiments/lada_recipe_text_rgda_ablation_four_20260617
TASKS='aircraft caltech101 dtd eurosat'
SEEDS='42'
GPUS='3'
PARALLEL_JOBS=1
BATCH_SIZE=64
ITERATIONS=800
bash scripts/run_lada_recipe_text_rgda_ablation.sh
```

启动时间：

```text
[2026-06-17 11:38:10] start base_current_seed42 gpu=3
```

当前状态：

```text
base_current_seed42 正在训练；
11:44 复查时日志已刷新到约 222/800 step；
当前尚无 four-task 结果 JSON。
```

11:52 再次复查：

```text
base_current_seed42 仍在正常训练；
日志已推进到约 534/800 step；
four-task 结果文件数量仍为 0，这是第一个 config 尚未完成时的正常状态；
result watcher 仍在后台运行，poll interval = 300s。
```

12:39 再次复查：

```text
base_current_seed42 已完成 Task 1 aircraft、Task 2 caltech101、Task 3 dtd 的训练与评估；
当前进入 Task 4 eurosat 训练，日志约为 37/800 step；
four-task 结果文件数量仍为 0，因为当前实现要等完整 config 完成后才写出 zs/rgda/ens JSON；
GPU3 正在工作，约 19GB 显存占用，utilization 约 99%；
result watcher 仍在后台运行。
```

17:34 再次复查：

```text
four-task pilot 已完成 6/7 个 config；
当前正在最后一个 config: text_only_no_fd_cd_lada_hybrid_seed42；
已产出 18/21 个结果 JSON；
缺少的是最后一个 config 的 zs/rgda/ens 三个结果；
最后一个 config 已完成 Task 1 aircraft 的训练与评估；
当前正在 Task 2 caltech101 训练，日志约为 407/800 step；
GPU3 仍在工作，result watcher 仍在后台运行。
```

17:49 再次复查：

```text
four-task pilot 已全部完成；
7/7 个 config 完成；
21/21 个结果 JSON 齐全；
incremental_summary.md 和 incremental_aggregate.md 已生成；
result watcher 仍在后台运行，但尚未把 four final 自动追加到 chat-history。
```

四任务 aggregate：

| method | n | seeds | K | Transfer | Average | Last |
|---|---:|---|---:|---:|---:|---:|
| ensemble | 7 | 42 | 4 | 52.74 +/- 1.78 | 59.52 +/- 1.61 | 73.17 +/- 2.09 |
| lr_rgda | 7 | 42 | 4 | 0.00 +/- 0.00 | 43.72 +/- 1.97 | 74.52 +/- 3.24 |
| zero_shot | 7 | 42 | 4 | 52.66 +/- 1.81 | 58.52 +/- 1.82 | 71.19 +/- 2.96 |

关键 per-config 结果：

| config | classifier | Transfer | Average | Last |
|---|---|---:|---:|---:|
| base_current_seed42 | zero-shot | 53.45 | 59.93 | 73.23 |
| base_current_seed42 | LR-RGDA | 0.00 | 44.43 | 75.87 |
| base_current_seed42 | ensemble | 53.47 | 60.86 | 74.77 |
| recipe_current_seed42 | zero-shot | 51.92 | 60.72 | 74.57 |
| recipe_current_seed42 | LR-RGDA | 0.00 | 45.89 | 77.63 |
| recipe_current_seed42 | ensemble | 52.13 | 61.46 | 75.63 |
| no_fd_cd_current_seed42 | zero-shot | 51.69 | 58.47 | 71.53 |
| no_fd_cd_current_seed42 | LR-RGDA | 0.00 | 44.48 | 75.90 |
| no_fd_cd_current_seed42 | ensemble | 51.74 | 59.42 | 73.18 |
| no_fd_cd_lada_hybrid_seed42 | zero-shot | 53.81 | 60.20 | 73.64 |
| no_fd_cd_lada_hybrid_seed42 | LR-RGDA | 0.00 | 44.48 | 75.90 |
| no_fd_cd_lada_hybrid_seed42 | ensemble | 53.86 | 60.94 | 74.98 |
| vanilla_no_fd_cd_current_seed42 | zero-shot | 50.04 | 55.99 | 68.90 |
| vanilla_no_fd_cd_current_seed42 | LR-RGDA | 0.00 | 43.57 | 74.35 |
| vanilla_no_fd_cd_current_seed42 | ensemble | 50.22 | 57.23 | 70.72 |
| vanilla_no_fd_cd_lada_hybrid_seed42 | zero-shot | 52.04 | 57.47 | 70.29 |
| vanilla_no_fd_cd_lada_hybrid_seed42 | LR-RGDA | 0.00 | 43.57 | 74.35 |
| vanilla_no_fd_cd_lada_hybrid_seed42 | ensemble | 52.04 | 58.52 | 72.56 |
| text_only_no_fd_cd_lada_hybrid_seed42 | zero-shot | 55.66 | 56.88 | 66.20 |
| text_only_no_fd_cd_lada_hybrid_seed42 | LR-RGDA | 0.00 | 39.60 | 67.63 |
| text_only_no_fd_cd_lada_hybrid_seed42 | ensemble | 55.75 | 58.22 | 70.35 |

四任务初步结论：

```text
recipe_current vs base_current:
  ensemble Average +0.60, Last +0.86
  LR-RGDA Average +1.46, Last +1.76
  zero-shot Average +0.79, Last +1.34
  Transfer 下降约 1.3-1.5

no_fd_cd_lada_hybrid vs no_fd_cd_current:
  zero-shot Average +1.73, Last +2.11
  ensemble Average +1.52, Last +1.80
  LR-RGDA 不变

vanilla_no_fd_cd_lada_hybrid vs vanilla_no_fd_cd_current:
  zero-shot Average +1.48, Last +1.39
  ensemble Average +1.29, Last +1.84
  LR-RGDA 不变

text_only_no_fd_cd_lada_hybrid:
  Transfer 最高，但 Average/Last 不占优；
  zero-shot Last 66.20，ensemble Last 70.35，低于 vanilla hybrid 和 no_fd_cd hybrid；
  LR-RGDA Average/Last 明显下降，说明只调文本侧不足以支撑 LR-RGDA。
```

后续检查重点：

```text
experiments/lada_recipe_text_rgda_ablation_four_20260617/logs/launch.log
experiments/lada_recipe_text_rgda_ablation_four_20260617/logs/base_current_seed42.log
experiments/lada_recipe_text_rgda_ablation_four_20260617/runs/*/*_{zs,rgda,ens}.json
```
