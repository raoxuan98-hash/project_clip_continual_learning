# Visual Tuning on Top of Text Semantic Anchors

**日期**: 2026-06-18
**目标**: 以“文本侧微调 + 每任务后缓存 semantic anchors / tuned text prototypes”为固定组件，系统评估视觉侧微调是否在此基础上带来稳定、可解释的额外收益。

---

## 1. 固定组件

所有配置固定：

```text
tune_text_encoder=true
text_classifier_mode=lada_hybrid
num_centers=4
artifact_num_centers=1,4
save_step_artifacts=true
skip_inline_eval=true
```

语义：

- 每个任务结束后缓存当前任务 tuned text prototypes；
- seen classes 使用 cached tuned text anchors；
- unseen classes 使用 frozen CLIP text features；
- step artifact 保存严格增量 compact stats；
- 不保存或 replay 旧任务 real features；
- 离线 evaluator 使用 `rgda_stats_by_m`、`gmm_memory` 和 alpha sweep。

---

## 2. 三个训练配置

| config | vision | text | lora | FD/CD | 目的 |
|---|---|---|---|---|---|
| `text_only_anchor` | frozen | tuned | `lora_vanilla` | off | 强 text-side baseline |
| `vanilla_vision_anchor` | tuned | tuned | `lora_vanilla` | off | 测普通视觉侧微调的边际收益/损伤 |
| `nsp_fd_cd_vision_anchor` | tuned | tuned | `lora_nsp` | on | 测 LoRA-NSP + FD/CD 是否能保留视觉收益并控制 Transfer 损伤 |

---

## 3. 离线评估

每个训练配置完成后运行：

```text
zero_shot
rgda_sc
rgda_mc4
rgda_mc4ft200:source=gmm_sample
ensemble alpha = 0,0.05,0.1,0.2,0.5,1.0
```

关键比较：

```text
B - A: vanilla vision tuning 的净收益
C - A: LoRA-NSP + FD/CD vision tuning 的净收益
C - B: NSP/FD/CD 是否减少普通视觉微调的 Transfer 损伤
```

---

## 4. 判定指标

优先看：

- zero-shot Transfer：视觉侧是否破坏 open-vocabulary alignment；
- LR-RGDA Last：视觉侧是否让 seen-task feature 更可分；
- ensemble Average/Last：最终综合收益；
- per-step matrix：是否出现明显旧任务退化。

临时 gate：

```text
视觉侧配置若 Average 和 Last 均提升 >= 1.0 point，且 Transfer 下降 <= 1.0-1.5 point，可认为有明确收益空间。

若 Average/Last 提升 <= 0.5 point 且 Transfer 下降 >= 2.0 point，则当前视觉侧微调不划算。

若 Last 明显提高但 Transfer 明显下降，则作为 tradeoff 记录，后续再考虑 lr=5e-4 或 adaptive alpha。
```

---

## 5. 失败与回撤规则

避免死循环：

- 同一个配置同一种错误最多重试 2 次；第三次不再盲跑，记录为 blocked/failed。
- 如果出现 NaN loss、class-count mismatch、artifact 缺失、sweep JSON 缺失，先停止解释结果，修代码或缩小 smoke。
- 如果远程服务器连续 3 次连接失败，尝试打开 EasyConnect app；仍失败则退出目标模式/标记 blocked。
- 如果某个配置训练日志 60 分钟无 task/step 更新，检查进程和 GPU；确认卡死后终止该配置，不继续无限等待。
- 不因为单个 alpha 的 best 结果好就下结论；best-alpha 只作为 diagnostic upper bound，fixed alpha 结果必须同时报告。

---

## 6. 新增 launcher

```text
scripts/run_visual_tuning_anchor_sweep.sh
```

默认 10-task：

```text
aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397
```

默认使用 GPU：

```text
0 1 5
```

默认每个配置训练后立即跑离线 RGDA/alpha sweep。

---

## 7. 正式启动记录

远程目录：

```text
/home/raoxuan/projects/project_clip_continual_learning
```

输出目录：

```text
experiments/visual_tuning_anchor_sweep_20260618_0112
```

启动命令：

```bash
OUT_DIR=experiments/visual_tuning_anchor_sweep_20260618_0112 \
GPUS="0 1 5" \
PARALLEL_JOBS=3 \
bash scripts/run_visual_tuning_anchor_sweep.sh
```

实际为 `nohup` 后台启动，主 launcher 进程：

```text
PID 12436: bash scripts/run_visual_tuning_anchor_sweep.sh
```

训练进程：

```text
PID 12482: text_only_anchor_seed42 on GPU 0
PID 12483: vanilla_vision_anchor_seed42 on GPU 1
PID 12484: nsp_fd_cd_vision_anchor_seed42 on GPU 5
```

启动时 GPU 0/1/5 空闲；启动后 3 个训练进程均正常占用对应 GPU。

启动后初始 artifact 进度：

```text
total artifact.pt count: 16
text_only_anchor: step_01 ... step_08
vanilla_vision_anchor: step_01 ... step_05
nsp_fd_cd_vision_anchor: step_01 ... step_03
```

这说明三个配置均已开始正常训练并写出 step artifacts。

监控命令：

```bash
ssh raoxuan@10.20.34.30 \
  "cd /home/raoxuan/projects/project_clip_continual_learning && \
   tail -n 80 experiments/visual_tuning_anchor_sweep_20260618_0112/logs/launch.log && \
   find experiments/visual_tuning_anchor_sweep_20260618_0112/async -path '*/artifact.pt' | wc -l && \
   find experiments/visual_tuning_anchor_sweep_20260618_0112/sweep -name 'rgda_sweep_summary.md' -print"
```

进程检查：

```bash
ssh raoxuan@10.20.34.30 \
  "ps -ef | grep -E 'run_visual_tuning_anchor_sweep|main_incremental|evaluate_incremental_rgda_sweep' | grep -v grep"
```

完成判据：

```text
3 个 rgda_sweep_summary.md 均存在；
每个 config 有 10 个 artifact.pt；
每个 config 生成 zero_shot / rgda_sc / rgda_mc4 / rgda_mc4ft200 / ensemble alpha sweep JSON。
```

---

## 8. 汇总工具

新增：

```text
scripts/summarize_visual_tuning_anchor_sweep.py
```

用途：

- 读取三个配置的 `sweep/*/rgda_sweep_summary.csv`；
- 生成跨配置 wide table；
- 生成同方法 delta table：
  - `vanilla_minus_text_only`
  - `nsp_minus_text_only`
  - `nsp_minus_vanilla`
- 生成每个配置按 Average 选择的 diagnostic best ensemble。

输出目录：

```text
experiments/visual_tuning_anchor_sweep_20260618_0112/reports/
```

当前验证：

```bash
python -m py_compile scripts/summarize_visual_tuning_anchor_sweep.py
python scripts/summarize_visual_tuning_anchor_sweep.py \
  experiments/visual_tuning_anchor_sweep_20260618_0112 \
  --allow_incomplete
```

当前状态符合预期：三个 sweep summary 都尚未生成，因为训练仍在进行。

最新进度检查：

```text
nsp_fd_cd_vision_anchor_seed42: 4 / 10 artifacts
text_only_anchor_seed42: 9 / 10 artifacts
vanilla_vision_anchor_seed42: 7 / 10 artifacts
sweep summaries: 0 / 3
result JSON count: 0
```

三份 train log 仍在更新；未观察到 Traceback、NaN 或 class-count mismatch。
