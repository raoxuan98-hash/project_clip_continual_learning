# 10-task 重消融 Loop 执行目标

**日期**: 2026-07-06
**起点**: Pre-Wave A 完成 → LoRA + always 胜出
**状态**: 种子方差 + Wave A 运行中

---

## 当前运行中

| 实验 | 目的 |
|------|------|
| lora_always_s43 | 种子方差：seed=43 vs seed=42（Pre-A lora_always=ZS A 70.01） |
| wa_lora_lr5e5 | Wave A: LoRA+always, lr=5e-5 |
| wa_lora_lr1e4 | Wave A: LoRA+always, lr=1e-4 |
| wa_lora_lr3e4 | Wave A: LoRA+always, lr=3e-4 |

公共配置：LoRA+NSP, text=always, cosine_with_warmup, cd=2.0, num_centers=4, rgda_train_iter=200, seed=43

---

## Objective 1: 种子方差 + Wave A 完成

**任务**：等待当前 4 个实验完成，读取结果。

**完成条件**：
- [ ] 4 个 `_zs_results.json` + `_ens_results.json` 全部存在
- [ ] 读取六指标，确定 σ_seed（lora_always seed=42 vs 43）
- [ ] Wave A 选出最优 lr（ZS Average 第一排序键）

**暂停条件**：
- 任何实验崩溃（JSON 缺失或空）→ 停止，排查日志
- 种子标准差 σ_seed > 1.0pp（ZS Average）→ 停止，讨论是否需要多种子
- Wave A 最优 lr 不在 {5e-5, 1e-4, 3e-4} 内部（lr 更极端方向更好）→ 停止，讨论是否扩展 lr 范围

**产出**：
- `chat-history/2026-07-06-waveA-results.md` 含完整六指标 + σ_seed + lr 决策

---

## Objective 2: Wave B（优化器）

**任务**：在 Wave A winner 配置上对比 optimizer ∈ {adamw, rmsprop, sgd(5e-3)}。

**前提**：Objective 1 完成，Wave A winner 已确定。

**启动脚本**：
```bash
bash scripts/launch_waves_b_f.sh B "<use_dora> <lr> <opt> <text_sched>"
# 实际：bash scripts/launch_waves_b_f.sh B "false 1e-4 adamw always"
```

**完成条件**：
- [ ] 3 个结果 JSON 存在
- [ ] 以 ZS Average 排名，adamw 仍为最优或差距 <0.5pp

**暂停条件**：
- SGD ZS Average 反超 AdamW > 0.5pp → 停止，重新评估优化器选择
- RMSprop ZS Average 反超 AdamW > 0.5pp → 停止，讨论是否切换

---

## Objective 3: Wave C（Batch Size）

**任务**：在 winner 配置上对比 bs ∈ {32, 64, 128}。

**前提**：Objective 2 完成（或确认 adamw 为最优）。

**启动脚本**：
```bash
bash scripts/launch_waves_b_f.sh C "<use_dora> <winner_lr> <winner_opt> <text_sched>"
```

**完成条件**：
- [ ] 3 个结果 JSON 存在
- [ ] 选出最优 bs（ZS Average 排序）

**暂停条件**：
- bs=32 ZS Average 反超 bs=128 → 停止，分析原因

---

## Objective 4: Wave D（CD 温度）

**任务**：对比 cd_temperature ∈ {1.0, 2.0, 4.0}。

**暂停条件**：
- temp=1.0 ZS Average 反超 temp=4.0 > 0.5pp → 停止，重新校准
- 三种温度差异 <0.2pp → 停止，取 temp=2.0 作为默认，无需继续对比

---

## Objective 5: Wave E（aux_weight）

**任务**：对比 aux_weight ∈ {0.0, 1.0}。

**暂停条件**：
- 差异 <0.3pp → 取 aux=0.0，不纠结

---

## Objective 6: Wave F（text_schedule 验证）

**任务**：在调优后配置上验证 text_schedule ∈ {freeze_after, low_lr_after, always, never}。

**前提**：Waves B–E 全部完成。

**暂停条件**：
- always 仍为最优 → 确认结论
- low_lr_after 反超 always → 停止，切换公共配置
- freeze_after 或 never 反超 → 停止，排查逻辑

---

## Objective 7: 汇总

**任务**：所有结果汇总到 `chat-history/2026-07-06-10task-reablation-results.md`。

**内容**：
- 完整实验表（23 行 × 六指标）
- 最终推荐配置
- 与历史 82.79 对标（恢复 mc4ft200 后）
- 种子方差报告

---

## 全局暂停条件

- SSH 连接超时 / VPN 断开 → 等待恢复，实验本身不受影响
- 服务器 OOM → 缩小 bs 或减少并行数
- 连续 2 个实验 JSON 缺失 → 停止，排查代码

---

## 服务器信息

| 项目 | 内容 |
|---|---|
| 地址 | raoxuan@10.20.34.30（需 EasyConnect VPN） |
| 路径 | /home/raoxuan/projects/project_clip_continual_learning |
| GPU | 0,1,2,3,4 可用；5 被 vLLM 占用 |
| 日志 | logs/10task_reablation/ |
| 结果 | experiments/10task_reablation/ |
| Python | /home/raoxuan/ENTER/envs/raoxuan/bin/python |
