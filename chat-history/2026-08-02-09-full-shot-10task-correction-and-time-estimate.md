# full-shot 进度修正：实际为 10 task 及更新时间预期

**日期**: 2026-08-02
**会话概况**: 在监控 W4 full-shot 过程中发现之前对 task 数量的误判，并更新了 `docs/average_push_log.md` 的时间预期。

---

## 1. 关键发现 / 决策

- **full-shot 并非 5 task，而是 10 task**（与 dataset_sequence 的 10 个数据集一一对应）。
- 之前误以为 seed42/43 已完成最终 task，实际上 seed42 仅到 Task 6/10，seed43 刚进入 Task 6/10（06:01 CST）。
- 因此完成时间大幅延长：
  - LoRA-NF seed42/43 预计还需 1–1.5 h；
  - 随后 LoRA-only seed42/43 各约 2.5 h；
  - B0 约 0.5 h；
  - 全程可能持续到 08:00–09:00 CST。

## 2. 已执行动作

- 更新 `docs/average_push_log.md` 中 W4 进度描述，明确“full-shot 共 10 task”及新的时间预期。
- 将文档改动 push 到 `average-push` 分支（commit `e321665`）。
- 启动 10 分钟间隔的后台状态轮询，避免频繁 SSH 检查。

## 3. 当前状态（截至 2026-08-02 06:07 CST）

| 项目 | 状态 |
|------|------|
| fs_loranf ens | 0/2 |
| fs_loraonly ens | 0/2 |
| b0 ens | 0/1 |
| seed42 最新 task | Task 6/10 训练（05:57 开始） |
| seed43 最新 task | Task 6/10 训练（06:01 开始） |

## 4. 待办事项

- [ ] 等待 full-shot LoRA-NF seed42/43 完成（10 task）。
- [ ] 等待 supervisor 自动调度 full-shot LoRA-only seed42/43。
- [ ] 等待 B0 零样本 + retrieval 评测。
- [ ] 最终聚合并更新 `docs/average_push_log.md` 与 chat-history。

## 5. 相关文件

- `docs/average_push_log.md`: 进度与时间预期。
- `scripts/pa_supervisor.sh`: 自动推进 W4/W5。
- `scripts/monitor_fullshot_b0.sh`: 完成检测与自动聚合。
