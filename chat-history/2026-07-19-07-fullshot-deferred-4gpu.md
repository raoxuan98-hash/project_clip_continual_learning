# full-shot 搁置 + 4 GPU 上限 + full_shot bug 修复（14:35）

**日期**: 2026-07-19
**会话概况**: 用户决策 full-shot 实验全部搁置、GPU 上限 4 张；修复 full_shot TypeError 并重排启动器。

## 1. 决策
- full-shot 搁置：Wave C LoRA full-shot 6 runs 与 LADA full-shot 3 runs 不再执行（计划 §10）。
- GPU 上限 4 张（0-3）；机器上有其他用户进程（GPU 1/2 有少量外部占用）。

## 2. full_shot bug（13:43 Wave C 6 runs 秒崩的原因）
- main_incremental.py:909  → loader 期望  表全量 → TypeError。
- 修复为 ；2 任务 full-shot smoke 验证训练/后处理通过（用户随后决定整体搁置 full-shot）。

## 3. 当前编排（14:35 启动）
- LADA 官方 16-shot ×3 seeds：GPU 0/1/2（）。
- Wave D 超参扫描：GPU 3 nspw×4 先行；LADA 完成后 GPU 0-2 接 cdw×3+layers×2 / cdt×3 / eps×3。
- 巡检 cron 更新为 2585848c（30 分钟）。

## 4. 待办
- [ ] LADA 16-shot 3 result.txt 验收（EVALUATION METRICS 段）
- [ ] Wave D 15 runs 验收
- [ ] Wave E / Wave F（含 E6 补充 artifact run + Frozen 行 + 汇总）
