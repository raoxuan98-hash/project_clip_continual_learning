# Part A 分类器离线扫描结果 + waveH 推进

**日期**: 2026-07-24
**会话概况**: Part A 离线分类器扫描（num_centers × rgda_train_iter）全部完成并通过对拍验收；waveH iter600/iter400 在 GPU 上推进至 Task 10；修复了 iter400 的 flickr30k 路径问题。

---

## 1. 关键讨论 / 决策

- **Part A 对拍验收通过**：离线 `rgda_m4_iter200` vs 内联 `inline_m4_iter200`，逐步 Average 差异在 ±0.39 以内，最终步仅 -0.024 → 离线扫描管线与内联 RGDA 一致，结果可信。
- **Part A 最终步 Average 排名（前 5）**：
  1. `ens_m2_iter200_alpha0.05` = 84.012
  2. `ens_m2_iter400_alpha0.05` = 83.951
  3. `ens_m1_iter200_alpha0.05` = 83.909
  4. `ens_m4_iter200_alpha0.05` = 83.907
  5. `ens_m1_iter800_alpha0.05` = 83.893
  - 内联基线 `inline_ens_alpha0.05` = 83.885，`zero_shot` = 80.428，`unfitted_m4` = 76.282。
  - 结论：num_centers 与 rgda_train_iter 在扫描范围内影响很小（顶部 ~0.1-0.15 差距）；M=2/iter=200 边际最优，当前默认 M=4/iter=200 不吃亏。
- **waveH iter400 的 flickr30k 路径修复**：iter400 启动命令中 `flickr30k_hf` 的 retrieval root 误写为 `/mnt/raoxuan/open_datasets/mscoco_2014_5k/test_hf`（不存在）。为保住已跑 33 分钟的进程，创建符号链接 `/mnt/raoxuan/open_datasets/mscoco_2014_5k/test_hf -> /mnt/raoxuan/open_datasets/flickr30k_hf`。**注意**：这引入了一个假的 `mscoco_2014_5k/` 目录（真实 mscoco 目录名为 `mscoco_2014_5k_test_hf`），waveH 结束后可删除。

## 2. 重要发现

- Part A 扫描产物：`experiments/paper_formal/WaveF_offline/classifier_sweep/sweep_summary.json`（10 步 × 43 个方法键），sweep 总耗时约 68 分钟（22:47–23:55，GPU4）。
- iter400 "卡住" 真相：此前判断的 GPU3 卡死实为误报中的重复启动担忧不成立——当前只有 1 个 iter400 父进程（pid 1320104），其余均为其 DataLoader worker。
- 训练运行状态：iter600（pid 1160216，GPU1）与 iter400（pid 1320104）均已进入 Task 10（sun397）阶段。

## 3. 待办事项 / 遗留问题

- [ ] waveH iter600/iter400 完成后读取 `experiments/paper_formal/WaveH_iterations/waveH__lora_nf_iter{600,400}__16shot__seed42.json`，与 iter800 对比分析步数影响
- [ ] 删除临时符号链接 `/mnt/raoxuan/open_datasets/mscoco_2014_5k/`（waveH 结束后）
- [ ] 主表扩展方案待用户拍板（论文线）
- [ ] waveG（CD 方向 A/B）——用户当前不推进

## 4. 相关文件

- `experiments/paper_formal/WaveF_offline/classifier_sweep/sweep_summary.json`：Part A 扫描汇总（服务器端，experiments/ 不入库）
- `scripts/offline_classifier_sweep.py`：Part A 扫描脚本（commit f8b2fcf）
- `experiments/paper_formal/WaveH_iterations/`：waveH 输出目录
