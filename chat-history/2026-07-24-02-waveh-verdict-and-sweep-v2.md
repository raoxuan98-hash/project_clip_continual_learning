# waveH 验收结论 + 分类器扫描 v2（含离线 LADA）

**日期**: 2026-07-24
**会话概况**: waveH（主干步数 600/400 vs 800）验收完成，结论为步数无显著影响；分类器离线扫描升级为 v2（LADA 离线重建 + 全任务评估 + 精确 T/A/L），正在 GPU4 重跑。

---

## 1. 关键讨论 / 决策

- **waveH 结论（seed42）**：主干训练步数 800→400，Ens Transfer 60.13→60.66（+0.5）、Last 83.91→83.43（-0.5）、Average 72.12→72.02、(T+L)/2 72.02→72.04 持平；检索 R@1 不降反微升（flickr i2t 80.80→81.50）。**400–800 步区间对 ID/OOD 平衡无实质影响**，800 步无过拟合证据，减步可省一半训练开销。
- **(T+L)/2 指标下分类器选择**：纯监督分类器（RGDA/LADA）Transfer=0（输出仅覆盖已见类），(T+L)/2 腰斩出局；必须选带 ZS 集成的分类器。集成家族内 Transfer 各变体基本相同，排名由 Last 决定；M=2/iter=200 与 M=4/iter=200 差距 ~0.1（噪声内），保持默认 M=4/iter=200。
- **扫描脚本 v2 三处修正**（commit e5a00af）：
  1. LADA 无条件从再抽取训练特征离线重建（v1 依赖 artifact 中的 lada state，而 WaveF artifact 的 lada=None）；
  2. 评估从"仅已见任务"改为全 10 任务——v1 的 Transfer 全为 0 是矩阵未填满的伪影，v2 的 T/A/L 为精确值；
  3. 输出新增 (T+L)/2 列；两个评估循环合并，特征只抽一次。
- **iter400 flickr 路径事故**：启动命令把 flickr30k_hf 的 root 误写为不存在的 `mscoco_2014_5k/test_hf`；通过临时符号链接抢救成功（iter400 仅 step 0–3 缺 flickr 检索，最终步完整）。事后已删除链接与新建目录。

## 2. 重要发现

- waveH 全套结果（含 LADA 列）在 `experiments/paper_formal/WaveH_iterations/waveH__lora_nf_iter{400,600}__16shot__seed42*.json`；iter800 对照为 `WaveA_main/waveA__lora_nf__16shot__seed42.json`。
- 离线扫描 step 1 验证：sanity 对拍 0.33%（≤0.5% 达标）；离线 zero_shot Transfer 59.97 vs 真实运行 59.78，吻合；离线 LADA 正常出数。
- 服务器上另有 `E0PA__lora_nf` seeds 42/43/44 进程（GPU1/3），非本会话启动，疑似另一会话任务。

## 3. 待办事项 / 遗留问题

- [ ] 扫描 v2（`classifier_sweep_v2/`）完成后出全方法 T/A/L 排名表（含 lada / lada_zs）
- [ ] 主表扩展方案待用户拍板（论文线）
- [ ] waveG（CD 方向 A/B）——用户当前不推进
- [ ] 本地 Mac 仓库 `.git` 目录为空（7月12日起损坏），提交只能在服务器侧进行；需找时间重建本地 git 元数据

## 4. 相关文件

- `scripts/offline_classifier_sweep.py`：v2 扫描脚本（commit e5a00af）
- `experiments/paper_formal/WaveF_offline/classifier_sweep_v2/`：v2 输出目录（服务器）
- `experiments/paper_formal/WaveH_iterations/`：waveH 结果
- `chat-history/2026-07-24-01-partA-sweep-results-waveh-progress.md`：Part A 结果记录
