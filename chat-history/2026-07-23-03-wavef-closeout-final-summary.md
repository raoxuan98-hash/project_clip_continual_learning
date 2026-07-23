# Wave F 收尾与 main_v3 战役终审

**日期**: 2026-07-23
**会话概况**: 完成 Wave F 最后一件（frozen 检索行）落盘确认，重跑全部 wave 的验收审计，溯源核对汇总数字，产出战役最终汇总文档（docs/rerun_campaign_main_v3_summary_2026-07-23.md）并完成终审提交。

---

## 1. 关键讨论 / 决策

- **终审复跑 audit 全 PASS**：Wave A 12/12、Wave B 9/9、Wave D（WaveD_hparams）15/15（含 `--require-reference`）、Wave E 3/3、Wave F artifact run 1/1。注意 Wave D 目录名是 `WaveD_hparams` 而非计划早期叫法 WaveD_sensitivity。
- **清理自己留下的巡检噪声**：alpha 扫描 smoke 的 `alpha_scan_smoke.{log,json}` 放在 WaveF_offline 根目录会被 audit 误认为 RUNNING/INCOMPLETE，已移入 `WaveF_offline/smoke/`（audit 只 glob 本层 `*.log` / `*_ens_results.json`，不递归），复审 Wave F 恢复 1/1 PASS 无残留告警。
- **汇总数字一律溯源**：分类/检索 3-seed 表 = `scripts/aggregate_campaign_results.py <wave_dir>` 原样引用；Wave D 单 seed 与各单点 = 对应 `*_ens_results.json` 直读；LADA = WaveC_lada/*.txt 解析后手算 mean±pstdev；v2 数字为引用值并在文档中标注。
- **Wave D 默认点取值**：任务交底给的"60.14/71.84/83.74 附近"与 JSON 不符，按"以 JSON 为准"原则取 waveA lora_nf seed43 单 seed 实际值 **60.14/72.13/83.86**。

## 2. 重要发现

- **全部数字核对一致**：Wave A/B/E 的 aggregate 输出、E6 alpha 扫描（α*=0.1，val 87.80；oracle 0.05→83.88 最优；K×K@α* T/A/L=60.19/72.00/83.49）、frozen 分类（mean 61.1）与检索（mscoco 52.32/33.32、flickr30k 81.50/61.06）、waveF↔waveA 对拍（T/A/L |Δ|≤0.03、逐格 max|diff|=0.007）均与交底值吻合，无一处对不上。
- **LADA 官方复现 3-seed**：seed42 61.6/72.4/82.8、seed43 61.8/72.5/83.1、seed44 62.2/72.5/82.8 → **61.9±0.25 / 72.5±0.05 / 82.9±0.14**，与 commit 922dfe1 记录一致。
- **文档撰写自查纠错**：Wave B 检索表一度把 flickr30k I2T 列错放进 mscoco I2T 列（C1 行），已修正为按 aggregate 列序（mscoco I2T/mscoco T2I/flickr30k I2T/flickr30k T2I）；"C1 flickr30k I2T 79.72/79.63 最低（FD 伤 I2T）"的结论不变。
- **v2 差异核心解释**：LoRA-Null 排序反转（v2 82.32>82.12 → main_v3 82.30<82.50）是 c14255e 文本塔对称 null 初始化修复的直接证据；v2 E7 疑似 CD 静默禁用（SigLIP tokenizer 无 attention_mask），main_v3 已修复，E7 对比须带此前提。

## 3. 待办事项 / 遗留问题

- [ ] full-shot（Wave C LoRA/LADA full-shot）整体搁置中，若论文需要再启动（重启前注意 243dcd4 的 full_shot num_shots=0 bug 已修但未跑）。
- [ ] 论文表格取用：以 docs/rerun_campaign_main_v3_summary_2026-07-23.md 为唯一数字来源；v2 对照数字只可在差异说明语境引用。
- [ ] `experiments/paper_formal/PortSmoke*/`、`WaveF_offline/smoke/` 为移植/扫描 smoke 残留，不入库不影响 audit，可在空间清理时删除。

## 4. 相关文件

- `docs/rerun_campaign_main_v3_summary_2026-07-23.md`：战役最终汇总（commit 79ca0a0）
- `docs/rerun_campaign_main_v3_2026-07-18.md`：执行计划（唯一有效）
- `scripts/aggregate_campaign_results.py`、`scripts/audit_campaign_run.py`：汇总与验收工具
- `experiments/paper_formal/Wave{A_main,B_components,C_lada,D_hparams,E_siglip2,F_offline}/`：全部结果数据（服务器本地）
- 关键 commits：c14255e（四项修复）、4f66332（Wave E seed/GPU）、18f4049（SigLIP2 移植）、aac989b（Wave F artifact run）、cb50e86（E6 alpha 扫描）、4ad6fa8/f69e967（Frozen 行）、79ca0a0（汇总文档）
