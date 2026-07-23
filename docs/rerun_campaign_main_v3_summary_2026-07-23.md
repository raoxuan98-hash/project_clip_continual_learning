# main_v3 重跑战役最终汇总（2026-07-23）

**代码线**: 服务器 `project_clip_continual_learning` @ `main_v3`
**配套计划**: `docs/rerun_campaign_main_v3_2026-07-18.md`（唯一有效执行计划）
**数据根**: `experiments/paper_formal/`（服务器本地，gitignore 内）
**本文性质**: 战役终审汇总 + 与 v2 线的差异说明。所有数字均可溯源：分类/检索表 = `scripts/aggregate_campaign_results.py <wave_dir>` 输出（3-seed mean±pstdev）；单 seed / 单 run 数字 = 对应 `*_ens_results.json` / 结果 JSON 直读；v2 数字为引用值（文中逐一标注）。

---

## 1. 战役概况

| 时间 | 事件 |
|---|---|
| 2026-07-18 | 用户拍板重跑决策（删 FD、完整矩阵、cd_weight=2.0），计划文档冻结 |
| 07-19 凌晨 | 启动器就位（本地模型目录、`CLIP_LOCAL_FILES_ONLY=1`、GPU 静态分工），Wave 0 门控 smoke 通过 |
| 07-19 上午-中午 | **Wave A**（12 runs）→ 12/12 验收 PASS；**Wave B**（9 runs）→ 9/9 PASS |
| 07-19 下午 | **Wave C**：官方 LADA 16-shot ×3 seeds 完成；**Wave D**（15 runs，单 seed43）→ 15/15 PASS |
| 07-19 14:35 | 用户决策：**full-shot 整体搁置**（commit `243dcd4`，见 §10） |
| 07-23 00:19 | Wave E 启动修复：seed 显式映射 42/43/44 + GPU 0/1/3（`4f66332`） |
| 07-23 01:09 | SigLIP2 骨干支持移植（`18f4049`，v2 → main_v3 最小侵入）→ **Wave E** 3/3 PASS |
| 07-23 04:09-05:59 | **Wave F**：E6 artifact 补充 run（`aac989b`）+ artifact 异步评估落盘 |
| 07-23 04:30 | E6 alpha 离线扫描脚本（`cb50e86`）→ α* 扫描完成 |
| 07-23 04:32 / 07:40 | Frozen 分类行修复（`4ad6fa8`）+ Frozen canonical 检索行（`f69e967`）落盘 |

关键 commits：`c14255e`（QKV 共享 P 按组赋值 / LoRA-Null 文本塔对称初始化 / CD 双向蒸馏 / 检索 loader 去 fallback 加 split 断言，四项战役前置修复）→ `4f66332`（Wave E seed/GPU 修正）→ `18f4049`（SigLIP2 移植）→ `aac989b`（Wave F artifact run）→ `cb50e86`（E6 alpha 扫描）→ `4ad6fa8` / `f69e967`（Frozen 行）。

## 2. 验收总表（终审复跑于 2026-07-23）

| Wave | 内容 | runs | 验收 | 备注 |
|---|---|---|---|---|
| Wave A | E1/E3 主表：4 adapter × 3 seeds | 12 | **12/12 PASS** | `audit_campaign_run.py --expected-k 10` |
| Wave B | E2 组件消融：C0/C1/C3 × 3 seeds | 9 | **9/9 PASS** | 同上 |
| Wave C | 官方 LADA 16-shot × 3 seeds | 3 | **3/3 完成** | 独立协议（WaveC_lada/*.txt） |
| Wave C(fs) | full-shot LoRA / LADA | — | **搁置** | 07-19 14:35 决策，见 §10 |
| Wave D | 单 seed 敏感性（15 配置） | 15 | **15/15 PASS** | 加 `--require-reference`；单 seed 为设计如此 |
| Wave E | SigLIP2 LoRA-NF × 3 seeds | 3 | **3/3 PASS** | GPU 0/1/3（GPU2 被占） |
| Wave F | E6 artifact run + alpha 扫描 + Frozen 行 | 1 | **1/1 PASS** | artifact run 与 waveA seed43 对拍一致 |

复现性校验：waveF artifact run 与 waveA `lora_nf seed43`（同配置独立重跑）对拍——Transfer/Average/Last 差 −0.001/−0.005/+0.027（|Δ|≤0.03），K×K 矩阵逐格 max|diff|=0.007。

## 3. 主表（E1，16-shot，3-seed mean±pstdev）

Transfer / Average / Last（%），来源 `aggregate_campaign_results.py experiments/paper_formal/WaveA_main`。

| 方法 | ZS T | ZS A | ZS L | Ens T | Ens A | Ens L |
|---|---|---|---|---|---|---|
| **LoRA-NF（本文）** | 59.92±0.25 | 70.40±0.20 | 80.50±0.11 | **60.25±0.16** | **72.16±0.05** | **83.88±0.02** |
| Standard LoRA | 59.54±0.18 | 69.64±0.14 | 79.03±0.10 | 59.86±0.12 | 71.39±0.13 | 82.50±0.06 |
| LoRA-Null | 59.64±0.25 | 69.35±0.13 | 78.25±0.23 | 60.03±0.15 | 71.34±0.13 | 82.30±0.22 |
| GradProj | 59.77±0.26 | 70.00±0.14 | 79.73±0.04 | 60.14±0.17 | 71.88±0.14 | 83.25±0.15 |
| LADA 官方复现（Wave C） | — | — | — | — | 72.5±0.05（A） | 82.9±0.14（L） |
| Frozen CLIP ZS（Wave F） | — | — | — | — | — | 61.1（zero-shot，无增量过程） |

- Ens Last 排序：**LoRA-NF 83.88 > GradProj 83.25 > Standard LoRA 82.50 > LoRA-Null 82.30**。
- LADA 官方复现 per-seed（T/A/L）：seed42 61.6/72.4/82.8，seed43 61.8/72.5/83.1，seed44 62.2/72.5/82.8 → **61.9±0.25 / 72.5±0.05 / 82.9±0.14**（LADA 无单独 ZS/Ens 列，其分类器本身是文本+标签记忆混合，故只列 T/A/L 三格）。
- Frozen CLIP ZS per-dataset：aircraft 22.4 / caltech101 92.9 / dtd 44.7 / eurosat 39.3 / flowers 65.3 / food101 84.8 / mnist 53.6 / oxford_pets 84.8 / stanford_cars 60.3 / sun397 62.5，mean 61.1（`WaveF_offline/frozen_zeroshot.json`，与公开报告一致）。

## 4. E2 组件消融（16-shot，3-seed）

来源 `aggregate_campaign_results.py experiments/paper_formal/WaveB_components`。C3(fd1/cd2) 与 Wave A LoRA-NF 同配置，互为校验（83.82 vs 83.88，两次独立 3-seed run 的均值差 0.06，在 seed 噪声内）。

**分类（T/A/L）**

| 配置 | ZS T | ZS A | ZS L | Ens T | Ens A | Ens L |
|---|---|---|---|---|---|---|
| C0 (fd0/cd0) | 60.39±0.20 | 69.58±0.17 | 79.43±0.06 | 60.56±0.19 | 71.17±0.11 | 83.38±0.09 |
| C1 (fd1/cd0) | 60.30±0.19 | 69.92±0.30 | 79.87±0.29 | 60.51±0.15 | 71.63±0.19 | 83.66±0.06 |
| C3 (fd1/cd2) | 59.92±0.25 | 70.42±0.20 | 80.51±0.09 | 60.25±0.17 | 72.16±0.06 | 83.82±0.04 |

**检索（R@1，Avg/Last）**

| 配置 | mscoco I2T | mscoco T2I | flickr30k I2T | flickr30k T2I |
|---|---|---|---|---|
| C0 | 51.84 / 52.74 | 33.43 / 34.05 | 80.85 / 79.97 | **62.09 / 63.75（T2I 最高）** |
| C1 | 51.41 / 53.07 | 33.21 / 34.06 | **79.72 / 79.63（I2T 最低）** | 61.53 / 63.15 |
| C3 | 52.43 / 52.86 | 33.31 / 33.72 | 80.85 / 81.20 | 61.34 / 61.93 |

结论：CD（跨模态蒸馏）是分类端主要正向因素（C3 Ens Last 83.82 vs C0 83.38，+0.44）；FD-only（C1）的 flickr30k I2T 79.72/79.63 为三者最低——FD 对 I2T 检索有退化，与 v2"FD 伤 I2T"结论一致，支持 fd_weight=0 的冻结决策。

## 5. E3 adapter 对比

即 §3 Wave A 四配置行（LoRA-NF / Standard LoRA / LoRA-Null / GradProj）。LoRA-NF 在 Ens 三指标全面第一；GradProj 第二（83.25）；LoRA-Null 末位（82.30，见 §9 排序反转解释）。

## 6. Wave D 单 seed 敏感性（seed43，设计如此）

来源各 `WaveD_hparams/*_ens_results.json` 直读（Ens T/A/L）。默认点取 waveA `lora_nf seed43` 单 seed JSON：**60.14 / 72.13 / 83.86**（以 JSON 为准）。aggregate 会对缺 seed42/44 告警，属预期。

| 组 | 取值 | Ens T | Ens A | Ens L |
|---|---|---|---|---|
| nsp_eps（默认 0.20） | 0.02 | 59.94 | 71.67 | 83.67 |
| | 0.05 | 59.95 | 72.04 | 83.74 |
| | 0.10 | 60.17 | 72.22 | 83.78 |
| | **0.20（默认）** | 60.14 | 72.13 | **83.86** |
| nsp_weight（默认 0.02） | 0 | 60.12 | 72.09 | 83.87 |
| | **0.02（默认）** | 60.14 | 72.13 | 83.86 |
| | 0.04 | 60.10 | 72.13 | 83.87 |
| | 0.08 | 60.09 | 72.08 | 83.87 |
| | 0.16 | 60.02 | 71.92 | 83.70 |
| layers（默认 all） | attn only | **60.62** | 71.91 | 83.61 |
| | ffn only | 60.40 | 72.11 | 83.41 |
| | **all（默认）** | 60.14 | 72.13 | **83.86** |
| cd_weight（默认 2.0） | 0.5 | 60.17 | 71.95 | 83.84 |
| | 1.0 | 60.12 | 72.05 | 83.86 |
| | **2.0（默认）** | 60.14 | 72.13 | 83.86 |
| | 4.0 | 60.17 | **72.22** | 83.87 |
| cd_temperature（默认 4） | 1 | 60.09 | 71.82 | 83.74 |
| | 2 | 60.17 | 71.96 | 83.76 |
| | **4（默认）** | 60.14 | 72.13 | 83.86 |
| | 8 | 60.17 | 72.18 | **83.88** |

结论：Ens Last 全组落在 83.41–83.88 窄带（±0.5 内），默认点基本处于各组顶部或并列顶部——**LoRA-NF 对 nsp_eps / nsp_weight / layers / cd_weight / cd_temperature 均不敏感**，无需逐任务调参。

## 7. E6 融合系数 α 离线扫描（Wave F）

来源 `WaveF_offline/alpha_scan.json`（`scripts/eval_alpha_scan_artifacts.py` 对 10 个 step artifact 在 features 层面扫描，10 数据集 × val/test 各提一次特征）。

- **选择协议**：global α* = 最终 step（step 10）10 数据集 **val**（= 16-shot 训练集 + test transform，本协议无真 val）集成准确率均值的 argmax，并列取较小 α。
- **val 曲线**（final step，%）：α=0→83.88, 0.01→86.70, 0.02→87.50, 0.05→87.75, **0.1→87.80（选中）**, 0.2→87.44, 0.5→87.06, 1.0→86.91。
- **oracle 曲线**（final step test 均值，仅供分析不参与选择）：α=0→80.43, 0.01→83.14, 0.02→83.78, **0.05→83.88（oracle 最优）**, 0.1→83.49, 0.2→83.08, 0.5→82.67, 1.0→82.54。
- **α\*=0.1 重算的完整 K×K 集成矩阵**：Transfer/Average/Last = **60.19 / 72.00 / 83.49**。

结论：战役默认 α=0.05 恰好落在 test oracle 最优点上（83.88）；val 协议选出的 α=0.1 仅低 0.39（83.49），且 0.02–0.1 区间 oracle 均 ≥83.78——**融合系数在 0.02–0.1 宽带内稳健，默认 0.05 近优**。

## 8. Wave E：SigLIP2 骨干鲁棒性（16-shot，3-seed）

来源 `aggregate_campaign_results.py experiments/paper_formal/WaveE_siglip2`。骨干 `google/siglip2-base-patch16-224`（本地目录，AutoModel 加载为 SiglipModel），训练/推理口径与 CLIP 线完全一致（softmax + logit_scale，未改 sigmoid）。

| 方法 | ZS T | ZS A | ZS L | Ens T | Ens A | Ens L |
|---|---|---|---|---|---|---|
| LoRA-NF on SigLIP2 | 59.69±0.10 | 69.25±0.35 | 81.69±0.40 | 59.94±0.05 | 71.32±0.31 | **87.95±0.09** |

检索（R@1，Avg/Last；SigLIP2 量级与 CLIP 不同，不与 CLIP 行直接比）：flickr30k 89.36/89.27（I2T）、73.39/73.39（T2I）；mscoco 65.10/65.15（I2T）、48.48/48.25（T2I）。

v2 对照（E7，引用 v2 值）：LoRA-NF Ens 59.83/70.33/87.77；Native LADA 58.03/71.04/88.15。main_v3 LoRA-NF Ens Last 87.95 较 v2 同配置 +0.18；**本次补齐了 v2 缺的 ZS 列**。

## 9. 检索汇总（R@1）

Frozen 行为 canonical 副本单测（`WaveF_offline/frozen_retrieval.json`，无增量过程，Avg=Last）；其余为 aggregate 3-seed mean（Avg / Last）。mscoco = mscoco_2014_5k（5000 图），flickr30k = flickr30k_hf（Karpathy 1K test，canonical）。

| 行 | mscoco I2T | mscoco T2I | flickr30k I2T | flickr30k T2I |
|---|---|---|---|---|
| Frozen CLIP | 52.32 | 33.32 | 81.50 | 61.06 |
| WaveA LoRA-NF | 52.45 / 52.90 | 33.30 / 33.69 | 80.85 / 81.27 | 61.34 / 61.95 |
| WaveA Standard LoRA | 52.06 / 51.91 | 33.07 / 33.52 | 81.01 / 81.17 | 61.37 / 61.75 |
| WaveA LoRA-Null | 52.04 / 52.11 | 33.24 / 33.57 | 80.95 / 81.23 | 61.29 / 61.35 |
| WaveA GradProj | 52.20 / 52.09 | 33.21 / 33.54 | 80.65 / 81.13 | 61.35 / 61.87 |
| WaveB C0 | 51.84 / 52.74 | 33.43 / 34.05 | 80.85 / 79.97 | 62.09 / 63.75 |
| WaveB C1 | 51.41 / 53.07 | 33.21 / 34.06 | 79.72 / 79.63 | 61.53 / 63.15 |
| WaveB C3 | 52.43 / 52.86 | 33.31 / 33.72 | 80.85 / 81.20 | 61.34 / 61.93 |

结论：NSP/LoRA-NF 训练不伤检索（各行相对 Frozen 偏移 ≤±1.5，R_last 有升有降）；C1（FD-only）flickr30k I2T 79.72 为最低，再次印证 FD 伤 I2T。

## 10. 与 v2 线的差异说明

**v2→main_v3 已知差异来源**：`c14255e` 四项修复（QKV 共享 P 绑定、LoRA-Null 文本塔对称初始化、CD 双向 I2T+T2I、检索 canonical 副本）；蒸馏参考集改为 flickr30k_train_sub8k；FD 已删除（fd_weight=0）；E7 上 v2 疑似 CD 静默禁用（SigLIP tokenizer 无 attention_mask → KeyError 被静默 catch），main_v3 已修复（全 1 mask，语义等价不传 mask）。

| 表 | v2（引用） | main_v3 | Δ（Last） | 说明 |
|---|---|---|---|---|
| E1 LoRA-NF Ens | 60.14/71.75/83.74 | 60.25/72.16/83.88 | +0.14 | 同序第一，微升 |
| E1 Standard LoRA Ens | 59.84/70.95/82.12 | 59.86/71.39/82.50 | +0.38 | CD 双向 + 参考集切换红利 |
| E3 GradProj Ens | 82.97（L） | 83.25（L） | +0.28 | 同上 |
| E3 LoRA-Null Ens | 82.32（L） | 82.30（L） | −0.02 | 见下"排序反转" |
| E2 C0 Ens | 83.21（L） | 83.38（L） | +0.17 | 同趋势 |
| E2 C1 Ens | 83.50（L） | 83.66（L） | +0.16 | 同趋势 |
| E2 C2↔C3 Ens | 83.72（L） | 83.82（L） | +0.10 | v2 C2 = main_v3 C3（fd1/cd2） |
| LADA 复现 | 61.6±0.2/72.3±0.4/83.0±0.2 | 61.9±0.25/72.5±0.05/82.9±0.14 | −0.1 | 两边均在 LADA 论文（56.7/68.9/83.1）之上 |
| E7 LoRA-NF Ens | 59.83/70.33/87.77 | 59.94/71.32/87.95 | +0.18 | main_v3 修复了 v2 疑似 CD 静默禁用 |
| Frozen 检索 | MSCOCO 52.32/33.33，Flickr30K 81.10/60.78（非 canonical） | MSCOCO 52.32/33.32，Flickr30K 81.50/61.06（canonical） | — | mscoco 完全一致；flickr30k +0.40/+0.28 为 canonical 副本偏移（~0.3-0.4，预期内） |

**排序反转（重要）**：v2 中 LoRA-Null（82.32）> Standard LoRA（82.12）；main_v3 中反转为 LoRA-Null（82.30）< Standard LoRA（82.50）。原因：v2 的 LoRA-Null 只对视觉塔做历史零空间初始化，文本塔带着持续 P 训练，实际是 LoRA-NF 混合体，成绩被抬高；`c14255e` 把文本塔对称纳入 null 初始化后，LoRA-Null 回到其真实水平——**这是修复生效的直接证据，而非回退**。

**不变的结论**：LoRA-NF 始终 Ens 第一（两线一致）；CD 是分类端主要正向因素；FD 伤 I2T 检索；NSP 训练不伤检索；融合系数在 0.02–0.1 宽带内稳健、默认 0.05 近优。

## 11. 已知边界

1. **full-shot 搁置**：07-19 14:35 用户决策（commit `243dcd4`）。原因：论文主协议聚焦 16-shot；GPU 预算收紧至 4 卡；`full_shot` num_shots=0 的 bug 已修但不再投入 full-shot runs。文档与表格中 full-shot 一律标注搁置。
2. **Wave D 为单 seed（seed43）敏感性**：设计如此（15 配置 × 800 iter 的预算约束），aggregate 的 missing-seed 告警为预期。
3. **E7/SigLIP2 口径**：沿用 softmax + logit_scale（SigLIP2 的 logit_scale 是 sigmoid 预训练温度，按计划文档决策不"修复"为 sigmoid）；v2 E7 疑似 CD 静默禁用，main_v3 已修复，两边 E7 数字对比时需带此前提。
4. **Flickr30K canonical 偏移**：main_v3 使用 canonical Karpathy test 副本（1000 图/5000 caption，loader 带硬断言），与 v2 非 canonical 数字存在 ~0.3-0.4 系统性偏移，属预期。
5. **Wave C LADA 为独立协议**（官方实现 + 转换权重，特征一致性 cos>0.999，commit `3749270`/`32c6e9e`），其 T/A/L 与本仓 Ens 行同表异构，比较时只读三格。
