# 修复后正式重跑战役计划（main_v3 线）

**创建日期**: 2026-07-18
**状态**: 当前唯一有效的重跑执行计划（无人值守战役）
**代码基线**: 服务器 `/home/raoxuan/projects/project_clip_continual_learning` @ main_v3（≥ `7f8b658`，含 `c14255e` 四项修复）
**决策来源**: 2026-07-18 用户拍板——删除 FD、完整矩阵、cd_weight 保持 2.0（Wave D 重扫验证）

---

## 0. 全局配置（冻结）

除被消融项外，所有 run 共用（相对旧候选配置的变化：**fd=0、CD 双向**）：

```text
adapter = LoRA-NF (hard), nsp_eps=0.20, nsp_weight=0.02
lora_rank=4, lora_alpha=4, target=q/k/v/out/fc1/fc2, use_dora=false
optimizer=AdamW, lr=1e-4, weight_decay=3e-5, batch_size=32
iterations=800, scheduler=cosine_with_warmup, warmup=0.1, eta_min=0.0
fd_weight=0.0, cd_weight=2.0, cd_divergence=kl_forward, cd_temperature=4.0, aux_weight=0.0
tune_vision=true, tune_text=true, text_lora_rank=4, text_tuning_schedule=always
text_classifier_mode=lada_hybrid, classifier_feature_transform=test
rgda_rank=32, rgda_alpha=(0.2, 2.0, 0.5), num_centers=4
rgda_train_iter=200, rgda_train_lr=0.01, rgda_fit_source=gmm_sample
alpha=0.05, ensemble_normalize=maxshift, temperature=1.0
reference_dataset=flickr8k, reference_batch_size=32
retrieval: mscoco_2014_5k + flickr30k_hf（canonical 副本，见 §0.1）
任务顺序: aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397
```

### 0.1 检索数据（canonical，唯一来源）

```text
--retrieval_roots mscoco_2014_5k=/mnt/raoxuan/open_datasets/mscoco_2014_5k_test_hf,flickr30k_hf=/mnt/raoxuan/open_datasets/flickr30k_hf
```

冻结基线已锚定（`experiments/retrieval_baseline/frozen_clip_retrieval.json`）：MSCOCO 52.32/33.32，Flickr30K 81.50/61.06。loader 已带硬断言（1000/5000、5000/≥25000），**任何 run 若断言失败 = 数据被污染，立即停止并报告**。

### 0.2 输出规范

- 根目录：`experiments/paper_formal/`（raoxuan 侧，gitignore 内）
- 实验名：`{WAVE}__{config}__{16shot|fs}__seed{42,43,44}[__{override}]`，禁用模糊名；重跑不覆盖，加 `__rerun1`
- 每 run 必须产出：`_zs_results.json`、`_rgda_results.json`、`_ens_results.json`、`_retrieval.json`、总配置 JSON、stdout log
- Python：`/home/raoxuan/ENTER/envs/raoxuan/bin/python`；每 GPU 严格串行；启动间隔 ≥15s（HF 429）

## 1. 每波启动前的代码 review 门（用户硬性要求）

每波实验启动**之前**必须完成并记录：

1. **CLI 接线核查**：启动命令中每个 `--参数` 都在 parser 注册且确实进入代码路径（`grep -rn "args\.<name>" src/ main_incremental.py` 交叉验证；历史教训：parser 注册 ≠ 接线）。
2. **配置核对**：启动脚本显式写出 §0 全部关键参数，不依赖 CLI 默认值；与结果 JSON 的 `arguments` 字段事后复核一致。
3. **输出冲突检查**：目标 stem 不存在；目录、命名符合 §0.2。
4. **修复生效验证**（Wave 0 一次性）：
   - QKV：1-task 后 q/k/v 三个模块的 P 均 ≠ 单位阵（打印 max|P−I|）；
   - LoRA-Null：null_init 后文本塔 P == 单位阵；
   - CD 双向：训练日志中 CD loss 量级正常（无 NaN/爆炸，约为单向同配置同量级）。
5. **smoke 通过**：Wave 0 的 2-task smoke（LoRA-NF + LoRA-Null 各一）指标落盘完整后，才允许启动 Wave A。

## 2. 实验矩阵（全部 16-shot 3 seeds={42,43,44}，除标注外）

### Wave 0：门控（不占用正式 runs）
- smoke ×2（LoRA-NF、LoRA-Null，2 任务，seed 43）

### Wave A：E1-16shot + E3 公平对比（12 runs）
| 实验 | 配置覆盖 | runs |
|---|---|---|
| A1 LoRA-NF | 默认（fd=0, cd=2） | 3 |
| A2 Standard LoRA | `--lora_type lora_vanilla --init_mode lora_vanilla --null_init_mode none` | 3 |
| A3 LoRA-Null | `--null_init_mode history_init_only`（其余同 A1） | 3 |
| A4 GradProj | `--use_gradient_projection`（其余同 A1） | 3 |

### Wave B：E2 组件消融（9 runs）
| 实验 | fd | cd | runs |
|---|---|---|---|
| B-C0 | 0 | 0 | 3 |
| B-C1 | 1 | 0 | 3 |
| B-C3 | 1 | 2 | 3 |

（C2 = A1，直接复用；Ensemble 链从 checkpoint 离线补。）

### Wave C：full-shot + LADA 基线（12 runs）
| 实验 | 配置 | runs |
|---|---|---|
| C-fs-LoRA-NF | `--full_shot`，其余同 A1 | 3 |
| C-fs-LoRA | `--full_shot` + A2 覆盖 | 3 |
| C-lada-16shot | 官方 LADA（自包含复跑，协议=`LADA_official` 原样） | 3 |
| C-lada-fs | 官方 LADA full-shot（先核查官方 num_shots 全量支持；不支持则记 N/A） | 3 |

### Wave D：超参重扫（seed 43 先行，~15 runs + 端点补种 ~6）
| 实验 | 网格（复用 A1 默认点） | runs |
|---|---|---|
| D1 cd_weight | {0.5, 1.0, 4.0} | 3 |
| D2 cd_temperature | {1.0, 2.0, 8.0} | 3 |
| D3 nsp_eps | {0.02, 0.05, 0.10} | 3 |
| D4 nsp_weight | {0, 0.04, 0.08, 0.16} | 4 |
| D5 应用层 | {attention only, ffn only} | 2 |
| D6 端点补种 | 默认点已 3 seeds；对最关键的 2-3 个端点补 42/44 | ~6 |

### Wave E：E7 SigLIP2 重跑（3 runs）
- LoRA-NF + SigLIP2（`google/siglip2-base-patch16-224`），fd=0, cd=2，16-shot，3 seeds；**保存 ZS/RGDA/Ens 全部输出**（v2 只存了 Ens）。
- Native LADA SigLIP2 复用 v2 数字（官方协议，不受我们的 bug 影响）。

### Wave F：离线收尾（无训练）
- E6：alpha ∈ {0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0} 离线扫描（Wave A checkpoint/artifact）；**validation protocol**：用各任务 val split 选全局 alpha，测试集只评一次；oracle 分析单列。
- Frozen CLIP 分类 K×K 行（无需训练）。
- 汇总：从原始 JSON 重算全部表格（mean ± population std）+ 检索 R_avg/R_last + 审计。

## 3. 单 run 验收标准（全部满足才算完成）

- 10/10 任务训练完成；zs/rgda/ens/retrieval 四类 JSON 齐全；`arguments` 与本计划配置一致；
- 无 NaN/OOM/中途恢复污染；检索断言未触发；
- 汇总值可从原始 JSON 重算（跑 `scripts/summarize_continual_metrics.py` 或等价物）。

失败处理：验证失败 → 诊断一次并重跑（`__rerun1`）；再失败 → 标记 blocked 并在最终报告中说明，不得静默跳过。

## 4. 执行顺序与依赖

```text
Wave 0（review+smoke）→ Wave A → Wave B ∥ Wave C（GPU 空闲即插）→ Wave D → Wave E → Wave F
```

- A 是全文主表，最高优先；B/C 可穿插；D 依赖 A 的 seed-43 默认点；E 独立；F 依赖 A 完成。
- 每波启动前执行 §1 的 review 门并记录到 chat-history。

## 5. 完成定义（goal 的验收线）

1. Wave A–E 全部 runs 通过 §3 验收（或明确标记 blocked 及原因）；
2. Wave F 全部离线产物完成：E6 alpha 表、Frozen 行、汇总表（分类 T/A/L 3-seed mean±std + 检索 R_avg/R_last）、审计通过；
3. 汇总文档落盘（`docs/` 或 `chat-history/`），含与 v2 数字的差异说明；
4. 向用户提交最终审查摘要（每个 wave 的 review 记录 + 结果 + 异常清单）。

## 6. 已知边界

- v2（zengyong 侧）数字与本战役不可直接混比：FD 删除、CD 双向、QKV/LoRA-Null 修复、检索 canonical 副本（Flickr30K 偏移 ~0.3-0.4）。
- E7 的 2-6% ZS 异常列是协议相关现象，结果照实记录并附说明。
- LADA full-shot 若官方代码不支持全量 shot，记 N/A 并在报告中说明，不自行改造官方代码。

---

## 7. 修订（2026-07-19）：蒸馏参考集切换 + vanilla 守卫

**决策（用户拍板）**：蒸馏参考从 Flickr8K 切换为 **Flickr30K Karpathy train split 子集**（8000 图、每图 1 条 caption、seed=42 采样）；检索评测**始终使用完整测试集**（MSCOCO 5K / Flickr30K 1K），不再使用任何"剔除重叠"子集。

**原因**：Flickr8K 几乎全是 Flickr30K 子集，与 Karpathy 1K test 有 243/1000 图像重合；改用 train split 后 split 层面不相交，并显式断言零文件名重叠。

**落地**（commit `231deae`）：

- 新参考集：`/mnt/raoxuan/open_datasets/flickr30k_train_sub8k/`（8000 图，构建脚本 `scripts/build_flickr30k_train_reference.py`，零重叠断言通过）。
- `src/utils/reference_loader.py`：`reference_dataset` 支持 `flickr30k_train_sub8k`（默认）/ `flickr8k`（仅历史对照）。
- `configs/base/default.yaml`、`main_incremental.py` 默认值、`scripts/rerun_campaign.sh` 同步切换。
- `src/utils/retrieval_eval.py`：移除 `flickr30k_hf_clean` 分支（不再需要）。
- `src/trainers/lora_nsp_trainer.py`：vanilla LoRA 三处 `hasattr` 守卫（`__init__` 协方差加载、两个 `update_*_covariance_history`），修复 waveA vanilla seed43 崩溃。

**影响**：所有在跑/待跑 wave 的训练数据变化，旧参考的 Wave A 部分运行（无任何完成 run）已全部终止并清除输出，Wave A 以新参考重新启动；Wave0 smoke 以新参考重跑（追加 vanilla 守卫路径验证）。v2 数字与 v2 时代 Flickr8K 数字均不与本战役混比，汇总文档需注明参考集差异。

---

## 8. Wave B–E 静态 review 门记录（2026-07-19，启动前预检）

§1 三项核查（CLI 接线 / 配置核对 / 输出冲突），对 `scripts/rerun_campaign.sh`（`d323470` 版）静态执行：

**Wave B（9 runs）**
- CLI 接线：`--fd_weight/--cd_weight` 覆盖在 BASE_ARGS（fd=0, cd=2）之后，顺序正确生效；C0/C1/C3 × seeds{42,43,44} 与计划矩阵一致；C2（fd0,cd2）复用 Wave A lora_nf，不重复跑。✓
- 配置核对：其余超参全继承 BASE_ARGS（与 E1 同协议）。✓
- 输出冲突：目录 `experiments/paper_formal/WaveB_components/`，与 WaveA_main 不相交；skip 逻辑按 ens JSON 断点续跑。✓

**Wave C（6 LoRA runs + 6 LADA runs）**
- CLI 接线：`--full_shot` flag 存在于 main_incremental.py:480（num_shots 置 None 路径 line 909）；LADA 侧 `waveC_lada` 分支调 `scripts/run_lada_official_campaign.sh`，epochs 与官方 run_TAIL_{16shot,fullshot}.sh 逐条一致，`num_shots=-1` 官方原生全量（scenario_datasets/utils.py:319）。✓
- 配置核对：LADA seed 经 CLI `seed N` 覆盖（官方 config 支持，main.py:38）；root 显式传 `/data1/open_datasets/X-TAIL`（TAIL.yaml 为占位符）。✓
- 输出冲突：LoRA 侧 `WaveC_fullshot/`；LADA 侧结果拷回 `WaveC_lada/waveC_TAIL_*_result.txt`，LADA_official/output 内目录名带 waveC_ 前缀不覆盖历史。✓

**Wave D（15 runs，seed43）**
- CLI 接线：cdw{0.5,1.0,4.0}、cdt{1.0,2.0,8.0}、eps{0.02,0.05,0.10}、nspw{0.00,0.04,0.08,0.16}、layers{attn,ffn} 共 15 个非默认点；默认点（cdw2.0/cdt4.0/eps0.20/nspw0.02/all layers）复用 waveA seed43。✓
- 输出冲突：`WaveD_hparams/`，GPU5 无分配（15=3+3+3+4+2）。✓
- 端点补种（~6 runs）在 seed43 结果出来后按计划另启。✓

**Wave E（3 runs，SigLIP2）**
- CLI 接线：seed=$((41+GPU)) 设计要求**只以 GPU 0/1/2 启动**（seeds 42/43/44）；`SIGLIP2_MODEL_DIR` 默认 `/mnt/raoxuan/models/siglip2-base-patch16-224`（9 文件已下载齐全）。✓
- 已知边界：SigLIP2 为 softmax 口径（计划 §6 已记 2-6% ZS 异常列现象，照实记录）。✓

**统一项**：所有分支均带 `--reference_dataset flickr30k_train_sub8k`（§7 修订后）、`--no-alpha_sensitivity`、检索 canonical roots；waveC_lada 不经 main_incremental，不适用参考集/检索约束（官方代码原样）。
