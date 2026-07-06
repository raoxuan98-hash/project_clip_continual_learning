# QKV 投影方案消融实验启动记录

**日期**: 2026-07-05
**目标**: 比较三种 QKV 投影处理方案在相同超参下的性能差距

---

## 1. 背景

在 LoRA-NSP 实现中，同一 transformer 层的 `q_proj/k_proj/v_proj` 输入空间相同，但旧实现会为三者各自收集协方差、各自构建投影矩阵 P，存在冗余。此前已完成两种改造：

- **方案 A（当前默认）**: 独立 q/k/v Linear + 独立 LoRA A/B，但同层 Q/K/V 共享同一个 `FixedProjection`（P 矩阵），协方差提取也只算一次。
- **方案 B（`--fused_qkv`）**: Timm 式 fused QKV，把 `q_proj/k_proj/v_proj` 合并成单个 `qkv_proj`。

本组消融引入第三方对照：
- **旧默认（独立 P）**: 通过环境变量 `SHARE_QKV=0` 临时禁用共享 P，恢复三个独立的 P 矩阵。

---

## 2. 实验配置

三组使用完全相同的超参：

```bash
--root /data1/open_datasets/X-TAIL
--dataset_sequence aircraft caltech101 dtd eurosat flowers oxford_pets
--num_shots 16
--batch_size 128
--lora_type lora_nsp
--use_dora false
--train_budget_mode uniform
--cd_weight 2.0
--aux_weight 0.0
--fd_weight 1.0
--cd_divergence kl_forward
--cd_temperature 4.0
--optimizer adamw
--lr 1e-4
--weight_decay 3e-5
--iterations 800
--scheduler cosine_with_warmup
--projection_param_mode full
--null_init_mode none
--seed 42
--lora_rank 4
--lora_target_modules q_proj,k_proj,v_proj,out_proj,fc1,fc2
```

---

## 3. 实验启动

| 方案 | 实验名 | GPU | 启动命令/备注 |
|---|---|---|---|
| 旧默认：独立 P | `independent_p` | GPU 0 | `SHARE_QKV=0 python main_incremental.py ... --experiment_name independent_p` |
| 当前默认：共享 P | `layer_all_rank4` | — | 复用 layer-rank 消融中已完成的 baseline（方案 A） |
| 方案 B：fused QKV | `fused_qkv` | GPU 2 | `python main_incremental.py ... --fused_qkv --experiment_name fused_qkv` |

输出目录：`experiments/qkv_ablation/`
日志目录：`experiments/qkv_ablation/logs/`

---

## 4. 代码改动

为支持独立 P 对照，临时修改了 `src/models/lora_sgp.py`：

- 在 `LoRACLIPVisionTransformer` 和 `LoRACLIPTextTransformer` 的 `__init__` 中读取环境变量 `SHARE_QKV`。
- `SHARE_QKV=1`（默认）时走方案 A（共享 P）。
- `SHARE_QKV=0` 时，q/k/v 各自创建独立的 `FixedProjection`。

已备份原文件：`src/models/lora_sgp.py.bak.<timestamp>`

---

## 5. 当前状态（2026-07-05 14:15 左右）

- `independent_p`: 运行中（GPU 0）
- `fused_qkv`: 运行中（GPU 2）
- 预计每组实验约 2 小时完成。

---

## 6. 下一步

- 等 `independent_p` 和 `fused_qkv` 都完成后，汇总三方结果到 `chat-history/2026-07-05-qkv-ablation-results.md`。
- 对比指标：ZS/Ens 的 Transfer、Average、Last。
- 根据结果决定默认方案是否保持共享 P，或进一步评估 fused QKV。
