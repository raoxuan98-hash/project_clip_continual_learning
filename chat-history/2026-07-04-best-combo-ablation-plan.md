# Best Config Combo 消融计划

**日期**: 2026-07-04
**目标**: 把近期 LoRA/优化器/学习率/温度/迭代次数/退火方式 消融中表现最好的单点配置组合起来，验证叠加效果。

## 一、各维度最佳单点（基于 seed=42）

| 维度 | 最佳候选 | 依据 | 关键指标 |
|---|---|---|---|
| LoRA 变体 | `current_nsp` (`projection_param_mode=full`, `null_init_mode=none`) | iter/scheduler 消融基线 | Last 最高 |
|  | `hist_null_init_runtime` | optimizer 消融 Transfer 略高 | Transfer 略高 |
|  | `hist_null_init_only` | null basis 消融 3-dataset 最优 | Average +0.56 |
| 优化器 | AdamW / RMSprop | optimizer 消融 | RMSprop Last 略高，AdamW 接近 |
| 学习率 | 1e-4（平衡）/ 3e-4（Last 最高） | lr 消融 | 1e-4 平衡；3e-4 Last 最高但 Transfer 降 |
| Batch size | 128 | batch size 消融 | Ens Avg 70.14，Ens Last 81.01 |
| cd_weight | 2.0 | distill/aux 消融 | Transfer+Last 最优 |
| aux_weight | 0.0 | distill/aux 消融 | Average 最优 |
| fd_weight | 1.0 | 不敏感 | 保持默认值 |
| cd_divergence | kl_forward | divergence 消融 | Transfer 最优 |
| cd_temperature | 4.0 | temperature 消融 | ZS Transfer 60.31，Ens Transfer 60.37 |
| iterations | 800（平衡）/ 1600（Last） | iter/scheduler 消融 | 1600 Last 最高，800 Transfer 更稳 |
| scheduler | cosine_with_warmup / linear / cosine | iter/scheduler 消融 | 三者接近，linear Last 略好 |

## 二、组合方案（4 组，占用 GPU0/2/3/4）

### Combo 1: Best Balanced
- `current_nsp + AdamW + lr=1e-4 + bs=128 + iter=800 + cosine_with_warmup`
- 目标：在 Transfer/Average/Last 之间取得最佳平衡。

### Combo 2: Last Max
- `current_nsp + AdamW + lr=3e-4 + bs=128 + iter=1600 + linear`
- 目标：牺牲少量 Transfer，最大化 Last。

### Combo 3: Transfer Max
- `hist_null_init_runtime + AdamW + lr=1e-4 + bs=128 + iter=800 + cosine`
- 目标：最大化 Transfer，同时保持较高 Last。

### Combo 4: Init-Only Mechanism
- `hist_null_init_only + AdamW + lr=1e-4 + bs=128 + iter=800 + cosine_with_warmup`
- 目标：验证 null basis 消融中推荐的 init-only 机制在完整训练配方下的表现。

## 三、公共配置

```text
--root /data1/open_datasets/X-TAIL
--dataset_sequence aircraft caltech101 dtd eurosat flowers oxford_pets
--num_shots 16
--batch_size 128
--lora_type lora_nsp --lora_rank 4 --use_dora false
--train_budget_mode uniform
--cd_weight 2.0 --aux_weight 0.0 --fd_weight 1.0
--cd_divergence kl_forward --cd_temperature 4.0
--weight_decay 3e-5
--seed 42
--output_dir experiments/best_combo_ablation
```

## 四、结果收集

结果文件：`experiments/best_combo_ablation/{name}_zs_results.json` 与 `{name}_ens_results.json`
指标：`metrics.transfer`, `metrics.average`, `metrics.last`（已 ×100，百分比）
