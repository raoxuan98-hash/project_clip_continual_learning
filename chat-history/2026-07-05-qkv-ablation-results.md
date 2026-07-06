# QKV 投影方案消融结果

**日期**: 2026-07-05

三组实验使用完全相同的超参：rank=4，target modules = q/k/v/out/ffn，其余为当前最佳配置（use_dora=false, lora_type=lora_nsp, cd_weight=2.0, cd_temperature=4.0, cd_divergence=kl_forward, lr=1e-4, iterations=800, scheduler=cosine_with_warmup）。

## 三方结果对比

| 方案 | ZS Transfer | ZS Average | ZS Last | Ens Transfer | Ens Average | Ens Last |
|---|---:|---:|---:|---:|---:|---:|
| 当前默认：共享 P（方案 A）| 59.96 | 69.61 | 79.18 | 60.04 | 70.65 | 81.41 |
| 旧默认：独立 P | 60.06 | 69.36 | 78.54 | 60.18 | 70.55 | 81.12 |
| 方案 B：fused QKV | 56.56 | 65.33 | 72.78 | 56.66 | 66.06 | 74.25 |

## 按 Ens Average 排序

1. **共享 P（方案 A，当前默认）**: Ens Avg **70.65**（T 60.04 / L 81.41）
2. **独立 P（旧默认）**: Ens Avg **70.55**（T 60.18 / L 81.12）
3. **fused QKV（方案 B）**: Ens Avg **66.06**（T 56.66 / L 74.25）

## 结论

- **共享 P 仍是三者中最优**，与 layer+rank 消融中的 baseline 一致。
- 独立 P（旧实现）与共享 P 差距很小（Ens Avg 相差约 0.1 个百分点），说明共享 P 的简化没有带来性能损失，同时减少了协方差计算和 SVD 的冗余。
- **fused QKV 明显落后**（Ens Avg 比共享 P 低约 4.6 个百分点），在该配置下不值得替换当前默认方案。

## 备注

- 独立 P 对照通过临时设置 `SHARE_QKV=0` 启动，运行日志与结果保存在 `experiments/qkv_ablation/`。
- fused QKV 通过 `--fused_qkv` 启动，结果同样保存在 `experiments/qkv_ablation/`。
