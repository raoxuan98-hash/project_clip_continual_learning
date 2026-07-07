# 最佳配置写入默认值 + reference_loader 修复

**日期**: 2026-07-07
**Commit**: `df92c9c`

---

## 1. CLI 默认值更新

基于 10-task 重消融完整结论，将 `main_incremental.py` 的 CLI 默认值更新为最佳配置：

| 参数 | 旧默认 | 新默认 | Phase |
|------|:---:|:---:|:---:|
| `--batch_size` | 64 | **32** | Phase 2: bs=32 ZS A=69.83, bs=64 ZS A=69.55 |
| `--seed` | 42 | **43** | Phase 1 统一 |
| `--scheduler` | cosine | **cosine_with_warmup** | Phase 1 公共配置 |
| `--use_dora` | true | **false** | Pre-A: LoRA > DoRA (3/3) |
| `--nsp_eps` | 0.05 | **0.20** | Phase 7b: ZS A +0.42pp vs 0.05 |
| `--cd_weight` | 1.0 | **2.0** | 公共配置 |
| `--cd_temperature` | 2.0 | **4.0** | Phase 3: ZS A 69.91 vs 69.83 |
| `--aux_weight` | 1.0 | **0.0** | 6-task 不敏感 |
| `--text_tuning_schedule` | freeze_after | **always** | Pre-A: always > low_lr > freeze |
| `--num_centers` | 1 | **4** | mc4ft200 |
| `--rgda_train_iter` | 0 | **200** | mc4ft200 (analytical → GMM fine-tuned) |

Docstring 同步更新。

## 2. reference_loader 硬编码修复

`src/utils/reference_loader.py:99` 中 `batch_size=32` 硬编码，导致 `--reference_batch_size` CLI 参数被完全忽略。Phase 9 的 4 个 ref_bs 变体均使用了 32，实验白跑。

**修复**：
```python
# 修复前
reference_loader = DataLoader(merged_ref_dataset, batch_size=32, ...)

# 修复后
reference_loader = DataLoader(merged_ref_dataset, batch_size=getattr(args, 'reference_batch_size', 32), ...)
```

此 bug 与之前的 `cd_temperature`/`cd_divergence`、`lora_target_modules`、`use_soft_projection` 同类：CLI 参数已注册但未接入实际代码路径。

## 3. 全消融最佳配置

```text
LoRA+NSP, hard projection, nsp_eps=0.20
text_tuning_schedule=always
lr=1e-4, batch_size=32
scheduler=cosine_with_warmup, eta_min=0.0
cd_weight=2.0, fd_weight=1.0, aux_weight=0.0
cd_divergence=kl_forward, cd_temperature=4.0
num_centers=4, rgda_train_iter=200, rgda_train_lr=0.01, rgda_fit_source=gmm_sample
ensemble_normalize=maxshift, seed=43
```

**ZS Average = 70.33 | Ens Average = 72.18 | Ens Last = 83.93**
