# Ablation Results Final Summary (2026-07-02)

Generated: 2026-07-02T10:39:58.215230

All experiments completed. Transfer metric aligned with LADA paper formula.

## 1. Optimizer Ablation (LoRA)

| variant | opt | extra | ZS T | ZS A | ZS L | Ens T | Ens A | Ens L |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| current_nsp | adagrad |  | 60.07 | 67.62 | 76.27 | 60.10 | 69.08 | 79.64 |
| current_nsp | adam |  | 60.87 | 68.59 | 77.54 | 60.91 | 69.85 | 80.30 |
| current_nsp | adamw |  | 60.89 | 68.59 | 77.55 | 60.92 | 69.85 | 80.25 |
| current_nsp | rmsprop |  | 60.87 | 68.64 | 77.58 | 60.89 | 69.89 | 80.36 |
| current_nsp | sgd |  | 60.88 | 65.02 | 71.25 | 60.92 | 66.72 | 74.43 |
| hist_null_init_runtime | adagrad |  | 60.45 | 67.94 | 76.33 | 60.49 | 69.26 | 79.52 |
| hist_null_init_runtime | adam |  | 61.24 | 68.32 | 76.63 | 61.27 | 69.74 | 79.71 |
| hist_null_init_runtime | adamw |  | 61.27 | 68.35 | 76.71 | 61.30 | 69.75 | 79.70 |
| hist_null_init_runtime | rmsprop |  | 60.86 | 68.52 | 77.65 | 60.91 | 69.74 | 80.24 |
| hist_null_init_runtime | sgd |  | 61.27 | 66.13 | 72.26 | 61.36 | 67.84 | 75.58 |

## 2. SGD Learning Rate Exploration

| variant | opt | extra | ZS T | ZS A | ZS L | Ens T | Ens A | Ens L |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| current_nsp | sgd |  | 60.88 | 65.02 | 71.25 | 60.92 | 66.72 | 74.43 |
| current_nsp | sgd | lr5e-2 | 61.35 | 26.39 | 0.17 | 61.35 | 38.19 | 70.95 |
| current_nsp | sgd | lr5e-3 | 57.44 | 68.53 | 79.00 | 57.59 | 69.99 | 82.24 |
| hist_null_init_runtime | sgd |  | 61.27 | 66.13 | 72.26 | 61.36 | 67.84 | 75.58 |
| hist_null_init_runtime | sgd | lr5e-2 | 61.35 | 26.39 | 0.17 | 61.35 | 38.19 | 70.97 |
| hist_null_init_runtime | sgd | lr5e-3 | 56.39 | 66.85 | 74.89 | 56.45 | 68.66 | 79.43 |

## 3. Learning Rate Ablation (current_nsp + AdamW)

| variant | opt | extra | ZS T | ZS A | ZS L | Ens T | Ens A | Ens L |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| current_nsp | adamw | lr1e-4 | 60.89 | 68.59 | 77.55 | 60.92 | 69.85 | 80.25 |
| current_nsp | adamw | lr3e-4 | 58.25 | 68.75 | 79.56 | 58.29 | 69.79 | 81.87 |
| current_nsp | adamw | lr5e-5 | 61.75 | 67.24 | 74.69 | 61.81 | 68.77 | 77.86 |
| current_nsp | adamw | lr6e-4 | 57.97 | 68.27 | 78.54 | 58.03 | 69.64 | 81.47 |

## 4. Batch Size Ablation (current_nsp + AdamW + lr=1e-4)

| variant | opt | extra | ZS T | ZS A | ZS L | Ens T | Ens A | Ens L |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| current_nsp | adamw | bs128 | 60.13 | 69.06 | 78.69 | 60.18 | 70.14 | 81.01 |
| current_nsp | adamw | bs32 | 61.66 | 67.78 | 75.80 | 61.68 | 69.31 | 79.02 |
| current_nsp | adamw | bs64 | 60.89 | 68.59 | 77.55 | 60.92 | 69.85 | 80.25 |

## 5. Distillation / Auxiliary Head Ablation (current_nsp + AdamW + lr=1e-4)

| variant | opt | extra | ZS T | ZS A | ZS L | Ens T | Ens A | Ens L |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| current_nsp | adamw | aux0p0 | 60.43 | 68.98 | 78.15 | 60.48 | 70.03 | 80.44 |
| current_nsp | adamw | aux1p0 | 60.89 | 68.59 | 77.55 | 60.92 | 69.85 | 80.25 |
| current_nsp | adamw | aux2p0 | 61.09 | 68.12 | 76.50 | 61.13 | 69.46 | 79.78 |
| current_nsp | adamw | cd0p0 | 60.51 | 68.02 | 77.09 | 60.54 | 69.41 | 80.27 |
| current_nsp | adamw | cd1p0 | 60.89 | 68.59 | 77.55 | 60.92 | 69.85 | 80.25 |
| current_nsp | adamw | cd2p0 | 60.96 | 68.71 | 77.67 | 61.01 | 69.95 | 80.27 |
| current_nsp | adamw | fd0p0 | 60.92 | 68.61 | 77.57 | 60.97 | 69.86 | 80.32 |
| current_nsp | adamw | fd1p0 | 60.89 | 68.59 | 77.55 | 60.92 | 69.85 | 80.25 |
| current_nsp | adamw | fd2p0 | 60.86 | 68.59 | 77.53 | 60.90 | 69.85 | 80.18 |

## Best Configs by (Transfer + Last) / 2

| setting | variant | opt/extra | ZS (T+L)/2 | Ens (T+L)/2 |
|---|---|---|---:|
| Optimizer (current_nsp) | current_nsp | rmsprop | 69.22 | 70.62 |
| Optimizer (hist) | hist_null_init_runtime | rmsprop | 69.26 | 70.57 |
| LR | current_nsp | lr1e-4 | 69.22 | 70.59 |
| Batch size | current_nsp | bs128 | 69.41 | 70.59 |
| fd_weight | current_nsp | fd0p0 | 69.24 | 70.65 |
| cd_weight | current_nsp | cd2p0 | 69.32 | 70.64 |
| aux_weight | current_nsp | aux0p0 | 69.29 | 70.59 |

## Best Configs by Last

| setting | variant | opt/extra | ZS Last | Ens Last |
|---|---|---:|---:|
| Optimizer (current_nsp) | current_nsp | rmsprop | 77.58 | 80.36 |
| LR | current_nsp | lr3e-4 | 79.56 | 81.87 |
| Batch size | current_nsp | bs128 | 78.69 | 81.01 |
| fd_weight | current_nsp | fd0p0 | 77.57 | 80.32 |
| cd_weight | current_nsp | cd2p0 | 77.67 | 80.27 |
| aux_weight | current_nsp | aux0p0 | 78.15 | 80.44 |

## Final Conclusions

- **Optimizer**: RMSprop, AdamW, Adam are all strong; RMSprop edges ahead on Last for both `current_nsp` and `hist_null_init_runtime`.
- **SGD LR**: 5e-2 destroys zero-shot; 5e-3 helps final accuracy for `current_nsp` but hurts Transfer. For `hist_null_init_runtime`, SGD 5e-3 is less effective.
- **Learning rate**: 1e-4 is the best balance; 3e-4 gives the highest Last but lower Transfer; 5e-5 is too small.
- **Batch size**: Larger batch (64/128) improves Average/Last over 32; 128 is slightly better than 64.
- **fd_weight**: 0.0, 1.0, 2.0 are nearly identical → feature distillation weight is insensitive.
- **cd_weight**: cd=2.0 gives the best Transfer and Last among {0,1,2}; class distillation helps modestly.
- **aux_weight**: aux=0.0 gives the best Average, aux=1.0 gives the best Last; effect is small.
- **Overall best single run (Ens Last)**: `current_nsp + AdamW + lr=3e-4` achieves Ens Last 81.87; `current_nsp + SGD 5e-3` reaches 82.24 but with much lower Transfer.
