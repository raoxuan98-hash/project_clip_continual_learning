# CD Temperature Ablation Summary (kl_forward)

Base config: current_nsp + AdamW + lr=1e-4 + cd_weight=2.0 + aux_weight=0.0 + fd_weight=1.0 + batch_size=64 + iterations=800 + lora_rank=4 + cd_divergence=kl_forward

## Zero-shot classifier (LADA metrics)

| cd_temperature | Transfer | Last | Average |
|---------------|----------|------:|---------:|
| 1.0 | 60.30 | 77.93 | 68.83 |
| 2.0 | 60.27 | 78.34 | 68.98 |
| 4.0 **best** | 60.31 | 78.40 | 69.05 |

Recommended temperature (zero-shot Transfer): `4.0`

## Ensemble classifier (LADA metrics)

| cd_temperature | Transfer | Last | Average |
|---------------|----------|------:|---------:|
| 1.0 | 60.34 | 80.56 | 70.00 |
| 2.0 | 60.30 | 80.72 | 70.09 |
| 4.0 | 60.37 | 80.70 | 70.14 |

Best ensemble Transfer at temperature: `4.0`
