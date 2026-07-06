# Layer × Rank 消融结果

| Config | ZS Transfer | ZS Average | ZS Last | Ens Transfer | Ens Average | Ens Last |
|---|---:|---:|---:|---:|---:|---:|
| Layer: All (baseline) | 59.96 | 69.61 | 79.18 | 60.04 | 70.65 | 81.41 |
| Layer: QK only | 61.17 | 63.97 | 69.85 | 61.37 | 66.04 | 73.90 |
| Layer: FFN only | 59.75 | 68.70 | 77.59 | 59.91 | 70.12 | 80.45 |
| Layer: Attention only | 61.29 | 67.79 | 75.35 | 61.39 | 69.21 | 78.44 |
| Rank: 16 | 59.48 | 69.63 | 79.52 | 59.63 | 70.59 | 81.43 |
| Rank: 2 | 60.38 | 68.40 | 76.95 | 60.49 | 69.79 | 79.78 |
| Rank: 4 | 59.87 | 68.95 | 78.15 | 59.97 | 70.12 | 80.57 |
| Rank: 8 | 59.32 | 69.30 | 79.19 | 59.46 | 70.39 | 81.56 |

## 按 Ens Average 排序

- **Layer: All (baseline)**: Ens Avg 70.65 (T 60.04 / L 81.41)
- **Rank: 16**: Ens Avg 70.59 (T 59.63 / L 81.43)
- **Rank: 8**: Ens Avg 70.39 (T 59.46 / L 81.56)
- **Layer: FFN only**: Ens Avg 70.12 (T 59.91 / L 80.45)
- **Rank: 4**: Ens Avg 70.12 (T 59.97 / L 80.57)
- **Rank: 2**: Ens Avg 69.79 (T 60.49 / L 79.78)
- **Layer: Attention only**: Ens Avg 69.21 (T 61.39 / L 78.44)
- **Layer: QK only**: Ens Avg 66.04 (T 61.37 / L 73.90)
