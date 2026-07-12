#!/bin/bash
# 自动监控实验完成并更新 ablation md 文件
# 用法: nohup bash scripts/auto_update_ablation_md.sh &
# 这个脚本独立于 VSCode 会话运行

PID=2424119
LOG_FILE="/data/home/zengyong1/projects/clip_ood_v2/experiments/inc_full_20260616_000303.log"
MD_FILE="/data/home/zengyong1/projects/clip_ood_v2/docs/ablation_num_centers.md"
PROJECT_DIR="/data/home/zengyong1/projects/clip_ood_v2"

echo "[auto_update] Monitoring experiment PID=$PID"
echo "[auto_update] Log: $LOG_FILE"
echo "[auto_update] Started at $(date)"

# 等待实验进程结束
while kill -0 $PID 2>/dev/null; do
    sleep 60
done

echo "[auto_update] Experiment PID=$PID finished at $(date)"
sleep 5  # 等待文件系统刷新

# 检查日志是否完整
FINAL_LINE=$(tail -1 "$LOG_FILE")
if ! echo "$FINAL_LINE" | grep -q "增量学习结果已保存"; then
    echo "[auto_update] WARNING: Log may be incomplete. Last line: $FINAL_LINE"
    echo "[auto_update] Waiting 60s and retrying..."
    sleep 60
fi

cd "$PROJECT_DIR"

# 提取结果并计算指标
source ~/anaconda3/etc/profile.d/conda.sh
conda activate clip

python3 -c "
import re

log_file = '${LOG_FILE}'
md_file = '${MD_FILE}'

# Parse results from log
lines = open(log_file).read()
pattern = r'\[Tested on Task (\d+):\s+(\S+)\s*\].*?Zero-shot:\s+([\d.]+)%.*?LR-RGDA:\s+([\d.]+)%.*?Ensemble:\s+([\d.]+)%'
matches = re.findall(pattern, lines)

if not matches:
    print('[auto_update] ERROR: No results found in log!')
    exit(1)

# Group by step
from collections import defaultdict
steps = defaultdict(list)
for task_id, task_name, zs, rgda, ens in matches:
    steps[int(task_id)].append((task_name, float(zs), float(rgda), float(ens)))

K = max(steps.keys())
task_names = [steps[i][-1][0] for i in range(1, K+1)]  # last occurrence gives task name

zs_mat = []; rgda_mat = []; ens_mat = []
for i in range(1, K+1):
    _, zs_row, rgda_row, ens_row = [], [], [], []
    for _, z, r, e in steps[i]:
        zs_row.append(z)
        rgda_row.append(r)
        ens_row.append(e)
    zs_mat.append(zs_row)
    rgda_mat.append(rgda_row)
    ens_mat.append(ens_row)

def compute_metrics(matrix):
    K = len(matrix)
    trans = [0.0]
    for k in range(1, K):
        trans.append(sum(matrix[k][j] for j in range(k)) / k)
    avgs = [sum(matrix[j][k] for j in range(k, K)) / (K - k) for k in range(K)]
    lasts = matrix[K-1]
    t_avg = sum(trans[1:]) / (K-1)
    a_avg = sum(avgs) / K
    l_avg = sum(lasts) / K
    return trans, avgs, lasts, t_avg, a_avg, l_avg

# Compute for all three classifiers
results = {}
for name, mat in [('ZS', zs_mat), ('RGDA', rgda_mat), ('Ensemble', ens_mat)]:
    trans, avgs, lasts, t_avg, a_avg, l_avg = compute_metrics(mat)
    results[name] = {
        'trans': trans, 'avgs': avgs, 'lasts': lasts,
        't_avg': t_avg, 'a_avg': a_avg, 'l_avg': l_avg
    }

# Generate markdown to append
md_section = f'''

---

## Exp 3: num_centers=4 + GMM mean replay + rgda_train_iter=200

**日志**: \`inc_full_20260616_000303.log\`
**配置**: num_centers=4, rgda_train_iter=200, gmm_sample_mode=mean, gaussian_samples_per_class=16, text_tuning_schedule=low_lr_after

### Zero-shot

| Task | Transfer | Average | Last |
|------|:-------:|:-------:|:----:|
'''
for i, t in enumerate(task_names):
    md_section += f'| {t} | {results[\"ZS\"][\"trans\"][i]:.2f} | {results[\"ZS\"][\"avgs\"][i]:.2f} | {results[\"ZS\"][\"lasts\"][i]:.2f} |\n'
md_section += f'| **Overall** | **{results[\"ZS\"][\"t_avg\"]:.2f}** | **{results[\"ZS\"][\"a_avg\"]:.2f}** | **{results[\"ZS\"][\"l_avg\"]:.2f}** |\n'

md_section += f'''
### LR-RGDA

| Task | Transfer | Average | Last |
|------|:-------:|:-------:|:----:|
'''
for i, t in enumerate(task_names):
    md_section += f'| {t} | {results[\"RGDA\"][\"trans\"][i]:.2f} | {results[\"RGDA\"][\"avgs\"][i]:.2f} | {results[\"RGDA\"][\"lasts\"][i]:.2f} |\n'
md_section += f'| **Overall** | **{results[\"RGDA\"][\"t_avg\"]:.2f}** | **{results[\"RGDA\"][\"a_avg\"]:.2f}** | **{results[\"RGDA\"][\"l_avg\"]:.2f}** |\n'

md_section += f'''
### Ensemble (α=0.05)

| Task | Transfer | Average | Last |
|------|:-------:|:-------:|:----:|
'''
for i, t in enumerate(task_names):
    md_section += f'| {t} | {results[\"Ensemble\"][\"trans\"][i]:.2f} | {results[\"Ensemble\"][\"avgs\"][i]:.2f} | {results[\"Ensemble\"][\"lasts\"][i]:.2f} |\n'
md_section += f'| **Overall** | **{results[\"Ensemble\"][\"t_avg\"]:.2f}** | **{results[\"Ensemble\"][\"a_avg\"]:.2f}** | **{results[\"Ensemble\"][\"l_avg\"]:.2f}** |\n'

# Read existing md to find the final comparison table, append new row
with open(md_file, 'r') as f:
    existing = f.read()

# Update the final comparison section
new_row = f'| Exp3 (4c+GMM+fit) Ensemble | {results[\"Ensemble\"][\"t_avg\"]:.2f} | {results[\"Ensemble\"][\"a_avg\"]:.2f} | {results[\"Ensemble\"][\"l_avg\"]:.2f} |'
new_row_rgda = f'| Exp3 (4c+GMM+fit) RGDA | {results[\"RGDA\"][\"t_avg\"]:.2f} | {results[\"RGDA\"][\"a_avg\"]:.2f} | {results[\"RGDA\"][\"l_avg\"]:.2f} |'

# Append new section and updated comparison
with open(md_file, 'a') as f:
    f.write(md_section)
    f.write(f'''
## 更新后的最终对比

| Method | Transfer | Average | Last |
|--------|:-------:|:-------:|:----:|
| Exp1 (4-center, 无微调) ZS | 68.87 | 76.55 | 74.06 |
| Exp1 (4-center, 无微调) RGDA | 70.05 | 76.87 | 75.94 |
| Exp1 (4-center, 无微调) Ensemble | 71.31 | 79.25 | 77.61 |
| Exp2 (1-center, 无微调) ZS | 68.88 | 76.56 | 74.09 |
| Exp2 (1-center, 无微调) RGDA | 74.41 | 81.49 | 80.63 |
| Exp2 (1-center, 无微调) Ensemble | 71.61 | 79.47 | 77.88 |
{new_row_rgda}
{new_row}
| Old code (May 29) Ensemble | — | — | 80.69 |

### 关键差异

| 对比 | Last Δ | Average Δ |
|------|:------:|:--------:|
| Ensemble: Exp3 (4c+GMM+fit) − Exp2 (1c, 无微调) | {results[\"Ensemble\"][\"l_avg\"] - 77.88:+.2f} | {results[\"Ensemble\"][\"a_avg\"] - 79.47:+.2f} |
| Ensemble: Exp3 (4c+GMM+fit) − Exp1 (4c, 无微调) | {results[\"Ensemble\"][\"l_avg\"] - 77.61:+.2f} | {results[\"Ensemble\"][\"a_avg\"] - 79.25:+.2f} |
| RGDA: Exp3 (4c+GMM+fit) − Exp1 (4c, 无微调) | {results[\"RGDA\"][\"l_avg\"] - 75.94:+.2f} | {results[\"RGDA\"][\"a_avg\"] - 76.87:+.2f} |

*Generated automatically at $(date)*
''')

print(f'[auto_update] MD file updated successfully at $(date)')
print(f'[auto_update] Exp3 Ensemble Last = {results[\"Ensemble\"][\"l_avg\"]:.2f}')
print(f'[auto_update] Exp3 RGDA Last = {results[\"RGDA\"][\"l_avg\"]:.2f}')
" 2>&1

echo "[auto_update] Done at $(date)"
