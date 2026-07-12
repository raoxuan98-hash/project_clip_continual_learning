#!/usr/bin/env python3
import json
from pathlib import Path

PROJECT = Path('/home/raoxuan/projects/project_clip_continual_learning')
OUT_DIR = PROJECT / 'experiments' / 'best_combo_ablation'
NAMES = [
    ('combo_balanced_adamw_1e-4_iter800_cwu', 'Combo 1: Best Balanced'),
    ('combo_last_adamw_3e-4_iter1600_linear', 'Combo 2: Last Max'),
    ('combo_transfer_hist_runtime_adamw_1e-4_iter800_cosine', 'Combo 3: Transfer Max'),
    ('combo_init_only_hist_only_adamw_1e-4_iter800_cwu', 'Combo 4: Init-Only'),
]

def load_metrics(name):
    zs_path = OUT_DIR / f'{name}_zs_results.json'
    ens_path = OUT_DIR / f'{name}_ens_results.json'
    with open(zs_path) as f:
        zs = json.load(f)['metrics']
    with open(ens_path) as f:
        ens = json.load(f)['metrics']
    return zs, ens

rows = []
for name, label in NAMES:
    try:
        zs, ens = load_metrics(name)
        rows.append({
            'name': label,
            'zs_t': zs['transfer'], 'zs_a': zs['average'], 'zs_l': zs['last'],
            'ens_t': ens['transfer'], 'ens_a': ens['average'], 'ens_l': ens['last'],
        })
    except Exception as e:
        print(f'Skipping {name}: {e}')

if not rows:
    print('No results yet.')
    exit(0)

md = '# Best Config Combo 消融结果\n\n'
md += '| Combo | ZS Transfer | ZS Average | ZS Last | Ens Transfer | Ens Average | Ens Last |\n'
md += '|---|---:|---:|---:|---:|---:|---:|\n'
for r in rows:
    md += f"| {r['name']} | {r['zs_t']:.2f} | {r['zs_a']:.2f} | {r['zs_l']:.2f} | {r['ens_t']:.2f} | {r['ens_a']:.2f} | {r['ens_l']:.2f} |\n"

md += '\n## 按 Ens Average 排序\n\n'
for r in sorted(rows, key=lambda x: x['ens_a'], reverse=True):
    md += f"- **{r['name']}**: Ens Avg {r['ens_a']:.2f} (T {r['ens_t']:.2f} / L {r['ens_l']:.2f})\n"

out_path = PROJECT / 'chat-history' / '2026-07-04-best-combo-ablation-results.md'
out_path.write_text(md, encoding='utf-8')
print(f'Summary written to {out_path}')
print(md)
