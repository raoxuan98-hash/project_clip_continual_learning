#!/usr/bin/env python3
"""从战役原始 JSON 重算 3-seed mean±std 汇总表（Wave F）。

口径与 v2 汇总一致：
- 分类：直接读取 <name>_{zs,rgda,ens}_results.json 的 metrics.{transfer,average,last}，
  跨 seed 取 mean ± population std（pstdev）。
- 检索：<name>_retrieval.json 按 dataset 聚合：Avg = 10 个任务步的均值，
  Last = 最后一个任务步；再跨 seed mean±pstdev。
- run 分组：run 名去掉尾部 _seedNN 作为配置名。

用法：
    python scripts/aggregate_campaign_results.py experiments/paper_formal/WaveA_main \
        [--variants zs ens] [--output-md path]
"""
import argparse
import json
import re
from pathlib import Path
from statistics import mean, pstdev


def group_runs(wave_dir: Path):
    groups = {}
    for p in sorted(wave_dir.glob("*_ens_results.json")):
        stem = p.name[: -len("_ens_results.json")]
        m = re.match(r"^(.*?)_+seed(\d+)$", stem)
        if not m:
            continue
        config, seed = m.group(1), int(m.group(2))
        groups.setdefault(config, {})[seed] = stem
    return groups


def cls_metrics(wave_dir: Path, stem: str, variant: str):
    d = json.load(open(wave_dir / f"{stem}_{variant}_results.json"))
    met = d["metrics"]
    return {k: float(met[k]) * (100.0 if float(met[k]) <= 1.0 else 1.0)
            for k in ("transfer", "average", "last")}


def retr_metrics(wave_dir: Path, stem: str):
    entries = json.load(open(wave_dir / f"{stem}_retrieval.json"))
    by_ds = {}
    for e in entries:
        by_ds.setdefault(e["dataset"], []).append(e)
    out = {}
    for ds, es in by_ds.items():
        es.sort(key=lambda e: e["step"])
        out[ds] = {
            "avg_i2t": mean(float(e["i2t_r@1"]) for e in es),
            "avg_t2i": mean(float(e["t2i_r@1"]) for e in es),
            "last_i2t": float(es[-1]["i2t_r@1"]),
            "last_t2i": float(es[-1]["t2i_r@1"]),
        }
    return out


def fmt(vals):
    if not vals:
        return "—"
    if len(vals) == 1:
        return f"{vals[0]:.2f}"
    return f"{mean(vals):.2f} ± {pstdev(vals):.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wave_dirs", nargs="+", type=Path)
    ap.add_argument("--variants", default="zs ens")
    ap.add_argument("--seeds", default="42 43 44")
    ap.add_argument("--output-md", type=Path)
    args = ap.parse_args()

    variants = args.variants.split()
    expected_seeds = {int(s) for s in args.seeds.split()}
    lines = []

    for wave_dir in args.wave_dirs:
        groups = group_runs(wave_dir)
        lines.append(f"\n## {wave_dir.name}（{len(groups)} 配置）\n")
        # 分类表
        header = "| Config | seeds |" + "".join(f" {v} {m} |" for v in variants for m in ("Transfer", "Average", "Last"))
        lines.append(header)
        lines.append("|" + "---|" * (2 + 3 * len(variants)))
        for config, seed_map in sorted(groups.items()):
            missing = expected_seeds - set(seed_map)
            row = f"| {config} | {len(seed_map)} |"
            if missing:
                row += f" | ⚠ missing seeds {sorted(missing)}" + "|" * (3 * len(variants) - 1)
            else:
                for v in variants:
                    per_seed = [cls_metrics(wave_dir, stem, v) for stem in seed_map.values()]
                    for key in ("transfer", "average", "last"):
                        row += f" {fmt([m[key] for m in per_seed])} |"
            lines.append(row)
        # 检索表
        lines.append("")
        lines.append("| Config | Dataset | Avg I2T R@1 | Avg T2I R@1 | Last I2T R@1 | Last T2I R@1 |")
        lines.append("|---|---|---|---|---|---|")
        for config, seed_map in sorted(groups.items()):
            if expected_seeds - set(seed_map):
                continue
            per_seed = [retr_metrics(wave_dir, stem) for stem in seed_map.values()]
            datasets = sorted(per_seed[0].keys())
            for ds in datasets:
                row = f"| {config} | {ds} |"
                for key in ("avg_i2t", "avg_t2i", "last_i2t", "last_t2i"):
                    row += f" {fmt([m[ds][key] for m in per_seed])} |"
                lines.append(row)

    text = "\n".join(lines)
    print(text)
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(text + "\n", encoding="utf-8")
        print(f"\nwritten -> {args.output_md}")


if __name__ == "__main__":
    main()
