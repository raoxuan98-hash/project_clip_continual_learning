#!/usr/bin/env python3
"""aggregate_pa_results.py — 聚合 PA_main（或任何 campaign 目录）下的实验结果。

每个 run 的五件套：
  <name>.json / <name>_zs_results.json / <name>_rgda_results.json /
  <name>_ens_results.json / <name>_retrieval.json

输出：
  1) 每 run 一行：ZS / RGDA / ENS 的 Transfer/Average/Last + 检索 R@1（最终 step）
  2) 同配置跨 seed 的 mean±std（按实验名去掉 seed 后缀分组）
  3) ENS 逐数据集 T/A/L（LADA 式主表用）

用法：
  python scripts/aggregate_pa_results.py --dir experiments/paper_formal/PA_main
  python scripts/aggregate_pa_results.py --dir .../AP_main --pattern 'AP__iters600*' --csv out.csv
"""

import argparse
import glob
import json
import os
import re
import statistics
import sys


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def final_retrieval(entries):
    """retrieval.json 是 list，按 (step, task, dataset) 记录；取最大 step 的条目。"""
    if not entries:
        return {}
    max_step = max(e.get("step", 0) for e in entries)
    out = {}
    for e in entries:
        if e.get("step", 0) == max_step:
            out[e["dataset"]] = e
    return out


def fmt(v, nd=2):
    if isinstance(v, (int, float)):
        return f"{v:.{nd}f}"
    return v if isinstance(v, str) else "--"


def mean_std(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    if not vals:
        return "--"
    if len(vals) == 1:
        return fmt(vals[0])
    return f"{statistics.mean(vals):.2f}±{statistics.stdev(vals):.2f}"


def collect(results_dir, pattern):
    ens_files = sorted(glob.glob(os.path.join(results_dir, pattern + "_ens_results.json")))
    runs = []
    for ef in ens_files:
        name = os.path.basename(ef)[: -len("_ens_results.json")]
        run = {"name": name}
        ok = True
        for key, suffix in [("zs", "_zs_results"), ("rgda", "_rgda_results"), ("ens", "_ens_results")]:
            d = load_json(os.path.join(results_dir, name + suffix + ".json"))
            if d is None or "metrics" not in d:
                ok = False
                break
            run[key] = d["metrics"]
            run[key + "_per_task"] = d.get("per_task_metrics", {})
        if not ok:
            run["incomplete"] = True
        ret = load_json(os.path.join(results_dir, name + "_retrieval.json"))
        run["retrieval"] = final_retrieval(ret) if ret else {}
        runs.append(run)
    return runs


def seed_group(name):
    return re.sub(r"seed\d+$", "seed*", name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="experiments/paper_formal/PA_main")
    ap.add_argument("--pattern", default="PA__*")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--per-dataset", action="store_true", help="打印 ENS 逐数据集 T/A/L")
    args = ap.parse_args()

    runs = collect(args.dir, args.pattern)
    if not runs:
        print(f"[warn] no runs matching {args.pattern}_ens_results.json in {args.dir}")
        sys.exit(1)

    datasets = None
    rows = []
    for r in runs:
        row = {"run": r["name"]}
        if r.get("incomplete"):
            row["note"] = "INCOMPLETE (missing json)"
        for head in ("zs", "rgda", "ens"):
            m = r.get(head, {})
            for metric in ("transfer", "average", "last"):
                row[f"{head}_{metric}"] = m.get(metric)
        for ds, e in r.get("retrieval", {}).items():
            row[f"{ds}_i2t_r1"] = e.get("i2t_r@1")
            row[f"{ds}_t2i_r1"] = e.get("t2i_r@1")
        if datasets is None and r.get("ens_per_task"):
            datasets = list(r["ens_per_task"].keys())
        rows.append(row)

    cols = ["run", "zs_transfer", "zs_average", "zs_last",
            "rgda_transfer", "rgda_average", "rgda_last",
            "ens_transfer", "ens_average", "ens_last",
            "mscoco_2014_5k_i2t_r1", "mscoco_2014_5k_t2i_r1",
            "flickr30k_hf_i2t_r1", "flickr30k_hf_t2i_r1"]

    print("## per-run summary")
    print("|" + "|".join(cols) + "|")
    print("|" + "|".join(["---"] * len(cols)) + "|")
    for row in rows:
        print("|" + "|".join(fmt(row.get(c)) if c != "run" else row["run"] for c in cols) + "|")

    # 跨 seed 聚合
    groups = {}
    for row in rows:
        groups.setdefault(seed_group(row["run"]), []).append(row)
    print("\n## seed-aggregated (mean±std)")
    gcols = cols[1:]
    print("|config|" + "|".join(gcols) + "|")
    print("|" + "|".join(["---"] * (len(gcols) + 1)) + "|")
    for g, grows in sorted(groups.items()):
        print("|" + g + "|" + "|".join(mean_std([r.get(c) for r in grows]) for c in gcols) + "|")

    if args.per_dataset and datasets:
        print("\n## ENS per-dataset (transfer / average / last)")
        print("|run|" + "|".join(datasets) + "|mean|")
        print("|" + "|".join(["---"] * (len(datasets) + 2)) + "|")
        for r in runs:
            pt = r.get("ens_per_task", {})
            for metric in ("transfer", "average", "last"):
                vals = [pt.get(ds, {}).get(metric) for ds in datasets]
                # Transfer 协议：首任务无定义（N/A），均值须剔除首列
                nums_src = vals[1:] if metric == "transfer" else vals
                nums = [v for v in nums_src if isinstance(v, (int, float))]
                mean = statistics.mean(nums) if nums else None
                if metric == "transfer":
                    vals = ["N/A"] + vals[1:]
                print(f"|{r['name']} {metric}|" + "|".join(fmt(v) for v in vals) + f"|{fmt(mean)}|")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for row in rows:
                w.writerow({c: (fmt(row.get(c)) if c != "run" else row["run"]) for c in cols})
        print(f"\n[csv] {args.csv}")


if __name__ == "__main__":
    main()
