#!/usr/bin/env python3
"""战役单 run / 整 wave 验收（计划 docs/rerun_campaign_main_v3_2026-07-18.md §3）。

对每个 run 检查：
1. 四类 JSON 齐全：<name>_{zs,rgda,ens}_results.json + <name>_retrieval.json
2. accuracy_matrix 为 K×K、值域 [0,1]、无 NaN/Inf
3. args.task_sequence 长度 == K，args.seed 与 run 名中的 seedN 一致
4. retrieval.json：K 任务 × 数据集条目齐全；num_images/num_captions 为
   canonical 计数（mscoco_2014_5k=5000/≥25000, flickr30k_hf=1000/5000），
   证明检索断言未触发、且未使用非 canonical 副本
5. 日志包含关键配置行（--reference_dataset flickr30k_train_sub8k）

用法：
    python scripts/audit_campaign_run.py <wave_dir> [--expected-k 10] [--require-reference]
退出码：全部 PASS 为 0，任一 FAIL 为 1；目录中尚无完成 run 时为 2（进行中）。
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

EXPECTED_COUNTS = {
    "mscoco_2014_5k": (5000, 25000),   # captions 为下限
    "flickr30k_hf": (1000, 5000),
}
RESULT_SUFFIXES = ("zs", "rgda", "ens")


def check_run(stem: str, wave_dir: Path, k: int, require_reference: bool):
    problems = []
    # 1. 四类 JSON
    paths = {s: wave_dir / f"{stem}_{s}_results.json" for s in RESULT_SUFFIXES}
    retr_path = wave_dir / f"{stem}_retrieval.json"
    missing = [str(p.name) for p in list(paths.values()) + [retr_path] if not p.exists()]
    if missing:
        return "INCOMPLETE", [f"missing: {', '.join(missing)}"]

    # 2/3. 分类结果
    m = re.search(r"seed(\d+)$", stem)
    expected_seed = int(m.group(1)) if m else None
    for s, p in paths.items():
        d = json.load(open(p))
        mat = d.get("accuracy_matrix")
        if not isinstance(mat, list) or len(mat) != k or any(not isinstance(r, list) or len(r) != k for r in mat):
            problems.append(f"{p.name}: matrix not {k}x{k}")
        else:
            flat = [v for r in mat for v in r]
            if any(not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v) for v in flat):
                problems.append(f"{p.name}: NaN/Inf in matrix")
            elif any(v < 0.0 or v > 1.0 for v in flat):
                problems.append(f"{p.name}: value out of [0,1]")
        args = d.get("args", {})
        seq = args.get("task_sequence") or []
        if len(seq) != k:
            problems.append(f"{p.name}: task_sequence len={len(seq)} != {k}")
        if expected_seed is not None and int(args.get("seed", -1)) != expected_seed:
            problems.append(f"{p.name}: seed={args.get('seed')} != {expected_seed}")
        met = d.get("metrics", {})
        for key in ("transfer", "average", "last"):
            if key not in met:
                problems.append(f"{p.name}: metrics missing '{key}'")

    # 4. 检索
    entries = json.load(open(retr_path))
    if not isinstance(entries, list):
        problems.append(f"{retr_path.name}: not a list")
    else:
        seen = {}
        for e in entries:
            ds = e.get("dataset")
            seen.setdefault(ds, set()).add(e.get("task"))
            exp = EXPECTED_COUNTS.get(ds)
            if exp:
                ni, nc = e.get("num_images", 0), e.get("num_captions", 0)
                if ni != exp[0] or nc < exp[1]:
                    problems.append(
                        f"{retr_path.name}: {ds}@{e.get('task')} counts {ni}/{nc}, expected {exp[0]}/>={exp[1]}")
            for kk in ("i2t_r@1", "t2i_r@1"):
                v = e.get(kk)
                if v is None or not (0.0 <= float(v) <= 100.0):
                    problems.append(f"{retr_path.name}: {ds}@{e.get('task')} bad {kk}={v}")
        for ds, exp in EXPECTED_COUNTS.items():
            tasks = seen.get(ds, set())
            if len(tasks) != k:
                problems.append(f"{retr_path.name}: {ds} has {len(tasks)}/{k} task entries")

    # 5. 配置一致性：汇总 <name>.json 的 arguments 字典
    summary_path = wave_dir / f"{stem}.json"
    if summary_path.exists():
        sargs = json.load(open(summary_path)).get("arguments", {}) or {}
        if require_reference and sargs.get("reference_dataset") != "flickr30k_train_sub8k":
            problems.append(
                f"{summary_path.name}: reference_dataset={sargs.get('reference_dataset')!r}")
        if expected_seed is not None and int(sargs.get("seed", -1)) != expected_seed:
            problems.append(f"{summary_path.name}: arguments.seed={sargs.get('seed')} != {expected_seed}")
    else:
        problems.append(f"{summary_path.name}: missing")

    return ("FAIL" if problems else "PASS"), problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wave_dir", type=Path)
    ap.add_argument("--expected-k", type=int, default=10)
    ap.add_argument("--require-reference", action="store_true")
    args = ap.parse_args()

    wave_dir = args.wave_dir
    if not wave_dir.is_dir():
        print(f"DIR-MISSING {wave_dir}")
        return 2

    stems = sorted(p.name[:-len("_ens_results.json")]
                   for p in wave_dir.glob("*_ens_results.json"))
    # 也列出只有日志（进行中/失败）的 run
    log_only = sorted(p.name[:-4] for p in wave_dir.glob("*.log")
                      if not (wave_dir / f"{p.name[:-4]}_ens_results.json").exists())
    n_fail = 0
    for stem in stems:
        status, problems = check_run(stem, wave_dir, args.expected_k, args.require_reference)
        print(f"[{status}] {stem}")
        for pb in problems:
            print(f"    - {pb}")
        n_fail += (status == "FAIL")
    for stem in log_only:
        print(f"[RUNNING/INCOMPLETE] {stem}")
    if not stems and not log_only:
        print("(empty dir)")
        return 2
    print(f"summary: {len(stems)} completed ({n_fail} FAIL), {len(log_only)} running/incomplete")
    return 1 if n_fail else (2 if log_only and not stems else 0)


if __name__ == "__main__":
    sys.exit(main())
