#!/usr/bin/env python3
"""Parse main_incremental.py train logs and extract inline alpha-sensitivity sweep tables."""
import argparse
import os
import re
import sys


def parse_section(lines, start_idx):
    """Parse a section that starts with [Alpha Sensitivity Sweep] or [LADA+ZS Alpha Sensitivity Sweep]."""
    records = []
    # header line is next non-dash line after start
    i = start_idx + 1
    n = len(lines)
    # skip separator lines
    while i < n and ("---" in lines[i] or "===" in lines[i]):
        i += 1
    if i >= n:
        return records, i
    # header line
    i += 1
    # skip dash separator
    while i < n and ("---" in lines[i]):
        i += 1
    # data lines
    while i < n:
        line = lines[i].rstrip("\n")
        if not line.strip() or "---" in line or "===" in line or "Best alpha" in line or "[LADA" in line:
            i += 1
            continue
        # stop at next blank or marker
        if "Alpha" in line and "Transfer" in line and "Average" in line:
            i += 1
            continue
        m = re.search(r"(\d+\.\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)", line)
        if m:
            alpha, transfer, avg, last = map(float, m.groups())
            records.append({"alpha": alpha, "transfer": transfer, "average": avg, "last": last})
        i += 1
    return records, i


def parse_log(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    sections = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        if "[Alpha Sensitivity Sweep] Transfer / Average / Last per alpha" in line:
            recs, i = parse_section(lines, i)
            sections["lr_ensemble"] = recs
        elif "[LADA+ZS Alpha Sensitivity Sweep]" in line:
            recs, i = parse_section(lines, i)
            sections["lada_zs"] = recs
        else:
            i += 1
    return sections


def format_table(records, name, decimals=3):
    header = f"{Alpha:>8} | {Transfer:>10} | {Average:>10} | {Last:>10}"
    lines = [f"## {name}", header, "-" * 50]
    fmt = f"{{:8.2f}} | {{:10.{decimals}f}} | {{:10.{decimals}f}} | {{:10.{decimals}f}}"
    best = max(records, key=lambda r: r["average"]) if records else None
    for r in records:
        marker = " <- best" if r is best else ""
        lines.append(fmt.format(r["alpha"], r["transfer"], r["average"], r["last"]) + marker)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", nargs="+", help="One or more train logs")
    parser.add_argument("--decimals", type=int, default=3)
    parser.add_argument("--csv", action="store_true", help="Output CSV instead of markdown")
    args = parser.parse_args()

    all_data = {}
    for log in args.log:
        combo_name = os.path.splitext(os.path.basename(log))[0]
        sections = parse_log(log)
        all_data[combo_name] = sections

    combos = sorted(all_data.keys())
    for combo in combos:
        sections = all_data[combo]
        print(f"\n# {combo}")
        if "lr_ensemble" in sections and sections["lr_ensemble"]:
            print(format_table(sections["lr_ensemble"], "LR Ensemble alpha sweep", args.decimals))
        else:
            print("No LR Ensemble alpha sweep table found.")
        if "lada_zs" in sections and sections["lada_zs"]:
            print(format_table(sections["lada_zs"], "LADA+ZS alpha sweep", args.decimals))

    if args.csv:
        print("\n# Combined CSV")
        print("combo,method,alpha,transfer,average,last")
        for combo in combos:
            for method, recs in all_data[combo].items():
                for r in recs:
                    print(f"{combo},{method},{r[alpha]},{r[transfer]},{r[average]},{r[last]}")


if __name__ == "__main__":
    main()
