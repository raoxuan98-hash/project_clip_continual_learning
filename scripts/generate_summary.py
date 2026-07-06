import json, glob, os
out_dir = "experiments/optimizer_ablation"
ens = sorted(glob.glob(os.path.join(out_dir, "*_seed42_ens_results.json")))
zs = sorted(glob.glob(os.path.join(out_dir, "*_seed42_zs_results.json")))
rows = []
for epath in ens:
    base = os.path.basename(epath).replace("_ens_results.json", "")
    variant, opt = base.rsplit("_seed42", 1)[0].rsplit("_", 1)
    with open(epath) as f:
        ed = json.load(f)
    with open(os.path.join(out_dir, base + "_zs_results.json")) as f:
        zd = json.load(f)
    rows.append({
        "variant": variant,
        "optimizer": opt,
        "zero_shot_last": zd["metrics"]["last"],
        "ensemble_last": ed["metrics"]["last"]
    })
# save json
summary = {"rows": rows}
for variant in ["current_nsp", "hist_null_init_runtime"]:
    sub = [r for r in rows if r["variant"] == variant]
    best_zs = max(sub, key=lambda x: x["zero_shot_last"])
    best_ens = max(sub, key=lambda x: x["ensemble_last"])
    summary.setdefault("best", {})[variant] = {
        "zero_shot": {"optimizer": best_zs["optimizer"], "value": best_zs["zero_shot_last"]},
        "ensemble": {"optimizer": best_ens["optimizer"], "value": best_ens["ensemble_last"]}
    }
with open(os.path.join(out_dir, "summary_optimizer_ablation.json"), "w") as f:
    json.dump(summary, f, indent=2)

# save markdown
lines = ["# Optimizer Ablation Summary (LoRA)", "", "| Variant | Optimizer | ZS last | Ens last |", "|---|---|---|---|"]
for r in rows:
    lines.append("| %s | %s | %.2f | %.2f |" % (r["variant"], r["optimizer"], r["zero_shot_last"], r["ensemble_last"]))
lines.append("")
for variant, best in summary["best"].items():
    lines.append("- **%s best ZS**: %s (%.2f)" % (variant, best["zero_shot"]["optimizer"], best["zero_shot"]["value"]))
    lines.append("- **%s best Ens**: %s (%.2f)" % (variant, best["ensemble"]["optimizer"], best["ensemble"]["value"]))
lines.append("")
with open(os.path.join(out_dir, "summary_optimizer_ablation.md"), "w") as f:
    f.write("\n".join(lines))
print("summary files written.")
