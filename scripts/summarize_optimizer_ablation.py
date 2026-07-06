import json, glob, os, sys
out_dir = "experiments/optimizer_ablation"
all_ens = sorted(glob.glob(os.path.join(out_dir, "*_seed42_ens_results.json")))
all_zs = sorted(glob.glob(os.path.join(out_dir, "*_seed42_zs_results.json")))
print("Found %d ens + %d zs result files (need 10 each)" % (len(all_ens), len(all_zs)))

def stem(p):
    return os.path.basename(p).replace("_ens_results.json","").replace("_zs_results.json","")

stems = sorted(set(stem(p) for p in all_ens + all_zs))
rows = []
for s in stems:
    variant, opt = s.rsplit("_seed42", 1)[0].rsplit("_", 1)
    opt = opt.replace("_seed42","")
    try:
        zs = json.load(open(os.path.join(out_dir, s + "_zs_results.json")))
        ens = json.load(open(os.path.join(out_dir, s + "_ens_results.json")))
        zs_last = zs["metrics"]["last"]
        ens_last = ens["metrics"]["last"]
        rows.append((variant, opt, zs_last, ens_last))
    except Exception as e:
        print("skip %s: %s" % (s, e))

print("\n%-25s %-12s %12s %12s" % ("variant", "optimizer", "ZS last", "Ens last"))
print("-" * 65)
for variant, opt, zs, ens in rows:
    print("%-25s %-12s %12.2f %12.2f" % (variant, opt, zs, ens))

for variant in ["current_nsp", "hist_null_init_runtime"]:
    sub = [r for r in rows if r[0] == variant]
    if sub:
        best_zs = max(sub, key=lambda x: x[2])
        best_ens = max(sub, key=lambda x: x[3])
        print("\n%s best ZS:  %s (%.2f)" % (variant, best_zs[1], best_zs[2]))
        print("%s best Ens: %s (%.2f)" % (variant, best_ens[1], best_ens[3]))
