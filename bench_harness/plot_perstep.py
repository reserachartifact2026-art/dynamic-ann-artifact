#!/usr/bin/env python3
import csv, os, sys
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV, OUT = sys.argv[1], sys.argv[2]
data = defaultdict(lambda: defaultdict(list))
with open(CSV) as f:
    r = csv.DictReader(f)
    for row in r:
        key = (row["dataset"], row["workload"])
        data[key]["step"].append(int(row["search_step"]))
        for c in r.fieldnames[3:]:
            v = row.get(c, "")
            data[key][c].append(float(v) if v not in ("", "None", None) else None)

SERIES = [("fda", "C0"), ("fda2", "C1"), ("svf", "C2")]

def plot(key, colfmt, ylabel, title, fname):
    x = data[key]["step"]
    plt.figure(figsize=(8, 4)); any_ = False
    for sysn, color in SERIES:
        y = data[key].get(colfmt % sysn, [])
        xs = [x[i] for i in range(len(y)) if y[i] is not None]
        ys = [v for v in y if v is not None]
        if ys:
            plt.plot(xs, ys, label=sysn, color=color, marker=".", ms=3, lw=1); any_ = True
    if not any_:
        plt.close(); return 0
    plt.xlabel("search step"); plt.ylabel(ylabel); plt.title(title)
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(fname, dpi=90); plt.close(); return 1

n = 0
for (ds, wl) in sorted(data):
    d = os.path.join(OUT, ds); os.makedirs(d, exist_ok=True)
    n += plot((ds, wl), "%s_recall", "recall@10 (%)", f"{ds} / {wl} — recall", os.path.join(d, f"{wl}_recall.png"))
    n += plot((ds, wl), "%s_qps",    "search QPS",    f"{ds} / {wl} — search throughput", os.path.join(d, f"{wl}_qps.png"))
    n += plot((ds, wl), "%s_ins/s",  "insert vec/s",  f"{ds} / {wl} — insert throughput", os.path.join(d, f"{wl}_insert.png"))
print(f"wrote {n} PNGs under {OUT}")
