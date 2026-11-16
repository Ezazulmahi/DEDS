r"""Capacity at equal quality (Supplementary Table S22) from the rate-distortion sweep of rev_cpu.py (rev_rd.csv).
For each cover and criterion, the capacity is the largest grid payload whose stego image met the criterion after the
PNG round trip: PSNR_Y >= 38 or 40 dB, SSIM_Y >= 0.98 or 0.99. Covers that meet a criterion at no grid point get 0.
Writes rev_capacity_equal_quality.csv and prints the table and the paired comparisons with the fine-tuned covers."""
import os
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

R = r"F:\DEDS_extra_checks\results"
CRIT = {"P38": ("psnr_y", 38.0), "P40": ("psnr_y", 40.0), "S0.98": ("ssim_y", 0.98), "S0.99": ("ssim_y", 0.99)}

rd = pd.read_csv(os.path.join(R, "rev_rd.csv"))
assert rd.exact.all(), "an embedding was not recovered exactly"
rows = []
for (cover, kind), g in rd.groupby(["cover", "kind"], sort=False):
    r = {"cover": cover, "kind": kind}
    for name, (col, thr) in CRIT.items():
        ok = g[g[col] >= thr]
        r[name] = float(ok.bpp.max()) if len(ok) else 0.0
    rows.append(r)
t = pd.DataFrame(rows)
t.to_csv(os.path.join(R, "rev_capacity_equal_quality.csv"), index=False)
print(t.groupby("kind")[list(CRIT)].mean().round(3))
t["i"] = t.cover.str.split(":").str[1]
f = t[t.kind == "finetuned"].set_index("i")
for k in ("baseline", "sharpen4", "noise8"):
    o = t[t.kind == k].set_index("i"); idx = f.index.intersection(o.index)
    for c in CRIT:
        a, b = f.loc[idx, c], o.loc[idx, c]
        print(f"finetuned vs {k} {c}: more {(a > b).sum()} same {(a == b).sum()} less {(a < b).sum()} "
              f"({(a.mean() / b.mean() - 1) * 100:+.0f}%, p={wilcoxon(a, b).pvalue:.1e})")
