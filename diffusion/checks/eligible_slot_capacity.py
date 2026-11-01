"""Eligible-slot capacity: at each cover's own K, how many of its raw slots hold an eligible coefficient (|q| >= 2).
Reads the stored corpus covers and cover_stats.csv only; writes results/eligible_slot_capacity.csv."""
import os, sys
import numpy as np, pandas as pd, cv2
from scipy.stats import wilcoxon
sys.path.insert(0, r"F:\DEDS_extra_checks")
import common as C

D = C.load_v5()
cs = pd.read_csv(r"F:\scirep results\09_expansion_1000\analysis1000\cover_stats.csv")
folder = {"base": "baseline", "ft": "finetuned"}
cache, rows = {}, []
for _, r in cs.iterrows():
    K = int(r.K)
    if K not in cache:
        cache[K] = D._slot_arrays(D._build_slot_order(K, 32, 32))
    I, J, U, V, Q = cache[K]
    blocks, _, _ = D._to_blocks(C.luma(cv2.imread(os.path.join(C.CORP, folder[r.set], r.image))))
    elig = int((np.abs(np.round(blocks[I, J, U, V] / Q)) >= 2).sum())
    assert len(I) == r.capacity_bits, (r.image, len(I), r.capacity_bits)
    rows.append(dict(image=r.image, set=r.set, fallback=r.fallback, raw_bits=len(I), eligible_bits=elig))
d = pd.DataFrame(rows); d["raw_bpp"] = d.raw_bits / 65536; d["elig_bpp"] = d.eligible_bits / 65536; d["share"] = d.eligible_bits / d.raw_bits
os.makedirs(r"F:\DEDS_extra_checks\results", exist_ok=True)
d.to_csv(r"F:\DEDS_extra_checks\results\eligible_slot_capacity.csv", index=False)

b = d[d.set == "base"].set_index("image"); f = d[d.set == "ft"].set_index("image")
keep = b.index[~b.fallback]
for name, idx in (("all 1,000 pairs", b.index), ("566 cover-derived pairs", keep)):
    print(f"{name}: raw bpp base {b.loc[idx].raw_bpp.mean():.3f} ft {f.loc[idx].raw_bpp.mean():.3f} | "
          f"eligible-slot bpp base {b.loc[idx].elig_bpp.mean():.3f} ft {f.loc[idx].elig_bpp.mean():.3f} | "
          f"eligible share base {b.loc[idx].share.mean():.3f} ft {f.loc[idx].share.mean():.3f}")
print("ft eligible-slot bpp: sd %.3f, range %.3f to %.3f; ft >= 0.30 eligible bpp: %d"
      % (f.elig_bpp.std(), f.elig_bpp.min(), f.elig_bpp.max(), (f.elig_bpp >= 0.30).sum()))
print("fallback baseline covers, eligible-slot bpp mean %.4f" % b[b.fallback].elig_bpp.mean())
print("ft > base eligible bpp on 566 pairs: %d, p = %.2g" % ((f.loc[keep].elig_bpp > b.loc[keep].elig_bpp).sum(),
      wilcoxon(f.loc[keep].elig_bpp, b.loc[keep].elig_bpp).pvalue))
