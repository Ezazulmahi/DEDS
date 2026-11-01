"""Test 4, step 3: summary of the broader-domain check, overall and per domain."""
import pandas as pd
from scipy.stats import wilcoxon

r = pd.read_csv(r"F:\DEDS_extra_checks\results\test4_domains.csv")
dom = pd.read_csv(r"F:\DEDS_extra_checks\test4_domains\prompts.tsv", sep="\t").set_index("index").domain
r["domain"] = r["index"].map(dom)
b = r[r.covers == "baseline"].set_index("index").sort_index(); f = r[r.covers == "finetuned"].set_index("index").sort_index()
print("rows", len(r), "| errors", r.error.notna().sum(), "| exact ft/base", int(f.exact.sum()), int(b.exact.sum()))
p = lambda a, c: wilcoxon(a, c).pvalue


def line(name, B, F):
    return (f"{name:13s} n={len(F):3d} | e_bar {B.e_bar.mean():5.2f} -> {F.e_bar.mean():5.2f} (ft higher {int((F.e_bar > B.e_bar).sum())}) | "
            f"fallback {int(B.fallback.sum()):3d}/{int(F.fallback.sum())} | elig-slot bpp {B.elig_slot_bpp.mean():.3f} -> {F.elig_slot_bpp.mean():.3f} | "
            f"PSNR {B.psnr_y.mean():5.2f} -> {F.psnr_y.mean():5.2f} (ft higher {int((F.psnr_y > B.psnr_y).sum())}) | "
            f"SSIM {B.ssim_y.mean():.3f} -> {F.ssim_y.mean():.3f} | HF% {100*B.hf_share.median():.2f} -> {100*F.hf_share.median():.2f}")


print(line("ALL", b, f))
for d in dom.unique():
    idx = dom[dom == d].index
    print(line(d, b.loc[idx], f.loc[idx]))
cd = b.index[~b.fallback]
print(f"cover-derived raw bpp on {len(cd)} pairs: base {b.loc[cd].raw_bpp.mean():.3f} ft {f.loc[cd].raw_bpp.mean():.3f}")
print(f"ft cover-derived capacity >= 0.30 bpp: {int(((~f.fallback) & (f.raw_bpp >= 0.30)).sum())} of 300")
print(f"p: e_bar {p(f.e_bar, b.e_bar):.2g}, PSNR {p(f.psnr_y, b.psnr_y):.2g}, SSIM {p(f.ssim_y, b.ssim_y):.2g}, elig {p(f.elig_slot_bpp, b.elig_slot_bpp):.2g}")
print(f"payload mean {f.bpp.mean():.3f}, PSNR ft range {f.psnr_y.min():.1f} to {f.psnr_y.max():.1f}")
