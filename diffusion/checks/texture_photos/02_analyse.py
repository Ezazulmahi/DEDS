"""Test 5 summary: DTD texture photographs against the corpus fine-tuned and baseline covers."""
import pandas as pd
from scipy.stats import wilcoxon

R = r"F:\DEDS_extra_checks\results"
s = pd.read_csv(fr"{R}\test5_texture_stats.csv")
cs = pd.read_csv(r"F:\scirep results\09_expansion_1000\analysis1000\cover_stats.csv")
el = pd.read_csv(fr"{R}\eligible_slot_capacity.csv")
print(f"DTD photos n={len(s)}: e_bar mean {s.e_bar.mean():.2f} median {s.e_bar.median():.2f} (max {s.e_bar.max():.2f}) | "
      f"fallback {int(s.fallback.sum())} | cover-derived >= 0.30 bpp: {int(s.cover_derived_ge030.sum())} | "
      f"eligible-slot bpp {s.elig_slot_bpp.mean():.3f} | HF% median {100*s.hf_share.median():.2f} | slope {s.slope.median():.2f}")
for name, k in (("fine-tuned corpus", "ft"), ("baseline corpus", "base")):
    c = cs[cs.set == k]
    print(f"{name}: e_bar mean {c.mean_eligible.mean():.2f} | eligible-slot bpp {el[el.set == k].elig_bpp.mean():.3f}")
print("share of DTD photos with e_bar above the fine-tuned corpus mean (16.64):", round((s.e_bar > 16.64).mean() * 100, 2), "%")
print("share of DTD photos with e_bar above the fine-tuned corpus minimum (6.74):", round((s.e_bar > 6.74).mean() * 100, 1), "%")
cat = s.groupby("category").e_bar.mean().sort_values()
print("lowest categories:", cat.head(3).round(2).to_dict())
print("highest categories:", cat.tail(3).round(2).to_dict())

e = pd.read_csv(fr"{R}\test5_texture_embed.csv")
print("embedding errors:", e.error.notna().sum(), "| exact:", e.groupby("kind").exact.sum().to_dict())
p = e.pivot(index="pair", columns="kind", values="psnr_y"); q = e.pivot(index="pair", columns="kind", values="ssim_y")
d = p.finetuned - p.photo
print(f"matched payload, 1,000 pairs: PSNR photo {p.photo.mean():.2f} | fine-tuned {p.finetuned.mean():.2f} | diff {d.mean():+.2f} dB "
      f"(ft higher in {int((d > 0).sum())}), p = {wilcoxon(p.finetuned, p.photo).pvalue:.2g}")
print(f"SSIM photo {q.photo.mean():.3f} | fine-tuned {q.finetuned.mean():.3f}")
