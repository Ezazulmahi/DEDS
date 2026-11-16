"""Fig. 2 with larger text (reviewer 1, comment 9): same data and layout as the Fig. 3 block of
F:\\Dataset_v5_build\\cover_statistics\\make_figs.py, but the cell values of panels a and b at 6.4 pt (were 4.6) and the
rest of the text 1 to 1.5 pt larger. Reads cover_stats.csv, elig_maps.npz and psd.npz there; writes only to
F:\\DEDS_extra_checks\\revision\\fig2_rev."""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

SRC = r"F:\Dataset_v5_build\cover_statistics"
OUT = r"F:\DEDS_extra_checks\revision\fig2_rev"; os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, SRC); os.chdir(SRC)
import common as C

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.linewidth": 0.6, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
BLUE, ORANGE, AQUA, INK, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#30363d", "#d9dde1"
TS, LS, CELL = 9.5, 7.5, 6.4
cs = pd.read_csv("cover_stats.csv"); b = cs[cs.set == "base"].set_index("image"); f = cs[cs.set == "ft"].set_index("image")
m = np.load("elig_maps.npz"); p = np.load("psd.npz")

fig = plt.figure(figsize=(7.4, 5.2)); gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.25], hspace=0.55, wspace=0.5)
for k, (key, ttl) in enumerate((("base", "baseline covers"), ("ft", "fine-tuned covers"))):
    ax = fig.add_subplot(gs[0, k]); ax.imshow(m[key], cmap="Blues", vmin=0, vmax=0.85)
    for u in range(8):
        for v in range(8):
            if C.STABLE[u, v]:
                ax.text(v, u, f"{m[key][u, v]:.2f}"[1:], ha="center", va="center", fontsize=CELL,
                        color="white" if m[key][u, v] > 0.45 else INK)
            else:
                ax.add_patch(Rectangle((v - .5, u - .5), 1, 1, fc="#eeeeec", ec="none"))
    ax.set_xticks(range(8)); ax.set_yticks(range(8)); ax.tick_params(length=0, labelsize=LS)
    ax.set_xlabel("horizontal frequency index $v$"); ax.set_ylabel("vertical frequency index $u$")
    ax.set_title(("a  " if k == 0 else "b  ") + ttl, loc="left", fontsize=TS, fontweight="bold")
    for s in ax.spines.values():
        s.set_visible(False)
ax = fig.add_subplot(gs[0, 2]); bins = np.linspace(0, 24, 49)
ax.hist(b.mean_eligible, bins=bins, color=BLUE, alpha=0.85, label=f"baseline (mean {b.mean_eligible.mean():.2f})")
ax.hist(f.mean_eligible, bins=bins, color=ORANGE, alpha=0.85, label=f"fine-tuned (mean {f.mean_eligible.mean():.2f})")
ax.set_xlabel(r"eligible coefficients per block, $\bar e$"); ax.set_ylabel("covers (of 1,000)")
ax.legend(fontsize=LS, loc="upper center"); ax.tick_params(labelsize=LS); ax.set_ylim(0, ax.get_ylim()[1] * 1.35)
ax.set_title("c  eligible coefficients", loc="left", fontsize=TS, fontweight="bold"); ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
ax = fig.add_subplot(gs[1, :2]); fr = np.arange(1, 128) / 256.0
for arr, col, lab, ls in ((p["pb"], BLUE, "baseline SD v1.5 covers (n = 1,000)", "-"), (p["pf"], ORANGE, "fine-tuned covers (n = 1,000)", "-"),
                          (p["pn"], AQUA, "photographs, Lorem Picsum (n = 195)", "--")):
    lg = np.log10(arr); ax.plot(fr, lg.mean(0), color=col, lw=1.4, ls=ls, label=lab)
    ax.fill_between(fr, np.percentile(lg, 25, 0), np.percentile(lg, 75, 0), color=col, alpha=0.15, lw=0)
ax.set_xscale("log"); ax.set_xlabel("spatial frequency (cycles per pixel)"); ax.set_ylabel(r"$\log_{10}$ radially averaged power of $Y$")
ax.legend(fontsize=LS, loc="lower left"); ax.grid(color=GRID, lw=0.5); ax.tick_params(labelsize=LS)
ax.set_title("d  luminance power spectrum", loc="left", fontsize=TS, fontweight="bold")
ax = fig.add_subplot(gs[1, 2])
ax.hist(b.capacity_bpp[~b.fallback], bins=np.linspace(0, 0.6, 31), color=BLUE, alpha=0.85, label="baseline (n = %d)" % (~b.fallback).sum())
ax.hist(f.capacity_bpp, bins=np.linspace(0, 0.6, 31), color=ORANGE, alpha=0.85, label="fine-tuned (n = 1,000)")
ax.set_xlabel("raw slot capacity (bits per pixel)"); ax.set_ylabel("number of covers"); ax.set_ylim(0, ax.get_ylim()[1] * 1.4); ax.legend(fontsize=LS, loc="upper center")
ax.tick_params(labelsize=LS)
ax.set_title("e  adaptive capacity", loc="left", fontsize=TS, fontweight="bold"); ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
fig.savefig(os.path.join(OUT, "fig2_dct_statistics.pdf"), bbox_inches="tight", pad_inches=0.25)
fig.savefig(os.path.join(OUT, "fig2_dct_statistics.png"), dpi=150, bbox_inches="tight", pad_inches=0.25)
print("saved", OUT)
