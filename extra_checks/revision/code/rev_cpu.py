r"""Revision analyses on the CPU (reviewer comments R1.2, R1.6, R2.1, R2.3, R2.5). Resumable; STOP file stops cleanly.
  offsets   (A8) within-bin offset distribution of the slots used at matched payload, by cover type and |q| class,
                 with the first-write distortion against Eq. 14 for each class
  split     (A4) 2x2 of the write rule: margin 0.5 or 0.15, with or without the |q'|>=2 preference, PSNR_Y after the
                 first write (pixels rounded), after the verify-fix loop and after colour repair (final image)
  rd        (A1, A2) rate-distortion sweep: each cover embedded with its full slot capacity at K = 1, 3, ..., 37, 38,
                 PNG round trip, exact recovery and PSNR_Y; equal-quality capacity is read from these curves
usage: python rev_cpu.py offsets|split|rd"""
import os, sys, io, glob, random, string, contextlib
import numpy as np, pandas as pd, cv2
sys.path.insert(0, r"F:\DEDS_extra_checks")
import common as C

OUT = r"F:\DEDS_extra_checks\results"
ALPHABET = string.ascii_letters + string.digits + " .,!?"
PAIRS = list(range(0, 1000, 5))
KGRID = list(range(1, 38, 2)) + [38]
D = None


def _init():
    global D
    cv2.setNumThreads(1)
    os.makedirs(os.path.join(OUT, "rev_tmp"), exist_ok=True)
    D = C.load_v5()


def text_bits(n_bits, seed):
    msg = "".join(random.Random(seed).choice(ALPHABET) for _ in range(n_bits // 8)).encode()
    return [int(b) for byte in msg for b in f"{byte:08b}"]


def psnr(a, b):
    m = ((a.astype(float) - b.astype(float)) ** 2).mean()
    return float("inf") if m == 0 else 10 * np.log10(255 ** 2 / m)


# ---------------------------------------------------------------- A8 offsets
def offsets():
    rows, mt = [], C.matched_tasks()
    D0 = C.load_v5()
    for i in PAIRS:
        K, n = mt[i]
        I, J, U, V, Q = D0._slot_arrays(D0._build_slot_order(K, 32, 32)[:n])
        bits = np.array(text_bits(n, 820000 + i))
        for cs in ("finetuned", "baseline"):
            blocks, _, _ = D0._to_blocks(C.luma(cv2.imread(os.path.join(C.CORP, cs, f"{i:03d}.png"))))
            c = blocks[I, J, U, V]; q = np.round(c / Q); f = c / Q - q
            new = np.array([D0._parity_write(cc, u, v, int(b), 0.15) for cc, u, v, b in zip(c, U, V, bits)])
            e2 = ((new - c) / Q) ** 2
            for cls, m in (("|q|<=1", np.abs(q) <= 1), ("|q|>=2", np.abs(q) >= 2)):
                if m.sum():
                    h, _ = np.histogram(f[m], bins=10, range=(-0.5, 0.5))
                    rows.append(dict(pair=i, covers=cs, cls=cls, n=int(m.sum()), share=float(m.mean()),
                                     ks=float(np.max(np.abs(np.cumsum(h) / m.sum() - np.arange(1, 11) / 10))),
                                     e2=float(e2[m].mean()), **{f"h{k}": int(x) for k, x in enumerate(h)}))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "rev_offsets.csv"), index=False)
    print("offsets done", len(rows))


# ---------------------------------------------------------------- A4 gain split
def split_work(task):
    key, i, cs, margin, pref = task
    K, n = TASKS[i]
    bits = text_bits(n, 720000 + i)
    cover = cv2.imread(os.path.join(C.CORP, cs, f"{i:03d}.png"))
    D.PAYLOAD_MARGIN = margin; D.PREFER_ELIGIBLE = pref
    Y, Cr, Cb = cv2.split(cv2.cvtColor(cover, cv2.COLOR_BGR2YCrCb))
    blocks, h8, w8 = D._to_blocks(Y)
    header = D._pack_header(n, K); D._write_header(blocks, n, K)
    slots = D._build_slot_order(K, 32, 32)
    for k in range(n):
        a, b, u, v = slots[k]
        blocks[a, b, u, v] = D._parity_write(blocks[a, b, u, v], u, v, bits[k])
    y1 = D._from_blocks(blocks, Y, h8, w8)
    first = psnr(Y, y1)
    y2, state = y1, {}
    for it in range(D.MAX_FIX_ITERATIONS):
        y2, rem = D._verify_and_fix(slots, y2, h8, w8, header, bits, K, state)
        if rem == 0:
            break
    fixed = psnr(Y, y2)
    with contextlib.redirect_stdout(io.StringIO()):
        out = D._exact_y_repair(cover, y2, Cr, Cb, header, bits, slots)
    st = C.png_roundtrip(out, os.path.join(OUT, "rev_tmp", f"{key.replace(':', '_')}.png"))
    got = C.read_bits(D, st)
    exact = got is not None and list(got) == bits
    return [dict(key=key, pair=i, covers=cs, margin=margin, prefer=pref, K=K, n_bits=n, psnr_first=first,
                 psnr_fixed=fixed, psnr_final=psnr(Y, C.luma(st)), exact=bool(exact), fix_passes=it + 1)]


# ---------------------------------------------------------------- A1, A2 rate-distortion
def cover_sets():
    s = {}
    for i in PAIRS:
        s[f"finetuned:{i}"] = ("finetuned", os.path.join(C.CORP, "finetuned", f"{i:03d}.png"))
        s[f"baseline:{i}"] = ("baseline", os.path.join(C.CORP, "baseline", f"{i:03d}.png"))
        s[f"sharpen4:{i}"] = ("sharpen4", ("sharpen", i))
        s[f"noise8:{i}"] = ("noise8", ("noise", i))
    for p in sorted(glob.glob(os.path.join(C.PHOTOS, "*.png")))[:195]:
        s[f"photo:{os.path.basename(p)}"] = ("photo", p)
    dtd = sorted(glob.glob(r"F:\DEDS_extra_checks\test5_textures\covers\*.png"))
    for p in dtd[::28][:200]:
        s[f"dtd:{os.path.basename(p)}"] = ("dtd", p)
    return s


def load_cover(src):
    if isinstance(src, str):
        return cv2.imread(src)
    kind, i = src
    x = cv2.imread(os.path.join(C.CORP, "baseline", f"{i:03d}.png")).astype(np.float64)
    if kind == "sharpen":     # x + a (x - G1 * x), a = 4, Gaussian of s.d. one pixel on every channel (Methods)
        y = x + 4.0 * (x - cv2.GaussianBlur(x, (0, 0), 1.0))
    else:                     # Gaussian noise, s.d. 8 grey levels, fixed seed per image
        y = x + np.random.default_rng(910000 + i).normal(0, 8, x.shape)
    return np.clip(np.round(y), 0, 255).astype(np.uint8)


def rd_work(task):
    key, kind, src = task
    cover = load_cover(src)
    st = C.cover_stats(D, cover)
    rows = []
    for K in KGRID:
        n = 8 * (1022 * (K + 1) // 8)
        bits = text_bits(n, 930000 + K)
        try:
            out, passes = C.embed_bits(D, cover, bits, K)
            out = C.png_roundtrip(out, os.path.join(OUT, "rev_tmp", f"{key.replace(':', '_')}.png"))
            got = C.read_bits(D, out)
            exact = got is not None and list(got) == bits
            qm = C.quality(cover, out); p, s = qm["psnr_y"], qm["ssim_y"]; err = ""
        except Exception as e:
            exact, p, s, passes, err = False, float("nan"), float("nan"), -1, repr(e)[:80]
        rows.append(dict(key=f"{key}:K{K}", cover=key, kind=kind, e_bar=st["e_bar"], K=K, n_bits=n, bpp=n / 65536,
                         psnr_y=p, ssim_y=s, exact=bool(exact), passes=passes, error=err))
    return rows


TASKS = C.matched_tasks()

if __name__ == "__main__":
    os.makedirs(os.path.join(OUT, "rev_tmp"), exist_ok=True)
    what = sys.argv[1]
    if what == "offsets":
        offsets()
    elif what == "split":
        tasks = [(f"{cs}:{i}:{m}:{int(p)}", i, cs, m, p) for i in PAIRS for cs in ("finetuned", "baseline")
                 for m in (0.5, 0.15) for p in (True, False)]
        sys.exit(0 if C.run_resumable(tasks, split_work, os.path.join(OUT, "rev_gain_split.csv"), init=_init, workers=3) else 1)
    elif what == "rd":
        tasks = [(k, kind, src) for k, (kind, src) in cover_sets().items()]
        # run_resumable skips by the first element; rows are keyed per K, so record the cover key as done via 'cover'
        sys.exit(0 if C.run_resumable(tasks, rd_work, os.path.join(OUT, "rev_rd.csv"), init=_init, workers=3, key="cover") else 1)
