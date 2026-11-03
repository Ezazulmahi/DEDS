r"""Test 5: texture-matched real photographs (Describable Textures Dataset, 5,640 photos in 47 texture categories).
Step a, prepare: centre crop to the shorter side, resize to 256x256 (area interpolation), store as PNG in covers/.
Step b, stats: eligible count, cover-derived K, raw and eligible-slot capacity, spectra for every photo.
Step c, matched payload: 1,000 photos (stratified over the 47 categories, seeded) each embedded with the K and
payload of one corpus pair j, and the fine-tuned corpus cover j embedded with the same K and the same message, so
the two are compared pair by pair. Resumable CSVs in F:\DEDS_extra_checks\results; a STOP file stops cleanly.
The downloaded images are only read with OpenCV; no code from the download is run."""
import os, sys, io, math, random, string, contextlib, glob
import numpy as np, pandas as pd, cv2
sys.path.insert(0, r"F:\DEDS_extra_checks")
import common as C

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "data", "images")
COV = os.path.join(HERE, "covers")
RES = r"F:\DEDS_extra_checks\results"
STATS = os.path.join(RES, "test5_texture_stats.csv")
EMB = os.path.join(RES, "test5_texture_embed.csv")
ALPHABET = string.ascii_letters + string.digits + " .,!?"
D = None


def _init():
    global D
    cv2.setNumThreads(1)
    D = C.load_v5()


def names():
    return sorted(os.path.relpath(p, SRC).replace("\\", "/") for p in glob.glob(os.path.join(SRC, "*", "*.jpg")))


def prepare(rel):
    out = os.path.join(COV, rel.replace("/", "__").replace(".jpg", ".png"))
    if not os.path.exists(out):
        im = cv2.imread(os.path.join(SRC, rel))
        h, w = im.shape[:2]; s = min(h, w)
        im = im[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
        cv2.imwrite(out, cv2.resize(im, (256, 256), interpolation=cv2.INTER_AREA))
    return out


def elig_slot_bits(bgr, K):
    I, J, U, V, Q = D._slot_arrays(D._build_slot_order(K, 32, 32))
    blocks, _, _ = D._to_blocks(C.luma(bgr))
    return int((np.abs(np.round(blocks[I, J, U, V] / Q)) >= 2).sum())


def stats_work(task):
    rel = task[0]
    img = cv2.imread(prepare(rel))
    st = C.cover_stats(D, img)
    K = st["K"] if st["K"] is not None else D.FALLBACK_K
    return [dict(key=rel, category=rel.split("/")[0], e_bar=st["e_bar"], own_K=K, fallback=st["K"] is None,
                 raw_bpp=1022 * (K + 1) / 65536, elig_slot_bpp=elig_slot_bits(img, K) / 65536,
                 cover_derived_ge030=(st["K"] is not None and st["capacity_bpp"] >= 0.30), **C.spectrum_stats(img))]


def embed_one(cover, msg, K):
    o = D._compute_K; D._compute_K = lambda b, K=K: K
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            st = D.embed(cover, msg)
        return st, None
    except Exception as e:
        return None, repr(e)
    finally:
        D._compute_K = o


def embed_work(task):
    key, j, rel = task
    K, n_bits = TASKS[j]
    msg = "".join(random.Random(520000 + j).choice(ALPHABET) for _ in range(n_bits // 8))
    rows = []
    for kind, path in (("photo", os.path.join(COV, rel.replace("/", "__").replace(".jpg", ".png"))),
                       ("finetuned", os.path.join(C.CORP, "finetuned", f"{j:03d}.png"))):
        cover = cv2.imread(path)
        stego, err = embed_one(cover, msg, K)
        row = dict(key=f"{j}:{kind}", pair=j, kind=kind, image=rel if kind == "photo" else f"{j:03d}.png",
                   K=K, n_bits=n_bits, bpp=n_bits / 65536, error=err)
        if stego is not None:
            stego = C.png_roundtrip(stego, os.path.join(HERE, "stego", f"{j:03d}_{kind}.png"))
            with contextlib.redirect_stdout(io.StringIO()):
                row["exact"] = D.extract(stego) == msg
            row.update(C.quality(cover, stego))
        rows.append(row)
    return rows


TASKS = C.matched_tasks()

if __name__ == "__main__":
    for d in (COV, os.path.join(HERE, "stego"), RES):
        os.makedirs(d, exist_ok=True)
    allnames = names()
    print("photos found:", len(allnames), flush=True)
    assert len(allnames) == 5640, len(allnames)
    if not C.run_resumable([(n,) for n in allnames], stats_work, STATS, init=_init, workers=3):
        sys.exit(1)
    # 1,000 photos stratified over the 47 categories (seeded), paired with corpus pairs 0..999 in order
    by_cat = {}
    for n in allnames:
        by_cat.setdefault(n.split("/")[0], []).append(n)
    rng = random.Random(2026)
    for v in by_cat.values():
        rng.shuffle(v)
    pick, k = [], 0
    while len(pick) < 1000:
        for c in sorted(by_cat):
            if k < len(by_cat[c]) and len(pick) < 1000:
                pick.append(by_cat[c][k])
        k += 1
    ok = C.run_resumable([(f"{j}:photo", j, rel) for j, rel in enumerate(pick)], embed_work, EMB, init=_init, workers=3)
    sys.exit(0 if ok else 1)
