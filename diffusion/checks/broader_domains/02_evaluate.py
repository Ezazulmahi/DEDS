"""Test 4, step 2: cover statistics and matched-payload embedding for the 300 broader-domain pairs.
Same protocol as the unseen-prompt check (Supplementary Table S7): a target payload drawn uniformly from 0.30 to
0.60 bpp per pair, K set from the target and forced on both covers, the same random 67-character ASCII message on
both, PNG round trip and byte-for-byte comparison. Also the eligible-slot capacity at each cover's own K.
Resumable (results/test4_domains.csv, one row per cover); a STOP file stops it cleanly."""
import os, sys, io, math, random, string, contextlib
import numpy as np, cv2
sys.path.insert(0, r"F:\DEDS_extra_checks")
import common as C

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(r"F:\DEDS_extra_checks\results", "test4_domains.csv")
ALPHABET = string.ascii_letters + string.digits + " .,!?"
D = None


def _init():
    global D
    cv2.setNumThreads(1)
    D = C.load_v5()


def eligible_slot_bits(bgr, K):
    I, J, U, V, Q = D._slot_arrays(D._build_slot_order(K, 32, 32))
    blocks, _, _ = D._to_blocks(C.luma(bgr))
    return int((np.abs(np.round(blocks[I, J, U, V] / Q)) >= 2).sum())


def work(task):
    key, i = task
    rng = random.Random(310000 + i)
    target = rng.uniform(0.30, 0.60)
    n_bits = 8 * int(target * 65536 / 8)
    K = math.ceil(n_bits / 1022) - 1
    msg = "".join(random.Random(320000 + i).choice(ALPHABET) for _ in range(n_bits // 8))
    rows = []
    for folder in ("finetuned", "baseline"):
        cover = cv2.imread(os.path.join(HERE, folder, f"{i:03d}.png"))
        st = C.cover_stats(D, cover)
        own_K = st["K"] if st["K"] is not None else D.FALLBACK_K
        row = dict(key=f"{folder}:{i}", index=i, covers=folder, e_bar=st["e_bar"], own_K=own_K, fallback=st["K"] is None,
                   raw_bpp=1022 * (own_K + 1) / 65536, elig_slot_bpp=eligible_slot_bits(cover, own_K) / 65536,
                   **C.spectrum_stats(cover), forced_K=K, n_bits=n_bits, bpp=n_bits / 65536)
        o = D._compute_K; D._compute_K = lambda b, K=K: K
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                stego = D.embed(cover, msg)
            err = None
        except Exception as e:
            stego, err = None, repr(e)
        finally:
            D._compute_K = o
        row["error"] = err
        if stego is not None:
            p = os.path.join(HERE, f"stego_{folder}", f"{i:03d}.png")
            stego = C.png_roundtrip(stego, p)
            with contextlib.redirect_stdout(io.StringIO()):
                row["exact"] = D.extract(stego) == msg
            row.update(C.quality(cover, stego))
        rows.append(row)
    return rows


if __name__ == "__main__":
    for d in ("stego_finetuned", "stego_baseline"):
        os.makedirs(os.path.join(HERE, d), exist_ok=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    ok = C.run_resumable([(f"finetuned:{i}", i) for i in range(300)], work, OUT, init=_init, workers=3)
    sys.exit(0 if ok else 1)
