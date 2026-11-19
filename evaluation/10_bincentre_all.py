"""Bin-centre reference rule (margin 1/2 with the |q'| >= 2 preference) on all 1,000 corpus pairs, with exactly the
same K, payload and message as the corrected margin-projected run (01_embed_normal_1000.py, message seed 424242 + i),
so that Table 2's comparison columns use one message per pair for both rules. Resumable.
Output: results/bincentre_per_image.csv"""
import os, sys, io, time, random, contextlib, importlib.util
from multiprocessing import Pool
import numpy as np, pandas as pd, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "bincentre_per_image.csv")
spec = importlib.util.spec_from_file_location("E", os.path.join(HERE, "01_embed_normal_1000.py"))
E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)
D = None


def _init():
    global D
    E._init(); D = E.D
    D.PAYLOAD_MARGIN = 0.5; D.PREFER_ELIGIBLE = True


def work(task):
    i, K, n_bits = task
    name = f"{i:03d}.png"
    msg = (lambda _rng: "".join(_rng.choice(E.ALPHABET) for _ in range(n_bits // 8)))(random.Random(424242 + i))
    rows = []
    for cover_set in ("finetuned", "baseline"):
        cover = cv2.imread(os.path.join(E.CORP, cover_set, name))
        orig = D._compute_K; D._compute_K = lambda blocks, K=K: K
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                stego = D.embed(cover, msg, mode="normal")
        finally:
            D._compute_K = orig
        ok, buf = cv2.imencode(".png", stego); stego = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        q = D.quality_metrics(cover, stego, n_bits)
        with contextlib.redirect_stdout(io.StringIO()):
            exact = D.extract(stego) == msg
        rows.append(dict(image=name, covers=cover_set, K=K, n_bits=n_bits, psnr_y=q["psnr_luma"], ssim_y=q["ssim_luma"], exact=exact))
    return rows


if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    c = pd.read_excel(os.path.join(E.CORP, "comparison_results.xlsx"), sheet_name="per_image_results")
    done = set(pd.read_csv(OUT).image) if os.path.exists(OUT) else set()
    tasks = [(int(r.image[:3]), int(r.K), int(r.n_bits)) for r in c.itertuples() if r.image not in done]
    print(f"{len(tasks)} pairs to do", flush=True)
    with Pool(workers, initializer=_init) as p:
        for k, rows in enumerate(p.imap_unordered(work, tasks), 1):
            pd.DataFrame(rows).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
            if k % 50 == 0:
                print(f"{k}/{len(tasks)}", flush=True)
    print("done", flush=True)
