r"""Step 15 (corrected-message rerun, 19 Nov 2026): the "DEDS, corpus" rows of Table 4 again, on the 1,000 fine-tuned
stego images of the rerun (F:\DEDS_rerun\4_evaluation\stego_finetuned), which carry a fresh random message per image.
Same detectors, nothing retrained except the SRM+FLD ensemble, which is retrained exactly as in step 11 (seed 2026,
same features) because step 11 does not save it.
    1. Y planes (OpenCV BGR2YCrCb) -> matched_0.4\deds_v5_rerun\paper_ft\
    2. SRNet-KV (srnet_kv_matched-deds_v5m, best.pt, 8-view TTA) -> results\srnet_kv_matched-deds_v5m_tta_rerun\
    3. SRM features -> features_srm\v5rerun_paper_ft\ ; SRM+FLD (deds_v5m) -> results\srm_fld_rerun\
Resumable: finished steps are skipped.   python 15_rescore_rerun.py [--workers 4]
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import sys, argparse, importlib.util, concurrent.futures as cf
from pathlib import Path
import numpy as np, pandas as pd, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def mod(name):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), C.ROOT / f"{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


SRC = Path(r"F:\DEDS_rerun\4_evaluation\stego_finetuned")
YDIR = C.MATCHED / "deds_v5_rerun" / "paper_ft"
COV = C.MATCHED / "covers_y" / "paper_ft"
N = 1000


def step_y():
    YDIR.mkdir(parents=True, exist_ok=True)
    for i in range(N):
        p = YDIR / f"{i:03d}.png"
        if p.exists():
            continue
        im = cv2.imread(str(SRC / f"{i:03d}.png")); assert im is not None, i
        tmp = p.with_name("tmp_" + p.name); cv2.imwrite(str(tmp), cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb)[:, :, 0]); os.replace(tmp, p)
    print("Y planes ready", flush=True)


def step_srnet():
    import torch
    T = mod("03_test")
    out = C.RESULTS / "srnet_kv_matched-deds_v5m_tta_rerun"; out.mkdir(parents=True, exist_ok=True)
    f = out / "v5_paper_ft_scores.csv"
    ck = torch.load(C.RUNS / "srnet_kv_matched-deds_v5m" / "best.pt", map_location="cpu", weights_only=False)
    thr = ck.get("threshold", ck.get("best", {}).get("thr", 0.5))
    pairs = [(COV / f"{i:03d}.png", YDIR / f"{i:03d}.png") for i in range(N)]
    if not f.exists():
        dev = torch.device("cuda")
        model = C.build_model(ck["cfg"]["model"], pretrained=False); model.load_state_dict(ck["model"])
        model = model.to(dev).to(memory_format=torch.channels_last)
        cov, stg = C.load_pairs(pairs)
        s = T.scores(model, np.concatenate([cov, stg]), dev, True)
        df = pd.DataFrame({"image": [str(c) for c, _ in pairs] + [str(t) for _, t in pairs], "label": [0] * N + [1] * N, "margin": s})
        df["p_stego"] = 1 / (1 + np.exp(-np.clip(s, -50, 50))); df["flagged"] = (df.margin > thr).astype(int); df.to_csv(f, index=False)
    df = pd.read_csv(f)
    r = {"run": "srnet_kv_matched-deds_v5m_tta_rerun", "set": "v5_paper_ft", "missing_pairs": 0, **T.summarise(df, thr)}
    pd.DataFrame([r]).to_csv(out / "summary.csv", index=False)
    print(f"SRNet-KV  AUC {r['auc']:.4f} [{r['auc_ci_low']:.4f}, {r['auc_ci_high']:.4f}]  P_E val {1 - r['acc_at_val_thr']:.4f}  P_E min {r['pe_min']:.4f}", flush=True)


def step_srm(workers):
    S = __import__("10_srm_extract"); K = __import__("11_srm_classify")
    from sealwatch.ensemble_classifier import FldEnsembleTrainer
    from sklearn.metrics import roc_auc_score
    fd = S.OUT / "v5rerun_paper_ft"; fd.mkdir(parents=True, exist_ok=True)
    paths = [YDIR / f"{i:03d}.png" for i in range(N)]
    (S.OUT / "v5rerun_paper_ft.txt").write_text("\n".join(str(p) for p in paths))
    tasks = [(fd / f"chunk_{k:03d}.npy", paths[k * S.CHUNK:(k + 1) * S.CHUNK]) for k in range((N + S.CHUNK - 1) // S.CHUNK)]
    tasks = [t for t in tasks if not t[0].exists()]
    with cf.ProcessPoolExecutor(workers) as ex:
        for o in ex.map(S.work, tasks):
            print("  saved", o.name, flush=True)
    out = C.RESULTS / "srm_fld_rerun"; out.mkdir(parents=True, exist_ok=True)
    split = np.array([r[3] for r in C.matched_split("suniward")])
    Xc, Xs = S.load("covers_universal"), S.load("deds_v5m_universal")
    Xpc, Xnew = S.load("covers_paper_ft"), S.load("v5rerun_paper_ft")
    tr, va = split == "train", split == "val"
    ens, _ = FldEnsembleTrainer(Xc[tr], Xs[tr], seed=2026, verbose=0).train()
    sv = K.soft(ens, np.r_[Xc[va], Xs[va]]); yv = np.r_[np.zeros(va.sum()), np.ones(va.sum())]
    _, thr = K.pe_thr(yv, sv)
    sc, ss = K.soft(ens, Xpc), K.soft(ens, Xnew); y = np.r_[np.zeros(N), np.ones(N)]; s = np.r_[sc, ss]
    pd.DataFrame({"label": y, "score": s}).to_csv(out / "deds_v5m_v5_paper_ft_scores.csv", index=False)
    pe, _ = K.pe_thr(y, s); lo, hi = K.boot_ci(sc, ss)
    r = {"method": "deds_v5m", "set": "v5_paper_ft", "pairs": N, "auc": roc_auc_score(y, s), "auc_ci_low": lo, "auc_ci_high": hi,
         "pe_min": pe, "acc_at_val_thr": float(((s > thr) == y).mean()), "stego_flagged": int((ss > thr).sum()),
         "covers_flagged": int((sc > thr).sum()), "L": ens.num_base_learners, "d_sub": ens.d_sub, "val_auc": roc_auc_score(yv, sv)}
    pd.DataFrame([r]).to_csv(out / "summary.csv", index=False)
    print(f"SRM+FLD  val AUC {r['val_auc']:.4f}  AUC {r['auc']:.4f} [{lo:.4f}, {hi:.4f}]  P_E val {1 - r['acc_at_val_thr']:.4f}  P_E min {pe:.4f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=4); a = ap.parse_args()
    step_y(); step_srnet(); step_srm(a.workers)
    print("DONE", flush=True)
