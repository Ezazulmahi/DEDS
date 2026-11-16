r"""Detectability of each ablation variant (reviewer 1, comment 3). For every variant: SRM features (sealwatch, as in
Steganalyzer_SOTA_2026) of its 200 covers and their matched-payload stego images (written by eval_variant.py), then
the Fisher linear discriminant ensemble with 5-fold cross-validation over pairs (a cover and its stego image always
in the same fold). Out-of-fold scores give AUC with a bootstrap interval and the minimum P_E.
Features are cached per variant, so the script can be rerun when more variants are evaluated.
usage: python rev_abl_detect.py baseline full_80k full reward_only filter_only ordinary [full_run2 ...]"""
import os, sys, concurrent.futures as cf
import numpy as np, pandas as pd, cv2
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, r"F:\DEDS_extra_checks"); import common as C
A = r"F:\DEDS_extra_checks\ablation"; OUT = r"F:\DEDS_extra_checks\results"
CACHE = os.path.join(OUT, "rev_abl_srm"); IDX = list(range(0, 1000, 5))


def cover(name, i):
    if name == "baseline":
        return os.path.join(C.CORP, "baseline", f"{i:03d}.png")
    if name == "full_80k":
        return os.path.join(C.CORP, "finetuned", f"{i:03d}.png")
    return os.path.join(A, "covers", name, f"{i:03d}.png")


def feat(path):
    import sealwatch as sw
    y = C.luma(cv2.imread(path))
    return np.concatenate([np.asarray(a, np.float32).ravel() for a in sw.srm.extract(y).values()])


def features(name):
    f = os.path.join(CACHE, f"{name}.npz")
    if os.path.exists(f):
        z = np.load(f); return z["c"], z["s"]
    cp = [cover(name, i) for i in IDX]; sp = [os.path.join(A, "stego_tmp", f"{name}_{i:03d}.png") for i in IDX]
    missing = [p for p in cp + sp if not os.path.exists(p)]
    assert not missing, f"{name}: {len(missing)} images missing, e.g. {missing[0]}"
    with cf.ProcessPoolExecutor(3) as ex:
        X = list(ex.map(feat, cp + sp, chunksize=8))
    c, s = np.stack(X[:200]), np.stack(X[200:])
    np.savez_compressed(f, c=c, s=s)
    return c, s


def evaluate(c, s, k=5, seed=2026):
    from sealwatch.ensemble_classifier import FldEnsembleTrainer
    rng = np.random.default_rng(seed); fold = rng.permutation(len(c)) % k
    sc, ss = np.zeros(len(c)), np.zeros(len(s))
    for f in range(k):
        tr, te = fold != f, fold == f
        ens, _ = FldEnsembleTrainer(c[tr], s[tr], seed=seed, verbose=0).train()
        soft = lambda X: np.mean([bl.predict(X) for bl in ens.base_learners], axis=0)
        sc[te], ss[te] = soft(c[te]), soft(s[te])
    y = np.r_[np.zeros(len(sc)), np.ones(len(ss))]; x = np.r_[sc, ss]
    auc = roc_auc_score(y, x); fpr, tpr, _ = roc_curve(y, x); pe = float(np.min((fpr + 1 - tpr) / 2))
    b = [roc_auc_score(np.r_[np.zeros(len(ii)), np.ones(len(ii))], np.r_[sc[ii], ss[ii]])
         for ii in (rng.integers(0, len(sc), len(sc)) for _ in range(2000))]
    return auc, float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), pe


if __name__ == "__main__":
    os.makedirs(CACHE, exist_ok=True)
    out = os.path.join(OUT, "rev_abl_detect.csv")
    done = pd.read_csv(out) if os.path.exists(out) else pd.DataFrame(columns=["variant"])
    rows = []
    for name in sys.argv[1:]:
        if name in set(done.variant):
            continue
        c, s = features(name)
        auc, lo, hi, pe = evaluate(c, s)
        rows.append(dict(variant=name, n_pairs=len(c), auc=auc, auc_lo=lo, auc_hi=hi, pe_min=pe))
        print(f"{name}: AUC {auc:.3f} ({lo:.3f} to {hi:.3f}), PE {pe:.3f}", flush=True)
        pd.concat([done, pd.DataFrame(rows)]).to_csv(out, index=False)
