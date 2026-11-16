"""Sanity checks for the matched SRM result (AUC 1.000 for deds_v5m). Uses the cached SRM features of
Steganalyzer_SOTA_2026; nothing is recomputed or overwritten.
  real       cover vs DEDS stego, random 80/20 split by pair        -> expected high
  shuffled   the same pairs with labels shuffled                     -> must be about 0.5
  cover_cov  two random halves of the covers, labelled 0 and 1       -> must be about 0.5"""
import sys, importlib
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, r"F:\Steganalyzer_SOTA_2026")
SRM = importlib.import_module("10_srm_extract")
from sealwatch.ensemble_classifier import FldEnsembleTrainer

Xc, Xs = SRM.load("covers_universal"), SRM.load("deds_v5m_universal")
assert len(Xc) == len(Xs)
rng = np.random.default_rng(7); n = len(Xc); perm = rng.permutation(n); tr, te = perm[: int(0.8 * n)], perm[int(0.8 * n):]


def auc(c_tr, s_tr, c_te, s_te):
    ens, _ = FldEnsembleTrainer(c_tr, s_tr, seed=2026, verbose=0).train()
    sc = lambda X: np.mean([bl.predict(X) for bl in ens.base_learners], axis=0)
    return roc_auc_score(np.r_[np.zeros(len(c_te)), np.ones(len(s_te))], np.r_[sc(c_te), sc(s_te)])


print(f"real      AUC {auc(Xc[tr], Xs[tr], Xc[te], Xs[te]):.3f}", flush=True)
both = np.r_[Xc[tr], Xs[tr]]; lab = rng.permutation(np.r_[np.zeros(len(tr)), np.ones(len(tr))])
both_te = np.r_[Xc[te], Xs[te]]; lab_te = rng.permutation(np.r_[np.zeros(len(te)), np.ones(len(te))])
print(f"shuffled  AUC {auc(both[lab == 0], both[lab == 1], both_te[lab_te == 0], both_te[lab_te == 1]):.3f}", flush=True)
h = rng.permutation(n); a, b = h[: n // 2], h[n // 2:]
print(f"cover_cov AUC {auc(Xc[a[:800]], Xc[b[:800]], Xc[a[800:]], Xc[b[800:]]):.3f}", flush=True)
