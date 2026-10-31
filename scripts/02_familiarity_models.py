"""Step 2. Per-concept hidden Markov models, fitted on each fold's training learners:
BKT, BKT-Forget, and DSFT's long-term familiarity layer (three levels, constant or time-dependent forgetting).
Writes out-of-fold predictions and the fitted parameters that step 3 builds on."""
import pickle, numpy as np
import common
from dsft import data, familiarity as fam
from dsft.data import N_KC
from sklearn.metrics import roc_auc_score

a = common.args()
z = data.load(a.data, a.quick)
user, kc, y, t = z["user"], z["kc"], z["y"].astype(np.float32), z["t"]
N = len(y)
us, folds = data.folds(user)
MODELS = {"BKT": (2, "none"), "BKT-Forget": (2, "const"), "Familiarity3": (3, "const"), "Familiarity3-Time": (3, "time")}
pred = {m: np.full(N, np.nan) for m in MODELS}
fold = np.full(N, -1)
params = {}
for fi, te_users in enumerate(folds):
    tr_users = np.setdiff1d(us, te_users)
    te = np.isin(user, te_users); fold[te] = fi
    Btr, Bte = fam.build_seqs(user, kc, y, t, tr_users), fam.build_seqs(user, kc, y, t, te_users)
    for name, (S, f) in MODELS.items():
        P = fam.fit(Btr, N_KC, S, f, iters=common.iters(a, 150))
        pred[name][te] = fam.predict(P, Bte, f, N)[te]
        params[(name, fi)] = {k: np.asarray(v) for k, v in P.items()}
        common.log(f"fold {fi} {name:18s} AUC {roc_auc_score(y[te], pred[name][te]):.4f}")
np.savez_compressed(common.path(a, "preds", "familiarity.npz"), fold=fold, **pred)
pickle.dump(params, open(common.path(a, "preds", "familiarity_params.pkl"), "wb"))
common.log("done")
