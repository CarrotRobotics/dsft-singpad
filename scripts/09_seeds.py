"""Step 9 (optional). Seed robustness for the three most accurate models: DSFT (final), tuned DKT and simpleKT.

Seed 0 is the main run (steps 3 and 4). Seeds 1 and 2 are new:
  * DKT and simpleKT: different network initialisation and batch order, same folds, validation split and settings;
  * DSFT: fitting is deterministic, so the seed perturbs the starting values of both layers
    (Gaussian noise with sd 0.3 on the unconstrained familiarity and take-layer parameters) and both layers are refitted.
Reports the five-fold mean AUC per seed and the DSFT margins over DKT and simpleKT with learner-level bootstrap CIs
(results/seeds.md). Runs after steps 3 and 4 with the same --out.
"""
import numpy as np
import common
import jax.numpy as jnp
from dsft import data, features, familiarity as fam, take_layer as tl, classic_kt as ckt, attention_kt as akt, evaluation
from dsft.data import N_KC
from sklearn.metrics import roc_auc_score

a = common.args()
z = data.load(a.data, a.quick)
user, kc, y, t, rec, pos, item = (z[k] for k in ["user", "kc", "y", "t", "rec", "pos", "item"])
yf = y.astype(np.float32); N, NI = len(y), int(item.max()) + 1
us, folds = data.folds(user)
F_ab = features.ability(user, rec, y)
EP = 1 if a.quick else None
IT_F, IT_T = common.iters(a, 150), common.iters(a, 300)

_fam_init, _tl_init = fam.init_params, tl.init


def noisy(P, rng, skip=()):
    return {k: (v if k in skip else jnp.asarray(np.asarray(v) + rng.normal(0, 0.3, np.shape(v)).astype(np.float32)))
            for k, v in P.items()}


def dsft_seed(s, fi, te_users):
    rng = np.random.default_rng(1000 * s + fi)
    fam.init_params = lambda n_kc, S, forget: noisy(_fam_init(n_kc, S, forget), rng)
    tl.init = lambda C, n_item, k=1: noisy(_tl_init(C, n_item, k), rng, skip=("beta",))
    try:
        tr_users = np.setdiff1d(us, te_users); tr, te = data.rows(user, tr_users), data.rows(user, te_users)
        P3 = fam.fit(fam.build_seqs(user, kc, yf, t, tr_users), N_KC, 3, "const", iters=IT_F)
        pf = np.full(N, np.nan)
        for g in (tr_users, te_users):
            m = np.isin(user, g); pf[m] = fam.predict(P3, fam.build_seqs(user, kc, yf, t, g), "const", N)[m]
        Gtr, Gte = tl.build_takes(rec, pos, tr), tl.build_takes(rec, pos, te)
        P = tl.fit(tl.arrays(Gtr, yf, pf, item, F_ab), 6, NI, True, iters=IT_T)
        return tl.predict(P, Gte, tl.arrays(Gte, yf, pf, item, F_ab), True, N)[0]
    finally:
        fam.init_params, tl.init = _fam_init, _tl_init


base_dsft, base_bl = np.load(common.path(a, "preds", "dsft.npz")), np.load(common.path(a, "preds", "baselines.npz"))
pred = {"DSFT": [base_dsft["DSFT"]], "DKT": [base_bl["DKT"]], "simpleKT": [base_bl["simpleKT"]]}
for s in (1, 2):
    cur = {m: np.full(N, np.nan) for m in pred}
    for fi, te_users in enumerate(folds):
        te = data.rows(user, te_users)
        fit_u, val_u = data.train_val(us, te_users, seed=fi)
        P, _, _ = ckt.dkt_fit(ckt.dkt_sequences(user, kc, y, fit_u, N_KC), ckt.dkt_sequences(user, kc, y, val_u, N_KC), y, N_KC,
                              n_h=100, lr=3e-3, seed=100 * s + fi, max_epochs=EP or 20)
        cur["DKT"][te] = ckt.dkt_predict(P, ckt.dkt_sequences(user, kc, y, te_users, N_KC), N)[te]
        akt.set_dim(64)
        sf, sv, ste = (akt.user_seqs(user, item, kc, y, g) for g in (fit_u, val_u, te_users))
        P = akt.fit("simplekt", sf, sv, item, kc, y, NI, N_KC, max_epochs=EP or 15, seed=100 * s + fi, lr=1e-3)
        cur["simpleKT"][te] = akt.predict(P, "simplekt", ste, item, kc, y, N)[te]
        cur["DSFT"][te] = dsft_seed(s, fi, te_users)[te]
        common.log(f"seed {s} fold {fi}: " + "  ".join(f"{m} {roc_auc_score(y[te], cur[m][te]):.4f}" for m in cur))
    for m in pred:
        pred[m].append(cur[m])
np.savez_compressed(common.path(a, "preds", "seeds.npz"), **{f"{m}-s{s}": p for m, ps in pred.items() for s, p in enumerate(ps)})

fold = np.full(N, -1)
for fi, g in enumerate(folds):
    fold[data.rows(user, g)] = fi
fa = lambda p: np.mean([roc_auc_score(y[fold == f], p[fold == f]) for f in range(5)])
lines = ["# Seed robustness (seed 0 = main run)", "", "| Model | Seed 0 | Seed 1 | Seed 2 |", "| --- | --- | --- | --- |"]
for m, ps in pred.items():
    lines.append(f"| {m} | " + " | ".join(f"{fa(p):.4f}" for p in ps) + " |")
lines += ["", "| DSFT minus | Seed | Difference | 95% CI |", "| --- | --- | --- | --- |"]
for other in ("DKT", "simpleKT"):
    for s in range(3):
        d, lo, hi = evaluation.bootstrap_diff(y, user, pred["DSFT"][s], pred[other][s])
        lines.append(f"| {other} | {s} | {d:+.4f} | {lo:+.4f} to {hi:+.4f} |")
open(common.path(a, "results", "seeds.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
common.log("done")
