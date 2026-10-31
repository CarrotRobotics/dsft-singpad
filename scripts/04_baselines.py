"""Step 4. Remaining baselines on the same folds: PFA, logistic regression, DKT, SAKT, AKT and simpleKT.

DKT and SAKT use the settings selected on validation learners of fold 0 (pass --tune to rerun that search:
DKT hidden size 100/200 x learning rate 1e-3/3e-3; SAKT embedding size 64/128 x the same learning rates).
Also fits the logistic regression with DSFT's three-take-state probabilities as extra features (needs step 3).
"""
import json, numpy as np, scipy.sparse as sp
import common
from dsft import data, features, classic_kt as ckt, attention_kt as akt
from dsft.data import N_KC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

a = common.args(lambda p: p.add_argument("--tune", action="store_true"))
z = data.load(a.data, a.quick)
user, kc, y, t, rec, item = (z[k] for k in ["user", "kc", "y", "t", "rec", "item"])
N, NI = len(y), int(item.max()) + 1
us, folds = data.folds(user)
EP = 1 if a.quick else None

s_f, f_f = features.pfa_counts(user, kc, y)
dense = np.column_stack([s_f, f_f, features.in_take(rec, y)])
ab = features.ability(user, rec, y)
oh = lambda v, n, ix: sp.csr_matrix((np.ones(len(ix)), (np.arange(len(ix)), v[ix])), shape=(len(ix), n))


def lr_X(ix, extra=None):
    cols = [dense[ix]] + ([extra[ix]] if extra is not None else [])
    return sp.hstack([oh(kc, N_KC, ix), oh(item, NI, ix), sp.csr_matrix(np.column_stack(cols))]).tocsr()


def lr(tr, te, extra=None):
    return LogisticRegression(max_iter=400).fit(lr_X(tr, extra), y[tr]).predict_proba(lr_X(te, extra))[:, 1]


# ---- optional search for DKT and SAKT on fold 0's validation learners ----
cfg = {"DKT": (100, 3e-3), "SAKT": (128, 3e-3)}
if a.tune:
    fit_u, val_u = data.train_val(us, folds[0], seed=0)
    va = data.rows(user, val_u); res = {}
    for n_h, lr_ in [(100, 1e-3), (100, 3e-3), (200, 1e-3), (200, 3e-3)]:
        _, v, ep = ckt.dkt_fit(ckt.dkt_sequences(user, kc, y, fit_u, N_KC), ckt.dkt_sequences(user, kc, y, val_u, N_KC), y, N_KC,
                               n_h=n_h, lr=lr_, seed=0, max_epochs=EP or 20)
        res[f"DKT hidden {n_h} lr {lr_}"] = v; common.log(f"tune DKT {n_h} {lr_}: val AUC {v:.4f} (epoch {ep})")
    for d, lr_ in [(64, 1e-3), (64, 3e-3), (128, 1e-3), (128, 3e-3)]:
        akt.set_dim(d)
        sf, sv = akt.user_seqs(user, item, kc, y, fit_u), akt.user_seqs(user, item, kc, y, val_u)
        P = akt.fit("sakt", sf, sv, item, kc, y, NI, N_KC, max_epochs=EP or 20, seed=0, lr=lr_)
        v = roc_auc_score(y[va], akt.predict(P, "sakt", sv, item, kc, y, N)[va])
        res[f"SAKT dim {d} lr {lr_}"] = v; common.log(f"tune SAKT {d} {lr_}: val AUC {v:.4f}")
    cfg["DKT"] = max([(100, 1e-3), (100, 3e-3), (200, 1e-3), (200, 3e-3)], key=lambda c: res[f"DKT hidden {c[0]} lr {c[1]}"])
    cfg["SAKT"] = max([(64, 1e-3), (64, 3e-3), (128, 1e-3), (128, 3e-3)], key=lambda c: res[f"SAKT dim {c[0]} lr {c[1]}"])
    json.dump(dict(validation_auc=res, chosen=cfg), open(common.path(a, "results", "deep_tuning.json"), "w"), indent=1)
common.log(f"DKT hidden/lr {cfg['DKT']}, SAKT dim/lr {cfg['SAKT']}")

import os
if not os.path.exists(common.path(a, "preds", "states3_fold0.npz")):
    raise SystemExit("Run scripts/03_dsft.py with the same --out first: this step uses DSFT's take states.")
names = ["PFA", "LR", "LR-noAbility", "LR-noAbility+DSFTstates", "DKT", "SAKT", "AKT", "simpleKT"]
pred = {m: np.full(N, np.nan) for m in names}
for fi, te_users in enumerate(folds):
    tr_users = np.setdiff1d(us, te_users)
    tr, te = data.rows(user, tr_users), data.rows(user, te_users)
    trm = np.isin(user, tr_users)
    pred["PFA"][te] = ckt.pfa_fit_predict(kc, y, s_f, f_f, trm, te, N_KC, iters=common.iters(a, 400))
    pred["LR"][te] = lr(tr, te, ab)
    pred["LR-noAbility"][te] = lr(tr, te)
    S = np.load(common.path(a, "preds", f"states3_fold{fi}.npz"))
    st = np.full((N, 3), np.nan); pfx = np.full(N, 0.5); st[S["rows"]] = S["states"]; pfx[S["rows"]] = S["pf"]
    pc = np.clip(pfx, 1e-4, 1 - 1e-4)
    extra = np.column_stack([st[:, 1:], np.log(pc / (1 - pc))])   # P(shaky), P(lost), familiarity log-odds
    pred["LR-noAbility+DSFTstates"][te] = lr(tr, te, extra)
    # deep models: 10% of training learners for early stopping
    fit_u, val_u = data.train_val(us, te_users, seed=fi)
    P, _, ep = ckt.dkt_fit(ckt.dkt_sequences(user, kc, y, fit_u, N_KC), ckt.dkt_sequences(user, kc, y, val_u, N_KC), y, N_KC,
                           n_h=cfg["DKT"][0], lr=cfg["DKT"][1], seed=fi, max_epochs=EP or 20)
    pred["DKT"][te] = ckt.dkt_predict(P, ckt.dkt_sequences(user, kc, y, te_users, N_KC), N)[te]
    sf, sv, ste = (akt.user_seqs(user, item, kc, y, g) for g in (fit_u, val_u, te_users))
    for name, kind, d, lr_, me in [("SAKT", "sakt", cfg["SAKT"][0], cfg["SAKT"][1], 20), ("AKT", "akt", 64, 1e-3, 15),
                                   ("simpleKT", "simplekt", 64, 1e-3, 15)]:
        akt.set_dim(d)
        P = akt.fit(kind, sf, sv, item, kc, y, NI, N_KC, max_epochs=EP or me, seed=fi, lr=lr_)
        pred[name][te] = akt.predict(P, kind, ste, item, kc, y, N)[te]
    common.log(f"fold {fi}: " + "  ".join(f"{m} {roc_auc_score(y[te], pred[m][te]):.4f}" for m in names))
np.savez_compressed(common.path(a, "preds", "baselines.npz"), **pred)
common.log("done")
