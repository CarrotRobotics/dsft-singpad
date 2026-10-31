"""Step 3. DSFT: choose the number of take states on validation learners, fit the final model, and run the ablations.

Per fold:
  * familiarity predictions p_fam for every note from the step-2 three-level model of that fold;
  * C in {2..6} chosen on 10% validation learners of the training set (take layer with note difficulty);
  * DSFT core (chosen C, note difficulty) and the final model (+ learner ability), fitted on all training learners;
  * ablations of the core: no note difficulty, no familiarity layer, two familiarity levels, time-dependent forgetting,
    and C = 2..5 for the sensitivity curve;
  * a three-take-state model whose states feed the logistic-regression analysis in step 4 and the take-state
    analysis in step 7.
"""
import json, pickle, numpy as np
import common
import jax.numpy as jnp
from dsft import data, familiarity as fam, take_layer as tl, features
from sklearn.metrics import roc_auc_score

a = common.args()
z = data.load(a.data, a.quick)
user, kc, y, t, rec, pos, item = (z[k] for k in ["user", "kc", "y", "t", "rec", "pos", "item"])
yf = y.astype(np.float32); N = len(y); NI = int(item.max()) + 1
us, folds = data.folds(user)
params = pickle.load(open(common.path(a, "preds", "familiarity_params.pkl"), "rb"))
F_ab = features.ability(user, rec, y)
IT = common.iters(a, 300)


def p_fam(key, forget, fi, groups):
    """Familiarity-layer predictions for the given learner groups, using fold fi's parameters."""
    P = {k: jnp.asarray(v) for k, v in params[(key, fi)].items()}
    out = np.full(N, np.nan)
    for g in groups:
        m = np.isin(user, g)
        out[m] = fam.predict(P, fam.build_seqs(user, kc, yf, t, g), forget, N)[m]
    return out


def fit_predict(tr, te, pf, C, use_item, feats=None):
    Gtr, Gte = tl.build_takes(rec, pos, tr), tl.build_takes(rec, pos, te)
    P = tl.fit(tl.arrays(Gtr, yf, pf, item, feats), C, NI, use_item, iters=IT)
    p, st = tl.predict(P, Gte, tl.arrays(Gte, yf, pf, item, feats), use_item, N)
    return P, p, st


names = ["DSFT", "DSFT-core", "Core-NoNoteDifficulty", "Core-NoFamiliarity", "Core-TwoLevelFamiliarity",
         "Core-TimeForgetting"] + [f"Core-C{c}" for c in (2, 3, 4, 5)]
pred = {m: np.full(N, np.nan) for m in names}
states3 = np.full((N, 3), np.nan)
pf_all = np.full(N, np.nan)
chosen, saved3 = {}, {}
for fi, te_users in enumerate(folds):
    tr_users = np.setdiff1d(us, te_users)
    tr, te = data.rows(user, tr_users), data.rows(user, te_users)
    pf = p_fam("Familiarity3", "const", fi, (tr_users, te_users)); pf_all[te] = pf[te]
    # choose C on validation learners only
    fit_u, val_u = data.train_val(us, te_users, seed=1000 + fi)
    fr, vr = data.rows(user, fit_u), data.rows(user, val_u)
    scores = {C: roc_auc_score(y[vr], fit_predict(fr, vr, pf, C, True)[1][vr]) for C in (2, 3, 4, 5, 6)}
    C = max(scores, key=scores.get); chosen[fi] = dict(validation_auc=scores, chosen=C)
    common.log(f"fold {fi} validation AUC by C: {scores} -> C = {C}")
    def run(name, pf_, C_, use_item, feats=None):
        pred[name][te] = fit_predict(tr, te, pf_, C_, use_item, feats)[1][te]

    run("DSFT-core", pf, C, True)
    run("DSFT", pf, C, True, F_ab)
    run("Core-NoNoteDifficulty", pf, C, False)
    run("Core-NoFamiliarity", np.full(N, 0.5, np.float32), C, True)
    run("Core-TwoLevelFamiliarity", p_fam("BKT-Forget", "const", fi, (tr_users, te_users)), C, True)
    run("Core-TimeForgetting", p_fam("Familiarity3-Time", "time", fi, (tr_users, te_users)), C, True)
    for c in (2, 3, 4, 5):
        run(f"Core-C{c}", pf, c, True)
    # three take states: readable states for the analyses (on all learners of the fold)
    P3, _, st_te = fit_predict(tr, te, pf, 3, True)
    Gtr = tl.build_takes(rec, pos, tr)
    _, st_tr = tl.predict(P3, Gtr, tl.arrays(Gtr, yf, pf, item), True, N)
    states3[te] = st_te[te]; states3[tr] = st_tr[tr]
    saved3[fi] = {k: np.asarray(v) for k, v in P3.items()}
    r = np.r_[tr, te]   # this fold's states and p_fam for training and test rows (used in steps 4 and 7)
    np.savez_compressed(common.path(a, "preds", f"states3_fold{fi}.npz"), rows=r, states=states3[r], pf=pf[r])
    common.log(f"fold {fi}: " + "  ".join(f"{m} {roc_auc_score(y[te], pred[m][te]):.4f}" for m in names))
np.savez_compressed(common.path(a, "preds", "dsft.npz"), pf_test=pf_all, **pred)
pickle.dump(saved3, open(common.path(a, "preds", "take3_params.pkl"), "wb"))
json.dump({str(k): {"validation_auc": {str(c): s for c, s in v["validation_auc"].items()}, "chosen": v["chosen"]} for k, v in chosen.items()},
          open(common.path(a, "results", "take_states_selection.json"), "w"), indent=1)
common.log("done")
