"""Step 8 (optional). Small training sets (Section 5.3): train on 2% and 5% of each fold's training learners
(17 and 43 learners) and test on the full test fold. DSFT here is the core model without learner ability."""
import json, numpy as np, scipy.sparse as sp
import common
from dsft import data, features, familiarity as fam, take_layer as tl, classic_kt as ckt, attention_kt as akt
from dsft.data import N_KC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

a = common.args()
z = data.load(a.data, a.quick)
user, kc, y, t, rec, pos, item = (z[k] for k in ["user", "kc", "y", "t", "rec", "pos", "item"])
yf = y.astype(np.float32); N, NI = len(y), int(item.max()) + 1
us, folds = data.folds(user)
s_f, f_f = features.pfa_counts(user, kc, y)
dense = np.column_stack([s_f, f_f, features.in_take(rec, y)])
oh = lambda v, n, ix: sp.csr_matrix((np.ones(len(ix)), (np.arange(len(ix)), v[ix])), shape=(len(ix), n))
X = lambda ix: sp.hstack([oh(kc, N_KC, ix), oh(item, NI, ix), sp.csr_matrix(dense[ix])]).tocsr()
out = {}
for frac in (0.02, 0.05):
    for fi, te_users in enumerate(folds):
        tr_all = np.setdiff1d(us, te_users)
        n_tr = max(5, int(round(frac * len(tr_all))))
        tr_users = np.random.default_rng(100 + fi).permutation(tr_all)[:n_tr]
        tr, te = data.rows(user, tr_users), data.rows(user, te_users)
        row = {"learners": int(n_tr)}
        row["LR"] = roc_auc_score(y[te], LogisticRegression(max_iter=400).fit(X(tr), y[tr]).predict_proba(X(te))[:, 1])
        P3 = fam.fit(fam.build_seqs(user, kc, yf, t, tr_users), N_KC, 3, "const", iters=common.iters(a, 150))
        pf = np.full(N, np.nan)
        for g in (tr_users, te_users):
            m = np.isin(user, g); pf[m] = fam.predict(P3, fam.build_seqs(user, kc, yf, t, g), "const", N)[m]
        Gtr, Gte = tl.build_takes(rec, pos, tr), tl.build_takes(rec, pos, te)
        Pd = tl.fit(tl.arrays(Gtr, yf, pf, item), 6, NI, True, iters=common.iters(a, 300))
        row["DSFT core"] = roc_auc_score(y[te], tl.predict(Pd, Gte, tl.arrays(Gte, yf, pf, item), True, N)[0][te])
        perm = np.random.default_rng(fi).permutation(tr_users); nv = max(2, len(perm) // 10)
        sv, sf = akt.user_seqs(user, item, kc, y, perm[:nv]), akt.user_seqs(user, item, kc, y, perm[nv:])
        akt.set_dim(64)
        Ps = akt.fit("simplekt", sf, sv, item, kc, y, NI, N_KC, max_epochs=1 if a.quick else 30, bs=16, seed=fi)
        row["simpleKT"] = roc_auc_score(y[te], akt.predict(Ps, "simplekt", akt.user_seqs(user, item, kc, y, te_users), item, kc, y, N)[te])
        Pk, _, _ = ckt.dkt_fit(ckt.dkt_sequences(user, kc, y, perm[nv:], N_KC), ckt.dkt_sequences(user, kc, y, perm[:nv], N_KC), y, N_KC,
                               n_h=100, lr=3e-3, seed=fi, max_epochs=1 if a.quick else 30, bs=16)
        row["DKT"] = roc_auc_score(y[te], ckt.dkt_predict(Pk, ckt.dkt_sequences(user, kc, y, te_users, N_KC), N)[te])
        out[f"{frac}|{fi}"] = row
        common.log(f"{n_tr} learners, fold {fi}: " + "  ".join(f"{k} {v:.4f}" for k, v in row.items() if k != "learners"))
json.dump(out, open(common.path(a, "results", "small_data.json"), "w"), indent=1)
for frac in (0.02, 0.05):
    rows_ = [v for k, v in out.items() if k.startswith(str(frac))]
    common.log(f"{rows_[0]['learners']} learners, mean AUC: " + "  ".join(f"{m} {np.mean([r[m] for r in rows_]):.4f}" for m in ["LR", "DSFT core", "DKT", "simpleKT"]))
