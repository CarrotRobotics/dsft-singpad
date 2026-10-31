"""Step 5. Paper tables: Table 1 (main comparison), Table 2 (ablations), AUC by practice gap (Fig. 2b),
and the learner-level bootstrap intervals. Writes results/tables.md and results/metrics.json."""
import json, numpy as np, jax
import common
from dsft import data, features, evaluation as ev, attention_kt as akt
from dsft.data import N_KC

a = common.args(lambda p: p.add_argument("--boot", type=int, default=300))
z = data.load(a.data, a.quick)
user, kc, y, t, item = (z[k] for k in ["user", "kc", "y", "t", "item"])
NI = int(item.max()) + 1
P = {}
for f in ("familiarity", "dsft", "baselines"):
    P.update(dict(np.load(common.path(a, "preds", f + ".npz"))))
fold = P["fold"]
gap = features.gap_days(user, kc, t)
B = 50 if a.quick else a.boot


def n_attn(kind, d):
    akt.set_dim(d)
    return sum(int(np.prod(v.shape)) for v in akt.init(jax.random.PRNGKey(0), kind, NI, N_KC, 200).values())


sel = json.load(open(common.path(a, "results", "take_states_selection.json")))
votes = [v["chosen"] for v in sel.values()]
C = max(set(votes), key=votes.count)   # 6 in every fold of the paper's run
params = {"BKT": 4 * N_KC, "PFA": 3 * N_KC, "BKT-Forget": 5 * N_KC, "LR": N_KC + NI + 9 + 1,
          "DKT": (2 * N_KC + 1) * 64 + (64 + 100) * 400 + 400 + 100 * N_KC + N_KC,
          "SAKT": n_attn("sakt", 128), "AKT": n_attn("akt", 64), "simpleKT": n_attn("simplekt", 64),
          "DSFT": 9 * N_KC + (C - 1) + C * (C - 1) + C + 1 + NI + 3}
labels = {"LR": "Logistic regression", "DKT": "DKT (tuned)", "SAKT": "SAKT (tuned)", "DSFT": "**DSFT (ours)**"}
out, M = [], {}

# ---- Table 1 ----
out += ["## Table 1. Main comparison (five-fold mean ± SD)", "",
        "| Model | Parameters | AUC | RMSE | DSFT minus model (95% bootstrap interval) |", "| --- | --- | --- | --- | --- |"]
order = ["BKT", "PFA", "BKT-Forget", "SAKT", "AKT", "LR", "simpleKT", "DKT", "DSFT"]
for m in order:
    au = ev.fold_auc(y, P[m], fold)
    M[m] = dict(auc=float(au.mean()), sd=float(au.std()), rmse=ev.rmse(y, P[m]), params=params[m],
                auc_gap1=float(ev.fold_auc(y, P[m], fold, gap > 1).mean()), auc_gap7=float(ev.fold_auc(y, P[m], fold, gap > 7).mean()))
    if m != "DSFT":
        M[m]["vs_dsft"] = ev.bootstrap_diff(y, user, P["DSFT"], P[m], B)
    d = M[m].get("vs_dsft")
    out.append(f"| {labels.get(m, m)} | {params[m]:,} | {M[m]['auc']:.4f} ± {M[m]['sd']:.4f} | {M[m]['rmse']:.4f} | "
               + (f"{d[0]:+.4f} ({d[1]:+.4f}, {d[2]:+.4f})" if d else "—") + " |")
    common.log(out[-1])

# ---- AUC by gap (Fig. 2b) ----
out += ["", "## AUC by gap since the previous attempt on the same concept (Fig. 2b)", "",
        "| Model | All | Gap > 1 day | Gap > 7 days |", "| --- | --- | --- | --- |"]
out += [f"| {labels.get(m, m)} | {M[m]['auc']:.4f} | {M[m]['auc_gap1']:.4f} | {M[m]['auc_gap7']:.4f} |" for m in order]

# ---- Table 2 ----
core = ev.fold_auc(y, P["DSFT-core"], fold)
out += ["", "## Table 2. Ablations of the DSFT core (Δ = core minus variant)", "",
        "| Variant | AUC | Δ (range over folds) | Bootstrap 95% interval of Δ |", "| --- | --- | --- | --- |",
        f"| DSFT core | {core.mean():.4f} | — | — |"]
abl = [("Without the take layer (familiarity layer only)", "Familiarity3"), ("Two take states instead of the chosen number", "Core-C2"),
       ("Without note difficulty", "Core-NoNoteDifficulty"), ("Without the familiarity layer", "Core-NoFamiliarity"),
       ("Two familiarity levels instead of three", "Core-TwoLevelFamiliarity"),
       ("Time-dependent instead of constant forgetting", "Core-TimeForgetting"), ("With learner ability (final model)", "DSFT")]
M["ablations"] = {}
for label, m in abl:
    au = ev.fold_auc(y, P[m], fold); d = core - au
    b = ev.bootstrap_diff(y, user, P["DSFT-core"], P[m], B)
    M["ablations"][label] = dict(auc=float(au.mean()), delta=float(d.mean()), fold_min=float(d.min()), fold_max=float(d.max()), boot=b)
    out.append(f"| {label} | {au.mean():.4f} | {d.mean():+.4f} ({d.min():+.4f} to {d.max():+.4f}) | {b[1]:+.4f} to {b[2]:+.4f} |")
    common.log(out[-1])
out += ["", "Take-state sensitivity (core): " + ", ".join(f"C={c}: {ev.fold_auc(y, P[f'Core-C{c}'], fold).mean():.4f}" for c in (2, 3, 4, 5))
        + f", chosen C: {core.mean():.4f}"]

# ---- latent states add information to the logistic regression ----
b = ev.bootstrap_diff(y, user, P["LR-noAbility+DSFTstates"], P["LR-noAbility"], B)
out += ["", f"Logistic regression (no ability features) {ev.fold_auc(y, P['LR-noAbility'], fold).mean():.4f} -> with DSFT states "
        f"{ev.fold_auc(y, P['LR-noAbility+DSFTstates'], fold).mean():.4f}; difference {b[0]:+.4f} ({b[1]:+.4f}, {b[2]:+.4f})"]
M["lr_plus_states"] = b
open(common.path(a, "results", "tables.md"), "w").write("\n".join(out) + "\n")
json.dump(M, open(common.path(a, "results", "metrics.json"), "w"), indent=1)
common.log("wrote results/tables.md and results/metrics.json")
