"""Step 6. Data analyses reported in Sections 5.4 and 5.5:
error runs within a take, forgetting after practice gaps, and the three readable take states."""
import pickle, numpy as np, pandas as pd, jax
import common
from dsft import data

a = common.args()
z = data.load(a.data, a.quick)
user, kc, y, t, rec = (z[k] for k in ["user", "kc", "y", "t", "rec"])
yf = y.astype(float); N = len(y)
lines = []

# error runs within a take
same = np.r_[False, rec[1:] == rec[:-1]]; prev = np.r_[np.nan, yf[:-1]]
lines += ["## Error runs within a take", "",
          f"P(correct | previous note correct) = {yf[same & (prev == 1)].mean():.3f}",
          f"P(correct | previous note wrong)   = {yf[same & (prev == 0)].mean():.3f}", "",
          "| Accuracy of the take so far | Next note correct | Note attempts |", "| --- | --- | --- |"]
d = pd.DataFrame({"r": rec, "y": yf})
d["so_far"] = d.groupby("r").y.transform(lambda s: s.shift(1).expanding().mean())
for lo, hi, lab in [(0, .5, "below 50%"), (.5, .75, "50% to 75%"), (.75, .9, "75% to 90%"), (.9, 1.01, "90% or above")]:
    s = d[(d.so_far >= lo) & (d.so_far < hi)]
    lines.append(f"| {lab} | {s.y.mean():.1%} | {len(s):,} |")

# forgetting after practice gaps
key = user.astype(np.int64) * 1000 + kc; o = np.lexsort((np.arange(N), key))
df = pd.DataFrame({"k": key[o], "y": yf[o], "t": t[o]}); g = df.groupby("k")
df["gap"] = g.t.diff(); df["prev5"] = g.y.transform(lambda s: s.shift(1).rolling(5, min_periods=3).mean()); df = df.dropna()
df["bin"] = pd.cut(df.gap, [-1, 1 / 24, 1, 7, 30, 1e9], labels=["under 1 hour", "1 hour to 1 day", "1 to 7 days", "7 to 30 days", "over 30 days"])
lines += ["", "## Accuracy after practice gaps (Section 5.5)", "",
          "| Gap since previous attempt | Attempts | Accuracy | Change from previous five attempts |", "| --- | --- | --- | --- |"]
for b, s in df.groupby("bin", observed=True):
    lines.append(f"| {b} | {len(s):,} | {s.y.mean():.1%} | {100 * (s.y.mean() - s.prev5.mean()):+.1f} points |")

# three readable take states (on track / shaky / lost)
P3 = pickle.load(open(common.path(a, "preds", "take3_params.pkl"), "rb"))
T = np.mean([np.asarray(jax.nn.softmax(p["A"], -1)) for p in P3.values()], 0)
lines += ["", "## Three-state take layer (Section 5.4)", "", "Mean transition matrix (rows: from on track / shaky / lost):"]
lines += ["  " + "  ".join(f"{v:.3f}" for v in row) for row in T]
lines.append("Initial distribution: " + ", ".join(f"{v:.3f}" for v in np.mean([np.asarray(jax.nn.softmax(p["pi"])) for p in P3.values()], 0)))
lines.append("Offsets b: " + ", ".join(f"{v:+.2f}" for v in np.mean([p["b"] for p in P3.values()], 0)))
st = np.full((N, 3), np.nan)
us, folds = data.folds(user)
for fi, te_users in enumerate(folds):
    S = np.load(common.path(a, "preds", f"states3_fold{fi}.npz"))
    te = np.isin(S["rows"], data.rows(user, te_users))
    st[S["rows"][te]] = S["states"][te]
state = np.argmax(st, 1)
lines += ["", "| Take state (most likely before the note) | Share of notes | Notes sung correctly | Share of all errors |", "| --- | --- | --- | --- |"]
for s, n in enumerate(["On track", "Shaky", "Lost"]):
    m = state == s
    lines.append(f"| {n} | {m.mean():.1%} | {yf[m].mean():.1%} | {((yf == 0) & m).sum() / (yf == 0).sum():.1%} |")
open(common.path(a, "results", "analyses.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
