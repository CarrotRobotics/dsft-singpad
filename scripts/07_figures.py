"""Step 7. The paper's three figures (PDF and 300-dpi PNG) in <out>/figures.
Fig. 1 is a static diagram; Fig. 2 reads results/metrics.json from step 5; Fig. 3 is computed from the data."""
import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle, FancyBboxPatch
import common
from dsft import data

a = common.args()
BLUE, GRAY, INK = "#2a78d6", "#8a8984", "#0b0b0b"


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(common.path(a, "figures", f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------- Fig. 1: graphical structure ----------------
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 10})
fig, ax = plt.subplots(figsize=(6.6, 3.7)); ax.set_xlim(0, 8.8); ax.set_ylim(0, 5.0); ax.axis("off"); ax.set_aspect("equal")
R = 0.33


def node(x, y, s, obs=False, fs=11):
    ax.add_patch(Circle((x, y), R, facecolor="#d9d9d9" if obs else "white", edgecolor="black", lw=1.0, zorder=3))
    ax.text(x, y, s, ha="center", va="center", fontsize=fs, zorder=4)


def square(x, y, s, w=0.62):
    ax.add_patch(Rectangle((x - w / 2, y - w / 2), w, w, facecolor="white", edgecolor="black", lw=1.0, zorder=3))
    ax.text(x, y, s, ha="center", va="center", fontsize=11, zorder=4)


def arrow(p, q, r1=R, r2=R, rad=0.0, ls="-"):
    p, q = np.array(p, float), np.array(q, float); v = (q - p) / np.linalg.norm(q - p)
    ax.add_patch(FancyArrowPatch(tuple(p + v * r1), tuple(q - v * r2), connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
                                 mutation_scale=9, lw=0.9, color="black", shrinkA=0, shrinkB=0, linestyle=ls, zorder=2))


for y0, h, label in [(3.25, 1.55, "Long-term layer: familiarity (per learner and concept)"),
                     (0.25, 2.65, r"Short-term layer: take state ($C$ states, per take)")]:
    ax.add_patch(FancyBboxPatch((0.35, y0), 4.95, h, boxstyle="round,pad=0,rounding_size=0.08", facecolor="none",
                                edgecolor="black", lw=0.8, ls=(0, (4, 2))))
    ax.text(0.5, y0 + h - 0.18, label, fontsize=9, va="center")
U, P_, F = (1.25, 3.85), (2.85, 3.85), (4.45, 3.85)
for c, s in [(U, "$U$"), (P_, "$P$"), (F, "$F$")]:
    node(*c, s)
for p, q, up, down in [(U, P_, r"$\ell_U$", r"$d_P$"), (P_, F, r"$\ell_P$", r"$d_F$")]:
    arrow(p, q, rad=-0.35); arrow(q, p, rad=-0.35)
    ax.text((p[0] + q[0]) / 2, p[1] + 0.42, up, ha="center", va="center"); ax.text((p[0] + q[0]) / 2, p[1] - 0.42, down, ha="center", va="center")
C0, C1, Y0, Y1 = (1.85, 2.0), (3.95, 2.0), (1.85, 0.8), (3.95, 0.8)
node(*C0, "$c_{j-1}$", fs=10); node(*C1, "$c_j$"); node(*Y0, "$y_{j-1}$", obs=True, fs=10); node(*Y1, "$y_j$", obs=True)
arrow(C0, C1); ax.text(2.9, 2.2, "$A$", ha="center")
arrow((0.8, 2.0), C0, 0, R, ls=(0, (2, 2))); arrow(C0, Y0); arrow(C1, Y1)
PF, BQ, AU = (6.35, 3.85), (6.35, 1.75), (6.35, 0.55)
square(*PF, r"$p^{\mathrm{fam}}_j$"); square(*BQ, r"$\beta_{q_j}$"); square(*AU, r"$\mathbf{a}_u$")
arrow((F[0] + R, F[1]), PF, 0, 0.31); ax.text(5.82, 4.05, r"$\theta$", ha="center")
for src in (PF, BQ, AU):
    arrow(src, Y1, 0.36, R)
for (x, yy), s in [(PF, "familiarity\nprediction"), (BQ, "note\ndifficulty"), (AU, "learner ability\n(earlier takes)")]:
    ax.text(6.75, yy, s, fontsize=8.5, va="center", ha="left")
save(fig, "fig1_model")

# ---------------- Fig. 2: accuracy against size, and by practice gap ----------------
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": GRAY, "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6, "axes.axisbelow": True})
M = json.load(open(common.path(a, "results", "metrics.json")))
fig, (pa, pb) = plt.subplots(1, 2, figsize=(7.4, 2.9), gridspec_kw={"width_ratios": [1.1, 0.8]})
spec = [("BKT", "i", (6, -3)), ("PFA", "i", (-6, 2)), ("BKT-Forget", "i", (6, 0)), ("LR", "i", (5, -12)), ("SAKT", "d", (7, -3)),
        ("AKT", "d", (7, -9)), ("simpleKT", "d", (7, 2)), ("DKT", "d", (-7, 5)), ("DSFT", "o", (-6, 8))]
names = {"LR": "Logistic regression", "DSFT": "DSFT (ours)"}
for m, k, (dx, dy) in spec:
    mk, c, s = {"i": ("o", GRAY, 26), "d": ("s", GRAY, 24), "o": ("*", BLUE, 110)}[k]
    x, v, sd = M[m]["params"], M[m]["auc"], M[m]["sd"]
    pa.errorbar(x, v, yerr=sd, fmt="none", ecolor=c, elinewidth=0.9, capsize=2, zorder=2)
    pa.scatter(x, v, marker=mk, s=s, color=c, edgecolor="white", linewidth=0.7, zorder=3)
    pa.annotate(names.get(m, m), (x, v), xytext=(dx, dy), textcoords="offset points", ha="left" if dx > 0 else "right",
                fontsize=7.3, color=BLUE if k == "o" else INK, fontweight="bold" if k == "o" else "normal")
pa.set_xscale("log"); pa.set_xlim(220, 1.6e6); pa.set_ylim(0.665, 0.865)
pa.set_xlabel("Number of free parameters (log scale)"); pa.set_ylabel("AUC (five-fold mean ± SD)")
pa.scatter([], [], marker="o", color=GRAY, s=20, label="Interpretable"); pa.scatter([], [], marker="s", color=GRAY, s=18, label="Deep")
pa.legend(loc="lower right", frameon=False, fontsize=7); pa.set_title("(a) Accuracy against model size", fontsize=8.5, loc="left")
xs = np.arange(3)
for m, mk in [("DKT", "s"), ("simpleKT", "D"), ("AKT", "^"), ("LR", "v"), ("SAKT", "o")]:
    pb.plot(xs, [M[m]["auc"], M[m]["auc_gap1"], M[m]["auc_gap7"]], color=GRAY, lw=1, marker=mk, ms=3.8, label={"DKT": "DKT (tuned)", "SAKT": "SAKT (tuned)"}.get(m, names.get(m, m)))
pb.plot(xs, [M["DSFT"]["auc"], M["DSFT"]["auc_gap1"], M["DSFT"]["auc_gap7"]], color=BLUE, lw=2, marker="*", ms=8, label="DSFT (ours)", zorder=3)
h, l = pb.get_legend_handles_labels(); o = [l.index("DSFT (ours)")] + [i for i in range(len(l)) if l[i] != "DSFT (ours)"]
pb.legend([h[i] for i in o], [l[i] for i in o], loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=7, handlelength=2.2)
pb.set_xticks(xs); pb.set_xticklabels(["All\nattempts", "Gap >\n1 day", "Gap >\n7 days"]); pb.set_xlim(-0.25, 2.25)
pb.set_ylabel("AUC (five-fold mean)"); pb.grid(axis="x", visible=False)
pb.set_title("(b) Accuracy by gap since the last attempt", fontsize=8.5, loc="left")
fig.tight_layout(w_pad=3); save(fig, "fig2_accuracy")

# ---------------- Fig. 3: error runs within a take ----------------
z = data.load(a.data, a.quick)
d = pd.DataFrame({"r": z["rec"], "y": z["y"].astype(float)})
d["so_far"] = d.groupby("r").y.transform(lambda s: s.shift(1).expanding().mean())
bins = [(0, .5, "below 50%"), (.5, .75, "50-75%"), (.75, .9, "75-90%"), (.9, 1.01, "90% or above")]
acc = [100 * d[(d.so_far >= lo) & (d.so_far < hi)].y.mean() for lo, hi, _ in bins]
plt.rcParams.update({"font.size": 8.5})
fig, ax = plt.subplots(figsize=(4.8, 2.6))
bars = ax.bar([b[2] for b in bins], acc, color=BLUE, width=0.6)
for r_, v in zip(bars, acc):
    ax.text(r_.get_x() + r_.get_width() / 2, v + 1.5, f"{v:.1f}%", ha="center", fontsize=8)
ax.set_ylim(0, 100); ax.set_ylabel("Next note sung correctly (%)"); ax.set_xlabel("Accuracy of the take so far"); ax.grid(axis="x", visible=False)
save(fig, "fig3_error_runs")
common.log("wrote figures to " + common.path(a, "figures"))
