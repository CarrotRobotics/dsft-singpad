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

# ---------------- Fig. 2: accuracy and model size, and accuracy by practice gap ----------------
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8984", "axes.linewidth": 0.8, "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6, "axes.axisbelow": True})
M = json.load(open(common.path(a, "results", "metrics.json")))
MUTED = "#52514e"
# one colour per model, shared by both panels (categorical slots 1-6, validated for colour-vision deficiency)
COL = {"DSFT": "#2a78d6", "DKT": "#eb6834", "simpleKT": "#1baf7a", "AKT": "#eda100", "LR": "#e87ba4", "SAKT": "#008300"}
LABEL = {"DSFT": "DSFT (ours)", "DKT": "DKT (tuned)", "SAKT": "SAKT (tuned)", "LR": "Logistic regression"}
fig, (pa, pb) = plt.subplots(1, 2, figsize=(6.6, 3.3), gridspec_kw={"width_ratios": [1.0, 1.0]})
order = sorted(["BKT", "PFA", "BKT-Forget", "SAKT", "AKT", "LR", "simpleKT", "DKT", "DSFT"], key=lambda m: M[m]["auc"])
fmt = lambda p: f"{p / 1000:.1f}K" if p < 10000 else f"{p / 1000:.0f}K"
for i, m in enumerate(order):
    ours, c = m == "DSFT", COL.get(m, "#c9c8c2")
    v, sd = M[m]["auc"], M[m]["sd"]
    pa.barh(i, v - 0.5, left=0.5, height=0.62, color=c, zorder=2)
    pa.errorbar(v, i, xerr=sd, fmt="none", ecolor=INK, elinewidth=0.8, capsize=1.8, zorder=3)
    r, g, bb = (int(c[k:k + 2], 16) / 255 for k in (1, 3, 5))
    pa.text(v - sd - 0.006, i, f"{v:.3f}", va="center", ha="right", fontsize=6.8, zorder=4, fontweight="bold" if ours else "normal",
            color=INK if 0.2126 * r + 0.7152 * g + 0.0722 * bb > 0.45 else "white")
    pa.text(1.03, i, fmt(M[m]["params"]), transform=pa.get_yaxis_transform(), va="center", ha="left", fontsize=7.2,
            color=INK if ours else MUTED, fontweight="bold" if ours else "normal")
pa.text(1.03, len(order) - 0.25, "Params", transform=pa.get_yaxis_transform(), va="bottom", ha="left", fontsize=7.2, color=MUTED)
pa.set_yticks(range(len(order))); pa.set_yticklabels([LABEL.get(m, m) for m in order])
for tk, m in zip(pa.get_yticklabels(), order):
    if m == "DSFT": tk.set_color(COL["DSFT"]); tk.set_fontweight("bold")
pa.set_xlim(0.5, 0.88); pa.set_xticks([0.5, 0.6, 0.7, 0.8]); pa.set_ylim(-0.6, len(order) - 0.2)
pa.grid(axis="y", visible=False); pa.spines["left"].set_visible(False); pa.tick_params(axis="y", length=0)
pa.set_box_aspect(1); pa.set_xlabel("AUC (five-fold mean ± SD; 0.5 = chance)")
pa.set_title("(a) Accuracy and model size", fontsize=8.5, loc="left", color=INK)
xs = np.arange(3)
for m in ["SAKT", "LR", "AKT", "simpleKT", "DKT", "DSFT"]:
    pb.plot(xs, [M[m]["auc"], M[m]["auc_gap1"], M[m]["auc_gap7"]], color=COL[m], lw=2.4 if m == "DSFT" else 1.5,
            zorder=3 if m == "DSFT" else 2, label=LABEL.get(m, m), solid_capstyle="round")
h, l = pb.get_legend_handles_labels()
pb.legend(h[::-1], l[::-1], loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, frameon=False, fontsize=7.2,
          handlelength=1.8, columnspacing=1.4, labelspacing=0.35)
pb.set_xticks(xs); pb.set_xticklabels(["All attempts", "> 1 day", "> 7 days"]); pb.set_xlim(-0.3, 2.3)
pb.set_box_aspect(1); pb.set_ylabel("AUC (five-fold mean)"); pb.set_xlabel("Gap since the last attempt on the concept")
pb.grid(axis="x", visible=False); pb.set_title("(b) Accuracy by gap since the last attempt", fontsize=8.5, loc="left", color=INK)
fig.tight_layout(w_pad=3.5); save(fig, "fig2_accuracy")

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
