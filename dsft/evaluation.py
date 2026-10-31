"""Per-fold AUC / RMSE and the learner-level bootstrap used for every comparison in the paper."""
import numpy as np
from sklearn.metrics import roc_auc_score


def fold_auc(y, p, fold, subset=None):
    sub = np.ones(len(y), bool) if subset is None else subset
    return np.array([roc_auc_score(y[sub & (fold == f)], p[sub & (fold == f)]) for f in np.unique(fold[fold >= 0])])


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def bootstrap_diff(y, user, pa, pb, n_boot=300, seed=7):
    """AUC(pa) - AUC(pb) on pooled out-of-fold predictions; learners resampled with replacement.
    Returns (difference on the full data, 2.5th percentile, 97.5th percentile)."""
    uu = np.unique(user); ui = np.searchsorted(uu, user)
    rows = np.split(np.argsort(ui, kind="stable"), np.cumsum(np.bincount(ui))[:-1])
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        s = np.concatenate([rows[j] for j in rng.integers(0, len(uu), len(uu))])
        d.append(roc_auc_score(y[s], pa[s]) - roc_auc_score(y[s], pb[s]))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(roc_auc_score(y, pa) - roc_auc_score(y, pb)), float(lo), float(hi)
