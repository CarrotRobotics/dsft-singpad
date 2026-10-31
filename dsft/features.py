"""Hand-crafted features: PFA counts, in-take context, and learner ability from earlier takes.
All features for a note use only that learner's earlier notes (data must be in the order produced by data.prepare)."""
import numpy as np


def pfa_counts(user, kc, y):
    """log(1 + prior successes) and log(1 + prior failures) of this learner on this concept."""
    key = user.astype(np.int64) * 1000 + kc
    order = np.lexsort((np.arange(len(key)), key))
    ks, ys = key[order], y[order].astype(float)
    first = np.r_[True, ks[1:] != ks[:-1]]
    grp = np.cumsum(first) - 1
    cs, cn = np.cumsum(ys), np.arange(1, len(ys) + 1)
    st = np.nonzero(first)[0]
    succ = (cs - ys) - (cs - ys)[st][grp]
    tot = (cn - 1) - (cn - 1)[st][grp]
    s = np.empty(len(y)); f = np.empty(len(y))
    s[order] = succ; f[order] = tot - succ
    return np.log1p(s).astype(np.float32), np.log1p(f).astype(np.float32)


def in_take(rec, y):
    """[previous note correct, previous note wrong, accuracy of the take so far, log(1 + notes so far)]."""
    y = y.astype(float)
    same = np.r_[False, rec[1:] == rec[:-1]]
    prev = np.r_[0.0, y[:-1]]
    grp = np.cumsum(~same) - 1
    st = np.nonzero(~same)[0]
    cs = np.cumsum(y)
    nb = np.arange(len(y)) - st[grp]
    cb = (cs - y) - (cs - y)[st][grp]
    return np.column_stack([np.where(same, prev, 0), np.where(same, 1 - prev, 0),
                            np.where(nb > 0, cb / np.maximum(nb, 1), 0.75), np.log1p(nb)]).astype(np.float32)


def ability(user, rec, y, prior=10.0, base_rate=0.75, decay=0.7):
    """Learner ability a_u for every note, from the learner's earlier takes only (never the current take):
    smoothed log-odds of cumulative accuracy, of exponentially weighted recent accuracy (decay per take),
    and log(1 + notes sung before). The first two are centred on the prior log-odds. Settings fixed in advance."""
    y = y.astype(float)
    new_take = np.r_[True, rec[1:] != rec[:-1]]
    take_id = np.cumsum(new_take) - 1
    st = np.nonzero(new_take)[0]
    tu, tn, tc = user[st], np.diff(np.r_[st, len(y)]), np.add.reduceat(y, st)
    cc, cn, ec, en = (np.zeros(len(st)) for _ in range(4))
    for i in range(1, len(st)):
        if tu[i] == tu[i - 1]:
            cc[i], cn[i] = cc[i - 1] + tc[i - 1], cn[i - 1] + tn[i - 1]
            ec[i], en[i] = decay * ec[i - 1] + tc[i - 1], decay * en[i - 1] + tn[i - 1]
    lg = lambda c, n: np.log((c + prior * base_rate) / (n - c + prior * (1 - base_rate)))
    base = lg(0, 0)
    per_take = np.column_stack([lg(cc, cn) - base, lg(ec, en) - base, np.log1p(cn)])
    return per_take[take_id].astype(np.float32)


def gap_days(user, kc, t):
    """Days since the learner's previous attempt on the same concept (NaN for a first attempt)."""
    key = user.astype(np.int64) * 1000 + kc
    o = np.lexsort((np.arange(len(key)), key))
    gap = np.full(len(key), np.nan)
    g = np.r_[np.nan, np.diff(t[o])]
    g[np.r_[True, key[o][1:] != key[o][:-1]]] = np.nan
    gap[o] = g
    return gap
