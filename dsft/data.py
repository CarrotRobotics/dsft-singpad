"""Loading SingPAD into flat note-level arrays, and the learner-level folds used throughout the paper."""
import json
import numpy as np
import pandas as pd

N_KC = 270


def prepare(raw_dir, out_path):
    """Read SingPAD's recordDS.csv and opernDS.csv and save one row per sung note.

    Rows are ordered by learner, take completion time, take id and note position, which every
    model relies on: a prediction for a row may only use rows of the same learner that come before it.
    Saved arrays: user, kc (0..269), item (note id, 0..2586), y (1 = correct), t (days since epoch),
    rec (take id), pos (note position in the take).
    """
    r = pd.read_csv(f"{raw_dir}/recordDS.csv", encoding="utf-8-sig")
    o = pd.read_csv(f"{raw_dir}/opernDS.csv", encoding="utf-8-sig")
    q2c = {}
    for _, row in o.iterrows():
        qs, cs = json.loads(row.qc_array)
        q2c.update(zip(qs, cs))
    r["t"] = pd.to_datetime(r.create_time, format="%Y/%m/%d %H:%M")
    r = r.sort_values(["user_id", "t", "record_id"], kind="stable")
    U, K, Q, Y, T, REC, POS = [], [], [], [], [], [], []
    for _, row in r.iterrows():
        qs, ys = json.loads(row.qa_array)
        day = row.t.value / 86400e9
        for i, (q, y) in enumerate(zip(qs, ys)):
            U.append(row.user_id); K.append(q2c[q]); Q.append(q); Y.append(y); T.append(day)
            REC.append(row.record_id); POS.append(i)
    kc_ids = {k: i for i, k in enumerate(np.unique(K))}
    q_ids = {q: i for i, q in enumerate(np.unique(Q))}
    np.savez_compressed(out_path, user=np.asarray(U), kc=np.array([kc_ids[k] for k in K]),
                        item=np.array([q_ids[q] for q in Q]), y=np.asarray(Y, np.int8), t=np.asarray(T),
                        rec=np.asarray(REC), pos=np.asarray(POS))
    return len(Y)


def load(path, quick=False):
    z = dict(np.load(path))
    if quick:   # a small learner subset for a fast end-to-end smoke test
        keep = np.isin(z["user"], np.random.default_rng(0).permutation(np.unique(z["user"]))[:150])
        z = {k: v[keep] for k, v in z.items()}
    z["y"] = z["y"].astype(np.int32)
    return z


def folds(user, k=5, seed=42):
    """Five learner-level folds; identical for every model."""
    us = np.random.default_rng(seed).permutation(np.unique(user))
    return us, np.array_split(us, k)


def train_val(us, test_users, seed, frac=0.1):
    """Training learners of a fold, shuffled, split into (fit, validation)."""
    tr = np.random.default_rng(seed).permutation(np.setdiff1d(us, test_users))
    nv = len(tr) // int(round(1 / frac))
    return tr[nv:], tr[:nv]


def rows(user, users):
    return np.nonzero(np.isin(user, users))[0]
