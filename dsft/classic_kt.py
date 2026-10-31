"""PFA and DKT baselines (JAX)."""
import numpy as np
import jax, jax.numpy as jnp
import optax
from sklearn.metrics import roc_auc_score


# ---------------- PFA ----------------
def pfa_fit_predict(kc, y, s, f, tr, te, n_kc, iters=400):
    """Per-concept logistic model on log prior successes (s) and failures (f)."""
    P = dict(b=jnp.zeros(n_kc), g=jnp.zeros(n_kc), r=jnp.zeros(n_kc))
    K, Y, S, F = map(jnp.asarray, (kc[tr], y[tr].astype(np.float32), s[tr], f[tr]))
    logit = lambda P, K, S, F: P["b"][K] + P["g"][K] * S + P["r"][K] * F
    loss = lambda P: optax.sigmoid_binary_cross_entropy(logit(P, K, S, F), Y).mean()
    opt = optax.adam(0.05); st = opt.init(P)

    @jax.jit
    def upd(P, st):
        v, g = jax.value_and_grad(loss)(P)
        u, st = opt.update(g, st, P)
        return optax.apply_updates(P, u), st, v

    for _ in range(iters):
        P, st, _ = upd(P, st)
    return np.asarray(jax.nn.sigmoid(logit(P, jnp.asarray(kc[te]), jnp.asarray(s[te]), jnp.asarray(f[te]))))


# ---------------- DKT ----------------
def _lstm_init(key, n_in, n_h, n_out):
    k1, k2, k3 = jax.random.split(key, 3)
    sc = 1 / np.sqrt(n_h)
    return dict(emb=jax.random.normal(k1, (n_in, 64)) * 0.1,
                W=jax.random.uniform(k2, (64 + n_h, 4 * n_h), minval=-sc, maxval=sc),
                bW=jnp.zeros(4 * n_h).at[n_h:2 * n_h].set(1.0),
                O=jax.random.uniform(k3, (n_h, n_out), minval=-sc, maxval=sc),
                bO=jnp.zeros(n_out))


def _dkt_logits(P, tok, qk, h0=None):
    """tok, qk: [T, B]. The state before consuming token t predicts the queried concept at t."""
    B = tok.shape[1]; n_h = P["O"].shape[0]
    if h0 is None:
        h0 = (jnp.zeros((B, n_h)), jnp.zeros((B, n_h)))

    def cell(c, x):
        h, cc = c
        t, q = x
        z = jnp.concatenate([P["emb"][t], h], 1) @ P["W"] + P["bW"]
        i, fg, g, o = jnp.split(z, 4, 1)
        cc = jax.nn.sigmoid(fg) * cc + jax.nn.sigmoid(i) * jnp.tanh(g)
        h = jax.nn.sigmoid(o) * jnp.tanh(cc)
        return (h, cc), (h @ P["O"] + P["bO"])[jnp.arange(B), q]

    hT, out = jax.lax.scan(cell, h0, (tok, qk))
    return out, hT


def dkt_sequences(user, kc, y, users, n_kc):
    """Per learner: input token at t = previous attempt (2*kc + y); start token = 2*n_kc."""
    seqs = []
    for u in users:
        idx = np.nonzero(user == u)[0]
        prev = np.r_[2 * n_kc, kc[idx][:-1] * 2 + y[idx][:-1].astype(int)]
        seqs.append((idx, prev, kc[idx], y[idx]))
    return seqs


def _windows(seqs, W):
    tok, q, yy, m = [], [], [], []
    for idx, prev, k, y in seqs:
        for s in range(0, len(idx), W):
            n = len(prev[s:s + W]); pad = W - n
            tok.append(np.r_[prev[s:s + W], np.zeros(pad, int)]); q.append(np.r_[k[s:s + W], np.zeros(pad, int)])
            yy.append(np.r_[y[s:s + W], np.zeros(pad)]); m.append(np.r_[np.ones(n), np.zeros(pad)])
    f = lambda a, dt: np.asarray(a, dt).T
    return f(tok, np.int32), f(q, np.int32), f(yy, np.float32), f(m, np.float32)


def dkt_predict(P, seqs_te, n_total, W=200):
    """Full-history inference: the hidden state is carried across windows of length W."""
    out = np.full(n_total, np.nan)
    f = jax.jit(_dkt_logits)
    for idx, prev, k, y in seqs_te:
        h = None
        for s in range(0, len(idx), W):
            tk = jnp.asarray(prev[s:s + W][:, None], jnp.int32); qq = jnp.asarray(k[s:s + W][:, None], jnp.int32)
            pad = W - len(tk)
            if pad:
                tk = jnp.pad(tk, ((0, pad), (0, 0))); qq = jnp.pad(qq, ((0, pad), (0, 0)))
            z, h = f(P, tk, qq, h)
            out[idx[s:s + W]] = np.asarray(jax.nn.sigmoid(z[:W - pad, 0]))
    return out


def dkt_fit(seq_fit, seq_val, y, n_kc, n_h=100, lr=3e-3, seed=0, max_epochs=20, patience=3, bs=128, W=200, n_total=None):
    """Adam training with early stopping on validation AUC. Returns (params, best val AUC, best epoch)."""
    tok, q, yy, m = _windows(seq_fit, W)
    P = _lstm_init(jax.random.PRNGKey(seed), 2 * n_kc + 1, n_h, n_kc)
    opt = optax.adam(lr); st = opt.init(P)
    n_total = n_total or len(y)
    va = np.concatenate([s[0] for s in seq_val])

    def loss(P, tk, qq, y_, mm):
        zz, _ = _dkt_logits(P, tk, qq)
        return (optax.sigmoid_binary_cross_entropy(zz, y_) * mm).sum() / mm.sum()

    @jax.jit
    def upd(P, st, tk, qq, y_, mm):
        v, g = jax.value_and_grad(loss)(P, tk, qq, y_, mm)
        g = jax.tree_util.tree_map(lambda x: jnp.clip(x, -1, 1), g)
        u, st = opt.update(g, st, P)
        return optax.apply_updates(P, u), st, v

    rng = np.random.default_rng(seed)
    best, bestP, bad, best_ep = -1, P, 0, 0
    for ep in range(max_epochs):
        perm = rng.permutation(tok.shape[1])
        for b in range(0, len(perm) - bs + 1, bs):
            j = perm[b:b + bs]
            P, st, _ = upd(P, st, tok[:, j], q[:, j], yy[:, j], m[:, j])
        a = roc_auc_score(y[va], dkt_predict(P, seq_val, n_total)[va])
        if a > best + 1e-4:
            best, bestP, bad, best_ep = a, P, 0, ep + 1
        else:
            bad += 1
            if bad >= patience:
                break
    return bestP, best, best_ep
