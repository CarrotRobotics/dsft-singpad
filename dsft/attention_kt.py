"""Attention-based KT baselines in JAX, re-implemented following the original designs (single attention block):
  SAKT     [Pandey & Karypis 2019]: query = current question embedding; keys/values = past interaction embeddings
           (question + response), learned positional embeddings, dot-product attention, FFN.
  AKT      [Ghosh et al. 2020]: Rasch-style embeddings (concept + difficulty * variation) and monotonic attention
           whose scores decay exponentially with the distance to past interactions (learned rate per head).
  simpleKT [Liu et al. 2023]: Rasch-style embeddings, plain dot-product attention, prediction from
           [retrieved knowledge ; current question] by an MLP.
Questions are notes (items); each note has one concept. Prediction at t uses interactions 1..t-1 only.
"""
import numpy as np
import jax, jax.numpy as jnp
import optax

D, H = 64, 4


def set_dim(d):
    """Embedding size used by init/forward (64 by default; tuned SAKT uses 128)."""
    global D
    D = d


def init(key, kind, n_item, n_kc, W):
    ks = iter(jax.random.split(key, 32))
    n = lambda *s: jax.random.normal(next(ks), s) * 0.05
    P = dict(Wq=n(D, D), Wk=n(D, D), Wv=n(D, D), Wo=n(D, D),
             F1=n(D, 2 * D), f1=jnp.zeros(2 * D), F2=n(2 * D, D), f2=jnp.zeros(D),
             O1=n(2 * D, D), o1=jnp.zeros(D), O2=n(D, 1), o2=jnp.zeros(1),
             null=n(D), ln1=jnp.ones(D), ln2=jnp.ones(D))
    if kind == "sakt":
        P.update(item=n(n_item, D), kc=n(n_kc, D), resp=n(2, D), posk=n(W, D))
    else:
        P.update(c=n(n_kc, D), d=n(n_kc, D), mu=jnp.zeros(n_item), e=n(2 * n_kc, D), f=n(2 * n_kc, D))
    if kind == "akt":
        P.update(theta=jnp.zeros(H))
    return P


def _sinusoid(T):
    pos = np.arange(T)[:, None]; i = np.arange(D)[None, :]
    ang = pos / np.power(10000, (2 * (i // 2)) / D)
    return jnp.asarray(np.where(i % 2 == 0, np.sin(ang), np.cos(ang)), jnp.float32)


def _ln(x, g):
    m = x.mean(-1, keepdims=True); v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / jnp.sqrt(v + 1e-5) * g


def forward(P, kind, it, kc, r):
    """it, kc, r: [B, T]. Returns logits [B, T] for each position given earlier positions."""
    B, T = it.shape
    if kind == "sakt":
        qe = P["item"][it] + P["kc"][kc]
        ie = qe + P["resp"][r] + P["posk"][:T][None]
        qe = qe + P["posk"][:T][None]
    else:
        mu = P["mu"][it][..., None]
        qe = P["c"][kc] + mu * P["d"][kc]
        ie = P["e"][kc * 2 + r] + mu * P["f"][kc * 2 + r] + qe
        if kind == "simplekt":   # simpleKT adds sinusoidal position encodings to both streams
            pe = _sinusoid(T)[None]
            qe, ie = qe + pe, ie + pe
    # keys/values: [null, interactions 0..T-1]; position t may see null and interactions j < t
    nullB = jnp.broadcast_to(P["null"], (B, 1, D))
    kin = jnp.concatenate([nullB, qe if kind != "sakt" else ie], 1)
    vin = jnp.concatenate([nullB, ie], 1)
    split = lambda x: x.reshape(B, -1, H, D // H).transpose(0, 2, 1, 3)
    q, k, v = split(qe @ P["Wq"]), split(kin @ P["Wk"]), split(vin @ P["Wv"])
    s = q @ k.transpose(0, 1, 3, 2) / np.sqrt(D // H)                      # [B, H, T, T+1]
    t_idx = jnp.arange(T)[:, None]; j_idx = jnp.arange(T + 1)[None, :] - 1  # j=-1 is null
    allowed = (j_idx < t_idx) | (j_idx == -1)
    if kind == "akt":
        dist = jnp.where(j_idx >= 0, t_idx - j_idx, 0).astype(jnp.float32)
        s = s - jax.nn.softplus(P["theta"])[None, :, None, None] * dist[None, None]
    s = jnp.where(allowed[None, None], s, -1e9)
    h = (jax.nn.softmax(s, -1) @ v).transpose(0, 2, 1, 3).reshape(B, T, D) @ P["Wo"]
    h = _ln(h + qe, P["ln1"])
    h = _ln(h + jax.nn.relu(h @ P["F1"] + P["f1"]) @ P["F2"] + P["f2"], P["ln2"])
    z = jax.nn.relu(jnp.concatenate([h, qe], -1) @ P["O1"] + P["o1"]) @ P["O2"] + P["o2"]
    return z[..., 0]


def user_seqs(user, item, kc, y, users):
    return [np.nonzero(user == u)[0] for u in users]


def windows(seqs, item, kc, y, W):
    out = []
    for idx in seqs:
        for s in range(0, len(idx), W):
            out.append(idx[s:s + W])
    I = np.zeros((len(out), W), np.int32); Kc = np.zeros_like(I); R = np.zeros_like(I); M = np.zeros((len(out), W), np.float32)
    for b, w in enumerate(out):
        n = len(w); I[b, :n] = item[w]; Kc[b, :n] = kc[w]; R[b, :n] = y[w]; M[b, :n] = 1
    return I, Kc, R, M


def fit(kind, seqs_fit, seqs_val, item, kc, y, n_item, n_kc, W=200, max_epochs=15, bs=64, seed=0, log=None, lr=1e-3):
    I, Kc, R, M = windows(seqs_fit, item, kc, y, W)
    P = init(jax.random.PRNGKey(seed), kind, n_item, n_kc, W)
    opt = optax.adamw(lr, weight_decay=1e-4); st = opt.init(P)

    def loss(P, i, k, r, m):
        z = forward(P, kind, i, k, r)
        return (optax.sigmoid_binary_cross_entropy(z, r.astype(jnp.float32)) * m).sum() / m.sum()

    @jax.jit
    def upd(P, st, i, k, r, m):
        v, g = jax.value_and_grad(loss)(P, i, k, r, m)
        u, st = opt.update(g, st, P)
        return optax.apply_updates(P, u), st, v

    rng = np.random.default_rng(seed)
    best, best_P, bad = -1, P, 0
    from sklearn.metrics import roc_auc_score
    for ep in range(max_epochs):
        perm = rng.permutation(len(I))
        for b in range(0, len(I), bs):
            j = perm[b:b + bs]
            P, st, v = upd(P, st, I[j], Kc[j], R[j], M[j])
        p = predict(P, kind, seqs_val, item, kc, y, len(y), W)
        va = np.concatenate(seqs_val)
        a = roc_auc_score(y[va], p[va])
        if log: log(f"      {kind} epoch {ep} val AUC {a:.4f}")
        if a > best + 1e-4:
            best, best_P, bad = a, P, 0
        else:
            bad += 1
            if bad >= 2: break
    return best_P


def predict(P, kind, seqs, item, kc, y, n_total, W=200):
    """Sliding windows with stride W/2 so every prediction (after the first W/2) sees >= W/2 history."""
    out = np.full(n_total, np.nan)
    f = jax.jit(forward, static_argnums=1)
    S = W // 2
    batch_i, batch_k, batch_r, targets = [], [], [], []

    def flush():
        if not batch_i: return
        z = np.asarray(jax.nn.sigmoid(f(P, kind, jnp.asarray(np.stack(batch_i)), jnp.asarray(np.stack(batch_k)), jnp.asarray(np.stack(batch_r)))))
        for b, (gidx, lo) in enumerate(targets):
            out[gidx] = z[b, lo:lo + len(gidx)]
        batch_i.clear(); batch_k.clear(); batch_r.clear(); targets.clear()

    for idx in seqs:
        starts = [0] + list(range(S, max(len(idx) - S, 1), S)) if len(idx) > W else [0]
        for s in starts:
            w = idx[s:s + W]; n = len(w)
            ii = np.zeros(W, np.int32); kk = np.zeros(W, np.int32); rr = np.zeros(W, np.int32)
            ii[:n], kk[:n], rr[:n] = item[w], kc[w], y[w]
            lo = 0 if s == 0 else S
            batch_i.append(ii); batch_k.append(kk); batch_r.append(rr); targets.append((w[lo:], lo))
            if len(batch_i) == 128: flush()
    flush()
    return out
