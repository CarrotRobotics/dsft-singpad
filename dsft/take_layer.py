"""Short-term take layer of DSFT: C take states, per-note difficulty, optional learner-ability features.

P(y_j = 1 | c_j) = sigmoid(w * logit(p_fam_j) + b_{c_j} + beta_{item_j} + g . a_u)
The take state c follows a C-state Markov chain shared by all learners; prediction for note j
uses the filtered distribution over c_j given earlier notes of the same take only.
"""
import numpy as np
import jax, jax.numpy as jnp
import optax


def build_takes(rec, pos, idx):
    r, p = rec[idx], pos[idx]
    o = np.lexsort((p, r))
    gi, rr = idx[o], r[o]
    starts = np.r_[0, np.nonzero(rr[1:] != rr[:-1])[0] + 1]
    lens = np.diff(np.r_[starts, len(rr)])
    T, n = lens.max(), len(lens)
    tid = np.repeat(np.arange(n), lens)
    step = np.arange(len(rr)) - np.repeat(starts, lens)
    G = np.full((T, n), -1, np.int64)
    G[step, tid] = gi
    return G


def arrays(G, y, pf, item, feats=None):
    m = G >= 0
    Gs = np.where(m, G, 0)
    pfc = np.clip(pf[Gs], 1e-4, 1 - 1e-4)
    d = dict(Y=jnp.asarray(np.where(m, y[Gs], 0).astype(np.float32)),
             Z=jnp.asarray(np.where(m, np.log(pfc / (1 - pfc)), 0).astype(np.float32)),
             I=jnp.asarray(np.where(m, item[Gs], 0).astype(np.int32)),
             M=jnp.asarray(m.astype(np.float32)))
    k = 0 if feats is None else feats.shape[1]
    d["F"] = jnp.asarray(np.where(m[..., None], feats[Gs], 0).astype(np.float32)) if k else jnp.zeros(m.shape + (1,), jnp.float32)
    return d


def init(C, n_item, k=1):
    A = np.full((C, C), -2.0, np.float32); np.fill_diagonal(A, 2.0)
    pi = np.zeros(C, np.float32); pi[0] = 1.5
    return dict(pi=jnp.asarray(pi), A=jnp.asarray(A), b=jnp.asarray(np.linspace(1.5, -1.5, C).astype(np.float32)),
                w=jnp.array(1.0), beta=jnp.zeros(n_item), g=jnp.zeros(k))


def run(P, D, use_item):
    C = P["b"].shape[0]
    A = jax.nn.softmax(P["A"], -1)
    n = D["Y"].shape[1]
    a0 = jnp.tile(jax.nn.softmax(P["pi"])[None, :], (n, 1))

    def step(a, xs):
        y, z, it, m, f = xs
        logits = P["w"] * z[:, None] + P["b"][None, :] + (f @ P["g"])[:, None]
        if use_item:
            logits = logits + P["beta"][it][:, None]
        pc = jax.nn.sigmoid(logits)
        p = jnp.clip((a * pc).sum(1), 1e-6, 1 - 1e-6)
        ll = m * (y * jnp.log(p) + (1 - y) * jnp.log(1 - p))
        post = a * jnp.where(y[:, None] > 0.5, pc, 1 - pc)
        post = post / post.sum(1, keepdims=True)
        return jnp.where(m[:, None] > 0, post @ A, a), (ll, p, a)

    _, (ll, p, a) = jax.lax.scan(step, a0, (D["Y"], D["Z"], D["I"], D["M"], D["F"]))
    return ll.sum(), p, a


def fit(D, C, n_item, use_item, iters=300, lr=0.05, l2_item=1.0):
    P = init(C, n_item, D["F"].shape[-1])
    n = float(D["M"].sum())
    opt = optax.adam(lr); st = opt.init(P)

    def loss(P):
        ll = run(P, D, use_item)[0]
        return (-ll + l2_item * (P["beta"] ** 2).sum()) / n

    @jax.jit
    def upd(P, st):
        v, g = jax.value_and_grad(loss)(P)
        u, st = opt.update(g, st, P)
        return optax.apply_updates(P, u), st, v

    for _ in range(iters):
        P, st, v = upd(P, st)
    return P


def predict(P, G, D, use_item, n_total):
    _, p, a = jax.jit(run, static_argnums=2)(P, D, use_item)
    out = np.full(n_total, np.nan); state = np.full((n_total, P["b"].shape[0]), np.nan)
    m = G >= 0
    out[G[m]] = np.asarray(p)[m]; state[G[m]] = np.asarray(a)[m]
    return out, state
