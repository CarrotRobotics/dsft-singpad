"""Long-term familiarity layer of DSFT, and the BKT-family baselines (JAX).

States are ordered from least to most familiar; each KC has its own parameters.
  S=2: (U, F)      S=3: (U, P, F)
Per attempt: forgetting over the gap since the previous attempt on this KC
(each state above the bottom drops one level w.p. d_s) -> predict -> Bayes
update with the outcome -> learning (each state below the top moves up one
level w.p. l_s).
Forgetting modes:
  none  : d_s = 0                                  (classic BKT)
  const : d_s = sigmoid(f_s) at every transition   (BKT-Forget style)
  time  : d_s = 1 - exp(-dt / tau_s), dt in days   (time-dependent forgetting ablation)
"""
import numpy as np
import jax, jax.numpy as jnp
import optax

jax.config.update("jax_enable_x64", False)


def _build(user, kc, y, t, users):
    """Pad per (user, KC) sequences into [N, L] arrays (time order kept)."""
    m = np.isin(user, users)
    gidx = np.nonzero(m)[0]
    u, k, yy, tt = user[m], kc[m], y[m], t[m]
    key = u.astype(np.int64) * 1000 + k
    order = np.lexsort((np.arange(len(key)), key))
    key, k, yy, tt, gidx = key[order], k[order], yy[order], tt[order], gidx[order]
    starts = np.r_[0, np.nonzero(np.diff(key))[0] + 1]
    lens = np.diff(np.r_[starts, len(key)])
    N, L = len(lens), lens.max()
    seq_id = np.repeat(np.arange(N), lens)
    step = np.arange(len(key)) - np.repeat(starts, lens)
    Y = np.zeros((N, L), np.float32); M = np.zeros((N, L), np.float32)
    DT = np.zeros((N, L), np.float32); G = np.full((N, L), -1, np.int64)
    Y[seq_id, step] = yy; M[seq_id, step] = 1; G[seq_id, step] = gidx
    dt = np.r_[0.0, np.diff(tt)]
    dt[step == 0] = 0.0
    DT[seq_id, step] = np.maximum(dt, 0.0)
    H = np.zeros((N, L), np.float32); H[seq_id, step] = (step > 0)
    return dict(kc=jnp.asarray(k[starts]), Y=jnp.asarray(Y.T), M=jnp.asarray(M.T),
                DT=jnp.asarray(DT.T), H=jnp.asarray(H.T), G=G.T, n=int(M.sum()))


def build_seqs(user, kc, y, t, users, edges=(4, 12, 32, 80, 160, 400)):
    """List of padded buckets grouped by (user, KC) sequence length."""
    m = np.isin(user, users)
    key = user.astype(np.int64) * 1000 + kc
    k_un, inv, cnt = np.unique(key[m], return_inverse=True, return_counts=True)
    seqlen = cnt[inv]
    out, lo = [], 0
    for hi in edges:
        sel = (seqlen > lo) & (seqlen <= hi)
        if sel.any():
            mm = np.zeros(len(user), bool); mm[np.nonzero(m)[0][sel]] = True
            out.append(_build_mask(user, kc, y, t, mm))
        lo = hi
    return out


def _build_mask(user, kc, y, t, m):
    return _build(np.where(m, user, -1), kc, y, t, np.unique(user[m]))


def init_params(n_kc, S, forget):
    pi = np.zeros((n_kc, S), np.float32); pi[:, 0] = 1.0
    th = np.full((n_kc, S), 0.5, np.float32); th[:, 0] = -0.5
    lr = np.full((n_kc, S - 1), -2.0, np.float32)
    fg = np.full((n_kc, S - 1), -3.0 if forget == "const" else np.log(7.0), np.float32)
    return dict(pi=jnp.asarray(pi), th=jnp.asarray(th), lr=jnp.asarray(lr), fg=jnp.asarray(fg))


def constrained(P):
    pi = jax.nn.softmax(P["pi"], -1)
    a = jax.nn.sigmoid(P["th"])
    th = [a[:, 0]]
    for s in range(1, a.shape[1]):
        th.append(th[-1] + (1 - th[-1]) * a[:, s])
    th = jnp.clip(jnp.stack(th, -1), 1e-4, 1 - 1e-4)
    return pi, th, jax.nn.sigmoid(P["lr"])


def make_run(forget):
    def run(P, D):
        pi, th, l = constrained(P)
        k = D["kc"]
        th_k, l_k, fg_k = th[k], l[k], P["fg"][k]
        if forget == "const":
            dconst = jax.nn.sigmoid(fg_k)
        tau = jnp.exp(fg_k)

        def step(prior, xs):
            y, m, dt, h = xs
            if forget == "none":
                a = prior
            else:
                d = dconst if forget == "const" else 1 - jnp.exp(-dt[:, None] / tau)
                d = d * (m * h)[:, None]               # no forgetting before the first attempt or on padding
                moved = prior[:, 1:] * d
                a = prior.at[:, 1:].add(-moved).at[:, :-1].add(moved)
            p = jnp.clip((a * th_k).sum(1), 1e-6, 1 - 1e-6)
            ll = m * (y * jnp.log(p) + (1 - y) * jnp.log(1 - p))
            like = jnp.where(y[:, None] > 0.5, th_k, 1 - th_k)
            post = a * like
            post = post / post.sum(1, keepdims=True)
            up = post[:, :-1] * l_k
            post = post.at[:, :-1].add(-up).at[:, 1:].add(up)
            new = jnp.where(m[:, None] > 0, post, prior)
            return new, (ll, p, a)

        # forgetting is applied using dt of the *current* attempt (gap since previous)
        _, (ll, p, a) = jax.lax.scan(step, pi[k], (D["Y"], D["M"], D["DT"], D["H"]))
        return ll.sum(), p, a
    return run


def fit(B, n_kc, S, forget, iters=150, lr=0.1, l2=1e-3, log=None):
    P = init_params(n_kc, S, forget)
    run = make_run(forget)

    n_tot = sum(D["n"] for D in B)
    arrs = [{k: v for k, v in D.items() if k not in ("G", "n")} for D in B]

    def loss(P):
        ll = sum(run(P, D)[0] for D in arrs)
        reg = sum((P[n] ** 2).sum() for n in ("pi", "th", "lr"))
        return (-ll + l2 * reg) / n_tot

    opt = optax.adam(lr)
    st = opt.init(P)

    @jax.jit
    def upd(P, st):
        v, g = jax.value_and_grad(loss)(P)
        u, st = opt.update(g, st, P)
        return optax.apply_updates(P, u), st, v

    for it in range(iters):
        P, st, v = upd(P, st)
        if log and (it % 50 == 0 or it == iters - 1):
            log(f"    it {it:3d} loss {float(v):.4f}")
    return P


def predict(P, B, forget, n_total):
    run = jax.jit(make_run(forget))
    out = np.full(n_total, np.nan)
    for D in B:
        _, p, _ = run(P, {k: v for k, v in D.items() if k not in ("G", "n")})
        G, p = D["G"], np.asarray(p)
        sel = G >= 0
        out[G[sel]] = p[sel]
    return out
