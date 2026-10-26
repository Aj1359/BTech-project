"""
FTRW -- Fairness-Targeted Random Walk for truth-seed selection (id: ftrw)

Implements the protected-set strategies from the BTP proposal document.
Steps 3-7 of the greedy loop are identical across all strategies;
only the definition of the protected set P changes.

GROUP_MODE OPTIONS (select via kwarg group_mode=):
  'quantile'    (default) P = worst p_frac share by current sp. Original rule.
  'stat_band'   Strategy 1: Statistically-tied band.
                  P = nodes whose sp confidence interval overlaps s*'s CI.
                  Specifically: sp - z*se <= s* + z*se(v*) where se=sqrt(sp(1-sp)/R).
                  Soft weights: w(v) = exp(-(sp-s*)/beta). Clipped to [k_min, p_max*|Vvul|].
  'rescuable'   Strategy 2: Rescuability filter (applied ON TOP of strategy 1 or quantile).
                  Drops "pinned" nodes from P that (a) are within d0 hops of M (so they
                  can never be saved because M arrives first), OR (b) showed no sp improvement
                  over a stagnation window w. Dropped nodes stay in the candidate pool but
                  not in the restart/walk mass.  Falls back to quantile P if filter empties it.
  'dual'        Strategy 4: Multiplicative-weights dual.
                  Persistent per-node weights λ updated each round as
                  λ ← clip(λ^rho * exp(-eta_d*(sp - s*)), lo, hi).
                  P is the smallest set carrying (1-alpha_d) of the λ mass (high-λ nodes).

Strategy 3 (community-stratified) is not implemented here; it requires community labels
and is a comparator variant best tested in ftrw_v3.

Parameters (all optional, pass via **kwargs):
  L=25            shortlist size
  R_cand=150      simulations per shortlisted candidate
  R_base=500      simulations for per-round sp estimate
  group_mode='quantile' | 'stat_band' | 'rescuable' | 'dual'
  p_frac=0.20     protected share (quantile / fallback)
  theta=None      fixed sp threshold instead of quantile
  phi_jump=0.80   FSPR mass share toward P
  phi_walk=0.70   LFPR local mass share toward P (None = plain PPR)
  eta=0.7, tau=2.0, eps=0.05, gamma=0.15, tail_frac=0.05
  -- Strategy 1 (stat_band) --
  z=1.96          z-score for the CI (95%)
  beta=0.02       temperature for soft weights
  p_max=0.40      upper clip for |P| / |Vvul|
  k_min=10        minimum |P|
  -- Strategy 2 (rescuable) --
  d0=2            hops from M inside which nodes are considered pinned
  stag_window=3   rounds of history to check for sp stagnation
  stag_eps=0.02   minimum sp improvement to count as "moving"
  -- Strategy 4 (dual) --
  rho=0.95        decay of the λ history
  eta_d=20.0      sensitivity to how far sp is above s*
  alpha_d=0.20    (1-alpha_d) of mass defines P
"""
import numpy as np
import networkx as nx
import scipy.sparse as sps
from collections import deque


# ---------------------------------------------------------------------------
# Graph utilities
# ---------------------------------------------------------------------------

def _dist_from(G, sources):
    dist = {s: 0 for s in sources}
    dq = deque(sources)
    while dq:
        u = dq.popleft()
        for v in (G.successors(u) if G.is_directed() else G.neighbors(u)):
            if v not in dist:
                dist[v] = dist[u] + 1
                dq.append(v)
    return dist


# ---------------------------------------------------------------------------
# Walk kernels
# ---------------------------------------------------------------------------

def _lfpr_walk(A, AT, v, gP, gamma, phi, iters=300, tol=1e-9):
    """LFPR_N power iteration (group-normalised transitions)."""
    gU = 1.0 - gP
    nP, nU = gP.sum(), gU.sum()
    if nP == 0 or nU == 0:
        phi = 1.0 if nU == 0 else 0.0
    outP = A @ gP
    outU = A @ gU
    invP = np.where(outP > 0, 1.0 / np.maximum(outP, 1e-12), 0.0)
    invU = np.where(outU > 0, 1.0 / np.maximum(outU, 1e-12), 0.0)
    danP, danU = outP == 0, outU == 0
    p = v.copy()
    for _ in range(iters):
        cP = AT @ (p * invP) + p[danP].sum() / max(nP, 1.0)
        cU = AT @ (p * invU) + p[danU].sum() / max(nU, 1.0)
        new = gamma * v + (1.0 - gamma) * (phi * gP * cP + (1.0 - phi) * gU * cU)
        done = np.abs(new - p).sum() < tol
        p = new
        if done:
            break
    return p / p.sum()


def _plain_ppr(A, AT, v, gamma, iters=300, tol=1e-9):
    """Standard personalised PageRank (ablation, phi_walk=None)."""
    deg = np.asarray(A.sum(axis=1)).ravel()
    inv = np.where(deg > 0, 1.0 / np.maximum(deg, 1e-12), 0.0)
    dang = deg == 0
    p = v.copy()
    for _ in range(iters):
        new = gamma * v + (1.0 - gamma) * (AT @ (p * inv) + p[dang].sum() * v)
        done = np.abs(new - p).sum() < tol
        p = new
        if done:
            break
    return p / p.sum()


# ---------------------------------------------------------------------------
# Protected-set strategies (Steps 1–2 only; steps 3–7 are shared)
# ---------------------------------------------------------------------------

def _group_quantile(sp, vul_idx, p_frac, theta, R_base, n):
    """Original strategy: worst p_frac quantile."""
    sp_vul = sp[vul_idx]
    if theta is None:
        cut = float(np.quantile(sp_vul, p_frac))
    else:
        cut = float(theta) - 1.0 / R_base
    in_P = np.zeros(n, dtype=bool)
    in_P[vul_idx[sp_vul <= cut]] = True
    # soft weights: uniform inside P
    w = in_P.astype(float)
    return in_P, w, cut


def _group_stat_band(sp, vul_idx, R_base, z, beta, p_max, k_min, n):
    """
    Strategy 1: Statistically-tied band.
    P = nodes v where  sp(v) - z*se(v) <= s* + z*se(v*)
    with se(v) = sqrt(sp(v)*(1-sp(v)) / R_base).
    Soft weight: exp(-(sp-s*)/beta).
    Clipped to at most p_max*|Vvul| nodes, at least k_min nodes.
    """
    sp_vul = sp[vul_idx]
    s_star = float(sp_vul.min())

    se = np.sqrt(np.clip(sp_vul * (1.0 - sp_vul), 0.0, None) / max(R_base, 1))
    # index of the worst node (argmin with tie-breaking by index)
    worst_idx = int(np.argmin(sp_vul))
    se_star = se[worst_idx]

    # band condition: lower CI of v overlaps upper CI of v*
    in_band = (sp_vul - z * se) <= (s_star + z * se_star)

    # clip size
    n_vul = len(vul_idx)
    max_size = max(k_min, int(p_max * n_vul))
    if in_band.sum() > max_size:
        # keep the max_size closest to s_star
        order = np.argsort(sp_vul[in_band])
        chosen = np.where(in_band)[0][order[:max_size]]
        in_band_clipped = np.zeros(n_vul, dtype=bool)
        in_band_clipped[chosen] = True
        in_band = in_band_clipped
    elif in_band.sum() < k_min:
        # force at least k_min by taking the k_min worst
        order = np.argsort(sp_vul)
        in_band = np.zeros(n_vul, dtype=bool)
        in_band[order[:k_min]] = True

    in_P = np.zeros(n, dtype=bool)
    in_P[vul_idx[in_band]] = True

    # soft weights
    sp_full = sp.copy()
    w = np.zeros(n)
    w[vul_idx[in_band]] = np.exp(
        -np.clip(sp_vul[in_band] - s_star, 0.0, None) / max(beta, 1e-9)
    )
    cut = float(np.max(sp_vul[in_band])) if in_band.any() else s_star

    return in_P, w, cut


def _group_rescuable(sp, vul_idx, dist_M, d0, sp_history, stag_window, stag_eps, n,
                     fallback_in_P, fallback_w, fallback_cut):
    """
    Strategy 2: Filter 'pinned' nodes from P.
    A node is pinned if:
      (a) it is within d0 hops of M  (geometry check), OR
      (b) its sp hasn't improved by stag_eps over the last stag_window rounds.
    Dropped nodes stay in the candidate pool but not in the restart/walk mass.
    Falls back to quantile P if the filter empties it.
    """
    sp_vul = sp[vul_idx]

    # (a) geometry: too close to M to be saved before M arrives
    close_to_M = dist_M[vul_idx] <= d0

    # (b) stagnation: sp hasn't moved
    stagnant = np.zeros(len(vul_idx), dtype=bool)
    if len(sp_history) >= stag_window:
        old_sp = sp_history[-stag_window][vul_idx]
        stagnant = (sp_vul - old_sp) < stag_eps

    pinned = close_to_M | stagnant

    # P = fallback_P minus pinned
    in_P_base = fallback_in_P.copy()
    # remove pinned from P
    in_P_base[vul_idx[pinned]] = False

    if in_P_base.any():
        sp_vul_P = sp[vul_idx[in_P_base[vul_idx]]]
        cut = float(sp_vul_P.max()) if len(sp_vul_P) > 0 else fallback_cut
        w = in_P_base.astype(float)
        return in_P_base, w, cut
    else:
        # fallback: use the full quantile set (stagnation filter too aggressive)
        return fallback_in_P, fallback_w, fallback_cut


def _group_dual(sp, vul_idx, lam, rho, eta_d, alpha_d, n):
    """
    Strategy 4: Multiplicative-weights dual.
    Update: lam <- lam^rho * exp(-eta_d * (sp - s*))
    P = smallest set (by descending λ) carrying at least (1-alpha_d) of total λ mass.
    Returns updated lam.
    """
    sp_vul = sp[vul_idx]
    s_star = float(sp_vul.min())

    # update λ for vulnerable nodes
    lam[vul_idx] = np.clip(
        lam[vul_idx] ** rho * np.exp(-eta_d * np.clip(sp_vul - s_star, 0.0, None)),
        1e-12, 1e6
    )
    # normalise
    total = lam[vul_idx].sum()
    if total <= 0:
        lam[vul_idx] = 1.0

    # P = high-λ nodes totalling (1-alpha_d) of mass
    lam_vul = lam[vul_idx]
    order = np.argsort(-lam_vul)  # descending
    cum = np.cumsum(lam_vul[order]) / lam_vul.sum()
    cutoff = np.searchsorted(cum, 1.0 - alpha_d) + 1
    cutoff = max(1, min(cutoff, len(order)))
    chosen = order[:cutoff]

    in_P = np.zeros(n, dtype=bool)
    in_P[vul_idx[chosen]] = True

    w = np.zeros(n)
    w[vul_idx[chosen]] = lam_vul[chosen]
    if w[in_P].sum() > 0:
        w[in_P] /= w[in_P].sum()

    sp_vul_P = sp_vul[chosen]
    cut = float(sp_vul_P.max()) if len(sp_vul_P) > 0 else s_star

    return in_P, w, cut, lam


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def get_seeds(G, M, budget, **kw):
    M = set(M)
    est = kw.get('estimate_sp')
    if est is None:
        raise ValueError("ftrw needs estimate_sp")

    # ── core hyper-parameters ──────────────────────────────────────────────
    model   = kw.get('model', 'COICM')
    pM, pT  = kw.get('pM', 0.1), kw.get('pT', 0.1)
    R       = int(kw.get('R', 1000))
    L       = int(kw.get('L', 25))
    R_cand  = int(kw.get('R_cand', min(150, R)))
    R_base  = int(kw.get('R_base', min(500, R)))
    p_frac  = float(kw.get('p_frac', 0.20))
    theta   = kw.get('theta', None)
    phi_jump = float(kw.get('phi_jump', 0.80))
    phi_walk = kw.get('phi_walk', 0.70)
    eta     = float(kw.get('eta', 0.7))
    tau     = float(kw.get('tau', 2.0))
    eps     = float(kw.get('eps', 0.05))
    gamma   = float(kw.get('gamma', 0.15))
    tail_frac = float(kw.get('tail_frac', 0.05))
    seed0   = int(kw.get('rng_seed', 42)) * 1000 + 17

    # ── group_mode ────────────────────────────────────────────────────────
    group_mode = kw.get('group_mode', 'quantile')   # 'quantile','stat_band','rescuable','dual'

    # Strategy 1 params
    z       = float(kw.get('z', 1.96))
    beta    = float(kw.get('beta', 0.02))
    p_max   = float(kw.get('p_max', 0.40))
    k_min   = int(kw.get('k_min', 10))

    # Strategy 2 params
    d0           = int(kw.get('d0', 2))
    stag_window  = int(kw.get('stag_window', 3))
    stag_eps     = float(kw.get('stag_eps', 0.02))

    # Strategy 4 params
    rho     = float(kw.get('rho', 0.95))
    eta_d   = float(kw.get('eta_d', 20.0))
    alpha_d = float(kw.get('alpha_d', 0.20))

    # ── graph setup ────────────────────────────────────────────────────────
    nodes = list(G.nodes())
    idx   = {v: i for i, v in enumerate(nodes)}
    n     = len(nodes)

    try:
        from sim_engine import GraphCSR
        graph_for_sim = kw.get('csr') or GraphCSR(G)
    except Exception:
        graph_for_sim = G

    def sim(T, runs, seed):
        return np.asarray(est(graph_for_sim, set(M), set(T),
                              model_type=model, pM=pM, pT=pT,
                              R_runs=runs, base_seed=seed), dtype=float)

    # vulnerable set: fixed from baseline (matches the evaluation objective)
    vul = kw.get('vulnerable_set')
    if vul is None:
        sp0 = sim(set(), R_base, seed0)
        vul = [v for v in nodes if v not in M and sp0[idx[v]] < 1.0]
    vul_idx = np.array(sorted(idx[v] for v in vul if v not in M), dtype=int)
    if len(vul_idx) == 0:
        return []

    # proximity to M
    dm   = _dist_from(G, list(M))
    dist = np.array([dm.get(v, np.inf) for v in nodes], dtype=float)
    prox = np.where(np.isfinite(dist), np.exp(-dist / tau), 0.0)
    # raw hop distance (for rescuable filter)
    dist_M = np.array([dm.get(v, np.inf) for v in nodes], dtype=float)

    # sparse walk matrices
    Gw = G.reverse(copy=False) if G.is_directed() else G
    A  = sps.csr_matrix(nx.to_scipy_sparse_array(Gw, nodelist=nodes, weight=None, format='csr'))
    AT = A.T.tocsr()

    # state
    blocked = np.zeros(n, dtype=bool)
    for v in M:
        blocked[idx[v]] = True
    T = []

    # Strategy 2: sp history ring buffer
    sp_history = deque(maxlen=stag_window + 1)

    # Strategy 4: persistent λ weights (initialised to 1)
    lam = np.ones(n, dtype=float)

    for it in range(min(budget, n - int(blocked.sum()))):
        seed_it = seed0 + 7919 * (it + 1)

        # 1. Current save probabilities
        sp = sim(T, R_base, seed_it)
        sp_history.append(sp.copy())   # for Strategy 2 stagnation check

        # 2. Compute protected set P using the chosen strategy ────────────
        # Always compute the quantile fallback (needed as rescuable fallback)
        fb_in_P, fb_w, fb_cut = _group_quantile(sp, vul_idx, p_frac, theta, R_base, n)

        if group_mode == 'quantile':
            in_P, w_P, cut = fb_in_P, fb_w, fb_cut

        elif group_mode == 'stat_band':
            in_P, w_P, cut = _group_stat_band(
                sp, vul_idx, R_base, z, beta, p_max, k_min, n)

        elif group_mode == 'rescuable':
            # Use stat_band as the base, then filter pinned nodes
            base_in_P, base_w, base_cut = _group_stat_band(
                sp, vul_idx, R_base, z, beta, p_max, k_min, n)
            in_P, w_P, cut = _group_rescuable(
                sp, vul_idx, dist_M, d0, sp_history, stag_window, stag_eps, n,
                base_in_P, base_w, base_cut)

        elif group_mode == 'dual':
            in_P, w_P, cut, lam = _group_dual(
                sp, vul_idx, lam, rho, eta_d, alpha_d, n)

        else:
            raise ValueError(f"Unknown group_mode: {group_mode!r}")

        # ── Steps 3–7: identical for all strategies ───────────────────────

        # 3. Deficit × proximity weights (use cut from selected strategy)
        d = np.where(in_P, np.maximum(0.0, cut + 1.0 / R_base - sp), 0.0)
        h = d * (eta + (1.0 - eta) * prox)

        # Blend with soft per-node weights from strategy if non-uniform
        if w_P.sum() > 0 and not np.allclose(w_P[in_P], w_P[in_P].mean()):
            h[in_P] *= w_P[in_P]

        # 4. FSPR-style jump vector
        in_Uvul = np.zeros(n, dtype=bool)
        in_Uvul[vul_idx] = True
        in_Uvul &= ~in_P
        v_jump = np.zeros(n)
        if in_P.any() and h.sum() > 0:
            if in_Uvul.any():
                v_jump[in_P]    = phi_jump * h[in_P] / h[in_P].sum()
                v_jump[in_Uvul] = (1.0 - phi_jump) / in_Uvul.sum()
            else:
                v_jump[in_P] = h[in_P] / h[in_P].sum()
        else:
            v_jump[:] = 1.0 / n
        # exploration: spread over vulnerable nodes (fixed from baseline)
        v_jump = eps / max(len(vul_idx), 1) * np.isin(np.arange(n), vul_idx).astype(float) \
               + (1.0 - eps) * v_jump
        v_jump /= v_jump.sum()

        # 5. Walk (LFPR or plain PPR)
        gP = in_P.astype(float)
        if phi_walk is None:
            pi = _plain_ppr(A, AT, v_jump, gamma)
        else:
            pi = _lfpr_walk(A, AT, v_jump, gP, gamma, float(phi_walk))

        # 6. Shortlist
        score = pi.copy()
        score[blocked] = -1.0
        C = [int(i) for i in np.argsort(-score)[:min(L, int((~blocked).sum()))]]

        # 7. CRN Monte Carlo evaluation: maximin + tail tie-break
        sp_ref   = sim(T, R_cand, seed_it + 1)
        ref_min  = sp_ref[vul_idx].min()
        r        = max(10, int(tail_frac * len(vul_idx)))
        W        = vul_idx[np.argsort(sp_ref[vul_idx])[:r]]
        ref_tail = sp_ref[W].mean()
        mins, tails = np.empty(len(C)), np.empty(len(C))
        for j, ci in enumerate(C):
            spx        = sim(T + [nodes[ci]], R_cand, seed_it + 1)
            mins[j]    = spx[vul_idx].min() - ref_min
            tails[j]   = spx[W].mean()     - ref_tail
        tol_eval = 1.0 / R_cand
        ok   = np.where(mins >= mins.max() - tol_eval)[0]
        best = max(ok, key=lambda j: (tails[j], mins[j], score[C[j]]))
        ci   = C[int(best)]
        T.append(nodes[ci])
        blocked[ci] = True

    return T
