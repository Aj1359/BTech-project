"""
FTRW-v2 -- Reverse-Rescue Dual-Walk (id: ftrw_v2)

Key improvements over FTRW v1:
1. DUAL-WALK SHORTLISTING: In addition to the forward LFPR walk (which finds nodes
   near M's influence region), runs a REVERSE RESCUE PPR from the worst-off nodes.
   The reverse walk identifies "savior" nodes — candidates from which truth can
   reach the most vulnerable nodes (walks backward from worst-off on the transpose
   of the original graph).

2. NEIGHBORHOOD RESCUE SCORING: A third signal that scores each candidate by the
   sum of deficits (1-sp) of its worst-off neighbors. Captures immediate rescue
   potential that random walks might smooth away.

3. ADAPTIVE PHI: phi_jump and phi_walk increase when the SP distribution is more
   skewed (many nodes with SP=0), focusing the walk more aggressively on the
   protected region. As seeds improve the distribution, phi relaxes.

4. COVERAGE-AWARE TIE-BREAKING: When candidates have similar maximin gain,
   prefers candidates whose neighborhood has less overlap with previously
   selected seeds, ensuring diverse spatial coverage.

Returns an ORDERED LIST of seeds (selection order), so prefix reuse is valid.

Parameters (all optional, pass via kwargs):
  L=30  shortlist size (wider than v1's 25)
  R_cand=150, R_base=500
  p_frac=0.20, theta=None
  phi_jump=0.80, phi_walk=0.70
  eta=0.7, tau=2.0, eps=0.05, gamma=0.15, tail_frac=0.05
"""
import numpy as np
import networkx as nx
import scipy.sparse as sps
from collections import deque


# ── Helpers ──────────────────────────────────────────────────────────────────

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


def _lfpr_walk(A, AT, v, gP, gamma, phi, iters=300, tol=1e-9):
    """LFPR_N: group-aware walk where a fixed share phi goes to protected nodes."""
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
    return p / max(p.sum(), 1e-15)


def _plain_ppr(A, AT, v, gamma, iters=300, tol=1e-9):
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
    return p / max(p.sum(), 1e-15)


def _rescue_ppr(A_rev, AT_rev, worst_mask, gamma=0.20, iters=300, tol=1e-9):
    """
    Personalized PageRank on the REVERSED original graph, starting from worst-off nodes.

    For undirected graphs A_rev = A_orig (no difference).
    For directed graphs, walking on the reverse finds nodes from which truth
    can propagate forward to the worst-off nodes.
    """
    n = A_rev.shape[0]
    v = worst_mask.astype(float)
    s = v.sum()
    if s > 0:
        v /= s
    else:
        v[:] = 1.0 / n
    deg = np.asarray(A_rev.sum(axis=1)).ravel()
    inv = np.where(deg > 0, 1.0 / np.maximum(deg, 1e-12), 0.0)
    dang = deg == 0
    p = v.copy()
    for _ in range(iters):
        new = gamma * v + (1.0 - gamma) * (AT_rev @ (p * inv) + p[dang].sum() * v)
        done = np.abs(new - p).sum() < tol
        p = new
        if done:
            break
    return p / max(p.sum(), 1e-15)


def _nbr_deficit_score(G, nodes, idx, worst_idx_set, sp, n):
    """
    For each candidate node, sum (1 - sp) of its neighbours that are among the
    worst-off nodes.  Identifies immediate "rescue" nodes.
    """
    score = np.zeros(n)
    for i in range(n):
        total = 0.0
        for nbr in G.neighbors(nodes[i]):
            ni = idx[nbr]
            if ni in worst_idx_set:
                total += max(0.0, 1.0 - sp[ni])
        score[i] = total
    return score


# ── Main entry ───────────────────────────────────────────────────────────────

def get_seeds(G, M, budget, **kw):
    M = set(M)
    est = kw.get('estimate_sp')
    if est is None:
        raise ValueError("ftrw_v2 needs estimate_sp")
    model = kw.get('model', 'COICM')
    pM, pT = kw.get('pM', 0.1), kw.get('pT', 0.1)
    R = int(kw.get('R', 1000))

    L         = int(kw.get('L', 30))
    R_cand    = int(kw.get('R_cand', min(150, R)))
    R_base    = int(kw.get('R_base', min(500, R)))
    p_frac    = float(kw.get('p_frac', 0.20))
    theta     = kw.get('theta', None)
    phi_jump_base = float(kw.get('phi_jump', 0.80))
    phi_walk_base = kw.get('phi_walk', 0.70)
    eta       = float(kw.get('eta', 0.7))
    tau       = float(kw.get('tau', 2.0))
    eps       = float(kw.get('eps', 0.05))
    gamma     = float(kw.get('gamma', 0.15))
    tail_frac = float(kw.get('tail_frac', 0.05))
    seed0     = int(kw.get('rng_seed', 42)) * 1000 + 19   # different stream from v1

    nodes = list(G.nodes())
    idx   = {v: i for i, v in enumerate(nodes)}
    n     = len(nodes)

    try:
        from sim_engine import GraphCSR
        graph_for_sim = kw.get('csr') or GraphCSR(G)
    except Exception:
        graph_for_sim = G

    def sim(T, runs, seed):
        return np.asarray(est(graph_for_sim, set(M), set(T), model_type=model,
                              pM=pM, pT=pT, R_runs=runs, base_seed=seed), dtype=float)

    # Vulnerable set
    vul = kw.get('vulnerable_set')
    if vul is None:
        sp0 = sim(set(), R_base, seed0)
        vul = [v for v in nodes if v not in M and sp0[idx[v]] < 1.0]
    vul_idx = np.array(sorted(idx[v] for v in vul if v not in M), dtype=int)
    if len(vul_idx) == 0:
        return []

    # Precompute: proximity to M
    dm = _dist_from(G, list(M))
    dist_m = np.array([dm.get(v, np.inf) for v in nodes], dtype=float)
    prox   = np.where(np.isfinite(dist_m), np.exp(-dist_m / tau), 0.0)

    # Adjacency matrices for the FORWARD walk (reversed graph for directed nets)
    Gw = G.reverse(copy=False) if G.is_directed() else G
    A_fwd  = sps.csr_matrix(nx.to_scipy_sparse_array(Gw, nodelist=nodes, weight=None, format='csr'))
    AT_fwd = A_fwd.T.tocsr()

    # Adjacency matrices for the RESCUE walk (reverse of original graph)
    # For undirected graphs, reverse = original, so A_rev = A_orig
    if G.is_directed():
        Gr = G.reverse(copy=False)
        A_rev  = sps.csr_matrix(nx.to_scipy_sparse_array(Gr, nodelist=nodes, weight=None, format='csr'))
        AT_rev = A_rev.T.tocsr()
    else:
        A_rev  = sps.csr_matrix(nx.to_scipy_sparse_array(G, nodelist=nodes, weight=None, format='csr'))
        AT_rev = A_rev.T.tocsr()

    blocked = np.zeros(n, dtype=bool)
    for v in M:
        blocked[idx[v]] = True

    seed_nbrs = np.zeros(n, dtype=float)     # coverage from already-selected seeds
    T = []

    for it in range(min(budget, n - int(blocked.sum()))):
        seed_it = seed0 + 7919 * (it + 1)

        # ── 1.  Current save probabilities ──────────────────────────────────
        sp = sim(T, R_base, seed_it)
        sp_vul = sp[vul_idx]

        # ── 2.  Dynamic protected / unprotected split ───────────────────────
        if theta is None:
            cut = float(np.quantile(sp_vul, p_frac))
        else:
            cut = float(theta) - 1.0 / R_base
        in_P = np.zeros(n, dtype=bool)
        in_P[vul_idx[sp_vul <= cut]] = True
        gP = in_P.astype(float)

        # ── ADAPTIVE PHI ────────────────────────────────────────────────────
        zero_frac = float(np.mean(sp_vul == 0.0)) if len(sp_vul) > 0 else 0.0
        phi_jump = min(0.95, phi_jump_base + 0.15 * zero_frac)
        phi_walk = (None if phi_walk_base is None
                    else min(0.90, float(phi_walk_base) + 0.10 * zero_frac))

        # ── 3.  Deficit × proximity weights ─────────────────────────────────
        d = np.where(in_P, cut + 1.0 / R_base - sp, 0.0)
        h = d * (eta + (1.0 - eta) * prox)

        # ── 4a. FSPR jump vector (same as v1) ──────────────────────────────
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
        v_jump = eps / n + (1.0 - eps) * v_jump
        v_jump /= v_jump.sum()

        # ── 5a. FORWARD WALK (LFPR) ────────────────────────────────────────
        if phi_walk is None:
            pi_fwd = _plain_ppr(A_fwd, AT_fwd, v_jump, gamma)
        else:
            pi_fwd = _lfpr_walk(A_fwd, AT_fwd, v_jump, gP, gamma, phi_walk)

        # ── 5b. REVERSE RESCUE WALK (NEW) ──────────────────────────────────
        n_worst  = max(10, int(0.10 * len(vul_idx)))
        ord_vul  = vul_idx[np.argsort(sp[vul_idx])]
        worst_mask = np.zeros(n, dtype=bool)
        worst_mask[ord_vul[:n_worst]] = True
        pi_rev = _rescue_ppr(A_rev, AT_rev, worst_mask, gamma=0.20)

        # ── 5c. NEIGHBOUR DEFICIT SCORE (NEW) ──────────────────────────────
        worst_set = set(ord_vul[:n_worst].tolist())
        nbr_score = _nbr_deficit_score(G, nodes, idx, worst_set, sp, n)
        nbr_max = nbr_score.max()
        if nbr_max > 0:
            nbr_score /= nbr_max

        # ── 6.  MERGE SHORTLISTS from three signals ────────────────────────
        L_each = max(8, L // 3)

        s_fwd = pi_fwd.copy();  s_fwd[blocked] = -1.0
        s_rev = pi_rev.copy();  s_rev[blocked] = -1.0
        s_nbr = nbr_score.copy(); s_nbr[blocked] = -1.0

        C_fwd = set(int(i) for i in np.argsort(-s_fwd)[:L_each])
        C_rev = set(int(i) for i in np.argsort(-s_rev)[:L_each])
        C_nbr = set(int(i) for i in np.argsort(-s_nbr)[:L_each])

        C = sorted(c for c in (C_fwd | C_rev | C_nbr) if not blocked[c])
        if not C:
            break

        # combined score (for tie-breaking only)
        combined = 0.40 * pi_fwd + 0.35 * pi_rev + 0.25 * nbr_score
        combined[blocked] = -1.0

        # ── 7.  MC evaluation with CRN ─────────────────────────────────────
        sp_ref   = sim(T, R_cand, seed_it + 1)
        ref_min  = sp_ref[vul_idx].min()
        r        = max(10, int(tail_frac * len(vul_idx)))
        W        = vul_idx[np.argsort(sp_ref[vul_idx])[:r]]
        ref_tail = sp_ref[W].mean()

        mins  = np.empty(len(C))
        tails = np.empty(len(C))
        for j, ci in enumerate(C):
            spx     = sim(T + [nodes[ci]], R_cand, seed_it + 1)
            mins[j]  = spx[vul_idx].min() - ref_min
            tails[j] = spx[W].mean()       - ref_tail

        tol_mc = 1.0 / R_cand
        ok   = np.where(mins >= mins.max() - tol_mc)[0]

        # Coverage-aware tie-breaking: penalise candidates near existing seeds
        best = max(ok, key=lambda j: (tails[j], combined[C[j]], -seed_nbrs[C[j]]))
        ci   = C[int(best)]
        T.append(nodes[ci])
        blocked[ci] = True

        # Update neighbourhood coverage
        for nbr in G.neighbors(nodes[ci]):
            seed_nbrs[idx[nbr]] += 1.0

    return T
