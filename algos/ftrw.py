"""
FTRW -- Fairness-Targeted Random Walk for truth-seed selection (id: ftrw)

Follows the project proposal (BTP-ProposedRandomWalkApproach.pdf) and adds the
FSPR / LFPR ideas with a protected vs. unprotected split that is RE-DRAWN AT EVERY STEP.

One round (repeated until `budget` seeds are chosen):
  1. Monte Carlo estimate of the save probability sp(v) under the current seeds T.
  2. Dynamic groups.  Protected P_i = the worst-protected vulnerable nodes right now
     (sp at or below the p_frac quantile, or sp < theta if theta is given).
     Unprotected U_i = every other node.  As seeds are added and sp rises, the nodes
     in P change, so the groups change from step to step.
  3. Deficit weight  h(v) = d(v) * [eta + (1 - eta) * exp(-dist(M, v) / tau)]  on P,
     d(v) = (cut + 1/R - sp(v)).
  4. FSPR-style jump vector: a fixed share `phi_jump` of the restart mass goes to P
     (spread in proportion to h), the rest to the vulnerable unprotected nodes,
     plus a small uniform part `eps` for exploration.
  5. LFPR_N-style walk: every node sends a fixed share `phi_walk` of its walk mass to
     protected nodes and the rest to unprotected nodes (group-normalised transitions,
     dangling mass spread uniformly inside the group), so the walk keeps entering the
     under-protected region.  Walk is on the reversed graph for directed networks.
  6. Shortlist the top-L walk scores outside M and T.
  7. Evaluate each shortlisted candidate with direct Monte Carlo (same random numbers for
     every candidate).  Choose the candidate with the largest maximin gain; candidates
     within one Monte Carlo step of the best maximin gain are ranked by the gain in the
     mean save probability of the r worst-off nodes (Eq. 15 of the proposal).  This
     tail tie-break matters: when many nodes are already lost, the plain maximin gain is
     0 for every candidate.

Returns an ORDERED LIST of seeds (selection order), so prefix reuse is valid.

Parameters (all optional, pass via kwargs):
  L=25 shortlist size            R_cand=150 simulations per candidate
  R_base=500 simulations for the per-round sp estimate
  p_frac=0.20 protected share    theta=None (fixed threshold instead of the quantile)
  phi_jump=0.80 FSPR mass        phi_walk=0.70 LFPR local mass (None = plain PageRank walk)
  eta=0.7, tau=2.0, eps=0.05, gamma=0.15 (restart), tail_frac=0.05
"""
import numpy as np
import networkx as nx
import scipy.sparse as sps
from collections import deque


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
    """LFPR_N: node-level update of Eq. (1) of the LFPR paper, vectorised."""
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


def get_seeds(G, M, budget, **kw):
    M = set(M)
    est = kw.get('estimate_sp')
    if est is None:
        raise ValueError("ftrw needs estimate_sp")
    model = kw.get('model', 'COICM')
    pM, pT = kw.get('pM', 0.1), kw.get('pT', 0.1)
    R = int(kw.get('R', 1000))
    L = int(kw.get('L', 25))
    R_cand = int(kw.get('R_cand', min(150, R)))
    R_base = int(kw.get('R_base', min(500, R)))
    p_frac = float(kw.get('p_frac', 0.20))
    theta = kw.get('theta', None)
    phi_jump = float(kw.get('phi_jump', 0.80))
    phi_walk = kw.get('phi_walk', 0.70)
    eta = float(kw.get('eta', 0.7))
    tau = float(kw.get('tau', 2.0))
    eps = float(kw.get('eps', 0.05))
    gamma = float(kw.get('gamma', 0.15))
    tail_frac = float(kw.get('tail_frac', 0.05))
    seed0 = int(kw.get('rng_seed', 42)) * 1000 + 17

    nodes = list(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    try:
        from sim_engine import GraphCSR
        graph_for_sim = kw.get('csr') or GraphCSR(G)       # build the fast graph once
    except Exception:
        graph_for_sim = G

    def sim(T, runs, seed):
        # KEYWORD call on purpose (a positional call binds pT to R in this engine)
        return np.asarray(est(graph_for_sim, set(M), set(T), model_type=model,
                              pM=pM, pT=pT, R_runs=runs, base_seed=seed), dtype=float)

    # vulnerable set (fixed): given by the runner, else from a baseline run
    vul = kw.get('vulnerable_set')
    if vul is None:
        sp0 = sim(set(), R_base, seed0)
        vul = [v for v in nodes if v not in M and sp0[idx[v]] < 1.0]
    vul_idx = np.array(sorted(idx[v] for v in vul if v not in M), dtype=int)
    if len(vul_idx) == 0:
        return []

    # proximity to misinformation (original graph, respects direction)
    dm = _dist_from(G, list(M))
    dist = np.array([dm.get(v, np.inf) for v in nodes], dtype=float)
    prox = np.where(np.isfinite(dist), np.exp(-dist / tau), 0.0)

    # walk graph: reversed for directed, original for undirected
    Gw = G.reverse(copy=False) if G.is_directed() else G
    A = sps.csr_matrix(nx.to_scipy_sparse_array(Gw, nodelist=nodes, weight=None, format='csr'))
    AT = A.T.tocsr()

    blocked = np.zeros(n, dtype=bool)
    for v in M:
        blocked[idx[v]] = True
    T = []

    for it in range(min(budget, n - int(blocked.sum()))):
        seed_it = seed0 + 7919 * (it + 1)

        # 1-2. current save probabilities and the dynamic protected / unprotected split
        sp = sim(T, R_base, seed_it)
        sp_vul = sp[vul_idx]
        if theta is None:
            cut = float(np.quantile(sp_vul, p_frac))
        else:
            cut = float(theta) - 1.0 / R_base
        in_P = np.zeros(n, dtype=bool)
        in_P[vul_idx[sp_vul <= cut]] = True
        gP = in_P.astype(float)

        # 3. deficit x proximity weights
        d = np.where(in_P, cut + 1.0 / R_base - sp, 0.0)
        h = d * (eta + (1.0 - eta) * prox)

        # 4. FSPR-style jump vector
        in_Uvul = np.zeros(n, dtype=bool)
        in_Uvul[vul_idx] = True
        in_Uvul &= ~in_P
        v = np.zeros(n)
        if in_P.any() and h.sum() > 0:
            if in_Uvul.any():
                v[in_P] = phi_jump * h[in_P] / h[in_P].sum()
                v[in_Uvul] = (1.0 - phi_jump) / in_Uvul.sum()
            else:
                v[in_P] = h[in_P] / h[in_P].sum()
        else:
            v[:] = 1.0 / n
        v = eps / n + (1.0 - eps) * v
        v /= v.sum()

        # 5. LFPR-style walk (or plain personalised PageRank for the ablation)
        if phi_walk is None:
            pi = _plain_ppr(A, AT, v, gamma)
        else:
            pi = _lfpr_walk(A, AT, v, gP, gamma, float(phi_walk))

        # 6. shortlist
        score = pi.copy()
        score[blocked] = -1.0
        C = [int(i) for i in np.argsort(-score)[:min(L, int((~blocked).sum()))]]

        # 7. direct Monte Carlo evaluation, common random numbers for all candidates
        sp_ref = sim(T, R_cand, seed_it + 1)
        ref_min = sp_ref[vul_idx].min()
        r = max(10, int(tail_frac * len(vul_idx)))
        W = vul_idx[np.argsort(sp_ref[vul_idx])[:r]]
        ref_tail = sp_ref[W].mean()
        mins, tails = np.empty(len(C)), np.empty(len(C))
        for j, ci in enumerate(C):
            spx = sim(T + [nodes[ci]], R_cand, seed_it + 1)
            mins[j] = spx[vul_idx].min() - ref_min
            tails[j] = spx[W].mean() - ref_tail
        tol = 1.0 / R_cand
        ok = np.where(mins >= mins.max() - tol)[0]
        best = max(ok, key=lambda j: (tails[j], mins[j], score[C[j]]))
        ci = C[int(best)]
        T.append(nodes[ci])
        blocked[ci] = True
    return T
