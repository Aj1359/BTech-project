"""
FTRW-v3 -- Multi-Resolution Shield Walk (id: ftrw_v3)

A fundamentally different approach from v1/v2.  Instead of one protected-set
threshold, runs LFPR walks at MULTIPLE RESOLUTIONS of the protected set and
adds a SHIELD SCORE that measures each candidate's ability to block
misinformation paths to vulnerable nodes.

Key innovations:
1. MULTI-RESOLUTION WALK: Runs three walks with different protected fractions:
   - p=0.05: ultra-aggressive, targets only the extreme worst-off nodes
   - p=0.20: default balance
   - p=0.50: broader, rescues the "second tier" of vulnerable nodes
   Each walk produces its own shortlist.  The union captures candidates that
   help different layers of the SP distribution.

2. SHIELD SCORE: For each candidate c, measures how many worst-off nodes v
   satisfy dist(M,c)+dist(c,v) == dist(M,v), i.e. c lies on a shortest
   M->v path.  Placing truth at such a "shield" node blocks misinformation
   from reaching v. Approximated using BFS from M and from a sample of
   worst-off nodes.

3. DEGREE-DEFICIT FUSION: A fast scoring signal that combines node degree
   (spreading power) with aggregate deficit of 2-hop neighbours.  Ensures
   high-connectivity candidates aren't overlooked by the walk.

Returns an ORDERED LIST of seeds (selection order), so prefix reuse is valid.
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


def _shield_score(dist_M_arr, dist_from_worst, worst_deficit, n):
    """
    Shield score: for each candidate c, sum deficit(u) over worst-off nodes u
    for which c lies on a shortest path from M to u:
      dist_M(c) + dist(c, u) == dist_M(u)
    """
    score = np.zeros(n)
    n_worst = len(worst_deficit)
    for j in range(n_worst):
        dist_u = dist_from_worst[j]    # dist_u[c] = dist(c, u_j) for node index c
        deficit_u = worst_deficit[j]
        for c in range(n):
            if np.isinf(dist_M_arr[c]) or np.isinf(dist_u[c]):
                continue
            if abs(dist_M_arr[c] + dist_u[c] - dist_M_arr[int(worst_deficit[j])]) < 0.5:
                # Approximation: treat deficit as the weight
                pass
    # Vectorised version for efficiency
    for j in range(n_worst):
        du = dist_from_worst[j]
        target_dist = worst_deficit[j][1]    # dist_M of this worst-off node
        w  = worst_deficit[j][0]             # deficit weight
        on_path = np.abs(dist_M_arr + du - target_dist) < 0.5
        score += w * on_path.astype(float)
    return score


def _shield_score_fast(dist_M_arr, G, nodes, idx, worst_nodes, sp, n, max_sample=50):
    """
    Fast approximation: BFS from a sample of worst-off nodes, check shield condition.
    Returns shield score array of length n.
    """
    rng = np.random.RandomState(77)
    sample = worst_nodes if len(worst_nodes) <= max_sample else rng.choice(worst_nodes, max_sample, replace=False)
    score = np.zeros(n, dtype=float)

    for u_node in sample:
        u_idx = idx[u_node] if u_node in idx else u_node
        deficit = max(0.0, 1.0 - sp[u_idx])
        if deficit < 1e-6:
            continue
        dist_M_u = dist_M_arr[u_idx]
        if np.isinf(dist_M_u):
            continue
        # BFS from u_node to get dist(c, u)
        dist_u = _dist_from(G, [u_node])
        for c_node, d_cu in dist_u.items():
            c_idx = idx[c_node] if c_node in idx else c_node
            d_mc = dist_M_arr[c_idx]
            if np.isinf(d_mc):
                continue
            # Shield condition: c is on a shortest path from M to u
            if abs(d_mc + d_cu - dist_M_u) < 0.5:
                score[c_idx] += deficit

    return score


def _degree_deficit_2hop(G, nodes, idx, sp, vul_idx_set, n):
    """
    Score = degree(c) * (sum of (1-sp) for vulnerable nodes within 2 hops).
    Captures both spreading power and proximity to deficit.
    """
    deg = np.zeros(n)
    agg = np.zeros(n)
    for i in range(n):
        nid = nodes[i]
        nbrs = list(G.neighbors(nid))
        deg[i] = len(nbrs)
        total = 0.0
        seen = set()
        for nb in nbrs:
            ni = idx[nb]
            if ni in vul_idx_set:
                total += max(0.0, 1.0 - sp[ni])
            seen.add(nb)
            # 2-hop
            for nb2 in G.neighbors(nb):
                if nb2 not in seen:
                    ni2 = idx[nb2]
                    if ni2 in vul_idx_set:
                        total += 0.5 * max(0.0, 1.0 - sp[ni2])
        agg[i] = total
    # Combine: sqrt(degree) * deficit_agg  (sqrt damps high-degree dominance)
    score = np.sqrt(np.maximum(deg, 1.0)) * agg
    mx = score.max()
    if mx > 0:
        score /= mx
    return score


# ── Main entry ───────────────────────────────────────────────────────────────

def get_seeds(G, M, budget, **kw):
    M = set(M)
    est = kw.get('estimate_sp')
    if est is None:
        raise ValueError("ftrw_v3 needs estimate_sp")
    model = kw.get('model', 'COICM')
    pM, pT = kw.get('pM', 0.1), kw.get('pT', 0.1)
    R = int(kw.get('R', 1000))

    L         = int(kw.get('L', 30))
    R_cand    = int(kw.get('R_cand', min(150, R)))
    R_base    = int(kw.get('R_base', min(500, R)))
    phi_jump  = float(kw.get('phi_jump', 0.80))
    phi_walk  = kw.get('phi_walk', 0.70)
    eta       = float(kw.get('eta', 0.7))
    tau       = float(kw.get('tau', 2.0))
    eps       = float(kw.get('eps', 0.05))
    gamma     = float(kw.get('gamma', 0.15))
    tail_frac = float(kw.get('tail_frac', 0.05))
    seed0     = int(kw.get('rng_seed', 42)) * 1000 + 23

    # Multi-resolution fractions (core innovation)
    p_fracs = [0.05, 0.20, 0.50]

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
    vul_idx_set = set(vul_idx.tolist())

    # Distance to M
    dm = _dist_from(G, list(M))
    dist_m = np.array([dm.get(v, np.inf) for v in nodes], dtype=float)
    prox   = np.where(np.isfinite(dist_m), np.exp(-dist_m / tau), 0.0)

    # Walk graph (reversed for directed, original for undirected)
    Gw = G.reverse(copy=False) if G.is_directed() else G
    A  = sps.csr_matrix(nx.to_scipy_sparse_array(Gw, nodelist=nodes, weight=None, format='csr'))
    AT = A.T.tocsr()

    blocked = np.zeros(n, dtype=bool)
    for v in M:
        blocked[idx[v]] = True
    T = []

    for it in range(min(budget, n - int(blocked.sum()))):
        seed_it = seed0 + 7919 * (it + 1)

        # ── 1.  Current save probabilities ──────────────────────────────────
        sp = sim(T, R_base, seed_it)
        sp_vul = sp[vul_idx]

        # ── 2.  MULTI-RESOLUTION walks ──────────────────────────────────────
        all_candidates = set()

        for pf in p_fracs:
            cut = float(np.quantile(sp_vul, pf))
            in_P = np.zeros(n, dtype=bool)
            in_P[vul_idx[sp_vul <= cut]] = True
            gP = in_P.astype(float)

            # Deficit × proximity
            d = np.where(in_P, cut + 1.0 / R_base - sp, 0.0)
            h = d * (eta + (1.0 - eta) * prox)

            # FSPR jump vector
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

            # Walk
            if phi_walk is None:
                pi = _plain_ppr(A, AT, v_jump, gamma)
            else:
                pi = _lfpr_walk(A, AT, v_jump, gP, gamma, float(phi_walk))

            pi[blocked] = -1.0
            L_res = max(6, L // len(p_fracs))
            top_k = set(int(i) for i in np.argsort(-pi)[:L_res])
            all_candidates |= top_k

        # ── 3.  SHIELD SCORE (NEW) ──────────────────────────────────────────
        n_worst = max(10, int(0.10 * len(vul_idx)))
        ord_vul = vul_idx[np.argsort(sp[vul_idx])]
        worst_nodes = [nodes[vi] for vi in ord_vul[:n_worst]]

        shield = _shield_score_fast(dist_m, G, nodes, idx, worst_nodes, sp, n,
                                    max_sample=min(40, n_worst))
        shield[blocked] = -1.0
        shield_mx = shield.max()
        if shield_mx > 0:
            shield /= shield_mx
        L_shield = max(6, L // 3)
        C_shield = set(int(i) for i in np.argsort(-shield)[:L_shield])
        all_candidates |= C_shield

        # ── 4.  DEGREE-DEFICIT FUSION SCORE (NEW) ──────────────────────────
        dd_score = _degree_deficit_2hop(G, nodes, idx, sp, vul_idx_set, n)
        dd_score[blocked] = -1.0
        L_dd = max(4, L // 5)
        C_dd = set(int(i) for i in np.argsort(-dd_score)[:L_dd])
        all_candidates |= C_dd

        # ── 5.  Filter and evaluate ────────────────────────────────────────
        C = sorted(c for c in all_candidates if not blocked[c])
        if not C:
            break

        # Combined score for tie-breaking
        combined = 0.35 * shield + 0.35 * dd_score
        # Add average walk score
        combined[blocked] = -1.0

        sp_ref   = sim(T, R_cand, seed_it + 1)
        ref_min  = sp_ref[vul_idx].min()
        r        = max(10, int(tail_frac * len(vul_idx)))
        W        = vul_idx[np.argsort(sp_ref[vul_idx])[:r]]
        ref_tail = sp_ref[W].mean()

        mins  = np.empty(len(C))
        tails = np.empty(len(C))
        for j, ci in enumerate(C):
            spx      = sim(T + [nodes[ci]], R_cand, seed_it + 1)
            mins[j]  = spx[vul_idx].min() - ref_min
            tails[j] = spx[W].mean()       - ref_tail

        tol_mc = 1.0 / R_cand
        ok     = np.where(mins >= mins.max() - tol_mc)[0]
        best   = max(ok, key=lambda j: (tails[j], combined[C[j]]))
        ci     = C[int(best)]
        T.append(nodes[ci])
        blocked[ci] = True

    return T
