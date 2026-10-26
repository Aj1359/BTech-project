"""
NSJ-LFPR -- Neighbour-Spread Jump LFPR (id: nsj_lfpr)

A fair-PageRank seed selector that changes the jump vector, the walk,
and the jump factor of LFPR/FSPR using 2-hop spreadability and risk dynamics.

Per step t:
  1. Effective residual risk rho_eff(v): baseline infection probability discounted by seeds
     already placed; nodes that are almost surely lost (>= tau_lost) are set to 0.
  2. Spreadability: kappa(u) = sum_{v within 2 hops} pT^dist(u,v) * rho_eff(v)
     Neighbour spreadability: kappa_N(u) = mean of kappa over the out-neighbours of u
     Combined score: sigma(u) = kappa(u) + lam * kappa_N(u)
  3. Dynamic groups: protected P_t = top p_frac highest-residual-risk nodes, U_t = rest.
  4. FSPR-style jump vector: phi_jump of restart mass on P (proportional to sigma),
     the rest on U (proportional to sigma).
  5. LFPR_N-style fair walk with NEIGHBOUR-BIASED edge weights a_ji = sigma(i)^s:
     every node sends phi_walk of its walk mass to protected neighbours and the rest to
     unprotected ones, in proportion to the neighbour's sigma.
  6. Adaptive jump factor: nu_t = nu_min + (nu_max - nu_min) * C_t,
     where C_t = share of residual risk held by P_t (concentrated risk -> stay near hotspots).
  7. Solve pi = nu_t * v + (1 - nu_t) * P^T pi (reversed graph if directed).
  8. Select:
       mode='fast'     -> argmax pi
       mode='verified' -> shortlist top-L by pi, evaluate each candidate by direct
                          Monte Carlo (common random numbers), pick largest maximin gain,
                          ties (within 1/R_cand) broken by gain in mean save probability
                          of the worst-off r nodes.
  9. Discount residual risk of nodes covered by new seed: rho_res(v) *= 1 - pT^dist.

Returns an ORDERED LIST of seeds (strictly prefix-reusable).
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import networkx as nx
import scipy.sparse as sps

from pr_common import baseline_risk, graph_without
from sj_common import spreadability, _lfpr_walk


def get_seeds(G, M, budget, **kw):
    M = set(M)
    pT = min(float(kw.get('pT', 0.1)), 1.0)
    pM = float(kw.get('pM', 0.1))
    model = kw.get('model', 'COICM')
    nu_min = float(kw.get('nu_min', 0.15))
    nu_max = float(kw.get('nu_max', 0.50))
    phi_jump = float(kw.get('phi_jump', 0.80))
    phi_walk = float(kw.get('phi_walk', 0.70))
    p_frac = float(kw.get('p_frac', 0.20))
    lam = float(kw.get('lam', 0.5))
    s_pow = float(kw.get('s', 1.0))
    tau_lost = float(kw.get('tau_lost', 0.95))
    L = int(kw.get('L', 25))
    R = int(kw.get('R', 1000))
    R_cand = int(kw.get('R_cand', min(150, R)))
    tail_frac = float(kw.get('tail_frac', 0.05))
    seed0 = int(kw.get('rng_seed', 42)) * 1000 + 31

    nodes, idx, rho0 = baseline_risk(G, M, **kw)
    n = len(nodes)
    undirected = not G.is_directed()

    # Outgoing adjacency matrix
    A_out = sps.csr_matrix(nx.to_scipy_sparse_array(G, nodelist=nodes, weight=None, format='csr'))
    deg_out = np.asarray(A_out.sum(axis=1)).ravel()
    inv_deg_out = np.where(deg_out > 0, 1.0 / np.maximum(deg_out, 1e-12), 0.0)

    # For walk: if directed, walk flows along incoming edges (reversed graph)
    Gw = G.reverse(copy=False) if G.is_directed() else G
    A_walk = sps.csr_matrix(nx.to_scipy_sparse_array(Gw, nodelist=nodes, weight=None, format='csr'))
    H = graph_without(G, M)

    blocked = np.zeros(n, dtype=bool)
    for v in M:
        if v in idx:
            blocked[idx[v]] = True

    # Check simulator for verified mode
    est = kw.get('estimate_sp')
    mode = kw.get('mode', 'verified' if est is not None else 'fast')

    sim = None
    if mode == 'verified' and est is not None:
        try:
            from sim_engine import GraphCSR
            graph_for_sim = kw.get('csr') or GraphCSR(G)
        except Exception:
            graph_for_sim = G

        def sim(T_curr, runs, seed):
            return np.asarray(est(graph_for_sim, set(M), set(T_curr), model_type=model,
                                  pM=pM, pT=pT, R_runs=runs, base_seed=seed), dtype=float)

    vul_idx = np.where((rho0 > 0) & ~blocked)[0]
    if len(vul_idx) == 0:
        vul_idx = np.where(~blocked)[0]

    rho_res = rho0.copy()
    T = []
    max_k = min(budget, int((~blocked).sum()))

    for it in range(max_k):
        # 1. Effective residual risk (lost nodes filtered out)
        rho_eff = np.where(rho_res < tau_lost, rho_res, 0.0)
        rho_eff[blocked] = 0.0

        # 2. Spreadability, neighbour spreadability, and combined score
        kappa = spreadability(A_out, rho_eff, pT, undirected=undirected)
        kappa[blocked] = 0.0
        kappa_N = (A_out @ kappa) * inv_deg_out
        sigma = kappa + lam * kappa_N
        sigma[blocked] = 0.0
        sigma = sigma + 1e-12 * (~blocked)

        # 3. Dynamic protected / unprotected split by residual risk
        risky = (rho_eff > 0) & ~blocked
        if risky.any():
            thr = float(np.quantile(rho_eff[risky], 1.0 - p_frac))
            inP = risky & (rho_eff >= thr)
        else:
            inP = np.zeros(n, dtype=bool)
        inU = ~inP & ~blocked

        # 4. FSPR-style jump vector driven by sigma
        v = np.zeros(n)
        sP = float((sigma * inP).sum())
        sU = float((sigma * inU).sum())
        if sP > 0 and sU > 0:
            v[inP] = phi_jump * sigma[inP] / sP
            v[inU] = (1.0 - phi_jump) * sigma[inU] / sU
        elif sP + sU > 0:
            v = sigma / sigma.sum()
        else:
            v[~blocked] = 1.0 / max(1, int((~blocked).sum()))

        v_sum = v.sum()
        if v_sum > 0:
            v /= v_sum

        # 5. Neighbour-biased LFPR_N walk: edge weight a_ji = sigma(i)^s
        # Scale sigma to avoid numerical overflow when raising to power s
        max_sig = sigma.max()
        scaled_sig = sigma / max_sig if max_sig > 0 else sigma
        W = sps.csr_matrix(A_walk @ sps.diags(np.power(scaled_sig, s_pow)))
        WT = W.T.tocsr()

        # 6. Adaptive jump factor from the concentration of residual risk
        tot_eff = float(rho_eff.sum())
        conc = float(rho_eff[inP].sum() / tot_eff) if tot_eff > 0 else 0.0
        nu_t = nu_min + (nu_max - nu_min) * conc

        # 7. Solve walk
        pi = _lfpr_walk(W, WT, v, inP.astype(float), nu_t, phi_walk)
        pi[blocked] = -1.0

        # 8. Selection
        avail_nodes = int((~blocked).sum())
        if avail_nodes == 0:
            break

        if sim is None or mode == 'fast':
            ci = int(np.argmax(pi))
        else:
            seed_it = seed0 + 7919 * (it + 1)
            cand_count = min(L, avail_nodes)
            C = [int(i) for i in np.argsort(-pi)[:cand_count] if not blocked[i]]
            if not C:
                ci = int(np.argmax(pi))
            else:
                sp_ref = sim(T, R_cand, seed_it)
                ref_min = float(sp_ref[vul_idx].min())
                r = max(10, int(tail_frac * len(vul_idx)))
                r = min(r, len(vul_idx))
                Wn = vul_idx[np.argsort(sp_ref[vul_idx])[:r]]
                ref_tail = float(sp_ref[Wn].mean())

                mins = np.empty(len(C))
                tails = np.empty(len(C))
                for j, c in enumerate(C):
                    spx = sim(T + [nodes[c]], R_cand, seed_it)
                    mins[j] = float(spx[vul_idx].min()) - ref_min
                    tails[j] = float(spx[Wn].mean()) - ref_tail

                max_gain = mins.max()
                ok = np.where(mins >= max_gain - 1.0 / R_cand)[0]
                best = max(ok, key=lambda j: (tails[j], mins[j], pi[C[j]]))
                ci = C[int(best)]

        chosen_node = nodes[ci]
        T.append(chosen_node)
        blocked[ci] = True

        # 9. Discount residual risk covered by the chosen truth seed
        if chosen_node in H:
            lengths = nx.single_source_shortest_path_length(H, chosen_node, cutoff=2)
            for u, d in lengths.items():
                if u in idx:
                    rho_res[idx[u]] *= (1.0 - (pT ** d))

    return T
