"""
sj_fspr.py -- Spreadability-Jump Fair Stochastic PageRank (id: sj_fspr)

FSPR where the jump vector is weighted by 2-hop spreadability instead of in-degree.
Dynamic protected group P is the top p_frac nodes by residual risk.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import networkx as nx
import scipy.sparse as sps

from pr_common import baseline_risk, graph_without
from sj_common import spreadability


def get_seeds(G, M, budget, **kw):
    M = set(M)
    pT = min(float(kw.get('pT', 0.1)), 1.0)
    p_frac = float(kw.get('p_frac', 0.20))
    phi_jump = float(kw.get('phi_jump', 0.80))
    gamma = float(kw.get('gamma', kw.get('nu', 0.30)))

    nodes, idx, rho0 = baseline_risk(G, M, **kw)
    n = len(nodes)
    undirected = not G.is_directed()

    A_out = sps.csr_matrix(nx.to_scipy_sparse_array(G, nodelist=nodes, weight=None, format='csr'))
    Gw = G.reverse(copy=False) if G.is_directed() else G
    A_walk = sps.csr_matrix(nx.to_scipy_sparse_array(Gw, nodelist=nodes, weight=None, format='csr'))
    H = graph_without(G, M)

    blocked = np.zeros(n, dtype=bool)
    for v in M:
        if v in idx:
            blocked[idx[v]] = True

    # Standard column-stochastic transition matrix for FSPR walk
    deg = np.asarray(A_walk.sum(axis=1)).ravel()
    inv_deg = np.where(deg > 0, 1.0 / np.maximum(deg, 1e-12), 0.0)
    # Transition probability P[j, i] = 1 / deg(j)
    # In power iteration: p_next = gamma * v + (1 - gamma) * (A_walk.T @ (p * inv_deg))
    AT = A_walk.T.tocsr()
    dangling = deg == 0

    rho_res = rho0.copy()
    T = []
    max_k = min(budget, int((~blocked).sum()))

    for it in range(max_k):
        rho_eff = rho_res.copy()
        rho_eff[blocked] = 0.0

        kappa = spreadability(A_out, rho_eff, pT, undirected=undirected)
        kappa[blocked] = 0.0
        kappa = kappa + 1e-12 * (~blocked)

        risky = (rho_eff > 0) & ~blocked
        if risky.any():
            thr = float(np.quantile(rho_eff[risky], 1.0 - p_frac))
            inP = risky & (rho_eff >= thr)
        else:
            inP = np.zeros(n, dtype=bool)
        inU = ~inP & ~blocked

        v = np.zeros(n)
        sP = float((kappa * inP).sum())
        sU = float((kappa * inU).sum())
        if sP > 0 and sU > 0:
            v[inP] = phi_jump * kappa[inP] / sP
            v[inU] = (1.0 - phi_jump) * kappa[inU] / sU
        elif sP + sU > 0:
            v = kappa / kappa.sum()
        else:
            v[~blocked] = 1.0 / max(1, int((~blocked).sum()))

        if v.sum() > 0:
            v /= v.sum()

        # FSPR power iteration
        p = v.copy()
        for _ in range(200):
            p_next = gamma * v + (1.0 - gamma) * (AT @ (p * inv_deg) + v * float((p * dangling).sum()))
            if np.abs(p_next - p).sum() < 1e-8:
                break
            p = p_next

        p[blocked] = -1.0
        ci = int(np.argmax(p))
        chosen_node = nodes[ci]
        T.append(chosen_node)
        blocked[ci] = True

        if chosen_node in H:
            lengths = nx.single_source_shortest_path_length(H, chosen_node, cutoff=2)
            for u, d in lengths.items():
                if u in idx:
                    rho_res[idx[u]] *= (1.0 - (pT ** d))

    return T
