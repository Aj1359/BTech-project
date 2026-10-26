"""
pr_target.py -- Target-Risk PageRank (id: pr_target)

PageRank on the reversed graph with personalization restarting exclusively
at vulnerable / at-risk nodes proportional to their residual risk.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import networkx as nx
import scipy.sparse as sps

from pr_common import baseline_risk, graph_without


def get_seeds(G, M, budget, **kw):
    M = set(M)
    pT = min(float(kw.get('pT', 0.1)), 1.0)
    gamma = float(kw.get('gamma', 0.15))

    nodes, idx, rho0 = baseline_risk(G, M, **kw)
    n = len(nodes)

    Gw = G.reverse(copy=False) if G.is_directed() else G
    A_walk = sps.csr_matrix(nx.to_scipy_sparse_array(Gw, nodelist=nodes, weight=None, format='csr'))
    AT = A_walk.T.tocsr()
    deg = np.asarray(A_walk.sum(axis=1)).ravel()
    inv_deg = np.where(deg > 0, 1.0 / np.maximum(deg, 1e-12), 0.0)
    dangling = deg == 0
    H = graph_without(G, M)

    blocked = np.zeros(n, dtype=bool)
    for v in M:
        if v in idx:
            blocked[idx[v]] = True

    rho_res = rho0.copy()
    T = []
    max_k = min(budget, int((~blocked).sum()))

    for it in range(max_k):
        rho_eff = rho_res.copy()
        rho_eff[blocked] = 0.0

        v = rho_eff.copy()
        s_v = v.sum()
        if s_v > 0:
            v /= s_v
        else:
            v[~blocked] = 1.0 / max(1, int((~blocked).sum()))

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
