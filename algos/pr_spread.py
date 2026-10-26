"""
pr_spread.py -- PageRank x Spreadability (id: pr_spread)

Multiplies reversed target-PageRank score by forward 2-hop spreadability,
discounting infection risk after each seed placement.
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
    gamma = float(kw.get('gamma', 0.15))

    nodes, idx, rho0 = baseline_risk(G, M, **kw)
    n = len(nodes)
    undirected = not G.is_directed()

    A_out = sps.csr_matrix(nx.to_scipy_sparse_array(G, nodelist=nodes, weight=None, format='csr'))
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

        kappa = spreadability(A_out, rho_eff, pT, undirected=undirected)
        score = p * (kappa + 1e-9)
        score[blocked] = -1.0

        ci = int(np.argmax(score))
        chosen_node = nodes[ci]
        T.append(chosen_node)
        blocked[ci] = True

        if chosen_node in H:
            lengths = nx.single_source_shortest_path_length(H, chosen_node, cutoff=2)
            for u, d in lengths.items():
                if u in idx:
                    rho_res[idx[u]] *= (1.0 - (pT ** d))

    return T
