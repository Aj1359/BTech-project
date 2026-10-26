"""
pr_common.py -- Shared graph and baseline risk estimation utilities for PageRank / Spreadability algorithms.
"""
import numpy as np
import networkx as nx
from collections import deque


def graph_without(G, M):
    """Returns a copy of graph G with nodes in M removed."""
    H = G.copy()
    H.remove_nodes_from(M)
    return H


def baseline_risk(G, M, **kw):
    """
    Computes baseline infection risk rho0(v) = 1 - sp0(v) for all nodes v in G.
    Returns:
        nodes: list of nodes in G
        idx: dict mapping node -> index in [0, len(nodes)-1]
        rho0: np.ndarray of shape (n,) with baseline infection probabilities in [0, 1]
    """
    nodes = list(G.nodes())
    n = len(nodes)
    idx = {node: i for i, node in enumerate(nodes)}
    rho0 = np.zeros(n, dtype=float)
    M_set = set(M)

    pM = float(kw.get('pM', 0.1))
    pT = float(kw.get('pT', 0.1))
    model = kw.get('model', 'COICM')
    R = int(kw.get('R', 1000))
    seed = int(kw.get('rng_seed', 42))

    est = kw.get('estimate_sp')
    if est is not None:
        try:
            from sim_engine import GraphCSR
            graph_for_sim = kw.get('csr') or GraphCSR(G)
        except Exception:
            graph_for_sim = G

        try:
            sp0 = est(graph_for_sim, M_set, set(), model_type=model,
                      pM=pM, pT=pT, R_runs=min(R, 500), base_seed=seed)
            for v in nodes:
                if v in M_set:
                    rho0[idx[v]] = 1.0
                else:
                    rho0[idx[v]] = max(0.0, 1.0 - float(sp0[v]))
            return nodes, idx, rho0
        except Exception as e:
            pass

    # Heuristic fallback if simulator is not provided: BFS distance from M
    dist = {}
    dq = deque([(m, 0) for m in M_set if m in idx])
    for m in M_set:
        if m in idx:
            dist[m] = 0
            rho0[idx[m]] = 1.0

    while dq:
        u, d = dq.popleft()
        for v in (G.successors(u) if G.is_directed() else G.neighbors(u)):
            if v not in dist:
                dist[v] = d + 1
                dq.append((v, d + 1))
                rho0[idx[v]] = float(pM ** (d + 1))

    return nodes, idx, rho0
