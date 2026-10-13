import random
import networkx as nx

def get_seeds(G, M, budget, **kwargs):
    candidates = [v for v in G.nodes() if v not in M]
    if not candidates:
        return []

    try:
        closeness = nx.closeness_centrality(G)
    except Exception:
        degrees = dict(G.degree())
        closeness = {v: -degrees.get(v, 0) for v in candidates}

    sorted_cands = sorted(candidates, key=lambda v: closeness.get(v, 0))
    return list(sorted_cands[:budget])
