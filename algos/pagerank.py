import networkx as nx

def get_seeds(G, M, budget, **kwargs):
    """
    Standard PageRank seed selection baseline.
    Ranks nodes according to classical stationary PageRank scores and returns top-budget candidates excluding M.
    """
    alpha = kwargs.get('alpha', 0.85)  # 1 - damping factor (damping nu = 0.15)
    candidates = set(G.nodes()) - set(M)
    
    try:
        pr = nx.pagerank(G, alpha=alpha)
    except Exception:
        # Fallback to in-degree if power iteration fails
        pr = dict(G.in_degree() if G.is_directed() else G.degree())
        
    sorted_cands = sorted(list(candidates), key=lambda x: pr.get(x, 0.0), reverse=True)
    return set(sorted_cands[:budget])
