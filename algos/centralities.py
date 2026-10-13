import networkx as nx

def get_seeds(G, M, budget, **kwargs):
    measure = kwargs.get('measure', 'pagerank')
    candidates = [v for v in G.nodes() if v not in M]
    
    if measure == 'pagerank':
        scores = nx.pagerank(G)
    elif measure == 'betweenness':
        scores = nx.betweenness_centrality(G)
    elif measure == 'k_shell' or measure == 'core_number':
        try:
            scores = nx.core_number(G)
        except Exception:
            scores = dict(G.degree())
    else:
        scores = dict(G.degree())
        
    sorted_cand = sorted(candidates, key=lambda v: scores.get(v, 0), reverse=True)
    return list(sorted_cand[:budget])
