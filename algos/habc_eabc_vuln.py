import networkx as nx

def get_seeds(G, M, budget, **kwargs):
    estimate_sp = kwargs.get('estimate_sp')
    pM = kwargs.get('pM', 0.1)
    pT = kwargs.get('pT', 0.1)
    model = kwargs.get('model', 'ICM')
    
    sp0 = estimate_sp(G, M, set(), model_type=model, pM=pM, pT=pT, R_runs=50)
    w = {v: 1.0 - float(sp0[v]) for v in G.nodes()}
    for m in M:
        w[m] = 0.0
        
    total_w = sum(w.values())
    if total_w == 0:
        total_w = 1.0
        
    candidates = [v for v in G.nodes() if v not in M]
    scores = {}
    
    for u in candidates:
        lengths = nx.single_source_shortest_path_length(G, u)
        sum_dist = 0.0
        for v, dist in lengths.items():
            if v != u and v not in M:
                sum_dist += w[v] / float(dist + 1)
        scores[u] = (w[u] + sum_dist) / total_w
        
    sorted_cand = sorted(candidates, key=lambda v: scores.get(v, 0), reverse=True)
    return list(sorted_cand[:budget])
