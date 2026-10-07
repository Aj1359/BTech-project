import random
import networkx as nx

def get_seeds(G, M, budget, **kwargs):
    candidates = set(G.nodes()) - set(M)
    if not candidates:
        return set()
        
    start_seed = random.choice(list(candidates))
    T = {start_seed}
    
    try:
        closeness = nx.closeness_centrality(G)
    except Exception:
        closeness = {v: 0 for v in G.nodes()}
        
    for _ in range(budget):
        non_seeds = [v for v in G.nodes() if v not in T and v not in M]
        if not non_seeds:
            break
        t = min(non_seeds, key=lambda v: closeness.get(v, 0))
        T.add(t)
        
    T.discard(start_seed)
    
    remaining = [v for v in G.nodes() if v not in T and v not in M]
    while len(T) < budget and remaining:
        t = random.choice(remaining)
        T.add(t)
        remaining.remove(t)
        
    return T
