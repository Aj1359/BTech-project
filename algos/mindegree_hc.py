import random
import networkx as nx

def get_seeds(G, M, budget, **kwargs):
    candidates = set(G.nodes()) - set(M)
    if not candidates:
        return set()
        
    start_seed = random.choice(list(candidates))
    T = {start_seed}
    
    try:
        harmonic = nx.harmonic_centrality(G)
    except Exception:
        harmonic = {v: 0 for v in G.nodes()}
        
    for _ in range(budget):
        non_seeds = [v for v in G.nodes() if v not in T and v not in M]
        if not non_seeds:
            break
        min_deg = min(G.degree(v) for v in non_seeds)
        pool = [v for v in non_seeds if G.degree(v) == min_deg]
        x = min(pool, key=lambda v: harmonic.get(v, 0))
        T.add(x)
        
    T.discard(start_seed)
    
    # Fill up to budget if initial seed removal dropped count
    remaining = [v for v in G.nodes() if v not in T and v not in M]
    while len(T) < budget and remaining:
        t = random.choice(remaining)
        T.add(t)
        remaining.remove(t)
        
    return T
