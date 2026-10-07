import random

def get_seeds(G, M, budget, **kwargs):
    candidates = set(G.nodes()) - set(M)
    if not candidates:
        return set()
        
    start_seed = random.choice(list(candidates))
    T = {start_seed}
    
    degrees = dict(G.degree())
    
    for _ in range(budget):
        non_seeds = [v for v in G.nodes() if v not in T and v not in M]
        if not non_seeds:
            break
        min_deg = min(degrees[v] for v in non_seeds)
        pool = [v for v in non_seeds if degrees[v] == min_deg]
        
        # Tie break by neighbor degree sum
        x = max(pool, key=lambda v: sum(degrees[u] for u in G.neighbors(v)))
        T.add(x)
        
    T.discard(start_seed)
    
    remaining = [v for v in G.nodes() if v not in T and v not in M]
    while len(T) < budget and remaining:
        t = random.choice(remaining)
        T.add(t)
        remaining.remove(t)
        
    return T
