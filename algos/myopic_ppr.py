import random
import networkx as nx

def get_seeds(G, M, budget, **kwargs):
    candidates = set(G.nodes()) - set(M)
    if not candidates:
        return set()
        
    start_seed = random.choice(list(candidates))
    T = {start_seed}
    
    for _ in range(budget):
        non_seeds = [v for v in G.nodes() if v not in T and v not in M]
        if not non_seeds:
            break
        
        # Personalized PageRank with restart set T
        reset_dict = {v: (1.0 / len(T) if v in T else 0.0) for v in G.nodes()}
        try:
            ppr = nx.pagerank(G, personalization=reset_dict)
        except Exception:
            ppr = {v: 0 for v in G.nodes()}
            
        t = min(non_seeds, key=lambda v: ppr.get(v, 0))
        T.add(t)
        
    T.discard(start_seed)
    
    remaining = [v for v in G.nodes() if v not in T and v not in M]
    while len(T) < budget and remaining:
        t = random.choice(remaining)
        T.add(t)
        remaining.remove(t)
        
    return T
