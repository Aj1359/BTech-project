import random

def get_seeds(G, M, budget, **kwargs):
    vulnerable_set = kwargs.get('vulnerable_set', set(G.nodes()))
    candidates = list(vulnerable_set - set(M))
    if not candidates:
        return []
    return list(random.sample(candidates, min(budget, len(candidates))))
