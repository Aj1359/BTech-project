import random

def get_seeds(G, M, budget, **kwargs):
    D = kwargs.get('D', {})
    D_M = kwargs.get('D_M', {})
    vulnerable_set = kwargs.get('vulnerable_set', set())
    
    candidates = set(G.nodes()) - set(M)
    T = []

    if not candidates:
        return T

    # Start with highest degree in candidates
    degrees = dict(G.degree())
    first_seed = max(candidates, key=lambda x: degrees.get(x, 0))
    T.append(first_seed)
    candidates.discard(first_seed)

    min_dist_to_T = {v: D.get(first_seed, {}).get(v, float('inf')) for v in vulnerable_set}

    while len(T) < budget and candidates:
        furthest_cand = None
        max_dist = -1

        for c in candidates:
            d_to_T = min((D.get(t, {}).get(c, float('inf')) for t in T), default=float('inf'))
            if d_to_T > max_dist and d_to_T != float('inf'):
                max_dist = d_to_T
                furthest_cand = c

        if furthest_cand is None:
            furthest_cand = random.choice(list(candidates))

        T.append(furthest_cand)
        candidates.discard(furthest_cand)

    return T
