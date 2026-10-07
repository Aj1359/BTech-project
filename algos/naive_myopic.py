def get_seeds(G, M, budget, **kwargs):
    estimate_sp = kwargs.get('estimate_sp')
    pM = kwargs.get('pM', 0.1)
    pT = kwargs.get('pT', 0.1)
    model = kwargs.get('model', 'ICM')
    R = kwargs.get('R', 50)
    
    sp0 = estimate_sp(G, M, set(), model=model, pM=pM, pT=pT, R=R)
    candidates = [v for v in G.nodes() if v not in M]
    sorted_cand = sorted(candidates, key=lambda v: sp0.get(v, 1.0))
    return set(sorted_cand[:budget])
