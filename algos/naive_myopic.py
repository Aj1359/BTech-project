def get_seeds(G, M, budget, **kwargs):
    import numpy as np
    estimate_sp = kwargs.get('estimate_sp')
    pM = kwargs.get('pM', 0.1)
    pT = kwargs.get('pT', 0.1)
    model = kwargs.get('model', 'COICM')
    R = kwargs.get('R', 50)

    sp0 = estimate_sp(G, M, set(), model_type=model, pM=pM, pT=pT, R_runs=R)
    # sp0 is a numpy array indexed by integer node id
    candidates = [v for v in G.nodes() if v not in M]
    sorted_cand = sorted(candidates, key=lambda v: float(sp0[v]))
    return list(sorted_cand[:budget])
