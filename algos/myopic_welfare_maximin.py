import random

def get_seeds(G, M, budget, **kwargs):
    pM = kwargs.get('pM', 0.1)
    pT = kwargs.get('pT', 0.1)
    R = kwargs.get('R', 10)
    vulnerable_set = kwargs.get('vulnerable_set', set(G.nodes()))
    estimate_sp = kwargs.get('estimate_sp')
    
    model = kwargs.get('model', 'COICM')
    T = []
    vulnerable = set(vulnerable_set)
    for _ in range(budget):
        sp = estimate_sp(G, M, set(T), model_type=model, pM=pM, pT=pT, R_runs=R)
        if not vulnerable: break

        min_sp_val = min(sp[v] for v in vulnerable)
        worst_nodes = [v for v in vulnerable if sp[v] == min_sp_val]
        t = random.choice(worst_nodes)

        T.append(t)
        vulnerable.remove(t)
    return T
