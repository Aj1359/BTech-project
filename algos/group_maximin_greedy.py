import random
import networkx as nx

def get_seeds(G, M, budget, **kwargs):
    estimate_sp = kwargs.get('estimate_sp')
    pM = kwargs.get('pM', 0.1)
    pT = kwargs.get('pT', 0.1)
    model = kwargs.get('model', 'ICM')
    R = kwargs.get('R', 50)
    vulnerable_set = kwargs.get('vulnerable_set', set(G.nodes()) - set(M))
    groups = kwargs.get('groups', [0, 1])
    
    groups_dict = nx.get_node_attributes(G, 'group')
    if not groups_dict:
        groups_dict = {v: 0 for v in G.nodes()}
        
    group_vuln = {g: [v for v in vulnerable_set if groups_dict.get(v, 0) == g] for g in groups}
    
    # Baseline infection with T = empty
    sp0 = estimate_sp(G, M, set(), model=model, pM=pM, pT=pT, R=R)
    base_inf = {}
    for g in groups:
        b_inf = sum(1.0 - sp0[v] for v in group_vuln[g])
        base_inf[g] = b_inf if b_inf > 0 else 1.0
        
    T = set()
    candidates = list(set(G.nodes()) - set(M))
    
    for _ in range(budget):
        scores = {}
        for c in candidates:
            if c in T: continue
            temp_T = T | {c}
            sp = estimate_sp(G, M, temp_T, model=model, pM=pM, pT=pT, R=R)
            
            S_g = {}
            for g in groups:
                if not group_vuln[g]:
                    S_g[g] = 1.0
                else:
                    curr_inf = sum(1.0 - sp[v] for v in group_vuln[g])
                    S_g[g] = 1.0 - (curr_inf / base_inf[g])
                    
            scores[c] = min(S_g.values())
            
        if not scores:
            break
            
        best_cand = max(scores, key=scores.get)
        T.add(best_cand)
        candidates.remove(best_cand)
        
    return T
