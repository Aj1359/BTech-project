import random
import networkx as nx

def get_seeds(G, M, budget, **kwargs):
    """
    DP-aware Fair Influence Blocking / Campaigning (arXiv 2601.22584):
    Greedy scalarization of Effectiveness(T) + lambda * DP_Fairness(T).
    """
    estimate_sp = kwargs.get('estimate_sp')
    pM = kwargs.get('pM', 0.1)
    pT = kwargs.get('pT', 0.1)
    model = kwargs.get('model', 'ICM')
    lam = kwargs.get('lam', 1.0)
    R = kwargs.get('R', 30)
    groups = kwargs.get('groups', [0, 1])
    vulnerable_set = kwargs.get('vulnerable_set', set(G.nodes()) - set(M))
    
    groups_dict = nx.get_node_attributes(G, 'group')
    if not groups_dict:
        groups_dict = {v: 0 for v in G.nodes()}
        
    vulnerable = list(vulnerable_set)
    total_vuln = float(len(vulnerable)) if vulnerable else 1.0
    
    sp0 = estimate_sp(G, M, set(), model=model, pM=pM, pT=pT, R=R)
    base_inf_g = {}
    for g in groups:
        g_vuln = [v for v in vulnerable if groups_dict.get(v, 0) == g]
        b_inf = sum(1.0 - sp0[v] for v in g_vuln)
        base_inf_g[g] = b_inf if b_inf > 0 else 1.0
        
    T = set()
    candidates = set(G.nodes()) - set(M)
    
    for _ in range(budget):
        if not candidates:
            break
            
        best_cand = None
        best_obj = -float('inf')
        
        for c in candidates:
            temp_T = T | {c}
            sp = estimate_sp(G, M, temp_T, model=model, pM=pM, pT=pT, R=R)
            
            # Effectiveness: total expected saved nodes
            exp_inf = sum(1.0 - sp[v] for v in vulnerable)
            effectiveness = sum(1.0 - sp0[v] for v in vulnerable) - exp_inf
            
            # DP Fairness: 1.0 - group gap in relative saved fraction
            S_g = {}
            for g in groups:
                g_vuln = [v for v in vulnerable if groups_dict.get(v, 0) == g]
                if not g_vuln:
                    S_g[g] = 1.0
                else:
                    curr_inf = sum(1.0 - sp[v] for v in g_vuln)
                    S_g[g] = 1.0 - (curr_inf / base_inf_g[g])
                    
            gap = max(S_g.values()) - min(S_g.values()) if S_g else 0.0
            dp_fair = 1.0 - gap
            
            obj = (effectiveness / total_vuln) + lam * dp_fair
            if obj > best_obj:
                best_obj = obj
                best_cand = c
                
        if best_cand is None:
            best_cand = random.choice(list(candidates))
            
        T.add(best_cand)
        candidates.remove(best_cand)
        
    return T
