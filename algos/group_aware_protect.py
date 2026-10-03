import random
import numpy as np

def get_seeds(G, M, budget, **kwargs):
    pM = kwargs.get('pM', 0.1)
    pT = kwargs.get('pT', 0.1)
    R = kwargs.get('R', 10)
    epsilon = kwargs.get('epsilon', 0.05)
    lam = kwargs.get('lam', 0.5)
    estimate_sp = kwargs.get('estimate_sp')
    D = kwargs.get('D')
    D_M = kwargs.get('D_M')
    vulnerable_set = kwargs.get('vulnerable_set')
    groups = kwargs.get('groups')
    
    Vg = {g: vulnerable_set.intersection({n for n in G.nodes() if G.nodes[n].get('group') == g}) for g in groups}
    
    sp0 = estimate_sp(G, M, set(), pM, pT, R)
    base_infected = {}
    for g in groups:
        base_inf = sum(1.0 - sp0[v] for v in Vg[g])
        base_infected[g] = base_inf if base_inf > 0 else 1.0
        
    D_T = {v: float('inf') for v in G.nodes()}
    T = set()
    candidates = set(G.nodes()) - set(M)
    
    for _ in range(budget):
        sp = estimate_sp(G, M, T, pM, pT, R)
        
        saved = {}
        for g in groups:
            if not Vg[g]: continue
            current_infected = sum(1.0 - sp[v] for v in Vg[g])
            saved[g] = 1.0 - (current_infected / base_infected[g])
            
        if not saved:
            break
            
        g_star = min(saved, key=saved.get)
        if not Vg[g_star]: break
        min_sp = min(sp[v] for v in Vg[g_star])
        worst = {v for v in Vg[g_star] if sp[v] == min_sp}
        
        cnt = {c: sum(1 for v in worst if D.get(c, {}).get(v, float('inf')) <= D_M[v]) for c in candidates}
        max_cnt = max(cnt.values()) if cnt else 0
        options = [c for c, val in cnt.items() if val == max_cnt]
        
        if len(options) > 1 and max_cnt == len(worst):
            next_worst = {v for v in Vg[g_star] if sp[v] <= min_sp + epsilon}
            cnt2 = {c: sum(1 for v in next_worst if D.get(c, {}).get(v, float('inf')) <= D_M[v]) for c in options}
            max_cnt2 = max(cnt2.values()) if cnt2 else 0
            options = [c for c, val in cnt2.items() if val == max_cnt2]
            
        if len(options) > 1:
            cnt3 = {c: sum(1 for v in vulnerable_set if min(D.get(c, {}).get(v, float('inf')), D_T.get(v, float('inf'))) <= D_M[v]) for c in options}
            max_cnt3 = max(cnt3.values()) if cnt3 else 0
            options = [c for c, val in cnt3.items() if val == max_cnt3]
            
        if len(options) > 1:
            scores = {}
            for c in options:
                arrivals_gstar = [min(D.get(c, {}).get(v, float('inf')), D_T.get(v, float('inf'))) for v in Vg[g_star]]
                arrivals_gstar = [x for x in arrivals_gstar if x < float('inf')]
                mean_gstar = np.mean(arrivals_gstar) if arrivals_gstar else float('inf')
                
                means = []
                for g in groups:
                    arrs = [min(D.get(c, {}).get(v, float('inf')), D_T.get(v, float('inf'))) for v in Vg[g]]
                    arrs = [x for x in arrs if x < float('inf')]
                    if arrs:
                        means.append(np.mean(arrs))
                spread = np.std(means) if means else 0
                
                scores[c] = mean_gstar + lam * spread
                
            min_score = min(scores.values()) if scores else 0
            options = [c for c, val in scores.items() if val == min_score]
            
        t = options[0] if options else random.choice(list(candidates))
        T.add(t)
        candidates.discard(t)
        for v in G.nodes():
            D_T[v] = min(D.get(t, {}).get(v, float('inf')), D_T.get(v, float('inf')))
            
    return T
