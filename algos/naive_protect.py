import random

def get_seeds(G, M, budget, **kwargs):
    epsilon = kwargs.get('epsilon', 0.05)
    D = kwargs.get('D', {})
    D_M = kwargs.get('D_M', {})
    vulnerable_set = kwargs.get('vulnerable_set', set())
    
    n_M = {v: sum(1 for m in M if D.get(m, {}).get(v, float('inf')) == D_M.get(v, float('inf'))) for v in vulnerable_set}
    n_T = {v: 0 for v in vulnerable_set}
    candidates = set(G.nodes()) - set(M)
    T = set()
    target = {v for v in vulnerable_set if D_M.get(v) == 1}
    
    while len(T) < budget and target:
        min_nT = min(n_T[v] for v in target)
        
        diff = {}
        if min_nT == 0:
            for v in target:
                diff[v] = n_M[v] if n_T[v] == 0 else float('-inf')
            eps_prime = 1
        else:
            for v in target:
                diff[v] = (n_M[v] - n_T[v]) / n_M[v] if n_M[v] > 0 else float('-inf')
            eps_prime = epsilon
            
        max_diff = max(diff.values()) if diff else 0
        worst = {v for v in target if diff[v] == max_diff}
        
        cnt = {c: sum(1 for v in worst if D.get(c, {}).get(v, float('inf')) <= D_M.get(v, float('inf'))) for c in candidates}
        max_cnt = max(cnt.values()) if cnt else 0
        options = [c for c, val in cnt.items() if val == max_cnt]
        
        if len(options) > 1 and max_cnt == len(worst):
            next_target = {v for v in target if diff[v] >= max_diff - eps_prime}
            cnt2 = {c: sum(1 for v in next_target if D.get(c, {}).get(v, float('inf')) <= D_M.get(v, float('inf'))) for c in options}
            max_cnt2 = max(cnt2.values()) if cnt2 else 0
            options = [c for c, val in cnt2.items() if val == max_cnt2]
            
        if len(options) > 1:
            cnt3 = {c: sum(1 for v in vulnerable_set if D.get(c, {}).get(v, float('inf')) <= D_M.get(v, float('inf'))) for c in options}
            max_cnt3 = max(cnt3.values()) if cnt3 else 0
            options = [c for c, val in cnt3.items() if val == max_cnt3]
            
        t = options[0] if options else random.choice(list(candidates))
        T.add(t)
        candidates.discard(t)
        
        for v in vulnerable_set:
            n_T[v] = sum(1 for s in T if D.get(s, {}).get(v, float('inf')) == D_M.get(v, float('inf')))
            
        target = target - {v for v in vulnerable_set if n_T[v] >= n_M[v]}
        
    if len(T) < budget:
        def coverage(c):
            return sum(1 for v in vulnerable_set if D.get(c, {}).get(v, float('inf')) <= D_M.get(v, float('inf')))
        sorted_cands = sorted(list(candidates), key=coverage, reverse=True)
        T.update(sorted_cands[:budget - len(T)])
        
    return T
