import random

def get_seeds(G, M, budget, **kwargs):
    estimate_sp = kwargs.get('estimate_sp')
    pM = kwargs.get('pM', 0.1)
    pT = kwargs.get('pT', 0.1)
    model = kwargs.get('model', 'ICM')
    R = kwargs.get('R', 50)
    vulnerable_set = kwargs.get('vulnerable_set', set(G.nodes()) - set(M))
    
    vulnerable = list(vulnerable_set)
    candidates = list(set(G.nodes()) - set(M))
    T = set()
    
    for _ in range(budget):
        min_sp = {}
        avg_sp = {}
        min_cnt = {}
        
        for c in candidates:
            temp_T = T | {c}
            sp = estimate_sp(G, M, temp_T, model=model, pM=pM, pT=pT, R=R)
            sp_vuln = [float(sp[v]) for v in vulnerable]
            
            if sp_vuln:
                m_sp = min(sp_vuln)
                min_sp[c] = m_sp
                avg_sp[c] = sum(sp_vuln) / len(sp_vuln)
                min_cnt[c] = sum(1 for val in sp_vuln if val == m_sp)
            else:
                min_sp[c] = 1.0
                avg_sp[c] = 1.0
                min_cnt[c] = 0
                
        if not min_sp:
            break
            
        best = max(min_sp.values())
        cand = [c for c in candidates if min_sp[c] == best]
        
        if len(cand) > 1:
            min_c_val = min(min_cnt[c] for c in cand)
            cand = [c for c in cand if min_cnt[c] == min_c_val]
            
        if len(cand) > 1:
            max_avg = max(avg_sp[c] for c in cand)
            cand = [c for c in cand if avg_sp[c] == max_avg]
            
        t = cand[0] if cand else random.choice(candidates)
        T.add(t)
        candidates.remove(t)
        
    return T
