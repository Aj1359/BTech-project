import random
import numpy as np

def get_seeds(G, M, budget, **kwargs):
    pM = kwargs.get('pM', 0.1)
    pT = kwargs.get('pT', 0.1)
    R = kwargs.get('R', 10)
    epsilon = kwargs.get('epsilon', 0.05)
    estimate_sp = kwargs.get('estimate_sp')
    D = kwargs.get('D')
    D_M = kwargs.get('D_M')
    vulnerable_set = kwargs.get('vulnerable_set')
    
    D_T = {v: float('inf') for v in vulnerable_set}
    candidates = set(G.nodes()) - set(M)
    model = kwargs.get('model', 'COICM')
    T = []

    for _ in range(budget):
        sp = estimate_sp(G, M, set(T), model_type=model, pM=pM, pT=pT, R_runs=R)
        if not vulnerable_set: break
        min_sp = min(sp[v] for v in vulnerable_set)
        worst = {v for v in vulnerable_set if sp[v] == min_sp}

        cnt = {}
        for c in candidates:
            cnt[c] = sum(1 for v in worst if D.get(c, {}).get(v, float('inf')) <= D_M[v])

        max_cnt = max(cnt.values()) if cnt else 0
        options = [c for c, val in cnt.items() if val == max_cnt]

        if len(options) > 1 and max_cnt == len(worst):
            next_worst = {v for v in vulnerable_set if sp[v] <= min_sp + epsilon}
            cnt2 = {c: sum(1 for v in next_worst if D.get(c, {}).get(v, float('inf')) <= D_M[v]) for c in options}
            max_cnt2 = max(cnt2.values()) if cnt2 else 0
            options = [c for c, val in cnt2.items() if val == max_cnt2]

        if len(options) > 1:
            cnt3 = {}
            for c in options:
                D_Tc = {v: min(D.get(c, {}).get(v, float('inf')), D_T[v]) for v in vulnerable_set}
                cnt3[c] = sum(1 for v in vulnerable_set if D_Tc[v] <= D_M[v])
            max_cnt3 = max(cnt3.values()) if cnt3 else 0
            options = [c for c, val in cnt3.items() if val == max_cnt3]

        if len(options) > 1:
            avg_arrival = {}
            for c in options:
                D_Tc = {v: min(D.get(c, {}).get(v, float('inf')), D_T[v]) for v in vulnerable_set}
                valid_arrivals = [D_Tc[v] for v in vulnerable_set if D_Tc[v] < float('inf')]
                avg_arrival[c] = np.mean(valid_arrivals) if valid_arrivals else float('inf')
            min_avg = min(avg_arrival.values()) if avg_arrival else 0
            options = [c for c, val in avg_arrival.items() if val == min_avg]

        t = options[0] if options else random.choice(list(candidates))
        T.append(t)
        candidates.discard(t)
        for v in vulnerable_set:
            D_T[v] = min(D.get(t, {}).get(v, float('inf')), D_T[v])

    return T
