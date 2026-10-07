import random
import networkx as nx

def get_seeds(G, M, budget, **kwargs):
    estimate_sp = kwargs.get('estimate_sp')
    pM = kwargs.get('pM', 0.1)
    pT = kwargs.get('pT', 0.1)
    model = kwargs.get('model', 'ICM')
    pop_size = kwargs.get('pop_size', 10)
    max_it = kwargs.get('max_it', 5)
    
    candidates = list(set(G.nodes()) - set(M))
    if len(candidates) <= budget:
        return set(candidates)
        
    vulnerable_set = kwargs.get('vulnerable_set', set(candidates))
    vulnerable = list(vulnerable_set)
    
    sp0 = estimate_sp(G, M, set(), model=model, pM=pM, pT=pT, R=20)
    exp_inf_0 = sum(1.0 - sp0[v] for v in vulnerable) if vulnerable else 1.0
    
    # Initialize population of k-subsets
    population = [set(random.sample(candidates, budget)) for _ in range(pop_size)]
    
    def fitness(X):
        sp = estimate_sp(G, M, X, model=model, pM=pM, pT=pT, R=20)
        exp_inf_T = sum(1.0 - sp[v] for v in vulnerable) if vulnerable else 0.0
        saved_frac = 1.0 - (exp_inf_T / exp_inf_0) if exp_inf_0 > 0 else 1.0
        maximin = min(sp[v] for v in vulnerable) if vulnerable else 1.0
        return 0.5 * saved_frac + 0.5 * maximin

    for it in range(max_it):
        scores = [fitness(X) for X in population]
        best_idx = scores.index(max(scores))
        alpha_wolf = population[best_idx]
        
        new_pop = []
        for X in population:
            # Shift wolf set toward alpha wolf
            EN = list(alpha_wolf - X)
            EO = list(X - alpha_wolf)
            step = min(1, len(EN), len(EO))
            if step > 0:
                new_X = (X - set(random.sample(EO, step))) | set(random.sample(EN, step))
                new_pop.append(new_X)
            else:
                new_pop.append(X)
        population = new_pop
        
    scores = [fitness(X) for X in population]
    best_idx = scores.index(max(scores))
    return population[best_idx]
