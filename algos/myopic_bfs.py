import networkx as nx

def get_seeds(G, M, budget, **kwargs):
    T = set()
    candidates = set(G.nodes()) - set(M)
    pT = kwargs.get('pT', 0.1)
    
    if not candidates:
        return T
        
    degrees = dict(G.degree())
    first_seed = max(candidates, key=lambda x: degrees.get(x, 0))
    T.add(first_seed)
    candidates.discard(first_seed)
    
    act_prob = {v: 0.0 for v in G.nodes()}
    act_prob[first_seed] = 1.0
    
    def update_act_prob(seed):
        q = [(seed, 1.0, 0)]
        visited = {seed: 1.0}
        while q:
            curr, prob, depth = q.pop(0)
            if depth >= 2: # Limiting depth to keep it fast as per Myopic BFS approximation
                continue
            for neighbor in G.neighbors(curr):
                if neighbor not in visited:
                    # Probabilistic OR combination: P(A or B) = P(A) + P(B) - P(A)*P(B)
                    act_prob[neighbor] = act_prob[neighbor] + (prob * pT) - (act_prob[neighbor] * prob * pT)
                    visited[neighbor] = prob * pT
                    q.append((neighbor, prob * pT, depth + 1))
                    
    update_act_prob(first_seed)
    
    while len(T) < budget and candidates:
        # Select the candidate with the lowest activation probability
        best_cand = min(candidates, key=lambda c: act_prob[c])
        T.add(best_cand)
        candidates.discard(best_cand)
        update_act_prob(best_cand)
        
    return T
