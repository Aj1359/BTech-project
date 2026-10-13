import networkx as nx

def get_seeds(G, M, budget, **kwargs):
    T = []
    candidates = set(G.nodes()) - set(M)
    pT = kwargs.get('pT', 0.1)

    if not candidates:
        return T

    degrees = dict(G.degree())
    first_seed = max(candidates, key=lambda x: degrees.get(x, 0))
    T.append(first_seed)
    candidates.discard(first_seed)

    act_prob = {v: 0.0 for v in G.nodes()}
    act_prob[first_seed] = 1.0

    def update_act_prob(seed):
        q = [(seed, 1.0, 0)]
        visited = {seed: 1.0}
        while q:
            curr, prob, depth = q.pop(0)
            if depth >= 2:
                continue
            for neighbor in G.neighbors(curr):
                if neighbor not in visited:
                    act_prob[neighbor] = act_prob[neighbor] + (prob * pT) - (act_prob[neighbor] * prob * pT)
                    visited[neighbor] = prob * pT
                    q.append((neighbor, prob * pT, depth + 1))

    update_act_prob(first_seed)

    while len(T) < budget and candidates:
        best_cand = min(candidates, key=lambda c: act_prob[c])
        T.append(best_cand)
        candidates.discard(best_cand)
        update_act_prob(best_cand)

    return T
