import networkx as nx

def compute_mf_fspr_scores(G, nu=0.15):
    """
    Computes Mean-Field Fairness-Sensitive PageRank (MF-FSPR) scores
    using the closed-form formulation derived in FSPR.pdf (Section II-A, Eq. 11-12):
    p_hat(u) = nu * phi_{C(u)} * (k_{in}(u) / D_{C(u)}) + (1 - nu) * (k_{in}(u) / <k_{in}>) * (1 / N)
    """
    n = G.number_of_nodes()
    if n == 0:
        return {}
        
    groups = nx.get_node_attributes(G, 'group')
    if not groups:
        groups = {node: 0 for node in G.nodes()}
        
    unique_groups = list(set(groups.values()))
    group_sizes = {g: sum(1 for node in G.nodes() if groups[node] == g) for g in unique_groups}
    phi = {g: group_sizes[g] / float(n) for g in unique_groups}
    
    in_degrees = dict(G.in_degree() if G.is_directed() else G.degree())
    total_in_degree = sum(in_degrees.values())
    avg_k_in = (total_in_degree / float(n)) if n > 0 else 1.0
    if avg_k_in == 0:
        avg_k_in = 1.0
        
    D_C = {g: sum(in_degrees[node] for node in G.nodes() if groups[node] == g) for g in unique_groups}
    
    scores = {}
    for u in G.nodes():
        g = groups[u]
        k_in = in_degrees[u]
        
        # Teleportation component
        if D_C[g] > 0:
            term1 = nu * phi[g] * (k_in / float(D_C[g]))
        else:
            term1 = nu * phi[g] / float(max(1, group_sizes[g]))
            
        # Propagation component
        term2 = (1.0 - nu) * (k_in / float(avg_k_in)) * (1.0 / float(n))
        
        scores[u] = term1 + term2
        
    # Normalize scores
    sum_scores = sum(scores.values())
    if sum_scores > 0:
        for u in scores:
            scores[u] /= sum_scores
            
    return scores

def get_seeds(G, M, budget, **kwargs):
    """
    Selects seeds using Mean-Field Fairness-Sensitive PageRank (MF-FSPR).
    Runs in linear time O(n + m) without iterative power iterations.
    """
    nu = kwargs.get('nu', 0.15)
    scores = compute_mf_fspr_scores(G, nu=nu)
    
    candidates = set(G.nodes()) - set(M)
    sorted_cands = sorted(list(candidates), key=lambda x: scores.get(x, 0.0), reverse=True)
    return list(sorted_cands[:budget])
