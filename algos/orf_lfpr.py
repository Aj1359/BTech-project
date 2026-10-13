import networkx as nx
import numpy as np

def compute_orf_lfpr_scores(G, gamma=0.15):
    """
    Computes One-Step Refinement Locally Fair PageRank (ORF-LFPR_N) from LSPR.pdf (Section IV & VI).
    1. Computes closed-form Mean-Field LFPR_N approximation (Eq. 12).
    2. Applies a single fairness-aware propagation step (Eq. 69) to incorporate node-specific local neighborhood.
    """
    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return {}
        
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    groups = nx.get_node_attributes(G, 'group')
    if not groups:
        groups = {node: 0 for node in nodes}
        
    unique_groups = list(set(groups.values()))
    group_sizes = {g: sum(1 for node in nodes if groups[node] == g) for g in unique_groups}
    alpha = {g: group_sizes[g] / float(n) for g in unique_groups}
    
    in_degrees = dict(G.in_degree() if G.is_directed() else G.degree())
    
    # d_in_g: total in-degree of group g divided by n (Eq. 12)
    d_in_g = {}
    for g in unique_groups:
        total_in_g = sum(in_degrees[node] for node in nodes if groups[node] == g)
        d_in_g[g] = total_in_g / float(n) if n > 0 else 1.0
        if d_in_g[g] == 0:
            d_in_g[g] = 1.0
            
    # Step 1: Mean-Field estimate p_hat_N(u)
    p_mf = np.zeros(n)
    for u in nodes:
        g = groups[u]
        term1 = gamma * (alpha[g] / float(max(1, group_sizes[g])))
        term2 = (1.0 - gamma) * alpha[g] * (in_degrees[u] / float(d_in_g[g])) * (1.0 / float(n))
        p_mf[node_to_idx[u]] = term1 + term2
        
    sum_mf = np.sum(p_mf)
    if sum_mf > 0:
        p_mf /= sum_mf
        
    # Step 2: One-Step Refinement (ORF)
    out_deg_g = {node: {g: 0 for g in unique_groups} for node in nodes}
    for u in nodes:
        nbrs = list(G.successors(u) if G.is_directed() else G.neighbors(u))
        for v in nbrs:
            out_deg_g[u][groups[v]] += 1
            
    mu = np.zeros(n)
    for node in nodes:
        g = groups[node]
        mu[node_to_idx[node]] = alpha[g] / float(max(1, group_sizes[g]))
        
    p_refined = gamma * mu
    
    # Uniform contribution from nodes with out_g(u) == 0
    uniform_contrib = {g: 0.0 for g in unique_groups}
    for u in nodes:
        u_idx = node_to_idx[u]
        p_u = p_mf[u_idx]
        if p_u == 0:
            continue
        for g in unique_groups:
            if out_deg_g[u][g] == 0:
                uniform_contrib[g] += (1.0 - gamma) * p_u * (alpha[g] / float(max(1, group_sizes[g])))
                
    for u in nodes:
        g = groups[u]
        p_refined[node_to_idx[u]] += uniform_contrib[g]
        
    # Local graph propagation step
    for u in nodes:
        u_idx = node_to_idx[u]
        p_u = p_mf[u_idx]
        if p_u == 0:
            continue
        nbrs = list(G.successors(u) if G.is_directed() else G.neighbors(u))
        for v in nbrs:
            g_v = groups[v]
            k_out_g = out_deg_g[u][g_v]
            if k_out_g > 0:
                trans_prob = alpha[g_v] / float(k_out_g)
                p_refined[node_to_idx[v]] += (1.0 - gamma) * p_u * trans_prob
                
    # Normalize refined vector
    s = np.sum(p_refined)
    if s > 0:
        p_refined /= s
        
    return {nodes[i]: float(p_refined[i]) for i in range(n)}

def get_seeds(G, M, budget, **kwargs):
    """
    Selects seeds using One-Step Refinement Locally Fair PageRank (ORF-LFPR_N).
    Achieves high accuracy in O(n + m) time with a single graph traversal.
    """
    gamma = kwargs.get('gamma', 0.15)
    scores = compute_orf_lfpr_scores(G, gamma=gamma)
    
    candidates = set(G.nodes()) - set(M)
    sorted_cands = sorted(list(candidates), key=lambda x: scores.get(x, 0.0), reverse=True)
    return list(sorted_cands[:budget])
