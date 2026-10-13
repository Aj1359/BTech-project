import networkx as nx
import numpy as np

def compute_lfpr_scores(G, gamma=0.15, max_iter=100, tol=1e-6):
    """
    Computes Locally Fair PageRank (LFPR_N) from LSPR.pdf (Section III-B & IV-A).
    Each node j allocates a fraction alpha_g of its outgoing probability mass to group g.
    P_N[j, i] = alpha_g * (a_{ji} / out_g(j)) for i in g.
    If out_g(j) == 0, the mass alpha_g is redistributed uniformly to group g.
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
    
    # Target fairness fractions alpha_g (proportional to group size)
    alpha = {g: group_sizes[g] / float(n) for g in unique_groups}
    
    # Teleportation vector mu
    mu = np.zeros(n)
    for node in nodes:
        g = groups[node]
        mu[node_to_idx[node]] = alpha[g] / float(max(1, group_sizes[g]))
        
    # Precompute group-specific outgoing degrees out_g(j)
    out_deg_g = {node: {g: 0 for g in unique_groups} for node in nodes}
    for u in nodes:
        nbrs = list(G.successors(u) if G.is_directed() else G.neighbors(u))
        for v in nbrs:
            out_deg_g[u][groups[v]] += 1
            
    # Power iteration
    p = np.copy(mu)
    for _ in range(max_iter):
        p_next = gamma * mu
        
        # Accumulate contributions from nodes where out_g(j) == 0 (dangling for group g)
        uniform_contrib = {g: 0.0 for g in unique_groups}
        for u in nodes:
            u_idx = node_to_idx[u]
            p_u = p[u_idx]
            if p_u == 0:
                continue
            for g in unique_groups:
                if out_deg_g[u][g] == 0:
                    uniform_contrib[g] += (1.0 - gamma) * p_u * (alpha[g] / float(max(1, group_sizes[g])))
                    
        for u in nodes:
            g = groups[u]
            p_next[node_to_idx[u]] += uniform_contrib[g]
            
        # Graph transition propagation
        for u in nodes:
            u_idx = node_to_idx[u]
            p_u = p[u_idx]
            if p_u == 0:
                continue
            nbrs = list(G.successors(u) if G.is_directed() else G.neighbors(u))
            for v in nbrs:
                g_v = groups[v]
                k_out_g = out_deg_g[u][g_v]
                if k_out_g > 0:
                    trans_prob = alpha[g_v] / float(k_out_g)
                    p_next[node_to_idx[v]] += (1.0 - gamma) * p_u * trans_prob
                    
        # Check convergence
        diff = np.sum(np.abs(p_next - p))
        p = p_next
        if diff < tol:
            break
            
    # Normalize
    s = np.sum(p)
    if s > 0:
        p = p / s
        
    return {nodes[i]: float(p[i]) for i in range(n)}

def get_seeds(G, M, budget, **kwargs):
    """
    Selects seeds using Neighborhood Locally Fair PageRank (LFPR_N).
    """
    gamma = kwargs.get('gamma', 0.15)
    scores = compute_lfpr_scores(G, gamma=gamma)
    
    candidates = set(G.nodes()) - set(M)
    sorted_cands = sorted(list(candidates), key=lambda x: scores.get(x, 0.0), reverse=True)
    return list(sorted_cands[:budget])
