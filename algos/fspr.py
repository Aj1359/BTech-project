import networkx as nx
import numpy as np

def compute_fspr_scores(G, nu=0.15):
    """
    Computes Fairness-Sensitive PageRank (FSPR) scores as defined in FSPR.pdf.
    FSPR adjusts the teleportation (jump) vector v such that each demographic group C
    receives jump probability proportional to target fairness mass phi_C and in-degree.
    """
    groups = nx.get_node_attributes(G, 'group')
    n = G.number_of_nodes()
    if not groups:
        groups = {node: 0 for node in G.nodes()}
        
    unique_groups = list(set(groups.values()))
    
    # Calculate group sizes and target fairness masses phi_C
    group_sizes = {g: sum(1 for node in G.nodes() if groups[node] == g) for g in unique_groups}
    phi = {g: group_sizes[g] / float(n) for g in unique_groups}
    
    # In-degrees and group in-degree sums D_C
    in_degrees = dict(G.in_degree() if G.is_directed() else G.degree())
    D_C = {g: sum(in_degrees[node] for node in G.nodes() if groups[node] == g) for g in unique_groups}
    
    # Construct jump vector v according to FSPR.pdf Section II-A
    v_vec = {}
    for node in G.nodes():
        g = groups[node]
        deg = in_degrees[node]
        if D_C[g] > 0:
            v_vec[node] = phi[g] * (deg / float(D_C[g]))
        else:
            v_vec[node] = phi[g] / float(max(1, group_sizes[g]))
            
    # Normalize jump vector
    sum_v = sum(v_vec.values())
    if sum_v > 0:
        for node in v_vec:
            v_vec[node] /= sum_v
    else:
        v_vec = {node: 1.0 / n for node in G.nodes()}
        
    # Standard PageRank with teleportation vector v (alpha = 1 - nu = 0.85)
    try:
        scores = nx.pagerank(G, alpha=(1.0 - nu), personalization=v_vec, max_iter=200)
    except Exception:
        scores = v_vec
        
    return scores

def get_seeds(G, M, budget, **kwargs):
    """
    Selects seeds using Fairness-Sensitive PageRank (FSPR).
    """
    nu = kwargs.get('nu', 0.15)
    scores = compute_fspr_scores(G, nu=nu)
    
    candidates = set(G.nodes()) - set(M)
    sorted_cands = sorted(list(candidates), key=lambda x: scores.get(x, 0.0), reverse=True)
    return list(sorted_cands[:budget])
