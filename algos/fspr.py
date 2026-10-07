import networkx as nx

def get_seeds(G, M, budget, **kwargs):
    groups = nx.get_node_attributes(G, 'group')
    if not groups:
        # Fallback if no groups
        groups = {n: 0 for n in G.nodes()}
        
    group_counts = {}
    for g in groups.values():
        group_counts[g] = group_counts.get(g, 0) + 1
        
    # FSPR biases the random jump vector towards the minority group
    # Probability of jumping to a node is inversely proportional to its group's size
    p_vector = {}
    for n in G.nodes():
        g = groups[n]
        p_vector[n] = 1.0 / group_counts[g]
        
    # Normalize the personalization vector
    sum_p = sum(p_vector.values())
    for n in p_vector:
        p_vector[n] /= sum_p
        
    pr = nx.pagerank(G, personalization=p_vector)
    
    candidates = set(G.nodes()) - set(M)
    sorted_cands = sorted(list(candidates), key=lambda x: pr[x], reverse=True)
    
    return set(sorted_cands[:budget])
