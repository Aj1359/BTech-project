import networkx as nx

def get_seeds(G, M, budget, **kwargs):
    groups = nx.get_node_attributes(G, 'group')
    if not groups:
        groups = {n: 0 for n in G.nodes()}
        
    # LFPR modifies the transition matrix so that a node distributes its PageRank
    # equally among the different groups it connects to, and then equally within those groups.
    W = nx.DiGraph() if G.is_directed() else nx.Graph()
    W.add_nodes_from(G.nodes())
    
    for u in G.nodes():
        neighbors = list(G.neighbors(u))
        if not neighbors:
            continue
            
        group_counts = {}
        for v in neighbors:
            g = groups[v]
            group_counts[g] = group_counts.get(g, 0) + 1
            
        num_groups_connected = len(group_counts)
        
        for v in neighbors:
            g = groups[v]
            # Weight = (1 / number of distinct groups connected) * (1 / number of nodes in that group connected)
            weight = 1.0 / (num_groups_connected * group_counts[g])
            W.add_edge(u, v, weight=weight)
            
    pr = nx.pagerank(W, weight='weight')
    
    candidates = set(G.nodes()) - set(M)
    sorted_cands = sorted(list(candidates), key=lambda x: pr[x], reverse=True)
    
    return set(sorted_cands[:budget])
