import networkx as nx
import random

def get_seeds(G, M, budget, **kwargs):
    T = set()
    candidates = set(G.nodes()) - set(M)
    if not candidates:
        return T
        
    degrees = dict(G.degree())
    
    # Start with a random seed as per algorithm, or highest degree for stability
    first_seed = max(candidates, key=lambda x: degrees.get(x, 0))
    T.add(first_seed)
    candidates.discard(first_seed)
    
    while len(T) < budget and candidates:
        min_deg = min(degrees[c] for c in candidates)
        min_deg_nodes = [c for c in candidates if degrees[c] == min_deg]
        
        # Sort by harmonic centrality (approximated via ego graph for speed on large networks)
        best_x = None
        min_hc = float('inf')
        
        for u in min_deg_nodes:
            # Approximate harmonic centrality using depth 2
            ego = nx.ego_graph(G, u, radius=2)
            lengths = nx.single_source_shortest_path_length(ego, u)
            hc = sum(1.0/d for v, d in lengths.items() if d > 0)
            
            if hc < min_hc:
                min_hc = hc
                best_x = u
                
        # Choose highest degree neighbor of the node with lowest harmonic centrality
        neighbors = list(G.neighbors(best_x))
        valid_neighbors = [n for n in neighbors if n in candidates]
        
        if valid_neighbors:
            y = max(valid_neighbors, key=lambda n: degrees[n])
        else:
            y = best_x
            
        T.add(y)
        candidates.discard(y)
        
    return T
