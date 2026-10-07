import random
import math
import numpy as np
import networkx as nx
from simulators import estimate_sp

def place_m(G, m=None, mode="highdeg", seed=42):
    """
    Selects misinformation seed set M according to mode:
    'highdeg', 'random', 'majority', 'minority', 'split_prop', 'split_equal', 'clustered', 'dispersed'.
    Ensures baseline validation: redraws seed if vulnerable set < 10% of network or empty.
    """
    rng = random.Random(seed)
    n = G.number_of_nodes()
    
    if m is None:
        m = max(1, math.ceil(0.03 * n))
        
    # Get largest connected component pool
    if G.is_directed():
        components = list(nx.strongly_connected_components(G))
        largest_comp = max(components, key=len) if components else set(G.nodes())
    else:
        components = list(nx.connected_components(G))
        largest_comp = max(components, key=len) if components else set(G.nodes())
        
    pool = list(largest_comp)
    
    # Get group mappings if available
    groups_dict = nx.get_node_attributes(G, 'group')
    if not groups_dict:
        # Default 80/20 group split if missing
        np_rng = np.random.RandomState(seed)
        group_arr = np_rng.choice([0, 1], size=n, p=[0.8, 0.2])
        groups_dict = {v: group_arr[i] for i, v in enumerate(G.nodes())}
        nx.set_node_attributes(G, groups_dict, 'group')
        
    group_counts = {}
    for v, g in groups_dict.items():
        group_counts[g] = group_counts.get(g, 0) + 1
        
    largest_group = max(group_counts.keys(), key=lambda g: group_counts[g])
    smallest_group = min(group_counts.keys(), key=lambda g: group_counts[g])
    
    for attempt in range(50):
        current_seed = seed + attempt
        rng = random.Random(current_seed)
        M = set()
        
        if mode in ["highdeg", "highest_degree"]:
            degrees = dict(G.degree(pool))
            # Sort by degree, break ties with rng
            sorted_pool = sorted(pool, key=lambda x: (degrees[x], rng.random()), reverse=True)
            M = set(sorted_pool[:m])
            
        elif mode in ["random", "Random"]:
            M = set(rng.sample(pool, min(m, len(pool))))
            
        elif mode in ["majority", "Majority"]:
            maj_pool = [v for v in pool if groups_dict[v] == largest_group]
            M = set(rng.sample(maj_pool, min(m, len(maj_pool))))
            
        elif mode in ["minority", "Minority"]:
            min_pool = [v for v in pool if groups_dict[v] == smallest_group]
            M = set(rng.sample(min_pool, min(m, len(min_pool))))
            
        elif mode == "split_prop":
            total_nodes = len(G)
            for g, cnt in group_counts.items():
                m_g = max(1, round(m * (cnt / float(total_nodes))))
                g_pool = [v for v in pool if groups_dict[v] == g]
                if g_pool:
                    M.update(rng.sample(g_pool, min(m_g, len(g_pool))))
            M = set(list(M)[:m])
            
        elif mode in ["split_equal", "Split"]:
            num_groups = len(group_counts)
            per_group = max(1, m // num_groups)
            for g in group_counts:
                g_pool = [v for v in pool if groups_dict[v] == g]
                if g_pool:
                    M.update(rng.sample(g_pool, min(per_group, len(g_pool))))
            M = set(list(M)[:m])
            
        elif mode == "clustered":
            r = rng.choice(pool)
            bfs_tree = nx.bfs_tree(G, r)
            M = set(list(bfs_tree.nodes())[:m])
            
        elif mode == "dispersed":
            # Farthest point selection
            start = rng.choice(pool)
            M = {start}
            while len(M) < m:
                # Find node with max min-distance to current M
                dist_map = {}
                for v in pool:
                    if v not in M:
                        min_d = float('inf')
                        for s in M:
                            try:
                                d = nx.shortest_path_length(G, s, v)
                            except nx.NetworkXNoPath:
                                d = 9999
                            if d < min_d:
                                min_d = d
                        dist_map[v] = min_d
                if not dist_map:
                    break
                next_node = max(dist_map, key=dist_map.get)
                M.add(next_node)
        else:
            M = set(rng.sample(pool, min(m, len(pool))))
            
        # Baseline validation with T = empty
        sp0 = estimate_sp(G, M, set(), model="ICM", pM=0.1, pT=0.1, R=100)
        vulnerable = {v for v in G.nodes() if v not in M and sp0[v] < 1.0}
        
        if len(vulnerable) >= max(1, int(0.1 * n)):
            return M, vulnerable, sp0
            
    # Fallback if validation retries exhausted
    sp0 = estimate_sp(G, M, set(), model="ICM", pM=0.1, pT=0.1, R=100)
    vulnerable = {v for v in G.nodes() if v not in M and sp0[v] < 1.0}
    return M, vulnerable, sp0
