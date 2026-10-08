import random
import networkx as nx

def get_seeds(G, M, budget, **kwargs):
    """
    FWRRS (Saxena et al., 2023):
    Fairness-aware truth campaigning for known misinformation starters.
    Builds weighted reversible reachable trees / path estimates for vulnerable nodes,
    and greedily selects seeds prioritizing the worst-off community g*.
    """
    vulnerable_set = kwargs.get('vulnerable_set', set(G.nodes()) - set(M))
    groups = kwargs.get('groups', [0, 1])
    D = kwargs.get('D', {})
    D_M = kwargs.get('D_M', {})
    
    groups_dict = nx.get_node_attributes(G, 'group')
    if not groups_dict:
        groups_dict = {v: 0 for v in G.nodes()}
        
    vulnerable_by_group = {
        g: [v for v in vulnerable_set if groups_dict.get(v, 0) == g]
        for g in groups
    }
    
    candidates = set(G.nodes()) - set(M)
    T = set()
    
    # Pre-calculate candidate protection power per vulnerable node
    # Candidate c protects v if dist(c, v) <= dist(M, v)
    c_protects_v = {}
    for c in candidates:
        c_dict = D.get(c, {})
        protected_nodes = set()
        for v in vulnerable_set:
            if c_dict.get(v, float('inf')) <= D_M.get(v, float('inf')):
                protected_nodes.add(v)
        c_protects_v[c] = protected_nodes
        
    protected_per_group = {g: set() for g in groups}
    
    for _ in range(budget):
        if not candidates:
            break
            
        # Find community g* with lowest current protected fraction
        protected_fracs = {}
        for g in groups:
            total_g = len(vulnerable_by_group[g])
            if total_g > 0:
                protected_fracs[g] = len(protected_per_group[g]) / float(total_g)
            else:
                protected_fracs[g] = 1.0
                
        g_star = min(protected_fracs, key=protected_fracs.get)
        
        # Candidate selection favoring block_power in g*
        best_cand = None
        best_gstar_power = -1
        best_total_power = -1
        
        for c in candidates:
            c_nodes = c_protects_v[c]
            # Newly protected nodes in g*
            gstar_nodes = set(v for v in c_nodes if groups_dict.get(v, 0) == g_star)
            new_gstar_power = len(gstar_nodes - protected_per_group[g_star])
            
            # Total newly protected nodes
            new_total_power = sum(
                len(set(v for v in c_nodes if groups_dict.get(v, 0) == g) - protected_per_group[g])
                for g in groups
            )
            
            if (new_gstar_power > best_gstar_power) or \
               (new_gstar_power == best_gstar_power and new_total_power > best_total_power):
                best_cand = c
                best_gstar_power = new_gstar_power
                best_total_power = new_total_power
                
        if best_cand is None:
            best_cand = random.choice(list(candidates))
            
        T.add(best_cand)
        candidates.remove(best_cand)
        
        # Update protected sets per group
        for v in c_protects_v[best_cand]:
            g_v = groups_dict.get(v, 0)
            if g_v in protected_per_group:
                protected_per_group[g_v].add(v)
                
    return T
