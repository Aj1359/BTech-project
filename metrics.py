import numpy as np
import networkx as nx
from scipy.stats import spearmanr, pearsonr
from simulators import simulate_cascade

def compute_gini(sp_values):
    n = len(sp_values)
    if n == 0:
        return 0.0
    arr = np.array(sp_values, dtype=float)
    mean_val = np.mean(arr)
    if mean_val == 0:
        return 0.0
    diff_sum = np.sum(np.abs(np.subtract.outer(arr, arr)))
    return float(diff_sum / (2.0 * n * n * mean_val))

def compute_consistency(sp_dict, sp0_dict, vulnerable):
    vuln = list(vulnerable)
    n = len(vuln)
    if n <= 1:
        return 1.0
    
    num, den = 0.0, 0.0
    for i in range(n):
        vi = vuln[i]
        for j in range(n):
            vj = vuln[j]
            s_ij = 1.0 - abs(sp0_dict[vi] - sp0_dict[vj])
            diff = abs(sp_dict[vi] - sp_dict[vj])
            num += diff * s_ij
            den += s_ij
            
    if den == 0:
        return 1.0
    return float(1.0 - (num / den))

def compute_group_metrics(G, sp_T_dict, sp0_dict, vulnerable):
    groups_dict = nx.get_node_attributes(G, 'group')
    group_nodes = {}
    for v in vulnerable:
        g = groups_dict.get(v, 0)
        if g not in group_nodes:
            group_nodes[g] = []
        group_nodes[g].append(v)
        
    S_g = {}
    for g, nodes in group_nodes.items():
        base_inf = sum(1.0 - sp0_dict[v] for v in nodes)
        if base_inf > 0:
            t_inf = sum(1.0 - sp_T_dict[v] for v in nodes)
            S_g[g] = 1.0 - (t_inf / base_inf)
        else:
            # Group has no baseline infection risk -> unthreatened, exclude from group gap/ratio
            pass
            
    if not S_g:
        return 0.0, 1.0, {}
        
    values = list(S_g.values())
    max_S = max(values)
    min_S = min(values)
    gap = max_S - min_S
    ratio = (min_S / max_S) if max_S > 0 else 1.0
    return float(gap), float(ratio), S_g

def inequality_aversion_welfare(S_g, alpha=0.5):
    """
    InequalityAversionWelfare(S_1..S_m, weights w_g, alpha):
    alpha near 1 favors total spread, alpha near 0 favors fairness.
    """
    if not S_g:
        return 0.0
    total_val = 0.0
    for g, sg in S_g.items():
        val = max(0.0, sg)
        total_val += (val ** alpha)
    return float(total_val / float(len(S_g)))

def demographic_parity_gap(S_g):
    """
    DemographicParityGap: max_g S_g - min_g S_g on group saved fractions.
    """
    if not S_g:
        return 0.0
    vals = list(S_g.values())
    return float(max(vals) - min(vals))


def compute_time_weighted_protection(G, M, T, model="ICM", pM=0.1, pT=0.1, R_eval=100, max_steps=20):
    groups_dict = nx.get_node_attributes(G, 'group')
    group_saved_by_step = {}
    
    for g in set(groups_dict.values()):
        group_saved_by_step[g] = [0] * (max_steps + 1)
        
    for _ in range(R_eval):
        state, time_step = simulate_cascade(G, M, T, model=model, pM=pM, pT=pT)
        for v, s in state.items():
            g = groups_dict.get(v, 0)
            if s == 1: # saved by truth
                step = min(max_steps, time_step.get(v, max_steps))
                group_saved_by_step[g][step] += 1
                
    time_weighted_scores = {}
    for g, counts in group_saved_by_step.items():
        score = 0.0
        total_g = sum(1 for v in G.nodes() if groups_dict.get(v, 0) == g)
        for step, count in enumerate(counts):
            weight = (max_steps - step + 1) / float(max_steps)
            score += count * weight
        time_weighted_scores[g] = score / float(total_g * R_eval) if total_g > 0 else 0.0
        
    return time_weighted_scores

def compute_all_metrics(G, M, T, model="ICM", pM=0.1, pT=0.1, gamma_T=1.0, c_T=1.0, c_M=1.0, tie_rule="truth", delay=0, R_eval=500, D_M=None):
    """
    Evaluates T on FRESH independent simulations and computes the full suite of v2 metrics.
    """
    # 1. Estimate sp using independent evaluation runs
    count_infected = {v: 0 for v in G.nodes()}
    for _ in range(R_eval):
        state, _ = simulate_cascade(
            G, M, T, model=model, pM=pM, pT=pT, 
            gamma_T=gamma_T, c_T=c_T, c_M=c_M, 
            tie_rule=tie_rule, delay=delay
        )
        for v, s in state.items():
            if s == -1:
                count_infected[v] += 1
                
    sp_T = {v: 1.0 - (count_infected[v] / float(R_eval)) for v in G.nodes()}
    
    # 2. Get baseline sp0
    count_infected_0 = {v: 0 for v in G.nodes()}
    for _ in range(R_eval):
        state0, _ = simulate_cascade(
            G, M, set(), model=model, pM=pM, pT=pT, 
            gamma_T=gamma_T, c_T=c_T, c_M=c_M, 
            tie_rule=tie_rule, delay=delay
        )
        for v, s in state0.items():
            if s == -1:
                count_infected_0[v] += 1
                
    sp0 = {v: 1.0 - (count_infected_0[v] / float(R_eval)) for v in G.nodes()}
    
    vulnerable = {v for v in G.nodes() if v not in M and sp0[v] < 1.0}
    if not vulnerable:
        vulnerable = set(G.nodes()) - set(M)
        
    sp_T_vuln = [sp_T[v] for v in vulnerable]
    
    # Saved Fraction
    expected_inf_0 = sum(1.0 - sp0[v] for v in vulnerable)
    expected_inf_T = sum(1.0 - sp_T[v] for v in vulnerable)
    saved_fraction = (1.0 - (expected_inf_T / expected_inf_0)) if expected_inf_0 > 0 else 1.0
    
    # Maximin sp
    maximin_sp = min(sp_T_vuln) if sp_T_vuln else 1.0
    
    # Gini & Consistency
    gini = compute_gini(sp_T_vuln)
    consistency = compute_consistency(sp_T, sp0, vulnerable)
    
    # Group Gap & Ratio
    group_gap, group_ratio, S_g = compute_group_metrics(G, sp_T, sp0, vulnerable)
    
    # Secondary metrics
    floor_share = sum(1 for v in vulnerable if sp_T[v] >= 0.5) / float(len(vulnerable)) if vulnerable else 1.0
    leximin_vec = sorted(sp_T_vuln)
    
    # Structural correlations
    degrees = dict(G.degree())
    deg_vuln = [degrees[v] for v in vulnerable]
    
    corr_deg = pearsonr(sp_T_vuln, deg_vuln)[0] if len(sp_T_vuln) > 1 and len(set(deg_vuln)) > 1 else 0.0
    
    if D_M is not None:
        dm_vuln = [D_M.get(v, 0) for v in vulnerable]
        corr_dist = pearsonr(sp_T_vuln, dm_vuln)[0] if len(sp_T_vuln) > 1 and len(set(dm_vuln)) > 1 else 0.0
    else:
        corr_dist = 0.0
        
    init_vuln = [1.0 - sp0[v] for v in vulnerable]
    corr_init = pearsonr(sp_T_vuln, init_vuln)[0] if len(sp_T_vuln) > 1 and len(set(init_vuln)) > 1 else 0.0
    
    # Time-weighted protection
    time_protection = compute_time_weighted_protection(G, M, T, model=model, pM=pM, pT=pT, R_eval=100)
    
    # Inequality Aversion & DP Gap
    ia_welfare = inequality_aversion_welfare(S_g, alpha=0.5)
    dp_gap = demographic_parity_gap(S_g)
    
    return {
        "Saved_Fraction": float(saved_fraction),
        "Maximin_SP": float(maximin_sp),
        "Gini": float(gini),
        "Consistency": float(consistency),
        "Group_Gap": float(group_gap),
        "Group_Ratio": float(group_ratio),
        "Inequality_Aversion_Welfare": float(ia_welfare),
        "DP_Gap": float(dp_gap),
        "Floor_Share": float(floor_share),
        "Corr_Degree": float(corr_deg),
        "Corr_Distance": float(corr_dist),
        "Corr_Init_Vuln": float(corr_init),
        "Time_Protection": time_protection,
        "Group_Saved_Frac": S_g,
        "Leximin_Sample": leximin_vec[:5]  # lowest 5 values
    }

