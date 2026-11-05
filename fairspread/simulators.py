import random
import numpy as np
import networkx as nx

def simulate_icm(G, M, T, pM=0.1, pT=0.1, tie_rule="truth", delay=0):
    """
    Independent Cascade Model (D1, D2, D4).
    Supports pT != pM (MCICM), custom tie rules ('truth', 'misinfo', 'coin'),
    and detection delay r (truth seeds activate at step r).
    """
    state = {v: 0 for v in G.nodes()}       # 0: inactive, 1: positive (truth), -1: negative (misinfo)
    time_step = {v: float('inf') for v in G.nodes()}
    
    # Misinformation activates at step 0
    for m in M:
        state[m] = -1
        time_step[m] = 0
        
    frontier_pos = set()
    frontier_neg = set(M)
    
    # If delay == 0, truth activates at step 0
    if delay == 0:
        for t in T:
            if state[t] != -1:  # truth seed node
                state[t] = 1
                time_step[t] = 0
                frontier_pos.add(t)
    
    step = 0
    max_steps = G.number_of_nodes() + 10
    
    while (frontier_pos or frontier_neg or step < delay) and step < max_steps:
        step += 1
        
        # Activate truth seeds at delay step if delay > 0
        if step == delay and delay > 0:
            for t in T:
                if state[t] == 0:
                    state[t] = 1
                    time_step[t] = step
                    frontier_pos.add(t)
        
        new_pos = set()
        new_neg = set()
        
        # Determine propagation probabilities (can be scalar or edge-dict/function)
        for u in frontier_pos:
            for v in G.neighbors(u):
                if state[v] == 0:
                    prob = pT(u, v) if callable(pT) else (pT.get((u, v), 0.1) if isinstance(pT, dict) else pT)
                    if random.random() < prob:
                        new_pos.add(v)
                        
        for u in frontier_neg:
            for v in G.neighbors(u):
                if state[v] == 0:
                    prob = pM(u, v) if callable(pM) else (pM.get((u, v), 0.1) if isinstance(pM, dict) else pM)
                    if random.random() < prob:
                        new_neg.add(v)
                        
        # Resolve conflicts (tie rule)
        conflicts = new_pos & new_neg
        final_pos = new_pos - conflicts
        final_neg = new_neg - conflicts
        
        for v in conflicts:
            if tie_rule == "truth":
                final_pos.add(v)
            elif tie_rule == "misinfo":
                final_neg.add(v)
            elif tie_rule == "coin":
                if random.random() < 0.5:
                    final_pos.add(v)
                else:
                    final_neg.add(v)
            else:
                final_pos.add(v) # default truth wins
                
        for v in final_pos:
            state[v] = 1
            time_step[v] = step
        for v in final_neg:
            state[v] = -1
            time_step[v] = step
            
        frontier_pos = final_pos
        frontier_neg = final_neg
        
    return state, time_step


def simulate_clt(G, M, T, gamma_T=1.0, tie_rule="truth", delay=0, in_weights=None):
    """
    Competitive Linear Threshold Model (D3).
    Thresholds theta_v ~ U[0,1] resampled per run.
    Weight of in-neighbors w(u,v) defaults to 1/in_degree(v).
    Truth-weight multiplier gamma_T.
    """
    state = {v: 0 for v in G.nodes()}
    time_step = {v: float('inf') for v in G.nodes()}
    
    # Resample thresholds per node
    thresholds = {v: random.random() for v in G.nodes()}
    
    # Calculate or use provided edge weights w(u, v)
    if in_weights is None:
        in_weights = {}
        for v in G.nodes():
            deg = G.in_degree(v) if G.is_directed() else G.degree(v)
            w = 1.0 / deg if deg > 0 else 0.0
            for u in (G.predecessors(v) if G.is_directed() else G.neighbors(v)):
                in_weights[(u, v)] = w
                
    # Activate misinfo at step 0
    for m in M:
        state[m] = -1
        time_step[m] = 0
        
    if delay == 0:
        for t in T:
            if state[t] != -1:
                state[t] = 1
                time_step[t] = 0
                
    step = 0
    changed = True
    max_steps = G.number_of_nodes() + 10
    
    while changed and step < max_steps:
        step += 1
        changed = False
        
        if step == delay and delay > 0:
            for t in T:
                if state[t] == 0:
                    state[t] = 1
                    time_step[t] = step
                    changed = True
                    
        nodes_to_activate = {}
        
        for v in G.nodes():
            if state[v] != 0:
                continue
                
            in_neighbors = G.predecessors(v) if G.is_directed() else G.neighbors(v)
            W_pos = 0.0
            W_neg = 0.0
            
            for u in in_neighbors:
                w = in_weights.get((u, v), 0.0)
                if state[u] == 1:
                    W_pos += gamma_T * w
                elif state[u] == -1:
                    W_neg += w
                    
            theta = thresholds[v]
            pos_reaches = (W_pos >= theta)
            neg_reaches = (W_neg >= theta)
            
            if pos_reaches and neg_reaches:
                if tie_rule == "truth":
                    nodes_to_activate[v] = 1
                elif tie_rule == "misinfo":
                    nodes_to_activate[v] = -1
                elif tie_rule == "coin":
                    nodes_to_activate[v] = 1 if random.random() < 0.5 else -1
            elif pos_reaches:
                nodes_to_activate[v] = 1
            elif neg_reaches:
                nodes_to_activate[v] = -1
                
        if nodes_to_activate:
            changed = True
            for v, st in nodes_to_activate.items():
                state[v] = st
                time_step[v] = step
                
    return state, time_step


def simulate_weighted_cascade(G, M, T, c_T=1.0, c_M=1.0, tie_rule="truth", delay=0):
    """
    Weighted Cascade Model (D4).
    p_T(u->v) = c_T / in_degree(v), p_M(u->v) = c_M / in_degree(v).
    """
    pT_dict = {}
    pM_dict = {}
    for v in G.nodes():
        deg = G.in_degree(v) if G.is_directed() else G.degree(v)
        p_base = 1.0 / deg if deg > 0 else 0.0
        for u in (G.predecessors(v) if G.is_directed() else G.neighbors(v)):
            pT_dict[(u, v)] = min(1.0, c_T * p_base)
            pM_dict[(u, v)] = min(1.0, c_M * p_base)
            
    return simulate_icm(G, M, T, pM=pM_dict, pT=pT_dict, tie_rule=tie_rule, delay=delay)


def simulate_cascade(G, M, T, model="ICM", pM=0.1, pT=0.1, gamma_T=1.0, c_T=1.0, c_M=1.0, tie_rule="truth", delay=0):
    """
    Unified entry point for running a single diffusion simulation.
    """
    if model in ["ICM", "D1", "D2", "COICM", "MCICM"]:
        return simulate_icm(G, M, T, pM=pM, pT=pT, tie_rule=tie_rule, delay=delay)
    elif model in ["CLT", "D3"]:
        return simulate_clt(G, M, T, gamma_T=gamma_T, tie_rule=tie_rule, delay=delay)
    elif model in ["WeightedCascade", "D4"]:
        return simulate_weighted_cascade(G, M, T, c_T=c_T, c_M=c_M, tie_rule=tie_rule, delay=delay)
    else:
        return simulate_icm(G, M, T, pM=pM, pT=pT, tie_rule=tie_rule, delay=delay)


def estimate_sp(G, M, T, model="ICM", pM=0.1, pT=0.1, gamma_T=1.0, c_T=1.0, c_M=1.0, tie_rule="truth", delay=0, R=100):
    """
    Estimates sp(v|T) = 1 - Pr[v ends up infected by misinformation] over R runs.
    Returns:
        sp_dict: {v: sp_value}
        avg_time_dict: {v: mean_activation_step}
        history_logs: list of per-run time steps for fine-grained analysis
    """
    count_infected = {v: 0 for v in G.nodes()}
    
    for _ in range(R):
        state, _ = simulate_cascade(
            G, M, T, model=model, pM=pM, pT=pT, 
            gamma_T=gamma_T, c_T=c_T, c_M=c_M, 
            tie_rule=tie_rule, delay=delay
        )
        for v, s in state.items():
            if s == -1:
                count_infected[v] += 1
                
    sp = {v: 1.0 - (count_infected[v] / float(R)) for v in G.nodes()}
    return sp
