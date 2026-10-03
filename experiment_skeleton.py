import networkx as nx
import numpy as np
import random
import os
import importlib.util
from collections import defaultdict

# ==========================================
# Shared Subroutines
# ==========================================

def simulate_icm(G, M, T, pM=0.1, pT=0.1):
    """
    Independent Cascade Model.
    Truth wins ties. Truth acts first.
    States: 0 (inactive), -1 (negative), 1 (positive)
    """
    state = {v: 0 for v in G.nodes()}
    time = {v: float('inf') for v in G.nodes()}
    
    for t in T:
        state[t] = 1
        time[t] = 0
        
    for m in M:
        if state[m] == 0:
            state[m] = -1
            time[m] = 0
            
    frontier_pos = set(T)
    frontier_neg = set(M) - set(T)
    step = 0
    
    while frontier_pos or frontier_neg:
        step += 1
        new_pos = set()
        new_neg = set()
        
        # Truth acts first
        for u in frontier_pos:
            for v in G.neighbors(u):
                if state[v] == 0:
                    if random.random() < pT:
                        new_pos.add(v)
                        
        # Misinformation acts
        for u in frontier_neg:
            for v in G.neighbors(u):
                if state[v] == 0 and v not in new_pos:
                    if random.random() < pM:
                        new_neg.add(v)
                        
        for v in new_pos:
            state[v] = 1
            time[v] = step
        for v in new_neg:
            state[v] = -1
            time[v] = step
            
        frontier_pos = new_pos
        frontier_neg = new_neg
        
    return state, time

def estimate_sp(G, M, T, pM, pT, R=100):
    count = {v: 0 for v in G.nodes()}
    for _ in range(R):
        state, _ = simulate_icm(G, M, T, pM, pT)
        for v, s in state.items():
            if s == -1:
                count[v] += 1
    sp = {v: 1.0 - (count[v] / R) for v in G.nodes()}
    return sp

def preprocess(G, M):
    G_prime = G.copy()
    if G_prime.is_directed():
        for m in M:
            in_edges = list(G_prime.in_edges(m))
            G_prime.remove_edges_from(in_edges)
            
    D_M = {v: float('inf') for v in G.nodes()}
    for m in M:
        lengths = nx.single_source_shortest_path_length(G, m)
        for v, dist in lengths.items():
            if dist < D_M[v]:
                D_M[v] = dist
                
    vulnerable = {v for v, d in D_M.items() if d < float('inf')} - set(M)
    
    try:
        D = dict(nx.all_pairs_shortest_path_length(G))
    except Exception:
        D = {}
        
    return D, D_M, vulnerable

# ==========================================
# Dynamic Algorithm Loader
# ==========================================

def load_algorithms(algos_dir='algos'):
    algorithms = {}
    if not os.path.exists(algos_dir):
        return algorithms
        
    for filename in os.listdir(algos_dir):
        if filename.endswith('.py') and not filename.startswith('__'):
            name = filename[:-3]
            filepath = os.path.join(algos_dir, filename)
            spec = importlib.util.spec_from_file_location(name, filepath)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if hasattr(module, 'get_seeds'):
                algorithms[name] = module.get_seeds
    return algorithms

# ==========================================
# Main Evaluation Loop
# ==========================================

def run_experiment():
    print("Loading algorithms...")
    algos_dir = os.path.join(os.path.dirname(__file__), 'algos')
    algorithms = load_algorithms(algos_dir)
    print(f"Loaded algorithms: {list(algorithms.keys())}")
    
    # 1. Generate Graph
    N = 100
    M_edges = 3
    G = nx.barabasi_albert_graph(N, M_edges)
    
    groups = [0, 1]
    group_ratios = [0.8, 0.2]
    group_assignments = np.random.choice(groups, size=N, p=group_ratios)
    for i, node in enumerate(G.nodes()):
        G.nodes[node]['group'] = group_assignments[i]
        
    M = set(random.sample(list(G.nodes()), int(N * 0.05)))
    D, D_M, vulnerable = preprocess(G, M)
    
    # Sweep parameters
    pM_values = [0.1, 0.3]
    pT_values = [0.1, 0.3]
    k_values = [2, 5]
    time_units = [3, 5, 10]
    
    for pM in pM_values:
        for pT in pT_values:
            model_name = "COICM" if pM == pT else "MCICM"
            print(f"\n=== {model_name} (pM={pM}, pT={pT}) ===")
            for k in k_values:
                print(f"  -- Budget k={k} --")
                for algo_name, algo_func in algorithms.items():
                    print(f"    -> Running {algo_name}")
                    T = algo_func(G, M, k, 
                                  pM=pM, pT=pT, R=10, epsilon=0.05, lam=0.5,
                                  estimate_sp=estimate_sp, D=D, D_M=D_M, 
                                  vulnerable_set=vulnerable, groups=groups)
                    
                    state, time = simulate_icm(G, M, T, pM=pM, pT=pT)
                    saved = sum(1 for v in vulnerable if state[v] != -1)
                    print(f"       Saved {saved}/{len(vulnerable)} vulnerable nodes.")
                    
                    for t_max in time_units:
                        spread_t = sum(1 for v in G.nodes() if time[v] <= t_max and state[v] == 1)
                        print(f"       Time <= {t_max}: Truth reached {spread_t} nodes.")

if __name__ == "__main__":
    run_experiment()
