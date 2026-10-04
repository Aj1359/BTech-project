import networkx as nx
import numpy as np
import random
import os
import importlib.util
from collections import defaultdict
import matplotlib.pyplot as plt

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

def load_real_network(filepath):
    G = nx.read_edgelist(filepath, create_using=nx.Graph())
    G = nx.convert_node_labels_to_integers(G)
    return G

# ==========================================
# Main Evaluation Loop
# ==========================================

def run_experiment():
    print("Loading algorithms...")
    algos_dir = os.path.join(os.path.dirname(__file__), 'algos')
    algorithms = load_algorithms(algos_dir)
    print(f"Loaded algorithms: {list(algorithms.keys())}")
    
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    networks_to_test = {}
    
    # Load synthetic network
    networks_to_test['Synthetic_BA_200'] = nx.barabasi_albert_graph(200, 3)
    
    # Load real networks (uncomment if you want to run on them, but note they take longer!)
    # nets_dir = os.path.join(os.path.dirname(__file__), 'networks')
    # if os.path.exists(nets_dir):
    #     for file in os.listdir(nets_dir):
    #         if file.endswith('.txt'):
    #             print(f"Loading {file}...")
    #             networks_to_test[file] = load_real_network(os.path.join(nets_dir, file))
    
    # Sweep parameters
    coicm_alphas = [0.1, 0.2, 0.3]
    mcicm_combinations = [(0.1, 0.2), (0.1, 0.3), (0.2, 0.1), (0.3, 0.1), (0.2, 0.3)]
    k_values = [2, 5, 10]
    time_units = [3, 5, 10]
    
    experiments = [("COICM", a, a) for a in coicm_alphas] + [("MCICM", pm, pt) for pm, pt in mcicm_combinations]
    
    for net_name, G in networks_to_test.items():
        print(f"\nProcessing network: {net_name} ({G.number_of_nodes()} nodes)")
        
        # Assign random groups
        groups = [0, 1]
        group_assignments = np.random.choice(groups, size=G.number_of_nodes(), p=[0.8, 0.2])
        for i, node in enumerate(G.nodes()):
            G.nodes[node]['group'] = group_assignments[i]
            
        M = set(random.sample(list(G.nodes()), max(1, int(G.number_of_nodes() * 0.05))))
        D, D_M, vulnerable = preprocess(G, M)
        
        # Track results for plotting
        res_vs_k = defaultdict(lambda: defaultdict(dict))
        res_vs_t = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

        for model_name, pM, pT in experiments:
            print(f"\n=== {model_name} (pM={pM}, pT={pT}) ===")
            for k in k_values:
                print(f"  -- Budget k={k} --")
                for algo_name, algo_func in algorithms.items():
                    print(f"    -> Running {algo_name}")
                    try:
                        T = algo_func(G, M, k, 
                                      pM=pM, pT=pT, R=200, epsilon=0.05, lam=0.5,
                                      estimate_sp=estimate_sp, D=D, D_M=D_M, 
                                      vulnerable_set=vulnerable, groups=groups)
                        
                        total_saved = 0
                        total_spread = {t: 0 for t in time_units}
                        
                        for _ in range(200):
                            state, time = simulate_icm(G, M, T, pM=pM, pT=pT)
                            total_saved += sum(1 for v in vulnerable if state[v] != -1)
                            
                            for t_max in time_units:
                                total_spread[t_max] += sum(1 for v in G.nodes() if time[v] <= t_max and state[v] == 1)
                                
                        avg_saved = total_saved / 200.0
                        res_vs_k[algo_name][(pM, pT)][k] = avg_saved
                        
                        for t_max in time_units:
                            res_vs_t[algo_name][(pM, pT)][k][t_max] = total_spread[t_max] / 200.0
                            
                    except Exception as e:
                        print(f"Error running {algo_name}: {e}")
                        
        # ---------------- Plotting ---------------- #
        print(f"\nGenerating plots for {net_name} in {results_dir}...")
        for model_name, pM, pT in experiments:
            
            # 1. Graph across seed nodes (k)
            plt.figure(figsize=(10, 6))
            for algo_name in algorithms.keys():
                if (pM, pT) in res_vs_k[algo_name]:
                    k_list = sorted(res_vs_k[algo_name][(pM, pT)].keys())
                    saved_list = [res_vs_k[algo_name][(pM, pT)][k] for k in k_list]
                    plt.plot(k_list, saved_list, marker='o', label=algo_name)
            plt.xlabel('Budget (k)')
            plt.ylabel('Vulnerable Nodes Saved')
            plt.title(f'{net_name} - {model_name} (pM={pM}, pT={pT})')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(results_dir, f'{net_name}_{model_name}_pM{pM}_pT{pT}_vs_k.png'))
            plt.close()
            
            # 2. Graph across time (plotted for k=5)
            target_k = 5
            plt.figure(figsize=(10, 6))
            for algo_name in algorithms.keys():
                if (pM, pT) in res_vs_t[algo_name] and target_k in res_vs_t[algo_name][(pM, pT)]:
                    t_list = sorted(res_vs_t[algo_name][(pM, pT)][target_k].keys())
                    reached_list = [res_vs_t[algo_name][(pM, pT)][target_k][t] for t in t_list]
                    plt.plot(t_list, reached_list, marker='s', label=algo_name)
            plt.xlabel('Time (t)')
            plt.ylabel('Truth Reached Nodes')
            plt.title(f'{net_name} - {model_name} (pM={pM}, pT={pT}, k={target_k})')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(results_dir, f'{net_name}_{model_name}_pM{pM}_pT{pT}_k{target_k}_vs_time.png'))
            plt.close()

if __name__ == "__main__":
    run_experiment()
