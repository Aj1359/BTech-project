import networkx as nx
import numpy as np
import random
import os
import json
import importlib.util
from collections import defaultdict
import matplotlib.pyplot as plt

# ==========================================
# Shared Subroutines
# ==========================================

def simulate_icm(G, M, T, pM=0.1, pT=0.1):
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
        for u in frontier_pos:
            for v in G.neighbors(u):
                if state[v] == 0:
                    if random.random() < pT:
                        new_pos.add(v)
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
# Metrics
# ==========================================

def get_gini_coefficient(sp_values):
    n = len(sp_values)
    if n == 0: return 0.0
    arr = np.array(sp_values)
    mean_val = np.mean(arr)
    if mean_val == 0: return 0.0
    diff_sum = np.sum(np.abs(np.subtract.outer(arr, arr)))
    return diff_sum / (2.0 * n * n * mean_val)

def get_consistency(sp_values_dict, sp0_values_dict):
    vuln = list(sp_values_dict.keys())
    n = len(vuln)
    if n <= 1: return 1.0
    
    num, den = 0.0, 0.0
    for i in range(n):
        vi = vuln[i]
        for j in range(n):
            vj = vuln[j]
            s_ij = 1.0 - abs(sp0_values_dict[vi] - sp0_values_dict[vj])
            diff = abs(sp_values_dict[vi] - sp_values_dict[vj])
            num += diff * s_ij
            den += s_ij
            
    if den == 0: return 1.0
    return 1.0 - (num / den)

def get_group_metrics(sp_values_dict, G, vulnerable):
    group_sps = defaultdict(list)
    for v in vulnerable:
        g = G.nodes[v].get('group', 0)
        group_sps[g].append(sp_values_dict[v])
        
    avg_sps = []
    for g, sps in group_sps.items():
        if sps:
            avg_sps.append(np.mean(sps))
            
    if len(avg_sps) < 2:
        return 0.0, 1.0
        
    max_sp = max(avg_sps)
    min_sp = min(avg_sps)
    group_gap = max_sp - min_sp
    group_ratio = min_sp / max_sp if max_sp > 0 else 1.0
    return group_gap, group_ratio

def load_real_network(filepath):
    G = nx.read_edgelist(filepath, create_using=nx.Graph(), data=False)
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
    
    networks_to_test = {}
    
    # Load real networks
    nets_dir = os.path.join(os.path.dirname(__file__), 'networks')
    if os.path.exists(nets_dir):
        for file in os.listdir(nets_dir):
            if file.endswith('.txt'):
                print(f"Loading {file}...")
                networks_to_test[file.replace('.txt', '')] = load_real_network(os.path.join(nets_dir, file))

    # Thesis settings
    coicm_alphas = [0.1, 0.3]
    mcicm_combs = [(0.1, 0.2), (0.1, 0.3), (0.2, 0.1), (0.3, 0.1), (0.2, 0.3)] # (pM, pT)
    experiments = [("COICM", a, a) for a in coicm_alphas] + [("MCICM", pm, pt) for pm, pt in mcicm_combs]
    
    k_values = [5, 10, 15, 20, 25]
    time_units = [3, 5, 10]
    
    for net_name, G in networks_to_test.items():
        print(f"\nProcessing network: {net_name} ({G.number_of_nodes()} nodes)")
        
        # Ensure groups exist
        groups = [0, 1]
        group_assignments = np.random.choice(groups, size=G.number_of_nodes(), p=[0.8, 0.2])
        for i, node in enumerate(G.nodes()):
            if 'group' not in G.nodes[node]:
                G.nodes[node]['group'] = group_assignments[i]
                
        group_0_nodes = [n for n in G.nodes() if G.nodes[n]['group'] == 0]
        group_1_nodes = [n for n in G.nodes() if G.nodes[n]['group'] == 1]
        majority = group_0_nodes if len(group_0_nodes) > len(group_1_nodes) else group_1_nodes
        minority = group_1_nodes if len(group_1_nodes) < len(group_0_nodes) else group_0_nodes
        
        m_size = max(1, int(G.number_of_nodes() * 0.05))
        
        # M-placements
        degrees = dict(G.degree())
        sorted_by_degree = sorted(degrees, key=degrees.get, reverse=True)
        m_highest_degree = set(sorted_by_degree[:m_size])
        m_random = set(random.sample(list(G.nodes()), m_size))
        m_majority = set(random.sample(majority, min(m_size, len(majority))))
        m_minority = set(random.sample(minority, min(m_size, len(minority))))
        
        half_m = m_size // 2
        m_split = set(random.sample(majority, min(half_m, len(majority)))) | set(random.sample(minority, min(m_size - half_m, len(minority))))
        
        placements = {
            'Highest_Degree': m_highest_degree, 
            'Random': m_random,
            'Majority': m_majority,
            'Minority': m_minority,
            'Split': m_split
        }
        
        for p_name, M in placements.items():
            D, D_M, vulnerable = preprocess(G, M)
            num_m_nodes = len(M)
            
            for model_name, pM, pT in experiments:
                
                # Base dir creation: network -> pT(alpha) -> pM(if MCICM) -> num_M_nodes
                if model_name == "COICM":
                    base_dir = os.path.join(results_dir, net_name, f"alpha_{pT}", f"M_{num_m_nodes}_{p_name}")
                else:
                    base_dir = os.path.join(results_dir, net_name, f"pT_{pT}", f"pM_{pM}", f"M_{num_m_nodes}_{p_name}")
                
                os.makedirs(base_dir, exist_ok=True)
                
                print(f"  Setup: {model_name} | pM={pM}, pT={pT} | M={p_name}")
                
                sp0 = estimate_sp(G, M, set(), pM, pT, R=100)
                sp0_vuln = {v: sp0[v] for v in vulnerable}
                expected_infected_empty = sum(1.0 - sp0[v] for v in vulnerable)
                
                # Track for plotting vs k at base level
                metrics_res = defaultdict(lambda: defaultdict(dict))
                
                for k in k_values:
                    # k directory
                    k_dir = os.path.join(base_dir, f"k_{k}")
                    os.makedirs(k_dir, exist_ok=True)
                    
                    # Track time metrics inside k
                    time_res = defaultdict(dict)
                    k_metrics_out = {}
                    
                    for algo_name, algo_func in algorithms.items():
                        try:
                            T = algo_func(G, M, k, 
                                          pM=pM, pT=pT, R=100, epsilon=0.05, lam=0.5,
                                          estimate_sp=estimate_sp, D=D, D_M=D_M, 
                                          vulnerable_set=vulnerable, groups=[0])
                                          
                            sp_T = estimate_sp(G, M, T, pM, pT, R=100)
                            sp_T_vuln = {v: sp_T[v] for v in vulnerable}
                            
                            expected_infected_T = sum(1.0 - sp_T[v] for v in vulnerable)
                            saved_frac = 1.0 - (expected_infected_T / expected_infected_empty) if expected_infected_empty > 0 else 1.0
                            maximin_sp = min(sp_T_vuln.values()) if sp_T_vuln else 1.0
                            consistency = get_consistency(sp_T_vuln, sp0_vuln)
                            gini = get_gini_coefficient(list(sp_T_vuln.values()))
                            
                            group_gap, group_ratio = get_group_metrics(sp_T_vuln, G, vulnerable)
                            
                            metrics_res['Saved_Fraction'][algo_name][k] = saved_frac
                            metrics_res['Maximin_SP'][algo_name][k] = maximin_sp
                            metrics_res['Consistency'][algo_name][k] = consistency
                            metrics_res['Gini'][algo_name][k] = gini
                            metrics_res['Group_Gap'][algo_name][k] = group_gap
                            metrics_res['Group_Ratio'][algo_name][k] = group_ratio
                            
                            k_metrics_out[algo_name] = {
                                "Saved_Fraction": saved_frac,
                                "Maximin_SP": maximin_sp,
                                "Consistency": consistency,
                                "Gini": gini,
                                "Group_Gap": group_gap,
                                "Group_Ratio": group_ratio
                            }
                            
                            # Time simulation (averaging over 50 runs for the curve)
                            total_spread = {t: 0 for t in time_units}
                            for _ in range(50):
                                state, time_arr = simulate_icm(G, M, T, pM=pM, pT=pT)
                                for t_max in time_units:
                                    total_spread[t_max] += sum(1 for v in G.nodes() if time_arr[v] <= t_max and state[v] == 1)
                                    
                            for t_max in time_units:
                                time_res[algo_name][t_max] = total_spread[t_max] / 50.0
                                
                        except Exception as e:
                            print(f"Error {algo_name}: {e}")
                            
                    # Save k-specific data
                    with open(os.path.join(k_dir, 'metrics.json'), 'w') as f:
                        json.dump(k_metrics_out, f, indent=4)
                        
                    # Plot Time Curve for this k
                    plt.figure(figsize=(8, 5))
                    for algo_name, t_data in time_res.items():
                        t_list = sorted(t_data.keys())
                        reached_list = [t_data[t] for t in t_list]
                        plt.plot(t_list, reached_list, marker='s', label=algo_name)
                    plt.xlabel('Time (t)')
                    plt.ylabel('Truth Reached Nodes')
                    plt.title(f'Time Spread | {model_name} pM={pM}, pT={pT} | k={k}')
                    plt.legend()
                    plt.grid(True)
                    plt.savefig(os.path.join(k_dir, 'spread_vs_time.png'))
                    plt.close()
                    
                # Plot Metrics vs k at base dir
                for metric_name, algos_data in metrics_res.items():
                    plt.figure(figsize=(10, 6))
                    for algo_name, k_data in algos_data.items():
                        k_list = sorted(k_data.keys())
                        val_list = [k_data[k] for k in k_list]
                        plt.plot(k_list, val_list, marker='o', label=algo_name)
                    plt.xlabel('Budget (k)')
                    plt.ylabel(metric_name.replace('_', ' '))
                    plt.title(f'{metric_name} | {net_name} | M={p_name} | {model_name}')
                    plt.legend()
                    plt.grid(True)
                    plt.savefig(os.path.join(base_dir, f'{metric_name}_vs_k.png'))
                    plt.close()
                    
    print("Experiment fully finished! Check the hierarchically structured results folder.")

if __name__ == "__main__":
    run_experiment()
