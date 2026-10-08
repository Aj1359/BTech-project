import os
import json
import random
import importlib.util
from collections import defaultdict
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from simulators import estimate_sp, simulate_cascade
from placem import place_m
from metrics import compute_all_metrics

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

def load_real_network(filepath, is_directed=False):
    create_using = nx.DiGraph() if is_directed else nx.Graph()
    G = nx.read_edgelist(filepath, create_using=create_using, data=False)
    
    # Extract largest connected component / SCC
    if G.is_directed():
        components = list(nx.strongly_connected_components(G))
    else:
        components = list(nx.connected_components(G))
        
    if components:
        largest_comp = max(components, key=len)
        G = G.subgraph(largest_comp).copy()
        
    G = nx.convert_node_labels_to_integers(G)
    return G

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

def run_experiment_v1():
    print("==================================================")
    print("Starting Experiment V1 Execution Suite")
    print("==================================================")
    
    base_dir = os.path.dirname(__file__)
    algos_dir = os.path.join(base_dir, 'algos')
    algorithms = load_algorithms(algos_dir)
    print(f"Loaded {len(algorithms)} algorithms: {list(algorithms.keys())}")
    
    results_dir = os.path.join(base_dir, 'results_v1')
    os.makedirs(results_dir, exist_ok=True)
    
    networks_to_test = {}
    nets_dir = os.path.join(base_dir, 'networks')
    if os.path.exists(nets_dir):
        for file in os.listdir(nets_dir):
            if file.endswith('.txt'):
                net_name = file.replace('.txt', '')
                print(f"Loading {net_name}...")
                networks_to_test[net_name] = load_real_network(os.path.join(nets_dir, file))
                
    if not networks_to_test:
        print("No networks found in networks/ directory.")
        return
        
    coicm_alphas = [0.1, 0.3]
    mcicm_combs = [(0.1, 0.2), (0.1, 0.3), (0.2, 0.1), (0.3, 0.1)]
    experiments = [("COICM", a, a) for a in coicm_alphas] + [("MCICM", pm, pt) for pm, pt in mcicm_combs]
    
    k_values = [5, 10, 15, 20, 25]
    placements = ["Highest_Degree", "Random", "Majority", "Minority", "Split"]
    
    for net_name, G in networks_to_test.items():
        print(f"\nProcessing Network V1 (LCC/SCC): {net_name} (|V|={G.number_of_nodes()})")
        groups_dict = nx.get_node_attributes(G, 'group')
        if not groups_dict:
            np_rng = np.random.RandomState(42)
            assigns = np_rng.choice([0, 1], size=G.number_of_nodes(), p=[0.8, 0.2])
            for i, v in enumerate(G.nodes()):
                G.nodes[v]['group'] = int(assigns[i])
            groups_dict = nx.get_node_attributes(G, 'group')
            
        unique_groups = list(set(groups_dict.values()))
        
        for p_name in placements:
            M, vulnerable, sp0 = place_m(G, mode=p_name, seed=42)
            D, D_M, vulnerable = preprocess(G, M)
            num_m_nodes = len(M)
            
            for model_name, pM, pT in experiments:
                if model_name == "COICM":
                    base_exp_dir = os.path.join(results_dir, net_name, f"alpha_{pT}", f"M_{num_m_nodes}_{p_name}")
                else:
                    base_exp_dir = os.path.join(results_dir, net_name, f"pT_{pT}", f"pM_{pM}", f"M_{num_m_nodes}_{p_name}")
                    
                os.makedirs(base_exp_dir, exist_ok=True)
                
                metrics_res = defaultdict(lambda: defaultdict(dict))
                
                for k in k_values:
                    k_dir = os.path.join(base_exp_dir, f"k_{k}")
                    os.makedirs(k_dir, exist_ok=True)
                    k_metrics_out = {}
                    
                    for algo_name, algo_func in algorithms.items():
                        try:
                            # 1. Selection Run
                            T = algo_func(
                                G, M, k, pM=pM, pT=pT, R=100,
                                epsilon=0.05, lam=0.5, estimate_sp=estimate_sp,
                                D=D, D_M=D_M, vulnerable_set=vulnerable, groups=unique_groups
                            )
                            T = set(T) - set(M)
                            if len(T) > k:
                                T = set(list(T)[:k])
                                
                            # 2. FRESH Independent Evaluation Run with full v3 metrics suite
                            eval_metrics = compute_all_metrics(
                                G, M, T, model="ICM", pM=pM, pT=pT,
                                tie_rule="truth", delay=0, R_eval=200, D_M=D_M
                            )
                            
                            k_metrics_out[algo_name] = eval_metrics
                            
                            for m_key, val in eval_metrics.items():
                                if isinstance(val, (int, float)):
                                    metrics_res[m_key][algo_name][k] = val
                                    
                        except Exception as e:
                            print(f"Error {algo_name} at k={k}: {e}")
                            
                    with open(os.path.join(k_dir, 'metrics_v1_full.json'), 'w') as f:
                        json.dump(k_metrics_out, f, indent=4)
                        
                # Plot All Metrics vs k
                for metric_name in ["Saved_Fraction", "Maximin_SP", "Group_Gap", "Group_Ratio", "Consistency", "Gini", "Inequality_Aversion_Welfare", "DP_Gap", "Floor_Share"]:
                    if metric_name in metrics_res:
                        plt.figure(figsize=(9, 5))
                        for algo_name, k_data in metrics_res[metric_name].items():
                            k_list = sorted(k_data.keys())
                            val_list = [k_data[k] for k in k_list]
                            plt.plot(k_list, val_list, marker='o', label=algo_name)
                        plt.xlabel('Budget (k)')
                        plt.ylabel(metric_name.replace('_', ' '))
                        plt.title(f'V1 {metric_name} | {net_name} | M={p_name}')
                        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                        plt.grid(True)
                        plt.tight_layout()
                        plt.savefig(os.path.join(base_exp_dir, f'{metric_name}_vs_k.png'))
                        plt.close()
                    
    print("Experiment V1 preparation with full paper metrics finished!")

if __name__ == "__main__":
    run_experiment_v1()
