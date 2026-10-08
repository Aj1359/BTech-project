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

# ==========================================
# Helpers & Setup
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


def generate_homophily_ba(n=500, m_attach=3, p_group0=0.8, h=0.5, seed=42):
    """
    Generates synthetic BA graph with group homophily.
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    G = nx.Graph()
    groups = np_rng.choice([0, 1], size=n, p=[p_group0, 1.0 - p_group0])
    
    for i in range(n):
        G.add_node(i, group=int(groups[i]))
        
    for i in range(1, n):
        possible_targets = list(range(i))
        weights = []
        for t in possible_targets:
            same_group = (groups[i] == groups[t])
            pref = h if same_group else (1.0 - h)
            deg = G.degree(t) + 1
            weights.append(pref * deg)
            
        sum_w = sum(weights)
        if sum_w > 0:
            probs = [w / sum_w for w in weights]
            targets = np_rng.choice(possible_targets, size=min(m_attach, i), replace=False, p=probs)
            for t in targets:
                G.add_edge(i, int(t))
        else:
            targets = rng.sample(possible_targets, min(m_attach, i))
            for t in targets:
                G.add_edge(i, t)
                
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

# ==========================================
# Main Experiment Runner V2
# ==========================================

def run_experiment_v2():
    print("==================================================")
    print("Starting Experiment V2 Framework Execution")
    print("==================================================")
    
    base_dir = os.path.dirname(__file__)
    algos_dir = os.path.join(base_dir, 'algos')
    algorithms = load_algorithms(algos_dir)
    print(f"Loaded {len(algorithms)} algorithms: {list(algorithms.keys())}")
    
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    networks_to_test = {}
    nets_dir = os.path.join(base_dir, 'networks')
    if os.path.exists(nets_dir):
        for file in os.listdir(nets_dir):
            if file.endswith('.txt'):
                net_name = file.replace('.txt', '')
                print(f"Loading network {net_name}...")
                networks_to_test[net_name] = load_real_network(os.path.join(nets_dir, file))
                
    # Add synthetic network if none loaded
    if not networks_to_test:
        print("Generating synthetic Homophily-BA network (n=300)...")
        networks_to_test['Homophily_BA_300'] = generate_homophily_ba(n=300, h=0.6, seed=42)
        
    # Core Grid Factor Configuration
    k_values = [5, 10, 15, 20, 25]
    m_placements = ["Highest_Degree", "Random", "Majority", "Minority", "Split"]
    
    # Models to test (Tier 1: Core COICM/MCICM, Tier 2: Extended CLT)
    model_configs = [
        ("COICM", {"pM": 0.1, "pT": 0.1, "tie_rule": "truth", "delay": 0}),
        ("MCICM_HighP", {"pM": 0.1, "pT": 0.3, "tie_rule": "truth", "delay": 0}),
        ("CLT_Default", {"model": "CLT", "gamma_T": 1.0, "tie_rule": "truth", "delay": 0})
    ]
    
    for net_name, G in networks_to_test.items():
        print(f"\n>>> Network: {net_name} (|V|={G.number_of_nodes()}, |E|={G.number_of_edges()})")
        
        # Ensure groups assigned
        groups_dict = nx.get_node_attributes(G, 'group')
        if not groups_dict:
            np_rng = np.random.RandomState(42)
            assigns = np_rng.choice([0, 1], size=G.number_of_nodes(), p=[0.8, 0.2])
            for i, v in enumerate(G.nodes()):
                G.nodes[v]['group'] = int(assigns[i])
            groups_dict = nx.get_node_attributes(G, 'group')
            
        unique_groups = list(set(groups_dict.values()))
        
        for place_mode in m_placements:
            print(f"  > Placement Mode: {place_mode}")
            M, vulnerable, sp0_base = place_m(G, mode=place_mode, seed=42)
            D, D_M, vulnerable = preprocess(G, M)
            num_m_nodes = len(M)
            
            for model_name, params in model_configs:
                model_type = params.get("model", "ICM")
                pM = params.get("pM", 0.1)
                pT = params.get("pT", 0.1)
                gamma_T = params.get("gamma_T", 1.0)
                tie_rule = params.get("tie_rule", "truth")
                delay = params.get("delay", 0)
                
                exp_path = os.path.join(
                    results_dir, net_name, model_name,
                    f"M_{num_m_nodes}_{place_mode}"
                )
                os.makedirs(exp_path, exist_ok=True)
                
                print(f"    Running Model: {model_name} | M size={num_m_nodes} | |Vulnerable|={len(vulnerable)}")
                
                metrics_vs_k = defaultdict(lambda: defaultdict(dict))
                
                for k in k_values:
                    k_path = os.path.join(exp_path, f"k_{k}")
                    os.makedirs(k_path, exist_ok=True)
                    
                    k_eval_summary = {}
                    
                    for algo_name, algo_func in algorithms.items():
                        try:
                            # 1. Selection Run
                            T = algo_func(
                                G, M, k,
                                pM=pM, pT=pT, gamma_T=gamma_T,
                                model=model_type, tie_rule=tie_rule, delay=delay,
                                R=100, epsilon=0.05, lam=0.5,
                                estimate_sp=estimate_sp, D=D, D_M=D_M,
                                vulnerable_set=vulnerable, groups=unique_groups
                            )
                            
                            # Ensure T excludes M and size <= k
                            T = set(T) - set(M)
                            if len(T) > k:
                                T = set(list(T)[:k])
                                
                            # 2. Independent Evaluation Run (R_eval = 200 for fast validation)
                            eval_metrics = compute_all_metrics(
                                G, M, T, model=model_type,
                                pM=pM, pT=pT, gamma_T=gamma_T,
                                tie_rule=tie_rule, delay=delay,
                                R_eval=200, D_M=D_M
                            )
                            
                            k_eval_summary[algo_name] = eval_metrics
                            
                            for metric_key, val in eval_metrics.items():
                                if isinstance(val, (int, float)):
                                    metrics_vs_k[metric_key][algo_name][k] = val
                                    
                        except Exception as e:
                            print(f"      [Error] {algo_name} at k={k}: {e}")
                            
                    # Save metric outputs for this budget k
                    with open(os.path.join(k_path, 'metrics_v2.json'), 'w') as f:
                        json.dump(k_eval_summary, f, indent=4)
                        
                # Generate plots for core metrics vs k
                for metric_key in ["Saved_Fraction", "Maximin_SP", "Group_Gap", "Group_Ratio", "Consistency", "Gini"]:
                    if metric_key in metrics_vs_k:
                        plt.figure(figsize=(9, 5))
                        for algo_name, k_dict in metrics_vs_k[metric_key].items():
                            sorted_k = sorted(k_dict.keys())
                            vals = [k_dict[k] for k in sorted_k]
                            plt.plot(sorted_k, vals, marker='o', label=algo_name)
                        plt.xlabel("Budget (k)")
                        plt.ylabel(metric_key.replace('_', ' '))
                        plt.title(f"{metric_key} vs Budget (k) | {net_name} | {model_name}")
                        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                        plt.grid(True)
                        plt.tight_layout()
                        plt.savefig(os.path.join(exp_path, f"{metric_key}_vs_k.png"))
                        plt.close()
                        
    print("\n==================================================")
    print("Experiment V2 completed successfully!")
    print(f"Results stored in: {results_dir}")
    print("==================================================")

if __name__ == "__main__":
    run_experiment_v2()
