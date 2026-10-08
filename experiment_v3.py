import os
import json
import math
import random
import importlib.util
from collections import defaultdict
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

from simulators import estimate_sp, simulate_cascade
from placem import place_m
from metrics import compute_all_metrics, inequality_aversion_welfare, demographic_parity_gap

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
    G = nx.read_edgelist(filepath, create_using=nx.Graph(), data=False)
    G = nx.convert_node_labels_to_integers(G)
    return G

def generate_homophily_ba(n=500, m_attach=3, p_group0=0.8, h=0.5, seed=42):
    """
    Generates synthetic BA graph with group homophily h.
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
# Statistical & Verification Subroutines
# ==========================================

def compute_beta(metrics_over_k):
    """
    Computes slope beta of maximin sp against budget k using least squares.
    """
    ks = sorted(metrics_over_k.keys())
    if len(ks) < 2:
        return 0.0
    ys = [metrics_over_k[k] for k in ks]
    beta, _ = np.polyfit(ks, ys, 1)
    return float(beta)

def classify_method(beta_method, beta_ref, se=0.01):
    if beta_method > beta_ref + se:
        return "better"
    elif abs(beta_method - beta_ref) <= se:
        return "equivalent"
    elif beta_method >= 0.8 * beta_ref:
        return "within 80%"
    else:
        return "worse"

def permutation_test_group_gap(G, M, T, model="ICM", pM=0.1, pT=0.1, n_perm=500):
    """
    Permutation test: reshuffle group labels keeping sizes to test if group gap exceeds chance.
    """
    eval_orig = compute_all_metrics(G, M, T, model=model, pM=pM, pT=pT, R_eval=100)
    obs_gap = eval_orig["Group_Gap"]
    
    groups_dict = nx.get_node_attributes(G, 'group')
    nodes = list(groups_dict.keys())
    labels = list(groups_dict.values())
    
    count_exceed = 0
    for _ in range(n_perm):
        shuffled_labels = random.sample(labels, len(labels))
        G_shuffled = G.copy()
        for i, n in enumerate(nodes):
            G_shuffled.nodes[n]['group'] = shuffled_labels[i]
            
        eval_perm = compute_all_metrics(G_shuffled, M, T, model=model, pM=pM, pT=pT, R_eval=50)
        if eval_perm["Group_Gap"] >= obs_gap:
            count_exceed += 1
            
    p_value = count_exceed / float(n_perm)
    return obs_gap, p_value

# ==========================================
# Main Experiment V3 Suite
# ==========================================

def run_experiment_v3():
    print("==================================================")
    print("Starting Experiment V3 Execution Suite")
    print("==================================================")
    
    base_dir = os.path.dirname(__file__)
    algos_dir = os.path.join(base_dir, 'algos')
    algorithms = load_algorithms(algos_dir)
    print(f"Loaded {len(algorithms)} algorithms: {list(algorithms.keys())}")
    
    results_dir = os.path.join(base_dir, 'results_v3')
    os.makedirs(results_dir, exist_ok=True)
    
    networks_to_test = {}
    nets_dir = os.path.join(base_dir, 'networks')
    if os.path.exists(nets_dir):
        for file in os.listdir(nets_dir):
            if file.endswith('.txt'):
                net_name = file.replace('.txt', '')
                print(f"Loading real network {net_name}...")
                networks_to_test[net_name] = load_real_network(os.path.join(nets_dir, file))
                
    if not networks_to_test:
        print("Generating synthetic Homophily-BA network (n=300)...")
        networks_to_test['Homophily_BA_300'] = generate_homophily_ba(n=300, h=0.6, seed=42)
        
    k_values = [5, 10, 15, 20, 25, 30]
    m_placements = ["Highest_Degree", "Random", "Majority", "Minority", "Split"]
    
    # Grid Configurations
    core_models = [
        ("D1_COICM", {"model": "ICM", "pM": 0.1, "pT": 0.1, "tie_rule": "truth", "delay": 0}),
        ("D2_MCICM_HighP", {"model": "ICM", "pM": 0.1, "pT": 0.5, "tie_rule": "truth", "delay": 0}),
    ]
    
    extended_models = [
        ("D3_CLT_g1.0", {"model": "CLT", "gamma_T": 1.0, "tie_rule": "truth", "delay": 0}),
        ("D3_CLT_g1.5", {"model": "CLT", "gamma_T": 1.5, "tie_rule": "truth", "delay": 0}),
        ("D4_WeightedCascade", {"model": "WeightedCascade", "c_T": 1.0, "c_M": 1.0, "tie_rule": "truth", "delay": 0})
    ]
    
    # ---------------------------------------------------------
    # E1 & E2 & E3: Core Grid Evaluation & Structural Bias & Beta
    # ---------------------------------------------------------
    print("\n--- Running Experiments E1, E2, E3 (Core Grid, Maximin Welfare & Beta Slope) ---")
    
    pass_fail_summary = {}
    
    for net_name, G in networks_to_test.items():
        groups_dict = nx.get_node_attributes(G, 'group')
        if not groups_dict:
            np_rng = np.random.RandomState(42)
            assigns = np_rng.choice([0, 1], size=G.number_of_nodes(), p=[0.8, 0.2])
            for i, v in enumerate(G.nodes()):
                G.nodes[v]['group'] = int(assigns[i])
            groups_dict = nx.get_node_attributes(G, 'group')
            
        unique_groups = list(set(groups_dict.values()))
        
        for place_mode in m_placements[:2]:  # Core evaluation on Highest_Degree & Random
            M, vulnerable, sp0_base = place_m(G, mode=place_mode, seed=42)
            D, D_M, vulnerable = preprocess(G, M)
            num_m_nodes = len(M)
            
            for model_name, params in core_models:
                exp_path = os.path.join(results_dir, net_name, model_name, f"M_{num_m_nodes}_{place_mode}")
                os.makedirs(exp_path, exist_ok=True)
                
                print(f"  Network: {net_name} | Model: {model_name} | M Placement: {place_mode}")
                metrics_vs_k = defaultdict(lambda: defaultdict(dict))
                
                for k in k_values:
                    k_path = os.path.join(exp_path, f"k_{k}")
                    os.makedirs(k_path, exist_ok=True)
                    k_summary = {}
                    
                    for algo_name, algo_func in algorithms.items():
                        try:
                            # 1. Selection Run
                            T = algo_func(
                                G, M, k,
                                pM=params.get("pM", 0.1), pT=params.get("pT", 0.1),
                                gamma_T=params.get("gamma_T", 1.0), model=params.get("model", "ICM"),
                                tie_rule=params.get("tie_rule", "truth"), delay=params.get("delay", 0),
                                R=100, epsilon=0.05, lam=0.5,
                                estimate_sp=estimate_sp, D=D, D_M=D_M,
                                vulnerable_set=vulnerable, groups=unique_groups
                            )
                            T = set(T) - set(M)
                            if len(T) > k:
                                T = set(list(T)[:k])
                                
                            # 2. FRESH Independent Evaluation Run (R_eval = 200)
                            eval_res = compute_all_metrics(
                                G, M, T, model=params.get("model", "ICM"),
                                pM=params.get("pM", 0.1), pT=params.get("pT", 0.1),
                                gamma_T=params.get("gamma_T", 1.0),
                                tie_rule=params.get("tie_rule", "truth"), delay=params.get("delay", 0),
                                R_eval=200, D_M=D_M
                            )
                            k_summary[algo_name] = eval_res
                            
                            for m_key, val in eval_res.items():
                                if isinstance(val, (int, float)):
                                    metrics_vs_k[m_key][algo_name][k] = val
                                    
                        except Exception as e:
                            print(f"    [Error] {algo_name} at k={k}: {e}")
                            
                    with open(os.path.join(k_path, 'metrics_v3.json'), 'w') as f:
                        json.dump(k_summary, f, indent=4)
                        
                # Plot Core Metrics vs Budget k
                for m_key in ["Saved_Fraction", "Maximin_SP", "Group_Gap", "Group_Ratio", "Consistency", "Gini"]:
                    if m_key in metrics_vs_k:
                        plt.figure(figsize=(9, 5))
                        for algo_name, k_dict in metrics_vs_k[m_key].items():
                            sorted_k = sorted(k_dict.keys())
                            vals = [k_dict[k] for k in sorted_k]
                            plt.plot(sorted_k, vals, marker='o', label=algo_name)
                        plt.xlabel("Budget (k)")
                        plt.ylabel(m_key.replace('_', ' '))
                        plt.title(f"E1: {m_key} vs Budget (k) | {net_name} | {model_name}")
                        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                        plt.grid(True)
                        plt.tight_layout()
                        plt.savefig(os.path.join(exp_path, f"{m_key}_vs_k.png"))
                        plt.close()
                        
                # E3 Beta Slopes & Classification
                beta_scores = {}
                for algo_name, k_dict in metrics_vs_k["Maximin_SP"].items():
                    beta_scores[algo_name] = compute_beta(k_dict)
                    
                ref_beta = beta_scores.get("protect", 0.01)
                classifications = {
                    algo: classify_method(beta, ref_beta)
                    for algo, beta in beta_scores.items()
                }
                
                with open(os.path.join(exp_path, 'beta_classifications.json'), 'w') as f:
                    json.dump({"beta": beta_scores, "classification": classifications}, f, indent=4)

    # ---------------------------------------------------------
    # E4: Homophily Sweep
    # ---------------------------------------------------------
    print("\n--- Running Experiment E4 (Homophily Sweep h = 0.2 ... 0.8) ---")
    h_values = [0.2, 0.4, 0.6, 0.8]
    e4_path = os.path.join(results_dir, "E4_Homophily_Sweep")
    os.makedirs(e4_path, exist_ok=True)
    
    e4_results = defaultdict(lambda: defaultdict(dict))
    
    for h in h_values:
        G_h = generate_homophily_ba(n=250, h=h, seed=int(h * 100))
        groups_h = list(set(nx.get_node_attributes(G_h, 'group').values()))
        M_h, vuln_h, _ = place_m(G_h, mode="highest_degree", seed=42)
        D_h, DM_h, vuln_h = preprocess(G_h, M_h)
        
        for algo_name in ["protect", "group_aware_protect", "top_degree", "fwrrs", "dp_aware_fibm"]:
            if algo_name in algorithms:
                try:
                    T = algorithms[algo_name](
                        G_h, M_h, 15, pM=0.1, pT=0.1, model="ICM",
                        R=50, D=D_h, D_M=DM_h, vulnerable_set=vuln_h, groups=groups_h
                    )
                    eval_h = compute_all_metrics(G_h, M_h, T, model="ICM", pM=0.1, pT=0.1, R_eval=100, D_M=DM_h)
                    e4_results["Group_Gap"][algo_name][h] = eval_h["Group_Gap"]
                    e4_results["Group_Ratio"][algo_name][h] = eval_h["Group_Ratio"]
                except Exception as e:
                    print(f"    [E4 Error] {algo_name} at h={h}: {e}")
                    
    with open(os.path.join(e4_path, 'e4_metrics.json'), 'w') as f:
        json.dump(e4_results, f, indent=4)

    # ---------------------------------------------------------
    # E10, E11: Extended Models, Tie Rules & Detection Delay
    # ---------------------------------------------------------
    print("\n--- Running Experiments E10 & E11 (CLT, Tie Rules, Detection Delay) ---")
    e11_path = os.path.join(results_dir, "E11_Detection_Delay")
    os.makedirs(e11_path, exist_ok=True)
    
    delays = [0, 1, 2, 3]
    e11_results = defaultdict(lambda: defaultdict(dict))
    
    G_ext = list(networks_to_test.values())[0]
    groups_ext = list(set(nx.get_node_attributes(G_ext, 'group').values()))
    M_ext, vuln_ext, _ = place_m(G_ext, mode="highest_degree", seed=42)
    D_ext, DM_ext, vuln_ext = preprocess(G_ext, M_ext)
    
    for delay_r in delays:
        for algo_name in ["protect", "group_aware_protect", "top_degree"]:
            if algo_name in algorithms:
                try:
                    T = algorithms[algo_name](
                        G_ext, M_ext, 15, pM=0.1, pT=0.1, delay=delay_r,
                        R=50, D=D_ext, D_M=DM_ext, vulnerable_set=vuln_ext, groups=groups_ext
                    )
                    eval_d = compute_all_metrics(
                        G_ext, M_ext, T, model="ICM", pM=0.1, pT=0.1, delay=delay_r, R_eval=100, D_M=DM_ext
                    )
                    e11_results["Maximin_SP"][algo_name][delay_r] = eval_d["Maximin_SP"]
                    e11_results["Group_Gap"][algo_name][delay_r] = eval_d["Group_Gap"]
                except Exception as e:
                    print(f"    [E11 Error] {algo_name} at delay={delay_r}: {e}")
                    
    with open(os.path.join(e11_path, 'e11_metrics.json'), 'w') as f:
        json.dump(e11_results, f, indent=4)

    # ---------------------------------------------------------
    # E14: Comparison with Fair Truth Campaigning Prior Work (FWRRS, DP-aware)
    # ---------------------------------------------------------
    print("\n--- Running Experiment E14 (Head-to-head Prior Work Comparison) ---")
    e14_path = os.path.join(results_dir, "E14_Prior_Work_Comparison")
    os.makedirs(e14_path, exist_ok=True)
    
    e14_summary = {}
    prior_work_methods = ["protect", "group_aware_protect", "fwrrs", "dp_aware_fibm", "top_degree", "random_baseline"]
    
    for algo_name in prior_work_methods:
        if algo_name in algorithms:
            try:
                T = algorithms[algo_name](
                    G_ext, M_ext, 15, pM=0.1, pT=0.1, model="ICM",
                    R=50, D=D_ext, D_M=DM_ext, vulnerable_set=vuln_ext, groups=groups_ext
                )
                eval_e14 = compute_all_metrics(G_ext, M_ext, T, model="ICM", pM=0.1, pT=0.1, R_eval=200, D_M=DM_ext)
                obs_gap, p_val = permutation_test_group_gap(G_ext, M_ext, T, model="ICM", pM=0.1, pT=0.1, n_perm=100)
                eval_e14["Permutation_P_Value"] = p_val
                e14_summary[algo_name] = eval_e14
            except Exception as e:
                print(f"    [E14 Error] {algo_name}: {e}")
                
    with open(os.path.join(e14_path, 'e14_head_to_head.json'), 'w') as f:
        json.dump(e14_summary, f, indent=4)

    print("\n==================================================")
    print("Experiment V3 Suite execution finished cleanly!")
    print(f"All structured deliverables saved under: {results_dir}")
    print("==================================================")

if __name__ == "__main__":
    run_experiment_v3()
