import os
import time
import json
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

from sim_engine import (
    GraphCSR, compute_distances_bfs,
    estimate_sp_engine, evaluate_metrics_engine
)
from algo_wrapper import AlgoManager, MODEL_AGNOSTIC_ALGOS, PREFIX_REUSABLE_ALGOS
from placem import place_m

# ==========================================
# Graph Processing (LCC/SCC)
# ==========================================

def load_and_extract_lcc(filepath, is_directed=False):
    """
    Loads network and strictly extracts the Largest Connected Component (LCC / LSCC).
    Removes all weak/disconnected components so propagation occurs exclusively on the main component.
    """
    create_using = nx.DiGraph() if is_directed else nx.Graph()
    G = nx.read_edgelist(filepath, create_using=create_using, data=False)

    if G.is_directed():
        components = list(nx.strongly_connected_components(G))
    else:
        components = list(nx.connected_components(G))

    if components:
        largest_comp = max(components, key=len)
        G = G.subgraph(largest_comp).copy()

    G = nx.convert_node_labels_to_integers(G)
    return G

def assign_communities(G, seed=42):
    """
    Assigns community group labels using Louvain modularity detection.
    """
    try:
        # pyrefly: ignore [missing-import]
        import community as community_louvain
        partition = community_louvain.best_partition(G, random_state=seed)
    except Exception:
        communities = list(nx.community.label_propagation_communities(G))
        partition = {}
        for g_id, comm in enumerate(communities):
            for node in comm:
                partition[node] = g_id

    nx.set_node_attributes(G, partition, 'group')
    return partition

# ==========================================
# Plotting Helpers
# ==========================================

def plot_metric_vs_k(k_values, metric_data, metric_name, title, output_path):
    """
    Generates a publication-quality line graph showing metric vs budget k across all algorithms.
    """
    plt.figure(figsize=(10, 6), dpi=150)
    markers = ['o', 's', '^', 'v', 'D', '<', '>', 'p', '*', 'h', 'H', 'd', 'P', 'X']

    algo_names = sorted(metric_data.keys())
    for idx, algo in enumerate(algo_names):
        vals = [metric_data[algo].get(k, 0.0) for k in k_values]
        marker = markers[idx % len(markers)]
        plt.plot(k_values, vals, marker=marker, linewidth=2, label=algo, markersize=6)

    plt.xlabel('Budget (k)', fontsize=12, fontweight='bold')
    ylabel = metric_name.replace('_', ' ').upper()
    plt.ylabel(ylabel, fontsize=12, fontweight='bold')
    plt.title(title, fontsize=13, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(k_values)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=9, frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

# ==========================================
# Main Experiment Pipeline
# ==========================================

def run_experiment_pipeline(
    results_dir="results",
    R_sim=1000,
    budgets=[5, 10, 15, 20, 25],
    m_size=10
):
    print("====================================================================")
    print("   Fair Truth-Spreader Placement Benchmark - Complete Runner       ")
    print(f"   Monte Carlo Runs: {R_sim} | Budgets k: {budgets} | M Size: {m_size}")
    print("====================================================================")

    os.makedirs(results_dir, exist_ok=True)
    algo_mgr = AlgoManager()
    print(f"Loaded {len(algo_mgr.loaded_funcs)} algorithms from algos/ directory.")

    # Datasets with calibrated/thesis propagation probabilities
    datasets = [
        {"name": "Facebook", "file": "networks/facebook_combined.txt", "alpha": 0.20, "run_mcicm": True},
        {"name": "Arenas_email", "file": "networks/email-Eu-core.txt", "alpha": 0.30, "run_mcicm": True},
        {"name": "CA-GrQc", "file": "networks/ca-GrQc.txt", "alpha": 0.15, "run_mcicm": False},
        {"name": "CA-HepTh", "file": "networks/ca-HepTh.txt", "alpha": 0.15, "run_mcicm": False},
        {"name": "CA-AstroPh", "file": "networks/ca-AstroPh.txt", "alpha": 0.25, "run_mcicm": False}
    ]

    # Diffusion models to execute: COICM and MCICM
    model_families = ["COICM", "MCICM"]

    for model_family in model_families:
        family_dir = os.path.join(results_dir, model_family)
        os.makedirs(family_dir, exist_ok=True)

        for ds in datasets:
            net_name = ds["name"]
            edge_file = ds["file"]

            if model_family == "MCICM" and not ds["run_mcicm"]:
                continue

            if not os.path.exists(edge_file):
                print(f"[Warning] Edge file not found: {edge_file}, skipping {net_name}...")
                continue

            print(f"\n>>> [{model_family}] Loading Network: {net_name} ({edge_file})")
            G = load_and_extract_lcc(edge_file)
            print(f"    Extracted LCC: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

            csr_graph = GraphCSR(G)
            groups = assign_communities(G, seed=42)
            unique_groups = sorted(list(set(groups.values())))

            # Network output folder inside model family
            net_dir = os.path.join(family_dir, net_name)
            os.makedirs(net_dir, exist_ok=True)

            # Determine probability parameters
            alpha = ds["alpha"]
            if model_family == "COICM":
                prob_folder_name = f"pT_{alpha}"
                pM_val, pT_val = alpha, alpha
            else:
                prob_folder_name = f"pM_{alpha}_pT_1.0"
                pM_val, pT_val = alpha, 1.0

            prob_dir = os.path.join(net_dir, prob_folder_name)
            os.makedirs(prob_dir, exist_ok=True)

            print(f"    Running: {prob_folder_name} (pM={pM_val}, pT={pT_val})")

            # Place M (highest degree starters within LCC)
            degrees = dict(G.degree())
            sorted_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)
            M_seeds = set(sorted_nodes[:m_size])

            D, D_M = compute_distances_bfs(G, M_seeds)

            # 1. Baseline Evaluation with T = empty
            sp0 = estimate_sp_engine(
                csr_graph, M_seeds, set(),
                model_type=model_family, pM=pM_val, pT=pT_val,
                R_runs=R_sim, base_seed=12345
            )
            vuln_nodes = [v for v in range(G.number_of_nodes()) if v not in M_seeds and sp0[v] < 1.0]
            if not vuln_nodes:
                vuln_nodes = [v for v in range(G.number_of_nodes()) if v not in M_seeds]

            baseline_outreach = float(np.mean([1.0 - sp0[v] for v in range(G.number_of_nodes())]))
            print(f"    Baseline T=empty: Vulnerable nodes = {len(vuln_nodes)}, Outreach = {baseline_outreach*100:.2f}%")

            # Metrics collectors for plotting vs k
            sp_vs_k = defaultdict(dict)
            saved_frac_vs_k = defaultdict(dict)

            max_k = max(budgets)
            prefix_cache = {}

            # Loop over budgets k
            for k in budgets:
                k_dir = os.path.join(prob_dir, f"k_{k}")
                os.makedirs(k_dir, exist_ok=True)

                k_seeds = {}
                k_metrics = {}

                print(f"      -> Evaluating Budget k = {k}...")

                for algo_id in algo_mgr.loaded_funcs.keys():
                    # Skip greedy maximin on large networks
                    if algo_id == 'greedy_maximin' and G.number_of_nodes() > 500:
                        continue

                    try:
                        start_time = time.time()

                        # Determine seed set T using prefix reuse if applicable
                        if algo_id in PREFIX_REUSABLE_ALGOS and algo_id in prefix_cache:
                            T = prefix_cache[algo_id][:k]
                        else:
                            run_k = max_k if algo_id in PREFIX_REUSABLE_ALGOS else k
                            T_full = algo_mgr.run_algo(
                                algo_id, G, M_seeds, run_k,
                                {'pM': pM_val, 'pT': pT_val, 'model': model_family},
                                R_select=R_sim, rng_seed=42 + k,
                                D=D, D_M=D_M, vulnerable_set=set(vuln_nodes),
                                groups=unique_groups, estimate_sp=estimate_sp_engine
                            )
                            if algo_id in PREFIX_REUSABLE_ALGOS:
                                prefix_cache[algo_id] = T_full
                                T = T_full[:k]
                            else:
                                T = T_full[:k]

                        sel_runtime = time.time() - start_time

                        # Independent Evaluation Run (R_sim runs)
                        eval_seed = (int(pM_val * 1000) + k * 101 + 54321) & 0xFFFFFFFF
                        metrics = evaluate_metrics_engine(
                            G, csr_graph, M_seeds, T, groups,
                            sp0, vuln_nodes,
                            model_type=model_family, pM=pM_val, pT=pT_val,
                            R_eval=R_sim, eval_seed=eval_seed
                        )
                        metrics["runtime_select_s"] = round(sel_runtime, 4)

                        k_seeds[algo_id] = [int(x) for x in T]
                        k_metrics[algo_id] = metrics

                        sp_vs_k[algo_id][k] = metrics["maximin_sp"]
                        saved_frac_vs_k[algo_id][k] = metrics["saved_fraction"]

                    except Exception as e:
                        print(f"         [Error] {algo_id} at k={k}: {e}")

                # Save seeds and metrics for this budget k
                with open(os.path.join(k_dir, "seeds.json"), "w") as f:
                    json.dump(k_seeds, f, indent=2)

                with open(os.path.join(k_dir, "metrics.json"), "w") as f:
                    json.dump(k_metrics, f, indent=2)

            # Generate Line Graphs at the probability folder level
            plot_metric_vs_k(
                budgets, sp_vs_k, "Maximin_SP",
                f"Maximin SP vs Budget (k) | {net_name} | {model_family} ({prob_folder_name})",
                os.path.join(prob_dir, "maximin_sp_vs_k.png")
            )

            plot_metric_vs_k(
                budgets, saved_frac_vs_k, "Saved_Fraction",
                f"Saved Fraction vs Budget (k) | {net_name} | {model_family} ({prob_folder_name})",
                os.path.join(prob_dir, "saved_fraction_vs_k.png")
            )

            # Save summary JSON for this probability setting
            summary_info = {
                "network": net_name,
                "model_family": model_family,
                "prob_setting": prob_folder_name,
                "pM": pM_val,
                "pT": pT_val,
                "budgets": budgets,
                "R_sim": R_sim,
                "baseline_outreach": baseline_outreach,
                "n_vulnerable": len(vuln_nodes),
                "sp_vs_k": sp_vs_k,
                "saved_frac_vs_k": saved_frac_vs_k
            }
            with open(os.path.join(prob_dir, "summary.json"), "w") as f:
                json.dump(summary_info, f, indent=2)

            print(f"    [Complete] Line graphs generated in {prob_dir}")

    print("\n====================================================================")
    print("   All Experiments Completed Successfully! Results in results/     ")
    print("====================================================================")

if __name__ == "__main__":
    run_experiment_pipeline()
