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

# ==========================================
# Misinformation Placement (M Modes)
# ==========================================

def select_m_seeds(G, m_size, mode="highdeg", groups=None, seed=42):
    """
    Selects misinformation seed set M of size m_size according to mode:
    - 'highdeg': Highest degree nodes
    - 'random': Uniform random selection
    - 'majority': Concentrated in the largest community
    - 'minority': Concentrated in the smallest community with >= m_size nodes
    - 'split_prop': Proportional allocation across all communities
    """
    rng = random.Random(seed)
    nodes = list(G.nodes())

    if mode in ["highdeg", "highest_degree"]:
        degrees = dict(G.degree())
        sorted_nodes = sorted(nodes, key=lambda x: (degrees[x], rng.random()), reverse=True)
        return set(sorted_nodes[:m_size])

    elif mode == "random":
        return set(rng.sample(nodes, min(m_size, len(nodes))))

    elif mode == "majority":
        if not groups:
            return set(rng.sample(nodes, min(m_size, len(nodes))))
        group_counts = defaultdict(list)
        for v, g in groups.items():
            group_counts[g].append(v)
        largest_g = max(group_counts.keys(), key=lambda g: len(group_counts[g]))
        pool = group_counts[largest_g]
        degrees = dict(G.degree(pool))
        sorted_pool = sorted(pool, key=lambda x: (degrees[x], rng.random()), reverse=True)
        return set(sorted_pool[:min(m_size, len(sorted_pool))])

    elif mode == "minority":
        if not groups:
            return set(rng.sample(nodes, min(m_size, len(nodes))))
        group_counts = defaultdict(list)
        for v, g in groups.items():
            group_counts[g].append(v)
        viable_groups = [g for g, vlist in group_counts.items() if len(vlist) >= m_size]
        target_g = min(viable_groups, key=lambda g: len(group_counts[g])) if viable_groups else min(group_counts.keys(), key=lambda g: len(group_counts[g]))
        pool = group_counts[target_g]
        degrees = dict(G.degree(pool))
        sorted_pool = sorted(pool, key=lambda x: (degrees[x], rng.random()), reverse=True)
        return set(sorted_pool[:min(m_size, len(sorted_pool))])

    elif mode in ["split_prop", "split_equal"]:
        if not groups:
            return set(rng.sample(nodes, min(m_size, len(nodes))))
        group_counts = defaultdict(list)
        for v, g in groups.items():
            group_counts[g].append(v)
        total_nodes = len(nodes)
        M = set()
        for g, vlist in sorted(group_counts.items()):
            m_g = max(1, round(m_size * (len(vlist) / float(total_nodes))))
            degrees = dict(G.degree(vlist))
            sorted_v = sorted(vlist, key=lambda x: (degrees[x], rng.random()), reverse=True)
            M.update(sorted_v[:m_g])
        return set(list(M)[:m_size])

    else:
        degrees = dict(G.degree())
        sorted_nodes = sorted(nodes, key=lambda x: degrees[x], reverse=True)
        return set(sorted_nodes[:m_size])

# ==========================================
# Main Experiment Pipeline
# ==========================================

def run_experiment_pipeline(
    results_dir="results",
    R_sim=1000,
    budgets=[5, 10, 15, 20, 25],
    m_size=10,
    m_modes=["highdeg"],
    target_networks=None
):
    print("====================================================================")
    print("   Fair Truth-Spreader Placement Benchmark - Complete Runner       ")
    print(f"   Monte Carlo Runs: {R_sim} | Budgets k: {budgets} | M Size: {m_size}")
    print(f"   M Placement Modes: {m_modes}")
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

    if target_networks:
        datasets = [ds for ds in datasets if ds["name"] in target_networks]

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
                prob_base = f"pT_{alpha}"
                pM_val, pT_val = alpha, alpha
            else:
                prob_base = f"pM_{alpha}_pT_1.0"
                pM_val, pT_val = alpha, 1.0

            for m_mode in m_modes:
                # Subfolder organization
                if len(m_modes) > 1:
                    prob_dir = os.path.join(net_dir, f"m_{m_mode}", prob_base)
                else:
                    prob_dir = os.path.join(net_dir, prob_base)
                os.makedirs(prob_dir, exist_ok=True)

                print(f"\n    [M Mode: {m_mode}] Running: {prob_base} (pM={pM_val}, pT={pT_val})")

                # Select M seeds according to mode
                M_seeds = select_m_seeds(G, m_size, mode=m_mode, groups=groups, seed=42)
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

                # Outreach strictly over vulnerable nodes (excluding M)
                baseline_outreach = float(np.mean([1.0 - sp0[v] for v in vuln_nodes]))
                sp0_vuln = [float(sp0[v]) for v in vuln_nodes]
                baseline_min_sp = min(sp0_vuln)
                baseline_zeros = sum(1 for x in sp0_vuln if x == 0.0)
                baseline_near_one = sum(1 for x in sp0_vuln if x < 0.001)
                sorted_sp0 = sorted(sp0_vuln)
                sp0_p01 = sorted_sp0[max(0, int(0.01 * len(sorted_sp0)))]
                sp0_p05 = sorted_sp0[max(0, int(0.05 * len(sorted_sp0)))]

                print(f"       Baseline T=empty:")
                print(f"          Vulnerable nodes = {len(vuln_nodes)} (out of {G.number_of_nodes()})")
                print(f"          Outreach = {baseline_outreach*100:.2f}% | Avg SP = {np.mean(sp0_vuln):.4f} | Min SP = {baseline_min_sp:.4f}")
                print(f"          Nodes with SP=0: {baseline_zeros} | Nodes with SP<0.001 (inf>99.9%): {baseline_near_one}")
                print(f"          Tail SP: p01 = {sp0_p01:.4f} | p05 = {sp0_p05:.4f}")

                # Save M.json for baseline & aggregator
                m_info = {
                    "M": [int(x) for x in M_seeds],
                    "m_mode": m_mode,
                    "baseline": {
                        "outreach_empty": baseline_outreach,
                        "n_vulnerable": len(vuln_nodes),
                        "min_sp": float(baseline_min_sp),
                        "avg_sp": float(np.mean(sp0_vuln))
                    }
                }
                with open(os.path.join(prob_dir, "M.json"), "w") as f:
                    json.dump(m_info, f, indent=2)

                # Metrics collectors for plotting vs k
                sp_vs_k = defaultdict(dict)
                p05_vs_k = defaultdict(dict)
                saved_frac_vs_k = defaultdict(dict)

                max_k = max(budgets)
                prefix_cache = {}

                # Loop over budgets k
                for k in budgets:
                    k_dir = os.path.join(prob_dir, f"k_{k}")
                    os.makedirs(k_dir, exist_ok=True)

                    k_seeds = {}
                    k_metrics = {}

                    print(f"         -> Evaluating Budget k = {k}...")

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
                                    groups=unique_groups, estimate_sp=estimate_sp_engine,
                                    csr=csr_graph,
                                    L=25, R_cand=150, R_base=500, p_frac=0.20, theta=None,
                                    phi_jump=0.80, phi_walk=0.70, eta=0.7, tau=2.0,
                                    eps=0.05, gamma=0.15, tail_frac=0.05
                                )
                                if algo_id in PREFIX_REUSABLE_ALGOS:
                                    prefix_cache[algo_id] = T_full
                                    T = T_full[:k]
                                else:
                                    T = T_full[:k]

                            sel_runtime = time.time() - start_time

                            # Paired Evaluation Run (Common Random Numbers with baseline sp0)
                            eval_seed = 12345
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
                            p05_vs_k[algo_id][k] = metrics["sp_p05"]
                            saved_frac_vs_k[algo_id][k] = metrics["saved_fraction"]

                            # ── Complete Per-iteration live progress ─────────────
                            print(
                                f"            [{algo_id:22s}] k={k:2d} | "
                                f"minSP={metrics['maximin_sp']:.4f} "
                                f"b5%={metrics['mean_bottom_5pct']:.4f} "
                                f"saved={metrics['saved_fraction']*100:4.1f}% "
                                f"avgSP={metrics['avg_sp']:.4f} "
                                f"gini={metrics['gini']:.4f} "
                                f"gap={metrics['group_gap']:.4f} "
                                f"rt={metrics['runtime_select_s']:.2f}s"
                            )

                        except Exception as e:
                            print(f"            [Error] {algo_id} at k={k}: {e}")

                    # Save seeds and metrics for this budget k
                    with open(os.path.join(k_dir, "seeds.json"), "w") as f:
                        json.dump(k_seeds, f, indent=2)

                    with open(os.path.join(k_dir, "metrics.json"), "w") as f:
                        json.dump(k_metrics, f, indent=2)

                # Generate Line Graphs at the probability folder level
                plot_metric_vs_k(
                    budgets, sp_vs_k, "Maximin_SP",
                    f"Maximin SP vs Budget (k) | {net_name} ({m_mode}) | {model_family} ({prob_base})",
                    os.path.join(prob_dir, "maximin_sp_vs_k.png")
                )

                plot_metric_vs_k(
                    budgets, p05_vs_k, "SP_P05 (5th Percentile)",
                    f"Tail SP (5th Percentile) vs Budget (k) | {net_name} ({m_mode}) | {model_family} ({prob_base})",
                    os.path.join(prob_dir, "sp_p05_vs_k.png")
                )

                plot_metric_vs_k(
                    budgets, saved_frac_vs_k, "Saved_Fraction",
                    f"Saved Fraction vs Budget (k) | {net_name} ({m_mode}) | {model_family} ({prob_base})",
                    os.path.join(prob_dir, "saved_fraction_vs_k.png")
                )

                # Save summary JSON for this setting
                summary_info = {
                    "network": net_name,
                    "model_family": model_family,
                    "m_mode": m_mode,
                    "prob_setting": prob_base,
                    "pM": pM_val,
                    "pT": pT_val,
                    "budgets": budgets,
                    "R_sim": R_sim,
                    "baseline_outreach": baseline_outreach,
                    "n_vulnerable": len(vuln_nodes),
                    "sp_vs_k": sp_vs_k,
                    "p05_vs_k": p05_vs_k,
                    "saved_frac_vs_k": saved_frac_vs_k
                }
                with open(os.path.join(prob_dir, "summary.json"), "w") as f:
                    json.dump(summary_info, f, indent=2)

                print(f"       [Complete] Summary and line graphs generated in {prob_dir}")

    print("\n====================================================================")
    print("   All Experiments Completed Successfully! Results in results/     ")
    print("====================================================================")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fair Truth-Spreader Placement Benchmark Runner")
    parser.add_argument("--m_modes", nargs="+", default=["highdeg"],
                        choices=["highdeg", "random", "majority", "minority", "split_prop", "all"],
                        help="Misinformation placement modes (default: highdeg, or 'all')")
    parser.add_argument("--R_sim", type=int, default=1000, help="Monte Carlo simulation runs (default: 1000)")
    parser.add_argument("--budgets", nargs="+", type=int, default=[5, 10, 15, 20, 25], help="Truth budgets k")
    parser.add_argument("--networks", nargs="+", default=None, help="Target network names (e.g. Facebook)")
    args = parser.parse_args()

    active_modes = args.m_modes
    if "all" in active_modes:
        active_modes = ["highdeg", "majority", "minority", "random", "split_prop"]

    run_experiment_pipeline(
        m_modes=active_modes,
        R_sim=args.R_sim,
        budgets=args.budgets,
        target_networks=args.networks
    )
