import os
import time
import json
import yaml
import hashlib
import random
import numpy as np
import networkx as nx
from collections import defaultdict

from sim_engine import (
    GraphCSR, compute_distances_bfs,
    estimate_sp_engine, evaluate_metrics_engine
)
from algo_wrapper import AlgoManager, MODEL_AGNOSTIC_ALGOS, PREFIX_REUSABLE_ALGOS
from placem import place_m

def load_network_and_preprocess(edge_file, is_directed=False):
    create_using = nx.DiGraph() if is_directed else nx.Graph()
    G = nx.read_edgelist(edge_file, create_using=create_using, data=False)

    # Induce on Largest Connected Component / SCC
    if G.is_directed():
        components = list(nx.strongly_connected_components(G))
    else:
        components = list(nx.connected_components(G))

    if components:
        largest_comp = max(components, key=len)
        G = G.subgraph(largest_comp).copy()

    G = nx.convert_node_labels_to_integers(G)
    return G

def get_or_create_groups(G, group_file=None, seed=42):
    if group_file and os.path.exists(group_file):
        groups = {}
        with open(group_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    groups[int(parts[0])] = int(parts[1])
        nx.set_node_attributes(G, groups, 'group')
        source = f"file:{group_file}"
    else:
        # Louvain community detection
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(G, random_state=seed)
        except Exception:
            # Fallback label propagation
            communities = list(nx.community.label_propagation_communities(G))
            partition = {}
            for g_id, comm in enumerate(communities):
                for node in comm:
                    partition[node] = g_id

        nx.set_node_attributes(G, partition, 'group')
        source = "louvain"

    return nx.get_node_attributes(G, 'group'), source

def calibrate_alpha(G, csr_graph, M_highdeg, R=1000):
    """
    On highest-degree M (|M|=10), pick smallest alpha on {0.05, 0.10, ..., 0.50}
    for which outreach with T=empty is at least 70%.
    """
    grid = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    n = G.number_of_nodes()

    for p in grid:
        sp0 = estimate_sp_engine(
            csr_graph, M_highdeg, set(),
            model_type="COICM", pM=p, pT=p, R_runs=R, base_seed=12345
        )
        outreach = np.mean([1.0 - sp0[v] for v in range(n)])
        if outreach >= 0.70:
            return p
    return 0.30  # default fallback

def run_benchmark(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    run_log_path = os.path.join(results_dir, "run_log.txt")

    def log(msg):
        print(msg)
        with open(run_log_path, 'a') as lf:
            lf.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")

    log("=== Starting Benchmark Run ===")
    algo_mgr = AlgoManager()
    log(f"Loaded algorithms: {list(algo_mgr.loaded_funcs.keys())}")

    budgets = config.get('budgets', [5, 10, 15, 20, 25])
    max_k = max(budgets)
    R_select = config.get('R_select', 1000)
    R_eval = config.get('R_eval', 10000)

    for net_cfg in config['networks']:
        net_name = net_cfg['name']
        edge_file = net_cfg['edge_file']
        if not os.path.exists(edge_file):
            log(f"[Skip] Edge file {edge_file} not found for {net_name}")
            continue

        log(f"\n>>> Loading Network: {net_name} ({edge_file})")
        G = load_network_and_preprocess(edge_file)
        csr_graph = GraphCSR(G)
        groups, group_source = get_or_create_groups(G, net_cfg.get('group_file'))

        # Store Network Metadata
        net_dir = os.path.join(results_dir, net_name)
        os.makedirs(net_dir, exist_ok=True)
        net_meta = {
            "name": net_name,
            "n": G.number_of_nodes(),
            "m": G.number_of_edges(),
            "directed": G.is_directed(),
            "group_source": group_source,
            "group_counts": {str(g): sum(1 for node in G if groups[node] == g) for g in set(groups.values())}
        }
        with open(os.path.join(net_dir, "network_meta.json"), 'w') as f:
            json.dump(net_meta, f, indent=2)

        # Alpha Calibration / Override
        if net_cfg.get('alpha_override') is not None:
            coicm_alpha = net_cfg['alpha_override']
        else:
            M_high, _, _ = place_m(G, m=10, mode="highdeg", seed=12345)
            coicm_alpha = calibrate_alpha(G, csr_graph, M_high)
        log(f"Configured COICM alpha for {net_name}: {coicm_alpha}")

        # Model settings to execute
        model_runs = [("COICM", {"model": "COICM", "pM": coicm_alpha, "pT": coicm_alpha, "p": coicm_alpha})]
        if net_name in config['models']['MCICM'].get('networks', []) or config['models']['MCICM'].get('networks') == 'all':
            model_runs.append(("MCICM", {"model": "MCICM", "pM": coicm_alpha, "pT": 1.0, "p": coicm_alpha}))

        # Loop over placements
        for placement in config['M']['placements']:
            repeats = config['M']['repeats'].get(placement, 1)

            for rep in range(repeats):
                rep_seed = config['M']['seed'] + rep * 777
                M_seeds, _, _ = place_m(G, m=config['M']['size'], mode=placement, seed=rep_seed)
                D, D_M = compute_distances_bfs(G, M_seeds)

                m_dir = os.path.join(net_dir, f"M_{placement}_rep{rep}")
                os.makedirs(m_dir, exist_ok=True)
                seeds_dir = os.path.join(m_dir, "seeds")
                os.makedirs(seeds_dir, exist_ok=True)

                # Loop over models
                for model_type, m_params in model_runs:
                    pM = m_params['pM']
                    pT = m_params['pT']

                    if model_type == "COICM":
                        model_dir = os.path.join(m_dir, "COICM", f"p_{m_params['p']}")
                    else:
                        model_dir = os.path.join(m_dir, "MCICM", f"pM_{pM}", f"pT_{pT}")

                    # 1. Baseline Run (T = empty)
                    sp0 = estimate_sp_engine(
                        csr_graph, M_seeds, set(),
                        model_type=model_type, pM=pM, pT=pT,
                        R_runs=R_eval, base_seed=rep_seed + 101
                    )
                    vuln_nodes = [v for v in range(G.number_of_nodes()) if v not in M_seeds and sp0[v] < 1.0]
                    baseline_outreach = float(np.mean([1.0 - sp0[v] for v in range(G.number_of_nodes())]))

                    baseline_data = {
                        "outreach_empty": baseline_outreach,
                        "n_vulnerable": len(vuln_nodes),
                        "vulnerable_per_group": {
                            str(g): sum(1 for v in vuln_nodes if groups.get(v, 0) == g)
                            for g in set(groups.values())
                        }
                    }
                    with open(os.path.join(m_dir, "M.json"), 'w') as f:
                        json.dump({"M": list(M_seeds), "baseline": baseline_data}, f, indent=2)

                    # Cache for prefix reuse: {algo_id: list_of_seeds_at_max_k}
                    prefix_cache = {}

                    for k in budgets:
                        k_dir = os.path.join(model_dir, f"k_{k}")
                        os.makedirs(k_dir, exist_ok=True)

                        for algo_id in config['algorithms']:
                            leaf_file = os.path.join(k_dir, f"{algo_id}.json")
                            # Checkpointing
                            if os.path.exists(leaf_file):
                                try:
                                    with open(leaf_file, 'r') as f:
                                        c_data = json.load(f)
                                        if c_data.get('status') == 'ok':
                                            continue
                                except Exception:
                                    pass

                            # Skip greedy maximin on large networks
                            if algo_id == 'greedy_maximin' and G.number_of_nodes() > 500:
                                res_leaf = {
                                    "network": net_name, "placement": placement, "rep": rep,
                                    "model": model_type, "pM": pM, "pT": pT, "k": k, "algo": algo_id,
                                    "status": "skipped", "reason": "graph too large for greedy maximin"
                                }
                                with open(leaf_file, 'w') as f:
                                    json.dump(res_leaf, f, indent=2)
                                continue

                            start_sel = time.time()
                            try:
                                # Determine seed set T
                                # Check if already computed in seeds/ for model-agnostic algos
                                seed_cache_file = os.path.join(seeds_dir, f"{algo_id}_k{k}.json")

                                if algo_id in MODEL_AGNOSTIC_ALGOS and os.path.exists(seed_cache_file):
                                    with open(seed_cache_file, 'r') as f:
                                        T = json.load(f)['T']
                                elif algo_id in PREFIX_REUSABLE_ALGOS and algo_id in prefix_cache:
                                    T = prefix_cache[algo_id][:k]
                                else:
                                    # Need full run at max_k for prefix reusable or at k
                                    run_k = max_k if algo_id in PREFIX_REUSABLE_ALGOS else k
                                    T_full = algo_mgr.run_algo(
                                        algo_id, G, M_seeds, run_k, m_params,
                                        R_select=R_select, rng_seed=rep_seed,
                                        D=D, D_M=D_M, vulnerable_set=set(vuln_nodes),
                                        groups=list(set(groups.values())),
                                        estimate_sp=estimate_sp_engine
                                    )
                                    if algo_id in PREFIX_REUSABLE_ALGOS:
                                        prefix_cache[algo_id] = T_full
                                        T = T_full[:k]
                                    else:
                                        T = T_full[:k]

                                    if algo_id in MODEL_AGNOSTIC_ALGOS:
                                        with open(seed_cache_file, 'w') as f:
                                            json.dump({"T": T}, f, indent=2)

                                time_sel = time.time() - start_sel

                                # Independent Fresh Evaluation
                                start_eval = time.time()
                                eval_seed = (rep_seed * 10007 + k * 97 + 54321) & 0xFFFFFFFF
                                metrics = evaluate_metrics_engine(
                                    G, csr_graph, M_seeds, T, groups,
                                    sp0, vuln_nodes,
                                    model_type=model_type, pM=pM, pT=pT,
                                    R_eval=R_eval, eval_seed=eval_seed
                                )
                                time_eval = time.time() - start_eval

                                leaf_record = {
                                    "network": net_name,
                                    "placement": placement,
                                    "rep": rep,
                                    "model": model_type,
                                    "pM": pM,
                                    "pT": pT,
                                    "k": k,
                                    "algo": algo_id,
                                    "T": [int(x) for x in T],
                                    "R_select": R_select,
                                    "R_eval": R_eval,
                                    "seed": rep_seed,
                                    "metrics": metrics,
                                    "baseline": baseline_data,
                                    "runtime_select_s": round(time_sel, 4),
                                    "runtime_eval_s": round(time_eval, 4),
                                    "status": "ok"
                                }
                            except Exception as e:
                                leaf_record = {
                                    "network": net_name, "placement": placement, "rep": rep,
                                    "model": model_type, "pM": pM, "pT": pT, "k": k, "algo": algo_id,
                                    "status": "failed", "error": str(e)
                                }

                            with open(leaf_file, 'w') as f:
                                json.dump(leaf_record, f, indent=2)

    log("=== Benchmark Run Completed Successfully ===")

if __name__ == "__main__":
    run_benchmark()
