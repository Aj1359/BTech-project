import math
import numpy as np
import networkx as nx
from collections import deque

try:
    from numba import njit, prange
    HAS_NUMBA = True
except Exception:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    prange = range


try:
    import fair_spread_cpp
    HAS_CPP = True
except ImportError:
    HAS_CPP = False


class GraphCSR:
    def __init__(self, G):
        nodes = list(G.nodes())
        self.num_nodes = len(nodes)
        self.node_to_idx = {node: i for i, node in enumerate(nodes)}
        self.idx_to_node = {i: node for i, node in enumerate(nodes)}

        row_ptr = [0]
        col_indices = []
        in_degrees = [0] * self.num_nodes
        out_degrees = [0] * self.num_nodes

        for i, u in enumerate(nodes):
            nbrs = list(G.successors(u) if G.is_directed() else G.neighbors(u))
            out_degrees[i] = len(nbrs)
            for v in nbrs:
                v_idx = self.node_to_idx[v]
                col_indices.append(v_idx)
                in_degrees[v_idx] += 1
            row_ptr.append(len(col_indices))

        self.row_ptr = np.array(row_ptr, dtype=np.int32)
        self.col_indices = np.array(col_indices, dtype=np.int32)
        self.in_degrees = np.array(in_degrees, dtype=np.int32)
        self.out_degrees = np.array(out_degrees, dtype=np.int32)

# ── SplitMix64 RNG ──────────────────────────────────────────────────────────
# When Numba is available the @njit path compiles to native uint64 (no warnings).
# When Numba is absent we use plain Python ints with a 64-bit mask; Python ints
# never overflow, so the RuntimeWarning is completely avoided.
if HAS_NUMBA:
    @njit(fastmath=True)
    def _splitmix64(seed):
        M = np.uint64(0xFFFFFFFFFFFFFFFF)
        z = (seed + np.uint64(0x9e3779b97f4a7c15)) & M
        z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)) & M
        z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)) & M
        return z ^ (z >> np.uint64(31))

    @njit(fastmath=True)
    def _rng_float(seed):
        next_s = _splitmix64(seed)
        val = float((next_s >> np.uint64(40)) & np.uint64(0xFFFFFF)) / 16777216.0
        return next_s, val
else:
    _M64 = 0xFFFFFFFFFFFFFFFF  # plain Python int — no overflow possible

    def _splitmix64(seed):
        seed = int(seed) & _M64
        z = (seed + 0x9e3779b97f4a7c15) & _M64
        z = ((z ^ (z >> 30)) * 0xbf58476d1ce4e5b9) & _M64
        z = ((z ^ (z >> 27)) * 0x94d049bb133111eb) & _M64
        return z ^ (z >> 31)

    def _rng_float(seed):
        next_s = _splitmix64(seed)
        val = float((next_s >> 40) & 0xFFFFFF) / 16777216.0
        return next_s, val

@njit(fastmath=True)
def _simulate_single_icm(
    row_ptr, col_indices, in_degrees,
    m_seeds, t_seeds,
    pM, pT, is_wc, c_T, c_M,
    tie_mode, delay_r, seed_val,
    state, time_step
):
    n = len(row_ptr) - 1
    state.fill(0)
    time_step.fill(32767)

    for m in m_seeds:
        state[m] = -1
        time_step[m] = 0

    frontier_pos = []
    frontier_neg = list(m_seeds)

    if delay_r == 0:
        for t in t_seeds:
            if state[t] != -1:
                state[t] = 1
                time_step[t] = 0
                frontier_pos.append(t)

    step = 0
    max_steps = n + 10
    rng_state = np.uint64(seed_val)

    while (len(frontier_pos) > 0 or len(frontier_neg) > 0 or step < delay_r) and step < max_steps:
        step += 1
        if step == delay_r and delay_r > 0:
            for t in t_seeds:
                if state[t] == 0:
                    state[t] = 1
                    time_step[t] = step
                    frontier_pos.append(t)

        new_pos = []
        new_neg = []

        for u in frontier_pos:
            s_idx = row_ptr[u]
            e_idx = row_ptr[u + 1]
            for e in range(s_idx, e_idx):
                v = col_indices[e]
                if state[v] == 0:
                    prob = pT
                    if is_wc:
                        prob = min(1.0, c_T / max(1, in_degrees[v]))
                    rng_state, rand_f = _rng_float(rng_state)
                    if rand_f < prob:
                        new_pos.append(v)

        for u in frontier_neg:
            s_idx = row_ptr[u]
            e_idx = row_ptr[u + 1]
            for e in range(s_idx, e_idx):
                v = col_indices[e]
                if state[v] == 0:
                    prob = pM
                    if is_wc:
                        prob = min(1.0, c_M / max(1, in_degrees[v]))
                    rng_state, rand_f = _rng_float(rng_state)
                    if rand_f < prob:
                        new_neg.append(v)

        # Conflict resolution
        pos_set = set(new_pos)
        neg_set = set(new_neg)
        conflicts = pos_set.intersection(neg_set)

        frontier_pos = []
        frontier_neg = []

        for v in pos_set:
            if v not in conflicts:
                frontier_pos.append(v)
            else:
                if tie_mode == 0:  # truth wins
                    frontier_pos.append(v)
                elif tie_mode == 1:  # misinfo wins
                    frontier_neg.append(v)
                else:  # coin flip
                    rng_state, coin = _rng_float(rng_state)
                    if coin < 0.5:
                        frontier_pos.append(v)
                    else:
                        frontier_neg.append(v)

        for v in neg_set:
            if v not in conflicts:
                frontier_neg.append(v)

        for v in frontier_pos:
            state[v] = 1
            time_step[v] = step
        for v in frontier_neg:
            state[v] = -1
            time_step[v] = step

@njit(parallel=True, fastmath=True)
def _estimate_sp_numba(
    row_ptr, col_indices, in_degrees,
    m_seeds, t_seeds,
    pM, pT, is_wc, c_T, c_M,
    tie_mode, delay_r,
    R_runs, base_seed
):
    n = len(row_ptr) - 1
    # Thread-safe matrix: each run writes exclusively to row r
    inf_matrix = np.zeros((R_runs, n), dtype=np.int8)

    for r in prange(R_runs):
        state = np.zeros(n, dtype=np.int8)
        time_step = np.zeros(n, dtype=np.int16)
        run_seed = np.uint64(base_seed) + np.uint64(r) * np.uint64(1000003)

        _simulate_single_icm(
            row_ptr, col_indices, in_degrees,
            m_seeds, t_seeds,
            pM, pT, is_wc, c_T, c_M,
            tie_mode, delay_r, run_seed,
            state, time_step
        )

        for v in range(n):
            if state[v] == -1:
                inf_matrix[r, v] = 1

    sp = np.zeros(n, dtype=np.float32)
    for v in range(n):
        cnt = 0
        for r in range(R_runs):
            cnt += inf_matrix[r, v]
        val = 1.0 - (float(cnt) / float(R_runs))
        if val < 0.0:
            val = 0.0
        elif val > 1.0:
            val = 1.0
        sp[v] = val
    return sp



def compute_distances_bfs(G, M_seeds):
    """
    Computes all-pairs shortest path distance on G with incoming edges to M removed.
    Returns (D, D_M)
    """
    G_prime = G.copy()
    if G_prime.is_directed():
        for m in M_seeds:
            for u in list(G_prime.predecessors(m)):
                G_prime.remove_edge(u, m)
    else:
        G_prime = nx.DiGraph(G_prime)
        for m in M_seeds:
            for u in list(G_prime.predecessors(m)):
                G_prime.remove_edge(u, m)

    D = dict(nx.all_pairs_shortest_path_length(G_prime))
    D_M = {v: float('inf') for v in G.nodes()}
    for m in M_seeds:
        lengths = D.get(m, {})
        for v, dist in lengths.items():
            if dist < D_M[v]:
                D_M[v] = dist

    return D, D_M

def estimate_sp_engine(
    graph_or_csr, M_seeds, T_seeds,
    model_type="COICM", pM=0.1, pT=0.1, gamma_T=1.0, c_T=1.0, c_M=1.0,
    tie_rule="truth", delay_r=0, R_runs=1000, base_seed=12345, *args, **kwargs
):
    # Support both GraphCSR and networkx.Graph
    if isinstance(graph_or_csr, GraphCSR):
        csr_graph = graph_or_csr
    elif hasattr(graph_or_csr, '_csr_cache'):
        csr_graph = graph_or_csr._csr_cache
    else:
        csr_graph = GraphCSR(graph_or_csr)

    # Handle positional args if passed by older thesis signatures: (G, M, T, pM, pT, R)
    if len(args) >= 1:
        pM = args[0]
    if len(args) >= 2:
        pT = args[1]
    if len(args) >= 3:
        R_runs = args[2]

    # Keyword overrides
    if 'R' in kwargs:
        R_runs = kwargs['R']
    if 'model' in kwargs:
        model_type = kwargs['model']  # alias: model= accepted alongside model_type=

    tie_mode = 0 if tie_rule == "truth" else (1 if tie_rule == "misinfo" else 2)
    is_wc = (model_type in ["WeightedCascade", "WC"])
    m_arr = np.array(list(M_seeds), dtype=np.int32)
    t_arr = np.array(list(T_seeds), dtype=np.int32)

    return _estimate_sp_numba(
        csr_graph.row_ptr, csr_graph.col_indices, csr_graph.in_degrees,
        m_arr, t_arr,
        float(pM), float(pT), is_wc, float(c_T), float(c_M),
        tie_mode, delay_r, int(R_runs), int(base_seed)
    )

def evaluate_metrics_engine(
    G, csr_graph, M_seeds, T_seeds, group_labels,
    sp0, vulnerable_nodes,
    model_type="COICM", pM=0.1, pT=0.1, gamma_T=1.0, c_T=1.0, c_M=1.0,
    tie_rule="truth", delay_r=0, R_eval=10000, eval_seed=54321
):
    n = csr_graph.num_nodes
    sp_T = estimate_sp_engine(
        csr_graph, M_seeds, T_seeds,
        model_type=model_type, pM=pM, pT=pT, gamma_T=gamma_T, c_T=c_T, c_M=c_M,
        tie_rule=tie_rule, delay_r=delay_r, R_runs=R_eval, base_seed=eval_seed
    )

    vuln_list = list(vulnerable_nodes)
    if not vuln_list:
        return {
            "maximin_sp": 1.0, "saved_fraction": 1.0, "speed_overall": 1.0,
            "speed_min_over_groups": 1.0, "avg_sp": 1.0, "gini": 0.0,
            "consistency": 1.0, "group_saved": {}, "group_gap": 0.0,
            "group_ratio": 1.0, "outreach": 0.0, "sp_p01": 1.0,
            "sp_median": 1.0, "sp_hist20": [0] * 20
        }

    sp_vuln = [float(sp_T[v]) for v in vuln_list]
    maximin_sp = min(sp_vuln)
    avg_sp = float(np.mean(sp_vuln))

    # Saved fraction
    exp_inf_0 = sum(1.0 - sp0[v] for v in vuln_list)
    exp_inf_T = sum(1.0 - sp_T[v] for v in vuln_list)
    saved_fraction = 1.0 - (exp_inf_T / exp_inf_0) if exp_inf_0 > 0 else 1.0

    # Outreach
    outreach = float(np.mean([1.0 - sp_T[v] for v in range(n)]))

    # Percentiles & Histogram
    sorted_sp = sorted(sp_vuln)
    sp_p01 = sorted_sp[max(0, int(0.01 * len(sorted_sp)))]
    sp_median = sorted_sp[len(sorted_sp) // 2]
    hist20, _ = np.histogram(sp_vuln, bins=20, range=(0.0, 1.0))

    # Gini
    arr = np.array(sp_vuln, dtype=float)
    mean_val = np.mean(arr)
    if mean_val > 0:
        diff_sum = np.sum(np.abs(np.subtract.outer(arr, arr)))
        gini = float(diff_sum / (2.0 * len(arr) * len(arr) * mean_val))
    else:
        gini = 0.0

    # Consistency
    num_cons, den_cons = 0.0, 0.0
    for i in range(len(vuln_list)):
        vi = vuln_list[i]
        for j in range(len(vuln_list)):
            vj = vuln_list[j]
            s_ij = 1.0 - abs(sp0[vi] - sp0[vj])
            diff = abs(sp_T[vi] - sp_T[vj])
            num_cons += diff * s_ij
            den_cons += s_ij
    consistency = float(1.0 - (num_cons / den_cons)) if den_cons > 0 else 1.0

    # Group saved fractions & gap/ratio
    unique_groups = sorted(list(set(group_labels.values())))
    group_saved = {}
    active_S = []

    for g in unique_groups:
        g_vuln = [v for v in vuln_list if group_labels.get(v, 0) == g]
        base_inf = sum(1.0 - sp0[v] for v in g_vuln)
        if base_inf > 0:
            t_inf = sum(1.0 - sp_T[v] for v in g_vuln)
            s_g = 1.0 - (t_inf / base_inf)
            group_saved[str(g)] = float(s_g)
            active_S.append(s_g)
        else:
            group_saved[str(g)] = 1.0  # unthreatened

    if active_S:
        group_gap = float(max(active_S) - min(active_S))
        group_ratio = float(min(active_S) / max(active_S)) if max(active_S) > 0 else 1.0
    else:
        group_gap = 0.0
        group_ratio = 1.0

    # Speed metric
    speed_overall = float(saved_fraction)
    speed_min_over_groups = float(min(active_S) if active_S else 1.0)

    return {
        "maximin_sp": float(maximin_sp),
        "saved_fraction": float(saved_fraction),
        "speed_overall": float(speed_overall),
        "speed_min_over_groups": float(speed_min_over_groups),
        "avg_sp": float(avg_sp),
        "gini": float(gini),
        "consistency": float(consistency),
        "group_saved": group_saved,
        "group_gap": float(group_gap),
        "group_ratio": float(group_ratio),
        "outreach": float(outreach),
        "sp_p01": float(sp_p01),
        "sp_median": float(sp_median),
        "sp_hist20": [int(x) for x in hist20]
    }
