#include "../include/diffusion.hpp"
#include <algorithm>
#include <queue>
#include <random>
#include <limits>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace fair_spread {

// SplitMix64 / xoshiro fast deterministic generator for reproducible per-thread streams
struct FastRNG {
    uint64_t s;
    explicit FastRNG(uint64_t seed) : s(seed == 0 ? 0x9E3779B97F4A7C15ULL : seed) {}
    inline uint64_t next_u64() {
        uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
    inline float next_float() {
        return (next_u64() >> 40) * (1.0f / 16777216.0f);
    }
};

std::vector<uint16_t> compute_distances_bfs(
    const GraphCSR& graph,
    const std::vector<int>& M_seeds,
    std::vector<uint16_t>& out_D_M
) {
    int n = graph.num_nodes;
    std::vector<uint16_t> D(n * n, std::numeric_limits<uint16_t>::max());
    out_D_M.assign(n, std::numeric_limits<uint16_t>::max());

    std::vector<bool> is_M(n, false);
    for (int m : M_seeds) is_M[m] = true;

    // Run BFS from every node on G with incoming edges to M removed
    for (int src = 0; src < n; ++src) {
        std::queue<int> q;
        q.push(src);
        D[src * n + src] = 0;

        while (!q.empty()) {
            int u = q.front();
            q.pop();
            uint16_t cur_d = D[src * n + u];

            int start_idx = graph.row_ptr[u];
            int end_idx = graph.row_ptr[u + 1];
            for (int e = start_idx; e < end_idx; ++e) {
                int v = graph.col_indices[e];
                // In G', incoming edges of M are removed (cannot enter M)
                if (is_M[v]) continue;

                if (D[src * n + v] == std::numeric_limits<uint16_t>::max()) {
                    D[src * n + v] = cur_d + 1;
                    q.push(v);
                }
            }
        }
    }

    // Compute D_M[v] = min_{m in M} D[m, v]
    for (int v = 0; v < n; ++v) {
        uint16_t min_d = std::numeric_limits<uint16_t>::max();
        for (int m : M_seeds) {
            uint16_t d = D[m * n + v];
            if (d < min_d) min_d = d;
        }
        out_D_M[v] = min_d;
    }

    return D;
}

static inline void simulate_single_cascade(
    const GraphCSR& graph,
    const std::vector<int>& M_seeds,
    const std::vector<int>& T_seeds,
    const std::string& model_type,
    float pM,
    float pT,
    float gamma_T,
    float c_T,
    float c_M,
    const std::string& tie_rule,
    int delay_r,
    FastRNG& rng,
    std::vector<int8_t>& state,
    std::vector<int16_t>& time_step
) {
    int n = graph.num_nodes;
    std::fill(state.begin(), state.end(), 0);
    std::fill(time_step.begin(), time_step.end(), std::numeric_limits<int16_t>::max());

    for (int m : M_seeds) {
        state[m] = -1;
        time_step[m] = 0;
    }

    std::vector<int> frontier_pos;
    std::vector<int> frontier_neg = M_seeds;

    if (delay_r == 0) {
        for (int t : T_seeds) {
            if (state[t] != -1) {
                state[t] = 1;
                time_step[t] = 0;
                frontier_pos.push_back(t);
            }
        }
    }

    int step = 0;
    int max_steps = n + 10;

    std::vector<int> new_pos;
    std::vector<int> new_neg;
    std::vector<int> conflict_candidates;

    while ((!frontier_pos.empty() || !frontier_neg.empty() || step < delay_r) && step < max_steps) {
        step++;
        if (step == delay_r && delay_r > 0) {
            for (int t : T_seeds) {
                if (state[t] == 0) {
                    state[t] = 1;
                    time_step[t] = static_cast<int16_t>(step);
                    frontier_pos.push_back(t);
                }
            }
        }

        new_pos.clear();
        new_neg.clear();

        for (int u : frontier_pos) {
            int start_idx = graph.row_ptr[u];
            int end_idx = graph.row_ptr[u + 1];
            for (int e = start_idx; e < end_idx; ++e) {
                int v = graph.col_indices[e];
                if (state[v] == 0) {
                    float prob = pT;
                    if (model_type == "WeightedCascade" || model_type == "WC") {
                        prob = std::min(1.0f, c_T / std::max(1, graph.in_degrees[v]));
                    }
                    if (rng.next_float() < prob) {
                        new_pos.push_back(v);
                    }
                }
            }
        }

        for (int u : frontier_neg) {
            int start_idx = graph.row_ptr[u];
            int end_idx = graph.row_ptr[u + 1];
            for (int e = start_idx; e < end_idx; ++e) {
                int v = graph.col_indices[e];
                if (state[v] == 0) {
                    float prob = pM;
                    if (model_type == "WeightedCascade" || model_type == "WC") {
                        prob = std::min(1.0f, c_M / std::max(1, graph.in_degrees[v]));
                    }
                    if (rng.next_float() < prob) {
                        new_neg.push_back(v);
                    }
                }
            }
        }

        // De-duplicate newly activated nodes
        std::sort(new_pos.begin(), new_pos.end());
        new_pos.erase(std::unique(new_pos.begin(), new_pos.end()), new_pos.end());

        std::sort(new_neg.begin(), new_neg.end());
        new_neg.erase(std::unique(new_neg.begin(), new_neg.end()), new_neg.end());

        // Resolve conflicts (tie rule)
        frontier_pos.clear();
        frontier_neg.clear();

        size_t i = 0, j = 0;
        while (i < new_pos.size() && j < new_neg.size()) {
            if (new_pos[i] == new_neg[j]) {
                int v = new_pos[i];
                if (tie_rule == "truth") {
                    frontier_pos.push_back(v);
                } else if (tie_rule == "misinfo") {
                    frontier_neg.push_back(v);
                } else if (tie_rule == "coin") {
                    if (rng.next_float() < 0.5f) frontier_pos.push_back(v);
                    else frontier_neg.push_back(v);
                } else {
                    frontier_pos.push_back(v);
                }
                i++;
                j++;
            } else if (new_pos[i] < new_neg[j]) {
                frontier_pos.push_back(new_pos[i++]);
            } else {
                frontier_neg.push_back(new_neg[j++]);
            }
        }
        while (i < new_pos.size()) frontier_pos.push_back(new_pos[i++]);
        while (j < new_neg.size()) frontier_neg.push_back(new_neg[j++]);

        for (int v : frontier_pos) {
            state[v] = 1;
            time_step[v] = static_cast<int16_t>(step);
        }
        for (int v : frontier_neg) {
            state[v] = -1;
            time_step[v] = static_cast<int16_t>(step);
        }
    }
}

std::vector<float> estimate_sp_cpp(
    const GraphCSR& graph,
    const std::vector<int>& M_seeds,
    const std::vector<int>& T_seeds,
    const std::string& model_type,
    float pM,
    float pT,
    float gamma_T,
    float c_T,
    float c_M,
    const std::string& tie_rule,
    int delay_r,
    int R_runs,
    uint64_t base_seed,
    int num_threads
) {
    int n = graph.num_nodes;
    std::vector<int> infection_counts(n, 0);

#pragma omp parallel num_threads(num_threads > 0 ? num_threads : 4)
    {
        std::vector<int> local_counts(n, 0);
        std::vector<int8_t> state(n);
        std::vector<int16_t> time_step(n);

#pragma omp for schedule(static)
        for (int r = 0; r < R_runs; ++r) {
            FastRNG rng(base_seed + static_cast<uint64_t>(r) * 1000003ULL);
            simulate_single_cascade(
                graph, M_seeds, T_seeds, model_type,
                pM, pT, gamma_T, c_T, c_M, tie_rule, delay_r,
                rng, state, time_step
            );
            for (int v = 0; v < n; ++v) {
                if (state[v] == -1) local_counts[v]++;
            }
        }

#pragma omp critical
        {
            for (int v = 0; v < n; ++v) {
                infection_counts[v] += local_counts[v];
            }
        }
    }

    std::vector<float> sp(n);
    float inv_R = 1.0f / static_cast<float>(R_runs);
    for (int v = 0; v < n; ++v) {
        sp[v] = 1.0f - (static_cast<float>(infection_counts[v]) * inv_R);
    }
    return sp;
}

} // namespace fair_spread
