#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <random>
#include <unordered_map>
#include <unordered_set>

namespace fair_spread {

struct SimulationResult {
    std::vector<int8_t> state;      // 0: inactive, 1: positive (truth), -1: negative (misinformation)
    std::vector<int16_t> step_time; // Step at which node was activated
};

struct GraphCSR {
    int num_nodes = 0;
    std::vector<int> row_ptr;
    std::vector<int> col_indices;
    std::vector<int> in_degrees;
    std::vector<int> out_degrees;
};

// Breadth-first search shortest path matrix computation
// with incoming edges to M removed
std::vector<uint16_t> compute_distances_bfs(
    const GraphCSR& graph,
    const std::vector<int>& M_seeds,
    std::vector<uint16_t>& out_D_M
);

// Unified multithreaded diffusion simulator
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
);

} // namespace fair_spread
