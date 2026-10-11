#include "../include/diffusion.hpp"
#include "../include/metrics.hpp"
#include <iostream>
#include <cassert>

using namespace fair_spread;

int main() {
    std::cout << "[Test C++ Core] Running validation on small graph..." << std::endl;

    // Small path graph: 0 -> 1 -> 2 -> 3 -> 4
    GraphCSR g;
    g.num_nodes = 5;
    g.row_ptr = {0, 1, 2, 3, 4, 4};
    g.col_indices = {1, 2, 3, 4};
    g.in_degrees = {0, 1, 1, 1, 1};
    g.out_degrees = {1, 1, 1, 1, 0};

    std::vector<int> M = {0};
    std::vector<int> T = {2};
    std::vector<int> groups = {0, 0, 1, 1, 1};

    std::vector<uint16_t> D_M;
    std::vector<uint16_t> D = compute_distances_bfs(g, M, D_M);
    assert(D_M[0] == 0);
    assert(D_M[1] == 1);
    assert(D_M[2] == 2);

    std::vector<float> sp0 = estimate_sp_cpp(g, M, {}, "COICM", 0.5f, 0.5f, 1.0f, 1.0f, 1.0f, "truth", 0, 1000, 42, 1);
    std::vector<float> spT = estimate_sp_cpp(g, M, T, "COICM", 0.5f, 0.5f, 1.0f, 1.0f, 1.0f, "truth", 0, 1000, 42, 1);

    assert(sp0[0] == 0.0f); // M is always infected
    assert(spT[2] == 1.0f); // T seed is protected

    std::vector<int> vuln = {1, 2, 3, 4};
    MetricSummary metrics = evaluate_metrics_cpp(
        g, M, T, groups, sp0, vuln, "COICM",
        0.5f, 0.5f, 1.0f, 1.0f, 1.0f, "truth", 0,
        1000, 12345, 1
    );

    assert(metrics.maximin_sp >= 0.0f && metrics.maximin_sp <= 1.0f);
    assert(metrics.saved_fraction >= 0.0f);
    std::cout << "[Test C++ Core] SUCCESS! maximin_sp=" << metrics.maximin_sp
              << " saved_fraction=" << metrics.saved_fraction << std::endl;
    return 0;
}
