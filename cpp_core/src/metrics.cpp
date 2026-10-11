#include "../include/metrics.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace fair_spread {

MetricSummary evaluate_metrics_cpp(
    const GraphCSR& graph,
    const std::vector<int>& M_seeds,
    const std::vector<int>& T_seeds,
    const std::vector<int>& group_labels,
    const std::vector<float>& sp0,
    const std::vector<int>& vulnerable_nodes,
    const std::string& model_type,
    float pM,
    float pT,
    float gamma_T,
    float c_T,
    float c_M,
    const std::string& tie_rule,
    int delay_r,
    int R_eval,
    uint64_t eval_seed,
    int num_threads
) {
    int n = graph.num_nodes;
    std::vector<int> infection_counts(n, 0);
    int max_it = n + 10;

    // Distinct groups
    std::vector<int> unique_groups;
    for (int g : group_labels) {
        if (std::find(unique_groups.begin(), unique_groups.end(), g) == unique_groups.end()) {
            unique_groups.push_back(g);
        }
    }

    std::map<int, int> vuln_per_group;
    for (int g : unique_groups) vuln_per_group[g] = 0;
    for (int v : vulnerable_nodes) {
        vuln_per_group[group_labels[v]]++;
    }

    // Speed accumulator per group: sum across runs
    std::map<int, double> speed_acc_g;
    for (int g : unique_groups) speed_acc_g[g] = 0.0;
    double speed_acc_overall = 0.0;

#pragma omp parallel num_threads(num_threads > 0 ? num_threads : 4)
    {
        std::vector<int> local_inf(n, 0);
        std::map<int, double> local_speed_g;
        for (int g : unique_groups) local_speed_g[g] = 0.0;
        double local_speed_overall = 0.0;

        std::vector<int8_t> state(n);
        std::vector<int16_t> time_step(n);

#pragma omp for schedule(static)
        for (int r = 0; r < R_eval; ++r) {
            // Distinct RNG stream per run
            FastRNG rng(eval_seed + static_cast<uint64_t>(r) * 982451653ULL);
            simulate_single_cascade(
                graph, M_seeds, T_seeds, model_type,
                pM, pT, gamma_T, c_T, c_M, tie_rule, delay_r,
                rng, state, time_step
            );

            for (int v = 0; v < n; ++v) {
                if (state[v] == -1) local_inf[v]++;
            }

            // Compute speed for this run
            for (int v : vulnerable_nodes) {
                if (state[v] == 1) { // saved by truth
                    int t_v = time_step[v];
                    double score = static_cast<double>(max_it - t_v + 1) / static_cast<double>(max_it);
                    local_speed_g[group_labels[v]] += score;
                    local_speed_overall += score;
                }
            }
        }

#pragma omp critical
        {
            for (int v = 0; v < n; ++v) infection_counts[v] += local_inf[v];
            for (int g : unique_groups) speed_acc_g[g] += local_speed_g[g];
            speed_acc_overall += local_speed_overall;
        }
    }

    MetricSummary summary;
    float inv_R = 1.0f / static_cast<float>(R_eval);

    // Compute sp(v)
    std::vector<float> sp(n);
    for (int v = 0; v < n; ++v) {
        sp[v] = 1.0f - (static_cast<float>(infection_counts[v]) * inv_R);
    }

    // Outreach
    int total_infected = 0;
    for (int v = 0; v < n; ++v) total_infected += infection_counts[v];
    summary.outreach = (static_cast<float>(total_infected) * inv_R) / static_cast<float>(n);

    if (vulnerable_nodes.empty()) {
        summary.maximin_sp = 1.0f;
        summary.saved_fraction = 1.0f;
        summary.avg_sp = 1.0f;
        return summary;
    }

    // Extract sp for vulnerable nodes
    std::vector<float> sp_vuln;
    sp_vuln.reserve(vulnerable_nodes.size());
    double sum_sp_vuln = 0.0;
    float min_sp = 1.0f;

    for (int v : vulnerable_nodes) {
        float val = sp[v];
        sp_vuln.push_back(val);
        sum_sp_vuln += val;
        if (val < min_sp) min_sp = val;
    }

    summary.maximin_sp = min_sp;
    summary.avg_sp = static_cast<float>(sum_sp_vuln / vulnerable_nodes.size());

    // Saved fraction
    double exp_inf_0 = 0.0;
    double exp_inf_T = 0.0;
    for (int v : vulnerable_nodes) {
        exp_inf_0 += (1.0f - sp0[v]);
        exp_inf_T += (1.0f - sp[v]);
    }
    summary.saved_fraction = (exp_inf_0 > 0.0)
        ? static_cast<float>(1.0 - (exp_inf_T / exp_inf_0))
        : 1.0f;

    // Speed calculation
    double total_vuln_count = static_cast<double>(vulnerable_nodes.size());
    summary.speed_overall = static_cast<float>(speed_acc_overall / (total_vuln_count * R_eval));

    float min_group_speed = std::numeric_limits<float>::max();
    for (int g : unique_groups) {
        int vg_count = vuln_per_group[g];
        if (vg_count > 0) {
            float g_spd = static_cast<float>(speed_acc_g[g] / (static_cast<double>(vg_count) * R_eval));
            if (g_spd < min_group_speed) min_group_speed = g_spd;
        }
    }
    summary.speed_min_over_groups = (min_group_speed == std::numeric_limits<float>::max()) ? 0.0f : min_group_speed;

    // Percentiles and histogram (20 bins)
    std::vector<float> sorted_sp = sp_vuln;
    std::sort(sorted_sp.begin(), sorted_sp.end());
    size_t idx_01 = std::min(sorted_sp.size() - 1, static_cast<size_t>(0.01f * sorted_sp.size()));
    size_t idx_med = sorted_sp.size() / 2;
    summary.sp_p01 = sorted_sp[idx_01];
    summary.sp_median = sorted_sp[idx_med];

    summary.sp_hist20.assign(20, 0);
    for (float val : sorted_sp) {
        int bin = static_cast<int>(val * 20.0f);
        if (bin >= 20) bin = 19;
        if (bin < 0) bin = 0;
        summary.sp_hist20[bin]++;
    }

    // Gini coefficient
    int num_vuln = static_cast<int>(sp_vuln.size());
    double diff_sum = 0.0;
    for (int i = 0; i < num_vuln; ++i) {
        for (int j = 0; j < num_vuln; ++j) {
            diff_sum += std::abs(sp_vuln[i] - sp_vuln[j]);
        }
    }
    summary.gini = (sum_sp_vuln > 0.0)
        ? static_cast<float>(diff_sum / (2.0 * num_vuln * sum_sp_vuln))
        : 0.0f;

    // Consistency
    double num_cons = 0.0;
    double den_cons = 0.0;
    for (int i = 0; i < num_vuln; ++i) {
        int vi = vulnerable_nodes[i];
        for (int j = 0; j < num_vuln; ++j) {
            int vj = vulnerable_nodes[j];
            double s_ij = 1.0 - std::abs(sp0[vi] - sp0[vj]);
            double d_sp = std::abs(sp[vi] - sp[vj]);
            num_cons += d_sp * s_ij;
            den_cons += s_ij;
        }
    }
    summary.consistency = (den_cons > 0.0) ? static_cast<float>(1.0 - (num_cons / den_cons)) : 1.0f;

    // Per-group saved fraction S_g and Group Gap / Ratio
    std::vector<float> active_S_g;
    for (int g : unique_groups) {
        double base_inf_g = 0.0;
        double t_inf_g = 0.0;
        for (int v : vulnerable_nodes) {
            if (group_labels[v] == g) {
                base_inf_g += (1.0f - sp0[v]);
                t_inf_g += (1.0f - sp[v]);
            }
        }
        if (base_inf_g > 0.0) {
            float s_g = static_cast<float>(1.0 - (t_inf_g / base_inf_g));
            summary.group_saved[g] = s_g;
            active_S_g.push_back(s_g);
        } else {
            // Group has no threatened nodes -> unthreatened
            summary.group_saved[g] = 1.0f;
        }
    }

    if (!active_S_g.empty()) {
        float max_S = *std::max_element(active_S_g.begin(), active_S_g.end());
        float min_S = *std::min_element(active_S_g.begin(), active_S_g.end());
        summary.group_gap = max_S - min_S;
        summary.group_ratio = (max_S > 0.0f) ? (min_S / max_S) : 1.0f;
    } else {
        summary.group_gap = 0.0f;
        summary.group_ratio = 1.0f;
    }

    return summary;
}

} // namespace fair_spread
