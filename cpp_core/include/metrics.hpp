#pragma once

#include "diffusion.hpp"
#include <vector>
#include <string>
#include <map>

namespace fair_spread {

struct MetricSummary {
    float maximin_sp = 0.0f;
    float saved_fraction = 0.0f;
    float speed_overall = 0.0f;
    float speed_min_over_groups = 0.0f;
    float avg_sp = 0.0f;
    float gini = 0.0f;
    float consistency = 0.0f;
    std::map<int, float> group_saved;
    float group_gap = 0.0f;
    float group_ratio = 1.0f;
    float outreach = 0.0f;
    float sp_p01 = 0.0f;
    float sp_median = 0.0f;
    std::vector<int> sp_hist20;
};

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
);

} // namespace fair_spread
