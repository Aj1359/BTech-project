#include "../include/diffusion.hpp"
#include "../include/metrics.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace fair_spread;

PYBIND11_MODULE(fair_spread_cpp, m) {
    m.doc() = "C++17 Multithreaded Core for Fair Truth-Spreader Placement";

    py::class_<GraphCSR>(m, "GraphCSR")
        .def(py::init<>())
        .def_readwrite("num_nodes", &GraphCSR::num_nodes)
        .def_readwrite("row_ptr", &GraphCSR::row_ptr)
        .def_readwrite("col_indices", &GraphCSR::col_indices)
        .def_readwrite("in_degrees", &GraphCSR::in_degrees)
        .def_readwrite("out_degrees", &GraphCSR::out_degrees);

    py::class_<MetricSummary>(m, "MetricSummary")
        .def_readonly("maximin_sp", &MetricSummary::maximin_sp)
        .def_readonly("saved_fraction", &MetricSummary::saved_fraction)
        .def_readonly("speed_overall", &MetricSummary::speed_overall)
        .def_readonly("speed_min_over_groups", &MetricSummary::speed_min_over_groups)
        .def_readonly("avg_sp", &MetricSummary::avg_sp)
        .def_readonly("gini", &MetricSummary::gini)
        .def_readonly("consistency", &MetricSummary::consistency)
        .def_readonly("group_saved", &MetricSummary::group_saved)
        .def_readonly("group_gap", &MetricSummary::group_gap)
        .def_readonly("group_ratio", &MetricSummary::group_ratio)
        .def_readonly("outreach", &MetricSummary::outreach)
        .def_readonly("sp_p01", &MetricSummary::sp_p01)
        .def_readonly("sp_median", &MetricSummary::sp_median)
        .def_readonly("sp_hist20", &MetricSummary::sp_hist20);

    m.def("compute_distances_bfs", [](const GraphCSR& graph, const std::vector<int>& M_seeds) {
        std::vector<uint16_t> D_M;
        std::vector<uint16_t> D = compute_distances_bfs(graph, M_seeds, D_M);
        return py::make_tuple(D, D_M);
    }, "Computes BFS shortest paths with incoming edges to M removed");

    m.def("estimate_sp", &estimate_sp_cpp,
        py::arg("graph"),
        py::arg("M_seeds"),
        py::arg("T_seeds"),
        py::arg("model_type") = "COICM",
        py::arg("pM") = 0.1f,
        py::arg("pT") = 0.1f,
        py::arg("gamma_T") = 1.0f,
        py::arg("c_T") = 1.0f,
        py::arg("c_M") = 1.0f,
        py::arg("tie_rule") = "truth",
        py::arg("delay_r") = 0,
        py::arg("R_runs") = 1000,
        py::arg("base_seed") = 12345ULL,
        py::arg("num_threads") = 4,
        "Multithreaded Monte Carlo simulation of save probability sp(v)");

    m.def("evaluate_metrics", &evaluate_metrics_cpp,
        py::arg("graph"),
        py::arg("M_seeds"),
        py::arg("T_seeds"),
        py::arg("group_labels"),
        py::arg("sp0"),
        py::arg("vulnerable_nodes"),
        py::arg("model_type") = "COICM",
        py::arg("pM") = 0.1f,
        py::arg("pT") = 0.1f,
        py::arg("gamma_T") = 1.0f,
        py::arg("c_T") = 1.0f,
        py::arg("c_M") = 1.0f,
        py::arg("tie_rule") = "truth",
        py::arg("delay_r") = 0,
        py::arg("R_eval") = 10000,
        py::arg("eval_seed") = 54321ULL,
        py::arg("num_threads") = 4,
        "Evaluates the complete suite of experiment metrics on independent simulations");
}
