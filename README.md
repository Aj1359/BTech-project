# Fair Truth-Spreader Placement Benchmark (Real Graphs Only)

Comprehensive experimental framework for evaluating fair truth-spreader placement algorithms under competing misinformation across real-world social networks.

---

## 1. Overview of Framework Components

- **`config.yaml`**: Main configuration specifying datasets, preprocessing, M placement policies, diffusion models, simulation counts (`R_select`, `R_eval`), and algorithms.
- **`cpp_core/`**: C++17 multithreaded (OpenMP) engine with CMake build system and `pybind11` bindings:
  - `include/diffusion.hpp`, `src/diffusion.cpp`: High-performance CSR simulation core supporting COICM, MCICM, CLT, and Weighted Cascade with deterministic RNG per (run, thread).
  - `include/metrics.hpp`, `src/metrics.cpp`: Fast metric computations (Maximin SP, Saved Fraction, Speed, Gini, Consistency, Group Gap/Ratio, Outreach, Histograms).
  - `src/bindings.cpp`: `pybind11` Python bindings.
  - `tests/test_core.cpp`: Standalone C++ verification suite.
- **`sim_engine.py`**: High-performance in-process simulation & metric calculation engine utilizing JIT parallel acceleration (via Numba / NumPy) and CSR representation for native speed on any platform.
- **`algo_wrapper.py`**: Unified interface `algo(G, M, k, model_setting, R_select, rng_seed) -> list of nodes` with prefix-reuse optimization and model-agnostic caching.
- **`runner.py`**: Automated experiment runner executing the benchmark grid, checkpointing individual runs, and writing leaf JSON files.
- **`aggregator.py`**: Compiles all leaf JSON results into `results/master.json` and outputs network sanity tables.
- **`test_validation.py`**: Validation suite executing all 9 verification tests required by the experiment specification.
- **`algos/`**: Modular directory containing all implemented seed selection algorithms.

---

## 2. Algorithms Roster

| Algorithm ID | Category | Description |
|---|---|---|
| `random` | Bounds | Random uniform baseline |
| `topdegree` | Agnostic | Top out-degree seed selector |
| `cmia_o` | Agnostic | Local arborescence heuristic |
| `naive_myopic` | Individual-fair | Lowest initial save probability $sp_0$ |
| `myopic` | Individual-fair | Myopic Welfare Maximin (min $sp$ selection) |
| `protect` | Thesis | PROTECT shortest-path enhancer heuristic |
| `naive_protect` | Thesis | Naive PROTECT distance-based heuristic |
| `fwrrs` | Group-fair | Weighted reversible reachable trees (*Saxena et al., 2023*) |
| `pagerank` | Ranking | Classical stationary PageRank |
| `lfpr` / `orf_lfpr` | Ranking | Locally Fair PageRank & One-Step Refinement (*Kumar et al., 2026*) |
| `fspr` / `mf_fspr` | Ranking | Fairness-Sensitive PageRank & Mean-Field (*Kumar et al., 2026*) |
| `proposed` | Proposed | Group-aware time-aware PROTECT |

---

## 3. Running Validation Tests

Before launching long-running benchmark evaluations, execute the 9-part validation suite:

```bash
python test_validation.py
```

This verifies:
1. $T = \emptyset$ yields `saved_fraction = 0`.
2. Exact probability agreement within 3 SE on a path graph.
3. Distance rule consistency when $p_T = 1.0$, delay 0, truth wins.
4. Monotonic non-decreasing behavior of `maximin_sp` over budget $k$.
5. Lower welfare for Random baseline.
6. Deterministic simulation reproducibility with identical seeds.
7. Prefix-reuse equivalence.
8. Boundedness of $sp \in [0, 1]$.
9. Ranking properties, non-negativity, and Pearson correlation $> 0.85$ between exact and mean-field fair PageRanks.

---

## 4. Launching the Benchmark

Run the automated benchmark runner:

```bash
python runner.py
```

### Execution Flow:
1. Loads datasets from `networks/`, cleans and extracts the **Largest Connected Component (LCC/LSCC)**.
2. Partitions/assigns demographic groups (from file or Louvain community detection).
3. Calibrates $\alpha$ so $M$ reaches $\ge 70\%$ baseline outreach (or uses Table 7.1 overrides).
4. Generates $M$ starter placements (`highdeg`, `random`, `majority`, `minority`, `split_prop`).
5. Evaluates $T = \emptyset$ baseline and stores `M.json`.
6. Executes algorithm selection with $R_{select}$ simulations.
7. Evaluates $T$ on **FRESH independent evaluation simulations** with $R_{eval}$.
8. Writes leaf JSONs matching Section 8 schema and checkpoints automatically.

---

## 5. Aggregating Results and Building Master Tables

After runs have been executed, build `master.json` and view network summary tables:

```bash
python aggregator.py
```

Output:
- Compiles `results/master.json` across all networks, placements, models, and budgets.
- Prints the network sanity summary table reporting baseline outreach, vulnerable set size, and group counts.
