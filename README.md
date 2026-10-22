# Fair Truth-Spreader Placement Benchmark

Experimental framework for evaluating **fair truth-spreader placement algorithms** under competing misinformation in real-world social networks. The goal is to place a limited budget of *truth seeds* to **maximise the minimum save probability** (maximin welfare) across all vulnerable nodes.

> **Proposed Method**: FTRW (Fairness-Targeted Random Walk) — a novel family of random walk models that dynamically re-draw protected vs. unprotected node groups at every seed-selection step, combining FSPR/LFPR-style walks with Monte Carlo evaluation to optimise the maximin welfare objective.

---

## 1. Project Structure

```
├── run_experiments.py      # Main experiment pipeline (CLI entry point)
├── sim_engine.py           # Numba-accelerated ICM simulation engine (CSR)
├── algo_wrapper.py         # Unified algorithm interface + prefix-reuse
├── aggregator.py           # Result compilation → master.json
├── test_validation.py      # 10-test verification suite
├── config.yaml             # Dataset/model configuration
├── placem.py               # M seed placement strategies
├── metrics.py              # Metric computation helpers
│
├── algos/                  # 26 seed-selection algorithms
│   ├── ftrw.py             # FTRW v1 — proposed (LFPR + dynamic groups)
│   ├── ftrw_v2.py          # FTRW v2 — Reverse-Rescue Dual Walk
│   ├── ftrw_v3.py          # FTRW v3 — Multi-Resolution Shield Walk
│   ├── protect.py          # PROTECT (shortest-path heuristic)
│   ├── fspr.py / lfpr.py   # Fair PageRank variants
│   └── ...                 # 20+ baseline/comparison algorithms
│
├── networks/               # Edge-list files (Facebook, Email, CA-*)
├── results/                # Output: seeds, metrics, plots per (net, k, model)
└── cpp_core/               # Optional C++17 engine (OpenMP + pybind11)
```

---

## 2. Proposed Method — FTRW Algorithm Family

All three variants share the core idea: **dynamic protected/unprotected groups that change at every step** based on the current save-probability distribution. Misinformation seeds $M$ are placed on the highest-degree nodes.

### FTRW v1 — Fairness-Targeted Random Walk ([`algos/ftrw.py`](algos/ftrw.py))

1. Monte Carlo estimate of save probability $sp(v)$ under current seeds $T$
2. **Dynamic groups**: Protected $P$ = worst-off vulnerable nodes (bottom $p_{\text{frac}}$ quantile of $sp$); Unprotected $U$ = remaining vulnerable nodes. Groups are re-drawn every iteration.
3. Deficit × proximity-to-$M$ weighting: $h(v) = d(v) \cdot [\eta + (1-\eta) \cdot e^{-\text{dist}(M,v)/\tau}]$
4. **FSPR-style jump** vector biased toward $P$ + **LFPR-style walk** with group-normalised transitions
5. Shortlist top-$L$ walk scores → evaluate with CRN Monte Carlo → pick maximin-best with tail tie-breaking

### FTRW v2 — Reverse-Rescue Dual Walk ([`algos/ftrw_v2.py`](algos/ftrw_v2.py))

Adds three innovations on top of v1:
- **Reverse rescue PPR**: walks *backward* from the worst-off nodes to find "savior" candidates that can spread truth to the most vulnerable
- **Neighbour deficit scoring**: scores each candidate by the total deficit of its worst-off neighbours (direct rescue potential)
- **Adaptive φ**: `phi_jump` and `phi_walk` increase when many nodes have $sp = 0$, relaxing as the distribution improves
- **Coverage-aware tie-breaking**: penalises candidates near already-selected seeds

### FTRW v3 — Multi-Resolution Shield Walk ([`algos/ftrw_v3.py`](algos/ftrw_v3.py))

A fundamentally different multi-signal approach:
- **Multi-resolution walks** at $p_{\text{frac}} \in \{0.05, 0.20, 0.50\}$ — captures candidates that help different layers of the SP distribution
- **Shield score**: identifies nodes on shortest $M \to$ vulnerable paths (blocking misinformation flow)
- **Degree-deficit fusion**: combines $\sqrt{\text{degree}}$ with 2-hop deficit aggregation
- Merges all shortlists into one wide candidate pool for evaluation

---

## 3. Full Algorithm Roster (26 algorithms)

| ID | Category | Description |
|---|---|---|
| `random` | Baseline | Uniform random seed selection |
| `topdegree` | Agnostic | Highest out-degree nodes |
| `cmia_o` | Agnostic | Local arborescence heuristic |
| `naive_myopic` | Individual-fair | Lowest initial $sp_0$ selection |
| `myopic` | Individual-fair | Myopic Welfare Maximin (greedy min $sp$) |
| `protect` | Thesis | PROTECT shortest-path enhancer |
| `naive_protect` | Thesis | Naive PROTECT distance-based |
| `fwrrs` | Group-fair | Fair Welfare-aware Reverse Reachable Sampling |
| `pagerank` | Ranking | Classical stationary PageRank |
| `lfpr` / `orf_lfpr` | Ranking | Locally Fair PageRank & One-Step Refinement |
| `fspr` / `mf_fspr` | Ranking | Fairness-Sensitive PageRank & Mean-Field |
| `proposed` | Group-aware | Group-aware time-aware PROTECT |
| `greedy_maximin` | Greedy | Exact greedy maximin (small graphs only) |
| `gonzalez` | Diversity | Gonzalez k-center heuristic |
| `mindegree_hcn` | Structural | Min-degree hub-centre-neighbour |
| `myopic_bfs` | Individual-fair | Myopic BFS-guided selection |
| `myopic_ppr` | Individual-fair | Myopic PPR-guided selection |
| `leastcentral` | Structural | Least central node heuristic |
| `betweenness` | Structural | Betweenness centrality ranking |
| `habc_vuln` | Structural | HABC/EABC vulnerable targeting |
| `group_maximin_greedy` | Group-fair | Group-aware maximin greedy |
| `dp_aware_fibm` | Group-fair | DP-aware fair influence blocking |
| **`ftrw`** | **Proposed** | **FTRW v1 — Dynamic group random walk** |
| **`ftrw_v2`** | **Proposed** | **FTRW v2 — Reverse-Rescue Dual Walk** |
| **`ftrw_v3`** | **Proposed** | **FTRW v3 — Multi-Resolution Shield Walk** |

---

## 4. Simulation Engine

[`sim_engine.py`](sim_engine.py) provides:
- **Numba JIT-compiled parallel ICM** simulation (`_estimate_sp_numba`) with deterministic SplitMix64 RNG
- **`GraphCSR`** — compressed sparse row graph representation for cache-friendly traversal
- **Common Random Numbers (CRN)** — paired evaluation seeds for variance reduction
- Support for **COICM** (equal propagation) and **MCICM** (asymmetric $p_M, p_T$) models
- Automatic legacy positional-argument detection for backward compatibility

---

## 5. Bug Fixes Applied

| Bug | Fix |
|---|---|
| **Unordered sets / broken prefix reuse** | All 26 algorithms return ordered lists; `algo_wrapper.py` preserves insertion order |
| **`estimate_sp` positional binding** | Auto-detects legacy `(G, M, T, pM, pT, R)` calls; strict `pM ≤ 1.0` / `pT ≤ 1.0` assertions |
| **Group noise & RNG pairing** | CRN variance reduction with paired base seeds; tiny community filtering (< 1%); saved fraction clamped to [0, 1] |

---

## 6. Running Validation Tests

```bash
python test_validation.py
```

Verifies 10 tests:
1. $T = \emptyset$ yields `saved_fraction = 0`
2. Exact probability agreement (± 3 SE) on a path graph
3. Distance rule consistency ($p_T = 1.0$, delay 0, truth wins)
4. Monotonic non-decreasing `maximin_sp` over budget $k$
5. Lower welfare for Random baseline
6. Deterministic reproducibility with identical seeds
7. Prefix-reuse equivalence
8. Boundedness of $sp \in [0, 1]$
9. Ranking properties & Pearson correlation > 0.85 (exact vs. mean-field fair PageRank)
10. Bug-fix regression tests (prefix matching, positional binding)

---

## 7. Running Experiments

### Quick single-network run:
```bash
python run_experiments.py --m_modes highdeg --networks Facebook
```

### Full benchmark:
```bash
python run_experiments.py --m_modes highdeg --R_sim 1000 --budgets 5 10 15 20 25
```

### CLI options:
| Flag | Default | Description |
|---|---|---|
| `--m_modes` | `highdeg` | M placement: `highdeg`, `random`, `majority`, `minority`, `split_prop`, `all` |
| `--R_sim` | `1000` | Monte Carlo simulation runs |
| `--budgets` | `5 10 15 20 25` | Truth seed budgets $k$ |
| `--networks` | all | Target networks (e.g., `Facebook`, `CA-GrQc`) |

### Output structure:
```
results/
└── COICM/
    └── Facebook/
        └── pT_0.2/
            ├── M.json                    # Baseline info
            ├── summary.json              # All metrics across k
            ├── maximin_sp_vs_k.png       # Line plot
            ├── sp_p05_vs_k.png           # Tail metric plot
            ├── saved_fraction_vs_k.png   # Coverage plot
            └── k_5/
                ├── seeds.json            # Selected seeds per algorithm
                └── metrics.json          # Full metrics per algorithm
```

---

## 8. Metrics

| Metric | Description |
|---|---|
| `maximin_sp` | Minimum save probability over all vulnerable nodes (↑ better) |
| `mean_bottom_5pct` | Mean SP of the worst-off 5% of vulnerable nodes |
| `avg_sp` | Mean save probability across all vulnerable nodes |
| `saved_fraction` | Fraction of expected infections prevented vs. $T = \emptyset$ |
| `gini` | Gini coefficient of SP distribution (↓ = more equitable) |
| `group_gap` | Max − min saved fraction across Louvain communities |
| `sp_p01` / `sp_p05` | 1st and 5th percentile of SP distribution |

---

## 9. Datasets

| Network | Nodes (LCC) | Edges | $\alpha$ | Source |
|---|---|---|---|---|
| Facebook | 4,039 | 88,234 | 0.20 | SNAP |
| Arenas Email | 1,005 | 25,571 | 0.30 | SNAP |
| CA-GrQc | 5,242 | 14,496 | 0.15 | SNAP |
| CA-HepTh | 9,877 | 25,998 | 0.15 | SNAP |
| CA-AstroPh | 18,772 | 198,110 | 0.25 | SNAP |

Download networks:
```bash
python download_networks.py
```

---

## 10. Requirements

- Python ≥ 3.8
- NumPy, SciPy, NetworkX, Matplotlib
- Numba (optional, for JIT-accelerated simulation)
- python-louvain (`community` package, for community detection)
