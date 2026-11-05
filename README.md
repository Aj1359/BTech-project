# Fair Truth-Spreader Placement against Misinformation

**B.Tech Project · Indian Institute of Technology Bhilai**

When misinformation starts spreading from a known set of sources $M$, where should we place a small budget of $k$ *truth seeds* so that **nobody is left behind**? Classical influence-blocking methods maximise the *total* number of people saved and routinely leave whole regions of the network unprotected. This project studies the **fair** version of the problem, maximising the save probability of the *worst-off* node, and proposes **FTRW (Fairness-Targeted Random Walk)**, a family of fairness-aware random-walk seed selectors, benchmarked against 23 baselines on five real social networks.

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#proposed-method-ftrw">Method</a> ·
  <a href="#repository-layout">Layout</a> ·
  <a href="#algorithms">Algorithms</a> ·
  <a href="#datasets">Datasets</a> ·
  <a href="#documentation--references">Docs</a>
</p>

---

## Problem

Given a graph $G=(V,E)$, misinformation seeds $M \subset V$ and a truth budget $k$, two cascades (truth $T$ and misinformation $M$) compete under an Independent Cascade model. The **save probability** $sp(v;T)$ is the probability that node $v$ is *not* captured by misinformation.

$$
\max_{T \subseteq V\setminus M,\ |T|\le k}\ \ \min_{v \in V_{\text{vuln}}} sp(v;T)
\qquad V_{\text{vuln}} = \{v \notin M : sp(v;\emptyset) < 1\}
$$

The maximin objective is non-submodular and expensive to evaluate (each $sp$ is a Monte Carlo estimate), which is what motivates walk-based heuristics that target the worst-off nodes directly.

**Diffusion models**

| Model | Propagation | Used on |
|---|---|---|
| **COICM** (Competitive Independent Cascade) | $p_M = p_T = \alpha$ | all networks |
| **MCICM** (high-effectiveness truth) | $p_M = \alpha,\ p_T = 1.0$ | Facebook, Arenas Email |

Ties are resolved in favour of truth; truth seeds activate with zero detection delay.

---

## Proposed method: FTRW

All FTRW variants share one idea: **the protected group is not fixed**. At every greedy step $t$, vulnerable nodes are re-partitioned by their *current* save probability into a protected set $P_t$ (worst off) and an unprotected set $U_t$. A fairness-aware PageRank walk biased towards $P_t$ produces a shortlist, and the shortlist is then scored by Common-Random-Numbers Monte Carlo to pick the candidate with the best maximin gain.

| Variant | File | Key idea |
|---|---|---|
| **FTRW v1** · Dynamic Group Walk | [`algos/ftrw.py`](algos/ftrw.py) | FSPR-style jump biased to $P_t$, weighted by deficit × proximity to $M$: $h(v) = (1-sp_t(v))\,[\eta + (1-\eta)e^{-d(M,v)/\tau}]$, plus an LFPR-style group-normalised walk. |
| **FTRW v2** · Reverse-Rescue Dual Walk | [`algos/ftrw_v2.py`](algos/ftrw_v2.py) | Adds a *reverse* PPR from the worst-off nodes to find their "saviours", neighbour-deficit scoring, adaptive $\varphi$ and coverage-aware tie-breaking. |
| **FTRW v3** · Multi-Resolution Shield Walk | [`algos/ftrw_v3.py`](algos/ftrw_v3.py) | Walks at several protected fractions $p_{\text{frac}} \in \{0.05, 0.20, 0.50\}$, a *shield* score for nodes on $M \to$ vulnerable shortest paths, and $\sqrt{\deg}$ × 2-hop deficit fusion. |

**Protected-set strategies** (FTRW v1, `group_mode=` kwarg)

| `group_mode` | Rule for $P_t$ |
|---|---|
| `quantile` *(default)* | Worst $p_{\text{frac}}$ share of vulnerable nodes by current $sp$. |
| `stat_band` | Nodes whose $sp$ confidence interval overlaps that of the current minimum; soft-weighted. |
| `rescuable` | Drops "pinned" nodes (within $d_0$ hops of $M$, or stagnant over a window) from the walk mass. |
| `dual` | Multiplicative-weights dual: persistent per-node weights $\lambda$; $P_t$ carries $(1-\alpha_d)$ of the $\lambda$ mass. |

**Spreadability-based variants:** the newer selectors replace degree-based jump vectors with *2-hop spreadability* and residual infection risk: `nsj_lfpr`, `sj_fspr`, `sj_lfpr`, `pr_target`, `pr_spread`, `pr_dist` (see [Algorithms](#algorithms)).

The full mathematical formulation is in [`docs/PROPOSED_METHOD.md`](docs/PROPOSED_METHOD.md) and the reports in [`docs/reports/`](docs/reports/).

---

## Quick start

```bash
git clone https://github.com/Aj1359/BTech-project.git
cd BTech-project
python -m venv .venv && source .venv/bin/activate     # optional
pip install -r requirements.txt
```

The five SNAP edge lists are already included in [`networks/`](networks/). To re-download them:

```bash
python scripts/download_networks.py
```

**1. Verify the installation** (12 tests, ~10 s):

```bash
python -m unittest discover -s tests -v
```

**2. Run a small experiment:**

```bash
python run_experiments.py --networks Arenas_email --budgets 5 10 --R_sim 200
```

**3. Run the full benchmark:**

```bash
python run_experiments.py --m_modes all --R_sim 1000 --budgets 5 10 15 20 25
```

**4. Print the ranked leaderboard for a finished run:**

```bash
python analyze_results.py --network Facebook --model COICM
```

> **Performance:** install `numba`. Without it the Monte Carlo engine falls back to pure Python and a full run is very slow; even a single small network with 32 algorithms takes many minutes.

### `run_experiments.py` options

| Flag | Default | Description |
|---|---|---|
| `--networks` | all five | Subset of `Facebook`, `Arenas_email`, `CA-GrQc`, `CA-HepTh`, `CA-AstroPh` |
| `--budgets` | `5 10 15 20 25` | Truth-seed budgets $k$ |
| `--R_sim` | `1000` | Monte Carlo runs per estimate |
| `--m_modes` | `highdeg` | Placement of $M$: `highdeg`, `random`, `majority`, `minority`, `split_prop`, or `all` |
| `--group_mode` | `binary` | Groups for FSPR/LFPR and group metrics: `binary` (protected vs. unprotected by baseline $sp$) or `community` (Louvain) |
| `--split_quantile` | `0.5` | Quantile of baseline $sp$ separating protected / unprotected in `binary` mode |

### Output

```
results/
└── COICM/                         # or MCICM/
    └── Facebook/
        └── [m_<mode>/]pT_0.2/     # m_<mode>/ level only when several --m_modes are given
            ├── M.json             # misinformation seeds + baseline diagnostics
            ├── summary.json       # every metric, every algorithm, every k
            ├── maximin_sp_vs_k.png
            ├── sp_p05_vs_k.png
            ├── saved_fraction_vs_k.png
            └── k_5/
                ├── seeds.json     # selected truth seeds per algorithm
                └── metrics.json   # full metrics per algorithm
```

`results/` is git-ignored. An earlier full COICM run is kept in [`results_archive/`](results_archive/) for reference.

---

## Repository layout

```
BTech-project/
├── run_experiments.py        # ▶ main benchmark pipeline (CLI entry point)
├── analyze_results.py        # ▶ ranked leaderboard from summary.json
├── requirements.txt
│
├── fairspread/               # core library
│   ├── sim_engine.py         #   CSR graph, Numba-parallel competing-ICM Monte Carlo, CRN, metrics
│   ├── algo_wrapper.py       #   algorithm registry + uniform get_seeds() interface, prefix reuse
│   ├── placem.py             #   misinformation-seed (M) placement strategies
│   ├── simulators.py         #   reference pure-Python cascade simulators
│   └── metrics.py            #   fairness / welfare metric helpers
│
├── algos/                    # one file per seed-selection algorithm (get_seeds(G, M, k, **kw))
├── tests/test_validation.py  # 12-test correctness suite
├── scripts/download_networks.py
├── networks/                 # SNAP edge lists
├── results_archive/          # archived results of an earlier COICM run
├── cpp_core/                 # experimental C++17 / OpenMP / pybind11 engine (see below)
├── legacy/                   # earlier experiment pipelines + config-driven runner
└── docs/
    ├── PROPOSED_METHOD.md    # FTRW formulation
    ├── reports/              # project reports (PDF)
    └── papers/               # reference papers (PDF)
```

**Adding an algorithm:** create `algos/my_algo.py` exposing `get_seeds(G, M, budget, **kwargs) -> list` (ordered, best first), then register it in `ALGO_FILES` in [`fairspread/algo_wrapper.py`](fairspread/algo_wrapper.py). Add it to `PREFIX_REUSABLE_ALGOS` if its seeds for smaller $k$ are a prefix of those for larger $k$; the pipeline then computes it once at $k_{\max}$.

---

## Algorithms

32 algorithms are registered and loaded by the pipeline.

| ID | Family | Description |
|---|---|---|
| **`ftrw`** | **Proposed** | FTRW v1: dynamic-group fair random walk + CRN Monte Carlo shortlist |
| **`ftrw_v2`** | **Proposed** | FTRW v2: Reverse-Rescue Dual Walk |
| **`ftrw_v3`** | **Proposed** | FTRW v3: Multi-Resolution Shield Walk |
| **`nsj_lfpr`** | **Proposed** | Neighbour-Spread Jump LFPR: spreadability-weighted jump, neighbour-biased fair walk, adaptive jump factor |
| **`sj_fspr`** | **Proposed** | FSPR with a 2-hop spreadability jump vector; $P_t$ = highest residual risk |
| **`sj_lfpr`** | **Proposed** | LFPR with a spreadability jump vector and locally fair transitions |
| **`pr_target`** | **Proposed** | Reverse PageRank restarting at at-risk nodes ∝ residual risk |
| **`pr_spread`** | **Proposed** | Target-PageRank × forward 2-hop spreadability, risk discounted per seed |
| **`pr_dist`** | **Proposed** | Target-PageRank shortlist + greedy distance coverage of still-savable nodes |
| `proposed` | Group-aware | Group-aware, time-aware PROTECT |
| `protect` | Thesis baseline | PROTECT shortest-path enhancer |
| `naive_protect` | Thesis baseline | Distance-based naive PROTECT |
| `myopic` | Individual-fair | Myopic welfare maximin (greedy on min $sp$) |
| `naive_myopic` | Individual-fair | Lowest initial $sp_0$ first |
| `myopic_bfs` | Individual-fair | BFS-guided myopic selection |
| `myopic_ppr` | Individual-fair | PPR-guided myopic selection |
| `greedy_maximin` | Greedy | Exact greedy maximin (small graphs only) |
| `group_maximin_greedy` | Group-fair | Group-aware maximin greedy |
| `fwrrs` | Group-fair | Fair welfare-aware reverse-reachable sampling |
| `dp_aware_fibm` | Group-fair | Demographic-parity-aware fair influence blocking |
| `pagerank` | Ranking | Classical PageRank |
| `fspr` / `mf_fspr` | Fair ranking | Fairness-Sensitive PageRank / mean-field approximation |
| `lfpr` / `orf_lfpr` | Fair ranking | Locally Fair PageRank / one-step refinement |
| `topdegree` | Structural | Highest degree |
| `betweenness` | Structural | Betweenness centrality |
| `leastcentral` | Structural | Least-central nodes |
| `mindegree_hcn` | Structural | Min-degree hub/centre/neighbour heuristic |
| `habc_vuln` | Structural | HABC/EABC centrality on vulnerable nodes |
| `gonzalez` | Diversity | Gonzalez $k$-center |
| `random` | Baseline | Uniform random |

`cmia_o` is listed in the registry but has no implementation file yet, so it is skipped at load time. `algos/pr_common.py` and `algos/sj_common.py` are shared helpers; `mindegree_hc.py`, `mindegree_nd.py` and `set_mogwo.py` are implemented but not registered.

---

## Metrics

| Metric | Meaning |
|---|---|
| `maximin_sp` ↑ | Minimum save probability over vulnerable nodes (the objective) |
| `mean_bottom_5pct` ↑ | Mean $sp$ of the worst-off 5% |
| `sp_p01`, `sp_p05` ↑ | 1st / 5th percentile of the $sp$ distribution |
| `avg_sp` ↑ | Mean save probability |
| `saved_fraction` ↑ | Fraction of expected infections prevented relative to $T=\emptyset$ (clamped to [0, 1]) |
| `gini` ↓ | Gini coefficient of the $sp$ distribution |
| `group_gap` ↓ | Max − min saved fraction across groups |

---

## Datasets

All networks are from [SNAP](https://snap.stanford.edu/data/), treated as undirected, and restricted to their largest connected component.

| Network | File | Nodes (LCC) | Edges (LCC) | $\alpha$ |
|---|---|---:|---:|---:|
| Facebook | `facebook_combined.txt` | 4,039 | 88,234 | 0.20 |
| Arenas Email | `email-Eu-core.txt` | 986 | 16,687 | 0.30 |
| CA-GrQc | `ca-GrQc.txt` | 4,158 | 13,428 | 0.15 |
| CA-HepTh | `ca-HepTh.txt` | 8,638 | 24,827 | 0.15 |
| CA-AstroPh | `ca-AstroPh.txt` | 17,903 | 197,031 | 0.25 |

$|M| = 10$ misinformation seeds by default.

---

## Validation suite

[`tests/test_validation.py`](tests/test_validation.py) checks:

1. $T=\emptyset$ ⇒ `saved_fraction = 0`
2. Monte Carlo $sp$ agrees with the exact value on a path graph (± 3 SE)
3. Distance rule ($p_T = 1$, no delay ⇒ truth wins ties)
4. `maximin_sp` is non-decreasing in $k$
5. Random baseline has maximin welfare ≤ PROTECT
6. Deterministic reproducibility under fixed seeds
7. Prefix-reuse equivalence
8. $sp \in [0, 1]$
9. PageRank / FSPR / LFPR scores form a distribution, never return $M$; exact vs. mean-field fair PageRank correlation > 0.85
10. Regression tests for fixed bugs (prefix matching, positional argument binding)
11. FTRW v1–v3 return valid, prefix-reusable seed lists
12. NSJ-LFPR and the spreadability variants return valid, prefix-reusable seed lists

---

## C++ core (experimental)

[`cpp_core/`](cpp_core/) contains a C++17 diffusion and metrics engine with OpenMP and an optional pybind11 module `fair_spread_cpp`:

```bash
cmake -S cpp_core -B cpp_core/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp_core/build
./cpp_core/build/test_core
```

`fairspread/sim_engine.py` detects the module if it is importable, but the Python pipeline does not dispatch to it yet; all results come from the Numba / Python engine.

---

## Documentation & references

**Project reports** ([`docs/reports/`](docs/reports/))

- `FTRW_Proposed_Method.pdf`: *Fairness-Targeted Random Walk for Negative Influence Minimization: A Unified Framework for Directed and Undirected Social Networks*
- `FTRW_Group_Segregation_Strategies.pdf`: *Group Segregation Strategies for FTRW*
- `Experiment_Design_v3.pdf`: *Fair Truth-Spreader Placement Against Known Misinformation: Experiment Design v3*

**Reference papers** ([`docs/papers/`](docs/papers/))

- D. Unnikrishnan, *Fairness in Negative Influence Minimization in Social Networks*, M.Sc. thesis, TU Eindhoven, 2023
- M. Kumar, G. Dixit, A. Saxena, *Fairness-Sensitive PageRank Approximation* (FSPR)
- M. Kumar, G. Dixit, A. Saxena, *Locally Fair PageRank: Mean-Field Approximation and One-Step Refinement* (LFPR)
- B. Razaghi, M. Roayaei, N. M. Charkari, *On the Group-Fairness-Aware Influence Maximization in Social Networks*, IEEE TCSS 10(6), 2023
- B. Fish et al., *Gaps in Information Access in Social Networks*, WWW 2019
- D. R. Windham et al., *Fast Algorithms to Improve Fair Information Access in Networks*
- K. Mohapatra, H. V. Singh, R. R. Singh, A. Saxena, *Fairness-aware Analysis of Centrality Measures in Social Networks*, CSoNet
- A. Singh, R. R. Singh, S. R. S. Iyengar, *Hybrid Centrality Measures for Service Coverage Problem*, CSoNet 2019
