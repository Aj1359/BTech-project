# Theoretical Formulation: Fairness-Targeted Random Walk (FTRW)

## 1. Problem Formulation and Welfare Objective

Given:
- A graph $G = (V, E)$
- A fixed set of misinformation seeds $M \subset V$ (e.g. placed on the highest-degree nodes)
- Independent Cascade Model parameters with competing cascades (COICM / MCICM)
- A truth seed budget $k$

The save probability $sp(v; T)$ of node $v$ is the probability that $v$ adopts the truth cascade $T$ or remains uninfected by misinformation $M$.

### Maximin Welfare Function
The objective is to maximize the minimum save probability across all vulnerable nodes:
$$\max_{T \subseteq V \setminus M, |T| \le k} \min_{v \in V_{\text{vuln}}} sp(v; T)$$
where $V_{\text{vuln}} = \{v \in V \setminus M : sp(v; \emptyset) < 1.0\}$.

---

## 2. Dynamic Group Partitions

Unlike static group fairness formulations, the FTRW family partitions nodes dynamically at each step $t \in \{1, \dots, k\}$:
- **Protected Group ($P_t$)**: Nodes in the lower tail of the current save probability distribution:
  $$P_t = \{ v \in V_{\text{vuln}} : sp_t(v) \le \text{quantile}(sp_t, p_{\text{frac}}) \}$$
- **Unprotected Group ($U_t$)**: The remaining vulnerable nodes:
  $$U_t = V_{\text{vuln}} \setminus P_t$$

As truth seeds are sequentially committed, protected nodes become saved, and the identity of the most disadvantaged group shifts dynamically.

---

## 3. Algorithm Variants

### 3.1 FTRW-v1: Dynamic Group Walk
- **Teleportation / Personalization**: Biased toward $P_t$ with deficit and distance weighting:
  $$h(v) = (1 - sp_t(v)) \cdot \left[\eta + (1-\eta) e^{-\text{dist}(M, v)/\tau}\right]$$
- **LFPR Transition**: Group-normalized transitions ensure walks give proportional influence back to $P_t$.
- **Shortlist & CRN Evaluation**: Top-$L$ candidates evaluated under Common Random Numbers (CRN) Monte Carlo.

### 3.2 FTRW-v2: Reverse-Rescue Dual Walk
- **Reverse Random Walk**: Walks backwards along incoming paths from the worst-off nodes to locate upstream truth propagators.
- **Neighbor Deficit Scoring**: Direct reward for candidates adjacent to dense clusters of vulnerable nodes.
- **Adaptive Bias ($\phi$)**: Increases focus on $P_t$ when the number of unsaved nodes is large, automatically softening as coverage expands.
- **Dispersion Regularizer**: Discourages placing seeds adjacent to existing truth seeds.

### 3.3 FTRW-v3: Multi-Resolution Shield Walk
- **Multi-Resolution Profiling**: Runs walks across multiple quantile thresholds ($p_{\text{frac}} \in \{0.05, 0.20, 0.50\}$) to capture both deep rescue nodes and general boundary protectors.
- **Shield Metric**: Intercepts shortest paths from $M$ to $V_{\text{vuln}}$, halting misinformation propagation early.
- **Degree-Deficit Fusion**: Balances hub broadcast power with local need.
