"""fairspread: core library for the Fair Truth-Spreader Placement benchmark.

Modules
-------
sim_engine    Numba-accelerated competing-cascade ICM engine (GraphCSR, CRN Monte Carlo, metrics)
algo_wrapper  Registry + uniform interface for the seed-selection algorithms in ../algos
placem        Misinformation seed (M) placement strategies
simulators    Reference pure-Python cascade simulators (used by placem and legacy scripts)
metrics       Fairness / welfare metric helpers (Gini, DP gap, inequality aversion, ...)
"""
