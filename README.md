# Group-Fairness-Aware Truth Spreading Experiments

This repository contains the simulation framework and algorithms for evaluating truth-spreading interventions against known misinformation in social networks. The focus is on analyzing the fairness of these interventions—both individual and group fairness—over time.

## Project Structure

- **`experiment_skeleton.py`**: The main evaluation loop. It sets up the graph (synthetic or real-world), selects misinformation seeds, and sweeps across different parameters (like budget `k`, and diffusion probabilities `pM`, `pT`). It dynamically loads any algorithm placed in the `algos/` folder and evaluates its performance across the Campaign-Oblivious and Misinformation-Aware Independent Cascade Models (COICM / MCICM).
- **`algos/`**: This directory contains the individual algorithmic strategies for truth seeding. Each script exposes a standard `get_seeds(...)` function.
  - `random_baseline.py`: A lower-bound baseline that picks nodes randomly.
  - `top_degree.py`: A fairness-agnostic baseline prioritizing highly connected nodes.
  - `myopic_welfare_maximin.py`: An individual-fair baseline that repeatedly picks the node with the lowest save probability.
  - `protect.py`: The individual-fair heuristic targeting maximin save probability through shortest-path enhancer logic.
  - `group_aware_protect.py`: The custom, group-aware, time-aware method that boosts the worst-off group based on relative risk reduction and arrival times.
- **`download_networks.py`**: A helper script to automatically download and extract standard real-world evaluation datasets from SNAP.
- **`networks/`**: The local storage for real-world datasets (e.g., `facebook_combined.txt`, `ca-GrQc.txt`, `ca-HepTh.txt`, `email-Eu-core.txt`).

## Adding New Algorithms

The `experiment_skeleton.py` script uses dynamic loading. To add a new algorithm to the evaluation pipeline:
1. Create a new `.py` file inside the `algos/` directory.
2. Implement a function named `get_seeds(G, M, budget, **kwargs)`.
3. Ensure it returns a `set()` of the selected seed nodes. 

The main experiment script will automatically detect the new file, run it across all parameters, and report its evaluation metrics!

## Running the Experiments

1. (Optional) Run `python download_networks.py` to fetch the real-world datasets.
2. Run `python experiment_skeleton.py` to execute the full evaluation suite across the defined parameter sweeps and time units.
