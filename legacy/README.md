# Legacy experiment scripts

Earlier iterations of the experiment pipeline, kept for reproducibility of
intermediate results. **The current pipeline is [`../run_experiments.py`](../run_experiments.py).**

| File | What it is |
|---|---|
| `experiment_skeleton.py` | First end-to-end prototype (self-contained ICM simulator). |
| `experiment_v1.py` | Baseline roster + full metric suite, writes `results_v1/`. |
| `experiment_v2.py` | COICM / MCICM model grid, writes `results/`. |
| `experiment_v3.py` | Experiment Design v3 (E4 homophily sweep, E11 detection delay, E14 prior-work comparison), writes `results_v3/`. |
| `runner.py` | Config-driven runner (reads `config.yaml`) built on the current `fairspread` engine. |
| `aggregator.py` | Merges the leaf JSONs written by `runner.py` into `results/master.json`. |
| `config.yaml` | Network / model / budget configuration used by `runner.py` and `aggregator.py`. |

The `experiment_*` scripts use the pure-Python simulators in
`fairspread/simulators.py` and are much slower than the Numba engine.

Run them from the repository root, e.g.

```bash
python legacy/experiment_v3.py
python legacy/runner.py
```
