import importlib.util
import os
import random
import networkx as nx

ALGO_FILES = {
    'random': 'random_baseline.py',
    'topdegree': 'top_degree.py',
    'cmia_o': 'cmia_o.py',
    'naive_myopic': 'naive_myopic.py',
    'myopic': 'myopic_welfare_maximin.py',
    'protect': 'protect.py',
    'naive_protect': 'naive_protect.py',
    'fwrrs': 'fwrrs.py',
    'pagerank': 'pagerank.py',
    'lfpr': 'lfpr.py',
    'fspr': 'fspr.py',
    'proposed': 'group_aware_protect.py',
    'greedy_maximin': 'greedy_welfare_maximin.py',
    'gonzalez': 'gonzalez.py',
    'mindegree_hcn': 'mindegree_hcn.py',
    'myopic_bfs': 'myopic_bfs.py',
    'myopic_ppr': 'myopic_ppr.py',
    'leastcentral': 'least_central.py',
    'betweenness': 'centralities.py',
    'habc_vuln': 'habc_eabc_vuln.py',
    'group_maximin_greedy': 'group_maximin_greedy.py',
    'dp_aware_fibm': 'dp_aware_fibm.py',
    'mf_fspr': 'mf_fspr.py',
    'orf_lfpr': 'orf_lfpr.py',
    'ftrw': 'ftrw.py',
    'ftrw_v2': 'ftrw_v2.py',
    'ftrw_v3': 'ftrw_v3.py',
    'nsj_lfpr': 'nsj_lfpr.py',
    'sj_fspr': 'sj_fspr.py',
    'sj_lfpr': 'sj_lfpr.py',
    'pr_target': 'pr_target.py',
    'pr_spread': 'pr_spread.py',
    'pr_dist': 'pr_dist.py'
}

# Algorithms whose seed selection is completely model-agnostic (independent of pM, pT)
MODEL_AGNOSTIC_ALGOS = {
    'random', 'topdegree', 'naive_protect', 'gonzalez',
    'mindegree_hcn', 'leastcentral', 'betweenness', 'habc_vuln',
    'pagerank', 'lfpr', 'fspr', 'mf_fspr', 'orf_lfpr'
}

# Algorithms where prefix-reuse is strictly valid (seeds for k < k_max are prefix of k_max)
# Making greedy/ranking algorithms prefix-reusable enforces nested seeds and avoids redundant computation
PREFIX_REUSABLE_ALGOS = {
    'myopic', 'protect', 'greedy_maximin', 'naive_myopic',
    'gonzalez', 'pagerank', 'lfpr', 'fspr', 'mf_fspr', 'orf_lfpr',
    'topdegree', 'betweenness', 'leastcentral', 'naive_protect',
    'habc_vuln', 'random', 'proposed', 'fwrrs',
    'group_maximin_greedy', 'dp_aware_fibm', 'mindegree_hcn', 'myopic_bfs',
    'ftrw', 'ftrw_v2', 'ftrw_v3',
    'nsj_lfpr', 'sj_fspr', 'sj_lfpr', 'pr_target', 'pr_spread', 'pr_dist'
}

class AlgoManager:
    def __init__(self, algos_dir='algos'):
        self.algos_dir = algos_dir
        self.loaded_funcs = {}
        self._load_all()

    def _load_all(self):
        for algo_id, filename in ALGO_FILES.items():
            path = os.path.join(self.algos_dir, filename)
            if os.path.exists(path):
                spec = importlib.util.spec_from_file_location(algo_id, path)
                mod = importlib.util.module_from_spec(spec)
                try:
                    spec.loader.exec_module(mod)
                    if hasattr(mod, 'get_seeds'):
                        self.loaded_funcs[algo_id] = mod.get_seeds
                except Exception as e:
                    print(f"[Warning] Could not load {algo_id}: {e}")

    def run_algo(self, algo_id, G, M, k, model_setting, R_select=1000, rng_seed=12345, **extra_kwargs):
        """
        Standard interface:
        algo(G, M, k, model_setting, R_select, rng_seed) -> list of nodes
        """
        if algo_id not in self.loaded_funcs:
            raise KeyError(f"Algorithm '{algo_id}' is not loaded or missing.")

        random.seed(rng_seed)
        pM = model_setting.get('pM', 0.1)
        pT = model_setting.get('pT', 0.1)
        model = model_setting.get('model', 'COICM')
        delay = model_setting.get('delay', 0)
        tie_rule = model_setting.get('tie_rule', 'truth')

        func = self.loaded_funcs[algo_id]
        kwargs = {
            'pM': pM,
            'pT': pT,
            'model': model,
            'delay': delay,
            'tie_rule': tie_rule,
            'R': R_select,
            'epsilon': 0.05,
            'lam': 0.5,
            **extra_kwargs
        }

        # Special kwargs for betweenness
        if algo_id == 'betweenness':
            kwargs['measure'] = 'betweenness'

        seeds = func(G, M, k, **kwargs)
        # Ensure T is a list strictly preserving selection order, excluding M and duplicates
        seed_list = []
        for v in seeds:
            if v not in M and v not in seed_list:
                seed_list.append(v)
        if len(seed_list) > k:
            seed_list = seed_list[:k]
        return seed_list
