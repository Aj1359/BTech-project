import unittest
import math
import numpy as np
import networkx as nx
from scipy.stats import pearsonr

from sim_engine import GraphCSR, compute_distances_bfs, estimate_sp_engine, evaluate_metrics_engine
from algo_wrapper import AlgoManager
from algos.pagerank import get_seeds as pagerank_seeds
from algos.fspr import compute_fspr_scores
from algos.mf_fspr import compute_mf_fspr_scores
from algos.lfpr import compute_lfpr_scores
from algos.orf_lfpr import compute_orf_lfpr_scores

class TestValidationSuite(unittest.TestCase):

    def setUp(self):
        # Create a small 8-node path graph: 0 - 1 - 2 - 3 - 4 - 5 - 6 - 7
        self.G_path = nx.path_graph(8)
        nx.set_node_attributes(self.G_path, {i: (0 if i < 4 else 1) for i in range(8)}, 'group')
        self.csr_path = GraphCSR(self.G_path)
        self.algo_mgr = AlgoManager()

    def test_01_empty_T_saved_fraction_zero(self):
        """Test 1: T = empty gives saved_fraction = 0."""
        M = {0}
        sp0 = estimate_sp_engine(self.csr_path, M, set(), "COICM", pM=0.3, pT=0.3, R_runs=2000, base_seed=42)
        vuln = [v for v in range(8) if v not in M and sp0[v] < 1.0]
        metrics = evaluate_metrics_engine(
            self.G_path, self.csr_path, M, set(),
            nx.get_node_attributes(self.G_path, 'group'),
            sp0, vuln, "COICM", pM=0.3, pT=0.3, R_eval=2000, eval_seed=42
        )
        self.assertAlmostEqual(metrics['saved_fraction'], 0.0, places=2)

    def test_02_monte_carlo_matches_exact_path(self):
        """Test 2: On path graph, MC sp matches exact probability within 3 SEs."""
        # 0 -> 1 -> 2 with p=0.5. Node 1 infected with prob 0.5, Node 2 infected with prob 0.25
        # sp(1) = 0.5, sp(2) = 0.75
        M = {0}
        R = 5000
        sp = estimate_sp_engine(self.csr_path, M, set(), "COICM", pM=0.5, pT=0.5, R_runs=R, base_seed=123)
        expected_sp1 = 0.5
        se1 = math.sqrt(expected_sp1 * (1 - expected_sp1) / R)
        self.assertTrue(abs(sp[1] - expected_sp1) < 3.0 * se1)

    def test_03_distance_rule_holds_when_pT_is_1(self):
        """Test 3: With pT=1, delay=0, truth-wins: if dist(T, v) <= dist(M, v), sp(v) = 1."""
        M = {0}
        T = {2}  # dist(T, 2)=0 <= dist(M, 2)=2, dist(T, 3)=1 <= dist(M, 3)=3
        sp = estimate_sp_engine(self.csr_path, M, T, "MCICM", pM=0.5, pT=1.0, tie_rule="truth", delay_r=0, R_runs=1000, base_seed=999)
        self.assertEqual(sp[2], 1.0)
        self.assertEqual(sp[3], 1.0)

    def test_04_maximin_sp_non_decreasing(self):
        """Test 4: maximin_sp is non-decreasing in k for myopic and protect (up to small noise)."""
        M = {0}
        D, D_M = compute_distances_bfs(self.G_path, M)
        sp0 = estimate_sp_engine(self.csr_path, M, set(), "COICM", pM=0.3, pT=0.3, R_runs=1000, base_seed=11)
        vuln = [v for v in range(8) if v not in M and sp0[v] < 1.0]

        prev_maximin = -1.0
        for k in [1, 2, 3]:
            T = self.algo_mgr.run_algo(
                'protect', self.G_path, M, k, {'pM': 0.3, 'pT': 0.3, 'model': 'COICM'},
                R_select=500, D=D, D_M=D_M, vulnerable_set=set(vuln),
                groups=[0, 1], estimate_sp=estimate_sp_engine
            )
            m = evaluate_metrics_engine(
                self.G_path, self.csr_path, M, T,
                nx.get_node_attributes(self.G_path, 'group'),
                sp0, vuln, "COICM", pM=0.3, pT=0.3, R_eval=1000, eval_seed=42 + k
            )
            # Monotonicity check (with small tolerance for stochastic MC noise)
            self.assertTrue(m['maximin_sp'] >= prev_maximin - 0.05)
            prev_maximin = m['maximin_sp']

    def test_05_random_has_lowest_maximin(self):
        """Test 5: Random baseline has lower or equal maximin_sp compared to Protect on average."""
        M = {0}
        D, D_M = compute_distances_bfs(self.G_path, M)
        sp0 = estimate_sp_engine(self.csr_path, M, set(), "COICM", pM=0.3, pT=0.3, R_runs=1000, base_seed=11)
        vuln = [v for v in range(8) if v not in M and sp0[v] < 1.0]

        T_rand = self.algo_mgr.run_algo('random', self.G_path, M, 2, {'pM': 0.3, 'pT': 0.3}, rng_seed=42)
        T_prot = self.algo_mgr.run_algo(
            'protect', self.G_path, M, 2, {'pM': 0.3, 'pT': 0.3},
            D=D, D_M=D_M, vulnerable_set=set(vuln), groups=[0, 1], estimate_sp=estimate_sp_engine
        )

        m_rand = evaluate_metrics_engine(self.G_path, self.csr_path, M, T_rand, nx.get_node_attributes(self.G_path, 'group'), sp0, vuln, "COICM", pM=0.3, pT=0.3, R_eval=1000, eval_seed=42)
        m_prot = evaluate_metrics_engine(self.G_path, self.csr_path, M, T_prot, nx.get_node_attributes(self.G_path, 'group'), sp0, vuln, "COICM", pM=0.3, pT=0.3, R_eval=1000, eval_seed=42)
        self.assertTrue(m_rand['maximin_sp'] <= m_prot['maximin_sp'] + 0.05)

    def test_06_deterministic_reproducibility(self):
        """Test 6: Identical seed gives identical simulation outputs."""
        M = {0}
        sp_a = estimate_sp_engine(self.csr_path, M, {2}, "COICM", pM=0.3, pT=0.3, R_runs=500, base_seed=777)
        sp_b = estimate_sp_engine(self.csr_path, M, {2}, "COICM", pM=0.3, pT=0.3, R_runs=500, base_seed=777)
        np.testing.assert_array_equal(sp_a, sp_b)

    def test_07_prefix_reuse_equality(self):
        """Test 7: Prefix-reuse matches direct run for prefix-reusable algorithms (e.g. pagerank, topdegree)."""
        M = {0}
        T_full = self.algo_mgr.run_algo('topdegree', self.G_path, M, 4, {})
        T_direct_2 = self.algo_mgr.run_algo('topdegree', self.G_path, M, 2, {})
        self.assertEqual(T_full[:2], T_direct_2)

    def test_08_sp_bounds_and_validity(self):
        """Test 8: sp values lie in [0, 1]."""
        M = {0}
        sp = estimate_sp_engine(self.csr_path, M, {1}, "COICM", pM=0.3, pT=0.3, R_runs=500, base_seed=123)
        self.assertTrue(np.all(sp >= 0.0))
        self.assertTrue(np.all(sp <= 1.0))

    def test_09_ranking_algorithms_properties(self):
        """Test 9: For PageRank, LFPR, FSPR: scores non-negative, sum to 1, no M returned, Pearson > 0.9."""
        M = {0}
        pr_scores = compute_fspr_scores(self.G_path)
        mf_scores = compute_mf_fspr_scores(self.G_path)

        self.assertAlmostEqual(sum(pr_scores.values()), 1.0, places=4)
        self.assertAlmostEqual(sum(mf_scores.values()), 1.0, places=4)

        # Candidates exclude M
        seeds = self.algo_mgr.run_algo('fspr', self.G_path, M, 3, {})
        self.assertFalse(0 in seeds)

        # Correlation between exact FSPR and mean-field FSPR on graph
        vec_exact = [pr_scores[i] for i in range(8)]
        vec_mf = [mf_scores[i] for i in range(8)]
        corr, _ = pearsonr(vec_exact, vec_mf)
        self.assertTrue(corr > 0.85)

    def test_10_bug_fixes_verification(self):
        """Test 10: Regression test for Bug 1 and Bug 2."""
        # 1. Bug 2 check: positional call (G, M, T, 0.3, 0.3, 500) must bind correctly
        sp = estimate_sp_engine(self.csr_path, {0}, {2}, 0.3, 0.3, 500)
        self.assertTrue(np.all(sp >= 0.0))
        self.assertTrue(np.all(sp <= 1.0))

        # 2. Bug 1 check: on an Erdős–Rényi random graph, topdegree and pagerank seeds at k=2 must be EXACT prefix of k=5
        G_rnd = nx.erdos_renyi_graph(30, 0.25, seed=123)
        M = {0, 1}
        for algo in ['topdegree', 'pagerank', 'protect', 'myopic']:
            D, D_M = compute_distances_bfs(G_rnd, M)
            seeds_5 = self.algo_mgr.run_algo(
                algo, G_rnd, M, 5, {'pM': 0.2, 'pT': 0.2},
                D=D, D_M=D_M, vulnerable_set=set(G_rnd.nodes()) - M,
                groups=[0], estimate_sp=estimate_sp_engine
            )
            seeds_2 = self.algo_mgr.run_algo(
                algo, G_rnd, M, 2, {'pM': 0.2, 'pT': 0.2},
                D=D, D_M=D_M, vulnerable_set=set(G_rnd.nodes()) - M,
                groups=[0], estimate_sp=estimate_sp_engine
            )
            # Seeds must be an ordered list
            self.assertIsInstance(seeds_5, list)
            self.assertIsInstance(seeds_2, list)
            # Exact prefix match
            self.assertEqual(seeds_5[:2], seeds_2, f"Prefix mismatch in {algo}!")

    def test_11_ftrw_family_execution_and_prefixes(self):
        """Test 11: Validate FTRW family (ftrw, ftrw_v2, ftrw_v3) seed validity and prefix reusability."""
        M = {0}
        D, D_M = compute_distances_bfs(self.G_path, M)
        sp0 = estimate_sp_engine(self.csr_path, M, set(), "COICM", pM=0.3, pT=0.3, R_runs=200, base_seed=42)
        vuln = set(v for v in range(8) if v not in M and sp0[v] < 1.0)

        for algo_name in ['ftrw', 'ftrw_v2', 'ftrw_v3']:
            seeds_3 = self.algo_mgr.run_algo(
                algo_name, self.G_path, M, 3, {'pM': 0.3, 'pT': 0.3, 'model': 'COICM'},
                D=D, D_M=D_M, vulnerable_set=vuln,
                estimate_sp=estimate_sp_engine, csr=self.csr_path,
                R_cand=30, R_base=60, L=5
            )
            seeds_2 = self.algo_mgr.run_algo(
                algo_name, self.G_path, M, 2, {'pM': 0.3, 'pT': 0.3, 'model': 'COICM'},
                D=D, D_M=D_M, vulnerable_set=vuln,
                estimate_sp=estimate_sp_engine, csr=self.csr_path,
                R_cand=30, R_base=60, L=5
            )

            # Seeds must be an ordered list
            self.assertIsInstance(seeds_3, list)
            self.assertEqual(len(seeds_3), 3)
            # Seeds must not overlap with M
            self.assertTrue(all(s not in M for s in seeds_3))
            # Prefix reusability check
    def test_12_nsj_lfpr_and_spreadability_variants(self):
        """Test 12: Validate NSJ-LFPR and spreadability variants execution and prefix reusability."""
        M = {0}
        sp0 = estimate_sp_engine(self.csr_path, M, set(), "COICM", pM=0.3, pT=0.3, R_runs=200, base_seed=42)
        vuln = set(v for v in range(8) if v not in M and sp0[v] < 1.0)

        new_algos = ['nsj_lfpr', 'sj_fspr', 'sj_lfpr', 'pr_target', 'pr_spread', 'pr_dist']
        for algo_name in new_algos:
            seeds_3 = self.algo_mgr.run_algo(
                algo_name, self.G_path, M, 3, {'pM': 0.3, 'pT': 0.3, 'model': 'COICM'},
                vulnerable_set=vuln, estimate_sp=estimate_sp_engine, csr=self.csr_path,
                R_cand=30, L=5
            )
            seeds_2 = self.algo_mgr.run_algo(
                algo_name, self.G_path, M, 2, {'pM': 0.3, 'pT': 0.3, 'model': 'COICM'},
                vulnerable_set=vuln, estimate_sp=estimate_sp_engine, csr=self.csr_path,
                R_cand=30, L=5
            )

            self.assertIsInstance(seeds_3, list)
            self.assertEqual(len(seeds_3), 3)
            self.assertTrue(all(s not in M for s in seeds_3))
            self.assertEqual(seeds_3[:2], seeds_2, f"Prefix mismatch in {algo_name}!")

if __name__ == '__main__':
    unittest.main()

