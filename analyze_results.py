"""
analyze_results.py -- Fast analysis and comparison tool for completed benchmark runs.
Prints a formatted comparison table ranking all algorithms on Maximin SP, 5th Percentile SP, and Saved Fraction.
"""
import os
import sys
import json
import argparse


def print_summary_table(data):
    """
    Prints a formatted, ranked leaderboard from a summary dictionary.
    Highlights our proposed algorithms and PROTECT baseline.
    """
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    net_name = data.get("network", "Unknown")
    model = data.get("model_family", "COICM")
    prob_setting = data.get("prob_setting", "")
    m_mode = data.get("m_mode", "highdeg")
    budgets = data.get("budgets", [5, 10, 15, 20, 25])
    sp_vs_k = data.get("sp_vs_k", {})
    p05_vs_k = data.get("p05_vs_k", {})
    saved_vs_k = data.get("saved_frac_vs_k", {})
    n_vuln = data.get("n_vulnerable", 0)

    print("\n" + "=" * 105)
    print(f"   [ANALYSIS REPORT] NETWORK: {net_name} | MODEL: {model} | SETTING: {prob_setting} | M: {m_mode} | Vuln: {n_vuln}")
    print("=" * 105)

    if not sp_vs_k:
        print("   [Warning] No algorithm data recorded in summary.")
        print("=" * 105)
        return

    max_k = max(budgets) if budgets else 25
    algos = sorted(sp_vs_k.keys())

    # Sort algorithms by primary objective (Maximin SP at max k), then b5%, then saved fraction
    def sort_key(algo):
        m_sp = sp_vs_k.get(algo, {}).get(str(max_k), sp_vs_k.get(algo, {}).get(max_k, 0.0))
        p05 = p05_vs_k.get(algo, {}).get(str(max_k), p05_vs_k.get(algo, {}).get(max_k, 0.0))
        sf = saved_vs_k.get(algo, {}).get(str(max_k), saved_vs_k.get(algo, {}).get(max_k, 0.0))
        return (m_sp, p05, sf)

    sorted_algos = sorted(algos, key=sort_key, reverse=True)

    header = f"{'Rank':<5} | {'Algorithm':<22} | {'MinSP (k=5)':<11} | {f'MinSP (k={max_k})':<12} | {f'b5% (k={max_k})':<11} | {f'Saved% (k={max_k})':<13} | Type"
    print(header)
    print("-" * len(header))

    protect_min = None
    ours_best_min = -1.0
    ours_best_algo = None

    for rank, algo in enumerate(sorted_algos, 1):
        s_k5 = sp_vs_k.get(algo, {}).get("5", sp_vs_k.get(algo, {}).get(5, 0.0))
        s_max = sp_vs_k.get(algo, {}).get(str(max_k), sp_vs_k.get(algo, {}).get(max_k, 0.0))
        p_max = p05_vs_k.get(algo, {}).get(str(max_k), p05_vs_k.get(algo, {}).get(max_k, 0.0))
        sv_max = saved_vs_k.get(algo, {}).get(str(max_k), saved_vs_k.get(algo, {}).get(max_k, 0.0))

        tag = "Baseline"
        if algo in ['nsj_lfpr', 'ftrw', 'ftrw_v2', 'ftrw_v3', 'sj_lfpr', 'sj_fspr', 'pr_target', 'pr_spread', 'pr_dist']:
            tag = "* [OURS]"
            if s_max > ours_best_min:
                ours_best_min = s_max
                ours_best_algo = algo
        elif algo == 'protect':
            tag = "<-- [PROTECT]"
            protect_min = s_max

        print(f"#{rank:<4} | {algo:<22} | {s_k5:>10.4f}  | {s_max:>11.4f}  | {p_max:>10.4f}  | {sv_max*100:>11.1f}%  | {tag}")

    print("-" * len(header))
    if protect_min is not None and ours_best_algo is not None:
        if ours_best_min > protect_min:
            print(f"   >>> WIN: Best proposed ({ours_best_algo}: {ours_best_min:.4f}) beats PROTECT ({protect_min:.4f}) by +{(ours_best_min - protect_min):.4f} on Maximin SP!")
        elif ours_best_min == protect_min:
            print(f"   >>> TIED: Best proposed ({ours_best_algo}: {ours_best_min:.4f}) tied PROTECT ({protect_min:.4f}) on Maximin SP.")
        else:
            print(f"   >>> PROTECT lead: {protect_min:.4f} vs {ours_best_algo}: {ours_best_min:.4f} (gap: {(protect_min - ours_best_min):.4f}).")
    print("=" * 105 + "\n")


def analyze_network(results_dir="results", model="COICM", network="Facebook"):
    net_path = os.path.join(results_dir, model, network)
    if not os.path.exists(net_path):
        print(f"[Error] Directory not found: {net_path}")
        return

    found_summaries = []
    for root, dirs, files in os.walk(net_path):
        if "summary.json" in files:
            found_summaries.append(os.path.join(root, "summary.json"))

    if not found_summaries:
        print(f"[Warning] No summary.json found in {net_path}")
        return

    for sum_path in found_summaries:
        with open(sum_path, 'r') as f:
            data = json.load(f)
        print_summary_table(data)


if __name__ == "__main__":
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description="Analyze and rank experiment results")
    parser.add_argument("--network", default="Facebook", help="Network name (e.g. Facebook, CA-GrQc, Arenas_email)")
    parser.add_argument("--model", default="COICM", help="Diffusion model (COICM or MCICM)")
    parser.add_argument("--results_dir", default="results", help="Directory where results are stored")
    args = parser.parse_args()

    analyze_network(results_dir=args.results_dir, model=args.model, network=args.network)
