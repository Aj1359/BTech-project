import os
import json
import yaml
import pandas as pd

def build_master_json(results_dir="results", config_path="config.yaml"):
    print("Building master.json from leaf JSONs...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    master_data = {
        "config": config,
        "networks": {},
        "rows": []
    }

    network_summary_rows = []

    for net_dir in os.listdir(results_dir):
        net_path = os.path.join(results_dir, net_dir)
        if not os.path.isdir(net_path):
            continue

        meta_file = os.path.join(net_path, "network_meta.json")
        if os.path.exists(meta_file):
            with open(meta_file, 'r') as f:
                net_meta = json.load(f)
                master_data["networks"][net_dir] = net_meta

        for root, dirs, files in os.walk(net_path):
            if "M.json" in files:
                m_file = os.path.join(root, "M.json")
                try:
                    with open(m_file, 'r') as f:
                        m_data = json.load(f)
                        baseline = m_data.get('baseline', {})
                        network_summary_rows.append({
                            "Network": net_dir,
                            "Placement_Folder": os.path.basename(root),
                            "Outreach_Empty": f"{baseline.get('outreach_empty', 0.0) * 100:.1f}%",
                            "Vulnerable_Set_Size": baseline.get('n_vulnerable', 0),
                            "Group_Vulnerable": baseline.get('vulnerable_per_group', {})
                        })
                except Exception:
                    pass

            for f_name in files:
                if f_name.endswith('.json') and f_name not in ["network_meta.json", "M.json", "master.json"]:
                    if os.path.basename(root) == "seeds":
                        continue
                    leaf_path = os.path.join(root, f_name)
                    try:
                        with open(leaf_path, 'r') as lf:
                            leaf = json.load(lf)
                            if leaf.get("status") == "ok":
                                # Exclude T and sp_hist20 as required
                                row = {k: v for k, v in leaf.items() if k not in ["T"]}
                                if "metrics" in row and "sp_hist20" in row["metrics"]:
                                    row["metrics"] = {mk: mv for mk, mv in row["metrics"].items() if mk != "sp_hist20"}
                                master_data["rows"].append(row)
                    except Exception:
                        pass

    out_master_path = os.path.join(results_dir, "master.json")
    with open(out_master_path, 'w') as f:
        json.dump(master_data, f, indent=2)

    print(f"master.json built successfully! Total evaluated records: {len(master_data['rows'])}")

    # Print short summary table per network
    print("\n" + "=" * 80)
    print("NETWORK SUMMARY SANITY CHECK (Outreach T=empty, Vulnerable Set Size, Group Counts)")
    print("=" * 80)
    if network_summary_rows:
        df_summary = pd.DataFrame(network_summary_rows)
        # Deduplicate to show clean summary
        print(df_summary.drop_duplicates(subset=["Network", "Placement_Folder"]).to_string(index=False))
    else:
        print("No M.json files found yet.")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    build_master_json()
