import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns



def main():
    results_file = Path("results/5fold_results.json")
    if not results_file.exists():
        print(f"Error: {results_file} not found.")
        return

    with open(results_file, "r") as f:
        data = json.load(f)
    
    city_results = data.get("city_level_results", [])
    if len(city_results) == 0:
        print("No city results found in JSON.")
        return

    print(f"Loaded {len(city_results)} cities from {results_file}")
    
    cpc_m0 = []
    cpc_m1 = []
    delta_cpcs = []
    city_names = []

    for res in city_results:
        m0 = res["M0"]["cpc_inter"]
        m1 = res["M1_city_oracle_obs"]["cpc_inter"]  # Using M1_city as primary M1
        delta = m1 - m0
        cpc_m0.append(m0)
        cpc_m1.append(m1)
        delta_cpcs.append(delta)
        city_names.append(res["city"])
        
    cpc_m0 = np.array(cpc_m0)
    cpc_m1 = np.array(cpc_m1)
    delta_cpcs = np.array(delta_cpcs)
    
    # ---------------------------------------------------------
    # 2. STATISTICAL TEST ON 50 CITIES (Wilcoxon Signed-Rank)
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("STATISTICAL TEST RESULTS (N=50 CITIES)")
    print("="*50)
    
    city_stats = data.get("rq1_delta_r", {}).get("city", {})
    if not city_stats:
        print("Missing rq1_delta_r.city in JSON.")
        return
        
    delta_stats = city_stats.get("delta_cpc_inter", {})
    
    n_cities = len(delta_cpcs)
    mean_delta = delta_stats.get("mean", np.mean(delta_cpcs))
    median_delta = delta_stats.get("median", np.median(delta_cpcs))
    ci_lower = delta_stats.get("ci_95_lower", 0.0)
    ci_upper = delta_stats.get("ci_95_upper", 0.0)
    effect_size = city_stats.get("rank_biserial_r", 0.0)
    p_value = city_stats.get("wilcoxon_one_sided_p", 1.0)
    p_value_two_sided = city_stats.get("wilcoxon_two_sided_p", 1.0)
    
    print(f"Number of cities: {n_cities}")
    print(f"Delta CPC (Mean):   {mean_delta:+.4f}")
    print(f"Delta CPC (Median): {median_delta:+.4f}")
    print(f"95% CI (Mean):      [{ci_lower:+.4f}, {ci_upper:+.4f}]")
    print(f"Matched-pairs Rank-biserial (r_rb): {effect_size:.4f}")
    print(f"Wilcoxon p-value (One-sided, M1 > M0): {p_value:.4e}")
    print(f"Wilcoxon p-value (Two-sided):          {p_value_two_sided:.4e}")
    print("="*50)

    # ---------------------------------------------------------
    # 3. STORYBOARD FIGURES
    # ---------------------------------------------------------
    plots_dir = Path("results/q3_figures")
    plots_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")
    
    # Figure 3: Distribution of Delta CPC by city (Sorted Bar Chart)
    sorted_indices = np.argsort(delta_cpcs)
    sorted_deltas = delta_cpcs[sorted_indices]
    sorted_cities = np.array(city_names)[sorted_indices]
    
    plt.figure(figsize=(12, 6))
    colors = ['#d62728' if x < 0 else '#2ca02c' for x in sorted_deltas]
    plt.bar(range(n_cities), sorted_deltas, color=colors, alpha=0.8)
    plt.axhline(0, color='black', linewidth=1.2)
    plt.axhline(mean_delta, color='blue', linestyle='--', linewidth=1.5, label=f'Mean Delta CPC = {mean_delta:.4f}')
    plt.xticks(range(n_cities), sorted_cities, rotation=90, fontsize=8)
    plt.ylabel("Delta CPC (M1 - M0)", fontsize=12)
    plt.title("Figure 3: Distribution of Delta CPC by City", fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "Figure_3_Delta_CPC_Distribution.png", dpi=300)
    plt.close()
    print(f"Saved Figure 3: {plots_dir / 'Figure_3_Delta_CPC_Distribution.png'}")

    # Figure 4: M0 vs M1 across cities
    plt.figure(figsize=(6, 6))
    plt.scatter(cpc_m0, cpc_m1, alpha=0.7, edgecolors='w', s=60, color='#1f77b4')
    
    min_val = min(np.min(cpc_m0), np.min(cpc_m1)) - 0.05
    max_val = max(np.max(cpc_m0), np.max(cpc_m1)) + 0.05
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='y = x')
    
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.xlabel("Zero-shot M0 CPC", fontsize=12)
    plt.ylabel("Support-conditioned M1 CPC", fontsize=12)
    plt.title("Figure 4: M0 vs M1 Performance per City", fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "Figure_4_M0_vs_M1_Scatter.png", dpi=300)
    plt.close()
    print(f"Saved Figure 4: {plots_dir / 'Figure_4_M0_vs_M1_Scatter.png'}")

if __name__ == "__main__":
    main()
