"""
Compare Urban GNN and Pairwise MLP backbones on Folds 2-5.
Reads results from `results/5fold_results.json` and `results/mlp_backbone_results.json`.
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from scipy import stats

def run_comparison(output_dir: str = "results", export_md: bool = True):
    print("\n" + "=" * 85)
    print("COMPARISON: Urban GNN vs Pairwise MLP (Backbone Robustness)")
    print("=" * 85)

    gnn_results_path = Path(output_dir) / "5fold_results.json"
    if not gnn_results_path.exists():
        print(f"ERROR: {gnn_results_path} not found. Please ensure the Urban GNN 5-fold test has been run first.")
        return

    mlp_results_path = Path(output_dir) / "mlp_backbone_results.json"
    if not mlp_results_path.exists():
        print(f"ERROR: {mlp_results_path} not found. Please ensure the MLP backbone test has been run first.")
        return

    with open(gnn_results_path, "r") as f:
        gnn_json = json.load(f)
        gnn_data = gnn_json.get("city_level_results")
        if not gnn_data:
            print("ERROR: 'city_level_results' not found in GNN results file.")
            return

    with open(mlp_results_path, "r") as f:
        mlp_json = json.load(f)
        if isinstance(mlp_json, dict) and "city_level_results" in mlp_json:
            all_mlp_results = mlp_json["city_level_results"]
        else:
            all_mlp_results = mlp_json

    folds_to_run = [2, 3, 4, 5]

    # Map GNN results
    gnn_map = {}
    for r in gnn_data:
        if r.get("fold") in folds_to_run:
            m0_data = r.get("M0")
            m1_data = r.get("M1_city_oracle_obs", r.get("M1_real_plus"))
            if m0_data and m1_data:
                gnn_map[r["city"]] = {
                    "m0_cpc_inter": m0_data.get("cpc_inter", 0.0),
                    "m1_cpc_inter": m1_data.get("cpc_inter", 0.0),
                    "delta_cpc": m1_data.get("cpc_inter", 0.0) - m0_data.get("cpc_inter", 0.0)
                }

    # Compile paired results
    paired_results = []
    for m in all_mlp_results:
        c = m.get("city")
        if not c or c not in gnn_map:
            continue
        g = gnn_map[c]
        
        if "M0" in m and "M1_city_oracle_obs" in m:
            mlp_m0 = m["M0"].get("cpc_inter", 0.0)
            mlp_m1 = m["M1_city_oracle_obs"].get("cpc_inter", 0.0)
            mlp_delta = mlp_m1 - mlp_m0
        else:
            mlp_m0 = m.get("m0_cpc_inter", 0.0)
            mlp_m1 = m.get("m1_cpc_inter", 0.0)
            mlp_delta = m.get("delta_cpc", 0.0)
            
        paired_results.append({
            "city": c,
            "fold": m.get("fold"),
            "gnn_m0": g["m0_cpc_inter"],
            "gnn_m1": g["m1_cpc_inter"],
            "gnn_delta": g["delta_cpc"],
            "mlp_m0": mlp_m0,
            "mlp_m1": mlp_m1,
            "mlp_delta": mlp_delta,
            "gamma": g["delta_cpc"] - mlp_delta
        })

    if not paired_results:
        print("ERROR: No matching cities found between GNN and MLP results.")
        return

    def summarize(vals, label):
        mean_v = float(np.mean(vals))
        median_v = float(np.median(vals))
        sd_v = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

        delta_by_fold = {f: [] for f in folds_to_run}
        for v, r in zip(vals, paired_results):
            delta_by_fold[r["fold"]].append(v)

        rng = np.random.default_rng(42)
        boot_means = []
        for _ in range(5000):
            samp = []
            for f in folds_to_run:
                fold_vals = delta_by_fold[f]
                if fold_vals:
                    samp.extend(rng.choice(fold_vals, size=len(fold_vals), replace=True))
            boot_means.append(np.mean(samp) if samp else 0.0)
        ci_l, ci_h = np.percentile(boot_means, [2.5, 97.5])

        return {
            "mean": mean_v,
            "std": sd_v,
            "median": median_v,
            "ci_95": (float(ci_l), float(ci_h))
        }

    gnn_deltas = np.array([r["gnn_delta"] for r in paired_results])
    mlp_deltas = np.array([r["mlp_delta"] for r in paired_results])
    gammas = np.array([r["gamma"] for r in paired_results])

    gnn_sum = summarize(gnn_deltas, "GNN Delta CPC")
    mlp_sum = summarize(mlp_deltas, "MLP Delta CPC")
    gamma_sum = summarize(gammas, "Gamma (GNN - MLP)")

    # Paired Tests
    _, gnn_w_p = stats.wilcoxon(gnn_deltas, alternative="greater")
    _, mlp_w_p = stats.wilcoxon(mlp_deltas, alternative="greater")
    _, gamma_w_p = stats.wilcoxon(gammas, alternative="two-sided")

    print(f"\n[Urban GNN Backbone] (n={len(paired_results)})")
    print(f"Mean M0: {np.mean([r['gnn_m0'] for r in paired_results]):.4f}")
    print(f"Mean M1: {np.mean([r['gnn_m1'] for r in paired_results]):.4f}")
    print(f"Mean Delta: {gnn_sum['mean']:+.4f} +- {gnn_sum['std']:.4f} | Median: {gnn_sum['median']:+.4f}")
    print(f"95% CI: [{gnn_sum['ci_95'][0]:+.4f}, {gnn_sum['ci_95'][1]:+.4f}]")
    print(f"Cities Improved: {np.sum(gnn_deltas > 0)}/{len(paired_results)} ({np.mean(gnn_deltas > 0)*100:.1f}%)")
    print(f"Wilcoxon (Delta > 0): p = {gnn_w_p:.4e}")

    print(f"\n[Pairwise Spatial MLP Backbone] (n={len(paired_results)})")
    print(f"Mean M0: {np.mean([r['mlp_m0'] for r in paired_results]):.4f}")
    print(f"Mean M1: {np.mean([r['mlp_m1'] for r in paired_results]):.4f}")
    print(f"Mean Delta: {mlp_sum['mean']:+.4f} +- {mlp_sum['std']:.4f} | Median: {mlp_sum['median']:+.4f}")
    print(f"95% CI: [{mlp_sum['ci_95'][0]:+.4f}, {mlp_sum['ci_95'][1]:+.4f}]")
    print(f"Cities Improved: {np.sum(mlp_deltas > 0)}/{len(paired_results)} ({np.mean(mlp_deltas > 0)*100:.1f}%)")
    print(f"Wilcoxon (Delta > 0): p = {mlp_w_p:.4e}")

    print("\n[Backbone Dependence Gamma = Delta_GNN - Delta_MLP]")
    print(f"Mean Gamma: {gamma_sum['mean']:+.4f} +- {gamma_sum['std']:.4f} | Median: {gamma_sum['median']:+.4f}")
    print(f"95% CI: [{gamma_sum['ci_95'][0]:+.4f}, {gamma_sum['ci_95'][1]:+.4f}]")
    print(f"Wilcoxon (Two-Sided): p = {gamma_w_p:.4e}")

    print("\n[Interpretation]")
    gnn_robust = gnn_sum['mean'] > 0 and gnn_w_p < 0.05
    mlp_robust = mlp_sum['mean'] > 0 and mlp_w_p < 0.05

    if gnn_robust and mlp_robust:
        print(r"=> Cả GNN và MLP đều cải thiện: Lợi ích của Y_D là robust across the two tested neural backbones.")
    elif gnn_robust and not mlp_robust:
        print(r"=> Chỉ GNN cải thiện: Đóng góp của Y_D phụ thuộc vào kiến trúc đồ thị.")
    elif not gnn_robust and mlp_robust:
        print(r"=> Chỉ MLP cải thiện.")
    else:
        print(r"=> Cả GNN và MLP đều không cho thấy sự cải thiện đáng kể.")

    if export_md:
        table_path = Path(output_dir) / "tables" / "table_gnn_vs_mlp_comparison.md"
        table_path.parent.mkdir(parents=True, exist_ok=True)
        with open(table_path, "w", encoding="utf-8") as f:
            f.write("# Neural Backbone Comparison: Urban GNN vs Pairwise MLP\n\n")
            f.write("| Backbone Architecture | Zero-Shot $M_0$ CPC | Calibrated $M_1$ CPC | Marginal Gain $\\Delta\\text{CPC}$ | 95% Bootstrap CI | Improved Cities | Wilcoxon $p$ |\n")
            f.write("|---|---|---|---|---|---|---|\n")
            f.write(f"| **Gravity-Informed Urban GNN** | {np.mean([r['gnn_m0'] for r in paired_results]):.4f} | **{np.mean([r['gnn_m1'] for r in paired_results]):.4f}** | **{gnn_sum['mean']:+.4f} +- {gnn_sum['std']:.4f}** | [{gnn_sum['ci_95'][0]:+.4f}, {gnn_sum['ci_95'][1]:+.4f}] | {np.sum(gnn_deltas > 0)}/{len(paired_results)} ({np.mean(gnn_deltas > 0)*100:.1f}%) | p = {gnn_w_p:.4e} |\n")
            f.write(f"| **Pairwise Spatial MLP** | {np.mean([r['mlp_m0'] for r in paired_results]):.4f} | **{np.mean([r['mlp_m1'] for r in paired_results]):.4f}** | **{mlp_sum['mean']:+.4f} +- {mlp_sum['std']:.4f}** | [{mlp_sum['ci_95'][0]:+.4f}, {mlp_sum['ci_95'][1]:+.4f}] | {np.sum(mlp_deltas > 0)}/{len(paired_results)} ({np.mean(mlp_deltas > 0)*100:.1f}%) | p = {mlp_w_p:.4e} |\n")
            f.write(f"| **Difference ($\\Gamma$)** | — | — | **{gamma_sum['mean']:+.4f} +- {gamma_sum['std']:.4f}** | [{gamma_sum['ci_95'][0]:+.4f}, {gamma_sum['ci_95'][1]:+.4f}] | — | p = {gamma_w_p:.4e} |\n")
        print(f"Comparison table exported to {table_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Urban GNN and Pairwise MLP backbones")
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()
    run_comparison(output_dir=args.output_dir)
