"""
Compare Urban GNN and Pairwise MLP backbones across the locked 5-fold evaluation (N=50 cities).
Reads results from `results/5fold_results.json` and `results/mlp_backbone_results.json`.
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from scipy import stats

def analyze_subset(gnn_map, all_mlp_results, folds_to_include, label):
    paired_results = []
    for m in all_mlp_results:
        c = m.get("city")
        f = m.get("fold")
        if f not in folds_to_include or not c or c not in gnn_map:
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
            "fold": f,
            "gnn_m0": g["m0_cpc_inter"],
            "gnn_m1": g["m1_cpc_inter"],
            "gnn_delta": g["delta_cpc"],
            "mlp_m0": mlp_m0,
            "mlp_m1": mlp_m1,
            "mlp_delta": mlp_delta,
            "gamma": g["delta_cpc"] - mlp_delta
        })

    if not paired_results:
        return None

    def summarize(vals):
        mean_v = float(np.mean(vals))
        median_v = float(np.median(vals))
        sd_v = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

        delta_by_fold = {f: [] for f in folds_to_include}
        for v, r in zip(vals, paired_results):
            delta_by_fold[r["fold"]].append(v)

        rng = np.random.default_rng(42)
        boot_means = []
        for _ in range(10000):
            samp = []
            for f in folds_to_include:
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

    gnn_sum = summarize(gnn_deltas)
    mlp_sum = summarize(mlp_deltas)
    gamma_sum = summarize(gammas)

    _, gnn_w_p = stats.wilcoxon(gnn_deltas, alternative="greater")
    _, mlp_w_p = stats.wilcoxon(mlp_deltas, alternative="greater")
    _, gamma_w_p = stats.wilcoxon(gammas, alternative="two-sided")

    return {
        "label": label,
        "n": len(paired_results),
        "gnn_m0_mean": float(np.mean([r["gnn_m0"] for r in paired_results])),
        "gnn_m1_mean": float(np.mean([r["gnn_m1"] for r in paired_results])),
        "gnn_sum": gnn_sum,
        "gnn_pos": int(np.sum(gnn_deltas > 0)),
        "gnn_p": float(gnn_w_p),
        "mlp_m0_mean": float(np.mean([r["mlp_m0"] for r in paired_results])),
        "mlp_m1_mean": float(np.mean([r["mlp_m1"] for r in paired_results])),
        "mlp_sum": mlp_sum,
        "mlp_pos": int(np.sum(mlp_deltas > 0)),
        "mlp_p": float(mlp_w_p),
        "gamma_sum": gamma_sum,
        "gamma_p": float(gamma_w_p),
    }


def run_comparison(output_dir: str = "results", export_md: bool = True):
    print("\n" + "=" * 85)
    print("COMPARISON: Gravity-Informed Urban GNN vs Pairwise Spatial MLP (Backbone Robustness)")
    print("=" * 85)

    gnn_results_path = Path(output_dir) / "5fold_results.json"
    mlp_results_path = Path(output_dir) / "mlp_backbone_results.json"

    with open(gnn_results_path, "r") as f:
        gnn_json = json.load(f)
        gnn_data = gnn_json.get("city_level_results", [])

    with open(mlp_results_path, "r") as f:
        mlp_json = json.load(f)
        all_mlp_results = mlp_json.get("city_level_results", mlp_json) if isinstance(mlp_json, dict) else mlp_json

    gnn_map = {}
    for r in gnn_data:
        m0_data = r.get("M0")
        m1_data = r.get("M1_city_oracle_obs", r.get("M1_city_oracle_obs"))
        if m0_data and m1_data:
            gnn_map[r["city"]] = {
                "m0_cpc_inter": m0_data.get("cpc_inter", 0.0),
                "m1_cpc_inter": m1_data.get("cpc_inter", 0.0),
                "delta_cpc": m1_data.get("cpc_inter", 0.0) - m0_data.get("cpc_inter", 0.0)
            }

    # part_a = analyze_subset(gnn_map, all_mlp_results, [2, 3, 4, 5], "Part A: Confirmatory Evaluation Set (Folds 2–5, n=40 Cities)")
    part_b = analyze_subset(gnn_map, all_mlp_results, [1, 2, 3, 4, 5], "Five-Fold Cross-City Evaluation Set (All 5 Folds, N=50 Cities)")

    for res in [part_b]:
        if not res:
            continue
        print(f"\n### {res['label']} (N={res['n']} Cities)")
        print(f"Urban GNN:     M0={res['gnn_m0_mean']:.4f} -> M1={res['gnn_m1_mean']:.4f} | dCPC={res['gnn_sum']['mean']:+.4f} +- {res['gnn_sum']['std']:.4f} | 95% CI [{res['gnn_sum']['ci_95'][0]:+.4f}, {res['gnn_sum']['ci_95'][1]:+.4f}] | Pos={res['gnn_pos']}/{res['n']} | p={res['gnn_p']:.2e}")
        print(f"Pairwise MLP:  M0={res['mlp_m0_mean']:.4f} -> M1={res['mlp_m1_mean']:.4f} | dCPC={res['mlp_sum']['mean']:+.4f} +- {res['mlp_sum']['std']:.4f} | 95% CI [{res['mlp_sum']['ci_95'][0]:+.4f}, {res['mlp_sum']['ci_95'][1]:+.4f}] | Pos={res['mlp_pos']}/{res['n']} | p={res['mlp_p']:.2e}")
        print(f"Difference G:  dCPC={res['gamma_sum']['mean']:+.4f} +- {res['gamma_sum']['std']:.4f} | 95% CI [{res['gamma_sum']['ci_95'][0]:+.4f}, {res['gamma_sum']['ci_95'][1]:+.4f}] | p={res['gamma_p']:.2e}")

    if export_md:
        table_path = Path(output_dir) / "tables" / "table_gnn_vs_mlp_comparison.md"
        table_path.parent.mkdir(parents=True, exist_ok=True)
        with open(table_path, "w", encoding="utf-8") as f:
            f.write("# Neural Backbone Comparison: Gravity-Informed Urban GNN vs Pairwise Spatial MLP\n\n")
            f.write("> **Evaluation Goal**: Assesses whether distance-binned aggregate distribution calibration ($Y_D^{\\text{target}}$) provides consistent reconstruction gain across distinct neural architectures (Spatial Graph Convolution vs Local Feature MLP).\n\n")
            
            for res in [part_b]:
                if not res:
                    continue
                f.write(f"## {res['label']}\n\n")
                f.write("| Backbone Architecture | Zero-Shot $M_0$ CPC | Calibrated $M_1$ CPC | Marginal Gain $\\Delta\\text{CPC}$ | 95% Fold-Stratified Bootstrap CI | Improved Cities | Wilcoxon $p$ |\n")
                f.write("|---|:---:|:---:|:---:|:---:|:---:|:---:|\n")
                gnn_mean = res['gnn_sum']['mean']
                gnn_std = res['gnn_sum']['std']
                mlp_mean = res['mlp_sum']['mean']
                mlp_std = res['mlp_sum']['std']
                gam_mean = res['gamma_sum']['mean']
                gam_std = res['gamma_sum']['std']
                f.write(f"| **Gravity-Informed Urban GNN** | {res['gnn_m0_mean']:.4f} | **{res['gnn_m1_mean']:.4f}** | **{gnn_mean:+.4f} +- {gnn_std:.4f}** | [{res['gnn_sum']['ci_95'][0]:+.4f}, {res['gnn_sum']['ci_95'][1]:+.4f}] | {res['gnn_pos']}/{res['n']} ({res['gnn_pos']/res['n']*100:.1f}%) | p = {res['gnn_p']:.2e} |\n")
                f.write(f"| **Pairwise Spatial MLP** | {res['mlp_m0_mean']:.4f} | **{res['mlp_m1_mean']:.4f}** | **{mlp_mean:+.4f} +- {mlp_std:.4f}** | [{res['mlp_sum']['ci_95'][0]:+.4f}, {res['mlp_sum']['ci_95'][1]:+.4f}] | {res['mlp_pos']}/{res['n']} ({res['mlp_pos']/res['n']*100:.1f}%) | p = {res['mlp_p']:.2e} |\n")
                f.write(f"| **Architecture Advantage ($\\Gamma = \\Delta_\\text{{GNN}} - \\Delta_\\text{{MLP}}$)** | — | — | **{gam_mean:+.4f} +- {gam_std:.4f}** | [{res['gamma_sum']['ci_95'][0]:+.4f}, {res['gamma_sum']['ci_95'][1]:+.4f}] | — | p = {res['gamma_p']:.2e} |\n\n")
            
        print(f"\nSaved comparison table to {table_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Urban GNN and Pairwise MLP backbones")
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()
    run_comparison(output_dir=args.output_dir)
