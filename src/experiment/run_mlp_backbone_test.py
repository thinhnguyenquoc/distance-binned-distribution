"""
Backbone Robustness Evaluation Experiment (Urban GNN vs Pairwise MLP).
"""

import os
import sys
import json
import time
import argparse
import torch
import numpy as np
from pathlib import Path
from scipy import stats
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.city_splits import generate_35_5_10_splits
from src.data.yd_extractor import compute_kbin_edges
from src.training.train import train_zero_shot_model
from src.experiment.run_experiment import run_target_city_experiments

def run_mlp_backbone_test(
    data_root: str = "data",
    output_dir: str = "results",
    epochs_per_fold: int = 200,
    lr: float = 3.2e-3,
    hidden_dim: int = 64,
    num_gnn_layers: int = 2,
    graph_type: str = "radius",
    radius_km: float = 5.0,
    knn_k: int = 10,
    loss_type: str = "ztnb",
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    os.makedirs(output_dir, exist_ok=True)
    splits = generate_35_5_10_splits(data_root=data_root)
    folds_to_run = [2, 3, 4, 5]
    seeds = [1, 10, 100]

    all_mlp_results = []
    
    # 1. Train and evaluate Pairwise MLP
    print("=" * 85)
    print("STARTING MLP BACKBONE TRAINING & EVALUATION")
    print(f"Folds: {folds_to_run} | Seeds: {seeds}")
    print("=" * 85)
    
    for fold_id in folds_to_run:
        split = splits[fold_id]
        train_cities = split["train"]
        val_cities = split["val"]
        test_cities = split["test"]

        print(f"\n# FOLD {fold_id}/5 (Test cities: {len(test_cities)})")
        models = []
        scalers = []
        
        for seed_idx, seed in enumerate(seeds):
            print(f"--- Training MLP Seed {seed} ---")
            _ckpt_path = Path(output_dir) / "checkpoints" / f"mlp_fold{fold_id}_seed{seed}.pt"
            
            model, scaler = train_zero_shot_model(
                train_city_names=train_cities,
                data_root=data_root,
                epochs=epochs_per_fold,
                lr=lr,
                hidden_dim=hidden_dim,
                num_gnn_layers=num_gnn_layers,
                graph_type=graph_type,
                radius_km=radius_km,
                knn_k=knn_k,
                loss_type=loss_type,
                backbone="mlp",  # <--- MLP Backbone
                device_str=device_str,
                verbose=False,
                val_city_names=val_cities,
                patience=16,
                checkpoint_path=_ckpt_path,
                run_tag=f"mlp_fold{fold_id}_seed{seed}",
                seed=seed,
            )
            models.append(model)
            scalers.append(scaler)
            
        bin_edges, K_active = compute_kbin_edges(train_cities, K=8, data_root=data_root)

        # Target City Evaluation
        for target_city in test_cities:
            seed_results = []
            for seed_idx, model in enumerate(models):
                scaler = scalers[seed_idx]
                res = run_target_city_experiments(
                    model=model,
                    city_name=target_city,
                    scaler=scaler,
                    data_root=data_root,
                    meta_prior_dir="meta_prior",
                    graph_type=graph_type,
                    radius_km=radius_km,
                    knn_k=knn_k,
                    num_trip_seeds=20, # dummy not used for kbins
                    device_str=device_str,
                    bin_edges=bin_edges,
                )
                seed_results.append(res)
                
            # Average over seeds
            avg_res = seed_results[0].copy()
            for key in ["M0", "M1_city_oracle_obs"]:
                if avg_res[key] is not None:
                    avg_res[key] = avg_res[key].copy()
                    for metric in ["cpc_inter"]:
                        avg_res[key][metric] = sum(r[key][metric] for r in seed_results) / len(seed_results)
            
            city_res = {
                "city": target_city,
                "fold": fold_id,
                "m0_cpc_inter": avg_res["M0"]["cpc_inter"],
                "m1_cpc_inter": avg_res["M1_city_oracle_obs"]["cpc_inter"],
                "delta_cpc": avg_res["M1_city_oracle_obs"]["cpc_inter"] - avg_res["M0"]["cpc_inter"]
            }
            all_mlp_results.append(city_res)
            print(f"  {target_city:15s} | M0: {city_res['m0_cpc_inter']:.4f} | M1: {city_res['m1_cpc_inter']:.4f} | d={city_res['delta_cpc']:+.4f}")
            
    # Save MLP results
    with open(Path(output_dir) / "mlp_backbone_results.json", "w") as f:
        json.dump(all_mlp_results, f, indent=2)

    # 2. Compare with Urban GNN
    print("\n" + "=" * 85)
    print("COMPARISON: GNN vs MLP")
    print("=" * 85)
    
    with open(Path(output_dir) / "5fold_results.json", "r") as f:
        gnn_data = json.load(f)["city_level_results"]
    
    # Map GNN results
    gnn_map = {}
    for r in gnn_data:
        if r["fold"] in folds_to_run:
            gnn_map[r["city"]] = {
                "m0_cpc_inter": r["M0"]["cpc_inter"],
                "m1_cpc_inter": r["M1_city_oracle_obs"]["cpc_inter"],
                "delta_cpc": r["M1_city_oracle_obs"]["cpc_inter"] - r["M0"]["cpc_inter"]
            }

    # Compile paired results
    paired_results = []
    for m in all_mlp_results:
        c = m["city"]
        g = gnn_map[c]
        paired_results.append({
            "city": c,
            "fold": m["fold"],
            "gnn_m0": g["m0_cpc_inter"],
            "gnn_m1": g["m1_cpc_inter"],
            "gnn_delta": g["delta_cpc"],
            "mlp_m0": m["m0_cpc_inter"],
            "mlp_m1": m["m1_cpc_inter"],
            "mlp_delta": m["delta_cpc"],
            "gamma": g["delta_cpc"] - m["delta_cpc"]
        })
        
    def summarize(vals, label):
        mean_v = np.mean(vals)
        median_v = np.median(vals)
        
        # Fold-stratified bootstrap for mean
        # Assuming vals correspond 1-to-1 with paired_results
        delta_by_fold = {f: [] for f in folds_to_run}
        for v, r in zip(vals, paired_results):
            delta_by_fold[r["fold"]].append(v)
            
        rng = np.random.default_rng(42)
        boot_means = []
        for _ in range(5000):
            samp = []
            for f in folds_to_run:
                fold_vals = delta_by_fold[f]
                samp.extend(rng.choice(fold_vals, size=len(fold_vals), replace=True))
            boot_means.append(np.mean(samp))
        ci_l, ci_h = np.percentile(boot_means, [2.5, 97.5])
        
        return {
            "mean": mean_v,
            "median": median_v,
            "ci_95": (ci_l, ci_h)
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
    
    print("\n[Urban GNN Backbone]")
    print(f"Mean M0: {np.mean([r['gnn_m0'] for r in paired_results]):.4f}")
    print(f"Mean M1: {np.mean([r['gnn_m1'] for r in paired_results]):.4f}")
    print(f"Mean Delta: {gnn_sum['mean']:+.4f} | Median: {gnn_sum['median']:+.4f}")
    print(f"95% CI: [{gnn_sum['ci_95'][0]:+.4f}, {gnn_sum['ci_95'][1]:+.4f}]")
    print(f"Cities Improved: {np.sum(gnn_deltas > 0)}/40 ({np.mean(gnn_deltas > 0)*100:.1f}%)")
    print(f"Wilcoxon (Delta > 0): p = {gnn_w_p:.4e}")
    
    print("\n[Pairwise MLP Backbone]")
    print(f"Mean M0: {np.mean([r['mlp_m0'] for r in paired_results]):.4f}")
    print(f"Mean M1: {np.mean([r['mlp_m1'] for r in paired_results]):.4f}")
    print(f"Mean Delta: {mlp_sum['mean']:+.4f} | Median: {mlp_sum['median']:+.4f}")
    print(f"95% CI: [{mlp_sum['ci_95'][0]:+.4f}, {mlp_sum['ci_95'][1]:+.4f}]")
    print(f"Cities Improved: {np.sum(mlp_deltas > 0)}/40 ({np.mean(mlp_deltas > 0)*100:.1f}%)")
    print(f"Wilcoxon (Delta > 0): p = {mlp_w_p:.4e}")
    
    print("\n[Backbone Dependence Gamma]")
    print(f"Mean Gamma: {gamma_sum['mean']:+.4f} | Median: {gamma_sum['median']:+.4f}")
    print(f"95% CI: [{gamma_sum['ci_95'][0]:+.4f}, {gamma_sum['ci_95'][1]:+.4f}]")
    print(f"Wilcoxon (Two-Sided): p = {gamma_w_p:.4e}")

    # Results by fold
    print("\n[Results by Fold]")
    for fold_id in folds_to_run:
        f_gammas = [r["gamma"] for r in paired_results if r["fold"] == fold_id]
        print(f"Fold {fold_id} (n={len(f_gammas)}): Mean Gamma = {np.mean(f_gammas):+.4f}")

if __name__ == "__main__":
    run_mlp_backbone_test()
