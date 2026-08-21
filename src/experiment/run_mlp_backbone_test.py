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
) -> None:
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
                
            if not seed_results:
                print(f"Warning: No valid results for {target_city}. Skipping.")
                continue

            # Average over seeds
            import copy
            avg_res = copy.deepcopy(seed_results[0])
            for key in ["M0", "M1_city_oracle_obs"]:
                if avg_res.get(key) is not None:
                    for metric in ["cpc_inter"]:
                        vals = [r[key][metric] for r in seed_results if not np.isnan(r[key][metric])]
                        avg_res[key][metric] = sum(vals) / len(vals) if vals else 0.0
            
            if "M0" not in avg_res or "M1_city_oracle_obs" not in avg_res:
                print(f"Warning: Missing required keys in results for {target_city}. Skipping.")
                continue

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

    print(f"\nSaved MLP backbone results to {Path(output_dir) / 'mlp_backbone_results.json'}")
    print("Run `compare_backbones.py` to compare MLP with Urban GNN.")

if __name__ == "__main__":
    run_mlp_backbone_test()
