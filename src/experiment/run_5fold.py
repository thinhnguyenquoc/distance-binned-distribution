"""
Master 5-Fold Cross-Validation Experiment Runner (Moving-Bin Calibration Framework).
"""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import sys
import json
import time
import argparse
import torch
from pathlib import Path

# Ensure root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.city_splits import generate_35_5_10_splits
from src.data.yd_extractor import compute_kbin_edges
from src.training.train import train_zero_shot_model
from src.experiment.run_experiment import run_target_city_experiments
from src.experiment.compute_delta_r import analyze_delta_r
from src.experiment.compute_qstar import analyze_qstar

from src.training.train import load_checkpoint


def run_5fold_experiment(
    data_root: str = "data",
    meta_prior_dir: str = "meta_prior",
    output_dir: str = "results",
    epochs_per_fold: int = 200,
    lr: float = 3.2e-3,
    hidden_dim: int = 64,
    num_gnn_layers: int = 2,
    graph_type: str = "radius",
    radius_km: float = 5.0,
    knn_k: int = 10,
    loss_type: str = "ztnb",
    backbone: str = "gnn",
    num_trip_seeds: int = 20,
    folds_to_run: list[int] | None = None,
    device_str: str | None = None,
):
    os.makedirs(output_dir, exist_ok=True)
    splits = generate_35_5_10_splits(data_root=data_root)

    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

    if folds_to_run is None:
        folds_to_run = [1, 2, 3, 4, 5]

    print("=" * 85)
    print("STARTING 5-FOLD CROSS-VALIDATION (MOVING-BIN CALIBRATION FRAMEWORK)")
    print(f"Device: {device_str} | Epochs: {epochs_per_fold} | Graph: {graph_type} (r={radius_km}km) | Seeds: {num_trip_seeds}")
    print(f"Primary Calibration Domain: Omega_c^+ (Interzonal moving bins 1, 2, 3)")
    print(f"Folds to run: {folds_to_run}")
    print("=" * 85)

    out_file_name = "5fold_results.json" if backbone == "gnn" else f"{backbone}_backbone_results.json"
    out_file = Path(output_dir) / out_file_name

    all_city_results = []
    if out_file.exists():
        try:
            with open(out_file, "r") as f:
                prev_json = json.load(f)
                all_city_results = prev_json.get("city_level_results", [])
                print(f"Loaded {len(all_city_results)} existing city records from {out_file}.")
        except Exception:
            all_city_results = []

    fold_summaries = {}

    start_total_time = time.time()

    for fold_id in folds_to_run:
        split = splits[fold_id]
        train_cities = split["train"]
        val_cities = split["val"]
        test_cities = split["test"]

        # Remove old records for this fold to ensure clean incremental update
        all_city_results = [r for r in all_city_results if r.get("fold") != fold_id]

        print("\n" + "#" * 85)
        print(f"FOLD {fold_id}/5: Training on {len(train_cities)} cities -> Testing on {len(test_cities)} held-out cities")
        print(f"Validation cities: {val_cities}")
        print(f"Held-out targets: {test_cities}")
        print("#" * 85)

        fold_start = time.time()
        models = []
        scalers = []
        seeds = [1, 10, 100]
        
        for seed_idx, seed in enumerate(seeds):
            _ckpt_dir  = Path(output_dir) / "checkpoints"
            _ckpt_name = f"5fold_fold{fold_id}_seed{seed}.pt" if backbone == "gnn" else f"5fold_{backbone}_fold{fold_id}_seed{seed}.pt"
            _ckpt_path = _ckpt_dir / _ckpt_name
            
            if _ckpt_path.exists():
                print(f"--- Found existing checkpoint {_ckpt_path}. Loading... ---")
                model, scaler, _ = load_checkpoint(_ckpt_path, device_str=device_str)
                model.eval()
            else:
                print(f"\n--- Training Seed {seed_idx+1}/{len(seeds)} (Seed: {seed}) [Backbone: {backbone.upper()}] ---")
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
                    backbone=backbone,
                    device_str=device_str,
                    verbose=True,
                    val_city_names=val_cities,
                    patience=16,
                    checkpoint_path=_ckpt_path,
                    run_tag=f"5fold_{backbone}_fold{fold_id}_seed{seed}",
                    seed=seed,
                )
            models.append(model)
            scalers.append(scaler)
        print(f"Fold {fold_id} models trained in {time.time() - fold_start:.1f}s.")




        # Compute Bin Edges from 35 train cities (K=8)
        bin_edges, K_active = compute_kbin_edges(train_cities, K=8, data_root=data_root)

        # Stage B: Target City Evaluation
        fold_city_results = []
        for target_city in test_cities:
            print(f"  -> Evaluating: {target_city:<18}", end="", flush=True)
            t0 = time.time()
            
            seed_results = []
            for seed_idx, model in enumerate(models):
                scaler = scalers[seed_idx]
                res = run_target_city_experiments(
                    model=model,
                    city_name=target_city,
                    scaler=scaler,
                    data_root=data_root,
                    graph_type=graph_type,
                    radius_km=radius_km,
                    knn_k=knn_k,
                    device_str=device_str,
                    bin_edges=bin_edges,
                )
                seed_results.append(res)
                
            # Average the results across 3 seeds
            avg_res = seed_results[0].copy()
            for key in ["M0", "M1_city_oracle_obs", "M1_county_oracle_obs", "M1_subzone_oracle_obs"]:
                if avg_res[key] is not None:
                    avg_res[key] = avg_res[key].copy()
                    for metric in ["cpc_inter", "mae_inter", "rmse_inter", "nrmse_inter", "rmse_log1p_inter", "spearman_inter", "rel_error_total", "cpc_inflow", "cpc_outflow"]:
                        if metric in avg_res[key]:
                            avg_res[key][metric] = sum(r[key][metric] for r in seed_results) / len(seed_results)
            
            for key in ["rho_c", "average_flow", "mean_distance"]:
                if key in avg_res and avg_res[key] is not None:
                    avg_res[key] = sum(r[key] for r in seed_results) / len(seed_results)
            
            # Compute Deltas (Primary Estimands)
            avg_res["delta_city"] = avg_res["M1_city_oracle_obs"]["cpc_inter"] - avg_res["M0"]["cpc_inter"]
            avg_res["delta_county"] = avg_res["M1_county_oracle_obs"]["cpc_inter"] - avg_res["M0"]["cpc_inter"]
            avg_res["delta_subzone"] = avg_res["M1_subzone_oracle_obs"]["cpc_inter"] - avg_res["M0"]["cpc_inter"]
            
            city_res = avg_res
            city_res["fold"] = fold_id
            fold_city_results.append(city_res)
            all_city_results.append(city_res)

            m0_c = city_res['M0']['cpc_inter']
            m1_city = city_res['M1_city_oracle_obs']['cpc_inter']
            m1_county = city_res['M1_county_oracle_obs']['cpc_inter']
            m1_sub = city_res['M1_subzone_oracle_obs']['cpc_inter']

            print(f" | M0: {m0_c:.4f} | M1_city: {m1_city:.4f} (d={avg_res['delta_city']:+.4f}) | M1_county: {m1_county:.4f} (d={avg_res['delta_county']:+.4f}) | M1_subzone: {m1_sub:.4f} (d={avg_res['delta_subzone']:+.4f}) | {time.time() - t0:.1f}s")

        fold_summaries[f"fold_{fold_id}"] = {
            "test_cities": test_cities,
            "mean_delta_city": float(sum(r["delta_city"] for r in fold_city_results) / max(1, len(fold_city_results))),
        }
        
        # Intermediate Save
        out_file_name = "5fold_results.json" if backbone == "gnn" else f"{backbone}_backbone_results.json"
        out_file = Path(output_dir) / out_file_name
        temp_delta_r = analyze_delta_r(all_city_results)
        temp_results = {
            "experiment_config": {
                "device": device_str,
                "epochs_per_fold": epochs_per_fold,
                "hidden_dim": hidden_dim,
                "graph_type": graph_type,
                "radius_km": radius_km,
                "knn_k": knn_k,
                "loss_type": loss_type,
                "total_cities_evaluated": len(all_city_results),
                "total_runtime_sec": time.time() - start_total_time,
            },
            "rq1_delta_r": temp_delta_r,
            "city_level_results": all_city_results,
        }
        with open(out_file, "w") as f:
            json.dump(temp_results, f, indent=2)

    # Cross-city Statistical Aggregation (Final)
    delta_r_analysis = analyze_delta_r(all_city_results)

    final_results = {
        "experiment_config": {
            "device": device_str,
            "epochs_per_fold": epochs_per_fold,
            "hidden_dim": hidden_dim,
            "graph_type": graph_type,
            "radius_km": radius_km,
            "knn_k": knn_k,
            "loss_type": loss_type,
            "total_cities_evaluated": len(all_city_results),
            "total_runtime_sec": time.time() - start_total_time,
        },
        "rq1_delta_r": delta_r_analysis,
        "city_level_results": all_city_results,
    }

    out_file_name = "5fold_results.json" if backbone == "gnn" else f"{backbone}_backbone_results.json"
    out_file = Path(output_dir) / out_file_name
    with open(out_file, "w") as f:
        json.dump(final_results, f, indent=2)

    print("\n" + "=" * 85)
    print("FINAL SUMMARY: UNIFIED RESOLUTION CALIBRATION (CITY / COUNTY / SUBZONE)")
    print("=" * 85)
    print(f"Total cities evaluated: {len(all_city_results)}/50")

    for scale in ["city", "county", "subzone"]:
        if scale in delta_r_analysis:
            s_data = delta_r_analysis[scale]
            scale_label = "GADM 4.1 LEVEL-2 COUNTY" if scale == "county" else f"{scale.upper()}"
            print(f"\n[{scale_label}-LEVEL CALIBRATION]")
            print(f"  M0 Interzonal CPC (Mean):                       {s_data['m0_cpc_inter']['mean']:.4f}")
            print(f"  M1 Interzonal CPC (Mean):                       {s_data['m1_cpc_inter']['mean']:.4f}")
            print(f"  Delta Mean +- Std:                              {s_data['delta_cpc_inter']['mean']:+.4f} +- {s_data['delta_cpc_inter']['std']:.4f}")
            print(f"  Delta 95% CI (Fold-Stratified Bootstrap):       [{s_data['delta_cpc_inter']['ci_95_lower']:+.4f}, {s_data['delta_cpc_inter']['ci_95_upper']:+.4f}]")
            print(f"  P(Delta > 0):                                   {s_data['p_improved'] * 100:.1f}%")
            if "wilcoxon_one_sided_p" in s_data:
                print(f"  Wilcoxon One-Sided p-value (H1: Delta > 0):     {s_data['wilcoxon_one_sided_p']:.4e}")

    print(f"\nSaved full results to: {out_file.resolve()}")
    print("=" * 85)
    return final_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--graph-type", type=str, default="radius", choices=["radius", "adaptive_radius", "knn"])
    parser.add_argument("--radius", type=float, default=5.0)
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--backbone", type=str, default="gnn", choices=["gnn", "mlp"])
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    run_5fold_experiment(
        epochs_per_fold=args.epochs,
        folds_to_run=args.folds,
        num_trip_seeds=args.seeds,
        graph_type=args.graph_type,
        radius_km=args.radius,
        knn_k=args.knn_k,
        loss_type="ztnb",
        backbone=args.backbone,
        device_str=args.device,
    )
