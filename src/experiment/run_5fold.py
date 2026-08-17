"""
Master 5-Fold Cross-Validation Experiment Runner across all 50 US cities.

Protocol:
    - 5 folds (each fold: 40 source cities for training, 10 held-out target cities for evaluation).
    - Every city appears as a held-out evaluation target exactly once.
    - Evaluates M0, M1_oracle, M1_real, Mq across all 50 cities (S=20 seeds per m grid point).
    - Saves detailed experiment results to results/5fold_results.json.
    - Produces final publication-ready summary tables for RQ1 (Delta R) and RQ2 (q*).
"""

import os
import sys
import json
import time
import argparse
import torch
from pathlib import Path

# Ensure root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.city_splits import generate_5fold_splits
from src.training.train import train_zero_shot_model
from src.experiment.run_experiment import run_target_city_experiments
from src.experiment.compute_delta_r import analyze_delta_r
from src.experiment.compute_qstar import analyze_qstar


def run_5fold_experiment(
    data_root: str = "data",
    meta_prior_dir: str = "meta_prior",
    output_dir: str = "results",
    epochs_per_fold: int = 25,
    lr: float = 1e-3,
    hidden_dim: int = 64,
    num_gnn_layers: int = 2,
    graph_type: str = "radius",
    radius_km: float = 5.0,
    knn_k: int = 10,
    loss_type: str = "ztnb",
    num_trip_seeds: int = 20,
    folds_to_run: list[int] | None = None,
    device_str: str | None = None,
):
    os.makedirs(output_dir, exist_ok=True)
    splits = generate_5fold_splits(data_root=data_root)

    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

    if folds_to_run is None:
        folds_to_run = [1, 2, 3, 4, 5]

    print("=" * 80)
    print("STARTING 5-FOLD CROSS-VALIDATION EXPERIMENT (50 US CITIES)")
    print(f"Device: {device_str} | Epochs/fold: {epochs_per_fold} | Loss: {loss_type} | Graph: {graph_type} (r={radius_km}km, k={knn_k}) | Seeds: {num_trip_seeds}")
    print(f"Folds to run: {folds_to_run}")
    print("=" * 80)

    all_city_results = []
    fold_summaries = {}

    start_total_time = time.time()

    for fold_id in folds_to_run:
        split = splits[fold_id]
        train_cities = split["train"]
        test_cities = split["test"]

        print("\n" + "#" * 80)
        print(f"FOLD {fold_id}/5: Training on {len(train_cities)} cities -> Testing on {len(test_cities)} held-out cities")
        print(f"Held-out targets: {test_cities}")
        print("#" * 80)

        # Stage A: Cross-city Training
        fold_start = time.time()
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
            device_str=device_str,
            verbose=True,
        )
        print(f"Fold {fold_id} model trained in {time.time() - fold_start:.1f}s.")

        # Stage B: Target City Evaluation
        fold_city_results = []
        for target_city in test_cities:
            print(f"  -> Evaluating: {target_city:<18}", end="", flush=True)
            t0 = time.time()
            city_res = run_target_city_experiments(
                model=model,
                city_name=target_city,
                scaler=scaler,
                data_root=data_root,
                meta_prior_dir=meta_prior_dir,
                graph_type=graph_type,
                radius_km=radius_km,
                knn_k=knn_k,
                num_trip_seeds=num_trip_seeds,
                device_str=device_str,
            )
            fold_city_results.append(city_res)
            all_city_results.append(city_res)

            real_str = f"{city_res['M1_real']['cpc']:.4f}" if city_res['M1_real'] is not None else "N/A"
            delta_real_str = f"{city_res['delta_r_real']:+.4f}" if city_res['delta_r_real'] is not None else "N/A"
            print(f" | M0: {city_res['M0']['cpc']:.4f} | M1_real: {real_str} (Delta: {delta_real_str}) | M1_oracle: {city_res['M1_oracle']['cpc']:.4f} | {time.time() - t0:.1f}s")

        fold_summaries[f"fold_{fold_id}"] = {
            "test_cities": test_cities,
            "mean_delta_r_oracle": float(sum(r["delta_r_oracle"] for r in fold_city_results) / len(fold_city_results)),
            "mean_delta_r_real": float(sum(r["delta_r_real"] for r in fold_city_results if r["delta_r_real"] is not None) / max(1, len([r for r in fold_city_results if r["delta_r_real"] is not None]))),
        }

    # Cross-city Statistical Aggregation
    delta_r_analysis = analyze_delta_r(all_city_results)
    qstar_analysis = analyze_qstar(all_city_results)

    final_results = {
        "experiment_config": {
            "device": device_str,
            "epochs_per_fold": epochs_per_fold,
            "hidden_dim": hidden_dim,
            "graph_type": graph_type,
            "radius_km": radius_km,
            "knn_k": knn_k,
            "loss_type": loss_type,
            "num_trip_seeds": num_trip_seeds,
            "total_cities_evaluated": len(all_city_results),
            "total_runtime_sec": time.time() - start_total_time,
        },
        "rq1_delta_r": delta_r_analysis,
        "rq2_qstar": qstar_analysis,
        "city_level_results": all_city_results,
    }

    out_file = Path(output_dir) / "5fold_results.json"
    with open(out_file, "w") as f:
        json.dump(final_results, f, indent=2)

    print("\n" + "=" * 80)
    print("FINAL 5-FOLD CROSS-VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Total cities evaluated: {len(all_city_results)}/50")

    if "real" in delta_r_analysis:
        print(f"\n[RQ1: Marginal Value of Real Y_D^Meta (Primary)]")
        print(f"  M1 Real CPC (Mean +- Std):        {delta_r_analysis['real']['m1_real_cpc_mean']:.4f}")
        print(f"  Delta R^real Mean +- Std:         {delta_r_analysis['real']['delta_r_mean']:+.4f} +- {delta_r_analysis['real']['delta_r_std']:.4f}")
        print(f"  Delta R^real Median:              {delta_r_analysis['real']['delta_r_median']:+.4f}")
        print(f"  P(Delta R^real > 0):              {delta_r_analysis['real']['p_improved'] * 100:.1f}%")
        if "wilcoxon_p" in delta_r_analysis["real"]:
            print(f"  Wilcoxon p-value:                 {delta_r_analysis['real']['wilcoxon_p']:.4e}")
        if delta_r_analysis['real']['realization_gap_mean'] is not None:
            print(f"  Realization Gap (Oracle - Real):  {delta_r_analysis['real']['realization_gap_mean']:+.4f}")

    print(f"\n[RQ1: Marginal Value of Oracle Y_D^GT (Ceiling)]")
    print(f"  M0 Zero-Shot CPC (Mean):          {delta_r_analysis['oracle']['m0_cpc_mean']:.4f}")
    print(f"  M1 Oracle CPC (Mean):             {delta_r_analysis['oracle']['m1_oracle_cpc_mean']:.4f}")
    print(f"  Delta R^oracle Mean +- Std:       {delta_r_analysis['oracle']['delta_r_mean']:+.4f} +- {delta_r_analysis['oracle']['delta_r_std']:.4f}")
    print(f"  P(Delta R^oracle > 0):            {delta_r_analysis['oracle']['p_improved'] * 100:.1f}%")

    if "real" in qstar_analysis:
        print(f"\n[RQ2: Observation-Equivalence q* and m* for Real Y_D^Meta (Primary)]")
        print(f"  m*_real (Trips required - Median): {qstar_analysis['real']['m_star']['median']:.1f} trips")
        print(f"  m*_real (Trips required - Mean):   {qstar_analysis['real']['m_star']['mean']:.1f} +- {qstar_analysis['real']['m_star']['std']:.1f}")
        print(f"  q*_real (Trip fraction - Median):  {qstar_analysis['real']['q_star']['median']:.6f}")
        print(f"  q*_real (Trip fraction - Mean):    {qstar_analysis['real']['q_star']['mean']:.6f} +- {qstar_analysis['real']['q_star']['std']:.6f}")

    print(f"\nSaved full results to: {out_file.resolve()}")
    print("=" * 80)
    return final_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--graph-type", type=str, default="radius", choices=["radius", "adaptive_radius", "knn"])
    parser.add_argument("--radius", type=float, default=5.0)
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    run_5fold_experiment(
        epochs_per_fold=args.epochs,
        folds_to_run=args.folds,
        num_trip_seeds=args.seeds,
        graph_type=args.graph_type,
        radius_km=args.radius,
        knn_k=args.knn_k,
        device_str=args.device,
    )
