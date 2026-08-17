"""
Master 5-Fold Cross-Validation Experiment Runner across all 50 US cities.

Protocol:
    - 5 folds (each fold: 40 source cities for training, 10 held-out target cities for evaluation).
    - Every city appears as a held-out evaluation target exactly once.
    - Evaluates M0, M1_oracle, M1_real, Mq across all 50 cities.
    - Saves detailed experiment results to results/5fold_results.json.
    - Produces final publication-ready summary tables for RQ1 (Delta R) and RQ2 (q*).
"""

import os
import json
import time
import argparse
from pathlib import Path

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
    knn_k: int = 10,
    loss_type: str = "ztnb",
    num_trip_seeds: int = 10,
    folds_to_run: list[int] | None = None,
):
    os.makedirs(output_dir, exist_ok=True)
    splits = generate_5fold_splits(data_root=data_root)

    if folds_to_run is None:
        folds_to_run = [1, 2, 3, 4, 5]

    print("=" * 80)
    print("STARTING 5-FOLD CROSS-VALIDATION EXPERIMENT (50 US CITIES)")
    print(f"Epochs per fold: {epochs_per_fold} | Loss: {loss_type} | k-NN: {knn_k}")
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
            knn_k=knn_k,
            loss_type=loss_type,
            verbose=True,
        )
        print(f"Fold {fold_id} model trained in {time.time() - fold_start:.1f}s.")

        # Stage B: Target City Evaluation
        fold_city_results = []
        for target_city in test_cities:
            print(f"  -> Evaluating held-out city: {target_city:<20}", end="", flush=True)
            t0 = time.time()
            city_res = run_target_city_experiments(
                model=model,
                city_name=target_city,
                scaler=scaler,
                data_root=data_root,
                meta_prior_dir=meta_prior_dir,
                knn_k=knn_k,
                num_trip_seeds=num_trip_seeds,
            )
            fold_city_results.append(city_res)
            all_city_results.append(city_res)
            print(f" | M0 CPC: {city_res['M0']['cpc']:.4f} | M1_oracle: {city_res['M1_oracle']['cpc']:.4f} | Delta R: {city_res['delta_r_oracle']:+.4f} | {time.time() - t0:.1f}s")

        fold_summaries[f"fold_{fold_id}"] = {
            "test_cities": test_cities,
            "mean_delta_r": float(sum(r["delta_r_oracle"] for r in fold_city_results) / len(fold_city_results)),
        }

    # Cross-city Statistical Aggregation
    delta_r_analysis = analyze_delta_r(all_city_results)
    qstar_analysis = analyze_qstar(all_city_results)

    final_results = {
        "experiment_config": {
            "epochs_per_fold": epochs_per_fold,
            "hidden_dim": hidden_dim,
            "knn_k": knn_k,
            "loss_type": loss_type,
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
    print("FINAL 5-FOLD CROSS-VALIDATION SUMMARY (ALL HELD-OUT CITIES)")
    print("=" * 80)
    print(f"Total cities evaluated: {len(all_city_results)}/50")
    print(f"\n[RQ1: Marginal Value of Y_D]")
    print(f"  M0 Zero-Shot CPC (Mean +- Std):    {delta_r_analysis['oracle']['m0_cpc_mean']:.4f}")
    print(f"  M1 Oracle CPC (Mean):             {delta_r_analysis['oracle']['m1_oracle_cpc_mean']:.4f}")
    print(f"  Delta R Mean +- Std:              {delta_r_analysis['oracle']['delta_r_mean']:+.4f} +- {delta_r_analysis['oracle']['delta_r_std']:.4f}")
    print(f"  Delta R Median (IQR):             {delta_r_analysis['oracle']['delta_r_median']:+.4f} ({delta_r_analysis['oracle']['delta_r_iqr']:.4f})")
    print(f"  P(Delta R > 0) [Improved fraction]:{delta_r_analysis['oracle']['p_improved'] * 100:.1f}%")
    if "wilcoxon_p" in delta_r_analysis["oracle"]:
        print(f"  Wilcoxon signed-rank p-value:     {delta_r_analysis['oracle']['wilcoxon_p']:.4e}")

    print(f"\n[RQ2: Observation-Equivalence q* and m*]")
    print(f"  m* (Trips required - Median):     {qstar_analysis['m_star']['median']:.1f} trips")
    print(f"  m* (Trips required - Mean +- Std): {qstar_analysis['m_star']['mean']:.1f} +- {qstar_analysis['m_star']['std']:.1f}")
    print(f"  q* (Trip fraction - Median):      {qstar_analysis['q_star']['median']:.6f}")
    print(f"  q* (Trip fraction - Mean +- Std):  {qstar_analysis['q_star']['mean']:.6f} +- {qstar_analysis['q_star']['std']:.6f}")

    print(f"\nSaved full results to: {out_file.resolve()}")
    print("=" * 80)
    return final_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    args = parser.parse_args()
    run_5fold_experiment(epochs_per_fold=args.epochs, folds_to_run=args.folds)
