"""
Master 5-Fold Cross-Validation Experiment Runner (Moving-Bin Calibration Framework).
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
from src.experiment.generate_tables import generate_tables


def run_5fold_experiment(
    data_root: str = "data",
    meta_prior_dir: str = "meta_prior",
    output_dir: str = "results",
    epochs_per_fold: int = 25,
    lr: float = 2e-3,
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

    print("=" * 85)
    print("STARTING 5-FOLD CROSS-VALIDATION (MOVING-BIN CALIBRATION FRAMEWORK)")
    print(f"Device: {device_str} | Epochs: {epochs_per_fold} | Graph: {graph_type} (r={radius_km}km) | Seeds: {num_trip_seeds}")
    print(f"Primary Calibration Domain: Omega_c^+ (Interzonal moving bins 1, 2, 3)")
    print(f"Folds to run: {folds_to_run}")
    print("=" * 85)

    all_city_results = []
    fold_summaries = {}

    start_total_time = time.time()

    for fold_id in folds_to_run:
        split = splits[fold_id]
        train_cities = split["train"]
        test_cities = split["test"]

        print("\n" + "#" * 85)
        print(f"FOLD {fold_id}/5: Training on {len(train_cities)} cities -> Testing on {len(test_cities)} held-out cities")
        print(f"Held-out targets: {test_cities}")
        print("#" * 85)

        # Stage A: Cross-city Training
        fold_start = time.time()
        _ckpt_dir  = Path(output_dir) / "checkpoints"
        _ckpt_path = _ckpt_dir / f"5fold_fold{fold_id}.pt"
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
            checkpoint_path=_ckpt_path,
            run_tag=f"5fold_fold{fold_id}",
        )
        print(f"Fold {fold_id} model trained in {time.time() - fold_start:.1f}s.")
        print(f"  -> Checkpoint: {_ckpt_path.resolve()}")


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

            m0_c = city_res['M0']['cpc_inter']
            m1_r = city_res['M1_real_plus']['cpc_inter'] if city_res['M1_real_plus'] else None
            m1_o = city_res['M1_oracle_plus']['cpc_inter']
            delta_r = city_res['delta_r_real_plus']
            overlap = city_res['distributional_overlap']

            r_str = f"{m1_r:.4f}" if m1_r is not None else "N/A"
            d_str = f"{delta_r:+.4f}" if delta_r is not None else "N/A"
            ov_str = f"{overlap*100:.1f}%" if overlap is not None else "N/A"
            print(f" | M0: {m0_c:.4f} | M1_real+: {r_str} (Delta: {d_str}, Overlap: {ov_str}) | M1_oracle+: {m1_o:.4f} | {time.time() - t0:.1f}s")

        fold_summaries[f"fold_{fold_id}"] = {
            "test_cities": test_cities,
            "mean_delta_r_inter": float(sum(r["delta_r_real_plus"] for r in fold_city_results if r["delta_r_real_plus"] is not None) / max(1, len([r for r in fold_city_results if r["delta_r_real_plus"] is not None]))),
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

    print("\n" + "=" * 85)
    print("FINAL SUMMARY: MOVING-BIN CALIBRATION ON OMEGA_c^+")
    print("=" * 85)
    print(f"Total cities evaluated: {len(all_city_results)}/50")

    if "real_plus" in delta_r_analysis:
        rp = delta_r_analysis["real_plus"]
        print(f"\n[RQ1: Primary Moving-Bin Meta Calibration (M1^real, + on Omega_c^+)]")
        print(f"  Distributional Overlap with Meta (Mean +- Std): {rp['distributional_overlap']['mean']*100:.2f}% +- {rp['distributional_overlap']['std']*100:.2f}%")
        print(f"  M1 Real+ Interzonal CPC (Mean +- Std):          {rp['m1_real_cpc_inter']['mean']:.4f} +- {rp['m1_real_cpc_inter']['std']:.4f}")
        print(f"  Delta R^real+ Mean +- Std:                      {rp['delta_r_inter']['mean']:+.4f} +- {rp['delta_r_inter']['std']:.4f}")
        print(f"  Delta R^real+ Median (IQR):                     {rp['delta_r_inter']['median']:+.4f} ({rp['delta_r_inter']['iqr']:.4f})")
        print(f"  P(Delta R^real+ > 0):                           {rp['p_improved'] * 100:.1f}%")
        if "wilcoxon_one_sided_p" in rp:
            print(f"  Wilcoxon One-Sided p-value (H1: Delta > 0):     {rp['wilcoxon_one_sided_p']:.4e}")
            print(f"  Wilcoxon Two-Sided p-value:                     {rp['wilcoxon_two_sided_p']:.4e}")
        rg = rp["realization_gap"]
        print(f"  Realization Gap (Oracle+ - Real+):              Mean = {rg['mean']:+.4f} | Median = {rg['median']:+.4f} | MAE = {rg['mae']:.4f}")

    if "4bin_ablation" in delta_r_analysis:
        ab = delta_r_analysis["4bin_ablation"]
        print(f"\n[Ablation: Legacy 4-Bin Calibration (M1^real, 4bin — keeping Bin 0 mismatch)]")
        p_imp_inter = ab.get('p_improved_inter', ab.get('p_improved', 0.0)) * 100
        print(f"  P(Delta R (4-bin) > 0 on Interzonal):           {p_imp_inter:.1f}%")

    print(f"\nSaved full results to: {out_file.resolve()}")
    tables_dir = Path(output_dir) / "tables"
    generate_tables(str(out_file), output_dir=str(tables_dir))
    print(f"Saved summary tables and ablation breakdown to: {tables_dir.resolve()}")
    print("=" * 85)
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
