"""
Backbone Robustness Evaluation Experiment (Urban GNN vs Pairwise MLP).
Trains and evaluates Pairwise MLP backbone (without graph convolutions)
across 5-Fold cross validation to assess calibration operator transferability.
"""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import sys
import json
import time
import argparse
import logging
import torch
import numpy as np
from pathlib import Path
from scipy import stats
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.city_splits import generate_35_5_10_splits
from src.data.dataset import load_city, load_raw_city
from src.data.urban_graph import build_radius_graph
from src.data.yd_extractor import compute_kbin_edges, extract_yd_kbins
from src.calibration.bin_calibration import calibrate_kbins
from src.training.evaluate import compute_cpc_pair
from src.training.train import train_zero_shot_model, infer_zero_shot


def fast_evaluate_city(model: torch.nn.Module, city_name: str, scaler: Any, bin_edges: np.ndarray, data_root: str = "data", device: str = "cpu") -> Dict[str, float]:
    """Fast, vectorized target city evaluation for M0 and M1 (City-level Oracle)."""
    raw = load_raw_city(city_name, data_root=data_root)
    dist_km = raw.dist_km
    inter_mask = (raw.pair_o_idx.numpy() != raw.pair_d_idx.numpy()) & (dist_km > 0.0)
    t_true_inter = raw.pair_trips.numpy()[inter_mask]

    edge_index, edge_dist = build_radius_graph(raw.lon_lat, radius_km=5.0, include_self_loop=True, cache_key=f"{city_name}_tracts")

    city_data = load_city(city_name, data_root=data_root, feature_scaler=scaler, fit_scaler=False)
    t_pred_zs_tensor = infer_zero_shot(model, city_data, edge_index, edge_dist, device=device)
    t_pred_zs = t_pred_zs_tensor.numpy().astype(np.float64)

    t0_inter = t_pred_zs[inter_mask]
    cpc_m0 = float(compute_cpc_pair(t_true_inter, t0_inter))

    yd_target = extract_yd_kbins(dist_km, raw.pair_trips.numpy(), bin_edges, inter_mask)
    t_cal = calibrate_kbins(t_pred_zs, dist_km, inter_mask, yd_target, bin_edges, q=1.0)
    t1_inter = t_cal[inter_mask]
    cpc_m1 = float(compute_cpc_pair(t_true_inter, t1_inter))

    return {
        "m0_cpc_inter": cpc_m0,
        "m1_cpc_inter": cpc_m1,
        "delta_cpc": cpc_m1 - cpc_m0
    }


def run_mlp_backbone_test(args: argparse.Namespace) -> None:
    data_root = args.data_root
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)

    log_file = os.path.join(output_dir, "mlp_backbone_execution.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)

    splits = generate_35_5_10_splits(data_root=data_root)
    
    if args.smoke:
        folds_to_run = [2]
        seeds = [1]
        epochs_per_fold = 2
        patience = 2
    else:
        folds_to_run = args.folds
        seeds = args.seeds
        epochs_per_fold = args.epochs
        patience = args.patience

    all_mlp_results = []
    
    logger.info("=" * 85)
    logger.info("STARTING PAIRWISE MLP BACKBONE TRAINING & EVALUATION")
    logger.info(f"Folds: {folds_to_run} | Seeds: {seeds} | Epochs: {epochs_per_fold} | Device: {args.device}")
    logger.info("=" * 85)
    
    for fold_id in folds_to_run:
        split = splits[fold_id]
        train_cities = split["train"]
        val_cities = split["val"]
        test_cities = split["test"] if not args.smoke else split["test"][:2]

        logger.info(f"\n# FOLD {fold_id}/5 (Train: {len(train_cities)}, Val: {len(val_cities)}, Test: {len(test_cities)})")
        models = []
        scalers = []
        
        for seed_idx, seed in enumerate(seeds):
            logger.info(f"--- Training MLP Seed {seed} (Fold {fold_id}) ---")
            _ckpt_path = Path(output_dir) / "checkpoints" / f"mlp_fold{fold_id}_seed{seed}.pt"
            
            model, scaler = train_zero_shot_model(
                train_city_names=train_cities,
                data_root=data_root,
                epochs=epochs_per_fold,
                lr=args.lr,
                hidden_dim=args.hidden_dim,
                num_gnn_layers=args.num_gnn_layers,
                graph_type=args.graph_type,
                radius_km=args.radius_km,
                knn_k=args.knn_k,
                loss_type=args.loss_type,
                backbone="mlp",  # <--- Pairwise Spatial MLP Backbone (No message passing)
                device_str=args.device,
                verbose=False,
                val_city_names=val_cities,
                patience=patience,
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
                res = fast_evaluate_city(
                    model=model,
                    city_name=target_city,
                    scaler=scaler,
                    bin_edges=bin_edges,
                    data_root=data_root,
                    device=args.device
                )
                seed_results.append(res)
                
            m0_cpc_inter = float(np.mean([r["m0_cpc_inter"] for r in seed_results]))
            m1_cpc_inter = float(np.mean([r["m1_cpc_inter"] for r in seed_results]))
            delta_cpc = m1_cpc_inter - m0_cpc_inter

            city_res = {
                "city": target_city,
                "fold": fold_id,
                "m0_cpc_inter": m0_cpc_inter,
                "m1_cpc_inter": m1_cpc_inter,
                "delta_cpc": delta_cpc,
                "seed_results": [
                    {
                        "seed": seeds[idx],
                        "m0_cpc_inter": r["m0_cpc_inter"],
                        "m1_cpc_inter": r["m1_cpc_inter"],
                        "delta_cpc": r["delta_cpc"]
                    }
                    for idx, r in enumerate(seed_results)
                ]
            }
            all_mlp_results.append(city_res)
            logger.info(f"  {target_city:15s} | M0: {m0_cpc_inter:.4f} | M1: {m1_cpc_inter:.4f} | d={delta_cpc:+.4f}")
            
    # Save MLP results
    mlp_json_path = Path(output_dir) / "mlp_backbone_results.json"
    with open(mlp_json_path, "w") as f:
        json.dump(all_mlp_results, f, indent=2)

    logger.info(f"\nSaved {len(all_mlp_results)} MLP backbone city results to {mlp_json_path}")
    logger.info("Run `python src/experiment/compare_backbones.py` to compare MLP with Urban GNN.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pairwise MLP Backbone Evaluation Experiment")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3.2e-3)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_gnn_layers", type=int, default=2)
    parser.add_argument("--graph_type", type=str, default="radius")
    parser.add_argument("--radius_km", type=float, default=5.0)
    parser.add_argument("--knn_k", type=int, default=10)
    parser.add_argument("--loss_type", type=str, default="ztnb")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--folds", nargs="+", type=int, default=[2, 3, 4, 5])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 10, 100])
    parser.add_argument("--smoke", action="store_true", help="Run quick 1-fold 1-seed smoke test")
    
    args = parser.parse_args()
    run_mlp_backbone_test(args)
