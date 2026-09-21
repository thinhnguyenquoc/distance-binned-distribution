"""Script to evaluate distance bin granularity scalability (K=4..40) and detect saturation."""

import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.calibration.bin_calibration import calibrate_kbins
from src.data.city_splits import load_splits_manifest_v2
from src.data.dataset import load_city
from src.data.urban_graph import build_radius_graph
from src.data.yd_extractor import (
    compute_equal_width_kbin_edges,
    compute_kbin_edges,
    extract_yd_kbins,
)
from src.training.evaluate import compute_cpc_pair
from src.training.train import infer_zero_shot, load_checkpoint

CANONICAL_SEEDS = [1, 10, 100]
K_LIST = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40]
DATA_ROOT = "data"
CHECKPOINT_DIR = Path("results/checkpoints")
OUTPUT_PATH = Path("results/bin_granularity_saturation.json")


def main():
    t_start = time.time()
    manifest_path = Path("results/e1/splits_manifest_v2.json")
    splits = load_splits_manifest_v2(str(manifest_path), data_root=DATA_ROOT)
    device = torch.device("cpu")

    # Step 1: Precompute / Cache zero-shot predictions for all 50 cities (averaged over seeds)
    print("Pre-computing Zero-Shot predictions on test cities across 5 folds...")
    city_cache = {}
    for fold in range(1, 6):
        split = splits[fold]
        models = {}
        for seed in CANONICAL_SEEDS:
            ckpt = CHECKPOINT_DIR / f"5fold_fold{fold}_seed{seed}.pt"
            model, scaler, _ = load_checkpoint(ckpt, device_str="cpu")
            models[seed] = (model, scaler)

        for city in sorted(split["test"]):
            city_data = load_city(city, data_root=DATA_ROOT, feature_scaler=models[1][1], fit_scaler=False)
            edge_index, edge_dist = build_radius_graph(city_data.lon_lat.numpy(), radius_km=5.0)
            truth = city_data.pair_trips.numpy().astype(np.float64)
            dist_km = np.asarray(city_data.dist_km, dtype=np.float64)
            inter_mask = (city_data.pair_o_idx.numpy() != city_data.pair_d_idx.numpy()) & (dist_km > 0.0)

            # seed predictions
            preds = []
            for seed in CANONICAL_SEEDS:
                m, _ = models[seed]
                pred = infer_zero_shot(m, city_data, edge_index, edge_dist, device=device).numpy().astype(np.float64)
                preds.append(pred)

            city_cache[city] = {
                "fold": fold,
                "truth": truth,
                "dist_km": dist_km,
                "inter_mask": inter_mask,
                "preds": preds,  # list of 3 arrays
            }

    print(f"Cached {len(city_cache)} cities. Starting K-sweep: {K_LIST}")

    results = {"quantile": {}, "equal_width": {}}

    for bin_method in ["quantile", "equal_width"]:
        edge_func = compute_kbin_edges if bin_method == "quantile" else compute_equal_width_kbin_edges
        results[bin_method] = {}

        for K in K_LIST:
            # Precompute edges per fold
            fold_edges = {}
            for fold in range(1, 6):
                edges, _ = edge_func(splits[fold]["train"], K=K, data_root=DATA_ROOT)
                fold_edges[fold] = edges

            city_deltas = []
            city_m0s = []
            city_m1s = []

            for city, item in city_cache.items():
                fold = item["fold"]
                edges = fold_edges[fold]
                truth = item["truth"]
                dist_km = item["dist_km"]
                inter_mask = item["inter_mask"]

                yd_target = extract_yd_kbins(dist_km, truth, edges, inter_mask)

                # Evaluate per seed and average
                m0_seeds = []
                m1_seeds = []
                for pred in item["preds"]:
                    m0 = float(compute_cpc_pair(truth[inter_mask], pred[inter_mask]))
                    cal = calibrate_kbins(pred, dist_km, inter_mask, yd_target, edges, q=1.0, tolerance=1e-5)
                    m1 = float(compute_cpc_pair(truth[inter_mask], cal[inter_mask]))
                    m0_seeds.append(m0)
                    m1_seeds.append(m1)

                avg_m0 = float(np.mean(m0_seeds))
                avg_m1 = float(np.mean(m1_seeds))
                city_m0s.append(avg_m0)
                city_m1s.append(avg_m1)
                city_deltas.append(avg_m1 - avg_m0)

            mean_m0 = float(np.mean(city_m0s))
            mean_m1 = float(np.mean(city_m1s))
            mean_delta = float(np.mean(city_deltas))
            median_delta = float(np.median(city_deltas))

            results[bin_method][K] = {
                "mean_m0_cpc": round(mean_m0, 5),
                "mean_m1_cpc": round(mean_m1, 5),
                "mean_delta_cpc": round(mean_delta, 5),
                "median_delta_cpc": round(median_delta, 5),
            }
            print(f"[{bin_method.upper()}] K={K:2d} -> M0: {mean_m0:.4f}, M1: {mean_m1:.4f}, Delta: {mean_delta:+.5f}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {OUTPUT_PATH} in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
