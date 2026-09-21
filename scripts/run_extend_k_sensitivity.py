"""Run K-sensitivity extension for K in [22, 24, 26, 28, 30] and merge with previous K=2..20 results."""

import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.calibration.bin_calibration import calibrate_kbins
from src.data.city_splits import load_splits_manifest_v2
from src.data.dataset import load_city
from src.data.urban_graph import build_radius_graph
from src.data.yd_extractor import compute_kbin_edges, extract_yd_kbins
from src.training.evaluate import evaluate_moving_and_full
from src.training.train import infer_zero_shot, load_checkpoint

CANONICAL_SEEDS = [1, 10, 100]
NEW_K_VALUES = [22, 24, 26, 28, 30]
DATA_ROOT = "data"
CHECKPOINT_DIR = Path("results/checkpoints")
PREV_SUMMARY_FILE = Path("results/interzonal_only/artifacts/k_sensitivity/k_sensitivity_summary.json")
OUTPUT_SUMMARY_FILE = Path("results/k_sensitivity_extended_summary.json")


def main():
    t_start = time.time()
    manifest_path = Path("results/e1/splits_manifest_v2.json")
    splits = load_splits_manifest_v2(str(manifest_path), data_root=DATA_ROOT)
    device = torch.device("cpu")

    results_new = []

    for fold in range(1, 6):
        split = splits[fold]
        train_cities = split["train"]
        test_cities = split["test"]

        # Compute bin edges for new K values
        bin_edges_by_k = {}
        for K in NEW_K_VALUES:
            edges, k_act = compute_kbin_edges(train_cities, K=K, data_root=DATA_ROOT)
            bin_edges_by_k[K] = {"edges": edges, "k_active": k_act}

        print(f"--- Fold {fold} ---")
        for city_idx, city in enumerate(test_cities, 1):
            print(f"  [{city_idx}/{len(test_cities)}] Processing {city}...")
            for seed in CANONICAL_SEEDS:
                ckpt_path = CHECKPOINT_DIR / f"5fold_fold{fold}_seed{seed}.pt"
                model, scaler, _ = load_checkpoint(str(ckpt_path), device_str="cpu")
                model.eval()

                city_data = load_city(city, data_root=DATA_ROOT, feature_scaler=scaler, fit_scaler=False)
                coords = city_data.lon_lat.numpy()
                edge_index, edge_dist = build_radius_graph(coords, radius_km=5.0)

                t_true = city_data.pair_trips.numpy().astype(np.float64)
                pair_o = city_data.pair_o_idx.numpy()
                pair_d = city_data.pair_d_idx.numpy()
                pair_dist_km = np.asarray(city_data.dist_km, dtype=np.float64)

                inter_mask = (pair_o != pair_d) & (pair_dist_km > 0.0)

                t_pred_zs_tensor = infer_zero_shot(model, city_data, edge_index, edge_dist, device=device)
                t0_np = t_pred_zs_tensor.numpy().astype(np.float64)

                m0_metrics = evaluate_moving_and_full(
                    city_data.pair_trips,
                    t_pred_zs_tensor,
                    city_data.pair_o_idx,
                    city_data.pair_d_idx,
                    city_data.bin_labels,
                    pair_distance=city_data.pair_distance,
                )
                m0_cpc = float(m0_metrics["cpc_inter"])

                for K in NEW_K_VALUES:
                    edges = bin_edges_by_k[K]["edges"]
                    k_active = bin_edges_by_k[K]["k_active"]
                    yd_target = extract_yd_kbins(pair_dist_km, t_true, edges, inter_mask)

                    t_cal = calibrate_kbins(t0_np, pair_dist_km, inter_mask, yd_target, edges, q=1.0, tolerance=1e-5)
                    m1_metrics = evaluate_moving_and_full(
                        city_data.pair_trips,
                        torch.tensor(t_cal),
                        city_data.pair_o_idx,
                        city_data.pair_d_idx,
                        city_data.bin_labels,
                        pair_distance=city_data.pair_distance,
                    )
                    m1_cpc = float(m1_metrics["cpc_inter"])
                    delta_cpc = m1_cpc - m0_cpc

                    results_new.append({
                        "city": city,
                        "fold": fold,
                        "seed": seed,
                        "K": K,
                        "m0_cpc_inter": m0_cpc,
                        "m1_cpc_inter": m1_cpc,
                        "delta_cpc": delta_cpc,
                        "k_active": k_active,
                    })

    df_new = pd.DataFrame(results_new)
    df_city_new = df_new.groupby(["city", "fold", "K"])[["m0_cpc_inter", "m1_cpc_inter", "delta_cpc", "k_active"]].mean().reset_index()

    # Calculate summary stats for each K in 22..30
    summary_new = []
    for K in NEW_K_VALUES:
        d = df_city_new[df_city_new["K"] == K]
        m0_mean = float(d["m0_cpc_inter"].mean())
        m1_mean = float(d["m1_cpc_inter"].mean())
        delta = d["delta_cpc"].values
        mean_d = float(np.mean(delta))
        std_d = float(np.std(delta, ddof=1))

        rng = np.random.default_rng(42)
        boot_means = []
        fold_vals_list = [d[d["fold"] == fold]["delta_cpc"].values for fold in range(1, 6)]
        for _ in range(10000):
            s = []
            for vals in fold_vals_list:
                s.extend(rng.choice(vals, size=len(vals), replace=True))
            boot_means.append(np.mean(s))
        ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

        summary_new.append({
            "K": K,
            "m0_cpc": m0_mean,
            "m1_cpc": m1_mean,
            "mean_delta": mean_d,
            "std_delta": std_d,
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "pos_cities": int(np.sum(delta > 0)),
            "total_cities": len(d),
            "k_act_mean": float(d["k_active"].mean()),
        })

    # Load previous summary K=2..20
    with open(PREV_SUMMARY_FILE, "r", encoding="utf-8") as f:
        prev_data = json.load(f)

    merged_summary = list(prev_data["summary"]) + summary_new
    output_obj = {
        "summary": merged_summary,
        "extended_k_tested": NEW_K_VALUES,
    }

    with open(OUTPUT_SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(output_obj, f, indent=2)

    print(f"\nCompleted in {time.time() - t_start:.1f}s. Summary saved to {OUTPUT_SUMMARY_FILE}")
    print("\n--- EXTENDED SUMMARY (K=2..30) ---")
    for s in merged_summary:
        print(f"K={s['K']:2d} | M0={s['m0_cpc']:.4f} | M1={s['m1_cpc']:.4f} | Delta={s['mean_delta']:+.5f} (CI: [{s['ci_low']:.5f}, {s['ci_high']:.5f}]) | +Cities={s['pos_cities']}/{s['total_cities']}")


if __name__ == "__main__":
    main()
