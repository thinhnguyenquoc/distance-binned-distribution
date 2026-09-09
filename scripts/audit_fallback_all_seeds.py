import json
import math
import itertools
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from src.data.city_splits import load_splits_manifest_v2
from src.data.dataset import load_city
from src.data.urban_graph import build_radius_graph
from src.data.yd_extractor import compute_kbin_edges, extract_yd_kbins
from src.training.train import load_checkpoint, infer_zero_shot

def main():
    manifest_path = Path("results/e1/splits_manifest_v2.json")
    splits = load_splits_manifest_v2(str(manifest_path), data_root="data")
    seeds = [1, 10, 100]
    K = 8
    epsilon = 1e-12

    def get_active_bins(yd, eps=1e-8):
        return yd > eps

    def safe_log_ratio(p, y_hat, active_mask, delta=1e-12):
        p = p.copy()
        y_hat = y_hat.copy()
        p_active = p[active_mask]
        if np.any(p_active < delta):
            p_active = np.maximum(p_active, delta)
            p_active = p_active / p_active.sum()
            p[active_mask] = p_active
        y_hat_active = np.maximum(y_hat[active_mask], delta)
        r = np.zeros_like(p)
        r[active_mask] = np.log(p_active) - np.log(y_hat_active)
        return r

    audit_results = {str(s): {"donor_total": 0, "donor_fallback": 0, "tm_total": 0, "tm_fallback": 0, "nan_inf": 0} for s in seeds}
    distinct_pairs_zero = set()
    distinct_pairs_below_delta = set()

    for fold_id in range(1, 6):
        split = splits[fold_id]
        train_cities = split["train"]
        test_cities = split["test"]
        bin_edges, _ = compute_kbin_edges(train_cities, K=K, data_root="data")

        train_yd_dict = {}
        for tc in train_cities:
            raw_c = load_city(tc, data_root="data", fit_scaler=False)
            dist_c = np.expm1(raw_c.pair_distance.numpy())
            inter_c = (raw_c.pair_o_idx.numpy() != raw_c.pair_d_idx.numpy()) & (dist_c > 0.0)
            t_gt_c = raw_c.pair_trips.numpy().astype(np.float64)
            train_yd_dict[tc] = extract_yd_kbins(dist_c, t_gt_c, bin_edges, inter_c)

        train_mean_yd = np.mean(list(train_yd_dict.values()), axis=0)

        test_yd_dict = {}
        test_data_dict = {}
        for tc in test_cities:
            raw_c = load_city(tc, data_root="data", fit_scaler=False)
            dist_c = np.expm1(raw_c.pair_distance.numpy())
            inter_c = (raw_c.pair_o_idx.numpy() != raw_c.pair_d_idx.numpy()) & (dist_c > 0.0)
            t_gt_c = raw_c.pair_trips.numpy().astype(np.float64)
            test_yd_dict[tc] = extract_yd_kbins(dist_c, t_gt_c, bin_edges, inter_c)
            test_data_dict[tc] = (raw_c, dist_c, inter_c, t_gt_c)

        # Check distinct target-donor combinations (50 * 35 = 1750)
        for tc in test_cities:
            yd_target = test_yd_dict[tc]
            act_mask = get_active_bins(yd_target)
            for d_city in train_cities:
                d_yd = train_yd_dict[d_city]
                d_active = d_yd[act_mask]
                if np.any(d_active == 0.0):
                    distinct_pairs_zero.add((tc, d_city))
                if np.any(d_active < epsilon):
                    distinct_pairs_below_delta.add((tc, d_city))

        for tc in test_cities:
            raw_c, dist_km, inter_mask, t_gt = test_data_dict[tc]
            ei, ed = build_radius_graph(raw_c.lon_lat, radius_km=5.0, include_self_loop=True, cache_key=f"{tc}_tracts")
            yd_target = test_yd_dict[tc]
            active_mask = get_active_bins(yd_target)
            dist_inter = dist_km[inter_mask]
            bin_masks = [((dist_inter > float(bin_edges[k])) & (dist_inter <= float(bin_edges[k + 1]))) for k in range(K)]

            for seed in seeds:
                ckpt_path = Path(f"results/checkpoints/5fold_fold{fold_id}_seed{seed}.pt")
                model, scaler, _ = load_checkpoint(ckpt_path, device_str="cpu")
                model.eval()

                city_data = load_city(tc, data_root="data", feature_scaler=scaler, fit_scaler=False)
                with torch.no_grad():
                    t0_tensor = infer_zero_shot(model, city_data, ei, ed, device="cpu")
                t0 = t0_tensor.numpy().astype(np.float64)
                t0_inter = t0[inter_mask]
                N_hat = t0_inter.sum()
                Y_hat = np.zeros(K, dtype=np.float64)
                for k in range(K):
                    if N_hat > 0:
                        Y_hat[k] = t0_inter[bin_masks[k]].sum() / N_hat

                r_T = safe_log_ratio(yd_target, Y_hat, active_mask, delta=epsilon)
                r_tilde_T = np.zeros_like(r_T)
                r_tilde_T[active_mask] = r_T[active_mask] - np.mean(r_T[active_mask])
                D_T = float(np.sqrt(np.mean(r_tilde_T[active_mask]**2)))

                # Check all 35 donors in train_cities for this seed
                for d_city in train_cities:
                    audit_results[str(seed)]["donor_total"] += 1
                    d_yd = train_yd_dict[d_city]
                    r_D = safe_log_ratio(d_yd, Y_hat, active_mask, delta=epsilon)
                    r_tilde_D = np.zeros_like(r_D)
                    r_tilde_D[active_mask] = r_D[active_mask] - np.mean(r_D[active_mask])
                    D_D = float(np.sqrt(np.mean(r_tilde_D[active_mask]**2)))
                    if D_D < 1e-12:
                        audit_results[str(seed)]["donor_fallback"] += 1

                    # Reconstruct and check NaN/Inf
                    r_tilde_D_star = np.zeros_like(r_tilde_D)
                    if D_D >= 1e-12:
                        r_tilde_D_star[active_mask] = r_tilde_D[active_mask] * (D_T / D_D)
                    p_D_star = np.zeros_like(Y_hat)
                    p_D_star[active_mask] = np.maximum(Y_hat[active_mask], epsilon) * np.exp(r_tilde_D_star[active_mask])
                    p_D_star[active_mask] /= p_D_star[active_mask].sum()
                    if np.isnan(p_D_star).any() or np.isinf(p_D_star).any() or np.isnan(D_D) or np.isinf(D_D):
                        audit_results[str(seed)]["nan_inf"] += 1

                # Check training mean for this seed
                audit_results[str(seed)]["tm_total"] += 1
                r_M = safe_log_ratio(train_mean_yd, Y_hat, active_mask, delta=epsilon)
                r_tilde_M = np.zeros_like(r_M)
                r_tilde_M[active_mask] = r_M[active_mask] - np.mean(r_M[active_mask])
                D_M = float(np.sqrt(np.mean(r_tilde_M[active_mask]**2)))
                if D_M < 1e-12:
                    audit_results[str(seed)]["tm_fallback"] += 1
                r_tilde_M_star = np.zeros_like(r_tilde_M)
                if D_M >= 1e-12:
                    r_tilde_M_star[active_mask] = r_tilde_M[active_mask] * (D_T / D_M)
                p_M_star = np.zeros_like(Y_hat)
                p_M_star[active_mask] = np.maximum(Y_hat[active_mask], epsilon) * np.exp(r_tilde_M_star[active_mask])
                p_M_star[active_mask] /= p_M_star[active_mask].sum()
                if np.isnan(p_M_star).any() or np.isinf(p_M_star).any() or np.isnan(D_M) or np.isinf(D_M):
                    audit_results[str(seed)]["nan_inf"] += 1

    out_data = {
        "audit_results": audit_results,
        "distinct_pairs_zero": len(distinct_pairs_zero),
        "distinct_pairs_below_delta": len(distinct_pairs_below_delta),
        "total_distinct_pairs": 50 * 35
    }
    Path("results").mkdir(parents=True, exist_ok=True)
    with open("results/audit_fallback_all_seeds.json", "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
    print("AUDIT_COMPLETE")
    print(json.dumps(out_data, indent=2))

if __name__ == "__main__":
    main()
