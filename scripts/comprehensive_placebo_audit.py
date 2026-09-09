import json
import sys
from pathlib import Path
import numpy as np
import torch
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.city_splits import load_splits_manifest_v2
from src.data.dataset import load_city
from src.data.urban_graph import build_radius_graph
from src.data.yd_extractor import compute_kbin_edges, extract_yd_kbins
from src.training.train import load_checkpoint, infer_zero_shot
from src.experiment.run_unified_placebo import (
    get_active_bins as orig_get_active_bins,
    safe_log_ratio as orig_safe_log_ratio,
    fast_eval_cpc as orig_fast_eval_cpc,
)

def run_comprehensive_audit():
    manifest_path = Path("results/e1/splits_manifest_v2.json")
    splits = load_splits_manifest_v2(str(manifest_path), data_root="data")
    seeds = [1, 10, 100]
    K = 8
    epsilon = 1e-12

    # Step 1: Active mask check across 50 cities
    # Compare active_by_support (pairs > 0 in evaluation support), active_by_experiment (yd > 1e-8), active_by_audit (yd > 1e-8)
    mask_diff_count = 0
    mask_diff_details = []

    # Step 2: safe_log_ratio comparison
    # We will track max absolute difference across:
    # - centered log-ratio
    # - RMS (D)
    # - reconstructed distribution p*
    max_diff_r_tilde = 0.0
    max_diff_D = 0.0
    max_diff_p_star = 0.0

    # Step 3: Granular stage-by-stage NaN/Inf checks + CPC check
    # Stages:
    # 1. baseline: t0, Y_hat
    # 2. log_ratio: r_D, r_tilde_D, r_M, r_tilde_M
    # 3. dose_matching: D_D, r_tilde_D_star, D_M, r_tilde_M_star
    # 4. reconstruction: p_D_star, p_M_star
    # 5. evaluation: cpc_matched, cpc_matched_tm
    nan_inf_counts = {
        str(s): {
            "baseline": 0,
            "log_ratio": 0,
            "dose_matching": 0,
            "reconstruction": 0,
            "evaluation": 0,
            "donor_fallback": 0,
            "tm_fallback": 0,
            "donor_trials": 0,
            "tm_trials": 0,
        }
        for s in seeds
    }

    # Step 4: Direct count of (yd > 0) & (yd < 1e-12)
    # Target-donor-bin level and target-donor combination level
    total_donor_bins = 0
    donor_bins_zero = 0
    donor_bins_positive_below_delta = 0
    pairs_with_zero = set()
    pairs_with_positive_below_delta = set()

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

            # Check Step 1: Active masks
            dist_inter = dist_c[inter_c]
            support_pair_counts = np.array([
                ((dist_inter > float(bin_edges[k])) & (dist_inter <= float(bin_edges[k + 1]))).sum()
                for k in range(K)
            ])
            active_by_support = support_pair_counts > 0
            active_by_experiment = orig_get_active_bins(test_yd_dict[tc], eps=1e-8)
            active_by_audit = (test_yd_dict[tc] > 1e-8)

            if not np.array_equal(active_by_support, active_by_experiment) or not np.array_equal(active_by_experiment, active_by_audit):
                mask_diff_count += 1
                mask_diff_details.append({
                    "city": tc,
                    "support_counts": support_pair_counts.tolist(),
                    "yd": test_yd_dict[tc].tolist(),
                    "by_support": active_by_support.tolist(),
                    "by_experiment": active_by_experiment.tolist(),
                })

            # Check Step 4: Target-donor bins
            act_mask = active_by_experiment
            for d_city in train_cities:
                d_yd = train_yd_dict[d_city]
                d_active = d_yd[act_mask]
                total_donor_bins += len(d_active)
                
                zeros = (d_active == 0.0)
                pos_below = (d_active > 0.0) & (d_active < epsilon)

                if np.any(zeros):
                    pairs_with_zero.add((tc, d_city))
                donor_bins_zero += int(zeros.sum())

                if np.any(pos_below):
                    pairs_with_positive_below_delta.add((tc, d_city))
                donor_bins_positive_below_delta += int(pos_below.sum())

        # Seed loops for Step 2, 3
        for tc in test_cities:
            raw_c, dist_km, inter_mask, t_gt = test_data_dict[tc]
            ei, ed = build_radius_graph(raw_c.lon_lat, radius_km=5.0, include_self_loop=True, cache_key=f"{tc}_tracts")
            yd_target = test_yd_dict[tc]
            active_mask = orig_get_active_bins(yd_target)
            t_true_inter = t_gt[inter_mask]
            dist_inter = dist_km[inter_mask]
            bin_masks = [((dist_inter > float(bin_edges[k])) & (dist_inter <= float(bin_edges[k + 1]))) for k in range(K)]

            for seed in seeds:
                s_key = str(seed)
                ckpt_path = Path(f"results/checkpoints/5fold_fold{fold_id}_seed{seed}.pt")
                model, scaler, _ = load_checkpoint(ckpt_path, device_str="cpu")
                model.eval()

                city_data = load_city(tc, data_root="data", feature_scaler=scaler, fit_scaler=False)
                with torch.no_grad():
                    t0_tensor = infer_zero_shot(model, city_data, ei, ed, device="cpu")
                t0 = t0_tensor.numpy().astype(np.float64)
                t0_inter = t0[inter_mask]
                denom = float(t_true_inter.sum() + t0_inter.sum())

                # Check Stage 1: Baseline
                if not np.isfinite(t0).all() or not np.isfinite(t0_inter).all():
                    nan_inf_counts[s_key]["baseline"] += 1

                N_hat = t0_inter.sum()
                Y_hat = np.zeros(K, dtype=np.float64)
                for k in range(K):
                    if N_hat > 0:
                        Y_hat[k] = t0_inter[bin_masks[k]].sum() / N_hat

                if not np.isfinite(Y_hat).all() or not np.isfinite(denom):
                    nan_inf_counts[s_key]["baseline"] += 1

                r_T = orig_safe_log_ratio(yd_target, Y_hat, active_mask, delta=epsilon)
                r_tilde_T = np.zeros_like(r_T)
                r_tilde_T[active_mask] = r_T[active_mask] - np.mean(r_T[active_mask])
                D_T = float(np.sqrt(np.mean(r_tilde_T[active_mask]**2)))

                # 35 donors in train
                for d_city in train_cities:
                    nan_inf_counts[s_key]["donor_trials"] += 1
                    d_yd = train_yd_dict[d_city]

                    # Stage 2: Log-ratio
                    r_D = orig_safe_log_ratio(d_yd, Y_hat, active_mask, delta=epsilon)
                    r_tilde_D = np.zeros_like(r_D)
                    r_tilde_D[active_mask] = r_D[active_mask] - np.mean(r_D[active_mask])
                    if not np.isfinite(r_D).all() or not np.isfinite(r_tilde_D).all():
                        nan_inf_counts[s_key]["log_ratio"] += 1

                    # Stage 3: Dose matching
                    D_D = float(np.sqrt(np.mean(r_tilde_D[active_mask]**2)))
                    if not np.isfinite(D_D):
                        nan_inf_counts[s_key]["dose_matching"] += 1

                    if D_D < 1e-12:
                        nan_inf_counts[s_key]["donor_fallback"] += 1
                        continue

                    r_tilde_D_star = np.zeros_like(r_tilde_D)
                    r_tilde_D_star[active_mask] = r_tilde_D[active_mask] * (D_T / D_D)
                    if not np.isfinite(r_tilde_D_star).all():
                        nan_inf_counts[s_key]["dose_matching"] += 1

                    # Stage 4: Reconstruction
                    p_D_star = np.zeros_like(Y_hat)
                    p_D_star[active_mask] = np.maximum(Y_hat[active_mask], epsilon) * np.exp(r_tilde_D_star[active_mask])
                    p_D_star[active_mask] /= p_D_star[active_mask].sum()
                    if not np.isfinite(p_D_star).all():
                        nan_inf_counts[s_key]["reconstruction"] += 1

                    # Stage 5: Evaluation (CPC)
                    cpc_matched = orig_fast_eval_cpc(p_D_star, active_mask, Y_hat, t0_inter, t_true_inter, bin_masks, denom, K)
                    if not np.isfinite(cpc_matched):
                        nan_inf_counts[s_key]["evaluation"] += 1

                # Training mean
                nan_inf_counts[s_key]["tm_trials"] += 1
                r_M = orig_safe_log_ratio(train_mean_yd, Y_hat, active_mask, delta=epsilon)
                r_tilde_M = np.zeros_like(r_M)
                r_tilde_M[active_mask] = r_M[active_mask] - np.mean(r_M[active_mask])
                if not np.isfinite(r_M).all() or not np.isfinite(r_tilde_M).all():
                    nan_inf_counts[s_key]["log_ratio"] += 1

                D_M = float(np.sqrt(np.mean(r_tilde_M[active_mask]**2)))
                if not np.isfinite(D_M):
                    nan_inf_counts[s_key]["dose_matching"] += 1

                if D_M < 1e-12:
                    nan_inf_counts[s_key]["tm_fallback"] += 1
                else:
                    r_tilde_M_star = np.zeros_like(r_tilde_M)
                    r_tilde_M_star[active_mask] = r_tilde_M[active_mask] * (D_T / D_M)
                    if not np.isfinite(r_tilde_M_star).all():
                        nan_inf_counts[s_key]["dose_matching"] += 1

                    p_M_star = np.zeros_like(Y_hat)
                    p_M_star[active_mask] = np.maximum(Y_hat[active_mask], epsilon) * np.exp(r_tilde_M_star[active_mask])
                    p_M_star[active_mask] /= p_M_star[active_mask].sum()
                    if not np.isfinite(p_M_star).all():
                        nan_inf_counts[s_key]["reconstruction"] += 1

                    cpc_matched_tm = orig_fast_eval_cpc(p_M_star, active_mask, Y_hat, t0_inter, t_true_inter, bin_masks, denom, K)
                    if not np.isfinite(cpc_matched_tm):
                        nan_inf_counts[s_key]["evaluation"] += 1

    # Step 5: Statistical reconciliation between per-city CSV and summary JSON
    df_city = pd.read_csv("results/unified_placebo_v1/unified_placebo_per_city.csv")
    with open("results/unified_placebo_v1/unified_placebo_reconciled_summary.json", "r") as f:
        summary_json = json.load(f)

    # Calculate macro averages directly from df_city
    rec_stats = {
        "Target - baseline": {
            "recomputed": float(df_city["d_cpc_target"].mean()),
            "summary": summary_json["target"]["mean_delta_cpc"],
            "manuscript": "+0.00354",
        },
        "Donor - baseline": {
            "recomputed": float(df_city["d_cpc_matched"].mean()),
            "summary": summary_json["matched_train_b"]["mean_delta_cpc"],
            "manuscript": "-0.00009",
        },
        "Training mean - baseline": {
            "recomputed": float(df_city["d_cpc_matched_train_mean"].mean()),
            "summary": summary_json["matched_train_mean"]["mean_delta_cpc"],
            "manuscript": "+0.00091",
        },
        "Permutation - baseline": {
            "recomputed": float(df_city["d_cpc_perm"].mean()),
            "summary": summary_json["permuted_b"]["mean_delta_cpc"],
            "manuscript": "-0.00696",
        },
        "Target - donor": {
            "recomputed": float((df_city["d_cpc_target"] - df_city["d_cpc_matched"]).mean()),
            "summary": summary_json["matched_train_b"]["specificity_gain_mean"],
            "manuscript": "+0.00363",
        },
        "Target - training mean": {
            "recomputed": float((df_city["d_cpc_target"] - df_city["d_cpc_matched_train_mean"]).mean()),
            "summary": summary_json["matched_train_mean"]["specificity_gain_mean"],
            "manuscript": "+0.00263",
        },
        "Target - permutation": {
            "recomputed": float((df_city["d_cpc_target"] - df_city["d_cpc_perm"]).mean()),
            "summary": summary_json["permuted_b"]["specificity_gain_mean"],
            "manuscript": "+0.01050",
        },
    }

    out_final = {
        "active_mask_diff_count": mask_diff_count,
        "mask_diff_details": mask_diff_details,
        "step4_counts": {
            "total_distinct_pairs": 50 * 35,
            "pairs_with_zero": len(pairs_with_zero),
            "pairs_with_positive_below_delta": len(pairs_with_positive_below_delta),
            "total_donor_bins": total_donor_bins,
            "donor_bins_zero": donor_bins_zero,
            "donor_bins_positive_below_delta": donor_bins_positive_below_delta,
        },
        "nan_inf_counts": nan_inf_counts,
        "recomputed_statistics_table": rec_stats,
    }

    out_path = Path("results/comprehensive_placebo_audit.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_final, f, indent=2)

    print("COMPREHENSIVE_AUDIT_SUCCESS")
    print(json.dumps(out_final, indent=2))

if __name__ == "__main__":
    run_comprehensive_audit()
