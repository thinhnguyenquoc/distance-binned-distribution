import os
import sys
import json
import time
import argparse
import random
import itertools
import numpy as np
import pandas as pd
import torch
import math
import logging
from pathlib import Path
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.city_splits import generate_35_5_10_splits
from src.data.yd_extractor import compute_kbin_edges, extract_yd_kbins
from src.data.dataset import load_raw_city, load_city
from src.data.urban_graph import build_radius_graph
from src.training.train import load_checkpoint
from src.experiment.run_experiment import infer_zero_shot
from src.training.evaluate import compute_cpc_pair
from src.calibration.bin_calibration import calibrate_kbins

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
        
    y_hat_active = y_hat[active_mask]
    y_hat_active = np.maximum(y_hat_active, delta)
    r = np.zeros_like(p)
    r[active_mask] = np.log(p_active) - np.log(y_hat_active)
    return r

def evaluate_cpc(t_true_inter, t_pred_inter):
    return compute_cpc_pair(t_true_inter, t_pred_inter)

def run_placebo_experiment(args):
    data_root = "data"
    output_dir = "results/placebo_matched_v2"
    os.makedirs(output_dir, exist_ok=True)
    
    log_file = f"{output_dir}/run.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)
    
    splits = generate_35_5_10_splits(data_root=data_root)
    placebo_seed = 20260823
    np.random.seed(placebo_seed)
    epsilon = 1e-12
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    raw_results = []
    B_perm = args.b
    folds_to_run = [2] if args.smoke else [1, 2, 3, 4, 5]
    
    for fold_id in folds_to_run:
        split = splits[fold_id]
        train_cities = split["train"]
        test_cities = split["test"]
        
        if args.smoke:
            test_cities = [test_cities[0]]
            train_cities = train_cities[:2]
            
        logger.info(f"\nProcessing Fold {fold_id}...")
        bin_edges, _ = compute_kbin_edges(split["train"], K=8, data_root=data_root)
        K = len(bin_edges) - 1
        train_yd_dict = {}
        
        for city_name in train_cities:
            raw = load_raw_city(city_name, data_root=data_root)
            dist_km = raw.dist_km
            inter_mask = (raw.pair_o_idx.numpy() != raw.pair_d_idx.numpy()) & (dist_km > 0.0)
            yd = extract_yd_kbins(dist_km, raw.pair_trips.numpy(), bin_edges, inter_mask)
            train_yd_dict[city_name] = yd
            
        train_mean_yd = np.mean(list(train_yd_dict.values()), axis=0)
        model_seeds = [1, 10] if args.smoke else [1, 10, 100]
        
        for c_idx, tc in enumerate(test_cities):
            logger.info(f"  Target City: {tc} ({c_idx+1}/{len(test_cities)})")
            raw = load_raw_city(tc, data_root=data_root)
            dist_km = raw.dist_km
            inter_mask = (raw.pair_o_idx.numpy() != raw.pair_d_idx.numpy()) & (dist_km > 0.0)
            t_true_inter = raw.pair_trips.numpy()[inter_mask]
            
            yd_target = extract_yd_kbins(dist_km, raw.pair_trips.numpy(), bin_edges, inter_mask)
            active_mask = get_active_bins(yd_target)
            target_active_bin_count = np.sum(active_mask)
            
            assert abs(yd_target.sum() - 1.0) < 1e-8, "yd_target prob mass must sum to 1"
            
            edge_index, edge_dist = build_radius_graph(
                lon_lat=raw.lon_lat, 
                radius_km=5.0, 
                include_self_loop=True, 
                cache_key=f"{tc}_tracts"
            )
            
            rng_perms = np.random.RandomState(placebo_seed)
            if math.factorial(target_active_bin_count) <= 40320:
                all_perms = list(itertools.permutations(np.arange(target_active_bin_count)))
                valid_perms = [p for p in all_perms if not np.array_equal(p, np.arange(target_active_bin_count))]
                if len(valid_perms) > B_perm:
                    chosen_perms = rng_perms.choice(len(valid_perms), size=B_perm, replace=False)
                    index_perms = [valid_perms[i] for i in chosen_perms]
                else:
                    index_perms = valid_perms
            else:
                perms_set = set()
                tries = 0
                while len(perms_set) < B_perm and tries < B_perm * 10:
                    tries += 1
                    p = tuple(rng_perms.permutation(np.arange(target_active_bin_count)))
                    if not np.array_equal(p, np.arange(target_active_bin_count)):
                        perms_set.add(p)
                index_perms = list(perms_set)

            for seed in model_seeds:
                logger.info(f"    Evaluating seed {seed}...")
                ckpt_path = Path(f"results/checkpoints/5fold_fold{fold_id}_seed{seed}.pt")
                if not ckpt_path.exists():
                    raise FileNotFoundError(f"Missing mandatory checkpoint {ckpt_path}.")
                    
                model, scaler, _ = load_checkpoint(ckpt_path, device_str="cpu")
                model.eval()
                
                city_data = load_city(tc, data_root=data_root, feature_scaler=scaler, fit_scaler=False)
                t_pred_zs_tensor = infer_zero_shot(model, city_data, edge_index, edge_dist, device="cpu")
                t_pred_zs = t_pred_zs_tensor.numpy().astype(np.float64)
                
                cpc_m0_inter = evaluate_cpc(t_true_inter, t_pred_zs[inter_mask])
                
                t0_inter = t_pred_zs[inter_mask]
                dist_inter = dist_km[inter_mask]
                N_hat = t0_inter.sum()
                
                bin_masks = []
                Y_hat = np.zeros(K, dtype=np.float64)
                if N_hat > 0:
                    for k in range(K):
                        lo, hi = float(bin_edges[k]), float(bin_edges[k+1])
                        in_bin = (dist_inter > lo) & (dist_inter <= hi)
                        bin_masks.append(in_bin)
                        Y_hat[k] = t0_inter[in_bin].sum() / N_hat
                        
                target_bins_with_zero_pred_mass = np.sum(active_mask & (Y_hat <= epsilon))
                model_supported_target_bin_count = target_active_bin_count - target_bins_with_zero_pred_mass
                model_target_bin_support_rate = model_supported_target_bin_count / target_active_bin_count if target_active_bin_count > 0 else 1.0
                
                r_T = safe_log_ratio(yd_target, Y_hat, active_mask, delta=epsilon)
                r_tilde_T = np.zeros_like(r_T)
                r_tilde_T[active_mask] = r_T[active_mask] - np.mean(r_T[active_mask])
                D_T = np.sqrt(np.mean(r_tilde_T[active_mask]**2))
                
                def equivalence_test(p_tgt, name):
                    t_cal_ref = calibrate_kbins(
                        t0_np=t_pred_zs.copy(), 
                        dist_km=dist_km, 
                        inter_mask=inter_mask, 
                        yd_target=p_tgt.copy(), 
                        bin_edges=bin_edges, 
                        q=1.0, 
                        tolerance=1e-5
                    )
                    cpc_ref = evaluate_cpc(t_true_inter, t_cal_ref[inter_mask])
                    
                    ref_bin_mass = np.zeros(K, dtype=np.float64)
                    ref_inter = t_cal_ref[inter_mask]
                    cal_mass_ref = ref_inter.sum()
                    for k in range(K):
                        if cal_mass_ref > 0:
                            ref_bin_mass[k] = ref_inter[bin_masks[k]].sum() / cal_mass_ref
                            
                    p_active = p_tgt[active_mask]
                    p_cond = p_active / np.sum(p_active) if np.sum(p_active) > 0 else Y_hat[active_mask] / np.sum(Y_hat[active_mask])
                    
                    w_raw = np.zeros(target_active_bin_count, dtype=np.float64)
                    n_w_infinite = 0
                    
                    for i, idx in enumerate(np.where(active_mask)[0]):
                        if Y_hat[idx] <= epsilon:
                            w_raw[i] = np.inf
                            n_w_infinite += 1
                        else:
                            w_raw[i] = p_cond[i] / Y_hat[idx]
                            
                    w_finite = w_raw[np.isfinite(w_raw)]
                    w_raw_min = float(np.min(w_finite)) if len(w_finite) > 0 else np.nan
                    w_raw_median = float(np.median(w_finite)) if len(w_finite) > 0 else np.nan
                    w_raw_p95 = float(np.percentile(w_finite, 95)) if len(w_finite) > 0 else np.nan
                    w_raw_max = float(np.max(w_finite)) if len(w_finite) > 0 else np.nan
                    
                    n_w_gt_2 = int((w_finite > 2.0).sum()) + n_w_infinite
                    n_w_gt_5 = int((w_finite > 5.0).sum()) + n_w_infinite
                    n_w_gt_10 = int((w_finite > 10.0).sum()) + n_w_infinite
                    
                    rate_w_gt_2 = n_w_gt_2 / target_active_bin_count
                    rate_w_gt_5 = n_w_gt_5 / target_active_bin_count
                    rate_w_gt_10 = n_w_gt_10 / target_active_bin_count
                    
                    w = np.ones(K, dtype=np.float64)
                    # For fast cal, clip Y_hat dynamically to avoid division by zero
                    y_hat_safe = np.maximum(Y_hat[active_mask], epsilon)
                    w_active = p_cond / y_hat_safe
                    w[active_mask] = w_active
                    
                    weighted_mass = float((Y_hat[active_mask] * w_active).sum())
                    s = np.ones(K, dtype=np.float64)
                    s_active = w_active / weighted_mass if weighted_mass > 0 else np.ones_like(w_active)
                    s[active_mask] = s_active
                    
                    t_cal_fast = t0_inter.copy()
                    for k in range(K):
                        if active_mask[k]:
                            t_cal_fast[bin_masks[k]] *= s[k]
                            
                    cal_mass_fast = t_cal_fast.sum()
                    if cal_mass_fast > 0:
                        t_cal_fast *= (N_hat / cal_mass_fast)
                        
                    cpc_fast = evaluate_cpc(t_true_inter, t_cal_fast)
                    
                    fast_bin_mass = np.zeros(K, dtype=np.float64)
                    for k in range(K):
                        if cal_mass_fast > 0:
                            fast_bin_mass[k] = t_cal_fast[bin_masks[k]].sum() / cal_mass_fast
                            
                    if not np.allclose(fast_bin_mass, ref_bin_mass, atol=1e-10):
                        raise ValueError(f"Equivalence failed for {name}: bin masses differ")
                    if not np.allclose(cpc_fast, cpc_ref, atol=1e-10):
                        raise ValueError(f"Equivalence failed for {name}: cpc differs")
                    if not np.allclose(t_cal_fast, t_cal_ref[inter_mask], atol=1e-10):
                        raise ValueError(f"Equivalence failed for {name}: t_cal differs")
                        
                    stats = {
                        "w_raw_min": w_raw_min, "w_raw_median": w_raw_median, "w_raw_p95": w_raw_p95, "w_raw_max": w_raw_max,
                        "n_w_gt_2": int(n_w_gt_2), "n_w_gt_5": int(n_w_gt_5), "n_w_gt_10": int(n_w_gt_10),
                        "rate_w_gt_2": float(rate_w_gt_2), "rate_w_gt_5": float(rate_w_gt_5), "rate_w_gt_10": float(rate_w_gt_10),
                        "n_w_infinite": int(n_w_infinite),
                        "target_active_bin_count": int(target_active_bin_count),
                        "model_supported_target_bin_count": int(model_supported_target_bin_count),
                        "target_bins_with_zero_pred_mass": int(target_bins_with_zero_pred_mass),
                        "model_target_bin_support_rate": float(model_target_bin_support_rate),
                    }
                    return cpc_fast, stats

                cpc_target, stats_tgt = equivalence_test(yd_target, "target")
                delta_cpc_target = cpc_target - cpc_m0_inter

                def build_row(cond, rep_id, donor_name, cpc_val, stats, dose, donor_stats={}):
                    d_cpc = cpc_val - cpc_m0_inter
                    row = {
                        "fold": int(fold_id), "model_seed": int(seed), "target_city": tc, "condition": cond,
                        "replicate_id": int(rep_id), "donor_city": donor_name, "placebo_seed": int(placebo_seed),
                        "q": 1.0, "D_T": float(D_T), "D_placebo": float(dose), 
                        "dose_error": float(abs(D_T - dose)),
                        "cpc_m0_inter": float(cpc_m0_inter), "cpc_m1_inter": float(cpc_val),
                        "delta_cpc_inter": float(d_cpc), "target_delta_cpc_inter": float(delta_cpc_target),
                        "specificity_gain": float(delta_cpc_target - d_cpc),
                    }
                    row.update(stats)
                    row.update(donor_stats)
                    return row

                raw_results.append(build_row("target", 0, tc, cpc_target, stats_tgt, D_T))

                seen_vecs = []
                p_idx = 0
                for perm_indices in index_perms:
                    r_tilde_P = np.zeros_like(r_tilde_T)
                    r_tilde_P[active_mask] = r_tilde_T[active_mask][list(perm_indices)]
                    
                    is_dup = False
                    for seen in seen_vecs:
                        if np.allclose(seen, r_tilde_P, atol=1e-12):
                            is_dup = True
                            break
                    if is_dup:
                        continue
                    seen_vecs.append(r_tilde_P.copy())
                    
                    D_P = np.sqrt(np.mean(r_tilde_P[active_mask]**2))
                    p_P = np.zeros_like(Y_hat)
                    p_P[active_mask] = np.maximum(Y_hat[active_mask], epsilon) * np.exp(r_tilde_P[active_mask])
                    p_P[active_mask] /= p_P[active_mask].sum()
                    
                    cpc_P, stats_P = equivalence_test(p_P, "permuted_bin")
                    raw_results.append(build_row("permuted", p_idx, "PERMUTED", cpc_P, stats_P, D_P))
                    p_idx += 1

                w_idx = 0
                for donor_name in train_cities:
                    donor_yd = train_yd_dict[donor_name]
                    r_D = safe_log_ratio(donor_yd, Y_hat, active_mask, delta=epsilon)
                    r_tilde_D = np.zeros_like(r_D)
                    r_tilde_D[active_mask] = r_D[active_mask] - np.mean(r_D[active_mask])
                    D_D = np.sqrt(np.mean(r_tilde_D[active_mask]**2))
                    if D_D < 1e-12: continue
                    
                    r_tilde_D_star = np.zeros_like(r_tilde_D)
                    r_tilde_D_star[active_mask] = r_tilde_D[active_mask] * (D_T / D_D)
                    D_D_star = np.sqrt(np.mean(r_tilde_D_star[active_mask]**2))
                    
                    p_D_star = np.zeros_like(Y_hat)
                    p_D_star[active_mask] = np.maximum(Y_hat[active_mask], epsilon) * np.exp(r_tilde_D_star[active_mask])
                    p_D_star[active_mask] /= p_D_star[active_mask].sum()
                    
                    cpc_wrong, stats_wrong = equivalence_test(p_D_star, "wrong_city")
                    
                    donor_target_mass_overlap = float(donor_yd[active_mask].sum())
                    donor_target_bin_overlap_count = int(np.sum((donor_yd > 0) & active_mask))
                    donor_target_bin_overlap_rate = donor_target_bin_overlap_count / target_active_bin_count if target_active_bin_count > 0 else 1.0
                    
                    d_stats = {
                        "donor_target_mass_overlap": donor_target_mass_overlap,
                        "donor_target_bin_overlap_count": donor_target_bin_overlap_count,
                        "donor_target_bin_overlap_rate": donor_target_bin_overlap_rate
                    }
                    raw_results.append(build_row("wrong_city", w_idx, donor_name, cpc_wrong, stats_wrong, D_D_star, d_stats))
                    w_idx += 1

                r_M = safe_log_ratio(train_mean_yd, Y_hat, active_mask, delta=epsilon)
                r_tilde_M = np.zeros_like(r_M)
                r_tilde_M[active_mask] = r_M[active_mask] - np.mean(r_M[active_mask])
                D_M = np.sqrt(np.mean(r_tilde_M[active_mask]**2))
                
                if D_M >= 1e-12:
                    r_tilde_M_star = np.zeros_like(r_tilde_M)
                    r_tilde_M_star[active_mask] = r_tilde_M[active_mask] * (D_T / D_M)
                    D_M_star = np.sqrt(np.mean(r_tilde_M_star[active_mask]**2))
                    
                    p_M_star = np.zeros_like(Y_hat)
                    p_M_star[active_mask] = np.maximum(Y_hat[active_mask], epsilon) * np.exp(r_tilde_M_star[active_mask])
                    p_M_star[active_mask] /= p_M_star[active_mask].sum()
                    
                    cpc_tm, stats_tm = equivalence_test(p_M_star, "train_mean")
                    
                    donor_target_mass_overlap = float(train_mean_yd[active_mask].sum())
                    donor_target_bin_overlap_count = int(np.sum((train_mean_yd > 0) & active_mask))
                    donor_target_bin_overlap_rate = donor_target_bin_overlap_count / target_active_bin_count if target_active_bin_count > 0 else 1.0
                    d_stats = {
                        "donor_target_mass_overlap": donor_target_mass_overlap,
                        "donor_target_bin_overlap_count": donor_target_bin_overlap_count,
                        "donor_target_bin_overlap_rate": donor_target_bin_overlap_rate
                    }
                    raw_results.append(build_row("trainmean", 0, "TRAIN_MEAN", cpc_tm, stats_tm, D_M_star, d_stats))

    df = pd.DataFrame(raw_results)
    df.to_csv(f"{output_dir}/matched_placebo_raw.csv", index=False)
    with open(f"{output_dir}/matched_placebo_raw.jsonl", "w") as f:
        for r in raw_results:
            f.write(json.dumps(r) + "\n")
            
    logger.info(f"Raw results saved to {output_dir}")
    
    agg_cols = {
        "delta_cpc_inter": "mean", "target_delta_cpc_inter": "mean", "specificity_gain": "mean",
        "cpc_m0_inter": "mean", "cpc_m1_inter": "mean",
        "w_raw_min": "mean", "w_raw_median": "mean", "w_raw_p95": "mean", "w_raw_max": "mean",
        "n_w_gt_2": "mean", "n_w_gt_5": "mean", "n_w_gt_10": "mean",
        "rate_w_gt_2": "mean", "rate_w_gt_5": "mean", "rate_w_gt_10": "mean",
        "n_w_infinite": "mean",
        "target_active_bin_count": "mean",
        "model_supported_target_bin_count": "mean",
        "target_bins_with_zero_pred_mass": "mean",
        "model_target_bin_support_rate": "mean",
        "donor_target_mass_overlap": "mean",
        "donor_target_bin_overlap_count": "mean",
        "donor_target_bin_overlap_rate": "mean",
        "D_placebo": "mean", "D_T": "mean"
    }
    
    # Fill missing donor stats for non-donor rows with 0/NaN appropriately for mean
    for col in ["donor_target_mass_overlap", "donor_target_bin_overlap_count", "donor_target_bin_overlap_rate"]:
        if col not in df.columns:
            df[col] = np.nan
    
    agg_df = df.groupby(["fold", "target_city", "condition", "replicate_id"]).agg(agg_cols).reset_index()
    agg_df.to_csv(f"{output_dir}/matched_placebo_seed_averaged.csv", index=False)
    
    city_stats = []
    for tc in agg_df.target_city.unique():
        c_df = agg_df[agg_df.target_city == tc]
        fold_val = c_df.fold.values[0]
        
        target_df = c_df[c_df.condition == "target"]
        target_val = target_df["delta_cpc_inter"].values[0] if len(target_df) > 0 else np.nan
        
        tm_df = c_df[c_df.condition == "trainmean"]
        trainmean_val = tm_df["delta_cpc_inter"].values[0] if len(tm_df) > 0 else np.nan
        
        wrong_df = c_df[c_df.condition == "wrong_city"]
        wrong_delta_mean = wrong_df["delta_cpc_inter"].mean() if len(wrong_df) > 0 else np.nan
        
        perm_df = c_df[c_df.condition == "permuted"]
        permuted_delta_mean = perm_df["delta_cpc_inter"].mean() if len(perm_df) > 0 else np.nan
            
        city_stats.append({
            "fold": fold_val, "city": tc,
            "target_delta_mean": target_val, "trainmean_delta_mean": trainmean_val,
            "wrong_delta_mean": wrong_delta_mean, "permuted_delta_mean": permuted_delta_mean,
            "specificity_wrong_mean": target_val - wrong_delta_mean if not np.isnan(wrong_delta_mean) else np.nan,
            "specificity_trainmean_mean": target_val - trainmean_val if not np.isnan(trainmean_val) else np.nan,
            "specificity_permuted_mean": target_val - permuted_delta_mean if not np.isnan(permuted_delta_mean) else np.nan,
            "n_permutations": len(perm_df),
            "target_w_raw_max": target_df["w_raw_max"].mean() if len(target_df) > 0 else np.nan,
            "target_w_raw_median": target_df["w_raw_median"].mean() if len(target_df) > 0 else np.nan,
            "target_w_raw_p95": target_df["w_raw_p95"].mean() if len(target_df) > 0 else np.nan,
            "target_n_w_gt_10": target_df["n_w_gt_10"].mean() if len(target_df) > 0 else np.nan,
            "target_rate_w_gt_10": target_df["rate_w_gt_10"].mean() if len(target_df) > 0 else np.nan,
            "target_n_w_infinite": target_df["n_w_infinite"].mean() if len(target_df) > 0 else np.nan,
            "target_active_bin_count": target_df["target_active_bin_count"].mean() if len(target_df) > 0 else np.nan,
            "target_model_target_bin_support_rate": target_df["model_target_bin_support_rate"].mean() if len(target_df) > 0 else np.nan,
            "target_bins_with_zero_pred_mass": target_df["target_bins_with_zero_pred_mass"].mean() if len(target_df) > 0 else np.nan,
            "wrong_donor_target_mass_overlap": wrong_df["donor_target_mass_overlap"].mean() if len(wrong_df) > 0 else np.nan,
            "wrong_donor_target_bin_overlap_rate": wrong_df["donor_target_bin_overlap_rate"].mean() if len(wrong_df) > 0 else np.nan,
            "target_dose": target_df["D_T"].mean() if len(target_df) > 0 else np.nan,
        })
        
    city_df = pd.DataFrame(city_stats)
    city_df.to_csv(f"{output_dir}/matched_placebo_per_city.csv", index=False)
    
    with open(f"{output_dir}/interpretation.md", "w", encoding="utf-8") as f:
        f.write("### Interpretation\n")
        f.write("> Permuted-bin chỉ kiểm tra mức độ nhạy của multiplicative calibration khi phân phối mass bị gán sai giữa các bin. Nó không ước lượng giá trị thông tin thực tế của Y_D, ngay cả sau khi các kiểm định kỹ thuật đều pass.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--b", type=int, default=1000)
    args = parser.parse_args()
    if args.smoke: args.b = 20
    run_placebo_experiment(args)
