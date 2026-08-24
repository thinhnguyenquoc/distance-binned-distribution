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

# Ensure root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.city_splits import generate_35_5_10_splits
from src.data.yd_extractor import compute_kbin_edges, extract_yd_kbins
from src.data.dataset import load_raw_city, load_city, assign_bins
from src.data.urban_graph import build_radius_graph
from src.training.train import load_checkpoint
from src.experiment.run_experiment import infer_zero_shot
from src.training.evaluate import compute_cpc_pair
from src.calibration.bin_calibration import calibrate_kbins

def get_active_bins(yd, eps=1e-8):
    return yd > eps

def mask_and_renormalize(donor_yd, active_mask):
    masked = donor_yd * active_mask
    s = masked.sum()
    if s <= 0:
        # Fallback to uniform on active bins if donor has 0 mass there
        num_active = active_mask.sum()
        if num_active > 0:
            return np.where(active_mask, 1.0 / num_active, 0.0)
        else:
            # If target has NO active bins (should not happen), return uniform over all
            return np.ones_like(donor_yd) / len(donor_yd)
    return masked / s

def get_permutations(yd, active_mask, max_perms=1000, seed=42):
    active_indices = np.where(active_mask)[0]
    active_vals = yd[active_indices]
    num_active = len(active_indices)
    
    # If the total number of permutations is small enough (<= 8!), generate all and sample
    if math.factorial(num_active) <= 40320:
        all_perms = set(itertools.permutations(active_vals))
        valid_perms = [p for p in all_perms if not np.allclose(p, active_vals)]
        if len(valid_perms) > max_perms:
            rng = np.random.RandomState(seed)
            indices = rng.choice(len(valid_perms), size=max_perms, replace=False)
            perms = [valid_perms[i] for i in indices]
        else:
            perms = valid_perms
    else:
        # For large K, randomly sample to avoid memory overflow and infinite loops
        rng = np.random.RandomState(seed)
        perms_set = set()
        tries = 0
        max_tries = max_perms * 10
        while len(perms_set) < max_perms and tries < max_tries:
            tries += 1
            p = tuple(rng.permutation(active_vals))
            if not np.allclose(p, active_vals):
                perms_set.add(p)
        perms = list(perms_set)
        
    res = []
    for p in perms:
        new_yd = yd.copy()
        new_yd[active_indices] = p
        res.append(new_yd)
    return res

def evaluate_cpc(t_true_inter, t_pred_inter):
    return compute_cpc_pair(t_true_inter, t_pred_inter)

def bootstrap_ci(data, n_boot=10000, seed=42):
    rng = np.random.RandomState(seed)
    means = []
    n = len(data)
    for _ in range(n_boot):
        sample = rng.choice(data, size=n, replace=True)
        means.append(np.mean(sample))
    return np.percentile(means, 2.5), np.percentile(means, 97.5)

def run_placebo_experiment(args):
    data_root = "data"
    output_dir = "results/placebo_v1"
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    log_file = f"{output_dir}/run.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Load 5-fold splits
    splits = generate_35_5_10_splits(data_root=data_root)
    
    # Load reference results to ensure M0 matching
    ref_results_path = Path("results/5fold_results.json")
    if ref_results_path.exists():
        with open(ref_results_path, "r") as f:
            ref_data = json.load(f)
        ref_city_res = {item["city"]: item for item in ref_data.get("city_level_results", [])}
    else:
        logger.warning("Warning: 5fold_results.json not found, skipping M0 validation.")
        ref_city_res = {}

    placebo_seed = 20260821
    np.random.seed(placebo_seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    raw_results = []
    
    B = args.b
    folds_to_run = [2] if args.smoke else [1, 2, 3, 4, 5]
    
    for fold_id in folds_to_run:
        split = splits[fold_id]
        train_cities = split["train"]
        test_cities = split["test"]
        
        if args.smoke:
            test_cities = [test_cities[0]] # Just one city
            
        logger.info(f"\nProcessing Fold {fold_id}...")
        
        # Load train Y_Ds for donor selection and train mean
        bin_edges, _ = compute_kbin_edges(train_cities, K=8, data_root=data_root)
        K = len(bin_edges) - 1
        train_yd_dict = {}
        
        logger.info("  Extracting donor Y_Ds...")
        for city_name in train_cities:
            raw = load_raw_city(city_name, data_root=data_root)
            dist_km = raw.dist_km
            inter_mask = (raw.pair_o_idx.numpy() != raw.pair_d_idx.numpy()) & (dist_km > 0.0)
            yd = extract_yd_kbins(dist_km, raw.pair_trips.numpy(), bin_edges, inter_mask)
            train_yd_dict[city_name] = yd
            
        train_mean_yd = np.mean(list(train_yd_dict.values()), axis=0)
        
        # Pre-sample donors for each test city so it's consistent across seeds
        donor_assignments = {}
        for tc in test_cities:
            valid_donors = [c for c in train_cities if c != tc]
            donor_assignments[tc] = np.random.choice(valid_donors, size=B, replace=True)
            
        model_seeds = [1, 10, 100]
        
        for c_idx, tc in enumerate(test_cities):
            logger.info(f"  Target City: {tc} ({c_idx+1}/{len(test_cities)})")
            raw = load_raw_city(tc, data_root=data_root)
            dist_km = raw.dist_km
            inter_mask = (raw.pair_o_idx.numpy() != raw.pair_d_idx.numpy()) & (dist_km > 0.0)
            t_true_inter = raw.pair_trips.numpy()[inter_mask]
            
            yd_target = extract_yd_kbins(dist_km, raw.pair_trips.numpy(), bin_edges, inter_mask)
            active_mask = get_active_bins(yd_target)
            
            # Ensure prob mass sums to 1
            assert abs(yd_target.sum() - 1.0) < 1e-8, "yd_target prob mass must sum to 1"
            
            # Prepare permuted YDs
            permuted_yds = get_permutations(yd_target, active_mask, max_perms=B, seed=placebo_seed)
            
            # Prepare donor YDs
            donor_yds = []
            for d_name in donor_assignments[tc]:
                d_yd = train_yd_dict[d_name]
                d_yd_masked = mask_and_renormalize(d_yd, active_mask)
                assert abs(d_yd_masked.sum() - 1.0) < 1e-8, f"Donor {d_name} prob mass must sum to 1 after renormalization"
                donor_yds.append((d_name, d_yd_masked))
                
            # Train mean YD
            train_mean_yd_masked = mask_and_renormalize(train_mean_yd, active_mask)
            
            # Graph
            edge_index, edge_dist = build_radius_graph(
                lon_lat=raw.lon_lat, 
                radius_km=5.0, 
                include_self_loop=True, 
                cache_key=f"{tc}_tracts"
            )
            
            seeds_run = 0
            for seed in model_seeds:
                logger.info(f"    Evaluating seed {seed}...")
                ckpt_path = Path(f"results/checkpoints/5fold_fold{fold_id}_seed{seed}.pt")
                if not ckpt_path.exists():
                    raise FileNotFoundError(f"Missing mandatory checkpoint {ckpt_path}. Placebo test requires all seeds to be present.")
                    
                model, scaler, _ = load_checkpoint(ckpt_path, device_str="cpu")
                model.eval()
                seeds_run += 1
                
                # Zero-shot inference
                city_data = load_city(tc, data_root=data_root, feature_scaler=scaler, fit_scaler=False)
                t_pred_zs_tensor = infer_zero_shot(model, city_data, edge_index, edge_dist, device="cpu")
                t_pred_zs = t_pred_zs_tensor.numpy().astype(np.float64)
                
                cpc_m0_inter = evaluate_cpc(t_true_inter, t_pred_zs[inter_mask])
                
                # Precompute bin masks and Y_hat for fast calibration
                t0_inter = t_pred_zs[inter_mask]
                dist_inter = dist_km[inter_mask]
                N_hat = t0_inter.sum()
                
                bin_masks = []
                Y_hat = np.zeros(K, dtype=np.float64)
                active = np.zeros(K, dtype=bool)
                if N_hat > 0:
                    for k in range(K):
                        lo, hi = float(bin_edges[k]), float(bin_edges[k+1])
                        in_bin = (dist_inter > lo) & (dist_inter <= hi)
                        bin_masks.append(in_bin)
                        Y_hat[k] = t0_inter[in_bin].sum() / N_hat
                        active[k] = bool(in_bin.any())
                
                def fast_cal_cpc(yd_tgt):
                    default_stats = {
                        "w_max": 1.0, "w_median": 1.0,
                        "s_min": 1.0, "s_max": 1.0, "s_median": 1.0, "s_p95": 1.0,
                        "s_gt_2": 0, "s_gt_5": 0, "s_gt_10": 0, 
                        "model_compat": 1.0, "donor_compat": 1.0
                    }
                    if N_hat <= 0:
                        return cpc_m0_inter, default_stats
                    
                    yd_raw = yd_tgt / yd_tgt.sum() if yd_tgt.sum() > 0 else np.ones(K)/K
                    yd_active = yd_raw * active.astype(np.float64)
                    active_sum = yd_active.sum()
                    Y_D_cond = yd_active / active_sum if active_sum > 0 else Y_hat.copy()
                    
                    w = np.ones(K, dtype=np.float64)
                    for k in range(K):
                        if active[k] and Y_hat[k] > 0:
                            w[k] = Y_D_cond[k] / Y_hat[k]  # q=1.0
                            
                    weighted_mass = float((Y_hat * w).sum())
                    s = w / weighted_mass if weighted_mass > 0 else np.ones(K)
                    
                    # We only need CPC on interzonal pairs, so we calibrate t0_inter directly
                    t_cal_inter = t0_inter.copy()
                    for k in range(K):
                        if active[k]:
                            t_cal_inter[bin_masks[k]] *= s[k]
                            
                    cal_mass = t_cal_inter.sum()
                    if cal_mass > 0:
                        t_cal_inter *= (N_hat / cal_mass)
                        s_final = s * (N_hat / cal_mass)
                    else:
                        s_final = s
                        
                    cpc = evaluate_cpc(t_true_inter, t_cal_inter)
                    
                    active_s = s_final[active]
                    active_w = w[active]
                    if len(active_s) > 0:
                        stats = {
                            "w_max": float(active_w.max()),
                            "w_median": float(np.median(active_w)),
                            "s_min": float(active_s.min()),
                            "s_max": float(active_s.max()),
                            "s_median": float(np.median(active_s)),
                            "s_p95": float(np.percentile(active_s, 95)),
                            "s_gt_2": int((active_s > 2.0).sum()),
                            "s_gt_5": int((active_s > 5.0).sum()),
                            "s_gt_10": int((active_s > 10.0).sum()),
                        }
                    else:
                        stats = default_stats.copy()
                        stats["w_max"] = 1.0
                        stats["w_median"] = 1.0
                        
                    # Model compatibility: fraction of calibration-target active bins that model predicted > 0
                    target_active_bins = yd_tgt > 0
                    model_active_bins = Y_hat > 0
                    stats["model_compat"] = float(np.sum(target_active_bins & model_active_bins) / np.sum(target_active_bins)) if np.sum(target_active_bins) > 0 else 1.0
                    
                    return cpc, stats
                
                # Correct target condition
                cpc_target_inter, stats_tgt = fast_cal_cpc(yd_target)
                delta_cpc_target = cpc_target_inter - cpc_m0_inter
                
                # Train mean condition
                cpc_trainmean_inter, stats_tm = fast_cal_cpc(train_mean_yd_masked)
                delta_cpc_trainmean = cpc_trainmean_inter - cpc_m0_inter
                
                # Helper to build dict
                def build_row(cond, rep_id, d_name, cpc_val, stats, d_compat):
                    d_cpc = cpc_val - cpc_m0_inter
                    row = {
                        "fold": int(fold_id), "model_seed": int(seed), "target_city": tc, "condition": cond,
                        "replicate_id": int(rep_id), "donor_city": d_name, "placebo_seed": int(placebo_seed),
                        "n_tracts": int(raw.n_tracts), "n_inter_pairs": int(inter_mask.sum()), "K": int(K), "active_bins": int(active_mask.sum()),
                        "q": 1.0, "cpc_m0_inter": float(cpc_m0_inter), "cpc_m1_inter": float(cpc_val),
                        "delta_cpc_inter": float(d_cpc), "target_delta_cpc_inter": float(delta_cpc_target), 
                        "specificity_gain": float(delta_cpc_target - d_cpc),
                        "donor_compat": float(d_compat)
                    }
                    row.update(stats)
                    return row

                # Record Target and Trainmean
                raw_results.append(build_row("target", 0, tc, cpc_target_inter, stats_tgt, 1.0))
                # donor_compat for trainmean is the mass of trainmean that falls into the target's active bins
                tm_compat = train_mean_yd[active_mask].sum()
                raw_results.append(build_row("trainmean", 0, "TRAIN_MEAN", cpc_trainmean_inter, stats_tm, tm_compat))
                
                # Wrong-city
                for b_idx, (d_name, d_yd_masked) in enumerate(donor_yds):
                    cpc_wrong, stats_wrong = fast_cal_cpc(d_yd_masked)
                    # For wrong city, we need the original donor Y_D to compute compat.
                    # Since we only stored masked donor Y_Ds in donor_yds, let's fetch it from train_yd_dict
                    orig_d_yd = train_yd_dict[d_name]
                    d_compat = orig_d_yd[active_mask].sum()
                    raw_results.append(build_row("wrong_city", b_idx, d_name, cpc_wrong, stats_wrong, d_compat))
                    
                # Permuted
                for p_idx, p_yd in enumerate(permuted_yds):
                    cpc_perm, stats_perm = fast_cal_cpc(p_yd)
                    raw_results.append(build_row("permuted", p_idx, "PERMUTED", cpc_perm, stats_perm, 1.0))

            assert seeds_run == len(model_seeds), f"Expected to run {len(model_seeds)} seeds, but ran {seeds_run}"

    df = pd.DataFrame(raw_results)
    df.to_csv(f"{output_dir}/placebo_raw.csv", index=False)
    with open(f"{output_dir}/placebo_raw.json", "w") as f:
        json.dump(raw_results, f, indent=2)
        
    logger.info(f"Raw results saved to {output_dir}")
    
    # Process Seed-Aggregated
    agg_df = df.groupby(["fold", "target_city", "condition", "replicate_id"]).agg({
        "delta_cpc_inter": "mean",
        "target_delta_cpc_inter": "mean",
        "specificity_gain": "mean",
        "cpc_m0_inter": "mean",
        "cpc_m1_inter": "mean",
        "w_max": "mean",
        "s_max": "mean",
        "model_compat": "mean",
        "donor_compat": "mean"
    }).reset_index()
    
    # M0 Verification
    m0_check_failed = False
    for tc in agg_df.target_city.unique():
        if tc in ref_city_res:
            ref_m0 = ref_city_res[tc]["M0"]["cpc_inter"]
            tc_m0 = agg_df[(agg_df.target_city == tc) & (agg_df.condition == "target")]["cpc_m0_inter"].values[0]
            if abs(ref_m0 - tc_m0) > 1e-6:
                logger.error(f"M0 mismatch for {tc}: ref={ref_m0}, new={tc_m0}")
                m0_check_failed = True
            
            ref_delta = ref_city_res[tc]["delta_city"]
            tc_delta = agg_df[(agg_df.target_city == tc) & (agg_df.condition == "target")]["delta_cpc_inter"].values[0]
            if abs(ref_delta - tc_delta) > 1e-6:
                logger.error(f"Delta mismatch for {tc}: ref={ref_delta}, new={tc_delta}")
                m0_check_failed = True

    if m0_check_failed and not args.smoke:
        logger.warning("Preflight checks failed! M0 or Delta CPC does not match 5fold_results.json.")
        
    city_stats = []
    
    for tc in agg_df.target_city.unique():
        c_df = agg_df[agg_df.target_city == tc]
        fold_val = c_df.fold.values[0]
        
        target_val = c_df[c_df.condition == "target"]["delta_cpc_inter"].values[0]
        
        seed_target = df[(df.target_city == tc) & (df.condition == "target")]
        target_delta_seed_sd = seed_target["delta_cpc_inter"].std(ddof=1) if len(seed_target) > 1 else 0.0
        
        trainmean_val = c_df[c_df.condition == "trainmean"]["delta_cpc_inter"].values[0]
        
        wrong_df = c_df[c_df.condition == "wrong_city"]
        wrong_delta_mean = wrong_df["delta_cpc_inter"].mean()
        wrong_delta_p2_5 = wrong_df["delta_cpc_inter"].quantile(0.025)
        wrong_delta_p97_5 = wrong_df["delta_cpc_inter"].quantile(0.975)
        pseudo_p_wrong = (1 + (wrong_df["delta_cpc_inter"] >= target_val).sum()) / (1 + len(wrong_df))
        specificity_wrong_mean = target_val - wrong_delta_mean
        
        perm_df = c_df[c_df.condition == "permuted"]
        n_permutations = len(perm_df)
        if len(perm_df) > 0:
            permuted_delta_mean = perm_df["delta_cpc_inter"].mean()
            pseudo_p_perm = (1 + (perm_df["delta_cpc_inter"] >= target_val).sum()) / (1 + len(perm_df))
            specificity_permuted_mean = target_val - permuted_delta_mean
        else:
            permuted_delta_mean = np.nan
            pseudo_p_perm = np.nan
            specificity_permuted_mean = np.nan
            
        target_df = c_df[c_df.condition == "target"]
            
        city_stats.append({
            "fold": fold_val, "city": tc,
            "target_delta_mean": target_val,
            "target_delta_seed_sd": target_delta_seed_sd,
            "trainmean_delta_mean": trainmean_val,
            "wrong_delta_mean": wrong_delta_mean,
            "wrong_delta_p2_5": wrong_delta_p2_5,
            "wrong_delta_p97_5": wrong_delta_p97_5,
            "pseudo_p_wrong": pseudo_p_wrong,
            "permuted_delta_mean": permuted_delta_mean,
            "pseudo_p_perm": pseudo_p_perm,
            "specificity_wrong_mean": specificity_wrong_mean,
            "specificity_trainmean_mean": target_val - trainmean_val,
            "specificity_permuted_mean": specificity_permuted_mean,
            "target_beats_wrong": int(target_val > wrong_delta_mean),
            "target_beats_trainmean": int(target_val > trainmean_val),
            "target_beats_permuted": int(target_val > permuted_delta_mean) if len(perm_df) > 0 else 1,
            "n_permutations": n_permutations,
            "target_w_max": target_df["w_max"].mean() if len(target_df) > 0 else np.nan,
            "target_model_compat": target_df["model_compat"].mean() if len(target_df) > 0 else np.nan,
            "wrong_w_max": wrong_df["w_max"].mean() if len(wrong_df) > 0 else np.nan,
            "wrong_s_max": wrong_df["s_max"].mean() if len(wrong_df) > 0 else np.nan,
            "wrong_donor_compat": wrong_df["donor_compat"].mean() if len(wrong_df) > 0 else np.nan,
            "perm_w_max": perm_df["w_max"].mean() if len(perm_df) > 0 else np.nan,
            "perm_s_max": perm_df["s_max"].mean() if len(perm_df) > 0 else np.nan,
            "perm_donor_compat": perm_df["donor_compat"].mean() if len(perm_df) > 0 else np.nan
        })
        
    city_df = pd.DataFrame(city_stats)
    city_df.to_csv(f"{output_dir}/placebo_per_city.csv", index=False)
    
    # Generate summary & plots
    if not args.smoke:
        generate_summary(city_df, df)

def generate_summary(city_df, raw_df):
    output_dir = "results/placebo_v1"
    
    evaluated_df = city_df[city_df.fold.isin([1, 2, 3, 4, 5])]
        
    summary = {
        "evaluated_n_cities": len(evaluated_df),
        "exploratory_n_cities": len(exploratory_df),
        "primary_test": {},
        "secondary_tests": {}
    }
    
    # Primary Test: H0: E[S_wrong] <= 0
    s_wrong = evaluated_df["specificity_wrong_mean"].values
    mean_s_wrong = float(np.mean(s_wrong))
    sd_s_wrong = float(np.std(s_wrong, ddof=1))
    ci_lower, ci_upper = bootstrap_ci(s_wrong)
    
    # Wilcoxon guards for zero-variance or identical samples
    try:
        _, p_val_one = wilcoxon(s_wrong, alternative="greater")
        _, p_val_two = wilcoxon(s_wrong, alternative="two-sided")
    except ValueError:
        p_val_one, p_val_two = 1.0, 1.0
    
    summary["primary_test"] = {
        "mean_specificity_wrong": mean_s_wrong,
        "sd_specificity_wrong": sd_s_wrong,
        "ci_95_lower": float(ci_lower),
        "ci_95_upper": float(ci_upper),
        "pct_greater_than_0": float(np.mean(s_wrong > 0)),
        "wilcoxon_one_sided_p": float(p_val_one),
        "wilcoxon_two_sided_p": float(p_val_two),
        "pass": bool(mean_s_wrong > 0 and ci_lower > 0 and p_val_one < 0.05)
    }
    
    # Train mean test
    s_trainmean = evaluated_df["specificity_trainmean_mean"].values
    mean_s_trainmean = float(np.mean(s_trainmean))
    try:
        _, p_val_trainmean = wilcoxon(s_trainmean, alternative="greater")
    except ValueError:
        p_val_trainmean = 1.0
    
    # Permuted test
    s_perm = evaluated_df["specificity_permuted_mean"].dropna().values
    mean_s_perm = float(np.mean(s_perm)) if len(s_perm) > 0 else np.nan
    
    if len(s_perm) > 0:
        try:
            _, p_val_perm = wilcoxon(s_perm, alternative="greater")
        except ValueError:
            p_val_perm = 1.0
    else:
        p_val_perm = 1.0
        
    summary["secondary_tests"] = {
        "trainmean": {
            "mean_specificity": mean_s_trainmean,
            "wilcoxon_one_sided_p": float(p_val_trainmean)
        },
        "permuted": {
            "mean_specificity": mean_s_perm,
            "wilcoxon_one_sided_p": float(p_val_perm)
        }
    }
    
    with open(f"{output_dir}/placebo_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    with open(f"{output_dir}/placebo_summary.md", "w") as f:
        f.write("# 5-Fold Target-Y_D Placebo Test Summary\n\n")
        f.write("## Full 5-Fold Statistics (Folds 1-5)\n")
        f.write(f"- Number of Cities: {len(evaluated_df)}\n")
        f.write(f"- Mean Specificity Gain vs Wrong City: {mean_s_wrong:.5f} (95% CI: [{ci_lower:.5f}, {ci_upper:.5f}])\n")
        f.write(f"- Primary Wilcoxon one-sided p-value: {p_val_one:.2e}\n")
        f.write(f"- Passed primary test: {summary['primary_test']['pass']}\n")
        
    # Generate Figures
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Figure 1: Distributions
    plt.figure(figsize=(10, 6))
    sns.kdeplot(evaluated_df["target_delta_mean"], label="Target", fill=True)
    sns.kdeplot(evaluated_df["wrong_delta_mean"], label="Wrong City Placebo Mean", fill=True)
    sns.kdeplot(evaluated_df["permuted_delta_mean"].dropna(), label="Permuted Placebo Mean", fill=True)
    plt.axvline(0, color='k', linestyle='--', alpha=0.5)
    plt.title("Distribution of Delta CPC (Target vs Placebos)")
    plt.xlabel("Delta CPC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig_target_vs_placebo.png")
    
    # Figure 2: Specificity by City
    plt.figure(figsize=(12, 6))
    df_plot = evaluated_df.sort_values("specificity_wrong_mean").reset_index()
    sns.barplot(data=df_plot, x=df_plot.index, y="specificity_wrong_mean", hue="fold", dodge=False)
    plt.axhline(0, color='red', linestyle='--')
    plt.xticks([])
    plt.title("Specificity Gain (S_wrong) by City")
    plt.xlabel("City (Sorted)")
    plt.ylabel("Specificity Gain (Delta CPC_target - Delta CPC_wrong)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig_specificity_by_city.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run smoke test")
    parser.add_argument("--b", type=int, default=1000, help="Placebo replicates")
    args = parser.parse_args()
    
    if args.smoke:
        args.b = 20
        
    run_placebo_experiment(args)
