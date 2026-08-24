"""
Partial-OD Information Equivalence Experiment (v1)
===================================================

Scientific Research Question:
    Under the same frozen zero-shot model and distance-bin calibration operator,
    what fraction of directly observed positive interzonal OD pairs is required
    to achieve reconstruction gain comparable to that obtained from the full
    target-city distance-binned mobility distribution?

Key Protocol Invariants:
    - 5-Fold Cross-City Evaluation (50 held-out test cities).
    - Model Seeds: {1, 10, 100} on frozen Gravity-Informed Urban GNN.
    - Zero retraining, fine-tuning, optimizer step, or backward pass.
    - K = 8 moving distance bins, q = 1.0 multiplier scaling.
    - Evaluation Support: strictly unseen positive interzonal pairs U = Omega_c^+ \ S_p.
    - Sampling: Uniform random sampling without replacement over positive interzonal pairs.
    - Nested permutation masks across p grid.
    - Shared masks across all 3 model seeds.
    - Statistical Unit: strictly the city (N = 50).
"""

import os
import sys
import time
import json
import hashlib
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt
import torch

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.city_splits import generate_35_5_10_splits
from src.data.dataset import load_city, load_cities, load_raw_city
from src.data.urban_graph import build_radius_graph
from src.data.yd_extractor import compute_kbin_edges, extract_yd_kbins
from src.calibration.bin_calibration import calibrate_kbins
from src.training.evaluate import compute_cpc_pair
from src.training.train import load_checkpoint, infer_zero_shot

PARTIAL_OD_BASE_SEED = 202608231
DEFAULT_GRID = [0.0, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80]


def get_stable_mask_seed(base_seed: int, fold: int, city: str, replicate_id: int) -> int:
    s = f"{base_seed}_{fold}_{city}_{replicate_id}"
    return int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % (2**32)


def holm_correction(p_vals: List[float]) -> np.ndarray:
    n = len(p_vals)
    sorted_indices = np.argsort(p_vals)
    adj_p = np.zeros(n)
    running_max = 0.0
    for i, idx in enumerate(sorted_indices):
        p_adj = p_vals[idx] * (n - i)
        running_max = max(running_max, p_adj)
        adj_p[idx] = min(1.0, running_max)
    return adj_p


def fold_stratified_bootstrap(
    city_df: pd.DataFrame, 
    metric_col: str, 
    p_val: float, 
    n_boot: int = 10000, 
    seed: int = 42
) -> Tuple[float, float]:
    rng = np.random.RandomState(seed)
    sub = city_df[city_df.p == p_val]
    
    vals: Dict[int, np.ndarray] = {}
    for f in range(1, 6):
        f_vals = sub[sub.fold == f][metric_col].values
        if len(f_vals) > 0:
            vals[f] = f_vals

    boot_means = np.empty(n_boot, dtype=np.float64)
    total_cities = sum(len(v) for v in vals.values())
    
    for b in range(n_boot):
        sample_sum = 0.0
        for f, arr in vals.items():
            idx = rng.randint(0, len(arr), size=len(arr))
            sample_sum += arr[idx].sum()
        boot_means[b] = sample_sum / total_cities

    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def run_partial_od_experiment(
    data_root: str = "data",
    output_dir: str = "results/partial_od_equivalence_v1",
    replicates: int = 500,
    p_grid: List[float] = None,
    smoke: bool = False,
    device: str = "cpu"
) -> None:
    if p_grid is None:
        p_grid = DEFAULT_GRID.copy()

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "figures").mkdir(parents=True, exist_ok=True)

    splits = generate_35_5_10_splits(data_root=data_root)
    model_seeds = [1, 10, 100] if not smoke else [1, 10]
    folds_to_run = list(range(1, 6)) if not smoke else [1]
    B = replicates if not smoke else 20

    print("=" * 85)
    print("PARTIAL-OD INFORMATION EQUIVALENCE EXPERIMENT (OPERATOR-MATCHED)")
    print(f"Folds: {folds_to_run} | Cities: {'All 50' if not smoke else 'Smoke 1'} | Replicates (B): {B}")
    print(f"Grid p: {p_grid} | Seeds: {model_seeds} | Output: {out_path}")
    print("=" * 85)

    # Cache for precalculated zero-shot models and scalers per fold
    fold_models: Dict[int, Dict[int, Tuple[Any, Any]]] = {}
    
    for f in folds_to_run:
        fold_models[f] = {}
        for s in model_seeds:
            ckpt_path = Path("results/checkpoints") / f"5fold_fold{f}_seed{s}.pt"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Missing required GNN checkpoint: {ckpt_path}")
            model, scaler, _ = load_checkpoint(ckpt_path, device_str=device)
            model.eval()
            fold_models[f][s] = (model, scaler)

    raw_records = []
    start_time = time.perf_counter()

    for fold_id in folds_to_run:
        split = splits[fold_id]
        train_cities = split["train"]
        test_cities = split["test"] if not smoke else split["test"][:1]
        
        # Compute K=8 bin edges from 35 train cities
        bin_edges, K_act = compute_kbin_edges(train_cities, K=8, data_root=data_root)
        assert K_act == 8 and len(bin_edges) == 9

        for city_idx, city_name in enumerate(test_cities):
            city_start = time.perf_counter()
            raw_data = load_raw_city(city_name, data_root=data_root)
            dist_km = raw_data.dist_km
            
            # Support Omega_c^+: strictly positive interzonal pairs
            inter_pos = (raw_data.pair_o_idx.numpy() != raw_data.pair_d_idx.numpy()) & (dist_km > 0.0) & (raw_data.pair_trips.numpy() > 0)
            n_pairs = int(inter_pos.sum())
            if n_pairs == 0:
                print(f"Warning: City {city_name} has 0 positive interzonal pairs. Skipping.")
                continue

            t_true_support = raw_data.pair_trips.numpy()[inter_pos].astype(np.float64)
            dist_support = dist_km[inter_pos]
            bin_idx_support = np.clip(np.digitize(dist_support, bin_edges) - 1, 0, 7)
            total_trip_mass = float(np.sum(t_true_support))
            
            # Extract clean full Y_D on support
            yd_full = np.bincount(bin_idx_support, weights=t_true_support, minlength=8).astype(np.float64)
            yd_full /= total_trip_mass

            # Precalculate M0 and full Y_D calibrated prediction for all model seeds
            seed_predictions: Dict[int, Dict[str, np.ndarray]] = {}
            for s in model_seeds:
                model, scaler = fold_models[fold_id][s]
                city_data = load_city(city_name, data_root=data_root, feature_scaler=scaler, fit_scaler=False)
                coords = city_data.lon_lat.numpy()
                ei, ed = build_radius_graph(coords, radius_km=5.0)
                
                with torch.no_grad():
                    m0_full = infer_zero_shot(model, city_data, ei, ed, device=device).numpy().astype(np.float64)
                
                t0_support = m0_full[inter_pos]
                N_hat_support = float(np.sum(t0_support))
                
                # Precompute full Y_D calibrated predictions
                Y_hat = np.bincount(bin_idx_support, weights=t0_support, minlength=8).astype(np.float64)
                Y_hat /= N_hat_support
                
                active = (yd_full > 1e-8) & (Y_hat > 1e-8)
                w_full = np.ones(8, dtype=np.float64)
                w_full[active] = yd_full[active] / Y_hat[active]
                weighted_mass_full = float(np.dot(Y_hat, w_full))
                s_full = w_full / weighted_mass_full if weighted_mass_full > 0 else np.ones(8)
                
                t_cal_full_support = t0_support * s_full[bin_idx_support]
                cal_mass_full = np.sum(t_cal_full_support)
                if cal_mass_full > 0:
                    t_cal_full_support *= (N_hat_support / cal_mass_full)
                    
                seed_predictions[s] = {
                    "t0": t0_support,
                    "N_hat": N_hat_support,
                    "Y_hat": Y_hat,
                    "t_cal_full": t_cal_full_support
                }

            # Run Replicate Sampling
            for rep_id in range(B):
                mask_seed = get_stable_mask_seed(PARTIAL_OD_BASE_SEED, fold_id, city_name, rep_id)
                rng = np.random.RandomState(mask_seed)
                
                # Single random permutation for nested masks
                perm = rng.permutation(n_pairs)
                
                for p_val in p_grid:
                    n_reveal = int(np.round(p_val * n_pairs))
                    # Nested revealed and unseen subsets
                    rev_indices = perm[:n_reveal]
                    unseen_indices = perm[n_reveal:]
                    n_unseen = len(unseen_indices)
                    
                    if n_unseen == 0:
                        # Full reveal corner case (p = 1.0)
                        continue

                    # Construct partial Y_D from revealed pairs S_p
                    if n_reveal == 0:
                        yd_partial = None
                        revealed_mass = 0.0
                        tv_partial = 0.0
                        js_partial = 0.0
                    else:
                        rev_trips = t_true_support[rev_indices]
                        rev_bins = bin_idx_support[rev_indices]
                        revealed_mass = float(np.sum(rev_trips))
                        
                        counts_k = np.bincount(rev_bins, weights=rev_trips, minlength=8).astype(np.float64)
                        if revealed_mass > 0:
                            yd_partial = counts_k / revealed_mass
                        else:
                            yd_partial = None
                            
                        if yd_partial is not None:
                            tv_partial = float(0.5 * np.sum(np.abs(yd_partial - yd_full)))
                            js_partial = float(jensenshannon(yd_partial, yd_full))
                        else:
                            tv_partial = 0.0
                            js_partial = 0.0

                    frac_pairs_rev = float(n_reveal) / float(n_pairs)
                    frac_mass_rev = float(revealed_mass) / float(total_trip_mass) if total_trip_mass > 0 else 0.0
                    
                    # Target ground truth on unseen set U_p
                    t_true_unseen = t_true_support[unseen_indices]
                    sum_true_unseen = float(np.sum(t_true_unseen))

                    # Evaluate across all 3 model seeds with identical mask
                    for s in model_seeds:
                        preds = seed_predictions[s]
                        t0_unseen = preds["t0"][unseen_indices]
                        t_full_unseen = preds["t_cal_full"][unseen_indices]
                        N_hat_total = preds["N_hat"]
                        Y_hat_total = preds["Y_hat"]
                        
                        # Compute M0 CPC on unseen set
                        denom_m0 = sum_true_unseen + float(np.sum(t0_unseen))
                        cpc_m0_unseen = (2.0 * np.sum(np.minimum(t_true_unseen, t0_unseen)) / denom_m0) if denom_m0 > 0 else 0.0
                        
                        # Compute Full Y_D CPC on unseen set
                        denom_full = sum_true_unseen + float(np.sum(t_full_unseen))
                        cpc_full_unseen = (2.0 * np.sum(np.minimum(t_true_unseen, t_full_unseen)) / denom_full) if denom_full > 0 else 0.0
                        
                        # Compute Partial Y_D Calibrated CPC on unseen set
                        if yd_partial is None:
                            cpc_part_unseen = cpc_m0_unseen
                        else:
                            active_p = (yd_partial > 1e-8) & (Y_hat_total > 1e-8)
                            w_p = np.ones(8, dtype=np.float64)
                            w_p[active_p] = yd_partial[active_p] / Y_hat_total[active_p]
                            weighted_mass_p = float(np.dot(Y_hat_total, w_p))
                            s_p = w_p / weighted_mass_p if weighted_mass_p > 0 else np.ones(8)
                            
                            # Calibrated predictions on unseen support
                            t_part_unseen = t0_unseen * s_p[bin_idx_support[unseen_indices]]
                            # Conservation scaling
                            pass # (Dead code removed)
                            
                            denom_part = sum_true_unseen + float(np.sum(t_part_unseen))
                            cpc_part_unseen = (2.0 * np.sum(np.minimum(t_true_unseen, t_part_unseen)) / denom_part) if denom_part > 0 else 0.0

                        gain_full = float(cpc_full_unseen - cpc_m0_unseen)
                        gain_part = float(cpc_part_unseen - cpc_m0_unseen)
                        diff_part_minus_yd = float(gain_part - gain_full)
                        rel_gain = float(gain_part / gain_full) if abs(gain_full) > 1e-8 else 1.0

                        raw_records.append({
                            "fold": fold_id,
                            "target_city": city_name,
                            "model_seed": s,
                            "replicate_id": rep_id,
                            "reveal_fraction_requested": p_val,
                            "n_pairs_total": n_pairs,
                            "n_pairs_revealed": n_reveal,
                            "fraction_pairs_revealed": frac_pairs_rev,
                            "total_trip_mass": total_trip_mass,
                            "revealed_trip_mass": revealed_mass,
                            "fraction_trip_mass_revealed": frac_mass_rev,
                            "n_unseen_pairs": n_unseen,
                            "empirical_tv_partial_vs_full": tv_partial,
                            "js_partial_vs_full": js_partial,
                            "cpc_m0_unseen": cpc_m0_unseen,
                            "cpc_full_yd_unseen": cpc_full_unseen,
                            "cpc_partial_od_unseen": cpc_part_unseen,
                            "gain_full_yd": gain_full,
                            "gain_partial_od": gain_part,
                            "difference_partial_minus_yd": diff_part_minus_yd,
                            "relative_gain_vs_yd": rel_gain,
                            "K": 8,
                            "q": 1.0,
                            "mask_seed": mask_seed
                        })

            city_elapsed = time.perf_counter() - city_start
            print(f"  [Fold {fold_id} ({city_idx+1}/10)] {city_name:<16} | Pairs: {n_pairs:>5} | B={B} reps done in {city_elapsed:.2f}s")

    raw_df = pd.DataFrame(raw_records)
    raw_csv_path = out_path / "partial_od_raw.csv"
    raw_df.to_csv(raw_csv_path, index=False)
    print(f"\nSaved {len(raw_df)} raw records to {raw_csv_path}")

    # Generate Statistical Summary and Figures
    generate_partial_od_summary(raw_df, out_path, p_grid, smoke=smoke)


def generate_partial_od_summary(
    raw_df: pd.DataFrame, 
    out_path: Path, 
    p_grid: List[float],
    smoke: bool = False
) -> None:
    print("\n" + "=" * 85)
    print("GENERATING PARTIAL-OD STATISTICAL SUMMARY (N=50 CITIES)")
    print("=" * 85)

    # Hierarchy: Replicates -> Model Seeds -> City Level (N=50)
    city_df = raw_df.groupby(["fold", "target_city", "reveal_fraction_requested"]).agg({
        "fraction_pairs_revealed": "mean",
        "fraction_trip_mass_revealed": "mean",
        "empirical_tv_partial_vs_full": "mean",
        "js_partial_vs_full": "mean",
        "cpc_m0_unseen": "mean",
        "cpc_full_yd_unseen": "mean",
        "cpc_partial_od_unseen": "mean",
        "gain_full_yd": "mean",
        "gain_partial_od": "mean",
        "difference_partial_minus_yd": "mean",
        "relative_gain_vs_yd": "mean"
    }).reset_index().rename(columns={"reveal_fraction_requested": "p"})

    summary_rows = []
    raw_p_values = []
    p_vals_tested = [p for p in p_grid if p > 0]

    # Precalculate raw Wilcoxon p-values for partial OD benefit vs M0
    for p_val in p_vals_tested:
        sub = city_df[city_df.p == p_val]
        gains = sub["gain_partial_od"].values
        _, p_w = stats.wilcoxon(gains, alternative="greater")
        raw_p_values.append(p_w)

    holm_p_vals = holm_correction(raw_p_values)
    holm_dict = {p: h_p for p, h_p in zip(p_vals_tested, holm_p_vals)}

    for p_val in p_grid:
        sub = city_df[city_df.p == p_val]
        n_cities = len(sub)
        
        mean_mass = float(sub["fraction_trip_mass_revealed"].mean())
        mean_tv = float(sub["empirical_tv_partial_vs_full"].mean())
        mean_m0 = float(sub["cpc_m0_unseen"].mean())
        mean_gain_full = float(sub["gain_full_yd"].mean())
        mean_gain_part = float(sub["gain_partial_od"].mean())
        mean_diff = float(sub["difference_partial_minus_yd"].mean())
        
        pos_cities = int((sub["gain_partial_od"] > 0).sum())
        match_yd_cities = int((sub["difference_partial_minus_yd"] >= 0).sum())
        
        ci_diff_l, ci_diff_h = fold_stratified_bootstrap(city_df, "difference_partial_minus_yd", p_val)
        ci_part_l, ci_part_h = fold_stratified_bootstrap(city_df, "gain_partial_od", p_val)
        ci_full_l, ci_full_h = fold_stratified_bootstrap(city_df, "gain_full_yd", p_val)
        
        h_pval = holm_dict.get(p_val, 1.0) if p_val > 0 else 1.0

        summary_rows.append({
            "p": p_val,
            "n_cities": n_cities,
            "mean_revealed_mass": mean_mass,
            "mean_tv": mean_tv,
            "mean_m0_cpc": mean_m0,
            "mean_gain_full_yd": mean_gain_full,
            "ci_95_gain_full": [ci_full_l, ci_full_h],
            "mean_gain_partial_od": mean_gain_part,
            "ci_95_gain_partial": [ci_part_l, ci_part_h],
            "mean_diff_vs_yd": mean_diff,
            "ci_95_diff": [ci_diff_l, ci_diff_h],
            "pos_cities_vs_m0": pos_cities,
            "match_yd_cities": match_yd_cities,
            "holm_pval_benefit": h_pval
        })

    summary_df = pd.DataFrame(summary_rows)

    # Calculate Equivalence Crossing
    p_eq_grid = None
    p_eq_interp = None

    for r in summary_rows:
        if r["mean_diff_vs_yd"] >= 0 and p_eq_grid is None:
            p_eq_grid = r["p"]

    # Linear interpolation for zero crossing of mean_diff_vs_yd
    for i in range(len(summary_rows) - 1):
        r1, r2 = summary_rows[i], summary_rows[i+1]
        d1, d2 = r1["mean_diff_vs_yd"], r2["mean_diff_vs_yd"]
        if d1 <= 0 and d2 >= 0 and (d2 - d1) > 0:
            p_eq_interp = r1["p"] + (-d1 / (d2 - d1)) * (r2["p"] - r1["p"])
            break

    # Save summary JSON
    summary_json_path = out_path / "partial_od_summary.json"
    summary_dict = {
        "experiment": "partial_od_information_equivalence",
        "n_evaluation_cities": len(city_df["target_city"].unique()),
        "p_eq_grid": p_eq_grid,
        "p_eq_interp": p_eq_interp,
        "results_by_p": summary_rows
    }
    with open(summary_json_path, "w") as f:
        json.dump(summary_dict, f, indent=2)

    # Save Manifest
    manifest_path = out_path / "partial_od_manifest.json"
    manifest_dict = {
        "experiment": "partial_od_information_equivalence",
        "protocol_version": "v1",
        "evaluation_cities": len(city_df["target_city"].unique()),
        "folds": [1, 2, 3, 4, 5] if not smoke else [1],
        "model_seeds": [1, 10, 100] if not smoke else [1, 10],
        "K": 8,
        "q": 1.0,
        "sampling": "uniform_OD_pair_without_replacement",
        "nested_masks": True,
        "evaluation_support": "unseen_positive_interzonal_pairs_only",
        "replicates": len(raw_df["replicate_id"].unique()),
        "mask_base_seed": PARTIAL_OD_BASE_SEED,
        "bootstrap_seed": 42
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest_dict, f, indent=2)

    # Save Markdown Table
    summary_md_path = out_path / "partial_od_summary.md"
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("# Table: Partial-OD Information Equivalence Summary\n\n")
        f.write("> **Evaluation Scope**: Assesses the operational value of information in target-city distance distribution $Y_D$ relative to observing $p\\%$ of positive interzonal OD pairs ($K=8, q=1.0$, seeds $s \\in \\{1, 10, 100\\}$) evaluated strictly on unseen pairs ($N=50$ held-out test cities).\n\n")
        
        if p_eq_interp is not None:
            pct_interp = p_eq_interp * 100.0
            f.write(f"**Estimated OD-Pair Equivalence Crossing ($p_\\text{{eq,interp}}$):** `{pct_interp:.2f}%` of positive interzonal OD pairs  \n")
        if p_eq_grid is not None:
            pct_grid = p_eq_grid * 100.0
            f.write(f"**Smallest Tested Grid Matching Fraction ($p_\\text{{eq,grid}}$):** `{pct_grid:.2f}%` of positive interzonal OD pairs  \n\n")
            
        f.write("| Revealed OD Pairs ($p$) | Mean Revealed Trip Mass | Mean TV to Full $Y_D$ | $M_0$ CPC (Unseen) | Full-$Y_D$ Gain | Partial-OD Gain | Difference vs Full $Y_D$ ($D(p)$) | 95% CI Difference | Partial Benefit Holm $p$ | Cities Partial $> M_0$ |\n")
        f.write("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        
        for r in summary_rows:
            p_pct = f"{r['p']*100:.2f}%"
            mass_pct = f"{r['mean_revealed_mass']*100:.2f}%"
            tv_pct = f"{r['mean_tv']*100:.2f}%"
            m0_str = f"{r['mean_m0_cpc']:.4f}"
            full_str = f"+{r['mean_gain_full_yd']:.5f}"
            part_str = f"{r['mean_gain_partial_od']:+.5f}"
            diff_str = f"{r['mean_diff_vs_yd']:+.5f}"
            ci_str = f"[{r['ci_95_diff'][0]:+.5f}, {r['ci_95_diff'][1]:+.5f}]"
            h_str = f"{r['holm_pval_benefit']:.4e}" if r['p'] > 0 else "—"
            pos_str = f"{r['pos_cities_vs_m0']}/{r['n_cities']}"
            
            f.write(f"| **{p_pct}** | {mass_pct} | {tv_pct} | {m0_str} | {full_str} | **{part_str}** | **{diff_str}** | {ci_str} | {h_str} | {pos_str} |\n")
            
        f.write("\n---\n\n### Empirical Interpretation\n")
        f.write("Under the frozen support-conditioned model and the same distance-bin calibration operator, the mean reconstruction benefit of the full target-city $Y_D$ was comparable to that obtained from directly observing approximately ")
        if p_eq_interp is not None:
            f.write(f"**{p_eq_interp*100:.2f}%** of the positive interzonal OD support.\n")
        else:
            f.write("the upper range of the tested partial-OD support.\n")

    print(f"Summary exported to {summary_md_path}")
    print(f"JSON exported to {summary_json_path}")

    # Generate Figures
    generate_figures(summary_df, out_path, p_eq_interp)


def generate_figures(summary_df: pd.DataFrame, out_path: Path, p_eq_interp: float) -> None:
    plt.rcParams.update({'font.sans-serif': 'Helvetica', 'axes.edgecolor': '#333333', 'axes.linewidth': 0.8})
    p_vals = summary_df["p"].values * 100.0 # to percentage
    
    # Figure 1: Gain vs Reveal Fraction
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    ax.axhline(0, color="#888888", linestyle="--", alpha=0.6)
    
    full_gain = summary_df["mean_gain_full_yd"].values
    part_gain = summary_df["mean_gain_partial_od"].values
    part_ci_l = np.array([ci[0] for ci in summary_df["ci_95_gain_partial"]])
    part_ci_h = np.array([ci[1] for ci in summary_df["ci_95_gain_partial"]])
    
    ax.plot(p_vals, full_gain, label="Full $Y_D$ Reference Gain", color="#1f77b4", linestyle="--", linewidth=2.0)
    ax.plot(p_vals, part_gain, label="Partial-OD Calibration Gain", color="#d62728", marker="o", linewidth=2.0)
    ax.fill_between(p_vals, part_ci_l, part_ci_h, color="#d62728", alpha=0.15, label="95% Fold Bootstrap CI")
    
    if p_eq_interp is not None:
        ax.axvline(p_eq_interp * 100.0, color="#2ca02c", linestyle=":", label=f"Equivalence $p_{{eq}} = {p_eq_interp*100:.2f}\\%$")
        
    ax.set_xlabel("Revealed Positive Interzonal OD Pairs (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean Marginal Gain $\\Delta\\mathrm{CPC}_U$ on Unseen OD", fontsize=11, fontweight="bold")
    ax.set_title("Marginal Reconstruction Value: Partial OD vs Full $Y_D$", fontsize=12, fontweight="bold", pad=12)
    ax.legend(frameon=True, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    fig.savefig(out_path / "figures" / "fig_partial_od_gain_vs_reveal_fraction.png")
    plt.close(fig)

    # Figure 2: Difference D(p) vs Fraction
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    ax.axhline(0, color="#333333", linestyle="-", linewidth=1.0)
    
    diff_vals = summary_df["mean_diff_vs_yd"].values
    diff_ci_l = np.array([ci[0] for ci in summary_df["ci_95_diff"]])
    diff_ci_h = np.array([ci[1] for ci in summary_df["ci_95_diff"]])
    
    ax.plot(p_vals, diff_vals, color="#9467bd", marker="s", linewidth=2.0, label="$\\bar{D}(p) = \\mathrm{Gain}_{\\mathrm{partial}} - \\mathrm{Gain}_{Y_D}$")
    ax.fill_between(p_vals, diff_ci_l, diff_ci_h, color="#9467bd", alpha=0.15, label="95% Fold Bootstrap CI")
    
    if p_eq_interp is not None:
        ax.scatter([p_eq_interp * 100.0], [0.0], color="#d62728", s=80, zorder=5, label=f"Crossing $p_{{eq}} = {p_eq_interp*100:.2f}\\%$")
        
    ax.set_xlabel("Revealed Positive Interzonal OD Pairs (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Gain Difference $D(p)$", fontsize=11, fontweight="bold")
    ax.set_title("Information Equivalence Zero-Crossing", fontsize=12, fontweight="bold", pad=12)
    ax.legend(frameon=True, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    fig.savefig(out_path / "figures" / "fig_partial_od_equivalence_crossing.png")
    plt.close(fig)

    # Figure 3: Fraction vs TV
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    tvs = summary_df["mean_tv"].values * 100.0
    ax.plot(p_vals, tvs, color="#ff7f0e", marker="^", linewidth=2.0)
    ax.set_xlabel("Revealed Positive Interzonal OD Pairs (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean Total Variation Error $\\mathrm{TV}(\\tilde{Y}_D, Y_D^{\\mathrm{full}})$ (%)", fontsize=11, fontweight="bold")
    ax.set_title("Distributional Convergence with Partial OD Observation", fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    fig.savefig(out_path / "figures" / "fig_partial_od_fraction_vs_tv.png")
    plt.close(fig)

    # Figure 4: Mass vs Gain
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    masses = summary_df["mean_revealed_mass"].values * 100.0
    ax.plot(masses, part_gain, color="#2ca02c", marker="d", linewidth=2.0, label="Partial-OD Gain")
    ax.axhline(full_gain[0], color="#1f77b4", linestyle="--", label="Full $Y_D$ Reference")
    ax.set_xlabel("Revealed Interzonal Trip Mass (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean Marginal Gain $\\Delta\\mathrm{CPC}_U$", fontsize=11, fontweight="bold")
    ax.set_title("Reconstruction Gain vs Revealed Trip Mass", fontsize=12, fontweight="bold", pad=12)
    ax.legend(frameon=True, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    fig.savefig(out_path / "figures" / "fig_partial_od_mass_vs_gain.png")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Partial-OD Information Equivalence Experiment")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="results/partial_od_equivalence_v1")
    parser.add_argument("--b", type=int, default=500, help="Number of Monte Carlo replicates per city")
    parser.add_argument("--smoke", action="store_true", help="Run fast smoke test")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    run_partial_od_experiment(
        data_root=args.data_root,
        output_dir=args.output_dir,
        replicates=args.b,
        smoke=args.smoke,
        device=args.device
    )
