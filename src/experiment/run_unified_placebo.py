"""
Unified Placebo Experiment across 50 Cities and 3 Seeds.

Evaluates 6 unified conditions under strictly identical calibration protocols:
  1. Target Y_D (Oracle: true city-specific distribution; an oracle reference, NOT a CPC upper bound)
  2. Raw Training Donors (B=1000 draws from 35 training cities in same fold)
  3. Raw Test Donors (9 other held-out test cities in same fold, both exact and B=1000 draws)
  4. Dose-Matched Training Donors (B=1000 draws from 35 training cities, matched dose D_T)
  5. Permuted Target Y_D (B=1000 permutations of active target bins)
  6. Train-Mean Global Y_D (Global average of 35 training cities in same fold)
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data.city_splits import load_splits_manifest_v2
from src.data.dataset import load_city
from src.data.urban_graph import build_radius_graph
from src.data.yd_extractor import compute_kbin_edges, extract_yd_kbins
from src.experiment.e1_core import active_bins_from_pairs, manifest_sha256, verify_checkpoint_provenance
from src.training.train import load_checkpoint, infer_zero_shot
from src.training.evaluate import compute_cpc_pair
from src.calibration.bin_calibration import calibrate_kbins


def safe_log_ratio(p: np.ndarray, y_hat: np.ndarray, active_mask: np.ndarray, delta: float = 1e-12) -> np.ndarray:
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


def fast_eval_cpc(
    yd_cand: np.ndarray,
    active_mask: np.ndarray,
    Y_hat: np.ndarray,
    t0_inter: np.ndarray,
    t_true_inter: np.ndarray,
    bin_masks: list[np.ndarray],
    denom: float,
    K: int = 8,
    epsilon: float = 1e-12,
) -> float:
    p_active = yd_cand[active_mask]
    p_sum = p_active.sum()
    if p_sum <= 0:
        p_cond = Y_hat[active_mask] / max(Y_hat[active_mask].sum(), 1e-12)
    else:
        p_cond = p_active / p_sum

    y_hat_safe = np.maximum(Y_hat[active_mask], epsilon)
    w_active = p_cond / y_hat_safe

    weighted_mass = float((Y_hat[active_mask] * w_active).sum())
    s_active = w_active / weighted_mass if weighted_mass > 0 else np.ones_like(w_active)

    s = np.ones(K, dtype=np.float64)
    s[active_mask] = s_active

    min_sum = 0.0
    for k in range(K):
        mask = bin_masks[k]
        if mask.any():
            t_scaled = t0_inter[mask] * s[k]
            min_sum += np.minimum(t_true_inter[mask], t_scaled).sum()

    return float(2.0 * min_sum / denom)


def assert_fast_eval_matches_operator(
    label: str,
    yd_cand: np.ndarray,
    active_mask: np.ndarray,
    Y_hat: np.ndarray,
    t0: np.ndarray,
    t0_inter: np.ndarray,
    t_true_inter: np.ndarray,
    dist_km: np.ndarray,
    inter_mask: np.ndarray,
    bin_edges: np.ndarray,
    bin_masks: list[np.ndarray],
    denom: float,
    K: int,
    tol: float = 1e-8,
) -> float:
    """Cross-check the fast CPC path against the canonical calibration operator."""
    t_cal_ref = calibrate_kbins(t0, dist_km, inter_mask, yd_cand, bin_edges, q=1.0)
    cpc_ref = float(compute_cpc_pair(t_true_inter, t_cal_ref[inter_mask]))
    cpc_fast = fast_eval_cpc(yd_cand, active_mask, Y_hat, t0_inter, t_true_inter, bin_masks, denom, K)
    if abs(cpc_fast - cpc_ref) >= tol:
        raise AssertionError(f"Equivalence check failed [{label}]: {cpc_fast} vs {cpc_ref}")
    return cpc_fast


def stratified_indices(fold_ids: np.ndarray, n_boot: int = 10000, seed: int = 42) -> np.ndarray:
    """Resample cities within each fold, preserving its original sample size."""
    folds = np.asarray(fold_ids)
    if folds.ndim != 1 or not len(folds) or pd.isna(folds).any() or n_boot < 1:
        raise ValueError("Nonempty fold IDs and a positive bootstrap count are required")
    rng = np.random.default_rng(seed)
    return np.concatenate([
        rng.choice(np.flatnonzero(folds == fold), size=(n_boot, int((folds == fold).sum())), replace=True)
        for fold in sorted(np.unique(folds))
    ], axis=1)


def bootstrap_ci(vals: np.ndarray, indices: np.ndarray) -> tuple[float, float]:
    """Apply shared city resamples to a condition or paired difference."""
    values = np.asarray(vals, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or indices.shape[1] != len(values):
        raise ValueError("Finite city-level values must match bootstrap indices")
    means = values[indices].mean(axis=1)
    return tuple(float(x) for x in np.percentile(means, [2.5, 97.5]))


def run_unified_placebo(
    b_draws: int = 1000,
    data_root: str = "data",
    output_dir: Path = Path("results/unified_placebo_v1"),
    checkpoint_dir: Path = Path("results/checkpoints"),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path("results/e1/splits_manifest_v2.json")
    splits = load_splits_manifest_v2(str(manifest_path), data_root=data_root)

    seeds = [1, 10, 100]
    K = 8
    placebo_seed = 20260823
    rng = np.random.RandomState(placebo_seed)
    epsilon = 1e-12

    city_results = []

    print(f"Starting Unified Placebo Experiment across 50 cities (B={b_draws})...")

    for fold_id in range(1, 6):
        split = splits[fold_id]
        train_cities = split["train"]
        test_cities = split["test"]

        bin_edges, _ = compute_kbin_edges(train_cities, K=K, data_root=data_root)

        # 1. Preload train Y_D
        train_yd_dict = {}
        for tc in train_cities:
            raw_c = load_city(tc, data_root=data_root, fit_scaler=False)
            dist_c = np.asarray(raw_c.dist_km, dtype=np.float64)
            inter_c = (raw_c.pair_o_idx.numpy() != raw_c.pair_d_idx.numpy()) & (dist_c > 0.0)
            t_gt_c = raw_c.pair_trips.numpy().astype(np.float64)
            train_yd_dict[tc] = extract_yd_kbins(dist_c, t_gt_c, bin_edges, inter_c)

        train_mean_yd = np.mean(list(train_yd_dict.values()), axis=0)

        # 2. Preload test Y_D
        test_yd_dict = {}
        test_data_dict = {}
        for tc in test_cities:
            raw_c = load_city(tc, data_root=data_root, fit_scaler=False)
            dist_c = np.asarray(raw_c.dist_km, dtype=np.float64)
            inter_c = (raw_c.pair_o_idx.numpy() != raw_c.pair_d_idx.numpy()) & (dist_c > 0.0)
            t_gt_c = raw_c.pair_trips.numpy().astype(np.float64)
            test_yd_dict[tc] = extract_yd_kbins(dist_c, t_gt_c, bin_edges, inter_c)
            test_data_dict[tc] = (raw_c, dist_c, inter_c, t_gt_c)

        for tc in test_cities:
            raw_c, dist_km, inter_mask, t_gt = test_data_dict[tc]
            ei, ed = build_radius_graph(raw_c.lon_lat, radius_km=5.0, include_self_loop=True, cache_key=f"{tc}_tracts")

            yd_target = test_yd_dict[tc]

            t_true_inter = t_gt[inter_mask]
            dist_inter = dist_km[inter_mask]
            bin_masks = [((dist_inter > float(bin_edges[k])) & (dist_inter <= float(bin_edges[k + 1]))) for k in range(K)]

            active_mask = active_bins_from_pairs(dist_inter, bin_edges)
            target_act_count = int(active_mask.sum())

            # A single active bin admits no non-identity permutation.
            if target_act_count < 2:
                index_perms = []
            elif math.factorial(target_act_count) <= 40320:
                all_p = list(itertools.permutations(np.arange(target_act_count)))
                valid_p = [p for p in all_p if not np.array_equal(p, np.arange(target_act_count))]
                if len(valid_p) > b_draws:
                    chosen_idx = rng.choice(len(valid_p), size=b_draws, replace=False)
                    index_perms = [valid_p[i] for i in chosen_idx]
                else:
                    index_perms = valid_p
            else:
                perms_set = set()
                while len(perms_set) < b_draws:
                    p = tuple(rng.permutation(np.arange(target_act_count)))
                    if not np.array_equal(p, np.arange(target_act_count)):
                        perms_set.add(p)
                index_perms = list(perms_set)

            # Sample B donors from train
            train_donor_sample = rng.choice(train_cities, size=b_draws, replace=True)
            # Other 9 test cities
            other_test_cities = [c for c in test_cities if c != tc]
            test_donor_sample = rng.choice(other_test_cities, size=b_draws, replace=True)

            seed_runs = []
            for seed in seeds:
                ckpt_path = checkpoint_dir / f"5fold_fold{fold_id}_seed{seed}.pt"
                model, scaler, metadata = load_checkpoint(ckpt_path, device_str="cpu")
                verify_checkpoint_provenance(
                    metadata,
                    ckpt_path,
                    expected_seed=seed,
                    expected_fold=fold_id,
                    expected_manifest_sha256=manifest_sha256(manifest_path),
                )
                model.eval()

                city_data = load_city(tc, data_root=data_root, feature_scaler=scaler, fit_scaler=False)
                with torch.no_grad():
                    t0_tensor = infer_zero_shot(model, city_data, ei, ed, device="cpu")
                t0 = t0_tensor.numpy().astype(np.float64)
                t0_inter = t0[inter_mask]

                denom = float(t_true_inter.sum() + t0_inter.sum())
                cpc0 = float(2.0 * np.minimum(t_true_inter, t0_inter).sum() / denom)

                N_hat = t0_inter.sum()
                Y_hat = np.zeros(K, dtype=np.float64)
                for k in range(K):
                    if N_hat > 0:
                        Y_hat[k] = t0_inter[bin_masks[k]].sum() / N_hat

                # Verification check: fast_eval vs calibrate_kbins
                check_args = (active_mask, Y_hat, t0, t0_inter, t_true_inter,
                              dist_km, inter_mask, bin_edges, bin_masks, denom, K)
                cpc_fast = assert_fast_eval_matches_operator("target", yd_target, *check_args)
                assert_fast_eval_matches_operator("donor_train", train_yd_dict[train_cities[0]], *check_args)
                assert_fast_eval_matches_operator("donor_test", test_yd_dict[other_test_cities[0]], *check_args)
                assert_fast_eval_matches_operator("train_mean", train_mean_yd, *check_args)

                # Low-mass probe: one active bin carries nearly all mass, the rest are near-zero.
                yd_low_mass = np.full(K, 1e-10)
                yd_low_mass[np.flatnonzero(active_mask)[0]] = 1.0
                yd_low_mass = yd_low_mass / yd_low_mass.sum()
                assert_fast_eval_matches_operator("low_mass", yd_low_mass, *check_args)

                # 1. Target Condition
                cpc_target = cpc_fast
                d_cpc_target = cpc_target - cpc0

                # 2. Raw Training Donor Condition (B=1000)
                d_cpc_raw_train_list = []
                for d_city in train_donor_sample:
                    d_yd = train_yd_dict[d_city]
                    cpc_wr = fast_eval_cpc(d_yd, active_mask, Y_hat, t0_inter, t_true_inter, bin_masks, denom, K)
                    d_cpc_raw_train_list.append(cpc_wr - cpc0)
                d_cpc_raw_train = float(np.mean(d_cpc_raw_train_list))

                # 3. Raw Test Donor Condition (both exact 9-donor average and B=1000 draws)
                d_cpc_raw_test_exact_list = []
                for d_city in other_test_cities:
                    d_yd = test_yd_dict[d_city]
                    cpc_wr = fast_eval_cpc(d_yd, active_mask, Y_hat, t0_inter, t_true_inter, bin_masks, denom, K)
                    d_cpc_raw_test_exact_list.append(cpc_wr - cpc0)
                d_cpc_raw_test_exact = float(np.mean(d_cpc_raw_test_exact_list))

                d_cpc_raw_test_b_list = []
                for d_city in test_donor_sample:
                    d_yd = test_yd_dict[d_city]
                    cpc_wr = fast_eval_cpc(d_yd, active_mask, Y_hat, t0_inter, t_true_inter, bin_masks, denom, K)
                    d_cpc_raw_test_b_list.append(cpc_wr - cpc0)
                d_cpc_raw_test_b = float(np.mean(d_cpc_raw_test_b_list))

                # 4. Dose-Matched Training Donor Condition (B=1000)
                r_T = safe_log_ratio(yd_target, Y_hat, active_mask, delta=epsilon)
                r_tilde_T = np.zeros_like(r_T)
                r_tilde_T[active_mask] = r_T[active_mask] - np.mean(r_T[active_mask])
                D_T = float(np.sqrt(np.mean(r_tilde_T[active_mask]**2)))

                d_cpc_matched_list = []
                for d_city in train_donor_sample:
                    d_yd = train_yd_dict[d_city]
                    r_D = safe_log_ratio(d_yd, Y_hat, active_mask, delta=epsilon)
                    r_tilde_D = np.zeros_like(r_D)
                    r_tilde_D[active_mask] = r_D[active_mask] - np.mean(r_D[active_mask])
                    D_D = float(np.sqrt(np.mean(r_tilde_D[active_mask]**2)))
                    if D_D < 1e-12:
                        d_cpc_matched_list.append(d_cpc_target)
                        continue

                    r_tilde_D_star = np.zeros_like(r_tilde_D)
                    r_tilde_D_star[active_mask] = r_tilde_D[active_mask] * (D_T / D_D)

                    p_D_star = np.zeros_like(Y_hat)
                    p_D_star[active_mask] = np.maximum(Y_hat[active_mask], epsilon) * np.exp(r_tilde_D_star[active_mask])
                    p_D_star[active_mask] /= p_D_star[active_mask].sum()

                    cpc_matched = fast_eval_cpc(p_D_star, active_mask, Y_hat, t0_inter, t_true_inter, bin_masks, denom, K)
                    d_cpc_matched_list.append(cpc_matched - cpc0)
                d_cpc_matched = float(np.mean(d_cpc_matched_list))

                # 5. Permuted Target Y_D Condition (B=1000)
                d_cpc_perm_list = []
                for p_idx, p_indices in enumerate(index_perms):
                    r_tilde_P = np.zeros_like(r_tilde_T)
                    r_tilde_P[active_mask] = r_tilde_T[active_mask][list(p_indices)]
                    p_P = np.zeros_like(Y_hat)
                    p_P[active_mask] = np.maximum(Y_hat[active_mask], epsilon) * np.exp(r_tilde_P[active_mask])
                    p_P[active_mask] /= p_P[active_mask].sum()

                    if p_idx == 0:
                        assert_fast_eval_matches_operator("permuted", p_P, *check_args)
                    cpc_perm = fast_eval_cpc(p_P, active_mask, Y_hat, t0_inter, t_true_inter, bin_masks, denom, K)
                    d_cpc_perm_list.append(cpc_perm - cpc0)
                # With a single active bin every permutation is the identity, so the
                # permuted condition degenerates to the target condition.
                d_cpc_perm = float(np.mean(d_cpc_perm_list)) if d_cpc_perm_list else d_cpc_target

                # 6. Global Train-Mean Condition (Raw)
                cpc_mean = fast_eval_cpc(train_mean_yd, active_mask, Y_hat, t0_inter, t_true_inter, bin_masks, denom, K)
                d_cpc_train_mean = float(cpc_mean - cpc0)

                # 7. Global Train-Mean Condition (Dose-Matched)
                r_M = safe_log_ratio(train_mean_yd, Y_hat, active_mask, delta=epsilon)
                r_tilde_M = np.zeros_like(r_M)
                r_tilde_M[active_mask] = r_M[active_mask] - np.mean(r_M[active_mask])
                D_M = float(np.sqrt(np.mean(r_tilde_M[active_mask]**2)))
                if D_M >= 1e-12:
                    r_tilde_M_star = np.zeros_like(r_tilde_M)
                    r_tilde_M_star[active_mask] = r_tilde_M[active_mask] * (D_T / D_M)
                    p_M_star = np.zeros_like(Y_hat)
                    p_M_star[active_mask] = np.maximum(Y_hat[active_mask], epsilon) * np.exp(r_tilde_M_star[active_mask])
                    p_M_star[active_mask] /= p_M_star[active_mask].sum()
                    cpc_matched_tm = fast_eval_cpc(p_M_star, active_mask, Y_hat, t0_inter, t_true_inter, bin_masks, denom, K)
                    d_cpc_matched_train_mean = float(cpc_matched_tm - cpc0)
                else:
                    d_cpc_matched_train_mean = d_cpc_target

                seed_runs.append({
                    "cpc0": cpc0,
                    "d_cpc_target": d_cpc_target,
                    "d_cpc_raw_train": d_cpc_raw_train,
                    "d_cpc_raw_test_exact": d_cpc_raw_test_exact,
                    "d_cpc_raw_test_b": d_cpc_raw_test_b,
                    "d_cpc_matched": d_cpc_matched,
                    "d_cpc_perm": d_cpc_perm,
                    "d_cpc_train_mean": d_cpc_train_mean,
                    "d_cpc_matched_train_mean": d_cpc_matched_train_mean,
                })

            # Average across 3 model seeds for city
            city_results.append({
                "fold": fold_id,
                "city": tc,
                "n_active_bins": target_act_count,
                "n_permutations": len(index_perms),
                "cpc0": float(np.mean([r["cpc0"] for r in seed_runs])),
                "d_cpc_target": float(np.mean([r["d_cpc_target"] for r in seed_runs])),
                "d_cpc_raw_train": float(np.mean([r["d_cpc_raw_train"] for r in seed_runs])),
                "d_cpc_raw_test_exact": float(np.mean([r["d_cpc_raw_test_exact"] for r in seed_runs])),
                "d_cpc_raw_test_b": float(np.mean([r["d_cpc_raw_test_b"] for r in seed_runs])),
                "d_cpc_matched": float(np.mean([r["d_cpc_matched"] for r in seed_runs])),
                "d_cpc_perm": float(np.mean([r["d_cpc_perm"] for r in seed_runs])),
                "d_cpc_train_mean": float(np.mean([r["d_cpc_train_mean"] for r in seed_runs])),
                "d_cpc_matched_train_mean": float(np.mean([r["d_cpc_matched_train_mean"] for r in seed_runs])),
            })

    df_city = pd.DataFrame(city_results)
    df_city.to_csv(output_dir / "unified_placebo_per_city.csv", index=False)

    summarize_placebo(df_city, output_dir)


def summarize_placebo(df_city: pd.DataFrame, output_dir: Path) -> dict:
    """Recompute summaries without inference or placebo resampling."""
    if len(df_city) != 50 or df_city["city"].nunique() != 50:
        raise ValueError("Expected exactly 50 distinct city-level records")
    expected = load_splits_manifest_v2(str(Path(__file__).resolve().parents[2] / "results/e1/splits_manifest_v2.json"),
                                     data_root=str(Path(__file__).resolve().parents[2] / "data"))
    expected_pairs = {(city, fold) for fold, split in expected.items() for city in split["test"]}
    if set(zip(df_city["city"], df_city["fold"])) != expected_pairs:
        raise ValueError("City/fold assignments do not match the locked split manifest")
    df_city = df_city.sort_values(["fold", "city"]).reset_index(drop=True)
    indices = stratified_indices(df_city["fold"].to_numpy())
    output_dir.mkdir(parents=True, exist_ok=True)
    # Compute Summary Statistics
    summary = {}
    cond_keys = [
        ("target", "d_cpc_target", "Target Y_D (Oracle)"),
        ("raw_test_exact", "d_cpc_raw_test_exact", "Raw Test Donors (E1-v2 exact 9 donors)"),
        ("raw_test_b", "d_cpc_raw_test_b", "Raw Test Donors (B=1000 draws)"),
        ("raw_train_b", "d_cpc_raw_train", "Raw Training Donors (B=1000 draws)"),
        ("matched_train_b", "d_cpc_matched", "Dose-Matched Training Donors (B=1000 draws)"),
        ("train_mean", "d_cpc_train_mean", "Raw Fold Train-Mean Y_D"),
        ("matched_train_mean", "d_cpc_matched_train_mean", "Dose-Matched Fold Train-Mean Y_D"),
        ("permuted_b", "d_cpc_perm", "Permuted Target Y_D (B=1000 draws)"),
    ]

    target_vals = df_city["d_cpc_target"].values

    for key, col, label in cond_keys:
        vals = df_city[col].values
        mean_v = float(np.mean(vals))
        median_v = float(np.median(vals))
        ci_low, ci_high = bootstrap_ci(vals, indices)

        if key == "target":
            # Pre/post calibration effect: two-sided.
            spec_gain_mean = 0.0
            spec_gain_median = 0.0
            spec_ci = [0.0, 0.0]
            win_rate = int((vals > 0).sum())
            p_effect = float(wilcoxon(vals, alternative="two-sided").pvalue)
            p_superiority = None
        else:
            # Target-versus-control superiority: one-sided.
            diffs = target_vals - vals
            spec_gain_mean = float(np.mean(diffs))
            spec_gain_median = float(np.median(diffs))
            spec_ci = list(bootstrap_ci(diffs, indices))
            win_rate = int((diffs > 0).sum())
            p_effect = float(wilcoxon(vals, alternative="two-sided").pvalue)
            p_superiority = float(wilcoxon(diffs, alternative="greater").pvalue)

        summary[key] = {
            "label": label,
            "mean_delta_cpc": mean_v,
            "median_delta_cpc": median_v,
            "ci_95": [ci_low, ci_high],
            "specificity_gain_mean": spec_gain_mean,
            "specificity_gain_median": spec_gain_median,
            "specificity_ci_95": spec_ci,
            "win_rate": f"{win_rate}/50",
            "p_effect_two_sided": p_effect,
            "p_superiority_one_sided": p_superiority,
        }

    with open(output_dir / "unified_placebo_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Markdown Summary Table
    def _fmt_p(p: float | None) -> str:
        if p is None:
            return "—"
        return f"{p:.2e}" if p < 0.001 else f"{p:.4f}"

    md = "# Unified Placebo Experiment Report (K=8, 50 Cities x 3 Seeds)\n\n"
    md += "### Reconciled Head-to-Head Placebo Comparison Table\n\n"
    md += ("| Condition | Mean $\\Delta$CPC | Median $\\Delta$CPC | Specificity Gain ($Target - Placebo$) "
           "| 95% Bootstrap CI | Win Rate | Two-sided $p$ (effect) | One-sided $p$ (target > placebo) |\n")
    md += "|---|---|---|---|---|---|---|---|\n"

    for key, col, label in cond_keys:
        s = summary[key]
        ci_str = f"[{s['ci_95'][0]:+.5f}, {s['ci_95'][1]:+.5f}]"
        spec_str = f"{s['specificity_gain_mean']:+.6f}" if key != "target" else "—"
        spec_ci_str = f"[{s['specificity_ci_95'][0]:+.5f}, {s['specificity_ci_95'][1]:+.5f}]" if key != "target" else ci_str
        md += (f"| **{s['label']}** | `{s['mean_delta_cpc']:+.6f}` | `{s['median_delta_cpc']:+.6f}` | "
               f"**`{spec_str}`** | `{spec_ci_str}` | **{s['win_rate']}** | "
               f"`{_fmt_p(s['p_effect_two_sided'])}` | `{_fmt_p(s['p_superiority_one_sided'])}` |\n")

    target = summary["target"]
    placebo_keys = [k for k, _, _ in cond_keys if k != "target"]
    n_beaten = sum(1 for k in placebo_keys if summary[k]["specificity_gain_mean"] > 0
                   and summary[k]["specificity_ci_95"][0] > 0)
    md += "\nBootstrap: 10,000 city resamples stratified by fold, seed 42. "
    md += "The same resample indices are used for every condition and paired contrast.\n"
    md += ("Test direction follows the protocol: pre/post $\\Delta$CPC effects are two-sided, "
           "target-versus-placebo superiority contrasts are one-sided (greater). "
           "P-values are raw and unadjusted.\n")
    md += ("The target condition is an oracle reference built from the true target $Y_D$; it is "
           "not a CPC upper bound, and a placebo condition may exceed it on individual cities.\n\n")
    md += "### Run-derived summary\n\n"
    md += (f"- Target mean $\\Delta$CPC = `{target['mean_delta_cpc']:+.6f}` "
           f"(95% CI `[{target['ci_95'][0]:+.5f}, {target['ci_95'][1]:+.5f}]`), "
           f"win rate {target['win_rate']}, two-sided $p$ = `{_fmt_p(target['p_effect_two_sided'])}`.\n")
    md += (f"- {n_beaten}/{len(placebo_keys)} placebo conditions show a strictly positive "
           "specificity gain with a bootstrap CI excluding zero.\n")
    metadata = {"bootstrap": "city-level, fold-stratified", "n_boot": 10000, "seed": 42,
                "shared_resamples": True, "fold_counts": {str(k): int(v) for k, v in df_city.groupby("fold").size().items()},
                "wilcoxon": {"effect": "two-sided", "superiority": "one-sided (greater)"},
                "statistical_unit": "city, after seed averaging"}
    (output_dir / "bootstrap_method.json").write_text(json.dumps(metadata, indent=2) + "\n")

    (output_dir / "unified_placebo_summary.md").write_text(md, encoding="utf-8")
    print(f"Placebo summary written to {output_dir}.")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--b", type=int, default=1000)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("results/checkpoints"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/unified_placebo_v1"))
    parser.add_argument("--summary-only", type=Path, help="Recompute bootstrap from an existing per-city CSV; no experiment runs")
    args = parser.parse_args()
    if args.summary_only is not None:
        import hashlib
        source = args.summary_only.read_bytes()
        summarize_placebo(pd.read_csv(args.summary_only), args.output_dir)
        (args.output_dir / "summary_source.json").write_text(json.dumps({"path": str(args.summary_only.resolve()), "sha256": hashlib.sha256(source).hexdigest()}, indent=2) + "\n")
        sys.exit(0)
    run_unified_placebo(
        b_draws=args.b,
        data_root=args.data_root,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
    )
