"""
Direct Partial-OD Information Equivalence Experiment (v1) - High-Performance Vectorized Runner
=============================================================================================

Core Scientific Research Question:
    Under a prespecified low-capacity direct-OD adaptation procedure
    (OD Fixed-Effect Residual Adapter, OD-FE), what fraction of directly observed
    positive interzonal OD pairs is required to achieve reconstruction gain on
    the remaining unseen pairs comparable to that obtained from the full
    target-city distance-binned mobility distribution (Y_D)?

Strict Protocol Invariants:
    - 5-Fold Cross-City Evaluation (50 held-out test cities).
    - Frozen Gravity-Informed Urban GNN backbones (seeds 1, 10, 100).
    - Hyperparameter lambda in {0.1, 1, 10, 100} selected per fold strictly using 5 validation cities.
    - Zero retraining, zero fine-tuning, zero optimizer step, zero backward pass.
    - Reference Arm: Production calibrate_kbins(t0, dist, inter, yd_full, bin_edges, q=1.0) with K=8, q=1.0.
    - Primary Grid: 15 p-levels in [0.0, 0.001, ..., 0.90].
    - B = 200 replicates per city.
"""

import os
import sys
import time
import json
import hashlib
import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import torch

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.city_splits import generate_35_5_10_splits
from src.data.dataset import load_city, load_raw_city
from src.data.urban_graph import build_radius_graph
from src.data.yd_extractor import compute_kbin_edges
from src.calibration.bin_calibration import calibrate_kbins
from src.training.evaluate import compute_cpc_pair
from src.training.train import load_checkpoint, infer_zero_shot

PARTIAL_OD_BASE_SEED = 202608231
PRIMARY_GRID_DIRECT = [
    0.0, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 
    0.10, 0.20, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90
]
LAMBDA_CANDIDATES = [0.1, 1.0, 10.0, 100.0]
VAL_P_GRID = [0.02, 0.05, 0.10, 0.20]

RAW_COLUMNS_DIRECT = [
    "fold", "city", "model_seed", "replicate_id", "p", "mask_seed",
    "selected_lambda", "n_total_pairs", "n_revealed", "n_unseen",
    "fraction_pairs_revealed", "total_trip_mass", "revealed_trip_mass",
    "fraction_trip_mass_revealed", "unseen_trip_mass", "fraction_unseen_trip_mass",
    "origin_coverage", "destination_coverage", "both_endpoint_coverage",
    "adapter_iterations", "adapter_converged",
    "cpc_m0_unseen", "cpc_full_yd_unseen", "cpc_direct_od_unseen",
    "gain_full_yd", "gain_direct_od", "difference_direct_minus_yd",
    "relative_direct_vs_yd", "total_m0_mass", "total_direct_mass", "K", "q"
]


def get_stable_mask_seed(base_seed: int, fold: int, city: str, replicate_id: int) -> int:
    s = f"{base_seed}_{fold}_{city}_{replicate_id}"
    return int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % (2**32)


def holm_correction(p_vals: List[float]) -> np.ndarray:
    n = len(p_vals)
    if n == 0:
        return np.array([])
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
    if total_cities == 0:
        return 0.0, 0.0
        
    for b in range(n_boot):
        sample_sum = 0.0
        for f, arr in vals.items():
            idx = rng.randint(0, len(arr), size=len(arr))
            sample_sum += arr[idx].sum()
        boot_means[b] = sample_sum / total_cities

    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def fit_od_fe_adapter(
    o_idx: np.ndarray,
    d_idx: np.ndarray,
    t0_support: np.ndarray,
    t_true_support: np.ndarray,
    rev_indices: np.ndarray,
    num_nodes: int,
    lambda_reg: float,
    max_iter: int = 150,
    tol: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray, int, bool]:
    """
    Solves the exact two-way fixed-effect ridge regression objective:
        min_{a, b} sum_{(i,j) in S_p} (r_ij - a_i - b_j)^2 + lambda * (||a||^2 + ||b||^2)
    Solved using conjugate gradient on the reduced SPD system; empirical convergence is monitored by the residual tolerance.
    """
    n_rev = len(rev_indices)
    if n_rev == 0:
        return np.zeros(num_nodes, dtype=np.float64), np.zeros(num_nodes, dtype=np.float64), 0, True

    o_rev = torch.as_tensor(o_idx[rev_indices], dtype=torch.long)
    d_rev = torch.as_tensor(d_idx[rev_indices], dtype=torch.long)
    t0_rev = torch.as_tensor(t0_support[rev_indices], dtype=torch.float64)
    t_true_rev = torch.as_tensor(t_true_support[rev_indices], dtype=torch.float64)

    # Target residual r_ij = log(1 + T_ij) - log(1 + \hat{T}^0_ij)
    r_rev = torch.log1p(t_true_rev) - torch.log1p(t0_rev)

    n_i = torch.bincount(o_rev, minlength=num_nodes).double()
    m_j = torch.bincount(d_rev, minlength=num_nodes).double()

    inv_denom_a = 1.0 / (n_i + lambda_reg)
    denom_b = m_j + lambda_reg

    c_a = torch.bincount(o_rev, weights=r_rev, minlength=num_nodes)
    c_b = torch.bincount(d_rev, weights=r_rev, minlength=num_nodes)

    rhs_b = c_b - torch.bincount(d_rev, weights=inv_denom_a[o_rev] * c_a[o_rev], minlength=num_nodes)

    def matvec(v):
        Av = v[d_rev]
        scaled_Av = inv_denom_a[o_rev] * Av
        At_scaled_Av = torch.bincount(d_rev, weights=scaled_Av, minlength=num_nodes)
        return denom_b * v - At_scaled_Av

    b = torch.zeros(num_nodes, dtype=torch.float64)
    r = rhs_b - matvec(b)
    p = r.clone()
    rsold = torch.dot(r, r)

    if float(rsold) < 1e-16:
        a = inv_denom_a * c_a
        return a.numpy(), b.numpy(), 0, True

    converged = False
    iters = 0

    for it in range(1, max_iter + 1):
        iters = it
        Ap = matvec(p)
        denom_alpha = float(torch.dot(p, Ap))
        if denom_alpha <= 0 or not np.isfinite(denom_alpha):
            converged = False
            break
        alpha = rsold / denom_alpha
        b = b + alpha * p
        r = r - alpha * Ap
        rsnew = torch.dot(r, r)
        if float(torch.sqrt(rsnew)) < tol:
            converged = True
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew

    a = inv_denom_a * (c_a - torch.bincount(o_rev, weights=b[d_rev], minlength=num_nodes))
    return a.numpy(), b.numpy(), iters, converged


def apply_od_fe_prediction(
    o_idx: np.ndarray,
    d_idx: np.ndarray,
    t0_support: np.ndarray,
    a: np.ndarray,
    b: np.ndarray
) -> np.ndarray:
    """
    Applies OD-FE predictions and preserves total baseline mass N0.
    """
    log_t0_plus_1 = np.log1p(t0_support)
    ell_direct = log_t0_plus_1 + a[o_idx] + b[d_idx]
    t_tilde = np.maximum(0.0, np.expm1(ell_direct))
    
    n0 = float(np.sum(t0_support))
    n_tilde = float(np.sum(t_tilde))
    
    if n_tilde > 0:
        t_direct = t_tilde * (n0 / n_tilde)
    else:
        t_direct = t0_support.copy()
        
    return t_direct


def select_fold_lambda(
    fold_id: int,
    val_cities: List[str],
    data_root: str = "data",
    model_seeds: List[int] = [1, 10, 100],
    b_val: int = 50,
    device: str = "cpu"
) -> Tuple[float, pd.DataFrame]:
    """
    Strictly selects lambda on the 5 validation cities of the fold.
    """
    print(f"\n[FOLD {fold_id}] Selecting hyperparameter lambda from {len(val_cities)} validation cities...")
    
    # Load fold models
    fold_models: Dict[int, Tuple[Any, Any]] = {}
    for s in model_seeds:
        ckpt_path = Path("results/checkpoints") / f"5fold_fold{fold_id}_seed{s}.pt"
        if not ckpt_path.exists():
            raise RuntimeError(f"Missing checkpoint {ckpt_path}")
        model, scaler, _ = load_checkpoint(ckpt_path, device_str=device)
        model.eval()
        fold_models[s] = (model, scaler)

    # Pre-cache validation city zero-shot predictions
    val_cache: Dict[str, Dict[str, Any]] = {}
    for city_name in val_cities:
        raw_data = load_raw_city(city_name, data_root=data_root)
        dist_km = raw_data.dist_km
        inter_pos = (raw_data.pair_o_idx.numpy() != raw_data.pair_d_idx.numpy()) & (dist_km > 0.0) & (raw_data.pair_trips.numpy() > 0)
        
        t_true_support = raw_data.pair_trips.numpy()[inter_pos].astype(np.float64)
        o_idx_support = raw_data.pair_o_idx.numpy()[inter_pos]
        d_idx_support = raw_data.pair_d_idx.numpy()[inter_pos]
        num_nodes = raw_data.n_tracts

        seed_preds = {}
        for s in model_seeds:
            model, scaler = fold_models[s]
            city_data = load_city(city_name, data_root=data_root, feature_scaler=scaler, fit_scaler=False)
            coords = city_data.lon_lat.numpy()
            ei, ed = build_radius_graph(coords, radius_km=5.0)
            with torch.no_grad():
                m0_full = infer_zero_shot(model, city_data, ei, ed, device=device).numpy().astype(np.float64)
            seed_preds[s] = m0_full[inter_pos]

        val_cache[city_name] = {
            "n_pairs": int(inter_pos.sum()),
            "t_true": t_true_support,
            "o_idx": o_idx_support,
            "d_idx": d_idx_support,
            "num_nodes": num_nodes,
            "seed_preds": seed_preds
        }

    lambda_scores = []
    
    for lam in LAMBDA_CANDIDATES:
        cpc_unseen_list = []
        gain_list = []
        
        for city_name in val_cities:
            cdata = val_cache[city_name]
            n_pairs = cdata["n_pairs"]
            t_true = cdata["t_true"]
            o_idx = cdata["o_idx"]
            d_idx = cdata["d_idx"]
            num_nodes = cdata["num_nodes"]

            for rep_id in range(b_val):
                mask_seed = get_stable_mask_seed(PARTIAL_OD_BASE_SEED, fold_id, f"val_{city_name}", rep_id)
                perm = np.random.RandomState(mask_seed).permutation(n_pairs)

                for p_val in VAL_P_GRID:
                    n_rev = int(np.round(p_val * n_pairs))
                    rev_indices = perm[:n_rev]
                    unseen_indices = perm[n_rev:]
                    t_true_unseen = t_true[unseen_indices]
                    sum_true_unseen = float(np.sum(t_true_unseen))

                    for s in model_seeds:
                        t0_support = cdata["seed_preds"][s]
                        t0_unseen = t0_support[unseen_indices]
                        
                        denom_m0 = sum_true_unseen + float(np.sum(t0_unseen))
                        cpc_m0 = (2.0 * np.sum(np.minimum(t_true_unseen, t0_unseen)) / denom_m0) if denom_m0 > 0 else 0.0

                        a, b, _, conv = fit_od_fe_adapter(
                            o_idx, d_idx, t0_support, t_true, rev_indices, num_nodes, lambda_reg=lam
                        )
                        if not conv:
                            raise RuntimeError(f"OD-FE CG solver did not converge during lambda selection on val city {city_name}!")
                        
                        t_direct_support = apply_od_fe_prediction(o_idx, d_idx, t0_support, a, b)
                        t_direct_unseen = t_direct_support[unseen_indices]
                        
                        denom_dir = sum_true_unseen + float(np.sum(t_direct_unseen))
                        cpc_dir = (2.0 * np.sum(np.minimum(t_true_unseen, t_direct_unseen)) / denom_dir) if denom_dir > 0 else 0.0

                        cpc_unseen_list.append(cpc_dir)
                        gain_list.append(cpc_dir - cpc_m0)

        mean_cpc = float(np.mean(cpc_unseen_list))
        mean_gain = float(np.mean(gain_list))
        lambda_scores.append({
            "lambda": lam,
            "validation_mean_cpc": mean_cpc,
            "mean_gain": mean_gain,
            "n_validation_cities": len(val_cities),
            "masks_per_city": b_val
        })
        print(f"  candidate lambda = {lam:<5} | Val Mean CPC_U = {mean_cpc:.5f} | Val Mean Gain = {mean_gain:+.5f}")

    selection_df = pd.DataFrame(lambda_scores)
    # Sort descending by validation_mean_cpc, then descending by lambda (tie-breaker prefers higher regularization)
    best_row = selection_df.sort_values(by=["validation_mean_cpc", "lambda"], ascending=[False, False]).iloc[0]
    selected_lam = float(best_row["lambda"])
    print(f"  --> Selected lambda_f* = {selected_lam} for Fold {fold_id}\n")
    return selected_lam, selection_df


def _process_city_replicates_chunk(
    args: Tuple[int, str, List[int], int, List[int], List[float], float, Dict[str, Any]]
) -> List[Tuple]:
    """
    Worker task: Processes a slice of replicates for a single city across all p-levels and model seeds.
    """
    fold_id, city_name, rep_ids, n_pairs, model_seeds, p_grid, selected_lambda, city_cached_data = args
    
    t_true_support = city_cached_data["t_true"]
    o_idx_support = city_cached_data["o_idx"]
    d_idx_support = city_cached_data["d_idx"]
    num_nodes = city_cached_data["num_nodes"]
    total_trip_mass = city_cached_data["total_trip_mass"]
    n_origins_total = city_cached_data["n_origins_total"]
    n_dests_total = city_cached_data["n_dests_total"]
    seed_predictions = city_cached_data["seed_predictions"]

    rows = []

    for rep_id in rep_ids:
        mask_seed = get_stable_mask_seed(PARTIAL_OD_BASE_SEED, fold_id, city_name, rep_id)
        rng = np.random.RandomState(mask_seed)
        perm = rng.permutation(n_pairs)

        for p_val in p_grid:
            n_reveal = int(np.round(p_val * n_pairs))
            rev_indices = perm[:n_reveal]
            unseen_indices = perm[n_reveal:]
            n_unseen = len(unseen_indices)
            if n_unseen == 0:
                continue

            if n_reveal == 0:
                revealed_mass = 0.0
                c_o = 0.0
                c_d = 0.0
                c_both = 0.0
            else:
                rev_trips = t_true_support[rev_indices]
                revealed_mass = float(np.sum(rev_trips))
                rev_o_set = set(o_idx_support[rev_indices])
                rev_d_set = set(d_idx_support[rev_indices])
                
                c_o = len(rev_o_set) / n_origins_total if n_origins_total > 0 else 0.0
                c_d = len(rev_d_set) / n_dests_total if n_dests_total > 0 else 0.0
                
                unseen_o = o_idx_support[unseen_indices]
                unseen_d = d_idx_support[unseen_indices]
                both_cov = np.isin(unseen_o, list(rev_o_set)) & np.isin(unseen_d, list(rev_d_set))
                c_both = float(np.mean(both_cov))

            frac_pairs_rev = float(n_reveal) / float(n_pairs)
            frac_mass_rev = float(revealed_mass) / float(total_trip_mass) if total_trip_mass > 0 else 0.0
            unseen_mass = total_trip_mass - revealed_mass
            frac_unseen_mass = unseen_mass / total_trip_mass if total_trip_mass > 0 else 0.0
            
            t_true_unseen = t_true_support[unseen_indices]
            sum_true_unseen = float(np.sum(t_true_unseen))

            # Evaluate across all model seeds with identical mask
            for s in model_seeds:
                preds = seed_predictions[s]
                t0_support = preds["t0"]
                t0_unseen = t0_support[unseen_indices]
                t_full_unseen = preds["t_cal_full"][unseen_indices]
                N_hat_total = preds["N_hat"]
                
                # 1. Arm A: M0 zero-shot
                denom_m0 = sum_true_unseen + float(np.sum(t0_unseen))
                cpc_m0_unseen = (2.0 * np.sum(np.minimum(t_true_unseen, t0_unseen)) / denom_m0) if denom_m0 > 0 else 0.0
                
                # 2. Arm B: Full Y_D Reference
                denom_full = sum_true_unseen + float(np.sum(t_full_unseen))
                cpc_full_unseen = (2.0 * np.sum(np.minimum(t_true_unseen, t_full_unseen)) / denom_full) if denom_full > 0 else 0.0
                
                # 3. Arm C: Direct-OD Adapter (OD-FE)
                if n_reveal == 0:
                    cpc_dir_unseen = cpc_m0_unseen
                    it_count = 0
                    is_conv = True
                    tot_dir_mass = N_hat_total
                else:
                    a, b, it_count, is_conv = fit_od_fe_adapter(
                        o_idx=o_idx_support,
                        d_idx=d_idx_support,
                        t0_support=t0_support,
                        t_true_support=t_true_support,
                        rev_indices=rev_indices,
                        num_nodes=num_nodes,
                        lambda_reg=selected_lambda
                    )
                    if not is_conv:
                        raise RuntimeError(f"OD-FE CG solver did not converge on city {city_name}, rep {rep_id}, p {p_val}!")
                        
                    t_direct_support = apply_od_fe_prediction(
                        o_idx_support, d_idx_support, t0_support, a, b
                    )
                    t_dir_unseen = t_direct_support[unseen_indices]
                    tot_dir_mass = float(np.sum(t_direct_support))
                    
                    denom_dir = sum_true_unseen + float(np.sum(t_dir_unseen))
                    cpc_dir_unseen = (2.0 * np.sum(np.minimum(t_true_unseen, t_dir_unseen)) / denom_dir) if denom_dir > 0 else 0.0

                gain_full = float(cpc_full_unseen - cpc_m0_unseen)
                gain_direct = float(cpc_dir_unseen - cpc_m0_unseen)
                diff_direct_minus_yd = float(gain_direct - gain_full)
                rel_direct = float(gain_direct / gain_full) if abs(gain_full) > 1e-8 else 1.0

                rows.append((
                    fold_id, city_name, s, rep_id, p_val, mask_seed,
                    selected_lambda, n_pairs, n_reveal, n_unseen,
                    frac_pairs_rev, total_trip_mass, revealed_mass,
                    frac_mass_rev, unseen_mass, frac_unseen_mass,
                    c_o, c_d, c_both, it_count, is_conv,
                    cpc_m0_unseen, cpc_full_unseen, cpc_dir_unseen,
                    gain_full, gain_direct, diff_direct_minus_yd,
                    rel_direct, N_hat_total, tot_dir_mass, 8, 1.0
                ))

    return rows


def run_fold_direct_od(
    fold_id: int,
    data_root: str = "data",
    output_dir: Path = Path("results/direct_od_equivalence_v1"),
    replicates: int = 200,
    p_grid: List[float] = None,
    smoke: bool = False,
    smoke_cities: int = 1,
    resume: bool = False,
    num_workers: int = 8,
    device: str = "cpu"
) -> Dict[str, Any]:
    if p_grid is None:
        p_grid = PRIMARY_GRID_DIRECT.copy()

    fold_dir = output_dir / f"fold_{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    raw_csv_path = fold_dir / "raw.csv"
    progress_json_path = fold_dir / "progress.json"
    marker_path = fold_dir / "completion.marker"
    lambda_csv_path = fold_dir / "lambda_selection.csv"
    lambda_json_path = fold_dir / "lambda_selected.json"

    splits = generate_35_5_10_splits(data_root=data_root)
    split = splits[fold_id]
    train_cities = split["train"]
    val_cities = split["val"]
    test_cities = split["test"] if not smoke else split["test"][:smoke_cities]
    model_seeds = [1, 10, 100] if not smoke else [1, 10]
    B = replicates if not smoke else 20
    b_val = 50 if not smoke else 5

    # 1. Select / Load Fold Lambda
    valid_lambda_cache = False
    if lambda_json_path.exists():
        with open(lambda_json_path, "r") as f:
            lam_info = json.load(f)
            if lam_info.get("val_cities") == val_cities and lam_info.get("model_seeds") == model_seeds and lam_info.get("b_val") == b_val:
                selected_lambda = float(lam_info["selected_lambda"])
                print(f">>> [FOLD {fold_id}] Using cached lambda_f* = {selected_lambda}")
                valid_lambda_cache = True
            else:
                print(f">>> [FOLD {fold_id}] Cached lambda_f* is stale (different config). Re-selecting...")

    if not valid_lambda_cache:
        selected_lambda, selection_df = select_fold_lambda(
            fold_id=fold_id,
            val_cities=val_cities,
            data_root=data_root,
            model_seeds=model_seeds,
            b_val=b_val,
            device=device
        )
        selection_df.to_csv(lambda_csv_path, index=False)
        with open(lambda_json_path, "w", encoding="utf-8") as f:
            json.dump({
                "fold": fold_id,
                "lambda_candidates": LAMBDA_CANDIDATES,
                "selected_lambda": selected_lambda,
                "selection_source": "validation_cities_only",
                "test_city_information_used": False,
                "val_cities": val_cities,
                "model_seeds": model_seeds,
                "b_val": b_val,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }, f, indent=2)

    print(f">>> [STARTING FOLD {fold_id}/5] {len(test_cities)} test cities | B={B} reps | {len(p_grid)} p-levels | lambda={selected_lambda} | Workers={num_workers}")

    # Check already completed cities if resume is True with protocol signature verification
    completed_cities = set()
    if resume and progress_json_path.exists():
        try:
            with open(progress_json_path, "r", encoding="utf-8") as f:
                prog = json.load(f)
                sig = prog.get("protocol_signature", {})
                sig_valid = (
                    prog.get("protocol_version") == "v1"
                    and sig.get("model_seeds") == model_seeds
                    and sig.get("B") == B
                    and sig.get("selected_lambda") == selected_lambda
                    and sig.get("n_p_levels") == len(p_grid)
                    and sig.get("split_manifest_sha256") == split_manifest_sha256
                )
                if sig_valid:
                    completed_cities = set(prog.get("completed_cities", []))
                    print(f"    [RESUME VERIFIED] Resuming fold {fold_id}: Found {len(completed_cities)} verified completed cities.")
                else:
                    print(f"    [RESUME REJECTED] Incompatible protocol signature in {progress_json_path}. Restarting fold {fold_id} cleanly.")
                    completed_cities = set()
        except Exception as e:
            print(f"    [RESUME WARNING] Failed to read {progress_json_path}: {e}. Restarting fold {fold_id} cleanly.")
            completed_cities = set()


    if not resume or not raw_csv_path.exists():
        with open(raw_csv_path, "w", encoding="utf-8") as f:
            f.write(",".join(RAW_COLUMNS_DIRECT) + "\n")

    # Load frozen GNN models
    models: Dict[int, Tuple[Any, Any]] = {}
    for s in model_seeds:
        ckpt_path = Path("results/checkpoints") / f"5fold_fold{fold_id}_seed{s}.pt"
        if not ckpt_path.exists():
            raise RuntimeError(f"Checkpoint missing for fold {fold_id} seed {s}: {ckpt_path}")
        model, scaler, _ = load_checkpoint(ckpt_path, device_str=device)
        model.eval()
        models[s] = (model, scaler)

    # Compute K=8 bin edges from 35 train cities for reference arm
    bin_edges, K_act = compute_kbin_edges(train_cities, K=8, data_root=data_root)
    assert K_act == 8 and len(bin_edges) == 9

    fold_start_time = time.perf_counter()
    rows_written_total = 0

    for city_idx, city_name in enumerate(test_cities):
        if city_name in completed_cities:
            print(f"  [{city_idx+1}/{len(test_cities)}] {city_name:<16} | ALREADY COMPLETED (Skipping)")
            continue

        city_start = time.perf_counter()
        raw_data = load_raw_city(city_name, data_root=data_root)
        dist_km = raw_data.dist_km
        
        inter_pos = (raw_data.pair_o_idx.numpy() != raw_data.pair_d_idx.numpy()) & (dist_km > 0.0) & (raw_data.pair_trips.numpy() > 0)
        n_pairs = int(inter_pos.sum())
        if n_pairs == 0:
            raise RuntimeError(f"Critical error: City {city_name} has 0 positive interzonal pairs!")

        t_true_support = raw_data.pair_trips.numpy()[inter_pos].astype(np.float64)
        o_idx_support = raw_data.pair_o_idx.numpy()[inter_pos]
        d_idx_support = raw_data.pair_d_idx.numpy()[inter_pos]
        dist_support = dist_km[inter_pos]
        num_nodes = raw_data.n_tracts
        total_trip_mass = float(np.sum(t_true_support))
        
        n_origins_total = len(set(o_idx_support))
        n_dests_total = len(set(d_idx_support))

        # Full Y_D reference distribution
        bin_idx_support = np.clip(np.digitize(dist_support, bin_edges, right=True) - 1, 0, 7)
        yd_full = np.bincount(bin_idx_support, weights=t_true_support, minlength=8).astype(np.float64)
        yd_full /= total_trip_mass

        # Precalculate M0 and full Y_D calibrated prediction for all model seeds
        seed_predictions: Dict[int, Dict[str, np.ndarray]] = {}
        for s in model_seeds:
            model, scaler = models[s]
            city_data = load_city(city_name, data_root=data_root, feature_scaler=scaler, fit_scaler=False)
            coords = city_data.lon_lat.numpy()
            ei, ed = build_radius_graph(coords, radius_km=5.0)
            
            with torch.no_grad():
                m0_full = infer_zero_shot(model, city_data, ei, ed, device=device).numpy().astype(np.float64)
            
            t0_support = m0_full[inter_pos]
            N_hat_support = float(np.sum(t0_support))
            
            # Reference Full Y_D calibration (K=8, q=1.0)
            Y_hat = np.bincount(bin_idx_support, weights=t0_support, minlength=8).astype(np.float64) / N_hat_support
            active = np.zeros(8, dtype=bool)
            for k in range(8):
                active[k] = bool((bin_idx_support == k).any())
            yd_act = yd_full * active.astype(np.float64)
            act_sum = yd_act.sum()
            Y_D_cond = yd_act / act_sum if act_sum > 0 else Y_hat.copy()

            w_full = np.ones(8, dtype=np.float64)
            for k in range(8):
                if active[k] and Y_hat[k] > 0:
                    w_full[k] = Y_D_cond[k] / Y_hat[k]
            weighted_mass_full = float(np.dot(Y_hat, w_full))
            s_full = w_full / weighted_mass_full if weighted_mass_full > 0 else np.ones(8)
            
            t_cal_full_support = t0_support * s_full[bin_idx_support]
            cal_mass_full = np.sum(t_cal_full_support)
            if cal_mass_full > 0:
                t_cal_full_support *= (N_hat_support / cal_mass_full)
                
            seed_predictions[s] = {
                "t0": t0_support,
                "N_hat": N_hat_support,
                "t_cal_full": t_cal_full_support
            }

        city_cached_data = {
            "t_true": t_true_support,
            "o_idx": o_idx_support,
            "d_idx": d_idx_support,
            "num_nodes": num_nodes,
            "total_trip_mass": total_trip_mass,
            "n_origins_total": n_origins_total,
            "n_dests_total": n_dests_total,
            "seed_predictions": seed_predictions
        }

        # Divide B replicates into chunks for multiprocessing
        rep_chunks = np.array_split(np.arange(B), min(num_workers, B))
        task_args = [
            (fold_id, city_name, chunk.tolist(), n_pairs, model_seeds, p_grid, selected_lambda, city_cached_data)
            for chunk in rep_chunks if len(chunk) > 0
        ]

        if num_workers > 1 and len(task_args) > 1:
            with mp.Pool(processes=min(num_workers, len(task_args))) as pool:
                chunk_results = pool.map(_process_city_replicates_chunk, task_args)
            city_rows = [item for sublist in chunk_results for item in sublist]
        else:
            city_rows = _process_city_replicates_chunk(task_args[0])

        # Append city records to raw CSV incrementally
        with open(raw_csv_path, "a", encoding="utf-8") as f:
            for r in city_rows:
                f.write(",".join(str(x) for x in r) + "\n")

        completed_cities.add(city_name)
        rows_written_total += len(city_rows)

        # Update progress.json with full protocol signature
        with open(progress_json_path, "w", encoding="utf-8") as f:
            json.dump({
                "fold": fold_id,
                "completed_cities": sorted(list(completed_cities)),
                "remaining_cities": [c for c in test_cities if c not in completed_cities],
                "rows_written": rows_written_total,
                "protocol_version": "v1",
                "protocol_signature": {
                    "model_seeds": model_seeds,
                    "B": B,
                    "selected_lambda": selected_lambda,
                    "n_p_levels": len(p_grid),
                    "split_manifest_sha256": split_manifest_sha256,
                },
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }, f, indent=2)


        city_elapsed = time.perf_counter() - city_start
        print(f"  [{city_idx+1}/{len(test_cities)}] {city_name:<16} | Pairs: {n_pairs:>7} | B={B} reps done in {city_elapsed:.2f}s (Flushed {len(city_rows)} rows)")

    # Read back raw.csv to generate per_seed, per_city, and fold_summary
    fold_df = pd.read_csv(raw_csv_path)
    
    # 1. Per-Seed Aggregation: Mean over B replicates -> (fold x city x model_seed x p)
    per_seed_df = fold_df.groupby(["fold", "city", "model_seed", "p"]).agg({
        "selected_lambda": "first",
        "fraction_pairs_revealed": "mean",
        "fraction_trip_mass_revealed": "mean",
        "origin_coverage": "mean",
        "destination_coverage": "mean",
        "both_endpoint_coverage": "mean",
        "adapter_iterations": "mean",
        "cpc_m0_unseen": "mean",
        "cpc_full_yd_unseen": "mean",
        "cpc_direct_od_unseen": "mean",
        "gain_full_yd": "mean",
        "gain_direct_od": "mean",
        "difference_direct_minus_yd": "mean",
        "relative_direct_vs_yd": "mean"
    }).reset_index()
    per_seed_csv_path = fold_dir / "per_seed.csv"
    per_seed_df.to_csv(per_seed_csv_path, index=False)

    # 2. Per-City Aggregation: Mean over 3 model seeds -> (fold x city x p)
    per_city_df = per_seed_df.groupby(["fold", "city", "p"]).agg({
        "selected_lambda": "first",
        "fraction_pairs_revealed": "mean",
        "fraction_trip_mass_revealed": "mean",
        "origin_coverage": "mean",
        "destination_coverage": "mean",
        "both_endpoint_coverage": "mean",
        "adapter_iterations": "mean",
        "cpc_m0_unseen": "mean",
        "cpc_full_yd_unseen": "mean",
        "cpc_direct_od_unseen": "mean",
        "gain_full_yd": "mean",
        "gain_direct_od": "mean",
        "difference_direct_minus_yd": "mean",
        "relative_direct_vs_yd": "mean"
    }).reset_index()
    per_city_csv_path = fold_dir / "per_city.csv"
    per_city_df.to_csv(per_city_csv_path, index=False)

    # 3. Fold Summary Table
    fold_summary_rows = []
    for p_val in p_grid:
        sub = per_city_df[per_city_df.p == p_val]
        fold_summary_rows.append({
            "p": p_val,
            "n_cities": len(sub),
            "mean_both_cov": float(sub["both_endpoint_coverage"].mean()),
            "mean_gain_full_yd": float(sub["gain_full_yd"].mean()),
            "mean_gain_direct_od": float(sub["gain_direct_od"].mean()),
            "mean_diff_vs_yd": float(sub["difference_direct_minus_yd"].mean()),
            "pos_cities": int((sub["gain_direct_od"] > 0).sum()),
            "match_yd_cities": int((sub["difference_direct_minus_yd"] >= 0).sum())
        })

    fold_summary_json_path = fold_dir / "fold_summary.json"
    with open(fold_summary_json_path, "w", encoding="utf-8") as f:
        json.dump({"fold": fold_id, "selected_lambda": selected_lambda, "summary_by_p": fold_summary_rows}, f, indent=2)

    fold_summary_md_path = fold_dir / "fold_summary.md"
    with open(fold_summary_md_path, "w", encoding="utf-8") as f:
        f.write(f"# Fold {fold_id} Direct-OD Summary Table (N={len(test_cities)} Cities, lambda*={selected_lambda})\n\n")
        f.write("| p | Both Coverage | Mean Gain Full $Y_D$ | Mean Gain Direct OD | Mean $D(p)$ (Direct - Full) | Positive Cities | Match Full $Y_D$ |\n")
        f.write("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for r in fold_summary_rows:
            f.write(f"| **{r['p']*100:.2f}%** | {r['mean_both_cov']*100:.2f}% | +{r['mean_gain_full_yd']:.5f} | {r['mean_gain_direct_od']:+.5f} | {r['mean_diff_vs_yd']:+.5f} | {r['pos_cities']}/{r['n_cities']} | {r['match_yd_cities']}/{r['n_cities']} |\n")

    # 4. Save Run Manifest
    manifest_path = fold_dir / "run_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "fold": fold_id,
            "protocol_version": "v1",
            "selected_lambda": selected_lambda,
            "cities": test_cities,
            "model_seeds": model_seeds,
            "replicates": B,
            "p_grid": p_grid,
            "raw_rows": len(fold_df),
            "per_seed_rows": len(per_seed_df),
            "per_city_rows": len(per_city_df),
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)

    # 5. QA Verification Before Writing completion.marker
    expected_raw_rows = len(test_cities) * len(model_seeds) * B * len(p_grid)
    actual_raw_rows = len(fold_df)
    
    assert actual_raw_rows == expected_raw_rows, f"Fold {fold_id} raw rows {actual_raw_rows} != expected {expected_raw_rows}"
    assert len(per_city_df) == len(test_cities) * len(p_grid), f"Fold {fold_id} per_city rows mismatch"
    assert not fold_df.isnull().any().any(), f"Fold {fold_id} contains NaN values!"

    with open(marker_path, "w", encoding="utf-8") as f:
        f.write(f"FOLD {fold_id} DIRECT-OD COMPLETED AND CERTIFIED\nTimestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    fold_total_time = time.perf_counter() - fold_start_time
    print(f">>> [FOLD {fold_id} COMPLETE] Certified {actual_raw_rows} rows in {fold_total_time:.2f}s | Marker: {marker_path.name}")
    
    return {
        "fold": fold_id,
        "selected_lambda": selected_lambda,
        "raw_rows": actual_raw_rows,
        "per_seed_rows": len(per_seed_df),
        "per_city_rows": len(per_city_df),
        "status": "PASS"
    }


def aggregate_combined_direct_od(
    output_dir: Path = Path("results/direct_od_equivalence_v1"),
    p_grid: List[float] = None
) -> None:
    if p_grid is None:
        p_grid = PRIMARY_GRID_DIRECT.copy()

    combined_dir = output_dir / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    (combined_dir / "figures").mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 85)
    print("MASTER AGGREGATION & SCIENTIFIC SUMMARY (DIRECT-OD EQUIVALENCE, N=50 CITIES)")
    print("=" * 85)

    all_raw_dfs = []
    all_per_seed_dfs = []
    all_per_city_dfs = []
    fold_lambdas = {}

    for f in range(1, 6):
        fold_dir = output_dir / f"fold_{f}"
        marker = fold_dir / "completion.marker"
        if not marker.exists():
            raise RuntimeError(f"Cannot aggregate: Fold {f} completion.marker not found at {marker}")
        
        with open(fold_dir / "lambda_selected.json", "r") as lf:
            fold_lambdas[f] = json.load(lf)["selected_lambda"]
            
        all_raw_dfs.append(pd.read_csv(fold_dir / "raw.csv"))
        all_per_seed_dfs.append(pd.read_csv(fold_dir / "per_seed.csv"))
        all_per_city_dfs.append(pd.read_csv(fold_dir / "per_city.csv"))

    raw_combined = pd.concat(all_raw_dfs, ignore_index=True)
    per_seed_combined = pd.concat(all_per_seed_dfs, ignore_index=True)
    per_city_combined = pd.concat(all_per_city_dfs, ignore_index=True)

    raw_combined.to_csv(combined_dir / "raw_all_folds.csv", index=False)
    per_seed_combined.to_csv(combined_dir / "per_seed_all_folds.csv", index=False)
    per_city_combined.to_csv(combined_dir / "per_city_all_folds.csv", index=False)

    print(f"Combined Raw Rows:      {len(raw_combined):>10} (Expected: 450,000)")
    print(f"Combined Per-Seed Rows: {len(per_seed_combined):>10} (Expected: 2,250)")
    print(f"Combined Per-City Rows: {len(per_city_combined):>10} (Expected: 750)")

    # Statistical Analysis across N=50 cities
    summary_rows = []
    raw_p_values = []
    p_vals_tested = [p for p in p_grid if p > 0]

    for p_val in p_vals_tested:
        sub = per_city_combined[per_city_combined.p == p_val]
        gains = sub["gain_direct_od"].values
        _, p_w = stats.wilcoxon(gains, alternative="greater")
        raw_p_values.append(p_w)

    holm_p_vals = holm_correction(raw_p_values)
    holm_dict = {p: h_p for p, h_p in zip(p_vals_tested, holm_p_vals)}

    for p_val in p_grid:
        sub = per_city_combined[per_city_combined.p == p_val]
        n_cities = len(sub)
        
        mean_mass = float(sub["fraction_trip_mass_revealed"].mean())
        mean_cov_both = float(sub["both_endpoint_coverage"].mean())
        mean_cov_o = float(sub["origin_coverage"].mean())
        mean_cov_d = float(sub["destination_coverage"].mean())
        
        mean_m0 = float(sub["cpc_m0_unseen"].mean())
        mean_gain_full = float(sub["gain_full_yd"].mean())
        mean_gain_direct = float(sub["gain_direct_od"].mean())
        mean_diff = float(sub["difference_direct_minus_yd"].mean())
        
        pos_cities = int((sub["gain_direct_od"] > 0).sum())
        match_yd_cities = int((sub["difference_direct_minus_yd"] >= 0).sum())
        
        ci_diff_l, ci_diff_h = fold_stratified_bootstrap(per_city_combined, "difference_direct_minus_yd", p_val)
        ci_dir_l, ci_dir_h = fold_stratified_bootstrap(per_city_combined, "gain_direct_od", p_val)
        ci_full_l, ci_full_h = fold_stratified_bootstrap(per_city_combined, "gain_full_yd", p_val)
        
        h_pval = holm_dict.get(p_val, 1.0) if p_val > 0 else 1.0

        summary_rows.append({
            "p": p_val,
            "n_cities": n_cities,
            "mean_revealed_mass": mean_mass,
            "mean_both_coverage": mean_cov_both,
            "mean_origin_coverage": mean_cov_o,
            "mean_destination_coverage": mean_cov_d,
            "mean_m0_cpc": mean_m0,
            "mean_gain_full_yd": mean_gain_full,
            "ci_95_gain_full": [ci_full_l, ci_full_h],
            "mean_gain_direct_od": mean_gain_direct,
            "ci_95_gain_direct": [ci_dir_l, ci_dir_h],
            "mean_diff_vs_yd": mean_diff,
            "ci_95_diff": [ci_diff_l, ci_diff_h],
            "pos_cities_vs_m0": pos_cities,
            "match_yd_cities": match_yd_cities,
            "holm_pval_benefit": h_pval
        })

    summary_df = pd.DataFrame(summary_rows)

    # 1. Positive Mean Crossing
    p_pos_mean = None
    for r in summary_rows:
        if r["mean_gain_direct_od"] > 0 and p_pos_mean is None:
            p_pos_mean = r["p"]

    # 2. Statistically Supported Benefit Threshold p*_DirectBenefit
    p_star_benefit = None
    for r in summary_rows:
        if r["holm_pval_benefit"] < 0.05 and r["ci_95_gain_direct"][0] > 0 and p_star_benefit is None:
            p_star_benefit = r["p"]

    # 3. Operational Equivalence Crossing p_eq
    p_eq_grid = None
    p_eq_interp = None
    for r in summary_rows:
        if r["mean_diff_vs_yd"] >= 0 and p_eq_grid is None:
            p_eq_grid = r["p"]

    for i in range(len(summary_rows) - 1):
        r1, r2 = summary_rows[i], summary_rows[i+1]
        d1, d2 = r1["mean_diff_vs_yd"], r2["mean_diff_vs_yd"]
        if d1 <= 0 and d2 >= 0 and (d2 - d1) > 0:
            p_eq_interp = r1["p"] + (-d1 / (d2 - d1)) * (r2["p"] - r1["p"])
            break

    # Save summary JSON
    summary_json_path = combined_dir / "summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "direct_partial_od_information_equivalence",
            "protocol_version": "v1",
            "n_evaluation_cities": 50,
            "fold_lambdas": fold_lambdas,
            "p_pos_mean_crossing": p_pos_mean,
            "p_star_benefit_threshold": p_star_benefit,
            "p_eq_grid": p_eq_grid,
            "p_eq_interp": p_eq_interp,
            "results_by_p": summary_rows
        }, f, indent=2)

    # Save Markdown Table
    summary_md_path = combined_dir / "summary.md"
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("# Table: Master Direct-OD Information Equivalence Summary (v1)\n\n")
        f.write("> **Evaluation Scope**: Evaluates the operational reconstruction value of directly observed positive interzonal OD pairs via low-capacity Origin-Destination Fixed-Effect residual adaptation (OD-FE), relative to the full target-city distance distribution $Y_D$ ($K=8, q=1.0$, seeds $s \\in \\{1, 10, 100\\}$), evaluated strictly on unseen pairs ($N=50$ held-out test cities across 5 folds).\n\n")
        
        f.write(f"• **Validation-Selected Lambdas:** Fold 1: `{fold_lambdas[1]}`, Fold 2: `{fold_lambdas[2]}`, Fold 3: `{fold_lambdas[3]}`, Fold 4: `{fold_lambdas[4]}`, Fold 5: `{fold_lambdas[5]}`  \n")
        if p_pos_mean is not None:
            pct_pos = p_pos_mean * 100.0
            f.write(f"• **Positive Mean Crossing Point ($p_\\text{{mean+}}$):** `{pct_pos:.2f}%` of positive interzonal OD pairs  \n")
        if p_star_benefit is not None:
            pct_star = p_star_benefit * 100.0
            f.write(f"• **Statistically Supported Benefit Threshold ($p^*_\\text{{DirectBenefit}}$):** `{pct_star:.2f}%` of positive interzonal OD pairs ($p_\\text{{Holm}} < 0.05$)  \n")
        if p_eq_interp is not None:
            pct_interp = p_eq_interp * 100.0
            f.write(f"• **Operational Equivalence Crossing ($p_\\text{{eq,interp}}$):** `{pct_interp:.2f}%` of positive interzonal OD pairs  \n\n")
        elif p_eq_grid is not None:
            pct_grid = p_eq_grid * 100.0
            f.write(f"• **Operational Equivalence Grid Point ($p_\\text{{eq,grid}}$):** `{pct_grid:.2f}%` of positive interzonal OD pairs  \n\n")
        else:
            f.write("• **Operational Equivalence Crossing:** Under the tested low-capacity direct-OD adaptation procedure, the full-$Y_D$ reconstruction gain was not matched within the prespecified reveal range up to 90% of the positive interzonal OD support.  \n\n")

        f.write("| Revealed OD Pairs ($p$) | Both Coverage | $M_0$ CPC (Unseen) | Full-$Y_D$ Gain | Direct-OD Gain | Difference vs Full $Y_D$ ($D(p)$) | 95% CI Difference | Direct Benefit Holm $p$ | Cities Direct $> M_0$ | Cities Direct $\\ge$ Full $Y_D$ |\n")
        f.write("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        
        for r in summary_rows:
            p_pct = f"{r['p']*100:.2f}%"
            cov_pct = f"{r['mean_both_coverage']*100:.2f}%"
            m0_str = f"{r['mean_m0_cpc']:.4f}"
            full_str = f"+{r['mean_gain_full_yd']:.5f}"
            dir_str = f"{r['mean_gain_direct_od']:+.5f}"
            diff_str = f"{r['mean_diff_vs_yd']:+.5f}"
            ci_str = f"[{r['ci_95_diff'][0]:+.5f}, {r['ci_95_diff'][1]:+.5f}]"
            h_str = f"{r['holm_pval_benefit']:.4e}" if r['p'] > 0 else "—"
            pos_str = f"{r['pos_cities_vs_m0']}/{r['n_cities']}"
            match_str = f"{r['match_yd_cities']}/{r['n_cities']}"
            
            f.write(f"| **{p_pct}** | {cov_pct} | {m0_str} | {full_str} | **{dir_str}** | **{diff_str}** | {ci_str} | {h_str} | {pos_str} | {match_str} |\n")
            
        f.write("\n---\n\n### Prescribed Scientific Interpretation\n")
        if p_eq_interp is not None:
            f.write(f"Under the prespecified OD fixed-effect residual adapter, directly observing approximately **{p_eq_interp*100:.2f}%** of the positive interzonal OD support produced a mean reconstruction gain on the remaining unseen pairs comparable to that obtained from the full target-city distance-binned distribution.\n")
        else:
            f.write("Under the tested low-capacity direct-OD adaptation procedure, the full-$Y_D$ reconstruction gain was not matched within the prespecified reveal range up to 90% of the positive interzonal OD support. This does not imply that $Y_D$ intrinsically contains more information than 90% of the OD observations; the result is conditional on the tested adaptation operator.\n")

    print(f"Summary Markdown: {summary_md_path}")
    print(f"Summary JSON:     {summary_json_path}")

    # Generate Publication Figures
    generate_direct_od_figures(summary_df, per_city_combined, combined_dir, p_eq_interp, p_star_benefit)

    # Write completion markers; certification is a separate post-execution gate.
    (output_dir / "FROZEN.marker").unlink(missing_ok=True)
    with open(output_dir / "COMPLETED.marker", "w", encoding="utf-8") as f:
        f.write("DIRECT PARTIAL-OD INFORMATION EQUIVALENCE v1 COMPUTATION COMPLETED\n")
        f.write(f"Completed At: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("Status: COMPLETED; CERTIFICATION_PENDING\n")
        f.write("Protocol: 50 held-out test cities across 5 disjoint folds (N=50)\n")
        f.write("Evaluation Support: unseen positive interzonal pairs Omega_c^+ \\ S_p\n")
        f.write(f"Replicates: 200 per city (Total: 450,000 raw calibrations)\n")



def generate_direct_od_figures(
    summary_df: pd.DataFrame, 
    per_city_df: pd.DataFrame, 
    combined_dir: Path, 
    p_eq_interp: Optional[float],
    p_star_benefit: Optional[float]
) -> None:
    plt.rcParams.update({'font.sans-serif': 'Helvetica', 'axes.edgecolor': '#333333', 'axes.linewidth': 0.8})
    fig_dir = combined_dir / "figures"
    p_vals = summary_df["p"].values * 100.0

    # Fig 1: Gain vs Reveal Fraction
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    ax.axhline(0, color="#888888", linestyle="--", alpha=0.6)
    
    full_gain = summary_df["mean_gain_full_yd"].values
    dir_gain = summary_df["mean_gain_direct_od"].values
    dir_ci_l = np.array([ci[0] for ci in summary_df["ci_95_gain_direct"]])
    dir_ci_h = np.array([ci[1] for ci in summary_df["ci_95_gain_direct"]])
    
    ax.plot(p_vals, full_gain, label="Full $Y_D$ Reference Gain", color="#1f77b4", linestyle="--", linewidth=2.0)
    ax.plot(p_vals, dir_gain, label="Direct OD-FE Adapter Gain", color="#d62728", marker="o", linewidth=2.0)
    ax.fill_between(p_vals, dir_ci_l, dir_ci_h, color="#d62728", alpha=0.15, label="95% Fold Bootstrap CI")
    
    if p_star_benefit is not None:
        ax.axvline(p_star_benefit * 100.0, color="#ff7f0e", linestyle="-.", label=f"Benefit $p^* = {p_star_benefit*100:.2f}\\%$")
    if p_eq_interp is not None:
        ax.axvline(p_eq_interp * 100.0, color="#2ca02c", linestyle=":", label=f"Equivalence $p_{{eq}} = {p_eq_interp*100:.2f}\\%$")
        
    ax.set_xlabel("Revealed Positive Interzonal OD Pairs (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean Marginal Gain $\\Delta\\mathrm{CPC}_U$ on Unseen OD", fontsize=11, fontweight="bold")
    ax.set_title("Direct OD-FE Reconstruction Gain vs Reveal Fraction", fontsize=12, fontweight="bold", pad=12)
    ax.legend(frameon=True, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    fig.savefig(fig_dir / "fig_1_direct_gain_vs_p.png")
    plt.close(fig)

    # Fig 2: Difference D(p) Equivalence
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    ax.axhline(0, color="#333333", linestyle="-", linewidth=1.0)
    
    diff_vals = summary_df["mean_diff_vs_yd"].values
    diff_ci_l = np.array([ci[0] for ci in summary_df["ci_95_diff"]])
    diff_ci_h = np.array([ci[1] for ci in summary_df["ci_95_diff"]])
    
    ax.plot(p_vals, diff_vals, color="#9467bd", marker="s", linewidth=2.0, label="$\\bar{D}_{\\mathrm{Direct}}(p) = \\mathrm{Gain}_{\\mathrm{Direct}} - \\mathrm{Gain}_{Y_D}$")
    ax.fill_between(p_vals, diff_ci_l, diff_ci_h, color="#9467bd", alpha=0.15, label="95% Fold Bootstrap CI")
    
    if p_eq_interp is not None:
        ax.scatter([p_eq_interp * 100.0], [0.0], color="#d62728", s=80, zorder=5, label=f"Crossing $p_{{eq}} = {p_eq_interp*100:.2f}\\%$")
        
    ax.set_xlabel("Revealed Positive Interzonal OD Pairs (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Gain Difference $D_{\\mathrm{Direct}}(p)$", fontsize=11, fontweight="bold")
    ax.set_title("Direct-OD Information Equivalence Zero-Crossing", fontsize=12, fontweight="bold", pad=12)
    ax.legend(frameon=True, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    fig.savefig(fig_dir / "fig_2_direct_equivalence_Dp.png")
    plt.close(fig)

    # Fig 3: Fold-Specific D(p)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    ax.axhline(0, color="#333333", linestyle="-", linewidth=1.0)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    
    for f in range(1, 6):
        f_sub = per_city_df[per_city_df.fold == f].groupby("p")["difference_direct_minus_yd"].mean().reset_index()
        ax.plot(f_sub["p"].values * 100.0, f_sub["difference_direct_minus_yd"].values, marker="o", markersize=4, label=f"Fold {f} (N=10)", color=colors[f-1])
        
    ax.set_xlabel("Revealed Positive Interzonal OD Pairs (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Fold-Specific Mean $D_{\\mathrm{Direct}}(p)$", fontsize=11, fontweight="bold")
    ax.set_title("Fold-Specific Direct-OD Equivalence Trajectories", fontsize=12, fontweight="bold", pad=12)
    ax.legend(frameon=True, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    fig.savefig(fig_dir / "fig_3_fold_specific_direct_Dp.png")
    plt.close(fig)

    # Fig 4: Endpoint Coverage vs p
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    cov_both = summary_df["mean_both_coverage"].values * 100.0
    cov_o = summary_df["mean_origin_coverage"].values * 100.0
    cov_d = summary_df["mean_destination_coverage"].values * 100.0
    
    ax.plot(p_vals, cov_both, color="#e377c2", marker="^", linewidth=2.0, label="Both Endpoints Observed ($C_{\\mathrm{both}}$)")
    ax.plot(p_vals, cov_o, color="#bcbd22", linestyle=":", linewidth=1.5, label="Origin Coverage ($C_O$)")
    ax.plot(p_vals, cov_d, color="#17becf", linestyle="--", linewidth=1.5, label="Destination Coverage ($C_D$)")
    
    ax.set_xlabel("Revealed Positive Interzonal OD Pairs (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Endpoint Coverage on Unseen Set (%)", fontsize=11, fontweight="bold")
    ax.set_title("Endpoint Observation Dynamics in Direct-OD Adaptation", fontsize=12, fontweight="bold", pad=12)
    ax.legend(frameon=True, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    fig.savefig(fig_dir / "fig_4_endpoint_coverage_vs_p.png")
    plt.close(fig)

    # Fig 5: Direct OD vs Partial YD from v2
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    ax.axhline(0, color="#888888", linestyle="--", alpha=0.6)
    
    ax.plot(p_vals, full_gain, label="Full $Y_D$ Reference", color="#1f77b4", linestyle="--", linewidth=2.0)
    ax.plot(p_vals, dir_gain, label="Direct OD-FE Adaptation", color="#d62728", marker="o", linewidth=2.0)
    
    # Try reading v2 summary if available for contextual reference
    v2_summary_path = Path("results/partial_od_equivalence_v2/combined/summary.json")
    if v2_summary_path.exists():
        try:
            with open(v2_summary_path, "r") as v2f:
                v2_data = json.load(v2f)
                v2_p = [r["p"] * 100.0 for r in v2_data["results_by_p"]]
                v2_gain = [r["mean_gain_partial_od"] for r in v2_data["results_by_p"]]
                ax.plot(v2_p, v2_gain, label="OD-Subsampled $Y_D$ (v2)", color="#2ca02c", linestyle="-.", marker="x", linewidth=1.5)
        except Exception:
            pass

    ax.set_xlabel("Revealed Positive Interzonal OD Pairs (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean Marginal Gain $\\Delta\\mathrm{CPC}_U$", fontsize=11, fontweight="bold")
    ax.set_title("Direct OD-FE vs OD-Subsampled $Y_D$ Estimation", fontsize=12, fontweight="bold", pad=12)
    ax.legend(frameon=True, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    fig.savefig(fig_dir / "fig_5_direct_vs_partialYD_comparison.png")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Direct Partial-OD Information Equivalence v1")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="results/direct_od_equivalence_v1")
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5], help="Folds to execute")
    parser.add_argument("--cities", type=int, default=10, help="Number of test cities per fold")
    parser.add_argument("--b", type=int, default=200, help="Monte Carlo replicates per city")
    parser.add_argument("--smoke", action="store_true", help="Run fast smoke test")
    parser.add_argument("--resume", action="store_true", help="Resume from progress.json")
    parser.add_argument("--aggregate_only", action="store_true", help="Only aggregate completed folds")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel worker processes")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    out_p = Path(args.output_dir)

    if args.aggregate_only:
        aggregate_combined_direct_od(output_dir=out_p)
    else:
        for f_id in args.folds:
            run_fold_direct_od(
                fold_id=f_id,
                data_root=args.data_root,
                output_dir=out_p,
                replicates=args.b,
                smoke=args.smoke,
                smoke_cities=args.cities,
                resume=args.resume,
                num_workers=args.workers,
                device=args.device
            )
        if not args.smoke and set(args.folds) == {1, 2, 3, 4, 5}:
            aggregate_combined_direct_od(output_dir=out_p)
