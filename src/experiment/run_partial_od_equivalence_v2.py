r"""
Partial-OD Information Equivalence Experiment v2 (Final Paper Protocol)
========================================================================

Core Scientific Research Question:
    Under the same frozen zero-shot model and the same production distance-bin
    calibration operator, what fraction of directly observed positive interzonal
    OD pairs is required to achieve reconstruction gain comparable to that
    obtained from the full target-city distance-binned mobility distribution?

Primary Estimands:
    1. Positive-Benefit Threshold p*_benefit (Holm p < 0.05, CI_lower > 0)
    2. Operational Equivalence Crossing p_eq (where mean D(p) = Gain_OD(p) - Gain_YD(p) >= 0)

Architectural Invariants:
    - 5 Folds, 50 held-out test cities (35 train / 5 val / 10 test per fold).
    - Model Seeds: {1, 10, 100} on frozen Gravity-Informed Urban GNN.
    - Zero retraining, zero fine-tuning, zero optimizer step, zero backward pass.
    - K = 8 distance bins, q = 1.0 within-tolerance multiplier scaling.
    - Calibration operator executed on full candidate support Omega_c^+, scored strictly on unseen U_p = Omega_c^+ \ S_p.
    - Nested permutation masks across 15 p-levels:
      [0.0, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90].
    - B = 500 Monte Carlo replicates per city.
    - Exact Per-Fold Storage Structure with incremental flush and completion markers.
"""

import os
import sys
import time
import json
import hashlib
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

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
PRIMARY_GRID_V2 = [
    0.0, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 
    0.10, 0.20, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90
]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_hashes(fold_id: int, model_seeds: List[int]) -> Dict[str, str]:
    hashes = {}
    for seed in model_seeds:
        path = Path("results/checkpoints") / f"5fold_fold{fold_id}_seed{seed}.pt"
        if not path.exists():
            raise RuntimeError(f"Required checkpoint missing for fold {fold_id} seed {seed}: {path}")
        hashes[str(seed)] = _sha256_file(path)
    return hashes

RAW_COLUMNS = [
    "fold", "city", "model_seed", "replicate_id", "p", "mask_seed",
    "n_total_pairs", "n_revealed", "n_unseen", "fraction_pairs_revealed",
    "total_trip_mass", "revealed_trip_mass", "fraction_trip_mass_revealed",
    "unseen_trip_mass", "fraction_unseen_trip_mass",
    "empirical_tv_partial_vs_full", "js_partial_vs_full",
    "cpc_m0_unseen", "cpc_full_yd_unseen", "cpc_partial_od_unseen",
    "gain_full_yd", "gain_partial_od", "difference_partial_minus_yd",
    "relative_gain_vs_yd", "K", "q"
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


import multiprocessing as mp


def _process_city_replicates_chunk(args_tuple: Tuple) -> List[Tuple]:
    (fold_id, city_name, rep_ids, n_pairs, model_seeds, p_grid, city_cached) = args_tuple
    
    t_true_support = city_cached["t_true_support"]
    bin_idx_support = city_cached["bin_idx_support"]
    total_trip_mass = city_cached["total_trip_mass"]
    yd_full = city_cached["yd_full"]
    t0_by_seed = city_cached["t0_by_seed"]
    t_full_by_seed = city_cached["t_full_by_seed"]
    Y_hat_by_seed = city_cached["Y_hat_by_seed"]
    active_by_seed = city_cached["active_by_seed"]
    
    chunk_rows = []
    
    for rep_id in rep_ids:
        mask_seed = get_stable_mask_seed(PARTIAL_OD_BASE_SEED, fold_id, city_name, rep_id)
        rng = np.random.RandomState(mask_seed)
        perm = rng.permutation(n_pairs)
        
        t_true_perm = t_true_support[perm]
        bin_idx_perm = bin_idx_support[perm]
        t0_perm = {s: t0_by_seed[s][perm] for s in model_seeds}
        t_full_perm = {s: t_full_by_seed[s][perm] for s in model_seeds}
        
        running_counts_k = np.zeros(8, dtype=np.float64)
        running_revealed_mass = 0.0
        prev_n_reveal = 0
        
        for p_val in p_grid:
            n_reveal = int(np.round(p_val * n_pairs))
            n_unseen = n_pairs - n_reveal
            if n_unseen == 0:
                continue
                
            if n_reveal == 0:
                yd_partial = None
                revealed_mass = 0.0
                tv_partial = np.nan
                js_partial = np.nan
            else:
                if n_reveal > prev_n_reveal:
                    delta_trips = t_true_perm[prev_n_reveal:n_reveal]
                    delta_bins = bin_idx_perm[prev_n_reveal:n_reveal]
                    running_revealed_mass += float(np.sum(delta_trips))
                    running_counts_k += np.bincount(delta_bins, weights=delta_trips, minlength=8)
                    prev_n_reveal = n_reveal
                
                revealed_mass = running_revealed_mass
                if revealed_mass > 0:
                    yd_partial = running_counts_k / revealed_mass
                    tv_partial = float(0.5 * np.sum(np.abs(yd_partial - yd_full)))
                    
                    # Exact Jensen-Shannon Divergence
                    m_dist = 0.5 * (yd_partial + yd_full)
                    mask_p = (yd_partial > 1e-15) & (m_dist > 1e-15)
                    mask_q = (yd_full > 1e-15) & (m_dist > 1e-15)
                    kl_p = np.sum(yd_partial[mask_p] * np.log(yd_partial[mask_p] / m_dist[mask_p]))
                    kl_q = np.sum(yd_full[mask_q] * np.log(yd_full[mask_q] / m_dist[mask_q]))
                    js_partial = float(np.sqrt(max(0.0, 0.5 * (kl_p + kl_q))))
                else:
                    yd_partial = None
                    tv_partial = np.nan
                    js_partial = np.nan
                    
            frac_pairs_rev = float(n_reveal) / float(n_pairs)
            frac_mass_rev = float(revealed_mass) / float(total_trip_mass) if total_trip_mass > 0 else 0.0
            unseen_mass = total_trip_mass - revealed_mass
            frac_unseen_mass = unseen_mass / total_trip_mass if total_trip_mass > 0 else 0.0
            
            t_true_u = t_true_perm[n_reveal:]
            sum_true_unseen = unseen_mass
            bin_idx_unseen = bin_idx_perm[n_reveal:]
            
            for s in model_seeds:
                t0_u = t0_perm[s][n_reveal:]
                t_full_u = t_full_perm[s][n_reveal:]
                
                sum_t0_u = float(np.sum(t0_u))
                denom_m0 = sum_true_unseen + sum_t0_u
                cpc_m0_unseen = (2.0 * np.sum(np.minimum(t_true_u, t0_u)) / denom_m0) if denom_m0 > 0 else 0.0
                
                sum_full_u = float(np.sum(t_full_u))
                denom_full = sum_true_unseen + sum_full_u
                cpc_full_unseen = (2.0 * np.sum(np.minimum(t_true_u, t_full_u)) / denom_full) if denom_full > 0 else 0.0
                
                if yd_partial is None:
                    cpc_part_unseen = cpc_m0_unseen
                else:
                    Y_hat = Y_hat_by_seed[s]
                    active = active_by_seed[s]
                    
                    yd_act = yd_partial * active.astype(np.float64)
                    act_sum = yd_act.sum()
                    Y_D_cond = yd_act / act_sum if act_sum > 0 else Y_hat.copy()
                    
                    w = np.ones(8, dtype=np.float64)
                    for k in range(8):
                        if active[k] and Y_hat[k] > 0:
                            w[k] = Y_D_cond[k] / Y_hat[k]
                    weighted_mass = float(np.dot(Y_hat, w))
                    s_mult = w / weighted_mass if weighted_mass > 0 else np.ones(8)
                    
                    t_part_u = t0_u * s_mult[bin_idx_unseen]
                    sum_part_u = float(np.sum(t_part_u))
                    denom_part = sum_true_unseen + sum_part_u
                    cpc_part_unseen = (2.0 * np.sum(np.minimum(t_true_u, t_part_u)) / denom_part) if denom_part > 0 else 0.0
                    
                gain_full = float(cpc_full_unseen - cpc_m0_unseen)
                gain_part = float(cpc_part_unseen - cpc_m0_unseen)
                diff_part_minus_yd = float(gain_part - gain_full)
                rel_gain = float(gain_part / gain_full) if abs(gain_full) > 1e-8 else 1.0
                
                chunk_rows.append((
                    fold_id, city_name, s, rep_id, p_val, mask_seed,
                    n_pairs, n_reveal, n_unseen, frac_pairs_rev,
                    total_trip_mass, revealed_mass, frac_mass_rev,
                    unseen_mass, frac_unseen_mass,
                    tv_partial, js_partial,
                    cpc_m0_unseen, cpc_full_unseen, cpc_part_unseen,
                    gain_full, gain_part, diff_part_minus_yd,
                    rel_gain, 8, 1.0
                ))
                
    return chunk_rows


def run_fold_partial_od(
    fold_id: int,
    data_root: str = "data",
    output_dir: Path = Path("results/partial_od_equivalence_v2"),
    replicates: int = 500,
    p_grid: List[float] = None,
    smoke: bool = False,
    smoke_cities: int = 1,
    resume: bool = False,
    num_workers: int = 8,
    device: str = "cpu"
) -> Dict[str, Any]:
    if p_grid is None:
        p_grid = PRIMARY_GRID_V2.copy()

    fold_dir = output_dir / f"fold_{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    raw_csv_path = fold_dir / "raw.csv"
    progress_json_path = fold_dir / "progress.json"
    marker_path = fold_dir / "completion.marker"

    splits = generate_35_5_10_splits(data_root=data_root)
    split = splits[fold_id]
    train_cities = split["train"]
    test_cities = split["test"] if not smoke else split["test"][:smoke_cities]
    model_seeds = [1, 10, 100] if not smoke else [1, 10]
    B = replicates if not smoke else 20
    manifest_path = Path("results/e1/splits_manifest_v2.json")
    split_manifest_sha256 = _sha256_file(manifest_path)

    print(f"\n>>> [STARTING FOLD {fold_id}/5] {len(test_cities)} test cities | B={B} reps | {len(p_grid)} p-levels | Seeds: {model_seeds} | Workers={num_workers}")

    checkpoint_sha256 = _checkpoint_hashes(fold_id, model_seeds)
    expected_signature = {
        "fold_id": fold_id,
        "model_seeds": model_seeds,
        "B": B,
        "p_grid": [float(p) for p in p_grid],
        "n_p_levels": len(p_grid),
        "split_manifest_sha256": split_manifest_sha256,
        "checkpoint_sha256": checkpoint_sha256,
    }

    # Check already completed cities if resume is True with protocol signature verification
    completed_cities = set()
    if resume and progress_json_path.exists():
        try:
            with open(progress_json_path, "r", encoding="utf-8") as f:
                prog = json.load(f)
                sig = prog.get("protocol_signature", {})
                if prog.get("protocol_version") != "v2" or sig != expected_signature:
                    raise RuntimeError(
                        f"Resume protocol mismatch in {progress_json_path}; use a fresh output directory."
                    )
                completed_cities = set(prog.get("completed_cities", []))
                print(f"    [RESUME VERIFIED] Resuming fold {fold_id}: Found {len(completed_cities)} verified completed cities.")
        except Exception as e:
            if isinstance(e, RuntimeError):
                raise
            raise RuntimeError(f"Cannot safely resume from {progress_json_path}: {e}") from e

    if resume and not progress_json_path.exists() and raw_csv_path.exists():
        raise RuntimeError(
            f"Resume state is incomplete: {raw_csv_path} exists without progress metadata; use a fresh output directory."
        )

    # If raw.csv doesn't exist or not resuming, initialize with header
    if not resume or not raw_csv_path.exists():
        with open(raw_csv_path, "w", encoding="utf-8") as f:
            f.write(",".join(RAW_COLUMNS) + "\n")

    # Load frozen GNN models for this fold
    models: Dict[int, Tuple[Any, Any]] = {}
    for s in model_seeds:
        ckpt_path = Path("results/checkpoints") / f"5fold_fold{fold_id}_seed{s}.pt"
        if not ckpt_path.exists():
            raise RuntimeError(f"Required checkpoint missing for fold {fold_id} seed {s}: {ckpt_path}")
        model, scaler, _ = load_checkpoint(ckpt_path, device_str=device)
        model.eval()
        models[s] = (model, scaler)

    # Compute K=8 bin edges from 35 train cities
    bin_edges, K_act = compute_kbin_edges(train_cities, K=8, data_root=data_root)
    if K_act != 8 or len(bin_edges) != 9:
        raise RuntimeError(f"Strict 8-bin invariant failed for fold {fold_id}: K_act={K_act}")

    fold_start_time = time.perf_counter()
    rows_written_total = 0

    for city_idx, city_name in enumerate(test_cities):
        if city_name in completed_cities:
            print(f"  [{city_idx+1}/{len(test_cities)}] {city_name:<16} | ALREADY COMPLETED (Skipping)")
            continue

        city_start = time.perf_counter()
        raw_data = load_raw_city(city_name, data_root=data_root)
        dist_km = raw_data.dist_km
        
        # Support Omega_c^+: strictly positive interzonal pairs
        inter_pos = (raw_data.pair_o_idx.numpy() != raw_data.pair_d_idx.numpy()) & (dist_km > 0.0) & (raw_data.pair_trips.numpy() > 0)
        n_pairs = int(inter_pos.sum())
        if n_pairs == 0:
            raise RuntimeError(f"Critical error: City {city_name} has 0 positive interzonal pairs!")

        t_true_support = raw_data.pair_trips.numpy()[inter_pos].astype(np.float64)
        dist_support = dist_km[inter_pos]
        bin_idx_support = np.clip(np.digitize(dist_support, bin_edges, right=True) - 1, 0, 7)
        total_trip_mass = float(np.sum(t_true_support))
        
        # Extract clean full Y_D on support
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
            
            # Precompute full Y_D calibrated predictions
            Y_hat = np.bincount(bin_idx_support, weights=t0_support, minlength=8).astype(np.float64)
            Y_hat /= N_hat_support
            
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
                "Y_hat": Y_hat,
                "active": active,
                "t_cal_full": t_cal_full_support
            }

        city_cached_data = {
            "t_true_support": t_true_support,
            "bin_idx_support": bin_idx_support,
            "total_trip_mass": total_trip_mass,
            "yd_full": yd_full,
            "t0_by_seed": {s: seed_predictions[s]["t0"] for s in model_seeds},
            "t_full_by_seed": {s: seed_predictions[s]["t_cal_full"] for s in model_seeds},
            "Y_hat_by_seed": {s: seed_predictions[s]["Y_hat"] for s in model_seeds},
            "active_by_seed": {s: seed_predictions[s]["active"] for s in model_seeds},
        }

        # Divide B replicates into chunks for multiprocessing
        n_chunks = max(1, min(num_workers, B))
        rep_chunks = np.array_split(np.arange(B), n_chunks)
        task_args = [
            (fold_id, city_name, chunk.tolist(), n_pairs, model_seeds, p_grid, city_cached_data)
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
                "protocol_version": "v2",
                "protocol_signature": {
                    **expected_signature,
                },
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }, f, indent=2)


        city_elapsed = time.perf_counter() - city_start
        global_city_idx = (fold_id - 1) * 10 + (city_idx + 1)
        total_cities_count = 50 if not smoke else len(test_cities) * 5
        pct = (global_city_idx / total_cities_count) * 100.0
        timestamp_str = time.strftime("%H:%M:%S")
        speed_str = f"{len(city_rows) / max(city_elapsed, 1e-4):.0f} rows/s"
        print(f"  [{timestamp_str}] [Fold {fold_id}/5 | City {city_idx+1:>2}/{len(test_cities)} | Total {global_city_idx:>2}/{total_cities_count} ({pct:>5.1f}%)] {city_name:<16} | Pairs: {n_pairs:>5} | Mass: {total_trip_mass:>9.1f} | Done in {city_elapsed:>5.2f}s ({len(city_rows):>5} rows | {speed_str})", flush=True)

    # Read back raw.csv to generate per_seed, per_city, and fold_summary
    fold_df = pd.read_csv(raw_csv_path)
    
    # 1. Per-Seed Aggregation: Mean over B replicates -> (fold x city x model_seed x p)
    per_seed_df = fold_df.groupby(["fold", "city", "model_seed", "p"]).agg({
        "fraction_pairs_revealed": "mean",
        "fraction_trip_mass_revealed": "mean",
        "fraction_unseen_trip_mass": "mean",
        "empirical_tv_partial_vs_full": "mean",
        "js_partial_vs_full": "mean",
        "cpc_m0_unseen": "mean",
        "cpc_full_yd_unseen": "mean",
        "cpc_partial_od_unseen": "mean",
        "gain_full_yd": "mean",
        "gain_partial_od": "mean",
        "difference_partial_minus_yd": "mean",
        "relative_gain_vs_yd": "mean"
    }).reset_index()
    per_seed_csv_path = fold_dir / "per_seed.csv"
    per_seed_df.to_csv(per_seed_csv_path, index=False)

    # 2. Per-City Aggregation: Mean over 3 model seeds -> (fold x city x p)
    per_city_df = per_seed_df.groupby(["fold", "city", "p"]).agg({
        "fraction_pairs_revealed": "mean",
        "fraction_trip_mass_revealed": "mean",
        "fraction_unseen_trip_mass": "mean",
        "empirical_tv_partial_vs_full": "mean",
        "js_partial_vs_full": "mean",
        "cpc_m0_unseen": "mean",
        "cpc_full_yd_unseen": "mean",
        "cpc_partial_od_unseen": "mean",
        "gain_full_yd": "mean",
        "gain_partial_od": "mean",
        "difference_partial_minus_yd": "mean",
        "relative_gain_vs_yd": "mean"
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
            "mean_gain_full_yd": float(sub["gain_full_yd"].mean()),
            "mean_gain_partial_od": float(sub["gain_partial_od"].mean()),
            "mean_diff_vs_yd": float(sub["difference_partial_minus_yd"].mean()),
            "mean_tv": float(sub["empirical_tv_partial_vs_full"].mean()),
            "pos_cities": int((sub["gain_partial_od"] > 0).sum()),
            "match_yd_cities": int((sub["difference_partial_minus_yd"] >= 0).sum())
        })

    fold_summary_json_path = fold_dir / "fold_summary.json"
    with open(fold_summary_json_path, "w", encoding="utf-8") as f:
        json.dump({"fold": fold_id, "summary_by_p": fold_summary_rows}, f, indent=2)

    fold_summary_md_path = fold_dir / "fold_summary.md"
    with open(fold_summary_md_path, "w", encoding="utf-8") as f:
        f.write(f"# Fold {fold_id} Partial-OD Summary Table (N={len(test_cities)} Cities)\n\n")
        f.write("| p | Mean Gain Full $Y_D$ | Mean Gain Partial OD | Mean $D(p)$ (Part - Full) | Mean TV | Positive Cities | Match Full $Y_D$ |\n")
        f.write("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for r in fold_summary_rows:
            f.write(f"| **{r['p']*100:.2f}%** | +{r['mean_gain_full_yd']:.5f} | {r['mean_gain_partial_od']:+.5f} | {r['mean_diff_vs_yd']:+.5f} | {r['mean_tv']*100:.2f}% | {r['pos_cities']}/{r['n_cities']} | {r['match_yd_cities']}/{r['n_cities']} |\n")

    # 4. Save Run Manifest
    manifest_path = fold_dir / "run_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "fold": fold_id,
            "protocol_version": "v2",
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
    assert actual_raw_rows == expected_raw_rows, (
        f"Fold {fold_id} raw rows {actual_raw_rows} != expected {expected_raw_rows}"
    )
    assert len(per_city_df) == len(test_cities) * len(p_grid), f"Fold {fold_id} per_city rows mismatch"
    
    # Non-null assertions:
    # By contract §15, empirical_tv_partial_vs_full and js_partial_vs_full are NaN at p=0 (undefined discrepancy)
    non_tv_cols = [c for c in fold_df.columns if c not in ["empirical_tv_partial_vs_full", "js_partial_vs_full"]]
    assert not fold_df[non_tv_cols].isnull().any().any(), f"Fold {fold_id} contains unexpected NaN values in required fields!"
    assert not fold_df[fold_df["p"] > 0]["empirical_tv_partial_vs_full"].isnull().any(), f"Fold {fold_id} contains NaN TV for p > 0!"

    with open(marker_path, "w", encoding="utf-8") as f:
        f.write(f"FOLD {fold_id} EXECUTION COMPLETE -- LOCAL QA PASS\nTimestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    fold_total_time = time.perf_counter() - fold_start_time
    print(f">>> [FOLD {fold_id} COMPLETE] Local QA passed for {actual_raw_rows} rows in {fold_total_time:.2f}s | Marker: {marker_path.name}")
    
    return {
        "fold": fold_id,
        "raw_rows": actual_raw_rows,
        "per_seed_rows": len(per_seed_df),
        "per_city_rows": len(per_city_df),
        "status": "PASS"
    }


def aggregate_combined_results(
    output_dir: Path = Path("results/partial_od_equivalence_v2"),
    p_grid: List[float] = None
) -> None:
    if p_grid is None:
        p_grid = PRIMARY_GRID_V2.copy()

    combined_dir = output_dir / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    (combined_dir / "figures").mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 85)
    print("MASTER AGGREGATION & SCIENTIFIC SUMMARY (COMBINING ALL 5 FOLDS, N=50 CITIES)")
    print("=" * 85)

    # Check that all 5 folds have completion.marker
    all_raw_dfs = []
    all_per_seed_dfs = []
    all_per_city_dfs = []

    for f in range(1, 6):
        fold_dir = output_dir / f"fold_{f}"
        marker = fold_dir / "completion.marker"
        manifest_path = fold_dir / "run_manifest.json"
        
        if not marker.exists() or not manifest_path.exists():
            raise RuntimeError(f"Cannot aggregate: Fold {f} completion.marker or run_manifest.json not found")
            
        with open(manifest_path, "r") as mf:
            manifest = json.load(mf)
            
        assert manifest.get("protocol_version") == "v2", f"Fold {f} protocol version mismatch"
        assert manifest.get("model_seeds") == [1, 10, 100], f"Fold {f} model seeds mismatch (not [1, 10, 100])"
        assert manifest.get("replicates") == 500, f"Fold {f} replicates != 500"
        
        from src.data.city_splits import generate_35_5_10_splits
        splits = generate_35_5_10_splits()
        locked_test_cities = splits[f]["test"]
        assert manifest.get("cities") == locked_test_cities, f"Fold {f} test cities mismatch with locked manifest"
        
        all_raw_dfs.append(pd.read_csv(fold_dir / "raw.csv"))
        all_per_seed_dfs.append(pd.read_csv(fold_dir / "per_seed.csv"))
        all_per_city_dfs.append(pd.read_csv(fold_dir / "per_city.csv"))

    raw_combined = pd.concat(all_raw_dfs, ignore_index=True)
    per_seed_combined = pd.concat(all_per_seed_dfs, ignore_index=True)
    per_city_combined = pd.concat(all_per_city_dfs, ignore_index=True)

    raw_combined.to_csv(combined_dir / "raw_all_folds.csv", index=False)
    per_seed_combined.to_csv(combined_dir / "per_seed_all_folds.csv", index=False)
    per_city_combined.to_csv(combined_dir / "per_city_all_folds.csv", index=False)

    expected_raw_rows = 50 * 3 * 500 * 15  # 15 p-levels
    expected_seed_rows = 50 * 3 * 15
    expected_city_rows = 50 * 15
    
    assert len(raw_combined) == expected_raw_rows, f"Combined raw rows mismatch: {len(raw_combined)} != {expected_raw_rows}"
    assert len(per_seed_combined) == expected_seed_rows, f"Combined seed rows mismatch"
    assert len(per_city_combined) == expected_city_rows, f"Combined city rows mismatch"

    print(f"Combined Raw Rows:      {len(raw_combined):>10} (Certified)")
    print(f"Combined Per-Seed Rows: {len(per_seed_combined):>10} (Certified)")
    print(f"Combined Per-City Rows: {len(per_city_combined):>10} (Certified)")

    # Statistical Analysis across N=50 cities
    summary_rows = []
    raw_p_values = []
    p_vals_tested = [p for p in p_grid if p > 0]

    # Precalculate raw Wilcoxon p-values for partial OD benefit vs M0
    for p_val in p_vals_tested:
        sub = per_city_combined[per_city_combined.p == p_val]
        gains = sub["gain_partial_od"].values
        _, p_w = stats.wilcoxon(gains, alternative="greater")
        raw_p_values.append(p_w)

    holm_p_vals = holm_correction(raw_p_values)
    holm_dict = {p: h_p for p, h_p in zip(p_vals_tested, holm_p_vals)}

    for p_val in p_grid:
        sub = per_city_combined[per_city_combined.p == p_val]
        n_cities = len(sub)
        
        mean_mass = float(sub["fraction_trip_mass_revealed"].mean())
        mean_unseen_mass = float(sub["fraction_unseen_trip_mass"].mean())
        mean_tv = float(sub["empirical_tv_partial_vs_full"].mean())
        mean_m0 = float(sub["cpc_m0_unseen"].mean())
        mean_gain_full = float(sub["gain_full_yd"].mean())
        mean_gain_part = float(sub["gain_partial_od"].mean())
        mean_diff = float(sub["difference_partial_minus_yd"].mean())
        
        pos_cities = int((sub["gain_partial_od"] > 0).sum())
        match_yd_cities = int((sub["difference_partial_minus_yd"] >= 0).sum())
        
        ci_diff_l, ci_diff_h = fold_stratified_bootstrap(per_city_combined, "difference_partial_minus_yd", p_val)
        ci_part_l, ci_part_h = fold_stratified_bootstrap(per_city_combined, "gain_partial_od", p_val)
        ci_full_l, ci_full_h = fold_stratified_bootstrap(per_city_combined, "gain_full_yd", p_val)
        
        h_pval = holm_dict.get(p_val, 1.0) if p_val > 0 else 1.0

        summary_rows.append({
            "p": p_val,
            "n_cities": n_cities,
            "mean_revealed_mass": mean_mass,
            "mean_unseen_mass": mean_unseen_mass,
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

    # Calculate 3 Key Thresholds
    # 1. Positive Mean Crossing
    p_pos_mean = None
    for r in summary_rows:
        if r["mean_gain_partial_od"] > 0 and p_pos_mean is None:
            p_pos_mean = r["p"]

    # 2. Statistically Supported Benefit Threshold p*_benefit (Holm p < 0.05, CI_lower > 0)
    p_star_benefit = None
    for r in summary_rows:
        if r["holm_pval_benefit"] < 0.05 and r["ci_95_gain_partial"][0] > 0 and p_star_benefit is None:
            p_star_benefit = r["p"]

    # 3. Operational Equivalence Crossing p_eq
    # NOTE (paper framing): p_eq is the MEAN-CROSSING CRITERION where D(p) = Gain_OD(p) - Gain_YD(p) >= 0.
    # This is NOT a formal statistical equivalence test (TOST) with pre-specified margin delta.
    # Report in paper as "operational equivalence point" or "operational equivalence crossing",
    # NOT as "the two information sources were statistically equivalent."
    # If TOST-style equivalence testing is desired in future work, add equivalence margin
    # delta and compute TOST p-value separately.
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
            "experiment": "partial_od_information_equivalence",
            "protocol_version": "v2",
            "n_evaluation_cities": 50,
            "p_pos_mean_crossing": p_pos_mean,
            "p_star_benefit_threshold": p_star_benefit,
            "p_eq_grid": p_eq_grid,
            "p_eq_interp": p_eq_interp,
            "results_by_p": summary_rows
        }, f, indent=2)

    # Save Markdown Table
    summary_md_path = combined_dir / "summary.md"
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("# Table: Master Partial-OD Information Equivalence Summary (v2)\n\n")
        f.write("> **Evaluation Scope**: Assesses the operational reconstruction value of target-city distance distribution $Y_D$ relative to observing $p\\%$ of positive interzonal OD pairs ($K=8, q=1.0$, seeds $s \\in \\{1, 10, 100\\}$) evaluated strictly on unseen pairs ($N=50$ held-out test cities across 5 folds).\n\n")
        
        if p_pos_mean is not None:
            pct_pos = p_pos_mean * 100.0
            f.write(f"• **Positive Mean Crossing Point:** `{pct_pos:.2f}%` of positive interzonal OD pairs  \n")
        if p_star_benefit is not None:
            pct_star = p_star_benefit * 100.0
            f.write(f"• **Statistically Supported Benefit Threshold ($p^*_\\text{{benefit}}$):** `{pct_star:.2f}%` of positive interzonal OD pairs ($p_\\text{{Holm}} < 0.05$)  \n")
        if p_eq_interp is not None:
            pct_interp = p_eq_interp * 100.0
            f.write(f"• **Operational Equivalence Crossing ($p_\\text{{eq,interp}}$):** `{pct_interp:.2f}%` of positive interzonal OD pairs  \n\n")
        elif p_eq_grid is not None:
            pct_grid = p_eq_grid * 100.0
            f.write(f"• **Operational Equivalence Grid Point ($p_\\text{{eq,grid}}$):** `{pct_grid:.2f}%` of positive interzonal OD pairs  \n\n")
        else:
            f.write("• **Operational Equivalence Crossing:** Full target-city $Y_D$ was not matched within the prespecified partial-OD range up to 90% of the positive interzonal OD support.  \n\n")

        f.write("| Revealed OD Pairs ($p$) | Mean Revealed Trip Mass | Mean TV to Full $Y_D$ | $M_0$ CPC (Unseen) | Full-$Y_D$ Gain | Partial-OD Gain | Difference vs Full $Y_D$ ($D(p)$) | 95% CI Difference | Partial Benefit Holm $p$ | Cities Partial $> M_0$ | Cities Partial $\\ge$ Full $Y_D$ |\n")
        f.write("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        
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
            match_str = f"{r['match_yd_cities']}/{r['n_cities']}"
            
            f.write(f"| **{p_pct}** | {mass_pct} | {tv_pct} | {m0_str} | {full_str} | **{part_str}** | **{diff_str}** | {ci_str} | {h_str} | {pos_str} | {match_str} |\n")
            
        f.write("\n---\n\n### Prescribed Scientific Interpretation\n")
        f.write("Under uniform random pair sampling, the mean revealed trip-mass fraction closely tracked the revealed pair fraction. ")
        if p_eq_interp is not None:
            f.write(f"Under the frozen support-conditioned model and the same production calibration operator, the mean reconstruction benefit provided by the full target-city $Y_D$ was matched at approximately **{p_eq_interp*100:.2f}%** of directly observed positive interzonal OD pairs.\n")
        else:
            f.write("Under the tested operator, directly observing up to 90% of the positive interzonal OD support did not fully match the mean reconstruction gain provided by the full target-city $Y_D$.\n")

    print(f"Summary Markdown: {summary_md_path}")
    print(f"Summary JSON:     {summary_json_path}")

    # Generate 5 Publication Figures
    generate_publication_figures(summary_df, per_city_combined, combined_dir, p_eq_interp, p_star_benefit)

    # Write execution completion markers with explicit verification semantics
    with open(combined_dir / "EXECUTION_COMPLETE.marker", "w", encoding="utf-8") as f:
        f.write(f"MASTER 5-FOLD AGGREGATION EXECUTION COMPLETE\nTimestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\nStatus: EXECUTION_COMPLETE\nCertification: PENDING_CONTRACT_VERIFICATION\n")

    exec_marker_path = output_dir / "EXECUTION_COMPLETE.marker"
    with open(exec_marker_path, "w", encoding="utf-8") as f:
        f.write("PARTIAL-OD INFORMATION EQUIVALENCE v2 EXECUTION COMPLETE\n")
        f.write(f"Completed At: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("Status: EXECUTION_COMPLETE\n")
        f.write("Certification: PENDING_CONTRACT_VERIFICATION (Run tests/test_partial_od_equivalence_v2_contract.py to certify)\n")
        f.write("Protocol: 50 held-out test cities across 5 disjoint folds (N=50)\n")
        f.write("Evaluation Support: unseen positive interzonal pairs Omega_c^+ \\ S_p\n")
        f.write(f"Replicates: 500 per city (Total: 1,125,000 raw calibrations)\n")

    # Invalidate any prior certification; execution completion is separate from post-execution certification.
    (output_dir / "FROZEN.marker").unlink(missing_ok=True)
    with open(output_dir / "COMPLETED.marker", "w", encoding="utf-8") as f:
        f.write("PARTIAL-OD INFORMATION EQUIVALENCE v2 COMPUTATION COMPLETED\n")
        f.write(f"Completed At: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("Status: COMPLETED; CERTIFICATION_PENDING\n")
        f.write("Protocol: 50 held-out test cities across 5 disjoint folds (N=50)\n")
        f.write("Evaluation Support: unseen positive interzonal pairs Omega_c^+ \\ S_p\n")
        f.write(f"Replicates: 500 per city (Total: 1,125,000 raw calibrations)\n")



def generate_publication_figures(
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
    part_gain = summary_df["mean_gain_partial_od"].values
    part_ci_l = np.array([ci[0] for ci in summary_df["ci_95_gain_partial"]])
    part_ci_h = np.array([ci[1] for ci in summary_df["ci_95_gain_partial"]])
    
    ax.plot(p_vals, full_gain, label="Full $Y_D$ Reference Gain", color="#1f77b4", linestyle="--", linewidth=2.0)
    ax.plot(p_vals, part_gain, label="Partial-OD Calibration Gain", color="#d62728", marker="o", linewidth=2.0)
    ax.fill_between(p_vals, part_ci_l, part_ci_h, color="#d62728", alpha=0.15, label="95% Fold Bootstrap CI")
    
    if p_star_benefit is not None:
        ax.axvline(p_star_benefit * 100.0, color="#ff7f0e", linestyle="-.", label=f"Benefit $p^* = {p_star_benefit*100:.2f}\\%$")
    if p_eq_interp is not None:
        ax.axvline(p_eq_interp * 100.0, color="#2ca02c", linestyle=":", label=f"Equivalence $p_{{eq}} = {p_eq_interp*100:.2f}\\%$")
        
    ax.set_xlabel("Revealed Positive Interzonal OD Pairs (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean Marginal Gain $\\Delta\\mathrm{CPC}_U$ on Unseen OD", fontsize=11, fontweight="bold")
    ax.set_title("Marginal Reconstruction Value: Partial OD vs Full $Y_D$", fontsize=12, fontweight="bold", pad=12)
    ax.legend(frameon=True, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    fig.savefig(fig_dir / "fig_1_gain_vs_p.png")
    plt.close(fig)

    # Fig 2: Difference D(p) Equivalence
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
    fig.savefig(fig_dir / "fig_2_Dp_equivalence.png")
    plt.close(fig)

    # Fig 3: TV vs p
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    tvs = summary_df["mean_tv"].values * 100.0
    ax.plot(p_vals, tvs, color="#ff7f0e", marker="^", linewidth=2.0)
    ax.set_xlabel("Revealed Positive Interzonal OD Pairs (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean Total Variation Error $\\mathrm{TV}(\\tilde{Y}_D, Y_D^{\\mathrm{full}})$ (%)", fontsize=11, fontweight="bold")
    ax.set_title("Distributional Convergence with Partial OD Observation", fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    fig.savefig(fig_dir / "fig_3_TV_vs_p.png")
    plt.close(fig)

    # Fig 4: Revealed Mass vs Gain
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
    fig.savefig(fig_dir / "fig_4_revealed_mass_vs_gain.png")
    plt.close(fig)

    # Fig 5: Fold-Specific D(p) Auditing
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    ax.axhline(0, color="#333333", linestyle="-", linewidth=1.0)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    
    for f in range(1, 6):
        f_sub = per_city_df[per_city_df.fold == f].groupby("p")["difference_partial_minus_yd"].mean().reset_index()
        ax.plot(f_sub["p"].values * 100.0, f_sub["difference_partial_minus_yd"].values, marker="o", markersize=4, label=f"Fold {f} (N=10)", color=colors[f-1])
        
    ax.set_xlabel("Revealed Positive Interzonal OD Pairs (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Fold-Specific Mean $D(p)$", fontsize=11, fontweight="bold")
    ax.set_title("Fold-Specific Equivalence Trajectories", fontsize=12, fontweight="bold", pad=12)
    ax.legend(frameon=True, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    fig.savefig(fig_dir / "fig_5_fold_specific_Dp.png")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Partial-OD Information Equivalence v2")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="results/partial_od_equivalence_v2")
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5], help="Folds to execute")
    parser.add_argument("--cities", type=int, default=10, help="Number of test cities per fold")
    parser.add_argument("--b", type=int, default=500, help="Monte Carlo replicates per city")
    parser.add_argument("--smoke", action="store_true", help="Run fast smoke test")
    parser.add_argument("--resume", action="store_true", help="Resume from progress.json")
    parser.add_argument("--aggregate_only", action="store_true", help="Only aggregate completed folds")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel worker processes")
    args = parser.parse_args()

    out_p = Path(args.output_dir)

    if args.aggregate_only:
        aggregate_combined_results(output_dir=out_p)
    else:
        global_start = time.perf_counter()
        print("=" * 85)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] STARTING PARTIAL-OD EQUIVALENCE EXPERIMENT (V2)")
        print(f"  Folds: {args.folds} | Replicates B={args.b} | Workers={args.workers} | Device={args.device}")
        print("=" * 85, flush=True)

        for f_id in args.folds:
            run_fold_partial_od(
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
            aggregate_combined_results(output_dir=out_p)

        global_elapsed = time.perf_counter() - global_start
        print("=" * 85)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ALL EXPERIMENTS COMPLETED IN {global_elapsed:.2f}s")
        print("=" * 85, flush=True)
