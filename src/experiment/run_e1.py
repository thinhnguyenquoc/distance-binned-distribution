"""Legacy E1 experiment runner; canonical training uses run_5fold.py.

This module is retained for historical reproduction of the original E1 training loop.
All reusable statistical infrastructure (run_city, fold_bootstrap, compute_summary,
write_tables, etc.) has been extracted to `src.experiment.e1_core`.
This module imports and re-exports those functions for backward-compatibility.

E1: Oracle Aggregated-Distance Existence Test (Amended Protocol v2)
==================================================================

Research Question:
    Does target-city distance-binned aggregate information (Y_D^{GT,+}) provide
    marginal value for OD reconstruction beyond a zero-shot gravity-informed urban GNN?

Formal Protocol (Amended Replication under Locked Protocol):
  - 5-Fold Stratified City-Level Cross-Validation (35 Train / 5 Validation / 10 Held-out Test per fold).
  - Split Manifest: Pre-locked at results/e1/splits_manifest_v2.json:
      * Outer test sets locked 100% from E1-v1 (zero test perturbation).
      * Inner validation: 5-stratum size stratification across 40 non-test pool (seed 20260818).
      * Manifest SHA-256 integrity check and full candidate audit logging.
  - Model Selection: Early stopping on validation set interzonal CPC (patience=15, max_epochs=100, cosine annealing).
  - Quantile Discretization: K_move = 8 moving-distance bins (Bin 0 intrazonal excluded), computed
    strictly from training cities of each fold. Strict invariant: K_active == 8.
  - Calibration Operator: q = 1.0 within-tolerance closed-form multiplier scaling (numerical tolerance 1e-5).
  - 3 Conditions per test city:
      * Condition A: Zero-Shot Baseline (M0: Theta* frozen, no target information)
      * Condition B: Treatment (+ Oracle Target Y_D: K=8 aggregate histogram from target ground truth)
      * Condition C: Multi-Donor Placebo (+ Oracle Wrong Donor Y_D: Average over all 9 other test cities in fold)
  - Primary Specificity Estimand:
      * Delta_c^specificity = Delta_c^target - bar{Delta}_c^wrong
      * Statistical unit is strictly the city (n=50 full_5_fold, n=50 full coverage).
  - Evaluation Domain: Interzonal pairs Omega_c^+ = {(i,j) in Omega_c : i != j, D_ij > 0}.
  - Statistical Analysis:
      * Primary Full 5-fold: Prospectively designated untouched Folds 1-5 (n=50 cities).
      * Exploratory / Development: Fold 1 (n=10 cities).
      * Descriptive Coverage: All 50 out-of-fold cities.
      * 95% Fold-Stratified Bootstrap CI (10,000 resamples) + Paired Wilcoxon Signed-Rank Test.
      * Sample standard deviation with ddof=1 recorded across all metadata.

Artifacts Generated:
  - results/e1/splits_manifest_v2.json
  - results/e1/e1_per_city_results.json
  - results/e1/e1_validation_manifest.json
  - results/e1/e1_summary.json
  - results/e1/tables/e1_main_table.md
  - results/e1/tables/e1_per_city.md
"""

import json
import argparse
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import torch

# Ensure repository root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.city_splits import load_splits_manifest_v2
from src.data.dataset import load_city, preload_all_cities
from src.data.urban_graph import build_radius_graph
from src.data.yd_extractor import compute_kbin_edges

# ---------------------------------------------------------------------------
# Public API — imported from e1_core (single source of truth for all
# statistical infrastructure shared with run_e1_specificity_from_checkpoints.py)
# ---------------------------------------------------------------------------
from src.experiment.e1_core import (
    # Constants
    K_MOVE,
    Q_CALIB,
    TOLERANCE,
    # Runtime utilities
    get_runtime_metadata,
    configure_cpu_threads,
    log_msg,
    # Statistical helpers
    build_inter_mask,
    safe_wilcoxon,
    compute_iqr,
    # Core evaluation functions
    run_city,
    fold_bootstrap,
    compute_summary,
    write_tables,
)
from src.training.train import train_zero_shot_model

# ---------------------------------------------------------------------------
# Legacy-runner-specific constants (not part of e1_core)
# ---------------------------------------------------------------------------
EPOCHS      = 200        # Maximum training epochs per fold
PATIENCE    = 15         # Early stopping patience based on validation CPC
MIN_DELTA   = 1e-4       # Minimum validation CPC improvement threshold
DATA_ROOT   = "data"     # Dataset root folder
RESULTS_DIR = Path("results/e1")
LOG_FILE    = RESULTS_DIR / "e1_execution.log"
MANIFEST_PATH = RESULTS_DIR / "splits_manifest_v2.json"
SEED        = 1          # Legacy single-seed default; canonical multi-seed uses run_full_experiment.py

def run_e1(
    smoke: bool = False,
    smoke_cities: list = None,
    device_str: str = "cpu",
    num_threads: int | None = None,
    seed: int = SEED,
):
    """
    Main E1 execution loop with structured step-by-step logging.
    """
    t_global_start = time.time()
    device = torch.device(device_str)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Configure CPU Threads
    active_threads = configure_cpu_threads(num_threads)
    runtime_meta = get_runtime_metadata()

    # -----------------------------------------------------------------------
    # STEP 1: Load Locked 5-Fold Splits Manifest v2 (35 Train / 5 Val / 10 Test)
    # -----------------------------------------------------------------------
    log_msg("=" * 75)
    log_msg("E1 EXPERIMENT PIPELINE (v2): ORACLE AGGREGATED-DISTANCE EXISTENCE TEST")
    log_msg("=" * 75)
    log_msg("  Runtime Environment & CPU Configuration:")
    log_msg(f"    - Platform: {runtime_meta['platform']}")
    log_msg(f"    - CPU Cores: {runtime_meta['cpu_count_logical']} logical / {runtime_meta['cpu_count_physical']} physical")
    log_msg(f"    - PyTorch Threads: {runtime_meta['torch_num_threads']} (interop: {runtime_meta['torch_num_interop_threads']})")
    log_msg(f"    - OpenMP / MKL Threads: OMP={runtime_meta['omp_num_threads']}, MKL={runtime_meta['mkl_num_threads']}")
    log_msg(f"[STEP 1/5] Loading locked splits manifest v2 from {MANIFEST_PATH}...")
    splits = load_splits_manifest_v2(str(MANIFEST_PATH), data_root=DATA_ROOT)
    with open(MANIFEST_PATH, "r", encoding="utf-8") as manifest_file:
        split_manifest_sha256 = json.load(manifest_file)["manifest_sha256"]
    
    log_msg("  -> Preloading all city datasets & spatial graphs into global in-memory cache...")
    preload_all_cities(data_root=DATA_ROOT, build_graphs=True, radius_km=5.0)

    all_results = []
    fold_manifest = {}
    total_test_cities = 50 if not smoke else len(smoke_cities or ["Portland", "Denver"])
    city_counter = 0

    for fold_id, split in splits.items():
        t_fold_start = time.time()
        train35 = split["train"]   # 35 training cities
        val5    = split["val"]     # 5 validation cities
        test10  = sorted(split["test"])  # 10 test cities

        run_test = test10
        if smoke or smoke_cities:
            target_filter = smoke_cities if smoke_cities else ["Portland", "Denver"]
            run_test = [c for c in target_filter if c in test10]
            if not run_test:
                continue

        fold_role = "Exploratory / Development" if fold_id == 1 else "Full 5-fold Out-of-Fold"
        log_msg("-" * 75)
        log_msg(f">>> [FOLD {fold_id}/5] {fold_role} | Train: {len(train35)} cities | Val: {len(val5)} cities | Test: {len(run_test)}/{len(test10)} cities")
        log_msg("-" * 75)

        # -------------------------------------------------------------------
        # STEP 2: Compute Pair-Weighted Quantile Bin Edges from 35 Train Cities
        # -------------------------------------------------------------------
        log_msg(f"  [STEP 2/5: Fold {fold_id}] Computing K_move={K_MOVE} quantile bin edges from {len(train35)} train cities...")
        bin_edges, K_active = compute_kbin_edges(train35, K=K_MOVE, data_root=DATA_ROOT)
        
        # Enforce strict 8-bin invariant
        if K_active != K_MOVE:
            raise RuntimeError(f"E1 invariant violated: Expected exactly {K_MOVE} active bins, got {K_active}")
        log_msg(f"    -> Strict {K_MOVE}-bin verified. Cut points (km): {np.round(bin_edges[1:-1], 2).tolist()}")

        # -------------------------------------------------------------------
        # STEP 3: Train Zero-Shot Backbone & Select Best Validation Checkpoint
        # -------------------------------------------------------------------
        log_msg(f"  [STEP 3/5: Fold {fold_id}] Training backbone model (max_epochs={EPOCHS}, patience={PATIENCE}, min_delta={MIN_DELTA})...")
        _ckpt_seed = seed
        _ckpt_dir  = RESULTS_DIR / "checkpoints"
        _ckpt_path = _ckpt_dir / f"fold{fold_id}_seed{_ckpt_seed}.pt"
        model, scaler, train_info = train_zero_shot_model(
            train_city_names=train35,
            data_root=DATA_ROOT,
            epochs=EPOCHS,
            device_str=device_str,
            verbose=True,
            val_city_names=val5,
            patience=PATIENCE,
            min_delta=MIN_DELTA,
            return_info=True,
            seed=_ckpt_seed,
            checkpoint_path=_ckpt_path,
            run_tag=f"e1_fold{fold_id}_seed{_ckpt_seed}",
            fold=fold_id,
            split_manifest_sha256=split_manifest_sha256,
        )

        # Verify per-fold convergence gate
        is_converged = bool(train_info["stopped_early"] or train_info["best_epoch"] <= (EPOCHS - 5))
        conv_gate_status = "PASSED" if is_converged else "FAILED (Ceiling Hit)"

        fold_manifest[fold_id] = {
            "fold_id": fold_id,
            "role": fold_role,
            "train_cities": train35,
            "val_cities": val5,
            "test_cities": test10,
            "K_active": K_active,
            "K_mode": "strict_8bin",
            "bin_edges_km": bin_edges.tolist(),
            "best_epoch": train_info["best_epoch"],
            "best_val_cpc": train_info["best_val_cpc"],
            "epochs_trained": train_info["epochs_trained"],
            "stopped_early": train_info["stopped_early"],
            "convergence_gate": conv_gate_status,
            "val_cpc_history": train_info["val_cpc_history"],
            "checkpoint_path": str(_ckpt_path.resolve()),
            "checkpoint_seed": _ckpt_seed,
        }
        log_msg(f"    -> [Fold {fold_id}] Frozen at best epoch {train_info['best_epoch']}/{train_info['epochs_trained']} (Validation CPC = {train_info['best_val_cpc']:.4f}) | Gate: {conv_gate_status}.")
        log_msg(f"    -> [Fold {fold_id}] Checkpoint: {_ckpt_path.resolve()}")


        # -------------------------------------------------------------------
        # STEP 4: Evaluate Held-Out Test Cities (Conditions A, B, C across all 9 donors)
        # -------------------------------------------------------------------
        log_msg(f"  [STEP 4/5: Fold {fold_id}] Precomputing test city structures & oracle Y_D for {len(test10)} test cities...")
        test_city_cache = {}
        for t_city in test10:
            cd_t = load_city(t_city, data_root=DATA_ROOT, feature_scaler=scaler)
            ei_t, ed_t = build_radius_graph(cd_t.lon_lat, radius_km=5.0)
            dist_t = np.expm1(cd_t.pair_distance.numpy())
            inter_t = build_inter_mask(cd_t, dist_t)
            t_gt_t = cd_t.pair_trips.numpy().astype(np.float64)
            yd_t = extract_yd_kbins(dist_t, t_gt_t, bin_edges, inter_t)
            test_city_cache[t_city] = {
                "city_data": cd_t,
                "edge_index": ei_t,
                "edge_dist": ed_t,
                "dist_km": dist_t,
                "inter_mask": inter_t,
                "t_gt": t_gt_t,
                "Y_D": yd_t,
            }

        log_msg(f"  [STEP 4/5: Fold {fold_id}] Evaluating {len(run_test)} held-out test cities with 9-donor placebo...")
        for i_city, city in enumerate(run_test):
            city_counter += 1
            t_city_start = time.time()
            
            res = run_city(
                city=city,
                model=model,
                scaler=scaler,
                bin_edges=bin_edges,
                K_active=K_active,
                test_cities=test10,
                fold_id=fold_id,
                device=device,
                test_city_cache=test_city_cache,
            )
            all_results.append(res)
            t_city_elapsed = time.time() - t_city_start
            
            d_target = res['delta_cpc_target']
            d_wrong  = res['delta_cpc_wrong']
            d_spec   = res['delta_cpc_specificity']
            spec_marker = f"(dSpec={d_spec:+.4f})"
            log_msg(
                f"    [Fold {fold_id} | City {i_city+1:02d}/{len(run_test):02d} (Total {city_counter:02d}/{total_test_cities:02d})] "
                f"{city:<16} ({t_city_elapsed:.2f}s) | "
                f"M0={res['cpc_baseline']:.4f} -> +Target={res['cpc_target_yd']:.4f} (dCPC={d_target:+.4f}) | "
                f"+WrongAvg9={res['cpc_wrong_yd']:.4f} (dCPC={d_wrong:+.4f}) {spec_marker}"
            )

        t_fold_elapsed = time.time() - t_fold_start
        log_msg(f"  [Fold {fold_id} Complete] Elapsed time: {t_fold_elapsed:.1f}s")

    # -----------------------------------------------------------------------
    # STEP 5: Statistical Aggregation, Verification, and Artifact Output
    # -----------------------------------------------------------------------
    log_msg("-" * 75)
    log_msg("[STEP 5/5] Synthesizing summary, fold-stratified bootstrap CI, and writing artifacts...")
    
    val_manifest_payload = {
        "protocol_version": "e1-v2-amended",
        "runtime_environment": get_runtime_metadata(),
        "folds": fold_manifest,
    }
    (RESULTS_DIR / "e1_per_city_results.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    (RESULTS_DIR / "e1_validation_manifest.json").write_text(json.dumps(val_manifest_payload, indent=2), encoding="utf-8")

    if len(all_results) < 2:
        log_msg("Warning: Fewer than 2 cities evaluated; skipping statistical synthesis.")
        return all_results, None

    summary = compute_summary(all_results, fold_manifest=fold_manifest, bootstrap_seed=seed)
    (RESULTS_DIR / "e1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_tables(all_results, summary)

    t_global_elapsed = time.time() - t_global_start
    is_conf = summary.get("is_full_5_fold_complete", False)
    is_full = summary.get("is_full_50_complete", False)
    conf = summary.get("full_5_fold_folds_2_5", {})

    log_msg("=" * 75)
    log_msg(f"E1 EXECUTION SUMMARY (Total Elapsed Time: {t_global_elapsed:.1f}s)")
    log_msg("=" * 75)
    log_msg(f"  Cities Evaluated: {len(all_results)}/50 | Mode: {'Full 50-City Protocol' if is_full else 'Exploratory Subset'}")
    log_msg(f"  Target Effect (dCPC): mean = {summary['delta_cpc_target_mean']:+.4f} (median = {summary['delta_cpc_target_median']:+.4f}, IQR = {summary['delta_cpc_target_iqr']:.4f})")
    log_msg(f"  95% Fold-Stratified Bootstrap CI: [{summary['delta_cpc_target_ci_l']:+.4f}, {summary['delta_cpc_target_ci_h']:+.4f}]")
    log_msg(f"  Placebo (9 Wrong Donors Avg): mean = {summary['delta_cpc_wrong_mean']:+.4f} (median = {summary['delta_cpc_wrong_median']:+.4f}, IQR = {summary['delta_cpc_wrong_iqr']:.4f})")
    log_msg(f"  Specificity Effect (Target - Wrong_Avg9): mean = {summary['delta_specificity_mean']:+.4f} (median = {summary['delta_specificity_median']:+.4f}, IQR = {summary['delta_specificity_iqr']:.4f})")
    log_msg(f"  Specificity 95% Bootstrap CI: [{summary['delta_specificity_ci_l']:+.4f}, {summary['delta_specificity_ci_h']:+.4f}]")
    log_msg(f"  Specificity Win Rate: {summary['win_rate_specificity']} | Wilcoxon p = {summary['p_specificity']:.2e}")

    if is_conf and conf.get("status") == "full_5_fold_complete":
        pass_ci = "PASS" if conf['ci_lower_bound_positive'] else "FAIL"
        pass_sci = "PASS" if conf['specificity_ci_lower_bound_positive'] else "FAIL"
        pass_tw = "PASS" if conf['target_beats_wrong'] else "FAIL"
        log_msg("\n  CONFIRMATORY HYPOTHESIS TEST OUTCOMES (Folds 1-5, n=50):")
        log_msg(f"    * Target 95% CI Lower Bound > 0      : {pass_ci} ([{conf['delta_cpc_target_ci_l']:+.4f}, {conf['delta_cpc_target_ci_h']:+.4f}])")
        log_msg(f"    * Specificity 95% CI Lower Bound > 0 : {pass_sci} ([{conf['delta_specificity_ci_l']:+.4f}, {conf['delta_specificity_ci_h']:+.4f}])")
        log_msg(f"    * Specificity Gain (Target > Wrong)  : {pass_tw} (+{conf['delta_specificity_mean']:+.4f} vs 0)")
        log_msg(f"    * Specificity Win Rate               : {conf['win_rate_specificity']} (Wilcoxon p = {conf['p_specificity']:.2e})")
    else:
        conf_count = len([r for r in all_results if r['fold'] >= 2])
        log_msg(f"\n  CONFIRMATORY STATUS: NOT AVAILABLE (Observed {conf_count}/50 test cities; strictly requires complete 50 test cities across Folds 1-5, with 10 test cities per fold).")
    log_msg("=" * 75)

    return all_results, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E1 Oracle Aggregated-Distance Existence Test (Amended Protocol v2)")
    parser.add_argument("--smoke",  action="store_true", help="Run smoke test on Portland (Fold 4) and Denver (Fold 5)")
    parser.add_argument("--cities", nargs="+", default=None, help="Custom list of test cities to run")
    parser.add_argument("--device", default="cpu", help="PyTorch device (cpu/cuda)")
    parser.add_argument("--num-threads", "-t", type=int, default=None, help="Number of CPU intra-op threads for PyTorch/OpenMP")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for model training and bootstrap")
    args = parser.parse_args()
    run_e1(smoke=args.smoke, smoke_cities=args.cities, device_str=args.device, num_threads=args.num_threads, seed=args.seed)
