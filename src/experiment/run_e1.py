"""
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
      * Statistical unit is strictly the city (n=40 confirmatory, n=50 full coverage).
  - Evaluation Domain: Interzonal pairs Omega_c^+ = {(i,j) in Omega_c : i != j, D_ij > 0}.
  - Statistical Analysis:
      * Primary Confirmatory: Prospectively designated untouched Folds 2–5 (n=40 cities).
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
import os
import platform
import time
import argparse
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
from scipy import stats
import torch

# Ensure repository root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.city_splits import load_splits_manifest_v2, get_wrong_donors
from src.data.dataset import load_city, preload_all_cities
from src.data.urban_graph import build_radius_graph
from src.data.yd_extractor import compute_kbin_edges, extract_yd_kbins
from src.calibration.bin_calibration import calibrate_kbins
from src.training.train import train_zero_shot_model, infer_zero_shot
from src.training.evaluate import compute_cpc_pair, compute_cpc_norm_pair


def get_runtime_metadata() -> dict:
    """
    Collect hardware, OS, and PyTorch runtime execution metadata.
    Enables auditability of multi-core CPU and multi-threading configuration.
    """
    cpu_physical = None
    cpu_logical = os.cpu_count()
    try:
        import psutil
        cpu_physical = psutil.cpu_count(logical=False)
    except Exception:
        cpu_physical = None

    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cpu_count_logical": cpu_logical,
        "cpu_count_physical": cpu_physical,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "not_set"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS", "not_set"),
    }


def configure_cpu_threads(num_threads: int | None = None) -> int:
    """
    Explicitly configure PyTorch CPU intra-op threads and OpenMP/MKL environment variables.
    If num_threads is specified and > 0, sets torch.set_num_threads(num_threads)
    and updates OMP_NUM_THREADS and MKL_NUM_THREADS.
    Returns the active torch.get_num_threads().
    """
    if num_threads is not None and num_threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
        os.environ["MKL_NUM_THREADS"] = str(num_threads)
        torch.set_num_threads(num_threads)

    return torch.get_num_threads()

# ---------------------------------------------------------------------------
# Global Experiment Parameters (Pre-specified, locked before evaluation)
# ---------------------------------------------------------------------------
K_MOVE      = 8          # Number of moving-distance bins (Bin 0 intrazonal excluded)
Q_CALIB     = 1.0        # Calibration strength (1.0 = exact within-tolerance distribution match)
EPOCHS      = 200        # Maximum training epochs per fold (with ReduceLROnPlateau & Early Stopping)
PATIENCE    = 15         # Early stopping patience based on validation CPC
MIN_DELTA   = 1e-4       # Minimum validation CPC improvement threshold
DATA_ROOT   = "data"     # Dataset root folder containing 50 city directories
RESULTS_DIR = Path("results/e1")
LOG_FILE    = RESULTS_DIR / "e1_execution.log"
MANIFEST_PATH = RESULTS_DIR / "splits_manifest_v2.json"
TOLERANCE   = 1e-5       # Floating-point tolerance for mass preservation & bin matching
SEED        = 3000       # Fixed random seed (Version 3.0) for deterministic training initialization


def log_msg(msg: str = "", print_to_console: bool = True):
    """Logs message with local timestamp to console (flush=True) and e1_execution.log."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {msg}" if msg else ""
    if print_to_console:
        try:
            print(formatted if formatted else "", flush=True)
        except Exception:
            try:
                print(formatted.encode("ascii", errors="replace").decode("ascii") if formatted else "", flush=True)
            except Exception:
                pass
    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write((formatted if formatted else "") + "\n")
    except Exception:
        pass


def build_inter_mask(cd, dist_km: np.ndarray) -> np.ndarray:
    """
    Construct boolean mask for interzonal candidate support Omega_c^+.
    Strict definition: Origin != Destination and Pairwise Distance > 0.
    """
    o = cd.pair_o_idx.numpy()
    d = cd.pair_d_idx.numpy()
    return (o != d) & (dist_km > 0.0)


def safe_wilcoxon(diff: np.ndarray, alternative: str = "greater") -> tuple[float, float]:
    """
    Defensive non-parametric Wilcoxon signed-rank test.
    Handles edge cases:
      - n < 2 observations (returns p=1.0)
      - All zero differences / ties (returns p=1.0)
      - Non-zero filtering using wilcox zero-method
    """
    diff_clean = diff[~np.isnan(diff)]
    if len(diff_clean) < 2:
        return 0.0, 1.0
    non_zero = diff_clean[diff_clean != 0.0]
    if len(non_zero) == 0:
        return 0.0, 1.0
    try:
        res = stats.wilcoxon(diff_clean, alternative=alternative, zero_method="wilcox")
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return 0.0, 1.0


def compute_iqr(values: np.ndarray) -> float:
    """Computes sample Interquartile Range (IQR = Q3 - Q1)."""
    if len(values) == 0:
        return 0.0
    return float(np.percentile(values, 75) - np.percentile(values, 25))


def run_city(
    city: str,
    model: torch.nn.Module,
    scaler: object,
    bin_edges: np.ndarray,
    K_active: int,
    test_cities: list[str],
    fold_id: int,
    device: torch.device,
    test_city_cache: dict[str, dict] | None = None,
    test_yd_cache: dict[str, np.ndarray] | None = None,
) -> dict:
    """
    Evaluates 3 experimental conditions on a single held-out test city.

    Condition A (Zero-Shot Baseline M0):
        Input: (X_c, G_c^urban, D_c). Forward pass of frozen model Theta*.
        Output: T_c^(0) = E[T | T >= 1].

    Condition B (Treatment: Target Oracle Y_D):
        Input: T_c^(0) and Y_{D,c}^{GT,+} (K-dim aggregate distance histogram).
        Operator: Moving-bin closed-form scaling on Omega_c^+.
        Output: T_c^(YD).

    Condition C (Placebo Control: Multi-Donor Wrong Y_D):
        Input: T_c^(0) and Y_{D,d}^{GT,+} for all 9 other test cities in fold d != c.
        Operator: Moving-bin closed-form scaling for each donor d.
        Output: Delta_c^(wrong) = 1/9 sum_{d != c} Delta_{c,d}^(wrong).
    """
    # 1. Retrieve precomputed test city structures if available, or load on-demand
    if test_city_cache is not None and city in test_city_cache:
        c_entry = test_city_cache[city]
        cd = c_entry["city_data"]
        ei = c_entry["edge_index"]
        ed = c_entry["edge_dist"]
        dist_km = c_entry["dist_km"]
        inter = c_entry["inter_mask"]
        t_gt = c_entry.get("t_gt", cd.pair_trips.numpy().astype(np.float64))
        Y_D_tgt = c_entry.get("Y_D")
    else:
        cd = load_city(city, data_root=DATA_ROOT, feature_scaler=scaler)
        ei, ed = build_radius_graph(cd.lon_lat, radius_km=5.0)
        dist_km = np.expm1(cd.pair_distance.numpy())
        inter = build_inter_mask(cd, dist_km)
        t_gt = cd.pair_trips.numpy().astype(np.float64)
        if test_yd_cache is not None and city in test_yd_cache:
            Y_D_tgt = test_yd_cache[city]
        else:
            Y_D_tgt = extract_yd_kbins(dist_km, t_gt, bin_edges, inter)

    # 2. Condition A: Zero-Shot Forward Inference
    T0 = infer_zero_shot(model, cd, ei, ed, device=device)
    t0 = T0.numpy().astype(np.float64)

    # Compute interzonal mask and ground truth flows
    n_inter = int(inter.sum())

    cpc0      = compute_cpc_pair(t_gt[inter], t0[inter])
    cpc0_norm = compute_cpc_norm_pair(t_gt[inter], t0[inter])

    # 3. Condition B: Target Oracle Y_D Extraction & Calibration
    if Y_D_tgt is None:
        if test_yd_cache is not None and city in test_yd_cache:
            Y_D_tgt = test_yd_cache[city]
        else:
            Y_D_tgt = extract_yd_kbins(dist_km, t_gt, bin_edges, inter)

    T_yd    = calibrate_kbins(t0, dist_km, inter, Y_D_tgt, bin_edges, q=Q_CALIB, tolerance=TOLERANCE)
    cpc_yd      = compute_cpc_pair(t_gt[inter], T_yd[inter])
    cpc_yd_norm = compute_cpc_norm_pair(t_gt[inter], T_yd[inter])
    delta_target = float(cpc_yd - cpc0)

    # 4. Condition C: Multi-Donor Placebo Control (All 9 Wrong Donors in Test Fold)
    wrong_donors = get_wrong_donors(city, test_cities)
    assert len(wrong_donors) == len(test_cities) - 1, f"Expected {len(test_cities) - 1} wrong donors, got {len(wrong_donors)}"

    wrong_cpc_list = []
    wrong_cpc_norm_list = []
    wrong_delta_list = []
    wrong_donor_details = []

    for donor in wrong_donors:
        if test_city_cache is not None and donor in test_city_cache:
            Y_D_wr = test_city_cache[donor]["Y_D"]
        elif test_yd_cache is not None and donor in test_yd_cache:
            Y_D_wr = test_yd_cache[donor]
        else:
            cd_d    = load_city(donor, data_root=DATA_ROOT, feature_scaler=scaler)
            dist_d  = np.expm1(cd_d.pair_distance.numpy())
            inter_d = build_inter_mask(cd_d, dist_d)
            t_gt_d  = cd_d.pair_trips.numpy().astype(np.float64)
            Y_D_wr  = extract_yd_kbins(dist_d, t_gt_d, bin_edges, inter_d)

        T_wr    = calibrate_kbins(t0, dist_km, inter, Y_D_wr, bin_edges, q=Q_CALIB, tolerance=TOLERANCE)
        cpc_wr_d      = compute_cpc_pair(t_gt[inter], T_wr[inter])
        cpc_wr_norm_d = compute_cpc_norm_pair(t_gt[inter], T_wr[inter])
        delta_wr_d    = cpc_wr_d - cpc0

        wrong_cpc_list.append(cpc_wr_d)
        wrong_cpc_norm_list.append(cpc_wr_norm_d)
        wrong_delta_list.append(delta_wr_d)
        wrong_donor_details.append({
            "donor_city": donor,
            "cpc_wrong_yd": float(cpc_wr_d),
            "cpc_wrong_yd_norm": float(cpc_wr_norm_d),
            "delta_cpc_wrong": float(delta_wr_d),
            "Y_D_wrong": Y_D_wr.tolist(),
        })

    # Exact arithmetic mean across all wrong donors in fold
    cpc_wr_mean      = float(np.mean(wrong_cpc_list))
    cpc_wr_norm_mean = float(np.mean(wrong_cpc_norm_list))
    delta_wr_mean    = float(np.mean(wrong_delta_list))
    delta_spec       = float(delta_target - delta_wr_mean)

    return {
        "city": city,
        "fold": fold_id,
        "donor_city": "all_9_fold_donors",
        "n_wrong_donors": len(wrong_donors),
        "n_inter_pairs": n_inter,
        "K_active": K_active,
        # Baseline (M0)
        "cpc_baseline": float(cpc0),
        "cpc_baseline_norm": float(cpc0_norm),
        # Target Oracle (+Y_D^target)
        "cpc_target_yd": float(cpc_yd),
        "cpc_target_yd_norm": float(cpc_yd_norm),
        "delta_cpc_target": delta_target,
        # Wrong Donor Placebo (+Y_D^wrong, 9-donor average)
        "cpc_wrong_yd": cpc_wr_mean,
        "cpc_wrong_yd_norm": cpc_wr_norm_mean,
        "delta_cpc_wrong": delta_wr_mean,
        # Specificity Estimand (Target - Wrong_Avg9)
        "delta_cpc_specificity": delta_spec,
        # Raw histogram vectors & per-donor breakdown for transparency
        "Y_D_target": Y_D_tgt.tolist(),
        "wrong_donor_breakdown": wrong_donor_details,
    }


def fold_bootstrap(
    values: np.ndarray,
    fold_ids: np.ndarray,
    n: int = 10000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple:
    """
    Computes Fold-Stratified Bootstrap 95% Confidence Interval.
    Resamples observations within each fold independently to account for shared model covariance.
    """
    rng = np.random.default_rng(seed)
    folds = sorted(set(fold_ids))
    boot = []
    for _ in range(n):
        s = []
        for f in folds:
            fd = values[fold_ids == f]
            if len(fd) > 0:
                s.extend(rng.choice(fd, size=len(fd), replace=True))
        if s:
            boot.append(np.mean(s))
    boot = np.array(boot)
    if len(boot) == 0:
        return 0.0, 0.0, np.array([0.0])
    return (
        float(np.percentile(boot, 100 * alpha / 2)),
        float(np.percentile(boot, 100 * (1 - alpha / 2))),
        boot,
    )


def compute_summary(results: list, fold_manifest: dict = None, bootstrap_seed: int = 2024) -> dict:
    """
    Aggregates per-city results into primary statistics:
      - Full out-of-fold descriptive coverage (n=50)
      - Confirmatory test set (n=40, Folds 2-5, gated strictly upon complete execution)
      - Per-fold breakdown (Folds 1 to 5)
    """
    dt  = np.array([r["delta_cpc_target"]      for r in results])
    dw  = np.array([r["delta_cpc_wrong"]       for r in results])
    ds  = np.array([r["delta_cpc_specificity"] for r in results])
    fid = np.array([r["fold"]                  for r in results])
    c0  = np.array([r["cpc_baseline"]          for r in results])
    cyd = np.array([r["cpc_target_yd"]        for r in results])
    cwr = np.array([r["cpc_wrong_yd"]         for r in results])

    n = len(results)
    ddof = 1 if n > 1 else 0

    ci_tl, ci_th, _ = fold_bootstrap(dt, fid, seed=bootstrap_seed)
    ci_wl, ci_wh, _ = fold_bootstrap(dw, fid, seed=bootstrap_seed)
    ci_sl, ci_sh, _ = fold_bootstrap(ds, fid, seed=bootstrap_seed)

    _, pt = safe_wilcoxon(dt, alternative="greater")
    _, pw = safe_wilcoxon(dw, alternative="greater")
    _, ps = safe_wilcoxon(ds, alternative="greater")

    # --- Confirmatory Evaluation Guard (Requires strictly complete 40 cities across Folds 2-5) ---
    conf_mask = (fid >= 2)
    conf_fid = fid[conf_mask]
    is_confirmatory_complete = bool(
        len(conf_fid) == 40
        and set(conf_fid.tolist()) == {2, 3, 4, 5}
        and all((conf_fid == f).sum() == 10 for f in [2, 3, 4, 5])
    )
    is_full_50_complete = bool(
        n == 50
        and set(fid.tolist()) == {1, 2, 3, 4, 5}
        and all((fid == f).sum() == 10 for f in range(1, 6))
    )

    conf_summary = None
    if is_confirmatory_complete:
        c_dt = dt[conf_mask]
        c_dw = dw[conf_mask]
        c_ds = ds[conf_mask]
        c_c0 = c0[conf_mask]
        c_cyd = cyd[conf_mask]
        c_n = 40
        c_ddof = 1

        c_ci_tl, c_ci_th, _ = fold_bootstrap(c_dt, conf_fid, seed=bootstrap_seed)
        c_ci_wl, c_ci_wh, _ = fold_bootstrap(c_dw, conf_fid, seed=bootstrap_seed)
        c_ci_sl, c_ci_sh, _ = fold_bootstrap(c_ds, conf_fid, seed=bootstrap_seed)
        _, c_pt = safe_wilcoxon(c_dt, alternative="greater")
        _, c_pw = safe_wilcoxon(c_dw, alternative="greater")
        _, c_ps = safe_wilcoxon(c_ds, alternative="greater")

        conf_summary = {
            "status": "confirmatory_complete",
            "protocol_role": "Amended Replication under Locked Protocol (Folds 2-5, n=40)",
            "n_cities": c_n,
            "cpc_baseline_mean": float(c_c0.mean()),
            "cpc_baseline_std": float(c_c0.std(ddof=c_ddof)),
            "cpc_target_yd_mean": float(c_cyd.mean()),
            "cpc_target_yd_std": float(c_cyd.std(ddof=c_ddof)),
            # Target Delta
            "delta_cpc_target_mean": float(c_dt.mean()),
            "delta_cpc_target_median": float(np.median(c_dt)),
            "delta_cpc_target_iqr": compute_iqr(c_dt),
            "delta_cpc_target_std": float(c_dt.std(ddof=c_ddof)),
            "delta_cpc_target_ci_l": c_ci_tl,
            "delta_cpc_target_ci_h": c_ci_th,
            "n_positive_target": int((c_dt > 0).sum()),
            "p_wilcoxon_target": float(c_pt),
            # Wrong Placebo Delta (9-donor avg)
            "delta_cpc_wrong_mean": float(c_dw.mean()),
            "delta_cpc_wrong_median": float(np.median(c_dw)),
            "delta_cpc_wrong_iqr": compute_iqr(c_dw),
            "delta_cpc_wrong_std": float(c_dw.std(ddof=c_ddof)),
            "delta_cpc_wrong_ci_l": c_ci_wl,
            "delta_cpc_wrong_ci_h": c_ci_wh,
            "n_positive_wrong": int((c_dw > 0).sum()),
            "p_wilcoxon_wrong": float(c_pw),
            # Specificity Estimand (Target - Wrong_Avg9)
            "delta_specificity_mean": float(c_ds.mean()),
            "delta_specificity_median": float(np.median(c_ds)),
            "delta_specificity_iqr": compute_iqr(c_ds),
            "delta_specificity_std": float(c_ds.std(ddof=c_ddof)),
            "delta_specificity_ci_l": c_ci_sl,
            "delta_specificity_ci_h": c_ci_sh,
            "n_positive_specificity": int((c_ds > 0).sum()),
            "p_specificity": float(c_ps),
            "ci_lower_bound_positive": bool(c_ci_tl > 0),
            "specificity_ci_lower_bound_positive": bool(c_ci_sl > 0),
            "target_beats_wrong": bool(float(c_ds.mean()) > 0),
            "win_rate_target": f"{int((c_dt > 0).sum())}/{c_n}",
            "win_rate_wrong": f"{int((c_dw > 0).sum())}/{c_n}",
            "win_rate_specificity": f"{int((c_ds > 0).sum())}/{c_n}",
        }
    else:
        conf_summary = {
            "status": "not_available",
            "reason": f"Incomplete confirmatory test set (observed {int(conf_mask.sum())}/40 required test cities across Folds 2–5; 10 test cities per fold required)",
        }

    # --- Per-Fold Breakdown ---
    per_fold = {}
    for f in sorted(set(fid)):
        idx = (fid == f)
        f_dt = dt[idx]
        f_dw = dw[idx]
        f_ds = ds[idx]
        f_c0 = c0[idx]
        f_n = int(idx.sum())
        f_ddof = 1 if f_n > 1 else 0
        per_fold[f"fold_{f}"] = {
            "n_cities": f_n,
            "role": "Exploratory / Development" if f == 1 else "Confirmatory Out-of-Fold",
            "cpc_baseline_mean": float(f_c0.mean()),
            "cpc_baseline_std": float(f_c0.std(ddof=f_ddof)),
            "delta_target_mean": float(f_dt.mean()),
            "delta_target_median": float(np.median(f_dt)),
            "delta_target_iqr": compute_iqr(f_dt),
            "delta_target_std": float(f_dt.std(ddof=f_ddof)),
            "delta_wrong_mean": float(f_dw.mean()),
            "delta_wrong_median": float(np.median(f_dw)),
            "delta_wrong_iqr": compute_iqr(f_dw),
            "delta_specificity_mean": float(f_ds.mean()),
            "delta_specificity_median": float(np.median(f_ds)),
            "n_positive_target": int((f_dt > 0).sum()),
            "n_positive_specificity": int((f_ds > 0).sum()),
            "win_rate_target": f"{int((f_dt > 0).sum())}/{f_n}",
            "win_rate_specificity": f"{int((f_ds > 0).sum())}/{f_n}",
            "best_epoch": fold_manifest.get(f, {}).get("best_epoch") if fold_manifest else None,
            "best_val_cpc": fold_manifest.get(f, {}).get("best_val_cpc") if fold_manifest else None,
            "convergence_gate": fold_manifest.get(f, {}).get("convergence_gate", "—") if fold_manifest else "—",
        }

    return {
        "n_cities": n,
        "protocol_version": "e1-v2-amended",
        "is_full_50_complete": is_full_50_complete,
        "is_confirmatory_complete": is_confirmatory_complete,
        "std_ddof": ddof,
        # Full Out-of-Fold (n=50)
        "cpc_baseline_mean": float(c0.mean()),
        "cpc_baseline_std":  float(c0.std(ddof=ddof)),
        "cpc_target_yd_mean": float(cyd.mean()),
        "cpc_target_yd_std": float(cyd.std(ddof=ddof)),
        # Target Delta
        "delta_cpc_target_mean":   float(dt.mean()),
        "delta_cpc_target_median": float(np.median(dt)),
        "delta_cpc_target_iqr":    compute_iqr(dt),
        "delta_cpc_target_std":    float(dt.std(ddof=ddof)),
        "delta_cpc_target_ci_l":   ci_tl,
        "delta_cpc_target_ci_h":   ci_th,
        "n_positive_target":       int((dt > 0).sum()),
        "p_wilcoxon_target":       float(pt),
        # Wrong Placebo Delta (9-donor avg)
        "cpc_wrong_yd_mean": float(cwr.mean()),
        "cpc_wrong_yd_std": float(cwr.std(ddof=ddof)),
        "delta_cpc_wrong_mean":    float(dw.mean()),
        "delta_cpc_wrong_median":  float(np.median(dw)),
        "delta_cpc_wrong_iqr":     compute_iqr(dw),
        "delta_cpc_wrong_std":     float(dw.std(ddof=ddof)),
        "delta_cpc_wrong_ci_l":    ci_wl,
        "delta_cpc_wrong_ci_h":    ci_wh,
        "n_positive_wrong":        int((dw > 0).sum()),
        "p_wilcoxon_wrong":        float(pw),
        # Specificity Estimand (Target - Wrong_Avg9)
        "delta_specificity_mean":   float(ds.mean()),
        "delta_specificity_median": float(np.median(ds)),
        "delta_specificity_iqr":    compute_iqr(ds),
        "delta_specificity_std":    float(ds.std(ddof=ddof)),
        "delta_specificity_ci_l":   ci_sl,
        "delta_specificity_ci_h":   ci_sh,
        "n_positive_specificity":   int((ds > 0).sum()),
        "p_specificity":            float(ps),
        "ci_lower_bound_positive":  bool(ci_tl > 0),
        "specificity_ci_lower_bound_positive": bool(ci_sl > 0),
        "target_beats_wrong":       bool(float(ds.mean()) > 0),
        "win_rate_target":          f"{int((dt > 0).sum())}/{n}",
        "win_rate_wrong":           f"{int((dw > 0).sum())}/{n}",
        "win_rate_specificity":     f"{int((ds > 0).sum())}/{n}",
        # Primary Confirmatory Subgroup (Folds 2-5, only valid when complete)
        "confirmatory_folds_2_5":   conf_summary,
        # Per-fold breakdown
        "per_fold": per_fold,
        "fold_validation_manifest": fold_manifest or {},
        # Runtime Environment & Multi-core CPU metadata
        "runtime_environment": get_runtime_metadata(),
    }


def write_tables(results: list, summary: dict, table_dir: Path | None = None):
    """
    Generates GitHub Markdown Tables following the Nature/PNAS reporting standard.
    """
    tdir = table_dir or (RESULTS_DIR / "tables")
    tdir.mkdir(parents=True, exist_ok=True)
    n  = summary["n_cities"]
    tl, th = summary["delta_cpc_target_ci_l"], summary["delta_cpc_target_ci_h"]
    wl, wh = summary["delta_cpc_wrong_ci_l"], summary["delta_cpc_wrong_ci_h"]
    sl, sh = summary["delta_specificity_ci_l"], summary["delta_specificity_ci_h"]
    c0m = summary["cpc_baseline_mean"]
    c0s = summary["cpc_baseline_std"]
    is_conf = summary.get("is_confirmatory_complete", False)
    is_full = summary.get("is_full_50_complete", False)

    run_type_str = "Full 50-City Protocol" if is_full else "Exploratory / Smoke Subset"

    lines = [
        f"# Table E1: Oracle Aggregated-Distance Existence Test ({run_type_str})",
        "",
        "> **Methodological Framing & Amendment Context**:",
        "> *\"We report the pooled five-fold out-of-fold benchmark across 50 cities as the primary cross-validated performance summary. Because Fold 1 contributed to protocol development, we additionally report the originally designated Folds 2–5 analysis as a confirmatory sensitivity analysis. Both analyses use five separately trained fold-specific models, and each city is evaluated exactly once when held out.\"*",
        "",
        "### Analysis Sets Hierarchy",
        "",
        "| Analysis set | n | Role |",
        "|---|---:|---|",
        "| All Folds 1–5 | 50 | Pooled out-of-fold benchmark |",
        "| Excluding Fold 1 | 40 | Confirmatory sensitivity |",
        "| Fold 1 | 10 | Development/exploratory diagnostic |",
        "",
        f"**Execution Status**: {len(results)}/50 test cities evaluated | is_confirmatory_complete={is_conf} | is_full_50_complete={is_full}",
        f"**Protocol**: 5-fold stratified city CV (35 train / 5 val / 10 test per fold, locked manifest v2)",
        f"**Parameters**: K_move={K_MOVE} bins (pair-weighted quantile), q={Q_CALIB} (within-tolerance calibration, tolerance 10⁻⁵), max_epochs={EPOCHS}, patience={PATIENCE}, std_ddof={summary['std_ddof']}",
        "",
    ]

    # Section 1: Primary Pooled Benchmark (Full Out-of-Fold 50 Cities)
    cov_label = "E1-A: Primary Pooled Out-of-Fold Benchmark (All Folds 1–5, n=50)" if is_full else f"E1-A: Primary Benchmark (Observed {n} Cities)"
    lines.extend([
        f"## {cov_label}",
        "",
        "| Condition / Estimand | Interzonal CPC (Mean ± SD) | Mean Δ | Median Δ | IQR | 95% Fold-Stratified Bootstrap CI | Win Rate | Wilcoxon p |",
        "|---|---|---|---|---|---|---|---|",
        f"| Zero-Shot Baseline (M₀) | {c0m:.4f} ± {c0s:.4f} | — | — | — | — | — | — |",
        (f"| + Oracle Y_D (target) | {summary['cpc_target_yd_mean']:.4f} ± {summary['cpc_target_yd_std']:.4f} | "
         f"+{summary['delta_cpc_target_mean']:.4f} | +{summary['delta_cpc_target_median']:.4f} | {summary['delta_cpc_target_iqr']:.4f} | "
         f"[{tl:+.4f}, {th:+.4f}] | {summary['win_rate_target']} | {summary['p_wilcoxon_target']:.2e} |"),
        (f"| + Oracle Y_D (wrong donors avg 9) | {summary['cpc_wrong_yd_mean']:.4f} ± {summary['cpc_wrong_yd_std']:.4f} | "
         f"{summary['delta_cpc_wrong_mean']:+.4f} | {summary['delta_cpc_wrong_median']:+.4f} | {summary['delta_cpc_wrong_iqr']:.4f} | "
         f"[{wl:+.4f}, {wh:+.4f}] | {summary['win_rate_wrong']} | {summary['p_wilcoxon_wrong']:.2e} |"),
        (f"| **Specificity Gain (Target − Wrong)** | — | "
         f"**+{summary['delta_specificity_mean']:.4f}** | **+{summary['delta_specificity_median']:.4f}** | {summary['delta_specificity_iqr']:.4f} | "
         f"**[{sl:+.4f}, {sh:+.4f}]** | **{summary['win_rate_specificity']}** | **{summary['p_specificity']:.2e}** |"),
        "",
    ])

    # Section 2: Confirmatory Sensitivity Analysis on Folds 2-5 (strictly when complete)
    conf = summary.get("confirmatory_folds_2_5")
    if is_conf and conf and conf.get("status") == "confirmatory_complete":
        c_tl, c_th = conf["delta_cpc_target_ci_l"], conf["delta_cpc_target_ci_h"]
        c_wl, c_wh = conf["delta_cpc_wrong_ci_l"], conf["delta_cpc_wrong_ci_h"]
        c_sl, c_sh = conf["delta_specificity_ci_l"], conf["delta_specificity_ci_h"]
        lines.extend([
            "## E1-B: Confirmatory Sensitivity Analysis (Excluding Fold 1: Folds 2–5, n=40)",
            "",
            "| Condition / Estimand | Interzonal CPC (Mean ± SD) | Mean Δ | Median Δ | IQR | 95% Fold-Stratified Bootstrap CI | Win Rate | Wilcoxon p |",
            "|---|---|---|---|---|---|---|---|",
            f"| Zero-Shot Baseline (M₀) | {conf['cpc_baseline_mean']:.4f} ± {conf['cpc_baseline_std']:.4f} | — | — | — | — | — | — |",
            (f"| + Oracle Y_D (target) | {conf['cpc_target_yd_mean']:.4f} ± {conf['cpc_target_yd_std']:.4f} | "
             f"+{conf['delta_cpc_target_mean']:.4f} | +{conf['delta_cpc_target_median']:.4f} | {conf['delta_cpc_target_iqr']:.4f} | "
             f"[{c_tl:+.4f}, {c_th:+.4f}] | {conf['win_rate_target']} | {conf['p_wilcoxon_target']:.2e} |"),
            (f"| + Oracle Y_D (wrong donors avg 9) | {conf['delta_cpc_wrong_mean'] + conf['cpc_baseline_mean']:.4f} ± {conf['delta_cpc_wrong_std']:.4f} | "
             f"{conf['delta_cpc_wrong_mean']:+.4f} | {conf['delta_cpc_wrong_median']:+.4f} | {conf['delta_cpc_wrong_iqr']:.4f} | "
             f"[{c_wl:+.4f}, {c_wh:+.4f}] | {conf['win_rate_wrong']} | {conf['p_wilcoxon_wrong']:.2e} |"),
            (f"| **Specificity Gain (Target − Wrong)** | — | "
             f"**+{conf['delta_specificity_mean']:.4f}** | **+{conf['delta_specificity_median']:.4f}** | {conf['delta_specificity_iqr']:.4f} | "
             f"**[{c_sl:+.4f}, {c_sh:+.4f}]** | **{conf['win_rate_specificity']}** | **{conf['p_specificity']:.2e}** |"),
            "",
        ])
    else:
        lines.extend([
            "## E1-B: Confirmatory Sensitivity Analysis (Excluding Fold 1: Folds 2–5, n=40)",
            "",
            f"> *Status: NOT AVAILABLE (Observed {len([r for r in results if r['fold']>=2])}/40 test cities; Confirmatory evaluation strictly requires complete 40 test cities across Folds 2–5, with 10 test cities per fold).* ",
            "",
        ])

    lines.extend([
        "## E1-C: Per-Fold Independent Training & Evaluation Breakdown",
        "",
        "| Fold | Role | Test Cities | Best Epoch | Best Val CPC | Convergence Gate | M₀ CPC | +Target CPC | Mean ΔTarget | Mean ΔWrong (9 Avg) | Specificity Win Rate |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ])

    for f_key, pf in summary.get("per_fold", {}).items():
        f_num = f_key.replace("fold_", "")
        b_ep = pf.get("best_epoch", "—")
        b_vc = f"{pf['best_val_cpc']:.4f}" if pf.get("best_val_cpc") is not None else "—"
        c_gate = pf.get("convergence_gate", "—")
        role = "Exploratory" if f_num == "1" else "Confirmatory"
        lines.append(
            f"| Fold {f_num} | {role} | {pf['n_cities']} | {b_ep} | {b_vc} | {c_gate} | "
            f"{pf['cpc_baseline_mean']:.4f} | {pf['cpc_baseline_mean'] + pf['delta_target_mean']:.4f} | "
            f"{pf['delta_target_mean']:+.4f} | {pf['delta_wrong_mean']:+.4f} | {pf['win_rate_specificity']} |"
        )

    # Section 3: Acceptance Criteria (Dynamic gating)
    if is_conf and conf and conf.get("status") == "confirmatory_complete":
        e_tl, e_th = conf["delta_cpc_target_ci_l"], conf["delta_cpc_target_ci_h"]
        e_sl, e_sh = conf["delta_specificity_ci_l"], conf["delta_specificity_ci_h"]
        lines.extend([
            "",
            "## Acceptance Criteria Verification (Confirmatory Folds 2–5, n=40)",
            "",
            "| Criterion | Required Condition | Observed Value | Verdict |",
            "|---|---|---|---|",
            f"| Confirmatory CI Lower Bound | CI_lower > 0 | [{e_tl:+.4f}, {e_th:+.4f}] | {'✓ PASS' if conf['ci_lower_bound_positive'] else '✗ FAIL'} |",
            f"| Specificity Superiority | Mean ΔCPC_target > Mean ΔCPC_wrong | {conf['delta_cpc_target_mean']:+.4f} vs {conf['delta_cpc_wrong_mean']:+.4f} (Diff: +{conf['delta_specificity_mean']:.4f}) | {'✓ PASS' if conf['target_beats_wrong'] else '✗ FAIL'} |",
            f"| Specificity CI Lower Bound | Specificity CI_lower > 0 | [{e_sl:+.4f}, {e_sh:+.4f}] | {'✓ PASS' if conf['specificity_ci_lower_bound_positive'] else '✗ FAIL'} |",
            f"| Specificity Significance | Paired Wilcoxon p < 0.05 | p = {conf['p_specificity']:.2e} | {'✓ PASS' if conf['p_specificity'] < 0.05 else '✗ FAIL'} |",
            f"| City-level Specificity Consistency | Win rate > 70% (>28/40) | {conf['win_rate_specificity']} | {'✓ PASS' if int(conf['win_rate_specificity'].split('/')[0]) >= 28 else '✗ FAIL'} |",
            "",
        ])
    else:
        lines.extend([
            "",
            "## Acceptance Criteria Verification",
            "",
            f"> *Status: PENDING FULL 50-CITY EXECUTION (Evaluated {n}/50 cities. Confirmatory criteria will be locked upon full completion).* ",
            "",
        ])

    (tdir / "e1_main_table.md").write_text("\n".join(lines), encoding="utf-8")

    hdr = "| City | Fold | n_pairs | CPC₀ | CPC_target | ΔCPC_target | CPC_wrong (avg 9) | ΔCPC_wrong (avg 9) | ΔSpecificity | Donors |"
    sep = "|---|---|---|---|---|---|---|---|---|---|"
    rows = [hdr, sep]
    for r in sorted(results, key=lambda x: x["city"]):
        rows.append(
            f"| {r['city']} | {r['fold']} | {r['n_inter_pairs']} | "
            f"{r['cpc_baseline']:.4f} | {r['cpc_target_yd']:.4f} | "
            f"{r['delta_cpc_target']:+.4f} | {r['cpc_wrong_yd']:.4f} | "
            f"{r['delta_cpc_wrong']:+.4f} | {r['delta_cpc_specificity']:+.4f} | 9 fold donors |"
        )
    (tdir / "e1_per_city.md").write_text("# E1: Complete Per-City Breakdown (50 Cities)\n\n" + "\n".join(rows) + "\n", encoding="utf-8")
    print(f"  [Artifact] Generated Markdown tables in {tdir}")


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

        fold_role = "Exploratory / Development" if fold_id == 1 else "Confirmatory Out-of-Fold"
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
        _ckpt_seed = seed + fold_id
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
    is_conf = summary.get("is_confirmatory_complete", False)
    is_full = summary.get("is_full_50_complete", False)
    conf = summary.get("confirmatory_folds_2_5", {})

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

    if is_conf and conf.get("status") == "confirmatory_complete":
        pass_ci = "PASS" if conf['ci_lower_bound_positive'] else "FAIL"
        pass_sci = "PASS" if conf['specificity_ci_lower_bound_positive'] else "FAIL"
        pass_tw = "PASS" if conf['target_beats_wrong'] else "FAIL"
        log_msg("\n  CONFIRMATORY HYPOTHESIS TEST OUTCOMES (Folds 2-5, n=40):")
        log_msg(f"    * Target 95% CI Lower Bound > 0      : {pass_ci} ([{conf['delta_cpc_target_ci_l']:+.4f}, {conf['delta_cpc_target_ci_h']:+.4f}])")
        log_msg(f"    * Specificity 95% CI Lower Bound > 0 : {pass_sci} ([{conf['delta_specificity_ci_l']:+.4f}, {conf['delta_specificity_ci_h']:+.4f}])")
        log_msg(f"    * Specificity Gain (Target > Wrong)  : {pass_tw} (+{conf['delta_specificity_mean']:+.4f} vs 0)")
        log_msg(f"    * Specificity Win Rate               : {conf['win_rate_specificity']} (Wilcoxon p = {conf['p_specificity']:.2e})")
    else:
        conf_count = len([r for r in all_results if r['fold'] >= 2])
        log_msg(f"\n  CONFIRMATORY STATUS: NOT AVAILABLE (Observed {conf_count}/40 test cities; strictly requires complete 40 test cities across Folds 2-5, with 10 test cities per fold).")
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
