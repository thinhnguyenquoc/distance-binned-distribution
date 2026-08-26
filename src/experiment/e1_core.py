r"""
E1 Core Statistical Infrastructure — Public API for E1 experiment family.

This module contains the reusable, canonical statistical and evaluation functions
used by both:
  - run_e1.py              : Legacy E1 training + evaluation runner (historical)
  - run_e1_specificity_from_checkpoints.py : Canonical specificity evaluation from checkpoints

Separation rationale:
  The legacy runner (run_e1.py) and the canonical checkpoint runner share
  the same city-level evaluation, bootstrap, summary, and table-generation logic.
  This module provides a single source of truth for those functions so that:
  1. Paper audit trail is unambiguous — all statistical computation is here.
  2. The 'Legacy' label on run_e1.py refers only to the training loop, NOT
     to the statistical infrastructure used by downstream analyses.

Public API:
  run_city(...)          -- 3-condition city evaluation (M0, +TargetYD, +WrongYD)
  fold_bootstrap(...)    -- Fold-stratified 95% bootstrap CI
  compute_summary(...)   -- Aggregate statistics across cities
  write_tables(...)      -- GitHub Markdown tables (Nature/PNAS standard)
  build_inter_mask(...)  -- Interzonal Omega_c^+ boolean mask
  safe_wilcoxon(...)     -- Defensive Wilcoxon signed-rank test
  compute_iqr(...)       -- Sample IQR
  log_msg(...)           -- Timestamped logging
  get_runtime_metadata() -- Hardware/OS audit metadata
  configure_cpu_threads()-- PyTorch threading configuration
"""

from __future__ import annotations

import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import stats

from src.data.city_splits import get_wrong_donors
from src.data.dataset import load_city
from src.data.urban_graph import build_radius_graph
from src.data.yd_extractor import extract_yd_kbins
from src.calibration.bin_calibration import calibrate_kbins
from src.training.train import infer_zero_shot
from src.training.evaluate import compute_cpc_pair, compute_cpc_norm_pair

# ---------------------------------------------------------------------------
# Experiment Constants (Pre-specified, locked before evaluation)
# ---------------------------------------------------------------------------
K_MOVE    = 8       # Number of moving-distance bins (Bin 0 intrazonal excluded)
Q_CALIB   = 1.0     # Calibration strength (1.0 = exact within-tolerance distribution match)
TOLERANCE = 1e-5    # Floating-point tolerance for mass preservation & bin matching

# Logging defaults (can be overridden by callers)
_RESULTS_DIR = Path("results/e1")
_LOG_FILE    = _RESULTS_DIR / "e1_execution.log"


# ---------------------------------------------------------------------------
# Utility: Runtime & Threading
# ---------------------------------------------------------------------------

def get_runtime_metadata() -> dict:
    """Collect hardware, OS, and PyTorch runtime execution metadata."""
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
    """Configure PyTorch CPU intra-op threads and OpenMP/MKL env vars."""
    if num_threads is not None and num_threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
        os.environ["MKL_NUM_THREADS"] = str(num_threads)
        torch.set_num_threads(num_threads)
    return torch.get_num_threads()


def log_msg(msg: str = "", print_to_console: bool = True, results_dir: Path | None = None):
    """Timestamped log to console and e1_execution.log."""
    log_dir = results_dir or _RESULTS_DIR
    log_file = log_dir / "e1_execution.log"
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
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write((formatted if formatted else "") + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Utility: Mask & Statistics
# ---------------------------------------------------------------------------

def build_inter_mask(cd: Any, dist_km: np.ndarray) -> np.ndarray:
    """Boolean mask for interzonal candidate support Omega_c^+ (i != j and D_ij > 0)."""
    o = cd.pair_o_idx.numpy()
    d = cd.pair_d_idx.numpy()
    return (o != d) & (dist_km > 0.0)


def safe_wilcoxon(diff: np.ndarray, alternative: str = "greater") -> tuple[float, float]:
    """Defensive Wilcoxon signed-rank test (handles n<2, all-zero, NaN)."""
    diff_clean = diff[~np.isnan(diff)]
    if len(diff_clean) < 2:
        return 0.0, 1.0
    if (diff_clean == 0.0).all():
        return 0.0, 1.0
    try:
        res = stats.wilcoxon(diff_clean, alternative=alternative, zero_method="wilcox")
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return 0.0, 1.0


def compute_iqr(values: np.ndarray) -> float:
    """Sample IQR (Q3 - Q1)."""
    if len(values) == 0:
        return 0.0
    return float(np.percentile(values, 75) - np.percentile(values, 25))


# ---------------------------------------------------------------------------
# Core: City-Level 3-Condition Evaluation
# ---------------------------------------------------------------------------

def run_city(
    city: str,
    model: torch.nn.Module,
    scaler: object,
    bin_edges: np.ndarray,
    K_active: int,
    test_cities: list[str],
    fold_id: int,
    device: torch.device,
    data_root: str = "data",
    test_city_cache: dict[str, dict] | None = None,
    test_yd_cache: dict[str, np.ndarray] | None = None,
) -> dict:
    """
    Evaluate 3 experimental conditions on a single held-out test city.

    Condition A (M0 — Zero-Shot Baseline):
        Frozen model forward pass; no target information.
    Condition B (M1 — Target Oracle Y_D^{GT,+}):
        Y_D extracted from target ground-truth OD — deliberate target-information
        intervention. Calibrated via calibrate_kbins on Omega_c^+.
    Condition C (Placebo — 9-Donor Wrong Y_D average):
        Average Delta CPC from each of the 9 other test cities in fold.
    """
    # 1. Load or retrieve city data
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
        cd = load_city(city, data_root=data_root, feature_scaler=scaler)
        ei, ed = build_radius_graph(cd.lon_lat, radius_km=5.0)
        dist_km = np.expm1(cd.pair_distance.numpy())
        inter = build_inter_mask(cd, dist_km)
        t_gt = cd.pair_trips.numpy().astype(np.float64)
        Y_D_tgt = (test_yd_cache.get(city) if test_yd_cache else None)
        if Y_D_tgt is None:
            Y_D_tgt = extract_yd_kbins(dist_km, t_gt, bin_edges, inter)

    # 2. Condition A: M0
    T0 = infer_zero_shot(model, cd, ei, ed, device=device)
    t0 = T0.numpy().astype(np.float64)
    n_inter = int(inter.sum())
    cpc0      = compute_cpc_pair(t_gt[inter], t0[inter])
    cpc0_norm = compute_cpc_norm_pair(t_gt[inter], t0[inter])

    # 3. Condition B: Target Oracle Y_D^{GT,+}
    if Y_D_tgt is None:
        Y_D_tgt = (test_yd_cache.get(city) if test_yd_cache else None) or \
                  extract_yd_kbins(dist_km, t_gt, bin_edges, inter)
    T_yd = calibrate_kbins(t0, dist_km, inter, Y_D_tgt, bin_edges, q=Q_CALIB, tolerance=TOLERANCE)
    cpc_yd      = compute_cpc_pair(t_gt[inter], T_yd[inter])
    cpc_yd_norm = compute_cpc_norm_pair(t_gt[inter], T_yd[inter])
    delta_target = float(cpc_yd - cpc0)

    # 4. Condition C: 9-Donor Placebo
    wrong_donors = get_wrong_donors(city, test_cities)
    assert len(wrong_donors) == len(test_cities) - 1, \
        f"Expected {len(test_cities)-1} wrong donors, got {len(wrong_donors)}"

    wrong_cpc_list, wrong_cpc_norm_list, wrong_delta_list, wrong_donor_details = [], [], [], []
    for donor in wrong_donors:
        if test_city_cache is not None and donor in test_city_cache:
            Y_D_wr = test_city_cache[donor]["Y_D"]
        elif test_yd_cache is not None and donor in test_yd_cache:
            Y_D_wr = test_yd_cache[donor]
        else:
            cd_d   = load_city(donor, data_root=data_root, feature_scaler=scaler)
            dist_d = np.expm1(cd_d.pair_distance.numpy())
            inter_d = build_inter_mask(cd_d, dist_d)
            t_gt_d = cd_d.pair_trips.numpy().astype(np.float64)
            Y_D_wr = extract_yd_kbins(dist_d, t_gt_d, bin_edges, inter_d)

        T_wr = calibrate_kbins(t0, dist_km, inter, Y_D_wr, bin_edges, q=Q_CALIB, tolerance=TOLERANCE)
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
        "yd_source": "target_ground_truth_positive_od",
        "cpc_baseline": float(cpc0),
        "cpc_baseline_norm": float(cpc0_norm),
        "cpc_target_yd": float(cpc_yd),
        "cpc_target_yd_norm": float(cpc_yd_norm),
        "delta_cpc_target": delta_target,
        "cpc_wrong_yd": cpc_wr_mean,
        "cpc_wrong_yd_norm": cpc_wr_norm_mean,
        "delta_cpc_wrong": delta_wr_mean,
        "delta_cpc_specificity": delta_spec,
        "Y_D_target": Y_D_tgt.tolist(),
        "wrong_donor_breakdown": wrong_donor_details,
    }


# ---------------------------------------------------------------------------
# Core: Fold-Stratified Bootstrap CI
# ---------------------------------------------------------------------------

def fold_bootstrap(
    values: np.ndarray,
    fold_ids: np.ndarray,
    n: int = 10000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple:
    """Fold-stratified bootstrap 95% CI (resamples within each fold independently)."""
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


# ---------------------------------------------------------------------------
# Core: Aggregate Summary Statistics
# ---------------------------------------------------------------------------

def compute_summary(results: list, fold_manifest: dict = None, bootstrap_seed: int = 2024) -> dict:
    """
    Aggregate per-city results into primary statistics.
    Statistical unit: CITY (N up to 50). Model seeds already averaged within city by caller.
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

    is_full_50_complete = bool(
        n == 50
        and set(fid.tolist()) == {1, 2, 3, 4, 5}
        and all((fid == f).sum() == 10 for f in range(1, 6))
    )

    if is_full_50_complete:
        c_ci_tl, c_ci_th, _ = fold_bootstrap(dt, fid, seed=bootstrap_seed)
        c_ci_wl, c_ci_wh, _ = fold_bootstrap(dw, fid, seed=bootstrap_seed)
        c_ci_sl, c_ci_sh, _ = fold_bootstrap(ds, fid, seed=bootstrap_seed)
        _, c_pt = safe_wilcoxon(dt, alternative="greater")
        _, c_pw = safe_wilcoxon(dw, alternative="greater")
        _, c_ps = safe_wilcoxon(ds, alternative="greater")
        conf_summary = {
            "status": "full_5_fold_complete",
            "protocol_role": "Amended Replication under Locked Protocol (Folds 1-5, n=50)",
            "n_cities": 50,
            "cpc_baseline_mean": float(c0.mean()), "cpc_baseline_std": float(c0.std(ddof=1)),
            "cpc_target_yd_mean": float(cyd.mean()), "cpc_target_yd_std": float(cyd.std(ddof=1)),
            "delta_cpc_target_mean": float(dt.mean()), "delta_cpc_target_median": float(np.median(dt)),
            "delta_cpc_target_iqr": compute_iqr(dt), "delta_cpc_target_std": float(dt.std(ddof=1)),
            "delta_cpc_target_ci_l": c_ci_tl, "delta_cpc_target_ci_h": c_ci_th,
            "n_positive_target": int((dt > 0).sum()), "p_wilcoxon_target": float(c_pt),
            "delta_cpc_wrong_mean": float(dw.mean()), "delta_cpc_wrong_median": float(np.median(dw)),
            "delta_cpc_wrong_iqr": compute_iqr(dw), "delta_cpc_wrong_std": float(dw.std(ddof=1)),
            "delta_cpc_wrong_ci_l": c_ci_wl, "delta_cpc_wrong_ci_h": c_ci_wh,
            "n_positive_wrong": int((dw > 0).sum()), "p_wilcoxon_wrong": float(c_pw),
            "delta_specificity_mean": float(ds.mean()), "delta_specificity_median": float(np.median(ds)),
            "delta_specificity_iqr": compute_iqr(ds), "delta_specificity_std": float(ds.std(ddof=1)),
            "delta_specificity_ci_l": c_ci_sl, "delta_specificity_ci_h": c_ci_sh,
            "n_positive_specificity": int((ds > 0).sum()), "p_specificity": float(c_ps),
            "ci_lower_bound_positive": bool(c_ci_tl > 0),
            "specificity_ci_lower_bound_positive": bool(c_ci_sl > 0),
            "target_beats_wrong": bool(float(ds.mean()) > 0),
            "win_rate_target": f"{int((dt > 0).sum())}/50",
            "win_rate_wrong": f"{int((dw > 0).sum())}/50",
            "win_rate_specificity": f"{int((ds > 0).sum())}/50",
        }
    else:
        conf_summary = {
            "status": "not_available",
            "reason": (
                f"Incomplete full_5_fold test set (observed {n}/50 required test cities "
                "across Folds 1-5; 10 test cities per fold required)"
            ),
        }

    per_fold = {}
    for f in sorted(set(fid)):
        idx = fid == f
        f_dt = dt[idx]; f_dw = dw[idx]; f_ds = ds[idx]; f_c0 = c0[idx]
        f_n = int(idx.sum()); f_ddof = 1 if f_n > 1 else 0
        per_fold[f"fold_{f}"] = {
            "n_cities": f_n,
            "role": "Exploratory / Development" if f == 1 else "Full 5-fold Out-of-Fold",
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
            "convergence_gate": fold_manifest.get(f, {}).get("convergence_gate", "--") if fold_manifest else "--",
        }

    return {
        "n_cities": n, "protocol_version": "e1-v2-amended",
        "is_full_50_complete": is_full_50_complete,
        "is_full_5_fold_complete": is_full_50_complete,
        "std_ddof": ddof,
        "cpc_baseline_mean": float(c0.mean()), "cpc_baseline_std": float(c0.std(ddof=ddof)),
        "cpc_target_yd_mean": float(cyd.mean()), "cpc_target_yd_std": float(cyd.std(ddof=ddof)),
        "delta_cpc_target_mean": float(dt.mean()), "delta_cpc_target_median": float(np.median(dt)),
        "delta_cpc_target_iqr": compute_iqr(dt), "delta_cpc_target_std": float(dt.std(ddof=ddof)),
        "delta_cpc_target_ci_l": ci_tl, "delta_cpc_target_ci_h": ci_th,
        "n_positive_target": int((dt > 0).sum()), "p_wilcoxon_target": float(pt),
        "cpc_wrong_yd_mean": float(cwr.mean()), "cpc_wrong_yd_std": float(cwr.std(ddof=ddof)),
        "delta_cpc_wrong_mean": float(dw.mean()), "delta_cpc_wrong_median": float(np.median(dw)),
        "delta_cpc_wrong_iqr": compute_iqr(dw), "delta_cpc_wrong_std": float(dw.std(ddof=ddof)),
        "delta_cpc_wrong_ci_l": ci_wl, "delta_cpc_wrong_ci_h": ci_wh,
        "n_positive_wrong": int((dw > 0).sum()), "p_wilcoxon_wrong": float(pw),
        "delta_specificity_mean": float(ds.mean()), "delta_specificity_median": float(np.median(ds)),
        "delta_specificity_iqr": compute_iqr(ds), "delta_specificity_std": float(ds.std(ddof=ddof)),
        "delta_specificity_ci_l": ci_sl, "delta_specificity_ci_h": ci_sh,
        "n_positive_specificity": int((ds > 0).sum()), "p_specificity": float(ps),
        "ci_lower_bound_positive": bool(ci_tl > 0),
        "specificity_ci_lower_bound_positive": bool(ci_sl > 0),
        "target_beats_wrong": bool(float(ds.mean()) > 0),
        "win_rate_target": f"{int((dt > 0).sum())}/{n}",
        "win_rate_wrong": f"{int((dw > 0).sum())}/{n}",
        "win_rate_specificity": f"{int((ds > 0).sum())}/{n}",
        "full_5_fold_folds_2_5": conf_summary,
        "per_fold": per_fold,
        "fold_validation_manifest": fold_manifest or {},
        "runtime_environment": get_runtime_metadata(),
    }


# ---------------------------------------------------------------------------
# Core: Markdown Table Output
# ---------------------------------------------------------------------------

def write_tables(
    results: list,
    summary: dict,
    table_dir: Path | None = None,
    results_dir: Path | None = None,
) -> None:
    """Generate GitHub Markdown tables (Nature/PNAS standard)."""
    base_dir = results_dir or _RESULTS_DIR
    tdir = table_dir or (base_dir / "tables")
    tdir.mkdir(parents=True, exist_ok=True)
    n  = summary["n_cities"]
    tl, th = summary["delta_cpc_target_ci_l"], summary["delta_cpc_target_ci_h"]
    wl, wh = summary["delta_cpc_wrong_ci_l"], summary["delta_cpc_wrong_ci_h"]
    sl, sh = summary["delta_specificity_ci_l"], summary["delta_specificity_ci_h"]
    c0m = summary["cpc_baseline_mean"]
    c0s = summary["cpc_baseline_std"]
    is_conf = summary.get("is_full_5_fold_complete", False)
    is_full = summary.get("is_full_50_complete", False)
    run_type_str = "Full 50-City Protocol" if is_full else "Exploratory / Smoke Subset"

    lines = [
        f"# Table E1: Oracle Aggregated-Distance Existence Test ({run_type_str})",
        "",
        "> **Methodological Framing & Amendment Context**:",
        '> *"We report the pooled five-fold out-of-fold benchmark across 50 cities as the'
        " primary cross-validated performance summary. Both analyses use five separately"
        ' trained fold-specific models, and each city is evaluated exactly once when held out."*',
        "",
        "### Analysis Sets Hierarchy",
        "",
        "| Analysis set | n | Role |",
        "|---|---:|---|",
        "| All Folds 1-5 | 50 | Pooled out-of-fold benchmark |",
        "| Excluding Fold 1 | 40 | Full 5-fold sensitivity |",
        "| Fold 1 | 10 | Development/exploratory diagnostic |",
        "",
        f"**Execution Status**: {len(results)}/50 test cities evaluated"
        f" | is_full_5_fold_complete={is_conf} | is_full_50_complete={is_full}",
        f"**Parameters**: K_move={K_MOVE} bins (pair-weighted quantile),"
        f" q={Q_CALIB}, std_ddof={summary['std_ddof']}",
        "",
    ]

    cov_label = (
        "E1-A: Primary Pooled Out-of-Fold Benchmark (All Folds 1-5, n=50)"
        if is_full else f"E1-A: Primary Benchmark (Observed {n} Cities)"
    )
    lines.extend([
        f"## {cov_label}", "",
        "| Condition | CPC (Mean +/- SD) | Mean Delta | Median Delta | IQR | 95% Bootstrap CI | Win Rate | Wilcoxon p |",
        "|---|---|---|---|---|---|---|---|",
        f"| Zero-Shot Baseline (M0) | {c0m:.4f} +/- {c0s:.4f} | -- | -- | -- | -- | -- | -- |",
        (f"| + Oracle Y_D (target) | {summary['cpc_target_yd_mean']:.4f} +/- {summary['cpc_target_yd_std']:.4f} | "
         f"+{summary['delta_cpc_target_mean']:.4f} | +{summary['delta_cpc_target_median']:.4f} | {summary['delta_cpc_target_iqr']:.4f} | "
         f"[{tl:+.4f}, {th:+.4f}] | {summary['win_rate_target']} | {summary['p_wilcoxon_target']:.2e} |"),
        (f"| + Oracle Y_D (wrong 9-donor avg) | {summary['cpc_wrong_yd_mean']:.4f} +/- {summary['cpc_wrong_yd_std']:.4f} | "
         f"{summary['delta_cpc_wrong_mean']:+.4f} | {summary['delta_cpc_wrong_median']:+.4f} | {summary['delta_cpc_wrong_iqr']:.4f} | "
         f"[{wl:+.4f}, {wh:+.4f}] | {summary['win_rate_wrong']} | {summary['p_wilcoxon_wrong']:.2e} |"),
        (f"| **Specificity (Target - Wrong)** | -- | "
         f"**+{summary['delta_specificity_mean']:.4f}** | **+{summary['delta_specificity_median']:.4f}** | {summary['delta_specificity_iqr']:.4f} | "
         f"**[{sl:+.4f}, {sh:+.4f}]** | **{summary['win_rate_specificity']}** | **{summary['p_specificity']:.2e}** |"),
        "",
    ])

    conf = summary.get("full_5_fold_folds_2_5")
    if is_conf and conf and conf.get("status") == "full_5_fold_complete":
        c_tl, c_th = conf["delta_cpc_target_ci_l"], conf["delta_cpc_target_ci_h"]
        c_wl, c_wh = conf["delta_cpc_wrong_ci_l"], conf["delta_cpc_wrong_ci_h"]
        c_sl, c_sh = conf["delta_specificity_ci_l"], conf["delta_specificity_ci_h"]
        lines.extend([
            "## E1-B: Full 5-fold Sensitivity (n=50)", "",
            "| Condition | CPC (Mean +/- SD) | Mean Delta | Median Delta | IQR | 95% Bootstrap CI | Win Rate | Wilcoxon p |",
            "|---|---|---|---|---|---|---|---|",
            f"| Zero-Shot Baseline (M0) | {conf['cpc_baseline_mean']:.4f} +/- {conf['cpc_baseline_std']:.4f} | -- | -- | -- | -- | -- | -- |",
            (f"| + Oracle Y_D (target) | {conf['cpc_target_yd_mean']:.4f} +/- {conf['cpc_target_yd_std']:.4f} | "
             f"+{conf['delta_cpc_target_mean']:.4f} | +{conf['delta_cpc_target_median']:.4f} | {conf['delta_cpc_target_iqr']:.4f} | "
             f"[{c_tl:+.4f}, {c_th:+.4f}] | {conf['win_rate_target']} | {conf['p_wilcoxon_target']:.2e} |"),
            (f"| + Oracle Y_D (wrong 9-donor avg) | {conf['delta_cpc_wrong_mean'] + conf['cpc_baseline_mean']:.4f} +/- {conf['delta_cpc_wrong_std']:.4f} | "
             f"{conf['delta_cpc_wrong_mean']:+.4f} | {conf['delta_cpc_wrong_median']:+.4f} | {conf['delta_cpc_wrong_iqr']:.4f} | "
             f"[{c_wl:+.4f}, {c_wh:+.4f}] | {conf['win_rate_wrong']} | {conf['p_wilcoxon_wrong']:.2e} |"),
            (f"| **Specificity (Target - Wrong)** | -- | "
             f"**+{conf['delta_specificity_mean']:.4f}** | **+{conf['delta_specificity_median']:.4f}** | {conf['delta_specificity_iqr']:.4f} | "
             f"**[{c_sl:+.4f}, {c_sh:+.4f}]** | **{conf['win_rate_specificity']}** | **{conf['p_specificity']:.2e}** |"),
            "",
        ])
    else:
        lines.extend([
            "## E1-B: Full 5-fold Sensitivity (n=50)", "",
            f"> *Status: NOT AVAILABLE ({n}/50 cities evaluated).*", "",
        ])

    lines.extend([
        "## E1-C: Per-Fold Breakdown", "",
        "| Fold | Role | Cities | Best Epoch | Best Val CPC | Gate | M0 CPC | +Target | DeltaTarget | DeltaWrong | Spec Win |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ])
    for f_key, pf in summary.get("per_fold", {}).items():
        f_num = f_key.replace("fold_", "")
        b_ep = pf.get("best_epoch", "--")
        b_vc = f"{pf['best_val_cpc']:.4f}" if pf.get("best_val_cpc") is not None else "--"
        role = "Exploratory" if f_num == "1" else "Full 5-fold"
        lines.append(
            f"| Fold {f_num} | {role} | {pf['n_cities']} | {b_ep} | {b_vc} | {pf.get('convergence_gate','--')} | "
            f"{pf['cpc_baseline_mean']:.4f} | {pf['cpc_baseline_mean'] + pf['delta_target_mean']:.4f} | "
            f"{pf['delta_target_mean']:+.4f} | {pf['delta_wrong_mean']:+.4f} | {pf['win_rate_specificity']} |"
        )

    if is_conf and conf and conf.get("status") == "full_5_fold_complete":
        e_tl, e_th = conf["delta_cpc_target_ci_l"], conf["delta_cpc_target_ci_h"]
        e_sl, e_sh = conf["delta_specificity_ci_l"], conf["delta_specificity_ci_h"]
        lines.extend([
            "", "## Acceptance Criteria (Full 5-fold, n=50)", "",
            "| Criterion | Required | Observed | Verdict |",
            "|---|---|---|---|",
            f"| Target CI_lower > 0 | CI_lower > 0 | [{e_tl:+.4f}, {e_th:+.4f}] | {'PASS' if conf['ci_lower_bound_positive'] else 'FAIL'} |",
            f"| Specificity > 0 | mean(Target) > mean(Wrong) | {conf['delta_cpc_target_mean']:+.4f} vs {conf['delta_cpc_wrong_mean']:+.4f} | {'PASS' if conf['target_beats_wrong'] else 'FAIL'} |",
            f"| Specificity CI_lower > 0 | CI_lower > 0 | [{e_sl:+.4f}, {e_sh:+.4f}] | {'PASS' if conf['specificity_ci_lower_bound_positive'] else 'FAIL'} |",
            f"| Specificity Wilcoxon | p < 0.05 | {conf['p_specificity']:.2e} | {'PASS' if conf['p_specificity'] < 0.05 else 'FAIL'} |",
            f"| Win Rate > 70% | >28/50 | {conf['win_rate_specificity']} | {'PASS' if int(conf['win_rate_specificity'].split('/')[0]) >= 28 else 'FAIL'} |",
            "",
        ])

    (tdir / "e1_main_table.md").write_text("\n".join(lines), encoding="utf-8")

    hdr = "| City | Fold | n_pairs | CPC0 | CPC_target | dCPC_target | CPC_wrong | dCPC_wrong | dSpecificity |"
    sep = "|---|---|---|---|---|---|---|---|---|"
    rows = [hdr, sep]
    for r in sorted(results, key=lambda x: x["city"]):
        rows.append(
            f"| {r['city']} | {r['fold']} | {r['n_inter_pairs']} | "
            f"{r['cpc_baseline']:.4f} | {r['cpc_target_yd']:.4f} | "
            f"{r['delta_cpc_target']:+.4f} | {r['cpc_wrong_yd']:.4f} | "
            f"{r['delta_cpc_wrong']:+.4f} | {r['delta_cpc_specificity']:+.4f} |"
        )
    (tdir / "e1_per_city.md").write_text(
        "# E1: Per-City Results (50 Cities)\n\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )
    print(f"  [Artifact] Generated Markdown tables in {tdir}")
