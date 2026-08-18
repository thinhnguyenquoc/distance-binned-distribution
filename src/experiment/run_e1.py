"""
E1: Oracle Aggregated-Distance Existence Test
=============================================
Protocol:
  - 5-fold stratified city CV: 35 train / 5 val / 10 test per fold
  - Model selection / early stopping on validation interzonal CPC (patience=5)
  - K_move=8 moving-distance bins (pair-weighted quantile from 35 train cities)
  - q=1.0 within-tolerance calibration (closed-form, mass-preservation and bin-matching errors < 10^-5)
  - 3 conditions: Zero-Shot (M0) / + Oracle Y_D (target) / + Oracle Y_D (wrong donor placebo)
  - Primary metric: Delta-CPC on Omega_c^+ (interzonal)
  - Fold-stratified bootstrap CI (10,000 resamples) + Wilcoxon signed-rank test
  - Sample standard deviation with ddof=1 recorded in metadata
  - Full per-fold summary and validation manifest

NOTE: Y_D is outcome-derived oracle aggregate -- not independent mobility data.
"""

import json
import time
import argparse
import sys
from pathlib import Path
import numpy as np
from scipy import stats
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.city_splits import generate_35_5_10_splits, get_donor_city

from src.data.dataset import load_city
from src.data.urban_graph import build_radius_graph
from src.data.yd_extractor import compute_kbin_edges, extract_yd_kbins
from src.calibration.bin_calibration import calibrate_kbins
from src.training.train import train_zero_shot_model, infer_zero_shot
from src.training.evaluate import compute_cpc_pair, compute_cpc_norm_pair

K_MOVE   = 8
Q_CALIB  = 1.0
EPOCHS   = 25
PATIENCE = 5
DATA_ROOT = "data"
RESULTS_DIR = Path("results/e1")


def build_inter_mask(cd, dist_km: np.ndarray) -> np.ndarray:
    """Mask for interzonal pairs Omega_c^+: i != j and D_ij > 0."""
    o, d = cd.pair_o_idx.numpy(), cd.pair_d_idx.numpy()
    return (o != d) & (dist_km > 0.0)


def safe_wilcoxon(diff: np.ndarray, alternative: str = "greater") -> tuple[float, float]:
    """
    Defensive Wilcoxon signed-rank test handling edge cases:
      - n < 2 observations
      - all differences are exactly zero
      - ties / degeneracies
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


def run_city(
    city: str,
    model,
    scaler,
    bin_edges: np.ndarray,
    K_active: int,
    donor: str,
    fold_id: int,
    device: torch.device,
) -> dict:
    # --- Target city load & Zero-Shot ---
    cd = load_city(city, data_root=DATA_ROOT, feature_scaler=scaler)
    ei, ed = build_radius_graph(cd.lon_lat.numpy(), radius_km=5.0)
    T0 = infer_zero_shot(model, cd, ei, ed, device=device)
    t0 = T0.numpy().astype(np.float64)
    dist_km = np.expm1(cd.pair_distance.numpy())
    inter = build_inter_mask(cd, dist_km)
    t_gt  = cd.pair_trips.numpy().astype(np.float64)

    cpc0      = compute_cpc_pair(t_gt[inter], t0[inter])
    cpc0_norm = compute_cpc_norm_pair(t_gt[inter], t0[inter])

    # --- Condition B: Target Oracle Y_D ---
    Y_D_tgt = extract_yd_kbins(dist_km, t_gt, bin_edges, inter)
    T_yd    = calibrate_kbins(t0, dist_km, inter, Y_D_tgt, bin_edges, q=Q_CALIB)
    cpc_yd      = compute_cpc_pair(t_gt[inter], T_yd[inter])
    cpc_yd_norm = compute_cpc_norm_pair(t_gt[inter], T_yd[inter])

    # --- Condition C: Wrong Donor Oracle Y_D (Placebo) ---
    cd_d   = load_city(donor, data_root=DATA_ROOT, feature_scaler=scaler)
    dist_d = np.expm1(cd_d.pair_distance.numpy())
    inter_d = build_inter_mask(cd_d, dist_d)
    t_gt_d  = cd_d.pair_trips.numpy().astype(np.float64)
    Y_D_wr  = extract_yd_kbins(dist_d, t_gt_d, bin_edges, inter_d)
    T_wr    = calibrate_kbins(t0, dist_km, inter, Y_D_wr, bin_edges, q=Q_CALIB)
    cpc_wr      = compute_cpc_pair(t_gt[inter], T_wr[inter])
    cpc_wr_norm = compute_cpc_norm_pair(t_gt[inter], T_wr[inter])

    return {
        "city": city,
        "fold": fold_id,
        "donor_city": donor,
        "n_inter_pairs": int(inter.sum()),
        "K_active": K_active,
        "cpc_baseline": cpc0,
        "cpc_baseline_norm": cpc0_norm,
        "cpc_target_yd": cpc_yd,
        "cpc_target_yd_norm": cpc_yd_norm,
        "delta_cpc_target": cpc_yd - cpc0,
        "cpc_wrong_yd": cpc_wr,
        "cpc_wrong_yd_norm": cpc_wr_norm,
        "delta_cpc_wrong": cpc_wr - cpc0,
        "Y_D_target": Y_D_tgt.tolist(),
        "Y_D_wrong":  Y_D_wr.tolist(),
    }


def fold_bootstrap(
    values: np.ndarray,
    fold_ids: np.ndarray,
    n: int = 10000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple:
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


def compute_summary(results: list, fold_manifest: dict = None) -> dict:
    dt  = np.array([r["delta_cpc_target"] for r in results])
    dw  = np.array([r["delta_cpc_wrong"]  for r in results])
    fid = np.array([r["fold"]             for r in results])
    c0  = np.array([r["cpc_baseline"]     for r in results])
    cyd = np.array([r["cpc_target_yd"]   for r in results])
    cwr = np.array([r["cpc_wrong_yd"]    for r in results])

    n = len(results)
    ddof = 1 if n > 1 else 0

    ci_tl, ci_th, _ = fold_bootstrap(dt, fid)
    ci_wl, ci_wh, _ = fold_bootstrap(dw, fid)

    _, pt = safe_wilcoxon(dt, alternative="greater")
    _, pw = safe_wilcoxon(dw, alternative="greater")
    _, ps = safe_wilcoxon(dt - dw, alternative="greater")

    # --- Confirmatory Evaluation Guard (Requires exact 40 cities across Folds 2-5) ---
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
        c_c0 = c0[conf_mask]
        c_cyd = cyd[conf_mask]
        c_n = 40
        c_ddof = 1

        c_ci_tl, c_ci_th, _ = fold_bootstrap(c_dt, conf_fid)
        c_ci_wl, c_ci_wh, _ = fold_bootstrap(c_dw, conf_fid)
        _, c_pt = safe_wilcoxon(c_dt, alternative="greater")
        _, c_pw = safe_wilcoxon(c_dw, alternative="greater")
        _, c_ps = safe_wilcoxon(c_dt - c_dw, alternative="greater")

        conf_summary = {
            "status": "confirmatory_complete",
            "n_cities": c_n,
            "cpc_baseline_mean": float(c_c0.mean()),
            "cpc_baseline_std": float(c_c0.std(ddof=c_ddof)),
            "cpc_target_yd_mean": float(c_cyd.mean()),
            "cpc_target_yd_std": float(c_cyd.std(ddof=c_ddof)),
            "delta_cpc_target_mean": float(c_dt.mean()),
            "delta_cpc_target_median": float(np.median(c_dt)),
            "delta_cpc_target_std": float(c_dt.std(ddof=c_ddof)),
            "delta_cpc_target_ci_l": c_ci_tl,
            "delta_cpc_target_ci_h": c_ci_th,
            "n_positive_target": int((c_dt > 0).sum()),
            "p_wilcoxon_target": float(c_pt),
            "delta_cpc_wrong_mean": float(c_dw.mean()),
            "delta_cpc_wrong_median": float(np.median(c_dw)),
            "delta_cpc_wrong_std": float(c_dw.std(ddof=c_ddof)),
            "delta_cpc_wrong_ci_l": c_ci_wl,
            "delta_cpc_wrong_ci_h": c_ci_wh,
            "n_positive_wrong": int((c_dw > 0).sum()),
            "p_wilcoxon_wrong": float(c_pw),
            "p_specificity": float(c_ps),
            "ci_lower_bound_positive": bool(c_ci_tl > 0),
            "target_beats_wrong": bool(float(c_dt.mean()) > float(c_dw.mean())),
            "win_rate_target": f"{int((c_dt > 0).sum())}/{c_n}",
            "win_rate_wrong": f"{int((c_dw > 0).sum())}/{c_n}",
        }
    else:
        conf_summary = {
            "status": "not_available",
            "reason": f"Incomplete confirmatory set (observed {int(conf_mask.sum())}/40 required cities)",
        }

    # --- Per-fold summary breakdown ---
    per_fold = {}
    for f in sorted(set(fid)):
        idx = (fid == f)
        f_dt = dt[idx]
        f_dw = dw[idx]
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
            "delta_target_std": float(f_dt.std(ddof=f_ddof)),
            "delta_wrong_mean": float(f_dw.mean()),
            "delta_wrong_median": float(np.median(f_dw)),
            "n_positive_target": int((f_dt > 0).sum()),
            "n_positive_wrong": int((f_dw > 0).sum()),
            "win_rate_target": f"{int((f_dt > 0).sum())}/{f_n}",
            "best_epoch": fold_manifest.get(f, {}).get("best_epoch") if fold_manifest else None,
            "best_val_cpc": fold_manifest.get(f, {}).get("best_val_cpc") if fold_manifest else None,
        }

    return {
        "n_cities": n,
        "is_full_50_complete": is_full_50_complete,
        "is_confirmatory_complete": is_confirmatory_complete,
        "std_ddof": ddof,
        # Full Out-of-Fold
        "cpc_baseline_mean": float(c0.mean()),
        "cpc_baseline_std":  float(c0.std(ddof=ddof)),
        "cpc_target_yd_mean": float(cyd.mean()),
        "cpc_target_yd_std": float(cyd.std(ddof=ddof)),
        "delta_cpc_target_mean":   float(dt.mean()),
        "delta_cpc_target_median": float(np.median(dt)),
        "delta_cpc_target_std":    float(dt.std(ddof=ddof)),
        "delta_cpc_target_ci_l":   ci_tl,
        "delta_cpc_target_ci_h":   ci_th,
        "n_positive_target":       int((dt > 0).sum()),
        "p_wilcoxon_target":       float(pt),
        "cpc_wrong_yd_mean": float(cwr.mean()),
        "cpc_wrong_yd_std": float(cwr.std(ddof=ddof)),
        "delta_cpc_wrong_mean":    float(dw.mean()),
        "delta_cpc_wrong_median":  float(np.median(dw)),
        "delta_cpc_wrong_std":     float(dw.std(ddof=ddof)),
        "delta_cpc_wrong_ci_l":    ci_wl,
        "delta_cpc_wrong_ci_h":    ci_wh,
        "n_positive_wrong":        int((dw > 0).sum()),
        "p_wilcoxon_wrong":        float(pw),
        "p_specificity":           float(ps),
        "ci_lower_bound_positive": bool(ci_tl > 0),
        "target_beats_wrong":      bool(float(dt.mean()) > float(dw.mean())),
        "win_rate_target":         f"{int((dt > 0).sum())}/{n}",
        "win_rate_wrong":          f"{int((dw > 0).sum())}/{n}",
        # Primary Confirmatory Subgroup (Folds 2-5, only valid when complete)
        "confirmatory_folds_2_5":  conf_summary,
        # Per-fold breakdown
        "per_fold": per_fold,
        "fold_validation_manifest": fold_manifest or {},
    }



def write_tables(results: list, summary: dict):
    tdir = RESULTS_DIR / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    n  = summary["n_cities"]
    tl = summary["delta_cpc_target_ci_l"]
    th = summary["delta_cpc_target_ci_h"]
    wl = summary["delta_cpc_wrong_ci_l"]
    wh = summary["delta_cpc_wrong_ci_h"]
    c0m = summary["cpc_baseline_mean"]
    c0s = summary["cpc_baseline_std"]
    is_conf = summary.get("is_confirmatory_complete", False)
    is_full = summary.get("is_full_50_complete", False)

    run_type_str = "Full 50-City Protocol" if is_full else "Exploratory / Smoke Subset"

    lines = [
        f"# Table E1: Oracle Aggregated-Distance Existence Test ({run_type_str})",
        "",
        "> **Methodological Grounding**:",
        "> 1. **Oracle Upper Bound**: Y_D^{GT,+} is an outcome-derived oracle aggregate from target ground truth.",
        "> 2. **Evaluation Split Role**: Folds 2–5 (n=40) serve as the prospectively designated confirmatory test set; Fold 1 (n=10) serves as exploratory; all 50 cities provide full out-of-fold descriptive coverage.",
        "",
        f"**Execution Status**: {len(results)}/50 test cities evaluated | is_confirmatory_complete={is_conf} | is_full_50_complete={is_full}",
        f"**Protocol**: 5-fold stratified city CV (35 train / 5 val / 10 test per fold)",
        f"**Parameters**: K_move={K_MOVE} bins (pair-weighted quantile), q={Q_CALIB} (within-tolerance calibration, tolerance 10⁻⁵), max_epochs={EPOCHS}, patience={PATIENCE}, std_ddof={summary['std_ddof']}",
        "",
    ]

    # Section 1: Confirmatory Results on Folds 2-5 (strictly when complete)
    conf = summary.get("confirmatory_folds_2_5")
    if is_conf and conf and conf.get("status") == "confirmatory_complete":
        c_tl, c_th = conf["delta_cpc_target_ci_l"], conf["delta_cpc_target_ci_h"]
        c_wl, c_wh = conf["delta_cpc_wrong_ci_l"], conf["delta_cpc_wrong_ci_h"]
        lines.extend([
            "## E1-A: Confirmatory Test Set Outcomes (Prospectively Untouched, Folds 2–5, n=40)",
            "",
            "| Condition | Interzonal CPC (Mean ± SD) | Mean ΔCPC | Median ΔCPC | 95% Fold-Stratified Bootstrap CI | Win Rate | Wilcoxon p |",
            "|---|---|---|---|---|---|---|",
            f"| Zero-Shot (M₀) | {conf['cpc_baseline_mean']:.4f} ± {conf['cpc_baseline_std']:.4f} | — | — | — | — | — |",
            (f"| + Oracle Y_D (target) | {conf['cpc_target_yd_mean']:.4f} ± {conf['cpc_target_yd_std']:.4f} | "
             f"+{conf['delta_cpc_target_mean']:.4f} | +{conf['delta_cpc_target_median']:.4f} | "
             f"[{c_tl:+.4f}, {c_th:+.4f}] | {conf['win_rate_target']} | {conf['p_wilcoxon_target']:.2e} |"),
            (f"| + Oracle Y_D (wrong donor) | {conf['delta_cpc_wrong_mean'] + conf['cpc_baseline_mean']:.4f} ± {conf['delta_cpc_wrong_std']:.4f} | "
             f"{conf['delta_cpc_wrong_mean']:+.4f} | {conf['delta_cpc_wrong_median']:+.4f} | "
             f"[{c_wl:+.4f}, {c_wh:+.4f}] | {conf['win_rate_wrong']} | {conf['p_wilcoxon_wrong']:.2e} |"),
            "",
        ])
    else:
        lines.extend([
            "## E1-A: Confirmatory Test Set Outcomes (Folds 2–5, n=40)",
            "",
            f"> *Status: NOT AVAILABLE ({len(results)}/50 cities run; Confirmatory evaluation strictly requires complete 40 test cities across Folds 2–5).* ",
            "",
        ])

    # Section 2: Observed Subset / Full Out-of-Fold Outcomes
    cov_label = "Full Out-of-Fold Descriptive Coverage (50 Cities, Folds 1–5)" if is_full else f"Observed Test Subset ({n} Cities)"
    lines.extend([
        f"## E1-B: {cov_label}",
        "",
        "| Condition | Interzonal CPC (Mean ± SD) | Mean ΔCPC | Median ΔCPC | 95% Fold-Stratified Bootstrap CI | Win Rate | Wilcoxon p |",
        "|---|---|---|---|---|---|---|",
        f"| Zero-Shot (M₀) | {c0m:.4f} ± {c0s:.4f} | — | — | — | — | — |",
        (f"| + Oracle Y_D (target) | {summary['cpc_target_yd_mean']:.4f} ± {summary['cpc_target_yd_std']:.4f} | "
         f"+{summary['delta_cpc_target_mean']:.4f} | +{summary['delta_cpc_target_median']:.4f} | "
         f"[{tl:+.4f}, {th:+.4f}] | {summary['win_rate_target']} | {summary['p_wilcoxon_target']:.2e} |"),
        (f"| + Oracle Y_D (wrong donor) | {summary['cpc_wrong_yd_mean']:.4f} ± {summary['cpc_wrong_yd_std']:.4f} | "
         f"{summary['delta_cpc_wrong_mean']:+.4f} | {summary['delta_cpc_wrong_median']:+.4f} | "
         f"[{wl:+.4f}, {wh:+.4f}] | {summary['win_rate_wrong']} | {summary['p_wilcoxon_wrong']:.2e} |"),
        "",
        "## E1-C: Per-Fold Independent Training & Evaluation Breakdown",
        "",
        "| Fold | Role | Test Cities Evaluated | Best Val Epoch | Best Val CPC | M₀ CPC | +Target Y_D CPC | Mean ΔCPC (Target) | Win Rate (Target) | Mean ΔCPC (Wrong) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ])

    for f_key, pf in summary.get("per_fold", {}).items():
        f_num = f_key.replace("fold_", "")
        b_ep = pf.get("best_epoch", "—")
        b_vc = f"{pf['best_val_cpc']:.4f}" if pf.get("best_val_cpc") is not None else "—"
        role = "Exploratory" if f_num == "1" else "Confirmatory"
        lines.append(
            f"| Fold {f_num} | {role} | {pf['n_cities']} | {b_ep} | {b_vc} | "
            f"{pf['cpc_baseline_mean']:.4f} | {pf['cpc_baseline_mean'] + pf['delta_target_mean']:.4f} | "
            f"{pf['delta_target_mean']:+.4f} | {pf['win_rate_target']} | {pf['delta_wrong_mean']:+.4f} |"
        )

    # Section 3: Acceptance Criteria (Dynamic gating)
    if is_conf and conf and conf.get("status") == "confirmatory_complete":
        e_tl, e_th = conf["delta_cpc_target_ci_l"], conf["delta_cpc_target_ci_h"]
        lines.extend([
            "",
            "## Acceptance Criteria Verification (Confirmatory Folds 2–5, n=40)",
            "",
            "| Criterion | Required Condition | Observed Value | Verdict |",
            "|---|---|---|---|",
            f"| Confirmatory CI Lower Bound | CI_lower > 0 | [{e_tl:+.4f}, {e_th:+.4f}] | {'✓ PASS' if conf['ci_lower_bound_positive'] else '✗ FAIL'} |",
            f"| Specificity Superiority | Mean ΔCPC_target > Mean ΔCPC_wrong | {conf['delta_cpc_target_mean']:+.4f} vs {conf['delta_cpc_wrong_mean']:+.4f} | {'✓ PASS' if conf['target_beats_wrong'] else '✗ FAIL'} |",
            f"| Specificity Significance | Paired Wilcoxon p < 0.05 | p = {conf['p_specificity']:.2e} | {'✓ PASS' if conf['p_specificity'] < 0.05 else '✗ FAIL'} |",
            f"| City-level Consistency | Win rate > 70% (>28/40) | {conf['win_rate_target']} | {'✓ PASS' if int(conf['win_rate_target'].split('/')[0]) >= 28 else '✗ FAIL'} |",
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

    hdr = "| City | Fold | n_pairs | CPC₀ | CPC_target | ΔCPC_target | CPC_wrong | ΔCPC_wrong | Donor City |"


    sep = "|---|---|---|---|---|---|---|---|---|"
    rows = [hdr, sep]
    for r in sorted(results, key=lambda x: x["city"]):
        rows.append(
            f"| {r['city']} | {r['fold']} | {r['n_inter_pairs']} | "
            f"{r['cpc_baseline']:.4f} | {r['cpc_target_yd']:.4f} | "
            f"{r['delta_cpc_target']:+.4f} | {r['cpc_wrong_yd']:.4f} | "
            f"{r['delta_cpc_wrong']:+.4f} | {r['donor_city']} |"
        )
    (tdir / "e1_per_city.md").write_text("# E1: Complete Per-City Breakdown (50 Cities)\n\n" + "\n".join(rows) + "\n", encoding="utf-8")
    print(f"  Generated Markdown tables in {tdir}")


def run_e1(smoke: bool = False, smoke_cities: list = None, device_str: str = "cpu"):
    t0 = time.time()
    device = torch.device(device_str)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    splits = generate_35_5_10_splits(DATA_ROOT)
    all_results = []
    fold_manifest = {}

    for fold_id, split in splits.items():
        train35 = split["train"]   # 35 cities
        val5    = split["val"]     # 5 cities
        test10  = sorted(split["test"])  # 10 cities

        run_test = test10
        if smoke:
            run_test = [c for c in (smoke_cities or ["Portland", "Denver"]) if c in test10]
            if not run_test:
                continue

        print(f"\n{'='*60}\nFOLD {fold_id}: train={len(train35)}, val={len(val5)}, test={len(run_test)}\n{'='*60}")

        # Step 1: Compute bin edges from 35 train cities
        print(f"  Computing K_move={K_MOVE} pair-weighted bin edges...")
        bin_edges, K_active = compute_kbin_edges(train35, K=K_MOVE, data_root=DATA_ROOT)
        if K_active != K_MOVE:
            raise RuntimeError(f"E1 requires exactly {K_MOVE} active bins, got {K_active}")
        print(f"  K_active={K_active} (verified {K_MOVE}-bin strict), internal edges (km): {np.round(bin_edges[1:-1], 2).tolist()}")

        # Step 2: Train backbone with validation early stopping
        print(f"  Training backbone (max_epochs={EPOCHS}, patience={PATIENCE})...")
        model, scaler, train_info = train_zero_shot_model(
            train_city_names=train35,
            data_root=DATA_ROOT,
            epochs=EPOCHS,
            device_str=device_str,
            verbose=True,
            val_city_names=val5,
            patience=PATIENCE,
            return_info=True,
        )

        fold_manifest[fold_id] = {
            "fold_id": fold_id,
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
            "val_cpc_history": train_info["val_cpc_history"],
        }
        print(f"  Backbone frozen (best epoch={train_info['best_epoch']}, best val CPC={train_info['best_val_cpc']:.4f}).")

        # Step 3: Evaluate test cities
        for city in run_test:
            donor = get_donor_city(city, test10)
            print(f"  [{fold_id}] {city} (donor: {donor})")
            res = run_city(city, model, scaler, bin_edges, K_active, donor, fold_id, device)
            all_results.append(res)
            print(f"    dCPC_target={res['delta_cpc_target']:+.4f}  dCPC_wrong={res['delta_cpc_wrong']:+.4f}  n_inter={res['n_inter_pairs']}")

    # Step 4: Save raw results and validation manifest
    (RESULTS_DIR / "e1_per_city_results.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    (RESULTS_DIR / "e1_validation_manifest.json").write_text(json.dumps(fold_manifest, indent=2), encoding="utf-8")
    print(f"\nSaved {len(all_results)} city results and validation manifest.")

    if len(all_results) < 2:
        print("Too few cities for statistics (smoke test).")
        return all_results, None

    # Step 5: Compute statistical summary with ddof=1 and Wilcoxon protections
    summary = compute_summary(all_results, fold_manifest=fold_manifest)
    (RESULTS_DIR / "e1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_tables(all_results, summary)

    elapsed = time.time() - t0
    is_conf = summary.get("is_confirmatory_complete", False)
    is_full = summary.get("is_full_50_complete", False)
    conf = summary.get("confirmatory_folds_2_5", {})

    print(f"\n{'='*60}\nE1 Complete in {elapsed:.0f}s ({'Full 50-City Run' if is_full else f'Subset Run with {len(all_results)} cities'})")
    print(f"  ΔCPC_target: mean={summary['delta_cpc_target_mean']:+.4f} (median={summary['delta_cpc_target_median']:+.4f}, std[ddof=1]={summary['delta_cpc_target_std']:.4f})")
    print(f"  95% Fold-stratified CI: [{summary['delta_cpc_target_ci_l']:+.4f}, {summary['delta_cpc_target_ci_h']:+.4f}]")
    print(f"  Win rate: {summary['win_rate_target']} cities | Wilcoxon p={summary['p_wilcoxon_target']:.2e}")
    print(f"  Placebo (wrong donor): mean={summary['delta_cpc_wrong_mean']:+.4f} | Win rate: {summary['win_rate_wrong']}")
    print(f"  Specificity Wilcoxon p: {summary['p_specificity']:.2e}")

    if is_conf and conf.get("status") == "confirmatory_complete":
        print(
            f"  Confirmatory Criteria (Folds 2-5, n=40): "
            f"CI_lower>0: {'✓ PASS' if conf['ci_lower_bound_positive'] else '✗ FAIL'} | "
            f"Target>Wrong: {'✓ PASS' if conf['target_beats_wrong'] else '✗ FAIL'} | "
            f"Win Rate: {conf['win_rate_target']}"
        )
    else:
        print(f"  Confirmatory Criteria (Folds 2-5): NOT AVAILABLE ({len(all_results)}/50 cities run; requires complete 40 cities across Folds 2-5).")
    print(f"{'='*60}")

    return all_results, summary



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke",  action="store_true", help="Run smoke test on 2 test cities (Portland, Denver)")
    parser.add_argument("--cities", nargs="+", default=None, help="Custom list of test cities to run")
    parser.add_argument("--device", default="cpu", help="PyTorch device")
    args = parser.parse_args()
    run_e1(smoke=args.smoke, smoke_cities=args.cities, device_str=args.device)
