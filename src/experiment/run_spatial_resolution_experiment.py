"""
Spatial Resolution Experiment (Origin County-Level vs. City-Level Calibration).

Evaluates whether providing finer spatial resolution in the aggregate distance distribution
(Y_D^(county) conditioned on origin county vs. macro city-wide Y_D^(city)) enhances mobility
prediction accuracy in heterogeneous metropolitan areas.

Key Estimands:
    1. City-Level Target Gain:        Δ_city        = CPC(M_city) - CPC(M0)
    2. County-Level Target Gain:      Δ_county      = CPC(M_county) - CPC(M0)
    3. Spatial Resolution Gain:       Δ_resolution  = CPC(M_county) - CPC(M_city)
    4. Specificity Gains:             Δ_spec_city   = CPC(M_city) - CPC(M_wrong)
                                      Δ_spec_county = CPC(M_county) - CPC(M_wrong)

Invariance Properties:
    - For single-county cities (n=45), M_county ≡ M_city, so Δ_resolution ≡ 0.0000 (Sanity Check).
    - For multi-county cities (n=5: Atlanta, Dallas, Kansas City, New York, Tulsa),
      heterogeneous origin distributions allow fine-grained spatial adaptation (Δ_resolution >= 0).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import load_city, preload_all_cities
from src.data.city_splits import load_splits_manifest_v2
from src.data.yd_extractor import compute_kbin_edges, extract_yd_kbins, extract_yd_kbins_grouped
from src.calibration.bin_calibration import calibrate_kbins, calibrate_kbins_grouped
from src.models.zero_shot_model import ZeroShotODModel
from src.training.train import (
    train_zero_shot_model,
    infer_zero_shot,
    build_radius_graph,
    load_checkpoint,
    save_checkpoint
)
from src.training.evaluate import compute_cpc_pair, compute_cpc_norm_pair

# Output directories & constants
RESULTS_DIR = PROJECT_ROOT / "results" / "spatial_resolution"
TABLES_DIR = RESULTS_DIR / "tables"
DATA_ROOT = "data"
K_MOVE = 8
Q_CALIB = 1.0
EPOCHS = 200
PATIENCE = 15
MIN_DELTA = 1e-4
DEFAULT_SEED = 2024


def log_msg(msg: str = "", print_to_console: bool = True):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {msg}" if msg else ""
    if print_to_console:
        print(formatted if formatted else "", flush=True)
    LOG_FILE = RESULTS_DIR / "spatial_resolution.log"
    try:
        with open(LOG_FILE, "a", encoding="utf-8", errors="replace") as f:
            f.write(formatted + "\n")
    except Exception:
        pass


def safe_wilcoxon(diff: np.ndarray, alternative: str = "greater") -> tuple[float, float]:
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


def fold_bootstrap(
    values: np.ndarray,
    fold_ids: np.ndarray,
    n: int = 10000,
    seed: int = 2024,
    alpha: float = 0.05,
) -> tuple[float, float]:
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
        return 0.0, 0.0
    return float(np.percentile(boot, 100 * (alpha / 2))), float(np.percentile(boot, 100 * (1 - alpha / 2)))


def run_spatial_resolution_city(
    city: str,
    model: torch.nn.Module,
    scaler: object,
    bin_edges: np.ndarray,
    test_cities: list[str],
    fold_id: int,
    device: torch.device,
    test_yd_cache: dict[str, np.ndarray],
) -> dict:
    t_start = time.time()
    
    cd = load_city(city, data_root=DATA_ROOT, feature_scaler=scaler)
    ei, ed = build_radius_graph(cd.lon_lat, radius_km=5.0)
    dist_km = np.expm1(cd.pair_distance.numpy())
    inter_mask = (cd.pair_o_idx.numpy() != cd.pair_d_idx.numpy()) & (dist_km > 0.0)
    t_gt = cd.pair_trips.numpy().astype(np.float64)
    
    # Extract county grouping from meta.csv
    meta_df = pd.read_csv(Path(DATA_ROOT) / city / "meta.csv")
    meta_df["county_id"] = meta_df["state_fips"].astype(str).str.zfill(2) + meta_df["county_fips"].astype(str).str.zfill(3)
    tract_to_county = dict(zip(meta_df["idx"], meta_df["county_id"]))
    pair_county_idx = np.array([tract_to_county[i] for i in cd.pair_o_idx.numpy()])
    
    unique_counties = sorted(list(set(pair_county_idx)))
    n_counties = len(unique_counties)
    
    # 1. Condition A: Zero-Shot Forward Pass (M0)
    T0 = infer_zero_shot(model, cd, ei, ed, device=device)
    t0_np = T0.numpy().astype(np.float64)
    cpc_0 = compute_cpc_pair(t_gt[inter_mask], t0_np[inter_mask])
    
    # 2. Condition B: City-Level Calibration (M_city)
    yd_city = test_yd_cache[city]
    t_city = calibrate_kbins(
        t0_np=t0_np,
        dist_km=dist_km,
        inter_mask=inter_mask,
        yd_target=yd_city,
        bin_edges=bin_edges,
        q=Q_CALIB,
    )
    cpc_city = compute_cpc_pair(t_gt[inter_mask], t_city[inter_mask])
    
    # 3. Condition C: County-Level Calibration (M_county)
    yd_county_dict = extract_yd_kbins_grouped(
        dist_km=dist_km,
        trips=t_gt,
        bin_edges=bin_edges,
        inter_mask=inter_mask,
        pair_group_idx=pair_county_idx,
    )
    t_county = calibrate_kbins_grouped(
        t0_np=t0_np,
        dist_km=dist_km,
        inter_mask=inter_mask,
        yd_target_dict=yd_county_dict,
        bin_edges=bin_edges,
        pair_group_idx=pair_county_idx,
        q=Q_CALIB,
    )
    cpc_county = compute_cpc_pair(t_gt[inter_mask], t_county[inter_mask])
    
    # 4. Condition D: Multi-Donor Wrong Placebo Y_D (9 wrong donors)
    wrong_cpcs = []
    other_donors = [d for d in test_cities if d != city]
    for donor in other_donors:
        yd_donor = test_yd_cache[donor]
        t_wrong_d = calibrate_kbins(
            t0_np=t0_np,
            dist_km=dist_km,
            inter_mask=inter_mask,
            yd_target=yd_donor,
            bin_edges=bin_edges,
            q=Q_CALIB,
        )
        wrong_cpcs.append(compute_cpc_pair(t_gt[inter_mask], t_wrong_d[inter_mask]))
    cpc_wrong = float(np.mean(wrong_cpcs))
    
    elapsed = time.time() - t_start
    
    return {
        "city": city,
        "fold": fold_id,
        "n_counties": n_counties,
        "is_multi_county": bool(n_counties > 1),
        "county_ids": unique_counties,
        "cpc_baseline": float(cpc_0),
        "cpc_city": float(cpc_city),
        "cpc_county": float(cpc_county),
        "cpc_wrong": float(cpc_wrong),
        "delta_cpc_city": float(cpc_city - cpc_0),
        "delta_cpc_county": float(cpc_county - cpc_0),
        "delta_cpc_resolution": float(cpc_county - cpc_city),
        "delta_cpc_spec_city": float(cpc_city - cpc_wrong),
        "delta_cpc_spec_county": float(cpc_county - cpc_wrong),
        "elapsed_sec": float(elapsed),
    }


def compute_resolution_summary(results: list[dict], bootstrap_seed: int = DEFAULT_SEED) -> dict:
    df = pd.DataFrame(results)
    
    # Global metrics
    fid = df["fold"].values
    d_res = df["delta_cpc_resolution"].values
    d_city = df["delta_cpc_city"].values
    d_county = df["delta_cpc_county"].values
    d_spec_city = df["delta_cpc_spec_city"].values
    d_spec_county = df["delta_cpc_spec_county"].values
    
    ci_res_l, ci_res_h = fold_bootstrap(d_res, fid, seed=bootstrap_seed)
    ci_city_l, ci_city_h = fold_bootstrap(d_city, fid, seed=bootstrap_seed)
    ci_county_l, ci_county_h = fold_bootstrap(d_county, fid, seed=bootstrap_seed)
    ci_scity_l, ci_scity_h = fold_bootstrap(d_spec_city, fid, seed=bootstrap_seed)
    ci_scounty_l, ci_scounty_h = fold_bootstrap(d_spec_county, fid, seed=bootstrap_seed)
    
    _, p_res = safe_wilcoxon(d_res, alternative="greater")
    _, p_scity = safe_wilcoxon(d_spec_city, alternative="greater")
    _, p_scounty = safe_wilcoxon(d_spec_county, alternative="greater")
    
    # Subgroup: Multi-County Cities (n=5)
    multi_df = df[df["is_multi_county"]]
    single_df = df[~df["is_multi_county"]]
    
    return {
        "n_total_cities": len(df),
        "n_multi_county_cities": len(multi_df),
        "n_single_county_cities": len(single_df),
        "pooled_50": {
            "cpc_baseline_mean": float(df["cpc_baseline"].mean()),
            "cpc_city_mean": float(df["cpc_city"].mean()),
            "cpc_county_mean": float(df["cpc_county"].mean()),
            "cpc_wrong_mean": float(df["cpc_wrong"].mean()),
            "delta_city_mean": float(d_city.mean()),
            "delta_city_ci": [ci_city_l, ci_city_h],
            "delta_county_mean": float(d_county.mean()),
            "delta_county_ci": [ci_county_l, ci_county_h],
            "delta_resolution_mean": float(d_res.mean()),
            "delta_resolution_median": float(np.median(d_res)),
            "delta_resolution_ci": [ci_res_l, ci_res_h],
            "delta_spec_city_mean": float(d_spec_city.mean()),
            "delta_spec_city_ci": [ci_scity_l, ci_scity_h],
            "delta_spec_county_mean": float(d_spec_county.mean()),
            "delta_spec_county_ci": [ci_scounty_l, ci_scounty_h],
            "win_rate_resolution": f"{(d_res > 0).sum()}/{len(df)}",
            "win_rate_spec_county": f"{(d_spec_county > 0).sum()}/{len(df)}",
            "wilcoxon_p_resolution": float(p_res),
            "wilcoxon_p_spec_county": float(p_scounty),
        },
        "multi_county_subset": {
            "cities": multi_df["city"].tolist(),
            "cpc_baseline_mean": float(multi_df["cpc_baseline"].mean()),
            "cpc_city_mean": float(multi_df["cpc_city"].mean()),
            "cpc_county_mean": float(multi_df["cpc_county"].mean()),
            "delta_city_mean": float(multi_df["delta_cpc_city"].mean()),
            "delta_county_mean": float(multi_df["delta_cpc_county"].mean()),
            "delta_resolution_mean": float(multi_df["delta_cpc_resolution"].mean()),
            "delta_resolution_max": float(multi_df["delta_cpc_resolution"].max()),
            "win_rate_resolution": f"{(multi_df['delta_cpc_resolution'] > 0).sum()}/{len(multi_df)}",
        },
        "single_county_subset": {
            "n_cities": len(single_df),
            "delta_resolution_mean": float(single_df["delta_cpc_resolution"].mean()),
            "delta_resolution_max": float(single_df["delta_cpc_resolution"].max()),
            "exact_zero_invariant": bool(np.allclose(single_df["delta_cpc_resolution"].values, 0.0, atol=1e-6)),
        }
    }


def write_resolution_tables(results: list[dict], summary: dict):
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Main Table
    p50 = summary["pooled_50"]
    mc = summary["multi_county_subset"]
    sc = summary["single_county_subset"]
    
    main_md = f"""# Table S1: Spatial Resolution Analysis (County-Level vs. City-Level Calibration)

> **Research Question**: Does conditioning the aggregated distance distribution $Y_D$ on origin counties ($M_{{\\text{{county}}}}$) improve zero-shot flow prediction over city-wide macro distributions ($M_{{\\text{{city}}}}$)?
> **Dataset**: 50 US Metropolitan Areas (45 Single-County, 5 Multi-County) under 5-Fold Stratified CV.
> **Calibration Protocol**: $K_{{\\text{{move}}}}=8$ quantile bins, $q=1.0$, within-tolerance distribution matching.

---

## S1-A: Overall Comparative Performance ($n=50$ Cities)

| Condition / Model | Mean Interzonal CPC | Mean Gain vs $M_0$ (Δ) | 95% Bootstrap CI | Specificity Gain vs Placebo | Specificity Win Rate |
|---|:---:|:---:|:---:|:---:|:---:|
| **Zero-Shot Baseline ($M_0$)** | {p50['cpc_baseline_mean']:.4f} | — | — | — | — |
| **+ City-Level Target $Y_D$ ($M_{{\\text{{city}}}}$)** | {p50['cpc_city_mean']:.4f} | {p50['delta_city_mean']:+.4f} | [{p50['delta_city_ci'][0]:+.4f}, {p50['delta_city_ci'][1]:+.4f}] | {p50['delta_spec_city_mean']:+.4f} | {p50['win_rate_spec_county']} |
| **+ County-Level Target $Y_D$ ($M_{{\\text{{county}}}}$)** | **{p50['cpc_county_mean']:.4f}** | **{p50['delta_county_mean']:+.4f}** | **[{p50['delta_county_ci'][0]:+.4f}, {p50['delta_county_ci'][1]:+.4f}]** | **{p50['delta_spec_county_mean']:+.4f}** | **{p50['win_rate_spec_county']}** |
| **Placebo Control ($M_{{\\text{{wrong}}}}$ 9-Donor Avg)** | {p50['cpc_wrong_mean']:.4f} | {p50['cpc_wrong_mean'] - p50['cpc_baseline_mean']:+.4f} | — | — | 0/50 |

---

## S1-B: Multi-County Metropolitan Focus ($n=5$ Heterogeneous Cities)

In multi-county metropolitan areas, distinct origin counties exhibit heterogeneous localized trip distributions.

| City | Origin Counties | Zero-Shot $M_0$ | City-Level $M_{{\\text{{city}}}}$ | County-Level $M_{{\\text{{county}}}}$ | Resolution Gain ($\\Delta_{{\\text{{res}}}}$) | Placebo $M_{{\\text{{wrong}}}}$ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
"""
    for r in sorted([r for r in results if r["is_multi_county"]], key=lambda x: x["delta_cpc_resolution"], reverse=True):
        main_md += f"| **{r['city']}** | {r['n_counties']} counties | {r['cpc_baseline']:.4f} | {r['cpc_city']:.4f} | **{r['cpc_county']:.4f}** | **{r['delta_cpc_resolution']:+.4f}** | {r['cpc_wrong']:.4f} |\n"
        
    main_md += f"""
**Multi-County Average ($n=5$)**:
- Mean Zero-Shot $M_0$: {mc['cpc_baseline_mean']:.4f}
- Mean City-Level $M_{{\\text{{city}}}}$: {mc['cpc_city_mean']:.4f} (Δ = {mc['delta_city_mean']:+.4f})
- Mean County-Level $M_{{\\text{{county}}}}$: **{mc['cpc_county_mean']:.4f}** (Δ = **{mc['delta_county_mean']:+.4f}**)
- **Mean Spatial Resolution Gain ($\\Delta_{{\\text{{res}}}}$)**: **{mc['delta_resolution_mean']:+.4f}** (Max: **{mc['delta_resolution_max']:+.4f}**)
- **Resolution Improvement Rate**: **{mc['win_rate_resolution']}**

---

## S1-C: Single-County Sanity Invariance ($n=45$ Single-County Cities)

For single-county cities, all tracts belong to the same origin county, meaning $M_{{\\text{{county}}}} \\equiv M_{{\\text{{city}}}}$ by definition.
- **Observed Mean $\\Delta_{{\\text{{resolution}}}}$**: {sc['delta_resolution_mean']:.6f}
- **Exact Mathematical Invariance**: {'✓ VERIFIED' if sc['exact_zero_invariant'] else '✗ FAILED'}
"""
    (TABLES_DIR / "spatial_resolution_main_table.md").write_text(main_md, encoding="utf-8")

    # 2. Per-City Breakdown Table
    rows = [
        "| City | Fold | Counties | Multi-County? | $M_0$ CPC | $M_{\\text{city}}$ CPC | $M_{\\text{county}}$ CPC | $\\Delta_{\\text{resolution}}$ | $M_{\\text{wrong}}$ | $\\Delta_{\\text{spec, county}}$ |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
    ]
    for r in sorted(results, key=lambda x: (not x["is_multi_county"], x["city"])):
        mc_flag = "Yes" if r["is_multi_county"] else "No"
        rows.append(
            f"| {r['city']} | {r['fold']} | {r['n_counties']} | {mc_flag} | "
            f"{r['cpc_baseline']:.4f} | {r['cpc_city']:.4f} | {r['cpc_county']:.4f} | "
            f"**{r['delta_cpc_resolution']:+.4f}** | {r['cpc_wrong']:.4f} | {r['delta_cpc_spec_county']:+.4f} |"
        )
    (TABLES_DIR / "spatial_resolution_per_city.md").write_text("# Complete Spatial Resolution Breakdown (50 Cities)\n\n" + "\n".join(rows) + "\n", encoding="utf-8")


def run_spatial_resolution_experiment(device_str: str = "cpu", seed: int = DEFAULT_SEED, smoke: bool = False):
    t_global_start = time.time()
    device = torch.device(device_str)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    
    log_msg("=" * 75)
    log_msg("SPATIAL RESOLUTION EXPERIMENT: ORIGIN COUNTY-LEVEL VS CITY-LEVEL CALIBRATION")
    log_msg("=" * 75)
    log_msg(f"  Configuration: K={K_MOVE} bins, q={Q_CALIB}, Seed={seed}, Device={device_str}")
    
    MANIFEST_PATH = PROJECT_ROOT / "results" / "e1" / "splits_manifest_v2.json"
    splits = load_splits_manifest_v2(str(MANIFEST_PATH), data_root=DATA_ROOT)
    
    log_msg("  Preloading datasets & spatial graphs...")
    preload_all_cities(data_root=DATA_ROOT, build_graphs=True, radius_km=5.0)
    
    all_results = []
    
    for fold_id, split in splits.items():
        t_fold_start = time.time()
        train35 = split["train"]
        val5 = split["val"]
        test10 = sorted(split["test"])
        
        if smoke:
            test10 = [c for c in ["Dallas", "Atlanta", "Denver", "Portland"] if c in test10]
            if not test10:
                continue
                
        log_msg("-" * 75)
        log_msg(f">>> [FOLD {fold_id}/5] Training Backbone & Evaluating Spatial Resolution...")
        log_msg("-" * 75)
        
        # 1. Compute Bin Edges
        bin_edges, K_active = compute_kbin_edges(train35, K=K_MOVE, data_root=DATA_ROOT)
        
        # 2. Train / Load Backbone Model
        fold_seed = seed + fold_id
        ckpt_path = RESULTS_DIR / "checkpoints" / f"fold{fold_id}_seed{fold_seed}.pt"
        
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
            seed=fold_seed,
            checkpoint_path=ckpt_path,
        )
        
        # 3. Precompute City-Level Y_D Oracles for all test cities in fold
        test_yd_cache = {}
        for t_city in test10:
            cd_t = load_city(t_city, data_root=DATA_ROOT, feature_scaler=scaler)
            dist_t = np.expm1(cd_t.pair_distance.numpy())
            inter_t = (cd_t.pair_o_idx.numpy() != cd_t.pair_d_idx.numpy()) & (dist_t > 0.0)
            t_gt_t = cd_t.pair_trips.numpy().astype(np.float64)
            test_yd_cache[t_city] = extract_yd_kbins(dist_t, t_gt_t, bin_edges, inter_t)
            
        # 4. Evaluate each held-out test city
        for city in test10:
            res = run_spatial_resolution_city(
                city=city,
                model=model,
                scaler=scaler,
                bin_edges=bin_edges,
                test_cities=test10,
                fold_id=fold_id,
                device=device,
                test_yd_cache=test_yd_cache,
            )
            all_results.append(res)
            
            mc_str = f" [Multi-County: {res['n_counties']} counties]" if res["is_multi_county"] else ""
            log_msg(
                f"  [{city:<16}] M0={res['cpc_baseline']:.4f} -> "
                f"M_city={res['cpc_city']:.4f} (d={res['delta_cpc_city']:+.4f}) -> "
                f"M_county={res['cpc_county']:.4f} (d={res['delta_cpc_county']:+.4f}) | "
                f"dRes={res['delta_cpc_resolution']:+.4f}{mc_str}"
            )
            
        t_fold_elapsed = time.time() - t_fold_start
        log_msg(f"  [Fold {fold_id} Complete] Elapsed time: {t_fold_elapsed:.1f}s")
        
    # Synthesize Summary & Tables
    summary = compute_resolution_summary(all_results, bootstrap_seed=seed)
    
    (RESULTS_DIR / "spatial_resolution_per_city.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    (RESULTS_DIR / "spatial_resolution_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    
    write_resolution_tables(all_results, summary)
    
    t_global_elapsed = time.time() - t_global_start
    log_msg("=" * 75)
    log_msg(f"SPATIAL RESOLUTION EXPERIMENT COMPLETED ({t_global_elapsed:.1f}s)")
    log_msg("=" * 75)
    p50 = summary["pooled_50"]
    mc = summary["multi_county_subset"]
    log_msg(f"  Total Cities Evaluated: {len(all_results)}/50")
    log_msg(f"  City-Level Gain (dCPC): mean = {p50['delta_city_mean']:+.4f} (CI: [{p50['delta_city_ci'][0]:+.4f}, {p50['delta_city_ci'][1]:+.4f}])")
    log_msg(f"  County-Level Gain (dCPC): mean = {p50['delta_county_mean']:+.4f} (CI: [{p50['delta_county_ci'][0]:+.4f}, {p50['delta_county_ci'][1]:+.4f}])")
    log_msg(f"  Multi-County Cities (n=5) Spatial Resolution Gain: mean = {mc['delta_resolution_mean']:+.4f} (Max: {mc['delta_resolution_max']:+.4f})")
    log_msg(f"  County-Level Specificity Win Rate: {p50['win_rate_spec_county']} (Wilcoxon p = {p50['wilcoxon_p_spec_county']:.2e})")
    log_msg("=" * 75)
    
    return all_results, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spatial Resolution Experiment: County vs City Calibration")
    parser.add_argument("--smoke", action="store_true", help="Run quick smoke test on subset of cities")
    parser.add_argument("--device", default="cpu", help="PyTorch device (cpu/cuda)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    args = parser.parse_args()
    
    run_spatial_resolution_experiment(device_str=args.device, seed=args.seed, smoke=args.smoke)
