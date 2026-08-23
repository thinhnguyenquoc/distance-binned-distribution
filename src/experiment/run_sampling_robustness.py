"""
Empirical Sampling Robustness Experiment (Task 2).
Measures empirical distance distribution error TV(Y_D^(m), Y_D^full) as a function of
observed sample size m in {100, 250, 500, 1000, 2500, 5000, 10000, 50000, 100000, inf},
and evaluates the resulting OD reconstruction benefit delta_CPC.
"""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import sys
import json
import hashlib
import argparse
import datetime
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import logging

from scipy.stats import spearmanr, wilcoxon
from scipy.spatial.distance import jensenshannon

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data.dataset import load_city, load_raw_city
from src.data.urban_graph import build_radius_graph
from src.training.train import load_checkpoint
from src.training.evaluate import compute_cpc_pair
from src.data.yd_extractor import compute_kbin_edges, extract_yd_kbins
from src.experiment.run_5fold import generate_35_5_10_splits
from src.experiment.run_experiment import infer_zero_shot


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


def get_stable_seed(base_seed: int, fold: int, city: str, m_val: Any, replicate_id: int) -> int:
    s = f"{base_seed}_{fold}_{city}_{m_val}_{replicate_id}"
    return int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % (2**32)


def sample_empirical_yd(yd_full: np.ndarray, m: float, seed: int) -> np.ndarray:
    """Draws m trips from categorical distribution over distance bins."""
    if np.isinf(m) or m is None:
        return yd_full.copy()
    m_int = int(m)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(m_int, yd_full)
    yd_m = counts.astype(np.float64) / float(m_int)
    return yd_m


def fold_stratified_bootstrap(city_df: pd.DataFrame, metric_col: str, m_val: float, confirmatory_folds: List[int], n_boot: int = 10000, seed: int = 42) -> Tuple[float, float]:
    rng = np.random.RandomState(seed)
    
    vals: Dict[int, np.ndarray] = {}
    for f in confirmatory_folds:
        mask = (city_df.fold == f) & (city_df.sample_m == m_val)
        vals[f] = city_df[mask][metric_col].values
        assert len(vals[f]) == 10, f"Expected 10 cities for fold {f}, got {len(vals[f])}"
        
    f_samples = [vals[f][rng.randint(0, 10, size=(n_boot, 10))] for f in confirmatory_folds]
    all_samples = np.hstack(f_samples)
    boot_means = np.mean(all_samples, axis=1)
        
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def fast_cal_metrics(
    yd_tgt: np.ndarray, 
    compute_spearman: bool, 
    N_hat: float, 
    K: int, 
    active: np.ndarray, 
    Y_hat: np.ndarray, 
    t0_inter: np.ndarray, 
    bin_idx: np.ndarray, 
    t_true_inter: np.ndarray, 
    cpc_m0: float, 
    yd_target: np.ndarray,
    inv_sum_denom: float,
    inv_N: float,
    t_cal_buf: np.ndarray,
    diff_buf: np.ndarray
) -> Tuple[float, float, float, float, float, float, Dict[str, float]]:
    
    if N_hat <= 0:
        return cpc_m0, 0.0, 0.0, 0.0, 0.0, 0.0, {}
    
    yd_raw = yd_tgt / yd_tgt.sum() if yd_tgt.sum() > 0 else np.ones(K) / K
    yd_active = yd_raw * active.astype(np.float64)
    active_sum = yd_active.sum()
    Y_D_cond = yd_active / active_sum if active_sum > 0 else Y_hat.copy()
    
    w = np.ones(K, dtype=np.float64)
    for k in range(K):
        if active[k] and Y_hat[k] > 0:
            w[k] = Y_D_cond[k] / Y_hat[k]
            
    weighted_mass = float(np.dot(Y_hat, w))
    s = w / weighted_mass if weighted_mass > 0 else np.ones(K)
    
    np.multiply(t0_inter, s[bin_idx], out=t_cal_buf)
            
    cal_mass = t_cal_buf.sum()
    if cal_mass > 0:
        t_cal_buf *= (N_hat / cal_mass)
        
    cpc = float(np.sum(np.minimum(t_true_inter, t_cal_buf)) * inv_sum_denom)
    
    np.subtract(t_true_inter, t_cal_buf, out=diff_buf)
    np.abs(diff_buf, out=diff_buf)
    mae = float(np.sum(diff_buf) * inv_N)
    
    np.square(diff_buf, out=diff_buf)
    rmse = float(np.sqrt(np.sum(diff_buf) * inv_N))
    
    spearman_val = float(spearmanr(t_true_inter, t_cal_buf)[0]) if compute_spearman else float('nan')
    
    active_w = w[active]
    w_gt_2 = float(np.mean(active_w > 2)) if len(active_w) > 0 else 0.0
    w_gt_5 = float(np.mean(active_w > 5)) if len(active_w) > 0 else 0.0
    w_gt_10 = float(np.mean(active_w > 10)) if len(active_w) > 0 else 0.0
    
    stats = {
        "w_min": float(active_w.min()) if len(active_w) > 0 else 1.0,
        "w_median": float(np.median(active_w)) if len(active_w) > 0 else 1.0,
        "w_p95": float(np.percentile(active_w, 95)) if len(active_w) > 0 else 1.0,
        "w_max": float(active_w.max()) if len(active_w) > 0 else 1.0,
        "w_gt_2": w_gt_2, "w_gt_5": w_gt_5, "w_gt_10": w_gt_10
    }
    
    tv_ach = float(0.5 * np.sum(np.abs(yd_tgt - yd_target)))
    js_div = float(jensenshannon(yd_tgt, yd_target))
    
    return cpc, mae, rmse, spearman_val, tv_ach, js_div, stats


def run_sampling_robustness(args: argparse.Namespace) -> None:
    data_root = "data"
    output_dir = getattr(args, "output_dir", None) or "results/sampling_robustness_v1"
    os.makedirs(output_dir, exist_ok=True)
    
    log_file = f"{output_dir}/run.log"
    logging.basicConfig(level=logging.INFO, format='%(message)s',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    
    sampling_base_seed = 20260823
    
    if args.smoke:
        m_grid = [100, 1000, float("inf")]
        B_sample = 20
        model_seeds_to_use = [1, 10]
        folds_to_run = [2]
    else:
        m_grid = [100, 250, 500, 1000, 2500, 5000, 10000, 50000, 100000, float("inf")]
        B_sample = args.b
        model_seeds_to_use = [1, 10, 100]
        folds_to_run = [1, 2, 3, 4, 5]
        
    splits = generate_35_5_10_splits(data_root=data_root)
    raw_results: List[Dict[str, Any]] = []
    
    for fold_id in folds_to_run:
        split = splits[fold_id]
        train_cities = split["train"]
        test_cities_to_use = split["test"] if not args.smoke else split["test"][:1]
            
        logger.info(f"\n=== Processing Fold {fold_id} ===")
        
        bin_edges, _ = compute_kbin_edges(train_cities, K=8, data_root=data_root)
        K = len(bin_edges) - 1
        
        for c_idx, tc in enumerate(test_cities_to_use):
            logger.info(f"  Target City: {tc} ({c_idx+1}/{len(test_cities_to_use)})")
            raw = load_raw_city(tc, data_root=data_root)
            dist_km = raw.dist_km
            inter_mask = (raw.pair_o_idx.numpy() != raw.pair_d_idx.numpy()) & (dist_km > 0.0)
            t_true_inter = raw.pair_trips.numpy()[inter_mask]
            
            yd_target = extract_yd_kbins(dist_km, raw.pair_trips.numpy(), bin_edges, inter_mask)
            
            # Pre-generate empirical sampled Y_D sets for all m and replicates
            logger.info("    Drawing empirical multinomial samples...")
            city_sample_sets: Dict[float, List[np.ndarray]] = {}
            for m in m_grid:
                if np.isinf(m):
                    city_sample_sets[m] = [yd_target.copy()]
                else:
                    samples_m = []
                    for b in range(B_sample):
                        seed_b = get_stable_seed(sampling_base_seed, fold_id, tc, int(m), b + 1)
                        yd_s = sample_empirical_yd(yd_target, m, seed_b)
                        samples_m.append(yd_s)
                    city_sample_sets[m] = samples_m
                    
            edge_index, edge_dist = build_radius_graph(
                lon_lat=raw.lon_lat, radius_km=5.0, include_self_loop=True, cache_key=f"{tc}_tracts"
            )
            
            dist_inter = dist_km[inter_mask]
            bin_idx = np.clip(np.digitize(dist_inter, bin_edges[1:-1], right=True), 0, K - 1).astype(np.int32)
            n_inter_pairs = len(dist_inter)
            inv_N = 1.0 / n_inter_pairs if n_inter_pairs > 0 else 0.0
            sum_t_true = float(t_true_inter.sum())
            
            t_cal_buf = np.empty(n_inter_pairs, dtype=np.float64)
            diff_buf = np.empty(n_inter_pairs, dtype=np.float64)
            
            for m_seed in model_seeds_to_use:
                logger.info(f"    Evaluating seed {m_seed}...")
                ckpt_path = Path(f"results/checkpoints/5fold_fold{fold_id}_seed{m_seed}.pt")
                if not ckpt_path.exists():
                    logger.warning(f"    Checkpoint {ckpt_path} not found. Skipping.")
                    continue
                try:
                    model, scaler, _ = load_checkpoint(ckpt_path, device_str="cpu")
                    model.eval()
                except Exception as e:
                    logger.error(f"    Failed to load checkpoint {ckpt_path}: {e}")
                    continue
                
                city_data = load_city(tc, data_root=data_root, feature_scaler=scaler, fit_scaler=False)
                t_pred_zs_tensor = infer_zero_shot(model, city_data, edge_index, edge_dist, device="cpu")
                t_pred_zs = t_pred_zs_tensor.numpy().astype(np.float64)
                
                t0_inter = t_pred_zs[inter_mask]
                N_hat = float(t0_inter.sum())
                cpc_m0 = float(compute_cpc_pair(t_true_inter, t0_inter))
                
                sum_denom = sum_t_true + N_hat
                inv_sum_denom = 2.0 / sum_denom if sum_denom > 0 else 0.0
                
                Y_hat = np.zeros(K, dtype=np.float64)
                active = np.zeros(K, dtype=bool)
                if N_hat > 0:
                    counts = np.bincount(bin_idx, weights=t0_inter, minlength=K)
                    Y_hat = counts / N_hat
                    pair_counts = np.bincount(bin_idx, minlength=K)
                    active = pair_counts > 0
                
                # 1. Oracle (m=inf)
                oracle_cpc, o_mae, o_rmse, o_spr, o_tv, o_js, o_stats = fast_cal_metrics(
                    yd_target, True, N_hat, K, active, Y_hat, t0_inter, bin_idx, t_true_inter, cpc_m0, yd_target,
                    inv_sum_denom, inv_N, t_cal_buf, diff_buf
                )
                
                def build_row(m_val: float, rep_id: int, cpc_val: float, mae: float, rmse: float, spr: float, tv_ach: float, js_div: float, st: Dict[str, float]) -> Dict[str, Any]:
                    row = {
                        "fold": fold_id, "target_city": tc, "model_seed": m_seed,
                        "sample_m": m_val, "replicate_id": rep_id,
                        "cpc_m0_inter": cpc_m0, "cpc_m1_inter": cpc_val,
                        "delta_cpc_inter": float(cpc_val - cpc_m0),
                        "degradation": float(oracle_cpc - cpc_val),
                        "mae": mae, "rmse": rmse, "spearman": spr,
                        "empirical_tv": tv_ach, "js_divergence": js_div,
                    }
                    row.update(st)
                    return row
                    
                raw_results.append(build_row(float("inf"), 0, oracle_cpc, o_mae, o_rmse, o_spr, o_tv, o_js, o_stats))
                
                # 2. Finite sample sizes m
                for m in m_grid:
                    if np.isinf(m):
                        continue
                    sample_list = city_sample_sets[m]
                    for b, yd_s in enumerate(sample_list):
                        n_cpc, n_mae, n_rmse, n_spr, n_tv, n_js, n_stats = fast_cal_metrics(
                            yd_s, False, N_hat, K, active, Y_hat, t0_inter, bin_idx, t_true_inter, cpc_m0, yd_target,
                            inv_sum_denom, inv_N, t_cal_buf, diff_buf
                        )
                        raw_results.append(build_row(float(m), b + 1, n_cpc, n_mae, n_rmse, n_spr, n_tv, n_js, n_stats))
                
    df = pd.DataFrame(raw_results)
    if not df.empty:
        df['spearman'] = df['spearman'].astype(float)
        
        df.to_csv(f"{output_dir}/sampling_raw.csv", index=False)
        df.to_json(f"{output_dir}/sampling_raw.jsonl", orient="records", lines=True)
        logger.info(f"Raw results saved with {len(df)} rows.")
        
        # Aggregation Step 1 & 2
        df_mean_b = df.groupby(["fold", "target_city", "model_seed", "sample_m"]).agg(
            delta_cpc_inter=("delta_cpc_inter", "mean"),
            degradation=("degradation", "mean"),
            empirical_tv=("empirical_tv", "mean"),
            js_divergence=("js_divergence", "mean"),
            cpc_m1_inter=("cpc_m1_inter", "mean"),
            prob_positive=("delta_cpc_inter", lambda x: float(np.mean(x > 0)))
        ).reset_index()
        
        df_seed_csv = df_mean_b.copy()
        df_seed_csv.to_csv(f"{output_dir}/sampling_per_seed.csv", index=False)
        
        city_df = df_mean_b.groupby(["fold", "target_city", "sample_m"]).agg(
            delta_cpc_mean=("delta_cpc_inter", "mean"),
            degradation_mean=("degradation", "mean"),
            empirical_tv_mean=("empirical_tv", "mean"),
            js_div_mean=("js_divergence", "mean"),
            prob_positive=("prob_positive", "mean"),
            cpc_m1_inter=("cpc_m1_inter", "mean")
        ).reset_index()
        
        city_df.to_csv(f"{output_dir}/sampling_per_city.csv", index=False)
        
        if not args.smoke:
            generate_sampling_summary(city_df, output_dir, m_grid)
    else:
        logger.warning("No results were generated. Check checkpoints.")


def generate_sampling_summary(city_df: pd.DataFrame, output_dir: str, m_grid: List[float]) -> None:
    confirmatory_folds = sorted([f for f in city_df.fold.unique() if f != 1])
    conf_df = city_df[city_df.fold.isin(confirmatory_folds)]
    
    if conf_df.empty:
        return
        
    sorted_m = sorted(m_grid, key=lambda x: (np.isinf(x), x))
    finite_m = [m for m in sorted_m if not np.isinf(m)]
    
    results: Dict[str, Dict[str, Any]] = {}
    p_benefit_onesided: List[float] = []
    p_degrad_onesided: List[float] = []
    
    # Get oracle delta_cpc per city for degradation paired test
    clean_vals_by_city: Dict[Tuple[int, str], float] = {}
    c_clean = conf_df[conf_df.sample_m.isin([float('inf')])]
    for _, row in c_clean.iterrows():
        clean_vals_by_city[(row["fold"], row["target_city"])] = row["delta_cpc_mean"]
        
    for m in sorted_m:
        m_str = "inf" if np.isinf(m) else str(int(m))
        c_m = conf_df[conf_df.sample_m == m]
        vals = c_m.delta_cpc_mean.values
        tv_vals = c_m.empirical_tv_mean.values
        
        mean_cpc1 = float(c_m.cpc_m1_inter.mean())
        mean_val = float(np.mean(vals))
        sd_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        median = float(np.median(vals))
        p25 = float(np.percentile(vals, 25))
        p75 = float(np.percentile(vals, 75))
        pos_cities = int(np.sum(vals > 0))
        harm_rate = float(np.sum(vals < 0) / len(vals))
        
        mean_tv = float(np.mean(tv_vals))
        tv_ci_lo, tv_ci_hi = fold_stratified_bootstrap(conf_df, "empirical_tv_mean", m, confirmatory_folds)
        ci_lower, ci_upper = fold_stratified_bootstrap(conf_df, "delta_cpc_mean", m, confirmatory_folds)
        
        # 1. Benefit Test (H1: delta_cpc > 0 vs M0)
        try:
            _, p_ben = wilcoxon(vals, alternative='greater')
        except Exception:
            p_ben = 1.0
            
        # 2. Degradation Test (H1: delta_cpc_oracle - delta_cpc_m > 0)
        degrad_vals = []
        for _, row in c_m.iterrows():
            clean_v = clean_vals_by_city.get((row["fold"], row["target_city"]), row["delta_cpc_mean"])
            degrad_vals.append(clean_v - row["delta_cpc_mean"])
        degrad_arr = np.array(degrad_vals)
        mean_degrad = float(np.mean(degrad_arr))
        
        if not np.isinf(m):
            try:
                _, p_deg = wilcoxon(degrad_arr, alternative='greater')
            except Exception:
                p_deg = 1.0
            p_benefit_onesided.append(float(p_ben))
            p_degrad_onesided.append(float(p_deg))
        else:
            p_deg = float('nan')
            
        results[m_str] = {
            "sample_m": m if not np.isinf(m) else None,
            "mean_cpc1": mean_cpc1,
            "mean_delta_cpc": mean_val, "sd": sd_val, "median": median,
            "p25": p25, "p75": p75, "ci_lower": ci_lower, "ci_upper": ci_upper,
            "pos_cities": pos_cities, "harm_rate": harm_rate,
            "mean_empirical_tv": mean_tv, "tv_ci_lo": tv_ci_lo, "tv_ci_hi": tv_ci_hi,
            "mean_degradation": mean_degrad,
            "wilcoxon_benefit_raw": float(p_ben),
            "wilcoxon_degrad_raw": float(p_deg) if not np.isnan(p_deg) else None
        }
        
    p_ben_adj = holm_correction(p_benefit_onesided)
    p_deg_adj = holm_correction(p_degrad_onesided)
    
    for i, m in enumerate(finite_m):
        m_str = str(int(m))
        results[m_str]["wilcoxon_benefit_holm"] = float(p_ben_adj[i])
        results[m_str]["wilcoxon_degrad_holm"] = float(p_deg_adj[i])
        
    oracle_gain = float(results["inf"]["mean_delta_cpc"])
    for m_str in results:
        if oracle_gain > 0:
            results[m_str]["relative_effect_pct"] = float(results[m_str]["mean_delta_cpc"] / oracle_gain * 100.0)
        else:
            results[m_str]["relative_effect_pct"] = None
            
    # Find crossover sample size m_cross where mean_delta_cpc >= 0
    m_cross = None
    for m in finite_m:
        if results[str(int(m))]["mean_delta_cpc"] >= 0:
            m_cross = int(m)
            break
            
    # Find significant benefit sample size m*
    m_star = None
    for m in finite_m:
        m_str = str(int(m))
        cond1 = results[m_str]["mean_delta_cpc"] > 0
        cond2 = results[m_str]["ci_lower"] > 0
        cond3 = results[m_str]["wilcoxon_benefit_holm"] < 0.05
        if cond1 and cond2 and cond3:
            m_star = int(m)
            break
            
    summary = {
        "confirmatory_n_cities": int(len(conf_df) // len(m_grid)),
        "m_cross_positive_dCPC": m_cross,
        "m_star_significant_benefit": m_star,
        "results_by_m": results
    }
    
    with open(f"{output_dir}/sampling_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    md = "# Empirical Y_D Sampling Robustness Summary\n\n"
    md += f"## Confirmatory Table (Folds 2-5, {int(len(conf_df)//len(m_grid))} Cities)\n\n"
    if m_cross is not None:
        md += f"**Minimum Sample Size for Positive Benefit ($m_{{\\text{{cross}}}}$):** `{m_cross:,}` trips\n\n"
    if m_star is not None:
        md += f"**Statistically Significant Benefit Sample Size ($m^*$):** `{m_star:,}` trips ($p < 0.05$ Holm)\n\n"
        
    md += "| Sample Size (m) | Mean Empirical TV | Mean M1 CPC | Mean dCPC | 95% CI | Pos Cities | Harm Rate | Rel Effect vs Clean (%) | Benefit p-val (vs M0) | Degrad p-val (vs Clean) |\n"
    md += "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
    
    for m in sorted_m:
        m_str = "inf" if np.isinf(m) else str(int(m))
        d = results[m_str]
        m_label = r"$\infty$ (Oracle)" if np.isinf(m) else f"{int(m):,}"
        tv_label = f"{d['mean_empirical_tv']:.4f} ({d['mean_empirical_tv']*100:.2f}%)"
        ci = f"[{d['ci_lower']:.5f}, {d['ci_upper']:.5f}]"
        
        ben_holm = d.get('wilcoxon_benefit_holm', d.get('wilcoxon_benefit_raw'))
        if isinstance(ben_holm, (float, np.floating)):
            ben_str = f"{ben_holm:.2e}" if ben_holm < 0.001 else f"{ben_holm:.4f}"
        else:
            ben_str = "N/A"
            
        deg_holm = d.get('wilcoxon_degrad_holm')
        if isinstance(deg_holm, (float, np.floating)):
            deg_str = f"{deg_holm:.2e}" if deg_holm < 0.001 else f"{deg_holm:.4f}"
        else:
            deg_str = "—"
            
        rel_eff = f"{d['relative_effect_pct']:+.1f}%" if d['relative_effect_pct'] is not None else "N/A"
        md += f"| {m_label} | {tv_label} | {d['mean_cpc1']:.5f} | {d['mean_delta_cpc']:+.5f} | {ci} | {d['pos_cities']}/{int(len(conf_df)//len(m_grid))} | {d['harm_rate']:.1%} | {rel_eff} | {ben_str} | {deg_str} |\n"
        
    with open(f"{output_dir}/sampling_summary.md", "w") as f:
        f.write(md)
        
    # --- Figure 1: Sample Size m vs Empirical TV Error ---
    plt.figure(figsize=(9, 6))
    finite_m_arr = np.array(finite_m)
    tv_means = [results[str(int(m))]["mean_empirical_tv"] for m in finite_m]
    tv_los = [results[str(int(m))]["tv_ci_lo"] for m in finite_m]
    tv_his = [results[str(int(m))]["tv_ci_hi"] for m in finite_m]
    
    plt.plot(finite_m_arr, tv_means, marker="o", color="darkblue", linewidth=2, label="Empirical TV Error")
    plt.fill_between(finite_m_arr, tv_los, tv_his, color="royalblue", alpha=0.25, label="95% Bootstrap CI")
    plt.axhline(0.0478, color="red", linestyle="--", linewidth=1.5, label="Theoretical Crossover $\\epsilon_{cross} = 4.78\\%$")
    plt.axhline(0.0300, color="darkorange", linestyle=":", linewidth=1.5, label="Significance Threshold $\\epsilon^* = 3.00\\%$")
    if m_star is not None:
        plt.axvline(m_star, color="green", linestyle="-.", label=f"Required $m^* = {m_star:,}$ trips")
    plt.xscale("log")
    plt.xlabel("Sample Size $m$ (Number of Observed Trips, log scale)")
    plt.ylabel("Empirical Total Variation Error $\\text{TV}(\\tilde{Y}_D^{(m)}, Y_D^{full})$")
    plt.title("Empirical TV Error vs Sample Size $m$")
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.legend()
    plt.savefig(f"{output_dir}/fig_sampling_m_vs_tv.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # --- Figure 2: Sample Size m vs Delta CPC ---
    plt.figure(figsize=(9, 6))
    dcpc_means = [results[str(int(m))]["mean_delta_cpc"] for m in finite_m]
    dcpc_los = [results[str(int(m))]["ci_lower"] for m in finite_m]
    dcpc_his = [results[str(int(m))]["ci_upper"] for m in finite_m]
    
    plt.plot(finite_m_arr, dcpc_means, marker="o", color="royalblue", linewidth=2, label="Mean $\\Delta$CPC")
    plt.fill_between(finite_m_arr, dcpc_los, dcpc_his, color="cornflowerblue", alpha=0.25, label="95% Bootstrap CI")
    plt.axhline(0, color="red", linestyle="--", alpha=0.7, label="Zero-Shot M0 Baseline")
    plt.axhline(oracle_gain, color="green", linestyle=":", label=f"Oracle Gain (+{oracle_gain:.5f})")
    if m_star is not None:
        plt.axvline(m_star, color="green", linestyle="-.", label=f"$m^* = {m_star:,}$ trips")
    plt.xscale("log")
    plt.xlabel("Sample Size $m$ (Number of Observed Trips, log scale)")
    plt.ylabel("Delta CPC ($M_1 - M_0$)")
    plt.title("OD Reconstruction Gain vs Observed Sample Size $m$")
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.legend()
    plt.savefig(f"{output_dir}/fig_sampling_m_vs_dcpc.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # --- Figure 3: Harm Rate vs Sample Size m ---
    plt.figure(figsize=(9, 6))
    harm_rates = [results[str(int(m))]["harm_rate"] for m in finite_m]
    plt.plot(finite_m_arr, harm_rates, marker="s", color="firebrick", linewidth=2, label="Harm Rate")
    plt.axhline(0.05, color="gray", linestyle=":", label="Oracle Baseline Harm Rate (5.0%)")
    plt.xscale("log")
    plt.xlabel("Sample Size $m$ (Number of Observed Trips, log scale)")
    plt.ylabel("Harm Rate (% Cities Worse than M0)")
    plt.ylim(-0.02, 1.02)
    plt.title("Harm Rate vs Observed Sample Size $m$")
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.legend()
    plt.savefig(f"{output_dir}/fig_sampling_harm_rate.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # --- Figure 4: Empirical TV vs Delta CPC (Direct Bridge to Synthetic Curve) ---
    plt.figure(figsize=(9, 6))
    plt.scatter(tv_means, dcpc_means, color="navy", s=60, zorder=3, label="Empirical Sampling ($m$)")
    for idx, m in enumerate(finite_m):
        plt.annotate(f"m={int(m):,}", (tv_means[idx], dcpc_means[idx]), textcoords="offset points", xytext=(5, 5), fontsize=8)
    plt.axhline(0, color="red", linestyle="--", alpha=0.7, label="Zero-Shot M0 Baseline")
    plt.axvline(0.0478, color="darkorange", linestyle=":", label="Synthetic Crossover $\\epsilon_{cross} = 4.78\\%$")
    plt.xlabel("Empirical Total Variation Error $\\text{TV}$")
    plt.ylabel("Mean $\\Delta$CPC")
    plt.title("Bridge: Empirical Sampling Error vs Reconstruction Benefit $\\Delta$CPC")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.savefig(f"{output_dir}/fig_sampling_tv_vs_dcpc_curve.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    manifest = {
        "experiment": "empirical_sampling_robustness",
        "timestamp": datetime.datetime.now().isoformat(),
        "m_grid": [m if not np.isinf(m) else "inf" for m in sorted_m],
        "m_cross": m_cross,
        "m_star": m_star
    }
    with open(f"{output_dir}/sampling_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--b", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_sampling_robustness(args)
