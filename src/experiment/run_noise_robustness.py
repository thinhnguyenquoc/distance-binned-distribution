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

from scipy.optimize import bisect
from scipy.stats import spearmanr, wilcoxon
from scipy.spatial.distance import jensenshannon

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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data.dataset import load_city, load_raw_city
from src.data.urban_graph import build_radius_graph
from src.training.train import load_checkpoint
from src.training.evaluate import compute_cpc_pair
from src.data.yd_extractor import compute_kbin_edges, extract_yd_kbins
from src.experiment.run_5fold import generate_35_5_10_splits
from src.experiment.run_experiment import infer_zero_shot

def evaluate_cpc(t_true_inter: np.ndarray, t_pred_inter: np.ndarray) -> float:
    return compute_cpc_pair(t_true_inter, t_pred_inter)

def get_active_bins(yd: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return yd > eps

def get_stable_seed(noise_seed: int, fold: int, city: str, replicate_id: int) -> int:
    s = f"{noise_seed}_{fold}_{city}_{replicate_id}"
    return int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % (2**32)

def generate_nested_noisy_yd(p_active: np.ndarray, epsilons: List[float], base_seed: int) -> Dict[float, np.ndarray]:
    K_act = len(p_active)
    if K_act == 1:
        return {eps: p_active.copy() for eps in epsilons}
        
    rng = np.random.RandomState(base_seed)
    
    for attempt in range(10000):
        z = rng.randn(K_act)
        z = z - np.mean(z)
        
        def get_p_sigma(sigma: float) -> np.ndarray:
            log_p = np.log(p_active) + sigma * z
            max_log = np.max(log_p)
            p_sigma = np.exp(log_p - max_log)
            p_sigma = p_sigma / np.sum(p_sigma)
            return p_sigma
            
        def tv_diff(sigma: float, eps: float) -> float:
            p_sigma = get_p_sigma(sigma)
            return float(0.5 * np.sum(np.abs(p_sigma - p_active)) - eps)
            
        max_idx = int(np.argmax(z))
        p_inf = np.zeros_like(p_active)
        p_inf[max_idx] = 1.0
        max_tv = float(0.5 * np.sum(np.abs(p_inf - p_active)))
        
        if max_tv <= max(epsilons) + 1e-6:
            continue
            
        try:
            results: Dict[float, np.ndarray] = {}
            for eps in epsilons:
                if eps == 0.0:
                    results[eps] = p_active.copy()
                    continue
                
                upper = 1.0
                while tv_diff(upper, eps) <= 0:
                    upper *= 2.0
                    if upper > 1e6:
                        raise ValueError("Upper bound too large")
                        
                sigma_opt = bisect(tv_diff, 0, upper, args=(eps,), xtol=1e-12, maxiter=1000)
                p_opt = get_p_sigma(float(sigma_opt))
                
                achieved_tv = float(0.5 * np.sum(np.abs(p_opt - p_active)))
                assert np.all(p_opt >= 0), "p_opt has negative values"
                assert np.abs(np.sum(p_opt) - 1.0) < 1e-8, "p_opt does not sum to 1"
                assert np.abs(achieved_tv - eps) < 1e-8, f"Achieved TV {achieved_tv} != requested {eps}"
                
                results[eps] = p_opt
            return results
        except (ValueError, AssertionError) as e:
            continue
            
    raise RuntimeError("Failed to generate valid noise direction after 10000 attempts.")


def fold_stratified_bootstrap(city_df: pd.DataFrame, metric_col: str, eps: float, confirmatory_folds: List[int], n_boot: int = 10000, seed: int = 42) -> Tuple[float, float]:
    rng = np.random.RandomState(seed)
    
    vals: Dict[int, np.ndarray] = {}
    for f in confirmatory_folds:
        mask = (city_df.fold == f) & (city_df.epsilon == eps)
        vals[f] = city_df[mask][metric_col].values
        assert len(vals[f]) == 10, f"Expected 10 cities for fold {f}, got {len(vals[f])}"
        
    f_samples = [vals[f][rng.randint(0, 10, size=(n_boot, 10))] for f in confirmatory_folds]
    all_samples = np.hstack(f_samples)
    boot_means = np.mean(all_samples, axis=1)
        
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def fast_cal_metrics(
    yd_tgt: np.ndarray, 
    eps_req: float, 
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
        return cpc_m0, 0.0, 0.0, 0.0, eps_req, 0.0, {}
    
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


def run_noise_robustness(args: argparse.Namespace) -> None:
    data_root = "data"
    grid_mode = getattr(args, "grid", "fine")
    if grid_mode == "fine":
        epsilons = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
        output_dir = getattr(args, "output_dir", None) or "results/noise_robustness_fine_v1"
    else:
        epsilons = [0.0, 0.05, 0.10, 0.20]
        output_dir = getattr(args, "output_dir", None) or "results/noise_robustness_v1"
        
    os.makedirs(output_dir, exist_ok=True)
    
    log_file = f"{output_dir}/run.log"
    logging.basicConfig(level=logging.INFO, format='%(message)s',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    
    noise_seed = 20260822
    nonzero_epsilons = [e for e in epsilons if e > 0]
    
    # Safely define parameters without mutating globals
    model_seeds_to_use = [1, 10, 100] if not args.smoke else [1, 10]
    B_noise = args.b if not args.smoke else 20
    folds_to_run = [1, 2, 3, 4, 5] if not args.smoke else [2]
        
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
            active_mask = get_active_bins(yd_target)
            
            p_active_orig = yd_target[active_mask]
            p_active_orig = p_active_orig / p_active_orig.sum()
            
            logger.info("    Generating noise nested directions...")
            city_noise_sets: List[Dict[float, np.ndarray]] = []
            for b in range(B_noise):
                seed_b = get_stable_seed(noise_seed, fold_id, tc, b+1)
                noisy_dict = generate_nested_noisy_yd(p_active_orig, epsilons, seed_b)
                full_dict: Dict[float, np.ndarray] = {}
                for eps, p_act in noisy_dict.items():
                    full_yd = np.zeros(K)
                    full_yd[active_mask] = p_act
                    full_dict[eps] = full_yd
                city_noise_sets.append(full_dict)
                
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
                
                cpc_m0 = float(evaluate_cpc(t_true_inter, t0_inter))
                
                sum_denom = sum_t_true + N_hat
                inv_sum_denom = 2.0 / sum_denom if sum_denom > 0 else 0.0
                
                Y_hat = np.zeros(K, dtype=np.float64)
                active = np.zeros(K, dtype=bool)
                if N_hat > 0:
                    counts = np.bincount(bin_idx, weights=t0_inter, minlength=K)
                    Y_hat = counts / N_hat
                    pair_counts = np.bincount(bin_idx, minlength=K)
                    active = pair_counts > 0
                
                # 1. Oracle (eps=0)
                oracle_cpc, o_mae, o_rmse, o_spr, o_tv, o_js, o_stats = fast_cal_metrics(
                    yd_target, 0.0, True, N_hat, K, active, Y_hat, t0_inter, bin_idx, t_true_inter, cpc_m0, yd_target,
                    inv_sum_denom, inv_N, t_cal_buf, diff_buf
                )
                
                if args.smoke:
                    assert o_tv < 1e-8, "Oracle TV is not 0"
                    
                def build_row(eps_val: float, rep_id: int, cpc_val: float, mae: float, rmse: float, spr: float, tv_ach: float, js_div: float, st: Dict[str, float]) -> Dict[str, Any]:
                    row = {
                        "fold": fold_id, "target_city": tc, "model_seed": m_seed,
                        "epsilon": eps_val, "replicate_id": rep_id,
                        "cpc_m0_inter": cpc_m0, "cpc_m1_inter": cpc_val,
                        "delta_cpc_inter": float(cpc_val - cpc_m0),
                        "degradation": float(oracle_cpc - cpc_val),
                        "mae": mae, "rmse": rmse, "spearman": spr,
                        "achieved_tv": tv_ach, "js_divergence": js_div,
                        "q": 1.0
                    }
                    row.update(st)
                    return row
                    
                raw_results.append(build_row(0.0, 0, oracle_cpc, o_mae, o_rmse, o_spr, o_tv, o_js, o_stats))
                
                # 2. Noise replicates
                for b, noisy_dict in enumerate(city_noise_sets):
                    for eps in nonzero_epsilons:
                        n_cpc, n_mae, n_rmse, n_spr, n_tv, n_js, n_stats = fast_cal_metrics(
                            noisy_dict[eps], eps, False, N_hat, K, active, Y_hat, t0_inter, bin_idx, t_true_inter, cpc_m0, yd_target,
                            inv_sum_denom, inv_N, t_cal_buf, diff_buf
                        )
                        if args.smoke:
                            assert np.abs(n_tv - eps) < 1e-8, f"TV mismatch in loop for eps {eps}"
                        raw_results.append(build_row(eps, b+1, n_cpc, n_mae, n_rmse, n_spr, n_tv, n_js, n_stats))
                
    df = pd.DataFrame(raw_results)
    if not df.empty:
        # Enforce explicit typing for consistency
        df['spearman'] = df['spearman'].astype(float)
        
        df.to_csv(f"{output_dir}/noise_raw.csv", index=False)
        df.to_json(f"{output_dir}/noise_raw.jsonl", orient="records", lines=True)
        logger.info(f"Raw results saved with {len(df)} rows.")
        
        # Aggregation Step 1 & 2
        df_mean_b = df.groupby(["fold", "target_city", "model_seed", "epsilon"]).agg(
            delta_cpc_inter=("delta_cpc_inter", "mean"),
            degradation=("degradation", "mean"),
            w_max=("w_max", "mean"),
            w_gt_2=("w_gt_2", "mean"),
            cpc_m1_inter=("cpc_m1_inter", "mean"),
            prob_positive=("delta_cpc_inter", lambda x: float(np.mean(x > 0)))
        ).reset_index()
        
        df_seed_csv = df_mean_b.copy()
        df_seed_csv.to_csv(f"{output_dir}/noise_per_seed.csv", index=False)
        
        city_df = df_mean_b.groupby(["fold", "target_city", "epsilon"]).agg(
            delta_cpc_mean=("delta_cpc_inter", "mean"),
            degradation_mean=("degradation", "mean"),
            prob_positive=("prob_positive", "mean"),
            cpc_m1_inter=("cpc_m1_inter", "mean"),
            w_max=("w_max", "mean"),
            w_gt_2=("w_gt_2", "mean")
        ).reset_index()
        
        city_df.to_csv(f"{output_dir}/noise_per_city.csv", index=False)
        
        if not args.smoke:
            generate_summary(city_df, output_dir, epsilons, nonzero_epsilons)
    else:
        logger.warning("No results were generated. Check checkpoints.")
        

def generate_summary(city_df: pd.DataFrame, output_dir: str, epsilons: List[float], nonzero_epsilons: List[float]) -> None:
    evaluation_folds = sorted(city_df.fold.unique().tolist())
    eval_df = city_df[city_df.fold.isin(evaluation_folds)]
    
    if eval_df.empty:
        return
        
    results: Dict[float, Dict[str, Any]] = {}
    p_benefit_onesided: List[float] = []
    p_degrad_onesided: List[float] = []
    
    # Get oracle delta_cpc per city for degradation paired test
    clean_vals_by_city: Dict[Tuple[int, str], float] = {}
    c_clean = eval_df[eval_df.epsilon == 0.0]
    for _, row in c_clean.iterrows():
        clean_vals_by_city[(row["fold"], row["target_city"])] = row["delta_cpc_mean"]
    
    for eps in epsilons:
        c_eps = eval_df[eval_df.epsilon == eps]
        vals = c_eps.delta_cpc_mean.values
        
        mean_cpc1 = float(c_eps.cpc_m1_inter.mean())
        mean_val = float(np.mean(vals))
        sd_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        median = float(np.median(vals))
        p25 = float(np.percentile(vals, 25))
        p75 = float(np.percentile(vals, 75))
        pos_cities = int(np.sum(vals > 0))
        harm_rate = float(np.sum(vals < 0) / len(vals))
        
        ci_lower, ci_upper = fold_stratified_bootstrap(eval_df, "delta_cpc_mean", eps, evaluation_folds)
        
        # 1. Benefit Test (H1: delta_cpc > 0 vs M0)
        try:
            _, p_ben = wilcoxon(vals, alternative='greater')
        except Exception:
            p_ben = 1.0
            
        # 2. Degradation Test (H1: delta_cpc_clean - delta_cpc_eps > 0 vs clean Y_D)
        degrad_vals = []
        for _, row in c_eps.iterrows():
            clean_v = clean_vals_by_city.get((row["fold"], row["target_city"]), row["delta_cpc_mean"])
            degrad_vals.append(clean_v - row["delta_cpc_mean"])
        degrad_arr = np.array(degrad_vals)
        mean_degrad = float(np.mean(degrad_arr))
        
        if eps > 0.0:
            try:
                _, p_deg = wilcoxon(degrad_arr, alternative='greater')
            except Exception:
                p_deg = 1.0
            p_benefit_onesided.append(float(p_ben))
            p_degrad_onesided.append(float(p_deg))
        else:
            p_deg = float('nan')
        
        results[eps] = {
            "mean_cpc1": mean_cpc1,
            "mean_delta_cpc": mean_val, "sd": sd_val, "median": median,
            "p25": p25, "p75": p75, "ci_lower": ci_lower, "ci_upper": ci_upper,
            "pos_cities": pos_cities, "harm_rate": harm_rate,
            "mean_degradation": mean_degrad,
            "wilcoxon_benefit_raw": float(p_ben),
            "wilcoxon_degrad_raw": float(p_deg) if not np.isnan(p_deg) else None
        }
        
    p_ben_adj = holm_correction(p_benefit_onesided)
    p_deg_adj = holm_correction(p_degrad_onesided)
    
    for i, e in enumerate(nonzero_epsilons):
        results[e]["wilcoxon_benefit_holm"] = float(p_ben_adj[i])
        results[e]["wilcoxon_degrad_holm"] = float(p_deg_adj[i])
        
    oracle_gain = float(results[0.0]["mean_delta_cpc"])
    for e in epsilons:
        if oracle_gain > 0:
            results[e]["relative_effect_pct"] = float(results[e]["mean_delta_cpc"] / oracle_gain * 100.0)
        else:
            results[e]["relative_effect_pct"] = None
            
    # Estimate exact crossover point epsilon_cross where mean_delta_cpc = 0
    eps_cross = None
    sorted_eps = sorted(epsilons)
    for i in range(len(sorted_eps) - 1):
        e1, e2 = sorted_eps[i], sorted_eps[i + 1]
        v1, v2 = results[e1]["mean_delta_cpc"], results[e2]["mean_delta_cpc"]
        if v1 >= 0 and v2 < 0:
            # Linear interpolation
            eps_cross = float(e1 + v1 / (v1 - v2) * (e2 - e1))
            break
        elif v1 > 0 and v2 == 0:
            eps_cross = float(e2)
            break
            
    # Estimate epsilon* (highest noise level with significant positive benefit)
    eps_star = 0.0
    for i, eps in enumerate(nonzero_epsilons):
        cond1 = results[eps]["mean_delta_cpc"] > 0
        cond2 = results[eps]["ci_lower"] > 0
        cond3 = results[eps]["wilcoxon_benefit_holm"] < 0.05
        if cond1 and cond2 and cond3:
            eps_star = eps
        else:
            break
            
    summary = {
        "n_evaluation_cities": int(len(eval_df) // len(epsilons)),
        "eps_cross_zero_dCPC": eps_cross,
        "eps_star_significant_benefit": float(eps_star),
        "results_by_eps": results
    }
    
    with open(f"{output_dir}/noise_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    md = "# 5-Fold Noise Robustness Summary\n\n"
    md += f"## Five-Fold Cross-City Evaluation Table (All 5 Folds, {int(len(eval_df)//len(epsilons))} Held-Out Test Cities)\n\n"
    if eps_cross is not None:
        md += f"**Crossover Threshold ($\\epsilon_{{\\text{{cross}}}}$, $\\Delta\\text{{CPC}}=0$):** `{eps_cross:.4f}` (TV $\\approx {eps_cross*100:.2f}\\%$)\n\n"
    md += "| Noise (eps) | Mean M1 CPC | Mean dCPC | 95% CI | Pos Cities | Harm Rate | Rel Effect vs Clean (%) | Benefit p-val (vs M0) | Degrad p-val (vs Clean) |\n"
    md += "| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
    for e in epsilons:
        d = results[e]
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
        md += f"| {e} | {d['mean_cpc1']:.5f} | {d['mean_delta_cpc']:+.5f} | {ci} | {d['pos_cities']}/{int(len(eval_df)//len(epsilons))} | {d['harm_rate']:.1%} | {rel_eff} | {ben_str} | {deg_str} |\n"
        
    with open(f"{output_dir}/noise_summary.md", "w") as f:
        f.write(md)
        
    # Figure 1: Dose-Response with CI
    plt.figure(figsize=(8, 6))
    means = [results[e]["mean_delta_cpc"] for e in epsilons]
    ci_lowers = [results[e]["ci_lower"] for e in epsilons]
    ci_uppers = [results[e]["ci_upper"] for e in epsilons]
    yerr_lower = [m - cl for m, cl in zip(means, ci_lowers)]
    yerr_upper = [cu - m for m, cu in zip(means, ci_uppers)]
    
    plt.errorbar(epsilons, means, yerr=[yerr_lower, yerr_upper], fmt='-o', color='royalblue', ecolor='gray', capsize=5, label='Confirmatory Mean (95% CI)')
    plt.axhline(0, color="red", linestyle="--", alpha=0.7, label='Zero-Shot M0 Baseline')
    if eps_cross is not None:
        plt.axvline(eps_cross, color="darkorange", linestyle=":", label=f'Crossover $\\epsilon_{{cross}} = {eps_cross:.3f}$')
    plt.xlabel("Noise Level (Epsilon TV)")
    plt.ylabel("Delta CPC (M1 - M0)")
    plt.title("Dose-Response: Noise Level vs Delta CPC")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.savefig(f"{output_dir}/fig_noise_dose_response.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Figure 2: Harm Rate
    hr = [results[e]["harm_rate"] for e in epsilons]
    plt.figure(figsize=(8, 6))
    plt.plot(epsilons, hr, marker="s", color='red', linewidth=2)
    if eps_cross is not None:
        plt.axvline(eps_cross, color="darkorange", linestyle=":", label=f'Crossover $\\epsilon_{{cross}} = {eps_cross:.3f}$')
    plt.title("Harm Rate vs Noise Level")
    plt.xlabel("Noise Level (Epsilon TV)")
    plt.ylabel("Harm Rate (% Cities Worse than M0)")
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.savefig(f"{output_dir}/fig_noise_harm_rate.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Figure 3: Per-City Response
    plt.figure(figsize=(10, 8))
    for city_name, g in city_df.groupby("target_city"):
        plt.plot(g["epsilon"], g["delta_cpc_mean"], alpha=0.35, color='gray')
    plt.plot(epsilons, means, marker="o", color='blue', linewidth=2.5, label='Overall Mean')
    plt.axhline(0, color="black", linestyle="--", linewidth=1.5)
    plt.title("Per-City Response to Noise")
    plt.xlabel("Noise Level (Epsilon TV)")
    plt.ylabel("Delta CPC (M1 - M0)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(f"{output_dir}/fig_noise_by_city.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    manifest = {
        "noise_definition": "multiplicative compositional noise on active bins, TV distance matching via bisection",
        "timestamp": datetime.datetime.now().isoformat(),
        "B_noise": 1000,
        "epsilons": epsilons,
        "eps_cross": eps_cross,
        "eps_star": eps_star
    }
    with open(f"{output_dir}/noise_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--b", type=int, default=1000)
    parser.add_argument("--grid", type=str, choices=["fine", "coarse"], default="fine", help="Grid: 'fine' [0..0.05] or 'coarse' [0..0.20]")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_noise_robustness(args)
