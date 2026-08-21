import os
import sys
import json
import hashlib
import argparse
import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from scipy.optimize import bisect
from scipy.stats import spearmanr, wilcoxon
from scipy.spatial.distance import jensenshannon

def holm_correction(p_vals):
    n = len(p_vals)
    sorted_indices = np.argsort(p_vals)
    adj_p = np.zeros(n)
    running_max = 0
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
def evaluate_cpc(t_true_inter, t_pred_inter):
    return compute_cpc_pair(t_true_inter, t_pred_inter)
from src.data.yd_extractor import compute_kbin_edges, extract_yd_kbins
from src.experiment.run_5fold import generate_35_5_10_splits
from src.experiment.run_experiment import infer_zero_shot

def get_active_bins(yd, eps=1e-8):
    return yd > eps

def get_stable_seed(noise_seed, fold, city, replicate_id):
    s = f"{noise_seed}_{fold}_{city}_{replicate_id}"
    return int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % (2**32)

def generate_nested_noisy_yd(p_active, epsilons, base_seed):
    K_act = len(p_active)
    if K_act == 1:
        return {eps: p_active.copy() for eps in epsilons}
        
    rng = np.random.RandomState(base_seed)
    
    for attempt in range(10000):
        z = rng.randn(K_act)
        z = z - np.mean(z)
        
        def get_p_sigma(sigma):
            log_p = np.log(p_active) + sigma * z
            max_log = np.max(log_p)
            p_sigma = np.exp(log_p - max_log)
            p_sigma = p_sigma / np.sum(p_sigma)
            return p_sigma
            
        def tv_diff(sigma, eps):
            p_sigma = get_p_sigma(sigma)
            return 0.5 * np.sum(np.abs(p_sigma - p_active)) - eps
            
        max_idx = np.argmax(z)
        p_inf = np.zeros_like(p_active)
        p_inf[max_idx] = 1.0
        max_tv = 0.5 * np.sum(np.abs(p_inf - p_active))
        
        if max_tv <= max(epsilons) + 1e-6:
            continue
            
        try:
            results = {}
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
                p_opt = get_p_sigma(sigma_opt)
                
                achieved_tv = 0.5 * np.sum(np.abs(p_opt - p_active))
                assert np.all(p_opt >= 0), "p_opt has negative values"
                assert np.abs(np.sum(p_opt) - 1.0) < 1e-8, "p_opt does not sum to 1"
                assert np.abs(achieved_tv - eps) < 1e-8, f"Achieved TV {achieved_tv} != requested {eps}"
                
                results[eps] = p_opt
            return results
        except (ValueError, AssertionError) as e:
            continue
            
    raise RuntimeError("Failed to generate valid noise direction after 10000 attempts.")


def fold_stratified_bootstrap(city_df, metric_col, eps, confirmatory_folds, n_boot=10000, seed=42):
    rng = np.random.RandomState(seed)
    
    vals = {}
    for f in confirmatory_folds:
        mask = (city_df.fold == f) & (city_df.epsilon == eps)
        vals[f] = city_df[mask][metric_col].values
        assert len(vals[f]) == 10, f"Expected 10 cities for fold {f}, got {len(vals[f])}"
        
    boot_means = np.zeros(n_boot)
    for i in range(n_boot):
        b_samples = []
        for f in confirmatory_folds:
            v = vals[f]
            b_samples.append(rng.choice(v, size=len(v), replace=True))
        boot_means[i] = np.concatenate(b_samples).mean()
        
    return np.percentile(boot_means, 2.5), np.percentile(boot_means, 97.5)


def fast_cal_metrics(yd_tgt, eps_req, compute_spearman, N_hat, K, active, Y_hat, t0_inter, bin_masks, t_true_inter, cpc_m0, yd_target):
    if N_hat <= 0:
        return cpc_m0, 0.0, 0.0, 0.0, eps_req, 0.0, {}
    
    yd_raw = yd_tgt / yd_tgt.sum() if yd_tgt.sum() > 0 else np.ones(K)/K
    yd_active = yd_raw * active.astype(np.float64)
    active_sum = yd_active.sum()
    Y_D_cond = yd_active / active_sum if active_sum > 0 else Y_hat.copy()
    
    w = np.ones(K, dtype=np.float64)
    for k in range(K):
        if active[k] and Y_hat[k] > 0:
            w[k] = Y_D_cond[k] / Y_hat[k]
            
    weighted_mass = float((Y_hat * w).sum())
    s = w / weighted_mass if weighted_mass > 0 else np.ones(K)
    
    t_cal_inter = t0_inter.copy()
    for k in range(K):
        if active[k]:
            t_cal_inter[bin_masks[k]] *= s[k]
            
    cal_mass = t_cal_inter.sum()
    if cal_mass > 0:
        t_cal_inter *= (N_hat / cal_mass)
        
    cpc = evaluate_cpc(t_true_inter, t_cal_inter)
    mae = float(np.mean(np.abs(t_true_inter - t_cal_inter)))
    rmse = float(np.sqrt(np.mean((t_true_inter - t_cal_inter)**2)))
    spearman = float(spearmanr(t_true_inter, t_cal_inter)[0]) if compute_spearman else np.nan
    
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
    
    tv_ach = 0.5 * np.sum(np.abs(yd_tgt - yd_target))
    js_div = float(jensenshannon(yd_tgt, yd_target))
    
    return cpc, mae, rmse, spearman, tv_ach, js_div, stats


def run_noise_robustness(args):
    data_root = "data"
    output_dir = "results/noise_robustness_v1"
    os.makedirs(output_dir, exist_ok=True)
    
    log_file = f"{output_dir}/run.log"
    logging.basicConfig(level=logging.INFO, format='%(message)s',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    
    noise_seed = 20260822
    model_seeds = [1, 10, 100]
    epsilons = [0.0, 0.05, 0.10, 0.20]
    nonzero_epsilons = [e for e in epsilons if e > 0]
    
    B_noise = args.b
    folds_to_run = [2] if args.smoke else [1, 2, 3, 4, 5]
    if args.smoke:
        B_noise = 20
        model_seeds = model_seeds[:2]
        
    splits = generate_35_5_10_splits(data_root=data_root)
    raw_results = []
    device = torch.device("cpu")
    
    for fold_id in folds_to_run:
        split = splits[fold_id]
        train_cities = split["train"]
        test_cities = split["test"]
        
        if args.smoke:
            test_cities = test_cities[:1]
            
        logger.info(f"\n=== Processing Fold {fold_id} ===")
        
        bin_edges, _ = compute_kbin_edges(train_cities, K=8, data_root=data_root)
        K = len(bin_edges) - 1
        
        for c_idx, tc in enumerate(test_cities):
            logger.info(f"  Target City: {tc} ({c_idx+1}/{len(test_cities)})")
            raw = load_raw_city(tc, data_root=data_root)
            dist_km = raw.dist_km
            inter_mask = (raw.pair_o_idx.numpy() != raw.pair_d_idx.numpy()) & (dist_km > 0.0)
            t_true_inter = raw.pair_trips.numpy()[inter_mask]
            
            yd_target = extract_yd_kbins(dist_km, raw.pair_trips.numpy(), bin_edges, inter_mask)
            active_mask = get_active_bins(yd_target)
            
            p_active_orig = yd_target[active_mask]
            p_active_orig = p_active_orig / p_active_orig.sum()
            
            logger.info("    Generating noise nested directions...")
            city_noise_sets = []
            for b in range(B_noise):
                seed_b = get_stable_seed(noise_seed, fold_id, tc, b+1)
                noisy_dict = generate_nested_noisy_yd(p_active_orig, epsilons, seed_b)
                full_dict = {}
                for eps, p_act in noisy_dict.items():
                    full_yd = np.zeros(K)
                    full_yd[active_mask] = p_act
                    full_dict[eps] = full_yd
                city_noise_sets.append(full_dict)
                
            edge_index, edge_dist = build_radius_graph(
                lon_lat=raw.lon_lat, radius_km=5.0, include_self_loop=True, cache_key=f"{tc}_tracts"
            )
            
            for m_seed in model_seeds:
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
                dist_inter = dist_km[inter_mask]
                N_hat = t0_inter.sum()
                
                cpc_m0 = evaluate_cpc(t_true_inter, t0_inter)
                
                bin_masks = []
                Y_hat = np.zeros(K, dtype=np.float64)
                active = np.zeros(K, dtype=bool)
                if N_hat > 0:
                    for k in range(K):
                        lo, hi = float(bin_edges[k]), float(bin_edges[k+1])
                        in_bin = (dist_inter > lo) & (dist_inter <= hi)
                        bin_masks.append(in_bin)
                        Y_hat[k] = t0_inter[in_bin].sum() / N_hat
                        active[k] = bool(in_bin.any())
                
                # 1. Oracle (eps=0)
                oracle_cpc, o_mae, o_rmse, o_spr, o_tv, o_js, o_stats = fast_cal_metrics(
                    yd_target, 0.0, True, N_hat, K, active, Y_hat, t0_inter, bin_masks, t_true_inter, cpc_m0, yd_target
                )
                
                if args.smoke:
                    assert o_tv < 1e-8, "Oracle TV is not 0"
                    
                def build_row(eps, rep_id, cpc_val, mae, rmse, spr, tv_ach, js_div, st):
                    row = {
                        "fold": int(fold_id), "target_city": tc, "model_seed": int(m_seed),
                        "epsilon": float(eps), "replicate_id": int(rep_id),
                        "cpc_m0_inter": float(cpc_m0), "cpc_m1_inter": float(cpc_val),
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
                            noisy_dict[eps], eps, False, N_hat, K, active, Y_hat, t0_inter, bin_masks, t_true_inter, cpc_m0, yd_target
                        )
                        if args.smoke:
                            assert np.abs(n_tv - eps) < 1e-8, f"TV mismatch in loop for eps {eps}"
                        raw_results.append(build_row(eps, b+1, n_cpc, n_mae, n_rmse, n_spr, n_tv, n_js, n_stats))
                
    # Save raw
    df = pd.DataFrame(raw_results)
    if not df.empty:
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
            prob_positive=("delta_cpc_inter", lambda x: np.mean(x > 0))
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
        

def generate_summary(city_df, output_dir, epsilons, nonzero_epsilons):
    confirmatory_folds = sorted([f for f in city_df.fold.unique() if f != 1])
    conf_df = city_df[city_df.fold.isin(confirmatory_folds)]
    
    if conf_df.empty:
        return
        
    results = {}
    p_onesided = []
    
    for eps in epsilons:
        c_eps = conf_df[conf_df.epsilon == eps]
        vals = c_eps.delta_cpc_mean.values
        
        mean_cpc1 = c_eps.cpc_m1_inter.mean()
        mean_val = np.mean(vals)
        sd_val = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
        median = np.median(vals)
        p25 = np.percentile(vals, 25)
        p75 = np.percentile(vals, 75)
        pos_cities = np.sum(vals > 0)
        harm_rate = np.sum(vals < 0) / len(vals)
        
        ci_lower, ci_upper = fold_stratified_bootstrap(conf_df, "delta_cpc_mean", eps, confirmatory_folds)
        
        _, p_1 = wilcoxon(vals, alternative='greater')
        _, p_2 = wilcoxon(vals, alternative='two-sided')
        if eps > 0.0:
            p_onesided.append(p_1)
        
        results[eps] = {
            "mean_cpc1": float(mean_cpc1),
            "mean_delta_cpc": float(mean_val), "sd": float(sd_val), "median": float(median),
            "p25": float(p25), "p75": float(p75), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper),
            "pos_cities": int(pos_cities), "harm_rate": float(harm_rate),
            "wilcoxon_one_sided_raw": float(p_1), "wilcoxon_two_sided": float(p_2)
        }
        
    p_adj = holm_correction(p_onesided)
    for i, e in enumerate(nonzero_epsilons):
        results[e]["wilcoxon_one_sided_holm"] = float(p_adj[i])
        
    oracle_gain = results[0.0]["mean_delta_cpc"]
    for e in epsilons:
        if oracle_gain > 0:
            results[e]["benefit_retention"] = float(results[e]["mean_delta_cpc"] / oracle_gain)
        else:
            results[e]["benefit_retention"] = None
            
    eps_star = 0.0
    for i, eps in enumerate(nonzero_epsilons):
        cond1 = results[eps]["mean_delta_cpc"] > 0
        cond2 = results[eps]["ci_lower"] > 0
        cond3 = results[eps]["wilcoxon_one_sided_holm"] < 0.05
        if cond1 and cond2 and cond3:
            eps_star = eps
        else:
            break
            
    summary = {
        "confirmatory_n_cities": int(len(conf_df) // len(epsilons)),
        "eps_star_estimate": float(eps_star),
        "results_by_eps": results
    }
    
    with open(f"{output_dir}/noise_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    md = "# 5-Fold Noise Robustness Summary\n\n"
    md += "## Confirmatory Table (Folds 2-5, 40 Cities)\n\n"
    md += "| Noise (eps) | Mean M1 CPC | Mean dCPC | 95% CI | Pos Cities | Harm Rate | Retention | Holm p-val |\n"
    md += "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
    for e in epsilons:
        d = results[e]
        ci = f"[{d['ci_lower']:.5f}, {d['ci_upper']:.5f}]"
        
        holm_val = d.get('wilcoxon_one_sided_holm')
        if isinstance(holm_val, float):
            holm = f"{holm_val:.2e}"
        else:
            holm = "N/A"
            
        ret = f"{d['benefit_retention']:.1%}" if d['benefit_retention'] is not None else "N/A"
        md += f"| {e} | {d['mean_cpc1']:.5f} | {d['mean_delta_cpc']:.5f} | {ci} | {d['pos_cities']}/{int(len(conf_df)//len(epsilons))} | {d['harm_rate']:.1%} | {ret} | {holm} |\n"
        
    with open(f"{output_dir}/noise_summary.md", "w") as f:
        f.write(md)
        
    plt.figure(figsize=(8,6))
    sns.lineplot(data=conf_df, x="epsilon", y="delta_cpc_mean", marker="o", errorbar=('ci', 95))
    plt.axhline(0, color="red", linestyle="--")
    plt.title("Dose-Response: Noise vs dCPC")
    plt.savefig(f"{output_dir}/fig_noise_dose_response.png")
    plt.close()
    
    hr = [results[e]["harm_rate"] for e in epsilons]
    plt.figure(figsize=(8,6))
    plt.plot(epsilons, hr, marker="s", color='red')
    plt.title("Harm Rate vs Noise Level")
    plt.ylim(0, 1.05)
    plt.savefig(f"{output_dir}/fig_noise_harm_rate.png")
    plt.close()
    
    plt.figure(figsize=(10,8))
    sns.lineplot(data=city_df, x="epsilon", y="delta_cpc_mean", hue="city", legend=False, alpha=0.5)
    plt.axhline(0, color="black", linestyle="--", linewidth=2)
    plt.title("Per-City Response to Noise")
    plt.savefig(f"{output_dir}/fig_noise_by_city.png")
    plt.close()
    
    manifest = {
        "noise_definition": "multiplicative compositional noise on active bins, TV distance matching via bisection",
        "timestamp": datetime.datetime.now().isoformat(),
        "B_noise": 1000,
        "epsilons": epsilons
    }
    with open(f"{output_dir}/noise_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--b", type=int, default=1000)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_noise_robustness(args)
