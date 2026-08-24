import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import datetime
import hashlib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.city_splits import generate_35_5_10_splits
from src.data.yd_extractor import compute_kbin_edges, extract_yd_kbins
from src.training.train import load_checkpoint, infer_zero_shot
from src.data.dataset import load_city
from src.data.urban_graph import build_radius_graph
from src.calibration.bin_calibration import calibrate_kbins
from src.training.evaluate import evaluate_moving_and_full

def holm_bonferroni(p_values: list[float]) -> list[float]:
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    adj_p = np.zeros(n)
    for i, idx in enumerate(sorted_indices):
        adj_p[idx] = min(1.0, p_values[idx] * (n - i))
    for i in range(1, n):
        idx = sorted_indices[i]
        prev_idx = sorted_indices[i-1]
        adj_p[idx] = max(adj_p[idx], adj_p[prev_idx])
    return adj_p.tolist()

def generate_file_hash(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()

def run_experiment(args):
    data_root = args.data_root
    output_dir = Path("results/k_sensitivity_v1")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(args.device)
    splits = generate_35_5_10_splits(data_root=data_root)
    K_values = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    seeds = [1, 10, 100]
    
    folds = [1, 2, 3, 4, 5] if not args.smoke_test else [2]
    
    results = []
    
    print("="*80)
    print("Starting 5-Fold Distance-Bin Number Sensitivity Test v1")
    if args.smoke_test:
        print("SMOKE TEST MODE: Fold 2 only, 1 city, seeds 1, 10")
        seeds = [1, 10]
    print("="*80)
    
    for fold_idx, fold in enumerate(folds, 1):
        train_cities = splits[fold]["train"]
        test_cities = splits[fold]["test"]
        
        if args.smoke_test:
            test_cities = test_cities[:1]
            
        print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] [Fold {fold_idx}/{len(folds)}] (Fold {fold}) Training cities: {len(train_cities)}, Test cities: {len(test_cities)}")
        
        bin_edges_by_k = {}
        for K in K_values:
            edges, k_act = compute_kbin_edges(train_cities, K=K, data_root=data_root)
            bin_edges_by_k[K] = {"edges": edges, "k_active": k_act}
            print(f"  [{datetime.datetime.now().strftime('%H:%M:%S')}] - K={K}: computed {k_act} active bins")
            
        for city_idx, target_city in enumerate(test_cities, 1):
            print(f"  [{datetime.datetime.now().strftime('%H:%M:%S')}] -> Evaluating City {city_idx}/{len(test_cities)}: {target_city}")
            
            for seed in seeds:
                ckpt_path = Path("results/checkpoints") / f"5fold_fold{fold}_seed{seed}.pt"
                if not ckpt_path.exists():
                    print(f"WARNING: Checkpoint {ckpt_path} missing. Skipping.")
                    continue
                
                model, scaler, _ = load_checkpoint(str(ckpt_path), device_str=args.device)
                model.eval()
                
                city_data = load_city(target_city, data_root=data_root, feature_scaler=scaler, fit_scaler=False)
                coords = city_data.lon_lat.numpy()
                edge_index, edge_dist = build_radius_graph(coords, radius_km=5.0)
                
                t_true = city_data.pair_trips.numpy().astype(np.float64)
                pair_o = city_data.pair_o_idx.numpy()
                pair_d = city_data.pair_d_idx.numpy()
                pair_dist = city_data.pair_distance.numpy()
                pair_dist_km = np.expm1(pair_dist)
                
                inter_mask = (pair_o != pair_d) & (pair_dist_km > 0.0)
                n_inter = inter_mask.sum()
                
                t_pred_zs_tensor = infer_zero_shot(model, city_data, edge_index, edge_dist, device=device)
                t0_np = t_pred_zs_tensor.numpy().astype(np.float64)
                
                m0_metrics = evaluate_moving_and_full(
                    city_data.pair_trips, t_pred_zs_tensor, city_data.pair_o_idx, city_data.pair_d_idx, city_data.bin_labels, pair_distance=city_data.pair_distance
                )
                
                m0_cpc_cache = m0_metrics["cpc_inter"]
                
                for K in K_values:
                    edges = bin_edges_by_k[K]["edges"]
                    k_active = bin_edges_by_k[K]["k_active"]
                    
                    yd_target = extract_yd_kbins(pair_dist_km, t_true, edges, inter_mask)
                    
                    yd_sum = float(np.sum(yd_target))
                    assert abs(yd_sum - 1.0) < 1e-6 or yd_sum == 0, f"Y_D sum={yd_sum} != 1.0"
                    
                    # Weights Diagnostics computation
                    inter_T0 = t0_np[inter_mask]
                    N_hat = inter_T0.sum()
                    inter_dist = pair_dist_km[inter_mask]
                    Y_hat = np.zeros(K, dtype=np.float64)
                    active = np.zeros(K, dtype=bool)
                    for k_idx in range(K):
                        lo, hi = float(edges[k_idx]), float(edges[k_idx + 1])
                        in_bin = (inter_dist > lo) & (inter_dist <= hi)
                        if N_hat > 0:
                            Y_hat[k_idx] = inter_T0[in_bin].sum() / N_hat
                        active[k_idx] = bool(in_bin.any())
                        
                    yd_raw = yd_target / yd_sum if yd_sum > 0 else np.ones(K)/K
                    yd_active = yd_raw * active.astype(np.float64)
                    active_sum = yd_active.sum()
                    Y_D_cond = yd_active / active_sum if active_sum > 0 else Y_hat.copy()
                    
                    w = np.ones(K, dtype=np.float64)
                    for k_idx in range(K):
                        if active[k_idx] and Y_hat[k_idx] > 0:
                            w[k_idx] = Y_D_cond[k_idx] / Y_hat[k_idx]  # q=1.0
                            
                    w_active = w[active]
                    if len(w_active) == 0: w_active = np.array([1.0])
                    
                    min_pred_mass = np.min(Y_hat[active]) if active.any() else 0.0
                    max_ratio = np.max(w_active)
                    
                    diag = {
                        "w_min": float(np.min(w_active)),
                        "w_median": float(np.median(w_active)),
                        "w_p95": float(np.percentile(w_active, 95)),
                        "w_max": float(max_ratio),
                        "frac_w_gt_2": float(np.mean(w_active > 2)),
                        "frac_w_gt_5": float(np.mean(w_active > 5)),
                        "frac_w_gt_10": float(np.mean(w_active > 10)),
                        "min_pred_mass": float(min_pred_mass),
                        "max_ratio": float(max_ratio),
                        "k_active": int(k_active),
                        "active_sum": float(active_sum)
                    }
                    
                    t_cal = calibrate_kbins(t0_np, pair_dist_km, inter_mask, yd_target, edges, q=1.0, tolerance=1e-5)
                    
                    m1_metrics = evaluate_moving_and_full(
                        city_data.pair_trips, torch.tensor(t_cal), city_data.pair_o_idx, city_data.pair_d_idx, city_data.bin_labels, pair_distance=city_data.pair_distance
                    )
                    
                    delta_cpc = m1_metrics["cpc_inter"] - m0_cpc_cache
                    
                    res_row = {
                        "city": target_city,
                        "fold": fold,
                        "seed": seed,
                        "K": K,
                        "q_K": 1.0,
                        "m0_cpc_inter": float(m0_cpc_cache),
                        "m1_cpc_inter": float(m1_metrics["cpc_inter"]),
                        "delta_cpc": float(delta_cpc),
                        "m1_mae_inter": float(m1_metrics["mae_inter"]),
                        "m1_rmse_inter": float(m1_metrics["rmse_inter"]),
                        "m1_spearman_inter": float(m1_metrics["spearman_inter"]),
                        "m1_cpc_inflow": float(m1_metrics.get("cpc_inflow", 0.0)),
                        "m1_cpc_outflow": float(m1_metrics.get("cpc_outflow", 0.0)),
                        "m1_rel_error_total": float(m1_metrics.get("rel_error_total", 0.0)),
                    }
                    res_row.update(diag)
                    results.append(res_row)
                    
    df = pd.DataFrame(results)
    df.to_csv(output_dir / "k_sensitivity_raw.csv", index=False)
    
    with open(output_dir / "k_sensitivity_raw.json", "w") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2)
        
    # Check M0 identical
    print("\nVerifying M0 consistency across K...")
    for (city, seed), group in df.groupby(['city', 'seed']):
        m0_vals = group['m0_cpc_inter'].values
        assert np.max(m0_vals) - np.min(m0_vals) < 1e-12, f"M0 changed across K for {city} seed {seed}!"
    print("M0 consistency passed.")
    
    # Aggregation
    print("\nAggregating over seeds...")
    avg_cols = ["m0_cpc_inter", "m1_cpc_inter", "delta_cpc", "m1_mae_inter", "m1_rmse_inter", "m1_spearman_inter", "w_max", "min_pred_mass", "k_active"]
    df_city = df.groupby(["city", "fold", "K"])[avg_cols].mean().reset_index()
    df_city.to_csv(output_dir / "k_sensitivity_per_city.csv", index=False)
    
    df_seed = df.copy()
    df_seed.to_csv(output_dir / "k_sensitivity_per_seed.csv", index=False)
    
    # Confirmatory Analysis
    df_conf = df_city[df_city["fold"].isin([2, 3, 4, 5])]
    print(f"\nConfirmatory cities: {df_conf['city'].nunique()}")
    
    summary_data = []
    
    for K in K_values:
        d = df_conf[df_conf["K"] == K]
        n_cities = len(d)
        if n_cities == 0:
            continue
            
        m0_mean = d["m0_cpc_inter"].mean()
        m1_mean = d["m1_cpc_inter"].mean()
        delta = d["delta_cpc"].values
        mean_d = np.mean(delta)
        std_d = np.std(delta, ddof=1) if n_cities > 1 else 0
        
        # Bootstrap
        rng = np.random.default_rng(42) # Bootstrap seed protocol
        boot_means = []
        for _ in range(10000):
            s = []
            for fold in [2, 3, 4, 5]:
                vals = d[d["fold"] == fold]["delta_cpc"].values
                if len(vals) > 0:
                    s.extend(rng.choice(vals, size=len(vals), replace=True))
            boot_means.append(np.mean(s))
        ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5]) if boot_means else (0,0)
        
        pos_cities = np.sum(delta > 0)
        
        _, p_1s = stats.wilcoxon(delta, alternative="greater") if len(delta) > 0 else (0, 1.0)
        _, p_2s = stats.wilcoxon(delta, alternative="two-sided") if len(delta) > 0 else (0, 1.0)
        
        summary_data.append({
            "K": K,
            "m0_cpc": m0_mean,
            "m1_cpc": m1_mean,
            "mean_delta": mean_d,
            "std_delta": std_d,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "pos_cities": int(pos_cities),
            "total_cities": n_cities,
            "k_act_mean": d["k_active"].mean(),
            "w_max_mean": d["w_max"].mean(),
            "p_1s_raw": p_1s,
            "p_2s_raw": p_2s,
        })
        
    # P-value adjustments
    secondary_ks = [K for K in K_values if K != 8]
    raw_ps = [next((s["p_1s_raw"] for s in summary_data if s["K"] == K), 1.0) for K in secondary_ks]
    adj_ps = holm_bonferroni(raw_ps)
    adj_p_map = dict(zip(secondary_ks, adj_ps))
    
    for s in summary_data:
        s["p_1s_adj"] = adj_p_map.get(s["K"], None)
        
    # Contrasts
    d8 = df_conf[df_conf["K"] == 8].set_index("city")
    mean_d8 = d8["delta_cpc"].mean()
    
    contrast_data = []
    raw_contrast_ps = []
    
    for K in secondary_ks:
        dk = df_conf[df_conf["K"] == K].set_index("city")
        common = d8.index.intersection(dk.index)
        
        d8_com = d8.loc[common]
        dk_com = dk.loc[common]
        
        ck = dk_com["delta_cpc"] - d8_com["delta_cpc"]
        _, p_ck = stats.wilcoxon(ck, alternative="two-sided") if len(ck) > 0 else (0, 1.0)
        
        raw_contrast_ps.append(p_ck)
        
        rk = dk["delta_cpc"].mean() / mean_d8 if mean_d8 > 0 else None
        
        contrast_data.append({
            "contrast": f"K{K} - K8",
            "mean_diff": float(ck.mean()) if len(ck)>0 else 0.0,
            "ci": [float(np.percentile(ck, 2.5)), float(np.percentile(ck, 97.5))] if len(ck)>0 else [0.0, 0.0],
            "p_adj": 1.0, # Placeholder, will be updated
            "r": rk
        })
        
    adj_contrast_ps = holm_bonferroni(raw_contrast_ps)
    for i in range(len(contrast_data)):
        contrast_data[i]["p_adj"] = adj_contrast_ps[i]
    
    # Save JSON summary
    out_sum = {
        "summary": summary_data,
        "contrasts": contrast_data
    }
    with open(output_dir / "k_sensitivity_summary.json", "w") as f:
        json.dump(out_sum, f, indent=2)
        
    # Generate Markdown
    md = []
    md.append("# 5-Fold Distance-Bin Number Sensitivity Test v1")
    md.append(f"\nConfirmatory cities (Folds 2-5): {df_conf['city'].nunique()}")
    md.append("\n## Primary Results")
    md.append("| K | Mean M0 CPC | Mean M1 CPC | Mean $\\Delta$ CPC | 95% CI | Positive cities | Mean $K_{active}$ | Mean $w_{max}$ | Adjusted p |")
    md.append("|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for s in summary_data:
        p_str = f"{s['p_1s_adj']:.4e}" if s['p_1s_adj'] is not None else "-"
        md.append(f"| {s['K']} | {s['m0_cpc']:.4f} | {s['m1_cpc']:.4f} | {s['mean_delta']:.4f} | [{s['ci_low']:.4f}, {s['ci_high']:.4f}] | {s['pos_cities']}/{s['total_cities']} | {s['k_act_mean']:.1f} | {s['w_max_mean']:.1f} | {p_str} |")
        
    md.append("\n## Contrasts (vs K=8)")
    md.append("| Contrast | Mean difference | 95% CI | Adjusted p |")
    md.append("|---|--:|--:|--:|")
    for c in contrast_data:
        md.append(f"| {c['contrast']} | {c['mean_diff']:.4f} | [{c['ci'][0]:.4f}, {c['ci'][1]:.4f}] | {c['p_adj']:.4e} |")
        
    with open(output_dir / "k_sensitivity_summary.md", "w") as f:
        f.write("\n".join(md))
        
    # Manifest
    manifest = {
        "split_seed": 20260818,
        "model_seeds": seeds,
        "bootstrap_seed": 42,
        "folds": folds,
        "confirmatory_folds": [2, 3, 4, 5] if not args.smoke_test else [2],
        "K_values": K_values,
        "primary_K": 8,
        "binning_method": "pair-weighted quantile",
        "q_policy": "q=1.0 fixed",
        "noise_level": 0.0,
        "checkpoint_hashes": {},
        "code_hash_version": generate_file_hash(__file__),
        "run_timestamp": datetime.datetime.now().isoformat()
    }
    with open(output_dir / "k_sensitivity_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        
    # Plotting
    # Fig 1: Mean gain by K
    plt.figure(figsize=(6, 4))
    ks = [s["K"] for s in summary_data]
    means = [s["mean_delta"] for s in summary_data]
    yerr = [[s["mean_delta"] - s["ci_low"] for s in summary_data], [s["ci_high"] - s["mean_delta"] for s in summary_data]]
    plt.errorbar(ks, means, yerr=yerr, marker='o', capsize=5)
    plt.axhline(0, color='red', linestyle='--')
    plt.xticks(K_values)
    plt.xlabel('K (number of bins)')
    plt.ylabel('Mean $\\Delta$ CPC')
    plt.title('Mean Gain by K (95% CI)')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "fig_delta_cpc_by_k.png", dpi=300)
    plt.close()
    
    # Fig 2: Per-city sensitivity
    plt.figure(figsize=(8, 5))
    for name, group in df_conf.groupby("city"):
        group = group.sort_values("K")
        fold = group["fold"].iloc[0]
        # In case we don't have enough colors, modulo by 10
        color = plt.cm.tab10(fold % 10)
        plt.plot(group["K"], group["delta_cpc"], marker='.', color=color, alpha=0.5, linewidth=1)
    plt.axhline(0, color='red', linestyle='--', linewidth=2)
    plt.xticks(K_values)
    plt.xlabel('K')
    plt.ylabel('$\\Delta$ CPC_c')
    plt.title('Per-City Sensitivity')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "fig_k_per_city.png", dpi=300)
    plt.close()
    
    # Fig 3: Calibration stability
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sns_k = [s["K"] for s in summary_data]
    
    axes[0].plot(sns_k, [s["w_max_mean"] for s in summary_data], marker='o')
    axes[0].set_title('Mean w_max')
    
    d_conf_minmass = df_conf.groupby("K")["min_pred_mass"].mean()
    axes[1].plot(d_conf_minmass.index, d_conf_minmass.values, marker='o')
    axes[1].set_title('Mean Min Predicted Mass')
    
    axes[2].plot(sns_k, [s["k_act_mean"] for s in summary_data], marker='o')
    axes[2].set_title('Mean K_active')
    
    for ax in axes:
        ax.set_xticks(K_values)
        ax.set_xlabel('K')
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(output_dir / "fig_weights_by_k.png", dpi=300)
    plt.close()
    
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()
    run_experiment(args)
