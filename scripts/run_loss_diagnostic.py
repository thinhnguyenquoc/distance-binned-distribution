"""
Diagnostic Script: Raw MSE vs Log1p-MSE on 5 Representative Cities.
Evaluates:
1. City flow statistics and heavy-tail severity.
2. 30/70 OD split manifest determinism.
3. Training stability (gradient norms, loss progression).
4. Within-city held-out evaluation on raw flow scale (CPC, MAE, MSE, RMSE).
5. Error breakdown across flow quantiles (Q1-Q5).
6. Systematic bias and volume ratio R_vol.
7. Preliminary zero-shot cross-city transfer (5x4 = 20 pairs) before/after DBD calibration.
"""

import os
import sys
import math
import time
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Project imports
from src.data.dataset import load_city, CityData, NODE_FEATURE_COLUMNS
from src.data.urban_graph import build_radius_graph
from src.models.zero_shot_model import ZeroShotODModel, ZeroShotMLPModel
from src.training.evaluate import compute_cpc_pair
from src.calibration.source_bins import (
    compute_source_distance_cap,
    build_source_bin_edges,
    assign_to_source_bins,
    compute_pure_calibration_ratios,
    apply_pure_dbd_calibration,
)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
OUTPUT_DIR = Path("diagnostics/loss_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CITIES = ["Arlington", "Seattle", "Charlotte", "San_Diego", "Chicago"]
SPLIT_SEED = 42
TRAIN_RATIO = 0.30
EPOCHS = 40
LR = 2e-3
WEIGHT_DECAY = 1e-4
CLIP_MAX_NORM = 5.0
K_BINS = 8


# ---------------------------------------------------------------------------
# Step 1: Data Distribution and Heavy-tail Statistics
# ---------------------------------------------------------------------------
def compute_flow_statistics(cities, data_root="data"):
    print("\n" + "="*70)
    print("STEP 1: Computing Flow Statistics & Heavy-Tail Diagnostic")
    print("="*70)
    
    stats_rows = []
    loaded_data = {}
    
    for city in cities:
        cd = load_city(city, data_root=data_root)
        loaded_data[city] = cd
        
        # Interzonal positive trips
        mask = (cd.pair_o_idx != cd.pair_d_idx) & (cd.pair_distance > 0) & (cd.pair_trips >= 1)
        flows = cd.pair_trips[mask].cpu().numpy().astype(np.float64)
        
        mean_val = float(np.mean(flows))
        med_val = float(np.median(flows))
        std_val = float(np.std(flows))
        p90 = float(np.percentile(flows, 90))
        p95 = float(np.percentile(flows, 95))
        p99 = float(np.percentile(flows, 99))
        max_val = float(np.max(flows))
        
        p99_to_med = p99 / max(med_val, 1e-6)
        max_to_med = max_val / max(med_val, 1e-6)
        
        # Log1p flows statistics
        log_flows = np.log1p(flows)
        log_std = float(np.std(log_flows))
        log_skew = float(((log_flows - np.mean(log_flows))**3).mean() / (log_std**3 + 1e-8))
        raw_skew = float(((flows - mean_val)**3).mean() / (std_val**3 + 1e-8))
        
        stats_rows.append({
            "city": city,
            "n_tracts": cd.n_tracts,
            "n_positive_pairs": len(flows),
            "mean": round(mean_val, 2),
            "median": round(med_val, 2),
            "std": round(std_val, 2),
            "P90": round(p90, 2),
            "P95": round(p95, 2),
            "P99": round(p99, 2),
            "max": round(max_val, 2),
            "P99_to_median": round(p99_to_med, 1),
            "max_to_median": round(max_to_med, 1),
            "raw_skewness": round(raw_skew, 2),
            "log1p_skewness": round(log_skew, 2)
        })
        
        print(f"[{city:12s}] Mean: {mean_val:6.1f} | Med: {med_val:4.1f} | P99: {p99:6.1f} | Max: {max_val:7.1f} | Max/Med: {max_to_med:6.1f}x | Skew: {raw_skew:5.1f} -> log1p: {log_skew:4.2f}")
        
    df_stats = pd.DataFrame(stats_rows)
    df_stats.to_csv(OUTPUT_DIR / "city_flow_statistics.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'city_flow_statistics.csv'}")
    return loaded_data, df_stats


# ---------------------------------------------------------------------------
# Step 2: 30/70 Manifest Creation (Fixed Seed)
# ---------------------------------------------------------------------------
def create_split_manifest(loaded_data, seed=SPLIT_SEED, train_ratio=TRAIN_RATIO):
    print("\n" + "="*70)
    print("STEP 2: Creating Fixed 30/70 OD Split Manifest (Seed = 42)")
    print("="*70)
    
    manifest_rows = []
    split_indices = {}
    
    rng = np.random.RandomState(seed)
    
    for city, cd in loaded_data.items():
        mask = ((cd.pair_o_idx != cd.pair_d_idx) & (cd.pair_distance > 0) & (cd.pair_trips >= 1)).cpu().numpy()
        valid_indices = np.where(mask)[0]
        n_total = len(valid_indices)
        
        # Deterministic shuffle
        shuffled = rng.permutation(valid_indices)
        n_train = int(np.floor(train_ratio * n_total))
        
        train_idx = shuffled[:n_train]
        eval_idx = shuffled[n_train:]
        
        split_indices[city] = {
            "train": train_idx,
            "eval": eval_idx
        }
        
        for idx in train_idx:
            manifest_rows.append({
                "source_city": city,
                "pair_idx": int(idx),
                "origin": int(cd.pair_o_idx[idx]),
                "destination": int(cd.pair_d_idx[idx]),
                "split": "train",
                "seed": seed
            })
        for idx in eval_idx:
            manifest_rows.append({
                "source_city": city,
                "pair_idx": int(idx),
                "origin": int(cd.pair_o_idx[idx]),
                "destination": int(cd.pair_d_idx[idx]),
                "split": "eval",
                "seed": seed
            })
            
        print(f"[{city:12s}] Total OD: {n_total:6d} | Train (30%): {len(train_idx):6d} | Held-out Eval (70%): {len(eval_idx):6d}")
        
    df_manifest = pd.DataFrame(manifest_rows)
    df_manifest.to_csv(OUTPUT_DIR / "od_split_manifest.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'od_split_manifest.csv'}")
    return split_indices


# ---------------------------------------------------------------------------
# Step 3: Model Factory & Training Loop
# ---------------------------------------------------------------------------
def create_model(model_name: str, node_in_dim: int):
    if model_name == "GNN":
        return ZeroShotODModel(
            node_in_dim=node_in_dim,
            node_hidden_dim=64,
            node_out_dim=64,
            num_gnn_layers=2,
            decoder_hidden_dim=64,
            dropout=0.1
        )
    elif model_name == "MLP":
        return ZeroShotMLPModel(
            node_in_dim=node_in_dim,
            node_hidden_dim=64,
            node_out_dim=64,
            num_gnn_layers=2,
            decoder_hidden_dim=64,
            dropout=0.1
        )
    else:
        raise ValueError(f"Unknown model {model_name}")


def train_single_run(
    city_name: str,
    city_data: CityData,
    edge_index: torch.Tensor,
    edge_dist: torch.Tensor,
    train_indices: np.ndarray,
    model_name: str,
    loss_mode: str,  # 'raw_mse' or 'log1p_mse'
    seed: int = 42,
    epochs: int = EPOCHS,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
    device: torch.device = DEVICE
):
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    node_in_dim = city_data.node_features.shape[1]
    model = create_model(model_name, node_in_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Pre-move tensors to device
    x = city_data.node_features.to(device)
    ei = edge_index.to(device)
    ed = edge_dist.to(device)
    p_o = city_data.pair_o_idx[train_indices].to(device)
    p_d = city_data.pair_d_idx[train_indices].to(device)
    p_dist = city_data.pair_distance[train_indices].to(device)
    pop = city_data.population.to(device)
    t_true = city_data.pair_trips[train_indices].to(device)
    
    # Target values
    if loss_mode == "log1p_mse":
        y_target = torch.log1p(t_true)
    else:
        y_target = t_true
        
    epoch_logs = []
    has_nan_inf = False
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass: returns mu >= 0
        mu = model(x, ei, ed, p_o, p_d, p_dist, pop, return_conditional_mean=False)
        
        if loss_mode == "raw_mse":
            y_pred = mu
            loss = F.mse_loss(y_pred, y_target)
        elif loss_mode == "log1p_mse":
            y_pred = torch.log1p(mu)
            loss = F.mse_loss(y_pred, y_target)
            
        if not torch.isfinite(loss):
            has_nan_inf = True
            print(f"ERROR: Non-finite loss at epoch {epoch} for {city_name} {model_name} {loss_mode}")
            break
            
        loss.backward()
        
        # Calculate gradient norm before clipping
        total_grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_grad_norm += param_norm.item() ** 2
        total_grad_norm = math.sqrt(total_grad_norm)
        
        # Clip gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP_MAX_NORM)
        optimizer.step()
        
        epoch_logs.append({
            "epoch": epoch,
            "loss": float(loss.item()),
            "grad_norm": float(total_grad_norm)
        })
        
    grad_norms = [e["grad_norm"] for e in epoch_logs]
    max_grad_norm = max(grad_norms) if grad_norms else 0.0
    mean_grad_norm = np.mean(grad_norms) if grad_norms else 0.0
    grad_norm_std = np.std(grad_norms) if grad_norms else 0.0
    final_loss = epoch_logs[-1]["loss"] if epoch_logs else 0.0
    
    stability_info = {
        "city": city_name,
        "model": model_name,
        "loss_mode": loss_mode,
        "seed": seed,
        "final_loss": round(final_loss, 4),
        "max_grad_norm": round(max_grad_norm, 2),
        "mean_grad_norm": round(mean_grad_norm, 2),
        "grad_norm_std": round(grad_norm_std, 2),
        "has_nan_inf": has_nan_inf,
        "n_clipped_epochs": sum(1 for g in grad_norms if g > CLIP_MAX_NORM)
    }
    
    return model, stability_info, epoch_logs


# ---------------------------------------------------------------------------
# Step 4: Within-City Held-out Evaluation (70%) on Raw Scale
# ---------------------------------------------------------------------------
def evaluate_source_holdout(
    model: nn.Module,
    city_name: str,
    city_data: CityData,
    edge_index: torch.Tensor,
    edge_dist: torch.Tensor,
    eval_indices: np.ndarray,
    model_name: str,
    loss_mode: str,
    device: torch.device = DEVICE
):
    model.eval()
    with torch.no_grad():
        x = city_data.node_features.to(device)
        ei = edge_index.to(device)
        ed = edge_dist.to(device)
        p_o = city_data.pair_o_idx[eval_indices].to(device)
        p_d = city_data.pair_d_idx[eval_indices].to(device)
        p_dist = city_data.pair_distance[eval_indices].to(device)
        pop = city_data.population.to(device)
        
        # Raw trips
        t_true = city_data.pair_trips[eval_indices].cpu().numpy().astype(np.float64)
        
        # Forward prediction mu >= 0
        mu = model(x, ei, ed, p_o, p_d, p_dist, pop, return_conditional_mean=False)
        t_pred = mu.cpu().numpy().astype(np.float64)
        t_pred = np.maximum(0.0, t_pred)
        
    # Metrics on original scale
    cpc = compute_cpc_pair(t_true, t_pred)
    mae = float(np.mean(np.abs(t_true - t_pred)))
    mse = float(np.mean((t_true - t_pred) ** 2))
    rmse = float(np.sqrt(mse))
    
    # Bias & Volume Ratio
    bias = float(np.mean(t_pred - t_true))
    r_vol = float(np.sum(t_pred) / max(np.sum(t_true), 1e-8))
    
    # Quantile metrics
    quantiles = [0, 50, 75, 90, 99, 100]
    q_thresholds = [np.percentile(t_true, q) for q in quantiles]
    quantile_records = []
    
    for k in range(len(quantiles) - 1):
        q_low, q_high = q_thresholds[k], q_thresholds[k+1]
        if k == len(quantiles) - 2:
            q_mask = (t_true >= q_low) & (t_true <= q_high)
        else:
            q_mask = (t_true >= q_low) & (t_true < q_high)
            
        if np.sum(q_mask) > 0:
            q_t_true = t_true[q_mask]
            q_t_pred = t_pred[q_mask]
            q_mae = float(np.mean(np.abs(q_t_true - q_t_pred)))
            q_rel_err = float(np.mean(np.abs(q_t_true - q_t_pred) / (q_t_true + 1.0)))
            q_label = f"Q{k+1}_{quantiles[k]}_{quantiles[k+1]}"
            
            quantile_records.append({
                "city": city_name,
                "model": model_name,
                "loss_mode": loss_mode,
                "quantile_bin": q_label,
                "n_pairs": int(np.sum(q_mask)),
                "flow_min": round(float(np.min(q_t_true)), 1),
                "flow_max": round(float(np.max(q_t_true)), 1),
                "mae": round(q_mae, 2),
                "rel_error": round(q_rel_err, 4)
            })
            
    summary_metric = {
        "city": city_name,
        "model": model_name,
        "loss_mode": loss_mode,
        "train_pairs": len(city_data.pair_trips) - len(eval_indices),
        "eval_pairs": len(eval_indices),
        "CPC": round(cpc, 4),
        "MAE": round(mae, 2),
        "MSE": round(mse, 2),
        "RMSE": round(rmse, 2),
        "Bias": round(bias, 2),
        "R_vol": round(r_vol, 3)
    }
    
    return summary_metric, quantile_records, t_pred


# ---------------------------------------------------------------------------
# Step 5: Preliminary Zero-Shot Cross-City Transfer (5x4 = 20 pairs)
# ---------------------------------------------------------------------------
def run_preliminary_zero_shot(
    trained_models: dict,
    loaded_data: dict,
    graphs: dict,
    k_bins: int = K_BINS,
    device: torch.device = DEVICE
):
    print("\n" + "="*70)
    print("STEP 5: Running Preliminary Zero-Shot Transfer & DBD Calibration (20 pairs)")
    print("="*70)
    
    zs_rows = []
    
    for (source_city, model_name, loss_mode), model in trained_models.items():
        model.eval()
        for target_city in CITIES:
            if source_city == target_city:
                continue
                
            tgt_cd = loaded_data[target_city]
            tgt_ei, tgt_ed = graphs[target_city]
            
            # Fixed Positive Interzonal OD Support (\Omega_t^+)
            tgt_o = tgt_cd.pair_o_idx.cpu().numpy()
            tgt_d = tgt_cd.pair_d_idx.cpu().numpy()
            tgt_dist = tgt_cd.dist_km if tgt_cd.dist_km is not None else np.expm1(tgt_cd.pair_distance.cpu().numpy())
            tgt_trips = tgt_cd.pair_trips.cpu().numpy().astype(np.float64)
            
            tgt_mask = (tgt_o != tgt_d) & (tgt_dist > 0.0) & (tgt_trips >= 1.0)
            tgt_indices = np.where(tgt_mask)[0]
            
            # Extract and validate support arrays
            o_supp = tgt_o[tgt_indices]
            d_supp = tgt_d[tgt_indices]
            dist_supp = tgt_dist[tgt_indices]
            t_true = tgt_trips[tgt_indices]
            
            # Mandatory sanity checks on \Omega_t^+
            assert np.all(o_supp != d_supp), f"{target_city}: Support contains intrazonal pairs!"
            assert np.all(dist_supp > 0.0), f"{target_city}: Support contains non-positive distances!"
            assert np.all(t_true >= 1.0), f"{target_city}: Support contains zero-flow pairs!"
            
            with torch.no_grad():
                x = tgt_cd.node_features.to(device)
                ei = tgt_ei.to(device)
                ed = tgt_ed.to(device)
                p_o = tgt_cd.pair_o_idx[tgt_indices].to(device)
                p_d = tgt_cd.pair_d_idx[tgt_indices].to(device)
                p_dist = tgt_cd.pair_distance[tgt_indices].to(device)
                pop = tgt_cd.population.to(device)
                
                mu = model(x, ei, ed, p_o, p_d, p_dist, pop, return_conditional_mean=False)
                t_zs = mu.cpu().numpy().astype(np.float64)
                t_zs = np.maximum(0.0, t_zs)
                
            # Baseline zero-shot metrics on \Omega_t^+
            cpc_before = compute_cpc_pair(t_true, t_zs)
            mae_before = float(np.mean(np.abs(t_true - t_zs)))
            mse_before = float(np.mean((t_true - t_zs) ** 2))
            
            # Target Distance-Binned Distribution Calibration (DBD) on \Omega_t^+
            d_km = dist_supp
            
            # Source-Specific Fixed-Width Distance Binning
            # Precomputed from source city 30% training distances
            src_train_idx = split_indices[source_city]["train"]
            src_cd = loaded_data[source_city]
            src_train_d = src_cd.dist_km[src_train_idx] if src_cd.dist_km is not None else np.expm1(src_cd.pair_distance[src_train_idx].cpu().numpy())
            d_cap_src = compute_source_distance_cap(src_train_d, percentile=99.0)
            src_bin_edges = build_source_bin_edges(d_cap_src, K=k_bins)
            
            # Assign target OD pairs to source-defined bins
            bin_assignments = assign_to_source_bins(d_km, src_bin_edges)
            
            total_t_true = np.sum(t_true)
            total_t_zs = np.sum(t_zs)
            
            p_b = np.zeros(k_bins, dtype=np.float64)
            q_b = np.zeros(k_bins, dtype=np.float64)
            
            for b in range(k_bins):
                in_b = (bin_assignments == b)
                p_b[b] = np.sum(t_true[in_b]) / max(total_t_true, 1e-8)
                q_b[b] = np.sum(t_zs[in_b]) / max(total_t_zs, 1e-8)
                
            # Pure Piecewise DBD Calibration (Zero epsilon smoothing, empty bin r_b=1, exact volume preservation)
            r_b = compute_pure_calibration_ratios(p_b, q_b)
            
            # Apply pure DBD calibration and assert |sum(t_cal) - sum(t_zs)| < 1e-10
            t_cal = apply_pure_dbd_calibration(t_zs, bin_assignments, r_b, tol=1e-10)
            
            cpc_after = compute_cpc_pair(t_true, t_cal)
            mae_after = float(np.mean(np.abs(t_true - t_cal)))
            mse_after = float(np.mean((t_true - t_cal) ** 2))
            
            delta_cpc = cpc_after - cpc_before
            delta_mae = mae_before - mae_after  # Positive = improved
            delta_mse = mse_before - mse_after  # Positive = improved
            
            # Optional Ablation: Target Total Volume Scaling (isolated from main DBD)
            # T_vol = T_zs * (total_t_true / total_t_zs)
            t_vol_ablation = t_zs * (total_t_true / max(total_t_zs, 1e-8))
            cpc_vol_ablation = compute_cpc_pair(t_true, t_vol_ablation)
            
            zs_rows.append({
                "source_city": source_city,
                "target_city": target_city,
                "model": model_name,
                "loss_mode": loss_mode,
                "cpc_before": round(cpc_before, 4),
                "cpc_after": round(cpc_after, 4),
                "delta_cpc": round(delta_cpc, 4),
                "mae_before": round(mae_before, 2),
                "mae_after": round(mae_after, 2),
                "delta_mae": round(delta_mae, 2),
                "mse_before": round(mse_before, 2),
                "mse_after": round(mse_after, 2),
                "delta_mse": round(delta_mse, 2),
                "cpc_vol_scaling_ablation": round(cpc_vol_ablation, 4)
            })
            
    df_zs = pd.DataFrame(zs_rows)
    df_zs.to_csv(OUTPUT_DIR / "preliminary_zero_shot.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'preliminary_zero_shot.csv'}")
    return df_zs


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------
def main():
    start_time = time.time()
    print("="*70)
    print(f"STARTING LOSS FUNCTION DIAGNOSTIC ON {len(CITIES)} REPRESENTATIVE CITIES")
    print(f"Device: {DEVICE}")
    print(f"Cities: {CITIES}")
    print("="*70)
    
    # 1. Flow Statistics
    loaded_data, df_stats = compute_flow_statistics(CITIES)
    
    # 2. Build Spatial Graphs
    print("\nPrecomputing spatial graphs...")
    graphs = {}
    for city, cd in loaded_data.items():
        coords = cd.lon_lat.numpy()
        ei, ed = build_radius_graph(coords, radius_km=5.0)
        graphs[city] = (ei, ed)
    print("Spatial graphs ready.")
    
    # 3. Create Manifest
    split_indices = create_split_manifest(loaded_data, seed=SPLIT_SEED, train_ratio=TRAIN_RATIO)
    
    # 4. Train Models & Evaluate Holdout
    print("\n" + "="*70)
    print("STEP 3 & 4: Training GNN & MLP with Raw MSE vs Log1p-MSE (30% train, eval on 70%)")
    print("="*70)
    
    all_stability = []
    all_holdout = []
    all_quantiles = []
    trained_models = {}
    
    models = ["GNN", "MLP"]
    losses = ["raw_mse", "log1p_mse"]
    
    for city in CITIES:
        cd = loaded_data[city]
        ei, ed = graphs[city]
        train_idx = split_indices[city]["train"]
        eval_idx = split_indices[city]["eval"]
        
        for model_name in models:
            for loss_mode in losses:
                run_key = f"{city}_{model_name}_{loss_mode}"
                print(f"\n---> Training: {city:12s} | Model: {model_name:3s} | Loss: {loss_mode:10s} ...", end="", flush=True)
                t0 = time.time()
                
                model, stab_info, epoch_logs = train_single_run(
                    city_name=city,
                    city_data=cd,
                    edge_index=ei,
                    edge_dist=ed,
                    train_indices=train_idx,
                    model_name=model_name,
                    loss_mode=loss_mode,
                    seed=42,
                    epochs=EPOCHS,
                    lr=LR
                )
                dt = time.time() - t0
                print(f" Done ({dt:.1f}s) | MaxGrad: {stab_info['max_grad_norm']} | Loss: {stab_info['final_loss']}")
                
                all_stability.append(stab_info)
                trained_models[(city, model_name, loss_mode)] = model
                
                # Evaluate on 70% held-out
                holdout_res, q_res, _ = evaluate_source_holdout(
                    model=model,
                    city_name=city,
                    city_data=cd,
                    edge_index=ei,
                    edge_dist=ed,
                    eval_indices=eval_idx,
                    model_name=model_name,
                    loss_mode=loss_mode
                )
                all_holdout.append(holdout_res)
                all_quantiles.extend(q_res)
                
    # Save stability & holdout CSVs
    df_stab = pd.DataFrame(all_stability)
    df_stab.to_csv(OUTPUT_DIR / "training_stability.csv", index=False)
    print(f"\nSaved: {OUTPUT_DIR / 'training_stability.csv'}")
    
    df_holdout = pd.DataFrame(all_holdout)
    df_holdout.to_csv(OUTPUT_DIR / "source_holdout_metrics.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'source_holdout_metrics.csv'}")
    
    df_quantiles = pd.DataFrame(all_quantiles)
    df_quantiles.to_csv(OUTPUT_DIR / "flow_quantile_metrics.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'flow_quantile_metrics.csv'}")
    
    # 5. Preliminary Zero-Shot Transfer
    df_zs = run_preliminary_zero_shot(trained_models, loaded_data, graphs, k_bins=K_BINS)
    
    total_time = time.time() - start_time
    print("\n" + "="*70)
    print(f"ALL DIAGNOSTIC RUNS COMPLETED IN {total_time:.1f}s ({total_time/60:.2f} mins)")
    print("="*70)


if __name__ == "__main__":
    main()
