"""
Audit Script: Dissecting the Zero-Shot CPC Jump (0.133 -> 0.577).
Answers:
1. Is the baseline CPC (~0.133) low due to severe scale mismatch between cities?
2. What is the scale-normalized CPC (1 - TVD) before calibration?
3. How much of the gain comes from:
   - (A) Oracle Total Volume Scaling alone?
   - (B) Pure DBD Shape Calibration alone (preserving predicted volume)?
   - (C) Joint DBD Shape + Target Volume scaling?
4. Verification that q_b correctly matches p_b after ratio adjustment.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from src.data.dataset import load_city
from src.data.urban_graph import build_radius_graph
from src.models.zero_shot_model import ZeroShotODModel
from src.training.evaluate import compute_cpc_pair, compute_cpc_norm_pair
from scripts.run_loss_diagnostic import create_split_manifest, train_single_run, CITIES, SPLIT_SEED, TRAIN_RATIO, K_BINS

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

def main():
    print("="*75)
    print("AUDIT: DECOMPOSING ZERO-SHOT TRANSFER GAIN ON 5 CITIES (LOG1P-MSE, GNN)")
    print("="*75)

    # 1. Load data & graphs
    loaded_data = {c: load_city(c, data_root="data") for c in CITIES}
    graphs = {c: build_radius_graph(loaded_data[c].lon_lat.numpy(), radius_km=5.0) for c in CITIES}
    splits = create_split_manifest(loaded_data, seed=SPLIT_SEED, train_ratio=TRAIN_RATIO)

    # 2. Train GNN with log1p-mse on 30% of each city
    trained_models = {}
    for city in CITIES:
        print(f"Training source GNN for {city:12s} ...", end="", flush=True)
        cd = loaded_data[city]
        ei, ed = graphs[city]
        train_idx = splits[city]["train"]
        model, stab, _ = train_single_run(
            city_name=city,
            city_data=cd,
            edge_index=ei,
            edge_dist=ed,
            train_indices=train_idx,
            model_name="GNN",
            loss_mode="log1p_mse",
            seed=42,
            epochs=40
        )
        trained_models[city] = model
        print(" Done.")

    # 3. Transfer audit across 20 pairs
    audit_rows = []
    
    print("\nAuditing 20 transfer pairs...")
    for src in CITIES:
        model = trained_models[src]
        model.eval()
        for tgt in CITIES:
            if src == tgt:
                continue
                
            tgt_cd = loaded_data[tgt]
            tgt_ei, tgt_ed = graphs[tgt]
            
            mask = ((tgt_cd.pair_o_idx != tgt_cd.pair_d_idx) & (tgt_cd.pair_distance > 0) & (tgt_cd.pair_trips >= 1)).cpu().numpy()
            indices = np.where(mask)[0]
            
            with torch.no_grad():
                x = tgt_cd.node_features.to(DEVICE)
                ei = tgt_ei.to(DEVICE)
                ed = tgt_ed.to(DEVICE)
                p_o = tgt_cd.pair_o_idx[indices].to(DEVICE)
                p_d = tgt_cd.pair_d_idx[indices].to(DEVICE)
                p_dist = tgt_cd.pair_distance[indices].to(DEVICE)
                pop = tgt_cd.population.to(DEVICE)
                
                t_true = tgt_cd.pair_trips[indices].cpu().numpy().astype(np.float64)
                mu = model(x, ei, ed, p_o, p_d, p_dist, pop, return_conditional_mean=False)
                t_zs = np.maximum(0.0, mu.cpu().numpy().astype(np.float64))

            total_true = float(np.sum(t_true))
            total_pred = float(np.sum(t_zs))
            vol_ratio = total_pred / max(total_true, 1e-8)
            
            # Baseline metrics
            cpc_base = compute_cpc_pair(t_true, t_zs)
            cpc_norm_base = compute_cpc_norm_pair(t_true, t_zs)  # 1 - TVD (pure shape, independent of scale)
            
            # Condition A: Volume scaling only (no DBD binning)
            # T_vol = T_zs * (total_true / total_pred)
            t_vol_only = t_zs * (total_true / max(total_pred, 1e-8))
            cpc_vol_only = compute_cpc_pair(t_true, t_vol_only)
            
            # Condition B: Pure DBD Shape Calibration (Preserving model's own predicted volume)
            d_km = tgt_cd.dist_km[indices] if tgt_cd.dist_km is not None else np.expm1(tgt_cd.pair_distance[indices].cpu().numpy())
            min_d, max_d = float(np.min(d_km)), float(np.max(d_km)) + 1e-4
            bin_edges = np.linspace(min_d, max_d, K_BINS + 1)
            bin_idx = np.clip(np.digitize(d_km, bin_edges) - 1, 0, K_BINS - 1)
            
            p_b = np.zeros(K_BINS, dtype=np.float64)
            q_b = np.zeros(K_BINS, dtype=np.float64)
            for b in range(K_BINS):
                in_b = (bin_idx == b)
                p_b[b] = np.sum(t_true[in_b]) / max(total_true, 1e-8)
                q_b[b] = np.sum(t_zs[in_b]) / max(total_pred, 1e-8)
                
            r_b = np.where(q_b > 1e-9, p_b / np.maximum(q_b, 1e-9), 1.0)
            
            # Pure shape DBD (re-normalized to keep sum equal to total_pred)
            t_pure_dbd = t_zs * r_b[bin_idx]
            t_pure_dbd = t_pure_dbd * (total_pred / max(np.sum(t_pure_dbd), 1e-8))
            cpc_pure_dbd = compute_cpc_pair(t_true, t_pure_dbd)
            cpc_norm_pure_dbd = compute_cpc_norm_pair(t_true, t_pure_dbd)
            
            # Condition C: Joint DBD Shape + Target Volume scaling
            t_joint = t_zs * r_b[bin_idx] * (total_true / max(total_pred, 1e-8))
            cpc_joint = compute_cpc_pair(t_true, t_joint)
            
            # Verify distribution matching
            # Check TVD between calibrated predicted DBD and target p_b
            cal_qb = np.zeros(K_BINS, dtype=np.float64)
            for b in range(K_BINS):
                cal_qb[b] = np.sum(t_joint[bin_idx == b]) / np.sum(t_joint)
            tvd_after = 0.5 * np.sum(np.abs(cal_qb - p_b))
            
            audit_rows.append({
                "src": src,
                "tgt": tgt,
                "vol_ratio_Rvol": round(vol_ratio, 4),
                "CPC_baseline": round(cpc_base, 4),
                "CPC_norm_baseline": round(cpc_norm_base, 4),
                "CPC_vol_only": round(cpc_vol_only, 4),
                "CPC_pure_dbd": round(cpc_pure_dbd, 4),
                "CPC_joint": round(cpc_joint, 4),
                "Gain_vol_only": round(cpc_vol_only - cpc_base, 4),
                "Gain_pure_dbd": round(cpc_pure_dbd - cpc_base, 4),
                "Gain_joint": round(cpc_joint - cpc_base, 4),
                "TVD_dbd_misfit_after": round(tvd_after, 6)
            })
            
    df_audit = pd.DataFrame(audit_rows)
    df_audit.to_csv("diagnostics/loss_comparison/audit_zero_shot_decomposition.csv", index=False)
    print("\nSaved: diagnostics/loss_comparison/audit_zero_shot_decomposition.csv")
    
    print("\n" + "="*75)
    print("AVERAGED DECOMPOSITION ACROSS ALL 20 TRANSFER PAIRS:")
    print("="*75)
    means = df_audit.mean(numeric_only=True)
    print(f"1. Volume Ratio (R_vol = pred/true):        {means['vol_ratio_Rvol']:.4f}  (Mô hình dự đoán lệch quy mô bao nhiêu lần)")
    print(f"2. CPC Baseline (raw unscaled):             {means['CPC_baseline']:.4f}")
    print(f"3. CPC-Norm Baseline (1 - TVD, no scale):   {means['CPC_norm_baseline']:.4f}  <-- Chất lượng shape thực sự khi bỏ qua scale!")
    print(f"4. CPC sau Volume Scaling đơn thuần:        {means['CPC_vol_only']:.4f}  (Gain: +{means['Gain_vol_only']:.4f})")
    print(f"5. CPC sau Pure DBD Shape (Volume giữ cựu): {means['CPC_pure_dbd']:.4f}  (Gain: +{means['Gain_pure_dbd']:.4f})")
    print(f"6. CPC sau Joint DBD + Volume Scaling:      {means['CPC_joint']:.4f}  (Gain: +{means['Gain_joint']:.4f})")
    print(f"7. TVD misfit sau calibration:              {means['TVD_dbd_misfit_after']:.6f}  (Kiểm tra ratio khớp phân phối)")
    print("="*75)

if __name__ == "__main__":
    main()
