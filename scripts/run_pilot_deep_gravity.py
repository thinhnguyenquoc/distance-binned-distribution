# Pilot runner for Adapted Deep Gravity (Fold 1, Seed 1)
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.city_splits import load_splits_manifest_v2
from src.data.dataset import load_city, load_cities, get_scaler_fingerprint, NODE_FEATURE_COLUMNS
from src.data.yd_extractor import compute_kbin_edges, extract_yd_kbins
from src.calibration.bin_calibration import calibrate_kbins
from src.experiment.e1_core import K_MOVE, Q_CALIB, TOLERANCE, build_inter_mask
from src.training.evaluate import compute_cpc_pair
from src.models.deep_gravity import DeepGravityAllocationModel
from src.models.origin_production_mlp import OriginProductionMLP


def build_origin_csr(pair_o_idx: torch.Tensor, num_origins: int):
    sort_perm = torch.argsort(pair_o_idx)
    sorted_o_idx = pair_o_idx[sort_perm]
    counts = torch.bincount(sorted_o_idx, minlength=num_origins)
    offsets = torch.zeros(num_origins + 1, dtype=torch.long)
    offsets[1:] = torch.cumsum(counts, dim=0)
    active_origins = torch.nonzero(counts > 0).squeeze(-1)
    return sort_perm, offsets, active_origins

def train_production_model(
    train_cities: list,
    val_cities: list,
    seed: int = 1,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 200,
    patience: int = 16,
    device: torch.device = torch.device("cpu"),
):
    print("--- [Step 1] Training OriginProductionMLP ---")
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = OriginProductionMLP(in_dim=26).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    train_data = []
    for c in train_cities:
        dist_km = np.expm1(c.pair_distance.numpy())
        omega_mask = torch.tensor(build_inter_mask(c, dist_km), dtype=torch.bool)
        
        assert (c.pair_o_idx[omega_mask] != c.pair_d_idx[omega_mask]).all()
        assert (c.pair_trips[omega_mask] > 0).all()
        
        x = c.node_features.to(device)
        O_i_omega = torch.zeros(c.n_tracts, dtype=torch.float32, device=device)
        O_i_omega.scatter_add_(0, c.pair_o_idx[omega_mask].to(device), c.pair_trips[omega_mask].to(device))
        z_i = torch.log1p(O_i_omega)
        train_data.append((x, z_i, O_i_omega, c.city_name))
        
    val_data = []
    for c in val_cities:
        dist_km = np.expm1(c.pair_distance.numpy())
        omega_mask = torch.tensor(build_inter_mask(c, dist_km), dtype=torch.bool)
        x = c.node_features.to(device)
        O_i_omega = torch.zeros(c.n_tracts, dtype=torch.float32, device=device)
        O_i_omega.scatter_add_(0, c.pair_o_idx[omega_mask].to(device), c.pair_trips[omega_mask].to(device))
        z_i = torch.log1p(O_i_omega)
        val_data.append((x, z_i, O_i_omega, c.city_name))
        
    best_val_rmse = float("inf")
    best_epoch = 0
    best_state = None
    patience_cnt = 0
    
    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        perm = np.random.permutation(len(train_data))
        for idx in perm:
            x, z_i, _, _ = train_data[idx]
            optimizer.zero_grad()
            z_hat = model(x)
            loss = nn.functional.mse_loss(z_hat, z_i)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_data)
        
        model.eval()
        val_rmses = []
        with torch.no_grad():
            for x, z_i, _, _ in val_data:
                z_hat = model(x)
                city_rmse = torch.sqrt(nn.functional.mse_loss(z_hat, z_i)).item()
                val_rmses.append(city_rmse)
        macro_val_rmse = float(np.mean(val_rmses))
        
        if macro_val_rmse < best_val_rmse - 1e-5:
            best_val_rmse = macro_val_rmse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_cnt = 0
        else:
            patience_cnt += 1
            
        if epoch % 20 == 0 or epoch == 1 or patience_cnt == 0:
            print(f"  Epoch {epoch:3d} | Train MSE: {train_loss:.4f} | Val Macro Log-RMSE: {macro_val_rmse:.4f} (best: {best_val_rmse:.4f} @ ep {best_epoch})")
            
        if patience_cnt >= patience:
            print(f"  Early stopping at epoch {epoch} (patience={patience})")
            break
            
    train_time = time.time() - t0
    print(f"  Production training completed in {train_time:.1f}s. Best epoch: {best_epoch} with Val Log-RMSE: {best_val_rmse:.4f}")
    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch, best_val_rmse, train_time
def train_deep_gravity_allocation(
    train_cities: list,
    val_cities: list,
    seed: int = 1,
    lr: float = 5e-6,
    momentum: float = 0.9,
    epochs: int = 20,
    batch_origins: int = 64,
    device: torch.device = torch.device("cpu"),
):
    print("--- [Step 2] Training DeepGravityAllocationModel with CSR Mini-batching ---")
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = DeepGravityAllocationModel(in_dim=53).to(device)
    optimizer = optim.RMSprop(model.parameters(), lr=lr, momentum=momentum)
    
    print("  Precomputing CSR index and features on Omega_c for 35 training cities...")
    train_items = []
    total_train_pairs = 0
    for c in train_cities:
        dist_km = np.expm1(c.pair_distance.numpy())
        omega_mask = torch.tensor(build_inter_mask(c, dist_km), dtype=torch.bool)
        o_idx_om = c.pair_o_idx[omega_mask]
        d_idx_om = c.pair_d_idx[omega_mask]
        dist_om = c.pair_distance[omega_mask].unsqueeze(1)
        trips_om = c.pair_trips[omega_mask]
        sort_perm, offsets, active_origins = build_origin_csr(o_idx_om, c.n_tracts)
        sorted_o = o_idx_om[sort_perm]
        sorted_d = d_idx_om[sort_perm]
        sorted_dist = dist_om[sort_perm]
        sorted_trips = trips_om[sort_perm].to(device)
        o_feat = c.node_features[sorted_o]
        d_feat = c.node_features[sorted_d]
        sorted_pair_feat = torch.cat([o_feat, d_feat, sorted_dist], dim=1).to(device)
        total_train_pairs += len(o_idx_om)
        train_items.append({
            "name": c.city_name,
            "sorted_pair_feat": sorted_pair_feat,
            "sorted_trips": sorted_trips,
            "offsets": offsets,
            "active_origins": active_origins,
            "n_tracts": c.n_tracts,
        })
    print(f"  Training CSR index ready. Total Omega_c pairs: {total_train_pairs:,}")
    
    print("  Precomputing CSR index for 5 validation cities...")
    val_items = []
    for c in val_cities:
        dist_km = np.expm1(c.pair_distance.numpy())
        omega_mask = torch.tensor(build_inter_mask(c, dist_km), dtype=torch.bool)
        o_idx_om = c.pair_o_idx[omega_mask]
        d_idx_om = c.pair_d_idx[omega_mask]
        dist_om = c.pair_distance[omega_mask].unsqueeze(1)
        trips_om = c.pair_trips[omega_mask]
        sort_perm, offsets, active_origins = build_origin_csr(o_idx_om, c.n_tracts)
        sorted_o = o_idx_om[sort_perm]
        sorted_d = d_idx_om[sort_perm]
        sorted_dist = dist_om[sort_perm]
        sorted_trips = trips_om[sort_perm].to(device)
        sorted_pair_feat = torch.cat([c.node_features[sorted_o], c.node_features[sorted_d], sorted_dist], dim=1).to(device)
        val_items.append({
            "name": c.city_name,
            "sorted_pair_feat": sorted_pair_feat,
            "sorted_trips": sorted_trips,
            "offsets": offsets,
            "active_origins": active_origins,
            "n_tracts": c.n_tracts,
        })
    
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        ep_t0 = time.time()
        model.train()
        train_loss = 0.0
        n_batches_total = 0
        perm_cities = np.random.permutation(len(train_items))
        for c_idx in perm_cities:
            item = train_items[c_idx]
            offsets = item["offsets"]
            active_origins = item["active_origins"]
            n_active = len(active_origins)
            shuffled_origs = active_origins[torch.randperm(n_active)]
            for i in range(0, n_active, batch_origins):
                batch_orig = shuffled_origs[i : i + batch_origins]
                num_b_orig = len(batch_orig)
                batch_feat_list = []
                batch_trips_list = []
                batch_compact_orig_list = []
                for compact_id, orig_id in enumerate(batch_orig):
                    st = offsets[orig_id].item()
                    en = offsets[orig_id + 1].item()
                    if en > st:
                        batch_feat_list.append(item["sorted_pair_feat"][st:en])
                        batch_trips_list.append(item["sorted_trips"][st:en])
                        batch_compact_orig_list.append(torch.full((en - st,), compact_id, dtype=torch.long, device=device))
                if not batch_feat_list:
                    continue
                b_feat = torch.cat(batch_feat_list, dim=0)
                b_trips = torch.cat(batch_trips_list, dim=0)
                b_compact_o = torch.cat(batch_compact_orig_list, dim=0)
                optimizer.zero_grad()
                loss = model.compute_loss_batch(b_feat, b_compact_o, b_trips, num_b_orig)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                train_loss += loss.item()
                n_batches_total += 1
        model.eval()
        val_losses = []
        with torch.no_grad():
            for item in val_items:
                offsets = item["offsets"]
                active_origins = item["active_origins"]
                city_orig_losses = []
                for i in range(0, len(active_origins), batch_origins):
                    batch_orig = active_origins[i : i + batch_origins]
                    num_b_orig = len(batch_orig)
                    b_feat_list, b_trips_list, b_compact_list = [], [], []
                    for compact_id, orig_id in enumerate(batch_orig):
                        st = offsets[orig_id].item()
                        en = offsets[orig_id + 1].item()
                        if en > st:
                            b_feat_list.append(item["sorted_pair_feat"][st:en])
                            b_trips_list.append(item["sorted_trips"][st:en])
                            b_compact_list.append(torch.full((en - st,), compact_id, dtype=torch.long, device=device))
                    if not b_feat_list:
                        continue
                    b_f = torch.cat(b_feat_list, dim=0)
                    b_t = torch.cat(b_trips_list, dim=0)
                    b_c = torch.cat(b_compact_list, dim=0)
                    b_loss = model.compute_loss_batch(b_f, b_c, b_t, num_b_orig).item()
                    city_orig_losses.append(b_loss)
                if city_orig_losses:
                    val_losses.append(float(np.mean(city_orig_losses)))
        mean_val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        ep_time = time.time() - ep_t0
        is_best = False
        if mean_val_loss < best_val_loss - 1e-5:
            best_val_loss = mean_val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            is_best = True
        print(f"  Epoch {epoch:2d}/{epochs} | Train CE: {train_loss / max(1, n_batches_total):.4f} | Val CE: {mean_val_loss:.4f} (best: {best_val_loss:.4f} @ ep {best_epoch}) {'*' if is_best else ''} | Time: {ep_time:.1f}s")
    last_state = copy.deepcopy(model.state_dict())
    total_time = time.time() - t0
    print(f"  DG training completed in {total_time:.1f}s. Best epoch: {best_epoch} with Val CE: {best_val_loss:.4f}")
    return model, best_state, last_state, best_epoch, best_val_loss, total_time
def evaluate_pilot_fold1(
    prod_model: OriginProductionMLP,
    dg_model: DeepGravityAllocationModel,
    test_cities: list[str],
    bin_edges: np.ndarray,
    scaler: object,
    data_root: str = "data",
    device: torch.device = torch.device("cpu"),
):
    print("--- [Step 3] Evaluating on Fold 1 Test Cities on Strict Omega_c Support ---")
    prod_model.eval()
    dg_model.eval()
    results = []
    for city_name in test_cities:
        cd = load_city(city_name, data_root=data_root, feature_scaler=scaler)
        n_t = cd.n_tracts
        dist_km = np.expm1(cd.pair_distance.numpy())
        omega_mask_np = build_inter_mask(cd, dist_km)
        omega_mask = torch.tensor(omega_mask_np, dtype=torch.bool)
        assert (cd.pair_o_idx[omega_mask] != cd.pair_d_idx[omega_mask]).all(), f"{city_name}: self-pairs in Omega_c!"
        assert (dist_km[omega_mask_np] > 0.0).all(), f"{city_name}: zero distance in Omega_c!"
        assert (cd.pair_trips[omega_mask] > 0).all(), f"{city_name}: zero trips in Omega_c!"
        o_idx_om = cd.pair_o_idx[omega_mask].to(device)
        d_idx_om = cd.pair_d_idx[omega_mask].to(device)
        dist_om = cd.pair_distance[omega_mask].to(device).unsqueeze(1)
        t_gt_om = cd.pair_trips[omega_mask].numpy().astype(np.float64)
        # Target OD intensities are used only for evaluation, oracle calibration,
        # and the privileged DG-Oracle-O diagnostic. They are not used to compute
        # DG-Predicted-O predictions. The known positive support is provided under
        # the common benchmark protocol.
        with torch.no_grad():
            o_feat = cd.node_features.to(device)[o_idx_om]
            d_feat = cd.node_features.to(device)[d_idx_om]
            pair_feat = torch.cat([o_feat, d_feat, dist_om], dim=1)
            probs_om = dg_model.compute_probs(pair_feat, o_idx_om, n_t).cpu().numpy().astype(np.float64)
            O_hat = prod_model.predict_outflow(cd.node_features.to(device)).cpu().numpy().astype(np.float64)
            t_pred_o_om = O_hat[o_idx_om.cpu().numpy()] * probs_om
            O_true_omega = np.zeros(n_t, dtype=np.float64)
            np.add.at(O_true_omega, o_idx_om.cpu().numpy(), t_gt_om)
            t_oracle_o_om = O_true_omega[o_idx_om.cpu().numpy()] * probs_om
        Y_D_tgt = extract_yd_kbins(dist_km, cd.pair_trips.numpy().astype(np.float64), bin_edges, omega_mask_np)
        t_pred_full = np.zeros(cd.n_pairs, dtype=np.float64)
        t_pred_full[omega_mask_np] = t_pred_o_om
        t_oracle_full = np.zeros(cd.n_pairs, dtype=np.float64)
        t_oracle_full[omega_mask_np] = t_oracle_o_om
        t_pred_cal_full = calibrate_kbins(t_pred_full, dist_km, omega_mask_np, Y_D_tgt, bin_edges, q=Q_CALIB, tolerance=TOLERANCE)
        t_oracle_cal_full = calibrate_kbins(t_oracle_full, dist_km, omega_mask_np, Y_D_tgt, bin_edges, q=Q_CALIB, tolerance=TOLERANCE)
        t_pred_cal_om = t_pred_cal_full[omega_mask_np]
        t_oracle_cal_om = t_oracle_cal_full[omega_mask_np]
        cpc_pred_before = compute_cpc_pair(t_gt_om, t_pred_o_om)
        cpc_pred_after  = compute_cpc_pair(t_gt_om, t_pred_cal_om)
        delta_pred      = cpc_pred_after - cpc_pred_before
        cpc_oracle_before = compute_cpc_pair(t_gt_om, t_oracle_o_om)
        cpc_oracle_after  = compute_cpc_pair(t_gt_om, t_oracle_cal_om)
        delta_oracle      = cpc_oracle_after - cpc_oracle_before
        tot_gt = float(np.sum(t_gt_om))
        tot_pred = float(np.sum(t_pred_o_om))
        tot_oracle = float(np.sum(t_oracle_o_om))
        ratio_pred = tot_pred / max(tot_gt, 1e-12)
        ratio_oracle = tot_oracle / max(tot_gt, 1e-12)
        assert abs(np.sum(t_pred_cal_om) - tot_pred) < 1e-4 * max(1.0, tot_pred)
        active_origs = np.unique(o_idx_om.cpu().numpy())
        orig_tot_before = np.zeros(n_t, dtype=np.float64)
        orig_tot_after = np.zeros(n_t, dtype=np.float64)
        np.add.at(orig_tot_before, o_idx_om.cpu().numpy(), t_pred_o_om)
        np.add.at(orig_tot_after, o_idx_om.cpu().numpy(), t_pred_cal_om)
        orig_drift = np.abs(orig_tot_after[active_origs] - orig_tot_before[active_origs]) / np.maximum(orig_tot_before[active_origs], 1.0)
        mean_orig_drift = float(np.mean(orig_drift))
        max_orig_drift = float(np.max(orig_drift))
        o_hat_act = O_hat[active_origs]
        o_true_act = O_true_omega[active_origs]
        prod_log_rmse = float(np.sqrt(np.mean((np.log1p(o_hat_act) - np.log1p(o_true_act)) ** 2)))
        prod_r = float(np.corrcoef(o_hat_act, o_true_act)[0, 1]) if np.std(o_hat_act) > 0 and np.std(o_true_act) > 0 else 0.0
        rec = {
            "city": city_name,
            "n_tracts": n_t,
            "n_pairs": cd.n_pairs,
            "n_omega_pairs": int(omega_mask.sum().item()),
            "pred_cpc_before": cpc_pred_before,
            "pred_cpc_after": cpc_pred_after,
            "pred_delta": delta_pred,
            "pred_flow_ratio": ratio_pred,
            "oracle_cpc_before": cpc_oracle_before,
            "oracle_cpc_after": cpc_oracle_after,
            "oracle_delta": delta_oracle,
            "oracle_flow_ratio": ratio_oracle,
            "prod_log_rmse": prod_log_rmse,
            "prod_r": prod_r,
            "mean_orig_drift": mean_orig_drift,
            "max_orig_drift": max_orig_drift,
        }
        results.append(rec)
        print(f"  {city_name:<16} | Pred: {cpc_pred_before:.4f} -> {cpc_pred_after:.4f} ({delta_pred:+.4f}, ratio {ratio_pred:.2f}) | Oracle: {cpc_oracle_before:.4f} -> {cpc_oracle_after:.4f} ({delta_oracle:+.4f}) | Drift: {mean_orig_drift:.2%}")
    return results
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="results/deep_gravity_pilot")
    args = parser.parse_args()
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    manifest_path = "results/e1/splits_manifest_v2.json"
    splits = load_splits_manifest_v2(manifest_path, data_root=args.data_root)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_meta = json.load(f)
    manifest_sha256 = manifest_meta.get("manifest_sha256", "unknown")
    
    fold_split = splits[args.fold]
    train_city_names = fold_split["train"]
    val_city_names = fold_split["val"]
    test_city_names = sorted(fold_split["test"])
    
    print(f"=== PILOT RUN: Adapted Deep Gravity on Fold {args.fold}, Seed {args.seed} ===")
    print(f"Manifest SHA256: {manifest_sha256[:10]}...")
    print(f"Train cities: {len(train_city_names)} | Val cities: {len(val_city_names)} | Test cities: {len(test_city_names)}")
    
    print("Fitting feature scaler on 35 training cities...")
    train_cities, scaler = load_cities(train_city_names, data_root=args.data_root)
    val_cities = [load_city(c, data_root=args.data_root, feature_scaler=scaler) for c in val_city_names]
    
    bin_edges, k_active = compute_kbin_edges(train_city_names, K=K_MOVE, data_root=args.data_root)
    assert k_active == K_MOVE, f"Expected {K_MOVE} active bins, got {k_active}"
    
    device = torch.device("cpu")
    
    prod_model, prod_epoch, prod_val_rmse, prod_time = train_production_model(
        train_cities=train_cities,
        val_cities=val_cities,
        seed=args.seed,
        device=device,
    )
    
    dg_model, best_dg_state, last_dg_state, dg_epoch, dg_val_ce, dg_time = train_deep_gravity_allocation(
        train_cities=train_cities,
        val_cities=val_cities,
        seed=args.seed,
        epochs=20,
        batch_origins=64,
        device=device,
    )
    
    prod_ckpt_path = out_dir / f"production_fold{args.fold}_seed{args.seed}.pt"
    torch.save({
        "model_state_dict": prod_model.state_dict(),
        "fold": args.fold,
        "seed": args.seed,
        "best_epoch": prod_epoch,
        "best_val_rmse": prod_val_rmse,
        "scaler_fingerprint": get_scaler_fingerprint(scaler),
        "scaler_feature_columns": list(NODE_FEATURE_COLUMNS),
        "manifest_sha256": manifest_sha256,
        "feature_scaler": scaler,
    }, prod_ckpt_path)
    print(f"Saved production model to {prod_ckpt_path}")
    
    dg_ckpt_path = out_dir / f"deep_gravity_fold{args.fold}_seed{args.seed}_best.pt"
    torch.save({
        "model_state_dict": best_dg_state,
        "fold": args.fold,
        "seed": args.seed,
        "best_epoch": dg_epoch,
        "best_val_ce": dg_val_ce,
        "scaler_fingerprint": get_scaler_fingerprint(scaler),
        "scaler_feature_columns": list(NODE_FEATURE_COLUMNS),
        "manifest_sha256": manifest_sha256,
        "feature_scaler": scaler,
    }, dg_ckpt_path)
    print(f"Saved Deep Gravity best checkpoint to {dg_ckpt_path}")
    
    dg_model.load_state_dict(best_dg_state)
    
    city_results = evaluate_pilot_fold1(
        prod_model=prod_model,
        dg_model=dg_model,
        test_cities=test_city_names,
        bin_edges=bin_edges,
        scaler=scaler,
        data_root=args.data_root,
        device=device,
    )
    
    pred_before_mean = float(np.mean([r["pred_cpc_before"] for r in city_results]))
    pred_after_mean  = float(np.mean([r["pred_cpc_after"] for r in city_results]))
    pred_delta_mean  = float(np.mean([r["pred_delta"] for r in city_results]))
    pred_win_rate    = f"{sum(r['pred_delta'] > 0 for r in city_results)}/{len(city_results)}"
    
    oracle_before_mean = float(np.mean([r["oracle_cpc_before"] for r in city_results]))
    oracle_after_mean  = float(np.mean([r["oracle_cpc_after"] for r in city_results]))
    oracle_delta_mean  = float(np.mean([r["oracle_delta"] for r in city_results]))
    oracle_win_rate    = f"{sum(r['oracle_delta'] > 0 for r in city_results)}/{len(city_results)}"
    mean_drift_all   = float(np.mean([r["mean_orig_drift"] for r in city_results]))
    
    summary = {
        "protocol": "adapted_deep_gravity_known_positive_support",
        "fold": args.fold,
        "seed": args.seed,
        "manifest_sha256": manifest_sha256,
        "torch_version": torch.__version__,
        "distance_transform": "log1p(d_km)",
        "prod_model": {
            "architecture": "MLP(26 -> 64 -> 32 -> 1) with ReLU",
            "best_epoch": prod_epoch,
            "best_val_rmse": prod_val_rmse,
            "training_time_sec": prod_time,
        },
        "dg_model": {
            "architecture": "15-hidden-layer MLP (6x256, 9x128) with LeakyReLU",
            "total_params": sum(p.numel() for p in dg_model.parameters()),
            "best_epoch": dg_epoch,
            "best_val_ce": dg_val_ce,
            "training_time_sec": dg_time,
            "batch_size_origins": 64,
            "epochs": 20,
            "optimizer": "RMSprop(lr=5e-6, momentum=0.9)",
        },
        "pred_o": {
            "description": "Zero-shot under known-positive support protocol",
            "cpc_before_mean": pred_before_mean,
            "cpc_after_mean": pred_after_mean,
            "delta_cpc_mean": pred_delta_mean,
            "win_rate": pred_win_rate,
            "mean_orig_drift": mean_drift_all,
        },
        "oracle_o": {
            "description": "Privileged information reference (non zero-shot diagnostic)",
            "cpc_before_mean": oracle_before_mean,
            "cpc_after_mean": oracle_after_mean,
            "delta_cpc_mean": oracle_delta_mean,
            "win_rate": oracle_win_rate,
        },
        "per_city": city_results,
    }
    summary_path = out_dir / f"pilot_fold{args.fold}_seed{args.seed}_results.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved pilot summary to {summary_path}")
    print("\n=== PILOT SUMMARY (Fold 1, Seed 1) ===")
    print(f"DG-Predicted-O: CPC {pred_before_mean:.5f} -> {pred_after_mean:.5f} (Delta: {pred_delta_mean:+.5f}, Win: {pred_win_rate}, Drift: {mean_drift_all:.2%})")
    print(f"DG-Oracle-O:    CPC {oracle_before_mean:.5f} -> {oracle_after_mean:.5f} (Delta: {oracle_delta_mean:+.5f}, Win: {oracle_win_rate})")

if __name__ == "__main__":
    main()
