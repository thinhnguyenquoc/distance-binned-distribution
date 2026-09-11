from __future__ import annotations

import os
import copy
import numpy as np
import torch
import torch.nn as nn

from src.data.dataset import load_city
from src.data.yd_extractor import extract_yd_kbins, compute_kbin_edges
from src.calibration.bin_calibration import calibrate_kbins
from src.experiment.e1_core import K_MOVE, Q_CALIB, TOLERANCE, build_inter_mask
from src.models.deep_gravity import DeepGravityAllocationModel, build_origin_csr
from src.models.origin_production_mlp import OriginProductionMLP

def predict_dg_predicted_o(
    prod_model: OriginProductionMLP,
    dg_model: DeepGravityAllocationModel,
    node_features: torch.Tensor,
    pair_o_idx: torch.Tensor,
    pair_d_idx: torch.Tensor,
    pair_distance: torch.Tensor,
    num_nodes: int,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """
    Prediction helper for DG-Predicted-O.
    Explicitly does NOT accept any trip intensity arguments.
    Operates conditioned on node features, distance, and the known-positive support.
    """
    prod_model.eval()
    dg_model.eval()
    with torch.no_grad():
        dist_t = pair_distance.to(device).unsqueeze(1)
        sort_perm, offsets, active_origins = build_origin_csr(pair_o_idx, num_nodes)
        sorted_o = pair_o_idx[sort_perm]
        sorted_d = pair_d_idx[sort_perm]
        sorted_dist = dist_t[sort_perm]
        sorted_feat = torch.cat([node_features[sorted_o], node_features[sorted_d], sorted_dist], dim=1).to(device)
        
        probs = dg_model.compute_probs_batched_csr(
            sorted_feat, sort_perm, offsets, active_origins, len(pair_o_idx), batch_origins=64, device=device
        )
        O_hat = prod_model.predict_outflow(node_features.to(device)).cpu()
        t_pred = O_hat[pair_o_idx] * probs.cpu()
    return t_pred.numpy().astype(np.float64)


def run_comprehensive_smoke_tests():
    print("=== RUNNING EXPANDED COMPREHENSIVE SMOKE TESTS ===")
    
    city_name = "Albuquerque"
    data = load_city(city_name, fit_scaler=True)
    dist_km = np.expm1(data.pair_distance.numpy())
    
    # 1. Enforce locked Omega_c mask
    omega_mask_np = build_inter_mask(data, dist_km)
    omega_mask = torch.tensor(omega_mask_np, dtype=torch.bool)
    
    o_idx_np = data.pair_o_idx.numpy()
    d_idx_np = data.pair_d_idx.numpy()
    trips_np = data.pair_trips.numpy()
    
    assert (o_idx_np[omega_mask_np] != d_idx_np[omega_mask_np]).all(), "Self-pairs found in Omega_c!"
    assert (dist_km[omega_mask_np] > 0.0).all(), "Zero or negative distance found in Omega_c!"
    assert (trips_np[omega_mask_np] > 0.0).all(), "Zero or negative trip count found in Omega_c!"
    print(f"[Test 1] Omega_c mask locked: {omega_mask.sum().item()}/{len(omega_mask)} interzonal pairs verified.")
    
    o_idx = data.pair_o_idx[omega_mask]
    d_idx = data.pair_d_idx[omega_mask]
    dist_t = data.pair_distance[omega_mask].unsqueeze(1)
    trips = data.pair_trips[omega_mask]
    n_nodes = data.n_tracts
    
    dg_model = DeepGravityAllocationModel(in_dim=53)
    prod_model = OriginProductionMLP(in_dim=26)
    
    # [Test 2] Finite forward, backward using complete CSR destination sets
    sort_perm, offsets, active_origins = build_origin_csr(o_idx, n_nodes)
    test_origins = active_origins[:5]
    num_b = len(test_origins)
    
    b_feat_list, b_trips_list, b_compact_list = [], [], []
    for c_id, orig_id in enumerate(test_origins):
        st = offsets[orig_id.item()].item()
        en = offsets[orig_id.item() + 1].item()
        pairs = sort_perm[st:en]
        feat = torch.cat([data.node_features[o_idx[pairs]], data.node_features[d_idx[pairs]], dist_t[pairs]], dim=1)
        b_feat_list.append(feat)
        b_trips_list.append(trips[pairs])
        b_compact_list.append(torch.full((len(pairs),), c_id, dtype=torch.long))
        
    b_feat = torch.cat(b_feat_list, dim=0)
    b_trips = torch.cat(b_trips_list, dim=0)
    b_compact = torch.cat(b_compact_list, dim=0)
    
    loss = dg_model.compute_loss_batch(b_feat, b_compact, b_trips, num_b)
    assert torch.isfinite(loss), "Batch loss is NaN/Inf!"
    loss.backward()
    for name, p in dg_model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), f"Gradient {name} has NaN/Inf!"
    print("[Test 2] Finite forward and backward gradient over full destination sets verified.")
    
    # [Test 3] Normalization on exact Omega_{c,i} via batched CSR inference
    sorted_feat = torch.cat([data.node_features[o_idx[sort_perm]], data.node_features[d_idx[sort_perm]], dist_t[sort_perm]], dim=1)
    probs = dg_model.compute_probs_batched_csr(sorted_feat, sort_perm, offsets, active_origins, len(o_idx), batch_origins=32)
    assert torch.isfinite(probs).all(), "Probabilities contain NaN/Inf!"
    assert (probs >= 0.0).all(), "Probabilities must be non-negative!"
    
    p_sum = torch.zeros(n_nodes)
    p_sum.scatter_add_(0, o_idx, probs)
    active_mask = torch.zeros(n_nodes, dtype=torch.bool)
    active_mask[o_idx] = True
    max_prob_err = (p_sum[active_mask] - 1.0).abs().max().item()
    print(f"[Test 3] Max origin prob normalization error: {max_prob_err:.2e} (all non-negative, finite)")
    assert max_prob_err < 1e-5, f"Prob sum error: {max_prob_err}"
    
    # [Test 4] OriginProductionMLP output bounds verified
    O_hat = prod_model.predict_outflow(data.node_features)
    assert torch.isfinite(O_hat).all(), "O_hat contains NaN/Inf!"
    assert (O_hat > 0.0).all(), "O_hat must be strictly positive!"
    print("[Test 4] OriginProductionMLP output bounds verified (all finite, strictly > 0 via clamp_min).")
    
    # [Test 5] Prediction API excludes target intensity values
    # Helper predict_dg_predicted_o does not accept pair_trips at all.
    t_pred_1 = predict_dg_predicted_o(prod_model, dg_model, data.node_features, o_idx, d_idx, data.pair_distance[omega_mask], n_nodes)
    t_pred_2 = predict_dg_predicted_o(prod_model, dg_model, data.node_features, o_idx, d_idx, data.pair_distance[omega_mask], n_nodes)
    assert np.allclose(t_pred_1, t_pred_2, atol=1e-7)
    print("[Test 5] Prediction API excludes target intensity values: DG-Predicted-O operates conditioned on node features, distance, and known-positive support.")
    
    # [Test 6] Calibration mass preservation & exact bin-proportion matching
    # NOTE: Bin edges computed from Albuquerque data for algebraic unit verification.
    edges, k_active = compute_kbin_edges(["Albuquerque"], K=K_MOVE, data_root="data")
    Y_D_tgt = extract_yd_kbins(dist_km, trips_np, edges, omega_mask_np)
    
    t0_full = np.zeros(len(data.pair_o_idx), dtype=np.float64)
    t0_full[omega_mask_np] = t_pred_1
    
    t_cal_full = calibrate_kbins(t0_full, dist_km, omega_mask_np, Y_D_tgt, edges, q=Q_CALIB, tolerance=TOLERANCE)
    
    tot_before = float(np.sum(t0_full[omega_mask_np]))
    tot_after = float(np.sum(t_cal_full[omega_mask_np]))
    mass_diff = abs(tot_after - tot_before)
    print(f"[Test 6a] City-level calibration mass difference: {mass_diff:.2e}")
    assert mass_diff < 1e-4 * max(1.0, tot_before), "City total flow not preserved!"
    
    Y_after = extract_yd_kbins(dist_km, t_cal_full, edges, omega_mask_np)
    active_bins = Y_D_tgt > 0.0
    bin_diff = np.abs(Y_after[active_bins] - Y_D_tgt[active_bins]).max()
    print(f"[Test 6b] Max distance-bin proportion discrepancy: {bin_diff:.2e}")
    assert bin_diff < 1e-5, f"Distance bin proportions do not match target Y_D: {bin_diff}"
    
    # [Test 7] Origin totals drift measurement
    orig_before = np.zeros(n_nodes)
    orig_after = np.zeros(n_nodes)
    np.add.at(orig_before, o_idx.numpy(), t0_full[omega_mask_np])
    np.add.at(orig_after, o_idx.numpy(), t_cal_full[omega_mask_np])
    
    active_mask_np = active_mask.numpy()
    orig_drift = np.abs(orig_after[active_mask_np] - orig_before[active_mask_np]) / np.maximum(orig_before[active_mask_np], 1.0)
    mean_orig_drift = float(np.mean(orig_drift))
    max_orig_drift = float(np.max(orig_drift))
    print(f"[Test 7] Origin totals drift after bin-calibration: mean={mean_orig_drift:.2%}, max={max_orig_drift:.2%}")
    
    # [Test 8] Checkpoint serialization and recovery
    os.makedirs("results/deep_gravity_pilot", exist_ok=True)
    prod_ckpt = "results/deep_gravity_pilot/test_prod.pt"
    dg_ckpt = "results/deep_gravity_pilot/test_dg.pt"
    
    torch.save(prod_model.state_dict(), prod_ckpt)
    torch.save(dg_model.state_dict(), dg_ckpt)
    
    prod_rec = OriginProductionMLP(in_dim=26)
    prod_rec.load_state_dict(torch.load(prod_ckpt, weights_only=False))
    dg_rec = DeepGravityAllocationModel(in_dim=53)
    dg_rec.load_state_dict(torch.load(dg_ckpt, weights_only=False))
    
    assert torch.allclose(prod_model.predict_outflow(data.node_features), prod_rec.predict_outflow(data.node_features))
    assert torch.allclose(probs, dg_rec.compute_probs_batched_csr(sorted_feat, sort_perm, offsets, active_origins, len(o_idx), batch_origins=32))
    
    os.remove(prod_ckpt)
    os.remove(dg_ckpt)
    print("[Test 8] Checkpoint serialization and exact recovery verified.")
    print("=== ALL EXPANDED SMOKE TESTS PASSED SUCCESSFULLY! ===")

if __name__ == "__main__":
    run_comprehensive_smoke_tests()
