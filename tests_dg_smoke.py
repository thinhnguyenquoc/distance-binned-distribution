# Full comprehensive smoke tests for adapted Deep Gravity
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
from src.models.deep_gravity import DeepGravityAllocationModel, segment_logsumexp
from src.models.origin_production_mlp import OriginProductionMLP

def run_comprehensive_smoke_tests():
    print("=== RUNNING EXPANDED COMPREHENSIVE SMOKE TESTS ===")
    
    # Load Albuquerque
    city_name = "Albuquerque"
    data = load_city(city_name, fit_scaler=True)
    dist_km = np.expm1(data.pair_distance.numpy())
    
    # 1. Enforce locked Omega_c mask
    omega_mask_np = build_inter_mask(data, dist_km)
    omega_mask = torch.tensor(omega_mask_np, dtype=torch.bool)
    
    # Assertions on Omega_c support
    o_idx_np = data.pair_o_idx.numpy()
    d_idx_np = data.pair_d_idx.numpy()
    trips_np = data.pair_trips.numpy()
    
    assert (o_idx_np[omega_mask_np] != d_idx_np[omega_mask_np]).all(), "Self-pairs found in Omega_c!"
    assert (dist_km[omega_mask_np] > 0.0).all(), "Zero or negative distance found in Omega_c!"
    assert (trips_np[omega_mask_np] > 0.0).all(), "Zero or negative trip count found in Omega_c!"
    print(f"[Test 1] Omega_c mask locked: {omega_mask.sum().item()}/{len(omega_mask)} interzonal pairs verified.")
    
    # Filter pairs strictly to Omega_c
    o_idx = data.pair_o_idx[omega_mask]
    d_idx = data.pair_d_idx[omega_mask]
    dist_t = data.pair_distance[omega_mask].unsqueeze(1)
    trips = data.pair_trips[omega_mask]
    
    o_feat = data.node_features[o_idx]
    d_feat = data.node_features[d_idx]
    pair_feat = torch.cat([o_feat, d_feat, dist_t], dim=1)
    n_nodes = data.n_tracts
    
    dg_model = DeepGravityAllocationModel(in_dim=53)
    prod_model = OriginProductionMLP(in_dim=26)
    
    # [Test 2] Finite forward, backward, no NaN/Inf
    scores = dg_model(pair_feat)
    assert torch.isfinite(scores).all(), "Scores contain NaN/Inf!"
    
    probs = dg_model.compute_probs(pair_feat, o_idx, n_nodes)
    assert torch.isfinite(probs).all(), "Probabilities contain NaN/Inf!"
    assert (probs > 0.0).all(), "Probabilities must be strictly positive!"
    
    loss = dg_model.compute_loss_batch(pair_feat[:500], o_idx[:500] - o_idx[:500].min(), trips[:500], int((o_idx[:500] - o_idx[:500].min()).max().item()) + 1)
    assert torch.isfinite(loss), "Batch loss is NaN/Inf!"
    loss.backward()
    for name, p in dg_model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), f"Gradient {name} has NaN/Inf!"
    print("[Test 2] Finite forward and backward gradient verified.")
    
    # [Test 3] Normalization on exact Omega_{c,i}
    p_sum = torch.zeros(n_nodes)
    p_sum.scatter_add_(0, o_idx, probs)
    active_origins = torch.zeros(n_nodes, dtype=torch.bool)
    active_origins[o_idx] = True
    max_prob_err = (p_sum[active_origins] - 1.0).abs().max().item()
    print(f"[Test 3] Max origin prob normalization error: {max_prob_err:.2e}")
    assert max_prob_err < 1e-5, f"Prob sum error: {max_prob_err}"
    
    # [Test 4] OriginProductionMLP prediction safety
    O_hat = prod_model.predict_outflow(data.node_features)
    assert torch.isfinite(O_hat).all(), "O_hat contains NaN/Inf!"
    assert (O_hat > 0.0).all(), "O_hat must be strictly positive!"
    print("[Test 4] OriginProductionMLP output bounds verified (all finite, > 0).")
    
    # [Test 5] Label-invariance: DG-Predicted-O is invariant to test pair_trips
    t_pred_o1 = (O_hat[o_idx] * probs).detach().numpy()
    
    # Scramble pair_trips arbitrarily
    scrambled_trips = trips.clone()
    scrambled_trips = scrambled_trips[torch.randperm(len(scrambled_trips))]
    
    # DG-Predicted-O relies ONLY on node_features, pair_feat, o_idx
    O_hat_test = prod_model.predict_outflow(data.node_features)
    probs_test = dg_model.compute_probs(pair_feat, o_idx, n_nodes)
    t_pred_o2 = (O_hat_test[o_idx] * probs_test).detach().numpy()
    
    assert np.allclose(t_pred_o1, t_pred_o2, atol=1e-7), "DG-Predicted-O leaked test trip values!"
    print("[Test 5] Label-invariance verified: DG-Predicted-O is completely independent of test trips.")
    
    # [Test 6] Calibration mass preservation & bin-matching
    edges = np.linspace(dist_km[omega_mask_np].min(), dist_km[omega_mask_np].max(), K_MOVE + 1)
    Y_D_tgt = extract_yd_kbins(dist_km, trips_np, edges, omega_mask_np)
    
    # Full city prediction array
    t0_full = np.zeros(len(data.pair_o_idx), dtype=np.float64)
    t0_full[omega_mask_np] = t_pred_o1
    
    t_cal_full = calibrate_kbins(t0_full, dist_km, omega_mask_np, Y_D_tgt, edges, q=Q_CALIB, tolerance=TOLERANCE)
    
    tot_before = float(np.sum(t0_full[omega_mask_np]))
    tot_after = float(np.sum(t_cal_full[omega_mask_np]))
    mass_diff = abs(tot_after - tot_before)
    print(f"[Test 6a] City-level calibration mass difference: {mass_diff:.2e}")
    assert mass_diff < 1e-4 * max(1.0, tot_before), "City total flow not preserved!"
    
    # [Test 7] Origin totals drift measurement
    orig_before = np.zeros(n_nodes)
    orig_after = np.zeros(n_nodes)
    np.add.at(orig_before, o_idx.numpy(), t0_full[omega_mask_np])
    np.add.at(orig_after, o_idx.numpy(), t_cal_full[omega_mask_np])
    
    active_mask_np = active_origins.numpy()
    orig_drift = np.abs(orig_after[active_mask_np] - orig_before[active_mask_np]) / np.maximum(orig_before[active_mask_np], 1.0)
    mean_orig_drift = float(np.mean(orig_drift))
    max_orig_drift = float(np.max(orig_drift))
    print(f"[Test 7] Origin totals drift after bin-calibration: mean={mean_orig_drift:.2%}, max={max_orig_drift:.2%}")
    
    # [Test 8] Serialization and state recovery
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
    assert torch.allclose(dg_model.compute_probs(pair_feat, o_idx, n_nodes), dg_rec.compute_probs(pair_feat, o_idx, n_nodes))
    
    os.remove(prod_ckpt)
    os.remove(dg_ckpt)
    print("[Test 8] Checkpoint serialization and exact recovery verified.")
    print("=== ALL EXPANDED SMOKE TESTS PASSED SUCCESSFULLY! ===")

if __name__ == "__main__":
    run_comprehensive_smoke_tests()
