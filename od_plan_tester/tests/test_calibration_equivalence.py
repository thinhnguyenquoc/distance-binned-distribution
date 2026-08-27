"""
Test to ensure manual Partial-OD calibration is exactly numerically equivalent 
to the canonical `calibrate_kbins` production operator.
"""

import pytest
import numpy as np

def test_manual_calibration_equivalence_to_canonical():
    """
    T27: Ensure any manual multiplicative calibration is numerically equivalent 
    to the canonical `calibrate_kbins` operator.
    """
    from src.calibration.bin_calibration import calibrate_kbins
    
    # 1. Setup mock data
    N = 100
    t0_np = np.random.uniform(1, 100, size=N)
    dist_km = np.random.uniform(1, 150, size=N)
    
    # Randomly assign pairs to interzonal
    inter_mask = np.random.rand(N) > 0.2
    
    # 3 target bins
    bin_edges = np.array([0.0, 10.0, 100.0, float('inf')])
    yd_target = np.array([0.5, 0.4, 0.1])
    
    # 2. Run canonical operator
    t_cal_canonical = calibrate_kbins(
        t0_np=t0_np.copy(),
        dist_km=dist_km,
        inter_mask=inter_mask,
        yd_target=yd_target,
        bin_edges=bin_edges,
        q=1.0
    )
    
    # 3. Emulate manual operator correctly (using active mask instead of target mask)
    t_cal_manual = t0_np.copy()
    n_inter = np.sum(t0_np[inter_mask])
    
    implied_b = np.zeros(3)
    active_mask = np.zeros(3, dtype=bool)
    
    # Determine bins
    bins = np.digitize(dist_km, bin_edges) - 1
    
    for k in range(3):
        mask_k = inter_mask & (bins == k)
        implied_b[k] = np.sum(t0_np[mask_k])
        active_mask[k] = np.any(mask_k)
        
    p_active = yd_target * active_mask
    p_cond = p_active / np.sum(p_active)
    
    implied_p = implied_b / n_inter
    
    w = np.zeros(3)
    for k in range(3):
        if active_mask[k] and implied_p[k] > 0:
            w[k] = p_cond[k] / implied_p[k]
        else:
            w[k] = 1.0
            
    weighted_mass = np.sum(implied_p * w)
    s = w / weighted_mass
    
    for k in range(3):
        mask_k = inter_mask & (bins == k)
        if np.any(mask_k):
            t_cal_manual[mask_k] *= s[k]
            
    # Rescale interzonal strictly
    cal_mass = np.sum(t_cal_manual[inter_mask])
    t_cal_manual[inter_mask] *= (n_inter / cal_mass)
    
    # 4. Assert exact numerical equivalence
    np.testing.assert_allclose(t_cal_manual, t_cal_canonical, atol=1e-5, rtol=1e-5)
