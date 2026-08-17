"""
Tests for Interzonal Moving-Bin Calibration on Omega_c^+ (Soft KL Projection).
(Tests T16 to T21)
"""

import pytest
import torch
import numpy as np
from od_plan_tester.project_adapter import calibrate_moving_bins


@pytest.mark.scientific
def test_moving_mass_preservation():
    """T16: Interzonal mass preservation on Omega_c^+ within numerical tolerance."""
    torch.manual_seed(42)
    # 100 pairs: 10 intrazonal (bin 0), 90 interzonal (bins 1, 2, 3)
    o_idx = torch.randint(0, 20, (100,))
    d_idx = torch.randint(0, 20, (100,))
    # force first 10 to be intrazonal
    for i in range(10):
        d_idx[i] = o_idx[i]

    bins = torch.randint(1, 4, (100,))
    bins[:10] = 0

    t0 = torch.rand(100) * 100.0 + 1.0
    target_moving = np.array([0.30, 0.50, 0.20])

    t_cal = calibrate_moving_bins(t0, bins, o_idx, d_idx, target_moving, q=1.0)

    inter_mask = (o_idx != d_idx) & (bins > 0)
    inter_mass0 = t0[inter_mask].sum().item()
    inter_mass_cal = t_cal[inter_mask].sum().item()
    rel_err = abs(inter_mass_cal - inter_mass0) / inter_mass0

    assert rel_err < 1e-5, f"Interzonal mass preservation relative error {rel_err} exceeds 1e-5"


@pytest.mark.scientific
def test_intrazonal_identity():
    """T17: Intrazonal pairs (i == j, D == 0, bin 0) are strictly preserved identically: T_cal == T_zs."""
    torch.manual_seed(42)
    o_idx = torch.tensor([0, 1, 0, 2, 3, 3])
    d_idx = torch.tensor([0, 1, 1, 3, 2, 3])  # pairs 0, 1, 5 are intrazonal
    bins = torch.tensor([0, 0, 1, 2, 3, 0])
    t0 = torch.tensor([50.0, 120.0, 30.0, 80.0, 20.0, 200.0])
    target_moving = np.array([0.25, 0.50, 0.25])

    t_cal = calibrate_moving_bins(t0, bins, o_idx, d_idx, target_moving, q=1.0)

    intra_mask = (o_idx == d_idx) | (bins == 0)
    assert torch.allclose(t_cal[intra_mask], t0[intra_mask], atol=1e-6)


@pytest.mark.contract
def test_q_zero_is_zero_shot_identity():
    """T18: At q=0, calibration outputs exactly zero-shot flows (T_cal == T_zs)."""
    torch.manual_seed(42)
    o_idx = torch.tensor([0, 0, 0])
    d_idx = torch.tensor([1, 2, 3])
    bins = torch.tensor([1, 2, 3])
    t0 = torch.tensor([10.0, 50.0, 100.0])
    target_moving = np.array([0.40, 0.40, 0.20])

    t_cal_0 = calibrate_moving_bins(t0, bins, o_idx, d_idx, target_moving, q=0.0)
    assert torch.allclose(t_cal_0, t0, atol=1e-6)


@pytest.mark.scientific
def test_q_one_matches_target_distribution():
    """T19: At q=1, active moving-bin proportions match target distribution within tolerance < 1e-5."""
    torch.manual_seed(42)
    o_idx = torch.tensor([0, 0, 0, 0, 0, 0])
    d_idx = torch.tensor([1, 2, 3, 4, 5, 6])
    bins = torch.tensor([1, 1, 2, 2, 3, 3])
    t0 = torch.tensor([10.0, 15.0, 40.0, 60.0, 100.0, 150.0])
    target_moving = np.array([0.20, 0.50, 0.30])

    t_cal_1 = calibrate_moving_bins(t0, bins, o_idx, d_idx, target_moving, q=1.0, tolerance=1e-5)

    inter_total = t_cal_1.sum().item()
    p_b1 = t_cal_1[bins == 1].sum().item() / inter_total
    p_b2 = t_cal_1[bins == 2].sum().item() / inter_total
    p_b3 = t_cal_1[bins == 3].sum().item() / inter_total

    assert abs(p_b1 - 0.20) < 1e-5
    assert abs(p_b2 - 0.50) < 1e-5
    assert abs(p_b3 - 0.30) < 1e-5


@pytest.mark.reference
def test_q_monotonic_soft_response():
    """T20: Soft multiplier w_k(q) = (p_cond / p_implied)^q smoothly interpolates between q=0 and q=1."""
    o_idx = torch.tensor([0, 0])
    d_idx = torch.tensor([1, 2])
    bins = torch.tensor([1, 2])
    t0 = torch.tensor([10.0, 90.0])  # implied: [0.1, 0.9]
    target_moving = np.array([0.5, 0.5, 0.0])  # target for bins 1, 2, 3

    t_q0 = calibrate_moving_bins(t0, bins, o_idx, d_idx, target_moving, q=0.0)
    t_q5 = calibrate_moving_bins(t0, bins, o_idx, d_idx, target_moving, q=0.5)
    t_q1 = calibrate_moving_bins(t0, bins, o_idx, d_idx, target_moving, q=1.0)

    # Bin 1 flow must strictly increase with q
    assert t_q0[0].item() < t_q5[0].item() < t_q1[0].item()
    # Bin 2 flow must strictly decrease with q
    assert t_q0[1].item() > t_q5[1].item() > t_q1[1].item()


@pytest.mark.reference
def test_inactive_bin_conditioning():
    """T21: For cities with diameter < 100 km (bin 3 absent), target is conditioned on active moving bins."""
    o_idx = torch.tensor([0, 0])
    d_idx = torch.tensor([1, 2])
    bins = torch.tensor([1, 2])  # only bins 1 and 2 present
    t0 = torch.tensor([30.0, 70.0])
    target_moving = np.array([0.40, 0.40, 0.20])  # has 0.20 on bin 3

    t_cal = calibrate_moving_bins(t0, bins, o_idx, d_idx, target_moving, q=1.0)

    # Conditioned target on {1, 2} is [0.4/0.8, 0.4/0.8] = [0.5, 0.5]
    inter_total = t_cal.sum().item()
    p_b1 = t_cal[bins == 1].sum().item() / inter_total
    p_b2 = t_cal[bins == 2].sum().item() / inter_total

    assert pytest.approx(0.5, abs=1e-5) == p_b1
    assert pytest.approx(0.5, abs=1e-5) == p_b2
