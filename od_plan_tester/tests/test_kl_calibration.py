"""
Tests for Support-Conditioned Mass-Preserving KL Calibration.
(Tests T16 to T21)
"""

import pytest
import torch
import numpy as np
from od_plan_tester.project_adapter import calibrate_by_distance_bins


@pytest.mark.scientific
def test_kl_mass_preservation_exact():
    """T16: Calibration exactly preserves total flow mass (sum T_cal == sum T_zs)."""
    torch.manual_seed(123)
    t0 = torch.rand(500) * 100.0 + 1.0
    bins = torch.randint(0, 4, (500,))
    p_target = np.array([0.15, 0.35, 0.40, 0.10])

    t_cal = calibrate_by_distance_bins(t0, bins, p_target, tolerance=1e-4)

    mass0 = t0.sum().item()
    mass_cal = t_cal.sum().item()
    rel_err = abs(mass_cal - mass0) / mass0

    assert rel_err < 1e-4, f"Mass preservation relative error {rel_err} exceeds 1e-4"


@pytest.mark.scientific
def test_kl_bin_proportions_exact():
    """T17: Calibration exactly matches target bin distribution Y_D on active bins."""
    torch.manual_seed(123)
    t0 = torch.rand(1000) * 50.0 + 1.0
    bins = torch.randint(0, 4, (1000,))
    p_target = np.array([0.20, 0.30, 0.40, 0.10])

    t_cal = calibrate_by_distance_bins(t0, bins, p_target, tolerance=1e-4)

    mass_cal = t_cal.sum().item()
    cal_proportions = np.array([(t_cal[bins == k].sum() / mass_cal).item() for k in range(4)])

    max_err = np.max(np.abs(cal_proportions - p_target))
    assert max_err < 1e-4, f"Bin matching error {max_err} exceeds 1e-4"


@pytest.mark.reference
def test_kl_support_conditioning_active_bins():
    """T18: Support-conditioning: If bin 3 has 0 pairs (city diameter < 100km), mass is re-normalized over bins 0, 1, 2."""
    t0 = torch.tensor([10.0, 30.0, 60.0])  # only bins 0, 1, 2
    bins = torch.tensor([0, 1, 2])
    p_target_4bin = np.array([0.20, 0.30, 0.30, 0.20])  # bin 3 has 0.20 weight

    t_cal = calibrate_by_distance_bins(t0, bins, p_target_4bin)

    assert t_cal.sum().item() == pytest.approx(100.0, rel=1e-4)
    # Target conditioned on bins 0, 1, 2 has sum 0.80 -> normalized weights: [0.20/0.8, 0.30/0.8, 0.30/0.8] = [0.25, 0.375, 0.375]
    prop0 = (t_cal[bins == 0].sum() / t_cal.sum()).item()
    prop1 = (t_cal[bins == 1].sum() / t_cal.sum()).item()
    prop2 = (t_cal[bins == 2].sum() / t_cal.sum()).item()

    assert pytest.approx(0.25, abs=1e-3) == prop0
    assert pytest.approx(0.375, abs=1e-3) == prop1
    assert pytest.approx(0.375, abs=1e-3) == prop2


@pytest.mark.contract
def test_kl_zero_shot_all_zero_safe_handling():
    """T19: All-zero flow prediction is safely returned without NaN/inf."""
    t0 = torch.zeros(10)
    bins = torch.zeros(10, dtype=torch.long)
    p_target = np.array([0.25, 0.25, 0.25, 0.25])

    t_cal = calibrate_by_distance_bins(t0, bins, p_target)
    assert torch.equal(t0, t_cal)
    assert not torch.isnan(t_cal).any()


@pytest.mark.scientific
def test_kl_no_epsilon_distortion_invariant():
    """T20: Mathematical correction: Exact invariant hold without epsilon distortion."""
    t0 = torch.tensor([100.0, 50.0, 50.0])
    bins = torch.tensor([0, 1, 2])
    p_target = np.array([0.5, 0.3, 0.2, 0.0])

    t_cal = calibrate_by_distance_bins(t0, bins, p_target, eps=1e-8)
    assert pytest.approx(200.0, rel=1e-4) == t_cal.sum().item()
    assert pytest.approx(100.0, rel=1e-3) == t_cal[0].item()
    assert pytest.approx(60.0, rel=1e-3) == t_cal[1].item()
    assert pytest.approx(40.0, rel=1e-3) == t_cal[2].item()


@pytest.mark.contract
def test_kl_multiplicative_scaling_property():
    """T21: Calibrated flows are scalar multiples of zero-shot flows within each bin."""
    t0 = torch.tensor([10.0, 20.0, 30.0, 40.0])
    bins = torch.tensor([0, 0, 1, 1])
    p_target = np.array([0.4, 0.6, 0.0, 0.0])

    t_cal = calibrate_by_distance_bins(t0, bins, p_target)
    # Ratio within bin 0 must be constant
    ratio_b0_1 = (t_cal[0] / t0[0]).item()
    ratio_b0_2 = (t_cal[1] / t0[1]).item()
    assert pytest.approx(ratio_b0_1, rel=1e-5) == ratio_b0_2

    # Ratio within bin 1 must be constant
    ratio_b1_1 = (t_cal[2] / t0[2]).item()
    ratio_b1_2 = (t_cal[3] / t0[3]).item()
    assert pytest.approx(ratio_b1_1, rel=1e-5) == ratio_b1_2
