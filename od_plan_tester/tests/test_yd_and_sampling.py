"""
Tests for Moving-Bin Y_D Extraction (Oracle & Real Meta) and Distributional Overlap.
(Tests T22 to T26)
"""

from pathlib import Path
import pytest
import torch
import numpy as np
from od_plan_tester.project_adapter import (
    extract_yd_moving_oracle,
    extract_M1_city_oracle_obs,
    compute_distributional_overlap,
    sample_multinomial_yd,
)


@pytest.mark.reference
def test_yd_moving_oracle_assignment():
    """T22: extract_yd_moving_oracle produces shape (3,), sums to 1.0, and excludes intrazonal trips."""
    # 4 pairs: pair 0 is intrazonal (0,0, bin 0, trips=100); pairs 1,2,3 are interzonal in bins 1,2,3 with trips 10, 20, 30
    o_idx = torch.tensor([0, 0, 0, 0])
    d_idx = torch.tensor([0, 1, 2, 3])
    bin_labels = torch.tensor([0, 1, 2, 3])
    pair_trips = torch.tensor([100.0, 10.0, 20.0, 30.0])

    yd_moving = extract_yd_moving_oracle(pair_trips, bin_labels, o_idx, d_idx)

    assert yd_moving.shape == (3,)
    assert pytest.approx(1.0, rel=1e-6) == float(np.sum(yd_moving))
    # Interzonal total = 60 -> proportions: [10/60, 20/60, 30/60] = [1/6, 1/3, 1/2]
    np.testing.assert_allclose(yd_moving, [1.0 / 6.0, 1.0 / 3.0, 0.5], atol=1e-5)


@pytest.mark.contract
def test_M1_city_oracle_obs_meta_sum():
    """T23: extract_M1_city_oracle_obs from Meta mobility data produces shape (3,) and sums strictly to 1.0."""
    if not Path("meta_prior").exists():
        pytest.skip("meta_prior directory is not present in this workspace")
    sample_cities = ["Philadelphia", "Denver", "Raleigh"]
    for c_name in sample_cities:
        yd_real = extract_M1_city_oracle_obs(c_name, meta_prior_dir="meta_prior")
        assert yd_real is not None, f"Missing moving Meta Y_D for {c_name}"
        assert yd_real.shape == (3,)
        assert pytest.approx(1.0, rel=1e-5) == float(np.sum(yd_real))
        assert (yd_real >= 0.0).all()


@pytest.mark.reference
def test_distributional_overlap_bounds():
    """T24: compute_distributional_overlap (Overlap / CPC_dist) is in [0, 1] and 1.0 on identical distributions."""
    p = np.array([0.25, 0.50, 0.25])
    q = np.array([0.30, 0.40, 0.30])

    overlap_self = compute_distributional_overlap(p, p)
    overlap_pq = compute_distributional_overlap(p, q)

    assert pytest.approx(1.0, rel=1e-6) == overlap_self
    assert 0.0 <= overlap_pq <= 1.0
    # Overlap = min(0.25, 0.30) + min(0.50, 0.40) + min(0.25, 0.30) = 0.25 + 0.40 + 0.25 = 0.90
    assert pytest.approx(0.90, rel=1e-6) == overlap_pq


@pytest.mark.reference
def test_multinomial_sampling_stochastic_validity():
    """T25: Multinomial sampling produces valid distributions across seeds."""
    t_true = torch.tensor([50.0, 150.0, 300.0, 500.0])
    bin_labels = torch.tensor([0, 1, 2, 3])

    for m in [100, 1000, 10000]:
        yd_m = sample_multinomial_yd(t_true, bin_labels, m=m, seed=42)
        assert len(yd_m) == 4
        assert pytest.approx(1.0, rel=1e-5) == float(np.sum(yd_m))
        assert (yd_m >= 0.0).all()


@pytest.mark.reference
def test_multinomial_sampling_asymptotic_convergence():
    """T26: When m=inf, Multinomial sampling converges to the exact underlying empirical distribution."""
    t_true = torch.tensor([100.0, 200.0, 300.0, 400.0])
    bin_labels = torch.tensor([0, 1, 2, 3])

    yd_inf = sample_multinomial_yd(t_true, bin_labels, m=np.inf, seed=42)
    expected = np.array([0.1, 0.2, 0.3, 0.4])
    np.testing.assert_allclose(yd_inf, expected, atol=1e-6)
