"""
Tests for Y_D Extraction (Oracle & Real Meta) and Multinomial Sampling.
(Tests T22 to T26)
"""

import pytest
import torch
import numpy as np
from od_plan_tester.project_adapter import (
    extract_yd_oracle,
    extract_yd_real,
    sample_multinomial_yd,
)


@pytest.mark.reference
def test_yd_oracle_exact_bin_assignment():
    """T22: Y_D^oracle aggregates ground truth flow volume exactly into the 4 distance bins."""
    # Pairs with flows [10, 20, 30, 40] in bins [0, 1, 2, 3] -> total = 100
    t_true = torch.tensor([10.0, 20.0, 30.0, 40.0])
    bin_labels = torch.tensor([0, 1, 2, 3])

    yd = extract_yd_oracle(t_true, bin_labels)
    expected = np.array([0.1, 0.2, 0.3, 0.4])
    np.testing.assert_allclose(yd, expected, atol=1e-5)


@pytest.mark.reference
def test_yd_oracle_sums_to_one():
    """T23: Extracted Y_D^oracle is a valid probability vector summing to 1.0."""
    t_true = torch.tensor([5.0, 15.0, 25.0])
    bin_labels = torch.tensor([0, 1, 1])

    yd = extract_yd_oracle(t_true, bin_labels)
    assert pytest.approx(1.0, rel=1e-6) == float(np.sum(yd))
    assert (yd >= 0.0).all()


@pytest.mark.contract
def test_yd_real_meta_4bins_sum():
    """T24: Extracted Y_D^real from Meta mobility data has 4 bins and sums to 1.0 across sample cities."""
    sample_cities = ["Philadelphia", "Denver", "Raleigh"]
    for c_name in sample_cities:
        yd_real = extract_yd_real(c_name, meta_prior_dir="meta_prior")
        assert yd_real is not None, f"Missing Meta Y_D for {c_name}"
        assert len(yd_real) == 4
        assert pytest.approx(1.0, rel=1e-4) == float(np.sum(yd_real))
        assert (yd_real >= 0.0).all()


@pytest.mark.reference
def test_multinomial_sampling_stochastic_validity():
    """T25: Multinomial sampling at finite m produces a valid 4-bin probability distribution."""
    t_true = torch.tensor([50.0, 150.0, 300.0, 500.0])
    bin_labels = torch.tensor([0, 1, 2, 3])

    for m in [100, 1000, 10000]:
        yd_m = sample_multinomial_yd(t_true, bin_labels, m=m, seed=42)
        assert len(yd_m) == 4
        assert pytest.approx(1.0, rel=1e-5) == float(np.sum(yd_m))
        assert (yd_m >= 0.0).all()


@pytest.mark.reference
def test_multinomial_sampling_asymptotic_convergence():
    """T26: When m=inf, Multinomial sampling exactly reproduces the exact Y_D^oracle."""
    t_true = torch.tensor([100.0, 200.0, 300.0, 400.0])
    bin_labels = torch.tensor([0, 1, 2, 3])

    yd_oracle = extract_yd_oracle(t_true, bin_labels)
    yd_inf = sample_multinomial_yd(t_true, bin_labels, m=np.inf, seed=42)

    np.testing.assert_allclose(yd_inf, yd_oracle, atol=1e-6)
