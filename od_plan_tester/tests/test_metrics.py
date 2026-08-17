"""
Tests for Evaluation Metrics Suite (CPC, CPC_norm, RMSE-log1p, Pearson r).
(Tests T27 to T31)
"""

import pytest
import torch
import numpy as np
from od_plan_tester.project_adapter import (
    compute_cpc,
    compute_cpc_norm,
    compute_rmse_log1p,
    compute_pearson_r,
    evaluate_all,
)


@pytest.mark.reference
def test_cpc_bounds_and_symmetry():
    """T27: CPC is bounded in [0, 1] and symmetric CPC(A, B) == CPC(B, A)."""
    t_true = torch.tensor([10.0, 50.0, 100.0, 500.0])
    t_pred = torch.tensor([15.0, 40.0, 120.0, 480.0])

    cpc1 = compute_cpc(t_true, t_pred)
    cpc2 = compute_cpc(t_pred, t_true)

    assert 0.0 <= cpc1 <= 1.0
    assert pytest.approx(cpc1, rel=1e-6) == cpc2

    # Perfect prediction has CPC = 1.0
    assert pytest.approx(1.0, rel=1e-6) == compute_cpc(t_true, t_true)


@pytest.mark.reference
def test_cpc_norm_1_minus_tvd():
    """T28: Normalized CPC matches 1 - 0.5 * sum |p_i - q_i| and is scale-invariant."""
    t_true = torch.tensor([10.0, 20.0, 30.0])
    t_pred = torch.tensor([20.0, 40.0, 60.0])  # identical shape, 2x scale

    cpc_norm = compute_cpc_norm(t_true, t_pred)
    assert pytest.approx(1.0, rel=1e-6) == cpc_norm


@pytest.mark.reference
def test_rmse_log1p_zero_on_identical():
    """T29: RMSE-log1p is strictly 0.0 on identical inputs and positive otherwise."""
    t_true = torch.tensor([5.0, 15.0, 50.0])
    t_pred = torch.tensor([10.0, 20.0, 60.0])

    assert pytest.approx(0.0, abs=1e-6) == compute_rmse_log1p(t_true, t_true)
    assert compute_rmse_log1p(t_true, t_pred) > 0.0


@pytest.mark.reference
def test_pearson_r_bounds():
    """T30: Pearson r is 1.0 on linear transformation and bounded in [-1, 1]."""
    t_true = torch.tensor([10.0, 20.0, 30.0, 40.0])
    t_pred = 3.0 * t_true + 5.0

    r = compute_pearson_r(t_true, t_pred)
    assert pytest.approx(1.0, rel=1e-5) == r


@pytest.mark.contract
def test_evaluate_all_contract():
    """T31: evaluate_all returns all locked primary and secondary metrics in a dictionary."""
    t_true = torch.tensor([10.0, 20.0, 30.0])
    t_pred = torch.tensor([12.0, 18.0, 35.0])

    res = evaluate_all(t_true, t_pred)
    assert isinstance(res, dict)
    assert "cpc" in res
    assert "cpc_norm" in res
    assert "rmse_log1p" in res
    assert "pearson_r" in res
    for v in res.values():
        assert isinstance(v, float)
