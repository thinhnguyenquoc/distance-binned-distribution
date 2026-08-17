"""
Tests for ZTNB Loss, Conditional Expectation, and Gravity Prior Oracle.
(Tests T01 to T06)
"""

import math
import pytest
import torch
import numpy as np
from od_plan_tester.project_adapter import (
    nb_log_prob,
    nb_log_prob_at_zero,
    ztnb_nll,
    compute_conditional_mean,
    GravityPrior,
)


@pytest.mark.reference
def test_ztnb_log_prob_zero_formula():
    """T01: Verify base NB log P(T=0) = phi * log(phi / (mu + phi))."""
    mu_vals = [0.5, 2.0, 10.0, 100.0]
    phi_vals = [0.1, 1.0, 5.0]

    for mu in mu_vals:
        for phi in phi_vals:
            log_phi = torch.tensor(math.log(phi))
            mu_t = torch.tensor([mu])
            log_p0 = nb_log_prob_at_zero(mu_t, log_phi).item()
            expected = phi * math.log(phi / (mu + phi))
            assert pytest.approx(expected, rel=1e-5) == log_p0


@pytest.mark.contract
@pytest.mark.scientific
def test_ztnb_nll_strictly_positive_support():
    """T02: ZTNB NLL enforces T >= 1 and computes finite exact likelihood on positive counts."""
    t_pos = torch.tensor([1.0, 2.0, 5.0, 20.0])
    mu = torch.tensor([1.5, 2.0, 4.0, 18.0])
    log_phi = torch.tensor(0.0)

    loss = ztnb_nll(t_pos, mu, log_phi)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0

    # Ensure zero counts trigger assertion error
    t_with_zero = torch.tensor([0.0, 1.0, 2.0])
    with pytest.raises(AssertionError):
        ztnb_nll(t_with_zero, torch.ones(3), log_phi)


@pytest.mark.reference
def test_ztnb_conditional_mean_strictly_greater():
    """T03: Conditional positive expectation E[T | T >= 1] = mu / (1 - P(0)) is strictly > mu."""
    mu = torch.tensor([0.1, 1.0, 5.0, 20.0])
    log_phi = torch.tensor(math.log(2.0))
    c_mean = compute_conditional_mean(mu, log_phi)

    assert (c_mean > mu).all()
    # Manual check for mu=1.0, phi=2.0
    # P(0) = (2 / (1 + 2))^2 = 4/9
    # E[T|T>=1] = 1.0 / (1 - 4/9) = 9/5 = 1.8
    expected_1 = 1.0 / (1.0 - (2.0 / 3.0) ** 2)
    assert pytest.approx(expected_1, rel=1e-4) == c_mean[1].item()


@pytest.mark.reference
def test_ztnb_conditional_mean_asymptotics():
    """T04: As mu -> infinity, P(0) -> 0 and E[T | T >= 1] -> mu."""
    mu_large = torch.tensor([10000.0, 50000.0])
    log_phi = torch.tensor(0.0)  # phi = 1.0
    c_mean = compute_conditional_mean(mu_large, log_phi)

    rel_diff = torch.abs(c_mean - mu_large) / mu_large
    assert (rel_diff < 1e-3).all()


@pytest.mark.reference
def test_gravity_prior_formula_and_decay():
    """T05: Gravity model formula: log T = G + log Pi + log Pj - alpha * log D."""
    grav = GravityPrior(init_G=0.5, init_alpha=1.5)
    pi = torch.tensor([1000.0, 2000.0])
    pj = torch.tensor([500.0, 4000.0])
    d = torch.tensor([10.0, 20.0])

    log_t = grav(pi, pj, d)
    expected_0 = 0.5 + math.log(1000.0) + math.log(500.0) - 1.5 * math.log(10.0)
    assert pytest.approx(expected_0, rel=1e-4) == log_t[0].item()

    # Verify distance decay: larger distance yields smaller predicted flow
    d_near = torch.tensor([5.0])
    d_far = torch.tensor([50.0])
    p_const = torch.tensor([1000.0])
    assert grav(p_const, p_const, d_near).item() > grav(p_const, p_const, d_far).item()


@pytest.mark.contract
def test_gravity_prior_learnable_gradients():
    """T06: Trainable shared gravity parameters G and log_alpha produce finite valid gradients."""
    grav = GravityPrior()
    pi = torch.tensor([1000.0])
    pj = torch.tensor([1000.0])
    d = torch.tensor([5.0])

    out = grav(pi, pj, d)
    loss = (out - torch.tensor([10.0])) ** 2
    loss.backward()

    assert grav.G.grad is not None and torch.isfinite(grav.G.grad)
    assert grav.log_alpha.grad is not None and torch.isfinite(grav.log_alpha.grad)
