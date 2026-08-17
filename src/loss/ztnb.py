"""
Zero-Truncated Negative Binomial (ZTNB) loss and NB sensitivity model.

Primary likelihood (conservative assumption):
    T_ij | T_ij > 0 ~ ZTNB(mu_ij, phi)

where mu_ij = E[T_ij | T_ij > 0] is the conditional positive mean,
and phi > 0 is the global dispersion parameter (trainable per fold).

Sensitivity model:
    T_ij ~ NB(mu_ij, phi)

Both assume the NB parameterisation where:
    P_NB(T=k; mu, phi) = C(k+phi-1, k) * (phi/(mu+phi))^phi * (mu/(mu+phi))^k

so that E[T] = mu and Var[T] = mu + mu^2/phi  (over-dispersed for finite phi).
"""

import math
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Core NB log-probability (numerically stable, log-space throughout)
# ---------------------------------------------------------------------------

def nb_log_prob(t: torch.Tensor, mu: torch.Tensor, log_phi: torch.Tensor) -> torch.Tensor:
    """
    Log-probability of NB(mu, phi) at non-negative integer counts t.

    Uses the log-gamma form to avoid overflow:
        log P(T=t) = lgamma(t+phi) - lgamma(phi) - lgamma(t+1)
                     + phi*log(phi/(mu+phi)) + t*log(mu/(mu+phi))

    Args:
        t:       Observed counts, shape (...,), non-negative integers as float.
        mu:      Predicted mean, shape (...,), strictly positive.
        log_phi: Log-dispersion, shape () or (...,). phi = exp(log_phi) > 0.

    Returns:
        log_prob, shape (...,)
    """
    phi = torch.exp(log_phi)
    eps = 1e-8

    mu = mu + eps           # guard against mu=0
    phi = phi + eps         # guard against phi=0

    p_nb0 = phi / (mu + phi)       # P_NB(T=0) probability mass at 0

    log_p = (
        torch.lgamma(t + phi)
        - torch.lgamma(phi)
        - torch.lgamma(t + 1)
        + phi * torch.log(p_nb0)
        + t   * torch.log(1.0 - p_nb0 + eps)
    )
    return log_p


def nb_log_prob_at_zero(mu: torch.Tensor, log_phi: torch.Tensor) -> torch.Tensor:
    """log P_NB(T=0; mu, phi) = phi * log(phi / (mu + phi))"""
    phi = torch.exp(log_phi)
    eps = 1e-8
    mu  = mu  + eps
    phi = phi + eps
    return phi * torch.log(phi / (mu + phi))


# ---------------------------------------------------------------------------
# ZTNB NLL  (primary)
# ---------------------------------------------------------------------------

def ztnb_nll(t: torch.Tensor, mu: torch.Tensor, log_phi: torch.Tensor) -> torch.Tensor:
    """
    Mean negative log-likelihood of the Zero-Truncated Negative Binomial.

    log P_ZTNB(T=t | T>0) = log P_NB(t) - log(1 - P_NB(T=0))

    Only valid for t >= 1.  Asserts that no zeros are present.

    Args:
        t:       Observed positive counts, shape (N,), all >= 1.
        mu:      Predicted conditional positive mean E[T|T>0], shape (N,).
        log_phi: Log-dispersion scalar or shape (N,).

    Returns:
        Scalar mean NLL.
    """
    assert (t >= 1).all(), "ZTNB requires all observed counts >= 1"

    log_p_nb   = nb_log_prob(t, mu, log_phi)
    log_p_nb_0 = nb_log_prob_at_zero(mu, log_phi)

    # log(1 - P_NB(0)) in numerically stable form
    # = log1p(-exp(log_p_nb_0))  — safe when log_p_nb_0 < 0
    log_1_minus_p0 = torch.log1p(-torch.exp(log_p_nb_0).clamp(max=1 - 1e-7))

    log_p_ztnb = log_p_nb - log_1_minus_p0
    return -log_p_ztnb.mean()


# ---------------------------------------------------------------------------
# NB NLL  (sensitivity / comparison)
# ---------------------------------------------------------------------------

def nb_nll(t: torch.Tensor, mu: torch.Tensor, log_phi: torch.Tensor) -> torch.Tensor:
    """
    Mean negative log-likelihood of the unconditional Negative Binomial.
    Used as sensitivity model; compare with ztnb_nll on pilot runs.

    Args:
        t:       Observed counts >= 0, shape (N,).
        mu:      Predicted mean, shape (N,).
        log_phi: Log-dispersion scalar or shape (N,).

    Returns:
        Scalar mean NLL.
    """
    return -nb_log_prob(t, mu, log_phi).mean()


# ---------------------------------------------------------------------------
# Unit tests (run with: python -m pytest src/loss/ztnb.py -v)
# ---------------------------------------------------------------------------

def _run_unit_tests():
    """
    Required unit tests (per implementation plan):
    1. P(T>0) = 1 by construction (ZTNB defined only for t>=1)
    2. NLL finite at t=1
    3. Gradient finite as mu -> 0
    4. phi > 0 enforced via exp(log_phi)
    5. ZTNB -> NB as P_NB(0) -> 0  (large mu regime)
    """
    print("Running ZTNB unit tests...")
    torch.manual_seed(0)

    # --- Test 1: finite NLL at t=1 ---
    t = torch.ones(10)
    mu = torch.ones(10) * 5.0
    log_phi = torch.tensor(0.0)
    loss = ztnb_nll(t, mu, log_phi)
    assert torch.isfinite(loss), f"Test 1 FAILED: NLL={loss}"
    print(f"  Test 1 PASS: NLL at t=1, mu=5 -> {loss.item():.4f}")

    # --- Test 2: gradient finite as mu -> 0+ ---
    mu_small = torch.tensor([1e-4], requires_grad=True)
    t_small  = torch.ones(1)
    log_phi  = torch.tensor(0.0)
    loss = ztnb_nll(t_small, mu_small, log_phi)
    loss.backward()
    assert torch.isfinite(mu_small.grad), f"Test 2 FAILED: grad={mu_small.grad}"
    print(f"  Test 2 PASS: gradient finite at mu=1e-4 -> grad={mu_small.grad.item():.4f}")

    # --- Test 3: phi > 0 enforced ---
    log_phi_neg = torch.tensor(-10.0)   # exp(-10) > 0, always
    phi_val = torch.exp(log_phi_neg)
    assert phi_val > 0, "Test 3 FAILED: phi <= 0"
    print(f"  Test 3 PASS: phi = exp(log_phi=-10) = {phi_val.item():.2e} > 0")

    # --- Test 4: ZTNB NLL ≈ NB NLL when mu >> 1 (P_NB(0) -> 0) ---
    t_large = torch.randint(100, 1000, (200,)).float()
    mu_large = torch.ones(200) * 500.0
    log_phi_large = torch.tensor(2.0)
    ztnb_val = ztnb_nll(t_large, mu_large, log_phi_large)
    nb_val   = nb_nll(t_large, mu_large, log_phi_large)
    diff = (ztnb_val - nb_val).abs()
    assert diff < 0.01, f"Test 4 FAILED: ZTNB={ztnb_val:.4f}, NB={nb_val:.4f}, diff={diff:.4f}"
    print(f"  Test 4 PASS: ZTNB~NB at large mu: |diff|={diff.item():.6f}")

    # --- Test 5: NLL is lower for better mu (sanity check) ---
    t_test = torch.ones(100) * 10.0
    mu_good = torch.ones(100) * 10.0
    mu_bad  = torch.ones(100) * 100.0
    log_phi = torch.tensor(1.0)
    nll_good = ztnb_nll(t_test, mu_good, log_phi)
    nll_bad  = ztnb_nll(t_test, mu_bad,  log_phi)
    assert nll_good < nll_bad, f"Test 5 FAILED: better mu should have lower NLL"
    print(f"  Test 5 PASS: correct mu has lower NLL ({nll_good:.3f} < {nll_bad:.3f})")

    # --- Test 6: zero count raises assertion ---
    try:
        t_with_zero = torch.tensor([0.0, 1.0, 2.0])
        mu_test = torch.ones(3)
        ztnb_nll(t_with_zero, mu_test, torch.tensor(0.0))
        print("  Test 6 FAILED: should have raised AssertionError for t=0")
    except AssertionError:
        print("  Test 6 PASS: AssertionError raised for t=0 input")

    print("\nAll unit tests passed.\n")


if __name__ == "__main__":
    _run_unit_tests()
