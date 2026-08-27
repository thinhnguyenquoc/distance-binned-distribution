r"""
Zero-Truncated Negative Binomial (ZTNB) Likelihood and Conditional Mean Conversion.

Exact Mathematical Formulation:
    Base NB Distribution:
        T ~ NB(mu_nb, phi) with mean mu_nb > 0 and dispersion phi > 0.
        P_NB(T=0) = (phi / (mu_nb + phi))^phi.

    Zero-Truncated NB (ZTNB):
        P_ZTNB(T=t | T >= 1) = P_NB(t; mu_nb, phi) / (1 - P_NB(0; mu_nb, phi))

    Conditional Expected Flow:
        E[T | T >= 1] = mu_nb / (1 - P_NB(0; mu_nb, phi))

The neural network outputs mu_nb > 0.
At training time: loss is -log P_ZTNB(T_ij; mu_nb, phi).
At inference time: predicted flow is \hat{T}^{ZS}_ij = E[T_ij | T_ij >= 1].
"""

import math
import torch
import torch.nn.functional as F


def nb_log_prob(t: torch.Tensor, mu_nb: torch.Tensor, log_phi: torch.Tensor) -> torch.Tensor:
    """
    Log-probability of base NB(mu_nb, phi) at integer count t.
    """
    log_phi_safe = torch.clamp(log_phi, min=-10.0, max=10.0)
    phi = torch.exp(log_phi_safe)
    eps = 1e-8

    mu = mu_nb + eps
    phi = phi + eps

    p_nb0 = phi / (mu + phi)  # probability parameter

    log_p = (
        torch.lgamma(t + phi)
        - torch.lgamma(phi)
        - torch.lgamma(t + 1)
        + phi * torch.log(p_nb0)
        + t * torch.log(1.0 - p_nb0 + eps)
    )
    return log_p


def nb_log_prob_at_zero(mu_nb: torch.Tensor, log_phi: torch.Tensor) -> torch.Tensor:
    """log P_NB(T=0; mu_nb, phi) = phi * log(phi / (mu_nb + phi))"""
    log_phi_safe = torch.clamp(log_phi, min=-10.0, max=10.0)
    phi = torch.exp(log_phi_safe)
    eps = 1e-8
    mu = mu_nb + eps
    phi = phi + eps
    return phi * torch.log(phi / (mu + phi))


def ztnb_nll(t: torch.Tensor, mu_nb: torch.Tensor, log_phi: torch.Tensor) -> torch.Tensor:
    """
    Exact Negative Log-Likelihood for Zero-Truncated Negative Binomial.
    log P_ZTNB(T=t | T>=1) = log P_NB(t; mu_nb, phi) - log(1 - P_NB(0; mu_nb, phi))
    """
    assert (t >= 1).all(), "ZTNB requires all observed counts >= 1"

    log_p_nb = nb_log_prob(t, mu_nb, log_phi)
    log_p_nb_0 = nb_log_prob_at_zero(mu_nb, log_phi)

    # Numerically stable log(1 - P_NB(0)) = log1p(-exp(log_p_nb_0))
    log_1_minus_p0 = torch.log1p(-torch.exp(log_p_nb_0).clamp(max=1.0 - 1e-7))

    log_p_ztnb = log_p_nb - log_1_minus_p0
    return -log_p_ztnb.mean()


def nb_nll(t: torch.Tensor, mu_nb: torch.Tensor, log_phi: torch.Tensor) -> torch.Tensor:
    """
    Mean negative log-likelihood of unconditional Negative Binomial (sensitivity model).
    """
    return -nb_log_prob(t, mu_nb, log_phi).mean()


def compute_conditional_mean(mu_nb: torch.Tensor, log_phi: torch.Tensor) -> torch.Tensor:
    """
    Converts base NB mean mu_nb to conditional positive mean E[T | T >= 1].
    E[T | T >= 1] = mu_nb / (1 - P_NB(0; mu_nb, phi))
    """
    log_phi_safe = torch.clamp(log_phi, min=-10.0, max=10.0)
    phi = torch.exp(log_phi_safe)
    eps = 1e-8
    p0 = (phi / (mu_nb + phi + eps)) ** phi
    # Clamp 1-p0 to avoid division by zero when mu_nb is tiny
    denom = torch.clamp(1.0 - p0, min=1e-6)
    return mu_nb / denom


def _run_unit_tests():
    print("Running updated ZTNB unit tests...")
    torch.manual_seed(0)

    # Test 1: Conditional mean is strictly > mu_nb
    mu = torch.tensor([1.0, 5.0, 10.0])
    log_phi = torch.tensor(0.0)
    c_mean = compute_conditional_mean(mu, log_phi)
    assert (c_mean > mu).all(), "Test 1 FAILED: Conditional mean must be > base mu"
    print(f"  Test 1 PASS: mu={mu.tolist()} -> E[T|T>=1]={c_mean.tolist()}")

    # Test 2: NLL at t=1 is finite
    t1 = torch.ones(5)
    loss = ztnb_nll(t1, torch.ones(5) * 2.0, log_phi)
    assert torch.isfinite(loss), "Test 2 FAILED: NLL not finite"
    print(f"  Test 2 PASS: NLL at t=1 -> {loss.item():.4f}")

    # Test 3: Gradient finite as mu -> 0
    mu_tiny = torch.tensor([1e-4], requires_grad=True)
    loss_tiny = ztnb_nll(torch.ones(1), mu_tiny, log_phi)
    loss_tiny.backward()
    assert torch.isfinite(mu_tiny.grad), "Test 3 FAILED: grad not finite"
    print(f"  Test 3 PASS: grad at mu=1e-4 -> {mu_tiny.grad.item():.4f}")

    print("All updated ZTNB unit tests passed.\n")


if __name__ == "__main__":
    _run_unit_tests()
