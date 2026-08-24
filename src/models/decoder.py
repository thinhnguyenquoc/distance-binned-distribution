"""
Pairwise OD Decoder with Single Base Magnitude Head (ZTNB).

Input edge representation:
    e_ij = [h_i, h_j, log(1 + D_ij), log(T^{grav}_ij)]

Single prediction head producing base Negative Binomial parameter via
residual-gravity initialization: mu_nb_ij = softplus(log_t_grav + residual_ij).

Exact ZTNB Likelihood & Predictions:
    At training: loss = -log P_ZTNB(T_ij; mu_nb_ij, phi) on positive observations in Omega_c.
    At inference: expected zero-shot prediction is the conditional expectation:
        \hat{T}^{ZS}_ij = E[T_ij | T_ij >= 1] = compute_conditional_mean(mu_nb_ij, log_phi).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PairwiseODDecoder(nn.Module):
    def __init__(
        self,
        node_dim: int = 64,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Input: [h_i, h_j, log_d, log_t_grav] -> dim = 2 * node_dim + 2
        in_dim = 2 * node_dim + 2

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Zero-init final layer so the gravity prior is supplied as a log-scale decoder feature/offset 
        # with a zero-initialized neural residual, yielding softplus(log_t_grav) at initialization.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        h_i: torch.Tensor,
        h_j: torch.Tensor,
        log_distance: torch.Tensor,
        log_t_grav: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            h_i:          (E, node_dim) origin node embeddings.
            h_j:          (E, node_dim) destination node embeddings.
            log_distance: (E,) or (E, 1) log1p(distance_km).
            log_t_grav:   (E,) or (E, 1) log gravity flow.

        Returns:
            mu_nb: (E,) positive base mean parameter mu_nb_ij > 0.
        """
        if log_distance.dim() == 1:
            log_distance = log_distance.unsqueeze(-1)
        if log_t_grav.dim() == 1:
            log_t_grav = log_t_grav.unsqueeze(-1)

        # Concatenate edge representation e_ij
        e_ij = torch.cat([h_i, h_j, log_distance, log_t_grav], dim=-1)

        residual = self.net(e_ij)  # (E, 1), ~0 at init
        # Residual-gravity: gravity prior serves as log-scale feature, GNN learns the deviation.
        # Yields softplus(log_t_grav) at initialization.
        log_mu_nb = log_t_grav + residual
        mu_nb = F.softplus(log_mu_nb.squeeze(-1)) + 1e-4
        return mu_nb


if __name__ == "__main__":
    dec = PairwiseODDecoder(node_dim=32, hidden_dim=64)
    h_i = torch.randn(100, 32)
    h_j = torch.randn(100, 32)
    ld = torch.randn(100)
    ltg = torch.randn(100)
    mu_nb = dec(h_i, h_j, ld, ltg)
    print("Decoder mu_nb output shape:", mu_nb.shape, "min:", mu_nb.min().item(), "max:", mu_nb.max().item())

    # At init, residual ~ 0, so mu_nb should track softplus(log_t_grav) closely
    expected = F.softplus(ltg) + 1e-4
    print("Max deviation from pure gravity at init:", (mu_nb - expected).abs().max().item())
