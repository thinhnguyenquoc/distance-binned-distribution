# Deep Gravity Allocation Model with CSR batching
from __future__ import annotations
import torch
import torch.nn as nn

def segment_logsumexp(scores: torch.Tensor, origin_idx: torch.Tensor, num_origins: int) -> torch.Tensor:
    """Numerically stable segment log-sum-exp over origins using PyTorch native scatter ops."""
    max_scores = torch.full((num_origins,), -float("inf"), dtype=scores.dtype, device=scores.device)
    max_scores.scatter_reduce_(0, origin_idx, scores, reduce="amax", include_self=False)
    shifted = scores - max_scores[origin_idx]
    sum_exp = torch.zeros((num_origins,), dtype=scores.dtype, device=scores.device)
    sum_exp.scatter_add_(0, origin_idx, torch.exp(shifted))
    return max_scores + torch.log(sum_exp.clamp_min(1e-12))


class DeepGravityAllocationModel(nn.Module):
    """
    Deep Gravity Destination Allocation Network (Simini et al., Nature Communications 2021).
    Adapted to the known-positive support Omega_c benchmark.

    Architecture:
        15 hidden layers feed-forward MLP:
        - 6 hidden layers of 256 units with LeakyReLU
        - 9 hidden layers of 128 units with LeakyReLU
        - 1 final linear layer to scalar score s_ij
        Total parameters: exactly 507,905.
    """

    def __init__(self, in_dim: int = 53):
        super().__init__()
        layers = []
        # First 6 hidden layers of 256 units
        layers.append(nn.Linear(in_dim, 256))
        layers.append(nn.LeakyReLU())
        for _ in range(5):
            layers.append(nn.Linear(256, 256))
            layers.append(nn.LeakyReLU())
        # Next 9 hidden layers of 128 units
        layers.append(nn.Linear(256, 128))
        layers.append(nn.LeakyReLU())
        for _ in range(8):
            layers.append(nn.Linear(128, 128))
            layers.append(nn.LeakyReLU())
        # Final linear layer to scalar score s_ij
        layers.append(nn.Linear(128, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, pair_feat: torch.Tensor) -> torch.Tensor:
        """Compute scalar logit s_ij for each pair."""
        return self.net(pair_feat).squeeze(-1)

    def compute_log_probs(self, pair_feat: torch.Tensor, origin_idx: torch.Tensor, num_origins: int) -> torch.Tensor:
        """Compute log p_ij = s_ij - logsumexp_{k in Omega_{c,i}}(s_ik) for candidate pairs."""
        scores = self.forward(pair_feat)
        lse = segment_logsumexp(scores, origin_idx, num_origins)
        return scores - lse[origin_idx]

    def compute_probs(self, pair_feat: torch.Tensor, origin_idx: torch.Tensor, num_origins: int) -> torch.Tensor:
        """Compute destination allocation probabilities p_ij."""
        log_p = self.compute_log_probs(pair_feat, origin_idx, num_origins)
        return torch.exp(log_p)

    def compute_loss_batch(
        self,
        batch_pair_feat: torch.Tensor,
        batch_origin_compact_idx: torch.Tensor,
        batch_trips: torch.Tensor,
        num_batch_origins: int,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss ONLY on pairs belonging to the current batch of origins.
        
        Args:
            batch_pair_feat: (E_batch, 53) feature matrix for pairs of the selected origins.
            batch_origin_compact_idx: (E_batch,) origin index remapped to [0, num_batch_origins - 1].
            batch_trips: (E_batch,) ground-truth flows on Omega_c for these pairs.
            num_batch_origins: Number of origins in this mini-batch (e.g. <= 64).
        """
        log_p = self.compute_log_probs(batch_pair_feat, batch_origin_compact_idx, num_batch_origins)
        
        # Origin total flow: O_i = sum_{j in Omega_{c,i}} t_ij
        O_i = torch.zeros(num_batch_origins, dtype=batch_trips.dtype, device=batch_trips.device)
        O_i.scatter_add_(0, batch_origin_compact_idx, batch_trips)
        
        # Empirical destination proportion: q_ij = t_ij / O_i
        q_ij = batch_trips / O_i[batch_origin_compact_idx].clamp_min(1e-12)
        
        # Pair cross-entropy component: - q_ij * log p_ij
        pair_ce = - q_ij * log_p
        
        # Origin loss: L_i = sum_{j in Omega_{c,i}} - q_ij * log p_ij
        origin_loss = torch.zeros(num_batch_origins, dtype=log_p.dtype, device=log_p.device)
        origin_loss.scatter_add_(0, batch_origin_compact_idx, pair_ce)
        
        active = O_i > 0
        if not active.any():
            return torch.tensor(0.0, device=batch_pair_feat.device, requires_grad=True)
        return origin_loss[active].mean()
