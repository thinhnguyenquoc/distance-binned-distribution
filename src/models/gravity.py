"""
Classical 2-parameter Physics Gravity Model Prior.

log T_ij^grav = G + log P_i + log P_j - alpha * log(D_ij)

Parameters:
    G: global scale parameter (learnable scalar)
    alpha: distance decay parameter (learnable scalar, initialized to ~1.0-2.0)

Both G and alpha are global trainable parameters shared across all cities in a fold,
providing the physics prior baseline for cross-city transfer.
"""

import math
import torch
import torch.nn as nn


class GravityPrior(nn.Module):
    def __init__(self, init_G: float = 0.0, init_alpha: float = 1.0):
        super().__init__()
        # Trainable physics parameters
        self.G = nn.Parameter(torch.tensor(init_G, dtype=torch.float32))
        self.log_alpha = nn.Parameter(torch.tensor(math.log(init_alpha), dtype=torch.float32))

    @property
    def alpha(self) -> torch.Tensor:
        return torch.exp(self.log_alpha)  # ensure alpha > 0

    def forward(
        self,
        population_i: torch.Tensor,
        population_j: torch.Tensor,
        distance_km: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes log T_ij^grav for each pair.

        Args:
            population_i: (E,) population of origin tract.
            population_j: (E,) population of destination tract.
            distance_km:  (E,) distance in km (not log).

        Returns:
            log_T_grav: (E,) log-expected gravity flow.
        """
        log_pi = torch.log(torch.clamp(population_i, min=1.0))
        log_pj = torch.log(torch.clamp(population_j, min=1.0))
        # Clamp at 0.1 km to avoid log(0) for intrazonal pairs (D_ii = 0).
        # This floor is an explicit design choice: intrazonal log_d = log(0.1) ≈ -2.3.
        # The model is trained on this behaviour; do not change without a full retrain.
        log_d  = torch.log(torch.clamp(distance_km, min=0.1))

        log_t_grav = self.G + log_pi + log_pj - self.alpha * log_d
        return log_t_grav


if __name__ == "__main__":
    grav = GravityPrior()
    p_i = torch.tensor([1000.0, 5000.0])
    p_j = torch.tensor([2000.0, 10000.0])
    d   = torch.tensor([5.0, 15.0])
    out = grav(p_i, p_j, d)
    print("Gravity prior output log_T:", out)
    print(f"Alpha: {grav.alpha.item():.4f}, G: {grav.G.item():.4f}")
