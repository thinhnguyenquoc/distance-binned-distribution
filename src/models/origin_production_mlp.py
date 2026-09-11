# Origin Production MLP
from __future__ import annotations
import torch
import torch.nn as nn

class OriginProductionMLP(nn.Module):
    """
    Independent 3-layer MLP predicting tract-level log outflow z_i = log(1 + O_i^Omega).

    Architecture:
        Linear(in_dim, 64) -> ReLU()
        Linear(64, 32) -> ReLU()
        Linear(32, 1)

    Strict constraints:
        - Input: strictly the 26 normalized node features.
        - No test-city OD labels, no GNN embeddings, no target city knowledge.
    """

    def __init__(self, in_dim: int = 26, hidden_1: int = 64, hidden_2: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_1),
            nn.ReLU(),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def predict_outflow(self, x: torch.Tensor) -> torch.Tensor:
        z_hat = torch.clamp(self.forward(x), min=0.0, max=20.0)
        return torch.expm1(z_hat).clamp_min(1e-8)
