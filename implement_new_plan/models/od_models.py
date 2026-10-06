"""
Three Distinct OD Flow Baseline Families:
1. Two-Parameter Gravity Model (TwoParameterGravity)
2. Pairwise MLP (PairwiseMLP)
3. Urban-GNN (UrbanGNN)

Implementation Invariant:
- These represent THREE DIFFERENT MODELING FAMILIES, not controlled ablations of each other.
- Shared protocol:
    same source city
    same 30% positive OD training pairs
    same held-out/source support
    same target positive support
    same model seeds (1, 10, 100) where applicable
    same evaluation metrics
    same zero-shot transfer protocol
    same DBD calibration pipeline
"""

import math
from typing import Optional, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from implement_new_plan.models.node_encoder import UrbanGNN as GNNNodeEncoder
from implement_new_plan.models.gravity import GravityPrior
from implement_new_plan.models.decoder import PairwiseODDecoder


# ===========================================================================
# MODEL 1 — TWO-PARAMETER GRAVITY MODEL
# ===========================================================================
class TwoParameterGravity(nn.Module):
    r"""
    Two-Parameter Physics-Based Gravity Model:
        \hat{T}_{ij} = \exp(G) * P_i * P_j * D_{ij}^{-\alpha}
    
    Exactly two trainable parameters: \theta_gravity = {G, \alpha}.
    - No neural network, no MLP, no graph, no node embeddings, no message passing.
    - G: Unconstrained global scale parameter (G \in R).
    - \alpha: Distance decay parameter \alpha = \exp(\log \alpha) > 0 (strictly matching baseline GravityPrior).
    - Numerically stable log-space computation:
        \log \hat{T}_{ij} = G + \log P_i + \log P_j - \alpha \log D_{ij}
      for pairs with P_i > 0 and P_j > 0. If P_i == 0 or P_j == 0, \hat{T}_{ij} = 0.
    """
    def __init__(self, init_G: float = 0.0, init_alpha: float = 1.0):
        super().__init__()
        self.G = nn.Parameter(torch.tensor(init_G, dtype=torch.float32))
        # Parameterize alpha = exp(log_alpha) > 0 strictly matching baseline GravityPrior
        if init_alpha <= 0.0:
            raise ValueError(f"init_alpha must be strictly positive, got {init_alpha}")
        self.log_alpha = nn.Parameter(torch.tensor(math.log(init_alpha), dtype=torch.float32))

    @property
    def alpha(self) -> torch.Tensor:
        return torch.exp(self.log_alpha)

    def forward(
        self,
        population_raw_o: torch.Tensor,
        population_raw_d: torch.Tensor,
        distance_km_raw: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Computes gravity flow directly on positive support:
            \hat{T}_{ij} = \exp(G) * P_i * P_j * D_{ij}^{-\alpha}
            
        Strict Contract:
        - population_raw_o: Raw origin tract population P_i (non-negative, finite).
          NO log1p, NO z-score, NO min-max, NO target-normalization.
        - population_raw_d: Raw destination tract population P_j (non-negative, finite).
          NO log1p, NO z-score, NO min-max, NO target-normalization.
        - distance_km_raw: Raw physical Haversine distance in km (> 0).
          NO log1p, NO standardization.
        """
        # 1. Population validity checks
        if not torch.isfinite(population_raw_o).all() or not torch.isfinite(population_raw_d).all():
            raise ValueError("TwoParameterGravity requires finite raw population values (no NaN/Inf).")
        if not (population_raw_o >= 0).all() or not (population_raw_d >= 0).all():
            raise ValueError("TwoParameterGravity requires non-negative raw population values. Found negative population.")
        
        # 2. Distance validity check
        if not torch.isfinite(distance_km_raw).all() or not (distance_km_raw > 0).all():
            raise ValueError("TwoParameterGravity requires strictly positive physical distance values (D_ij > 0).")

        # 3. Handle zero-population pairs explicitly without log(0)
        pos_pop_mask = (population_raw_o > 0) & (population_raw_d > 0)
        
        # Initialize output flow tensor with zeros
        t_hat = torch.zeros_like(distance_km_raw, dtype=torch.float32)

        if torch.any(pos_pop_mask):
            p_o_pos = population_raw_o[pos_pop_mask]
            p_d_pos = population_raw_d[pos_pop_mask]
            d_pos = distance_km_raw[pos_pop_mask]

            alpha_val = self.alpha
            log_flow = self.G + torch.log(p_o_pos) + torch.log(p_d_pos) - alpha_val * torch.log(d_pos)

            # Check for potential floating-point overflow before exp
            if torch.any(log_flow > 88.0):
                max_log = float(torch.max(log_flow).item())
                min_log = float(torch.min(log_flow).item())
                raise OverflowError(
                    f"TwoParameterGravity prediction log_flow exceeded floating point limits (max={max_log:.2f}, min={min_log:.2f}, G={self.G.item():.4f}, alpha={alpha_val.item():.4f})."
                )

            t_hat[pos_pop_mask] = torch.exp(log_flow)

        # 4. Final numerical sanity assertions (no arbitrary clipping)
        if not torch.isfinite(t_hat).all():
            raise ValueError("TwoParameterGravity produced non-finite flow predictions (NaN/Inf).")
        if not (t_hat >= 0).all():
            raise ValueError("TwoParameterGravity produced negative flow predictions.")

        return t_hat


# ===========================================================================
# MODEL 2 — PAIRWISE MLP
# ===========================================================================
class PairwiseMLP(nn.Module):
    r"""
    DeepGravity-Inspired Pairwise MLP (Direct Flow-Intensity Regression).
    
    Adapted to direct OD-flow regression without target origin outflow constraints:
        \hat{T}_{ij} = Softplus(f_MLP([x_i || x_j || d'_ij]))
    
    Critical Separation Invariant:
    - NO G, NO alpha, NO gravity equation, NO gravity residual/prior/initialization.
    - NO graph, NO message passing, NO adjacency matrix.
    - Input: [x_i || x_j || d'_ij] of dimension 2 * node_in_dim + 1.
    """
    def __init__(
        self,
        node_in_dim: int = 26,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Input: origin features (F), destination features (F), source-scaled distance (1)
        in_dim = 2 * node_in_dim + 1

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        x_o: torch.Tensor,
        x_d: torch.Tensor,
        distance_std: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Forward pass predicting flow directly:
            x_ij = [x_o || x_d || distance_std]
            \hat{T}_{ij} = Softplus(net(x_ij)) >= 0
            
        Strict Input Contract:
        - x_o: Origin node features (shape: (..., F)) using canonical NODE_FEATURE_COLUMNS.
        - x_d: Destination node features (shape: (..., F)) using the exact same canonical schema.
        - distance_std: Source-standardized log-distance: (log1p(d_raw) - mu_s) / sigma_s.
        - Dimension check: [x_o || x_d || distance_std] must have exactly 2 * F + 1 dimensions.
        """
        if distance_std.dim() == 1:
            distance_std = distance_std.unsqueeze(-1)

        F_dim = x_o.shape[-1]
        if x_d.shape[-1] != F_dim:
            raise ValueError(
                f"Origin and destination feature dimension mismatch: x_o has {F_dim} dims, "
                f"x_d has {x_d.shape[-1]} dims. Must use identical canonical schema."
            )

        pair_feat = torch.cat([x_o, x_d, distance_std], dim=-1)
        expected_pair_dim = 2 * F_dim + 1
        if pair_feat.shape[-1] != expected_pair_dim:
            raise AssertionError(
                f"Pairwise MLP input dimension mismatch: expected {expected_pair_dim} (2 * {F_dim} + 1), "
                f"got {pair_feat.shape[-1]}"
            )

        z_ij = self.net(pair_feat).squeeze(-1)
        t_hat = F.softplus(z_ij)
        return t_hat


# ===========================================================================
# MODEL 3 — URBAN-GNN
# ===========================================================================
class UrbanGNN(nn.Module):
    r"""
    Spatial Graph Neural Network Model with Message Passing and Trainable Gravity.
    
    Architecture (Inherited baseline implementation):
    1. Spatial radius graph (r=5.0 km) with distance-modulated message passing:
           m_ij = W_msg * [h_j || log(1 + d_ij)]
    2. Node representations h_i, h_j from 2-layer GraphConvLayer.
    3. Jointly trained classical 2-parameter gravity component:
           log T_ij^grav = G + log P_i + log P_j - alpha * log(D_ij)
    4. Neural transfer decoder with residual-gravity combination:
           mu_nb_ij = Softplus(log T_ij^grav + residual_ij)
    """
    def __init__(
        self,
        node_in_dim: int = 26,
        node_hidden_dim: int = 64,
        node_out_dim: int = 64,
        num_gnn_layers: int = 2,
        decoder_hidden_dim: int = 64,
        dropout: float = 0.1,
        init_G: float = 0.0,
        init_alpha: float = 1.0,
    ):
        super().__init__()
        # 1. Spatial GNN Node Encoder (Message Passing)
        self.node_encoder = GNNNodeEncoder(
            in_dim=node_in_dim,
            hidden_dim=node_hidden_dim,
            out_dim=node_out_dim,
            num_layers=num_gnn_layers,
            dropout=dropout,
        )

        # 2. Jointly trainable Gravity component (G, alpha)
        self.gravity = GravityPrior(init_G=init_G, init_alpha=init_alpha)

        # 3. Residual-Gravity Pairwise Decoder
        self.decoder = PairwiseODDecoder(
            node_dim=node_out_dim,
            hidden_dim=decoder_hidden_dim,
            dropout=dropout,
        )

    @property
    def G(self) -> torch.Tensor:
        return self.gravity.G

    @property
    def alpha(self) -> torch.Tensor:
        return self.gravity.alpha

    def forward(
        self,
        x: torch.Tensor,
        spatial_edge_index: torch.Tensor,
        spatial_edge_dist_raw: torch.Tensor,
        pair_o_idx: torch.Tensor,
        pair_d_idx: torch.Tensor,
        pair_distance_km_raw: torch.Tensor,
        population_raw: Optional[torch.Tensor] = None,
        pair_distance_log: Optional[torch.Tensor] = None,
        population: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""
        Forward pass with spatial message passing and residual gravity decoding.
        
        Strict Contract:
        - x: Processed node features (includes source-standardized log1p population).
        - spatial_edge_dist_raw: Raw Haversine physical distance in km (<= 5.0 km radius).
        - pair_distance_km_raw: Raw physical distance D_ij in km (> 0) for explicit gravity prior.
        - population_raw: Raw tract population P_i (non-negative, finite) for explicit gravity prior.
        - pair_distance_log: log(1 + d_raw) used as edge feature in decoder. If None, computed on the fly.
        """
        # Support both population_raw and legacy population keyword argument
        pop_input = population_raw if population_raw is not None else population
        if pop_input is None:
            raise ValueError("UrbanGNN requires population_raw tensor for its gravity prior component.")

        # Population validity check for raw gravity prior
        if not torch.isfinite(pop_input).all():
            raise ValueError("UrbanGNN gravity prior requires finite raw population values.")
        if not (pop_input >= 0).all():
            raise ValueError("UrbanGNN gravity prior requires non-negative raw population values.")

        # 1. Node message passing: internally uses log1p(spatial_edge_dist_raw)
        h = self.node_encoder(x, spatial_edge_index, spatial_edge_dist_raw)
        h_o = h[pair_o_idx]
        h_d = h[pair_d_idx]

        # 2. Gravity prior component: MUST use raw physical distance in km and raw population
        pop_o = pop_input[pair_o_idx]
        pop_d = pop_input[pair_d_idx]
        log_t_grav = self.gravity(pop_o, pop_d, pair_distance_km_raw)

        # 3. Decoder combination: uses log1p(d_raw)
        if pair_distance_log is None:
            pair_distance_log = torch.log1p(pair_distance_km_raw)
            
        t_hat = self.decoder(h_o, h_d, pair_distance_log, log_t_grav)
        return t_hat


# ===========================================================================
# SANITY CHECKS & DISTANCE CONTRACT VERIFICATION
# ===========================================================================
def verify_distance_contract():
    """Validates the strict distance contract across all three model families."""
    print("Running Distance Contract Verification...")
    
    # 1. Two-Parameter Gravity: requires raw physical km and raw population
    grav = TwoParameterGravity()
    population_raw_o = torch.tensor([10000.0, 50000.0])
    population_raw_d = torch.tensor([20000.0, 30000.0])
    distance_km_raw = torch.tensor([5.2, 14.8])
    
    # Assert raw distance and population are valid
    assert torch.all(distance_km_raw > 0)
    assert torch.all(population_raw_o >= 0)
    assert torch.all(population_raw_d >= 0)
    out_grav = grav(population_raw_o, population_raw_d, distance_km_raw)
    assert out_grav.shape == (2,)
    assert torch.all(out_grav > 0)

    # Test zero population handling (P=0 -> flow=0 without log(0) errors)
    out_zero_pop = grav(torch.tensor([0.0, 50000.0]), population_raw_d, distance_km_raw)
    assert out_zero_pop[0].item() == 0.0
    assert out_zero_pop[1].item() > 0.0

    # Test alpha non-negativity property
    assert grav.alpha.item() >= 0.0

    # Test error handling on negative/non-finite population
    try:
        grav(torch.tensor([-1.0, 10.0]), population_raw_d, distance_km_raw)
        assert False, "Should fail on negative population"
    except ValueError:
        pass

    # Test error handling on non-positive distance
    try:
        grav(population_raw_o, population_raw_d, torch.tensor([0.0, 10.0]))
        assert False, "Should fail on non-positive distance"
    except ValueError:
        pass
    
    # 2. Pairwise MLP: requires source-standardized log-distance
    mlp = PairwiseMLP(node_in_dim=26)
    x_o = torch.randn(2, 26)
    x_d = torch.randn(2, 26)
    
    # Simulate source stats
    source_mean = 2.5
    source_std = 0.8
    distance_log = torch.log1p(distance_km_raw)
    distance_std = (distance_log - source_mean) / source_std
    
    # Verify relations
    assert torch.allclose(distance_log, torch.log1p(distance_km_raw))
    assert torch.allclose(distance_std, (distance_log - source_mean) / source_std)
    
    out_mlp = mlp(x_o, x_d, distance_std)
    assert out_mlp.shape == (2,)
    assert torch.all(out_mlp >= 0)
    
    # 3. Urban-GNN: spatial radius edge raw <= 5.0, pair raw km, pair log
    gnn = UrbanGNN(node_in_dim=26)
    x_all = torch.randn(5, 26)
    spatial_edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    spatial_edge_dist_raw = torch.tensor([1.2, 3.4, 4.8])  # <= 5.0 km
    assert torch.all(spatial_edge_dist_raw <= 5.0)
    
    pop_all = torch.tensor([1000.0, 2000.0, 1500.0, 3000.0, 2500.0])
    pair_o = torch.tensor([0, 3])
    pair_d = torch.tensor([1, 4])
    
    out_gnn = gnn(
        x=x_all,
        spatial_edge_index=spatial_edge_index,
        spatial_edge_dist_raw=spatial_edge_dist_raw,
        pair_o_idx=pair_o,
        pair_d_idx=pair_d,
        pair_distance_km_raw=distance_km_raw,
        population_raw=pop_all,
    )
    assert out_gnn.shape == (2,)
    assert torch.all(out_gnn >= 0)
    print("Distance & Raw Population Contract Verification PASSED.")


if __name__ == "__main__":
    verify_distance_contract()
