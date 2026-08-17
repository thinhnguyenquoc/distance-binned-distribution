"""
Full Physics-Informed Zero-Shot OD Model (M_0).

Coordinates:
    1. UrbanGNN: X, G^urban -> h_i
    2. GravityPrior: P_i, P_j, D_ij -> log T_ij^grav
    3. PairwiseODDecoder: [h_i, h_j, log D_ij, log T_ij^grav] -> mu_ij
    4. Learnable global dispersion parameter phi for ZTNB likelihood.

Zero-Shot inference:
    \hat{T}^{ZS}_ij = mu_ij  for all candidate pairs in Omega_c.
"""

import torch
import torch.nn as nn
from src.models.node_encoder import UrbanGNN
from src.models.gravity import GravityPrior
from src.models.decoder import PairwiseODDecoder


class ZeroShotODModel(nn.Module):
    def __init__(
        self,
        node_in_dim: int = 26,
        node_hidden_dim: int = 64,
        node_out_dim: int = 64,
        num_gnn_layers: int = 2,
        decoder_hidden_dim: int = 64,
        dropout: float = 0.1,
        init_log_phi: float = 0.0,
    ):
        super().__init__()
        # 1. Urban GNN
        self.node_encoder = UrbanGNN(
            in_dim=node_in_dim,
            hidden_dim=node_hidden_dim,
            out_dim=node_out_dim,
            num_layers=num_gnn_layers,
            dropout=dropout,
        )

        # 2. Gravity Prior
        self.gravity_prior = GravityPrior()

        # 3. Pairwise Decoder
        self.decoder = PairwiseODDecoder(
            node_dim=node_out_dim,
            hidden_dim=decoder_hidden_dim,
            dropout=dropout,
        )

        # 4. Global trainable dispersion parameter phi (phi = exp(log_phi))
        self.log_phi = nn.Parameter(torch.tensor(init_log_phi, dtype=torch.float32))

    @property
    def phi(self) -> torch.Tensor:
        return torch.exp(self.log_phi)

    def forward(
        self,
        x: torch.Tensor,
        spatial_edge_index: torch.Tensor,
        spatial_edge_dist: torch.Tensor,
        pair_o_idx: torch.Tensor,
        pair_d_idx: torch.Tensor,
        pair_distance_log1p: torch.Tensor,
        population: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass predicting flow magnitude mu_ij for requested candidate pairs.

        Args:
            x:                   (N, F) normalized node features.
            spatial_edge_index:  (2, E_graph) G^urban edges.
            spatial_edge_dist:   (E_graph,) G^urban edge distances (km).
            pair_o_idx:          (E_pairs,) origin tract indices.
            pair_d_idx:          (E_pairs,) destination tract indices.
            pair_distance_log1p: (E_pairs,) log1p(distance_km).
            population:          (N,) raw tract population.

        Returns:
            mu: (E_pairs,) predicted expected flows \hat{T}^{ZS}_ij.
        """
        # Step 1: Compute node embeddings from observable urban graph
        h = self.node_encoder(x, spatial_edge_index, spatial_edge_dist)  # (N, d)

        # Gather origin and destination embeddings for candidate pairs
        h_o = h[pair_o_idx]  # (E_pairs, d)
        h_d = h[pair_d_idx]  # (E_pairs, d)

        # Step 2: Compute Physics Gravity prior
        # Recover un-logged distance in km for gravity formulation
        dist_km = torch.expm1(pair_distance_log1p)
        pop_o = population[pair_o_idx]
        pop_d = population[pair_d_idx]
        log_t_grav = self.gravity_prior(pop_o, pop_d, dist_km)  # (E_pairs,)

        # Step 3: Decode pairwise flows
        mu = self.decoder(h_o, h_d, pair_distance_log1p, log_t_grav)  # (E_pairs,)
        return mu


if __name__ == "__main__":
    from src.data.dataset import load_city
    from src.data.urban_graph import build_knn_graph

    cd = load_city("Raleigh", "data")
    ei, ed = build_knn_graph(cd.lon_lat.numpy(), k=10)

    model = ZeroShotODModel()
    mu = model(
        cd.node_features,
        ei,
        ed,
        cd.pair_o_idx,
        cd.pair_d_idx,
        cd.pair_distance,
        cd.population,
    )
    print(f"ZeroShotODModel test forward pass:")
    print(f"  mu shape: {mu.shape}, min: {mu.min().item():.3f}, max: {mu.max().item():.3f}")
    print(f"  phi: {model.phi.item():.4f}")
