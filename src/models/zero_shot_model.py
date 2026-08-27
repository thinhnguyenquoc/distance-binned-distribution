r"""
Gravity-Informed Urban-GNN Support-Conditioned Zero-Shot Model (M_0).
(neuroGravity-inspired neural transferable architecture)

Mathematical Formulation:
    1. Classical Gravity Prior:
        T_ij^grav = exp(G_0) * P_i * P_j * D_ij^(-alpha_0)
    2. Urban GNN Representation:
        h_i = GNN_theta(X_i, G^urban)
    3. Neural Transfer Decoder:
        \hat{T}_ij^ZS = f_theta*(X_i, X_j, D_ij, T_ij^grav)
        where f_theta maps [h_i, h_j, log(1+D_ij), log(T_ij^grav)] to conditional mean E[T_ij | T_ij >= 1].
    4. Learnable global dispersion parameter phi for ZTNB likelihood.
"""

import torch
import torch.nn as nn
from src.models.node_encoder import UrbanGNN
from src.models.gravity import GravityPrior
from src.models.decoder import PairwiseODDecoder
from src.loss.ztnb import compute_conditional_mean
from src.models.node_encoder import NodeMLP

class ZeroShotMLPModel(nn.Module):
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
        # 1. Urban MLP (no message passing)
        self.node_encoder = NodeMLP(
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

        # 4. Global trainable dispersion parameter phi
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
        return_conditional_mean: bool = False,
    ) -> torch.Tensor:
        h = self.node_encoder(x, spatial_edge_index, spatial_edge_dist)
        h_o = h[pair_o_idx]
        h_d = h[pair_d_idx]

        dist_km = torch.expm1(pair_distance_log1p)
        pop_o = population[pair_o_idx]
        pop_d = population[pair_d_idx]
        log_t_grav = self.gravity_prior(pop_o, pop_d, dist_km)

        mu_nb = self.decoder(h_o, h_d, pair_distance_log1p, log_t_grav)

        if return_conditional_mean:
            return compute_conditional_mean(mu_nb, self.log_phi)
        return mu_nb

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

        # 3. Pairwise Decoder (outputs base mean mu_nb > 0)
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
        return_conditional_mean: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass predicting flows for candidate pairs on Omega_c.

        Args:
            return_conditional_mean:
                If False (training): returns base parameter mu_nb for ZTNB likelihood.
                If True (inference): returns exact conditional expectation E[T | T >= 1].
        """
        # Step 1: Compute node embeddings from observable urban graph G^urban
        h = self.node_encoder(x, spatial_edge_index, spatial_edge_dist)  # (N, d)

        # Gather origin and destination embeddings for candidate pairs
        h_o = h[pair_o_idx]  # (E_pairs, d)
        h_d = h[pair_d_idx]  # (E_pairs, d)

        # Step 2: Compute Physics Gravity prior
        dist_km = torch.expm1(pair_distance_log1p)
        pop_o = population[pair_o_idx]
        pop_d = population[pair_d_idx]
        log_t_grav = self.gravity_prior(pop_o, pop_d, dist_km)  # (E_pairs,)

        # Step 3: Decode pairwise flows (mu_nb > 0)
        mu_nb = self.decoder(h_o, h_d, pair_distance_log1p, log_t_grav)  # (E_pairs,)

        if return_conditional_mean:
            # \hat{T} = E[T | T >= 1]
            return compute_conditional_mean(mu_nb, self.log_phi)
        return mu_nb


if __name__ == "__main__":
    from src.data.dataset import load_city
    from src.data.urban_graph import build_knn_graph

    cd = load_city("Raleigh", "data")
    ei, ed = build_knn_graph(cd.lon_lat.numpy(), k=10)

    model = ZeroShotODModel()
    mu_nb = model(cd.node_features, ei, ed, cd.pair_o_idx, cd.pair_d_idx, cd.pair_distance, cd.population, return_conditional_mean=False)
    t_hat = model(cd.node_features, ei, ed, cd.pair_o_idx, cd.pair_d_idx, cd.pair_distance, cd.population, return_conditional_mean=True)
    print("Forward pass base mu_nb shape:", mu_nb.shape, "min:", mu_nb.min().item())
    print("Forward pass t_hat shape:", t_hat.shape, "min:", t_hat.min().item())
    assert (t_hat >= mu_nb).all(), "Conditioning must increase or maintain expectation"
    print("Model check passed.")
