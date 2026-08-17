"""
Cross-City Training and Transfer Pipeline.

Stage A: Cross-city Training
    Trains ZeroShotODModel on a list of source cities using ZTNB likelihood:
        L_train = - 1 / |Omega^+| * sum_{(i,j) in Omega^+} log P_ZTNB(T_ij; mu_ij, phi)
    After convergence, freezes parameters -> theta*.

Stage B: Zero-Shot Transfer Evaluation
    Evaluates theta* on held-out target city: (X^{c*}, D^{c*}) -> \hat{T}^{ZS}.
"""

import time
import torch
import torch.optim as optim
from typing import List, Dict

from src.data.dataset import CityData, load_cities, load_city
from src.data.urban_graph import build_knn_graph
from src.models.zero_shot_model import ZeroShotODModel
from src.loss.ztnb import ztnb_nll, nb_nll
from src.training.evaluate import evaluate_all


def train_epoch(
    model: ZeroShotODModel,
    train_cities: List[CityData],
    city_graphs: List[tuple[torch.Tensor, torch.Tensor]],
    optimizer: optim.Optimizer,
    loss_type: str = "ztnb",
    device: torch.device = torch.device("cpu"),
) -> float:
    """Trains one epoch iterating across all source cities."""
    model.train()
    total_loss = 0.0
    num_cities = len(train_cities)

    for city_data, (edge_index, edge_dist) in zip(train_cities, city_graphs):
        optimizer.zero_grad()

        # Move tensors to device
        x = city_data.node_features.to(device)
        ei = edge_index.to(device)
        ed = edge_dist.to(device)
        p_o = city_data.pair_o_idx.to(device)
        p_d = city_data.pair_d_idx.to(device)
        p_dist = city_data.pair_distance.to(device)
        pop = city_data.population.to(device)
        t_true = city_data.pair_trips.to(device)

        # Forward pass
        mu = model(x, ei, ed, p_o, p_d, p_dist, pop)

        # Loss computation
        if loss_type == "ztnb":
            loss = ztnb_nll(t_true, mu, model.log_phi)
        elif loss_type == "nb":
            loss = nb_nll(t_true, mu, model.log_phi)
        else:
            raise ValueError(f"Unknown loss type {loss_type}")

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN/Inf loss encountered for {city_data.city_name}, skipping batch.")
            continue

        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(num_cities, 1)


@torch.no_grad()
def infer_zero_shot(
    model: ZeroShotODModel,
    city_data: CityData,
    edge_index: torch.Tensor,
    edge_dist: torch.Tensor,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Runs pure zero-shot forward inference on a held-out city."""
    model.eval()
    x = city_data.node_features.to(device)
    ei = edge_index.to(device)
    ed = edge_dist.to(device)
    p_o = city_data.pair_o_idx.to(device)
    p_d = city_data.pair_d_idx.to(device)
    p_dist = city_data.pair_distance.to(device)
    pop = city_data.population.to(device)

    mu = model(x, ei, ed, p_o, p_d, p_dist, pop)
    return mu.cpu()


def train_zero_shot_model(
    train_city_names: List[str],
    data_root: str = "data",
    epochs: int = 30,
    lr: float = 1e-3,
    hidden_dim: int = 64,
    num_gnn_layers: int = 2,
    knn_k: int = 10,
    loss_type: str = "ztnb",
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = True,
) -> tuple[ZeroShotODModel, object]:
    """
    Full cross-city training pipeline on source cities.

    Returns:
        (trained frozen ZeroShotODModel, fitted feature scaler)
    """
    device = torch.device(device_str)
    if verbose:
        print(f"Loading {len(train_city_names)} source cities onto {device}...")

    train_cities, scaler = load_cities(train_city_names, data_root=data_root)

    # Precompute spatial graphs G^urban
    city_graphs = []
    for c in train_cities:
        ei, ed = build_knn_graph(c.lon_lat.numpy(), k=knn_k)
        city_graphs.append((ei, ed))

    model = ZeroShotODModel(
        node_in_dim=train_cities[0].node_features.shape[1],
        node_hidden_dim=hidden_dim,
        node_out_dim=hidden_dim,
        num_gnn_layers=num_gnn_layers,
        decoder_hidden_dim=hidden_dim,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        loss_val = train_epoch(
            model=model,
            train_cities=train_cities,
            city_graphs=city_graphs,
            optimizer=optimizer,
            loss_type=loss_type,
            device=device,
        )
        scheduler.step()

        if verbose and (epoch % 5 == 0 or epoch == 1 or epoch == epochs):
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:02d}/{epochs:02d} | Loss ({loss_type}): {loss_val:.4f} | phi: {model.phi.item():.3f} | alpha: {model.gravity_prior.alpha.item():.3f} | {elapsed:.1f}s")

    # Freeze all parameters
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return model, scaler
