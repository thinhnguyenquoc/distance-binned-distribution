"""
Cross-City Training and Transfer Pipeline.

Stage A: Cross-city Training
    Trains ZeroShotODModel on a list of source cities using ZTNB likelihood:
        L_train = - 1 / |Omega^+| * sum_{(i,j) in Omega^+} log P_ZTNB(T_ij; mu_nb_ij, phi)
    After convergence, freezes parameters -> theta*.

Stage B: Zero-Shot Transfer Evaluation
    Evaluates theta* on held-out target city:
        (X^{c*}, G^{urban, c*}, D^{c*}) -> \hat{T}^{ZS} = E[T | T >= 1].
"""

import time
import torch
import torch.optim as optim
from typing import List, Dict

from src.data.dataset import CityData, load_cities, load_city
from src.data.urban_graph import build_radius_graph, build_knn_graph
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
    model.train()
    total_loss = 0.0
    num_cities = len(train_cities)

    for city_data, (edge_index, edge_dist) in zip(train_cities, city_graphs):
        optimizer.zero_grad()

        x = city_data.node_features.to(device)
        ei = edge_index.to(device)
        ed = edge_dist.to(device)
        p_o = city_data.pair_o_idx.to(device)
        p_d = city_data.pair_d_idx.to(device)
        p_dist = city_data.pair_distance.to(device)
        pop = city_data.population.to(device)
        t_true = city_data.pair_trips.to(device)

        # Training pass returns base mean mu_nb
        mu_nb = model(x, ei, ed, p_o, p_d, p_dist, pop, return_conditional_mean=False)

        if loss_type == "ztnb":
            loss = ztnb_nll(t_true, mu_nb, model.log_phi)
        elif loss_type == "nb":
            loss = nb_nll(t_true, mu_nb, model.log_phi)
        else:
            raise ValueError(f"Unknown loss type {loss_type}")

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN/Inf loss encountered for {city_data.city_name}, skipping.")
            continue

        loss.backward()
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
    """Runs zero-shot forward inference returning exact conditional expectation E[T | T >= 1]."""
    model.eval()
    x = city_data.node_features.to(device)
    ei = edge_index.to(device)
    ed = edge_dist.to(device)
    p_o = city_data.pair_o_idx.to(device)
    p_d = city_data.pair_d_idx.to(device)
    p_dist = city_data.pair_distance.to(device)
    pop = city_data.population.to(device)

    # Returns E[T | T >= 1]
    t_hat = model(x, ei, ed, p_o, p_d, p_dist, pop, return_conditional_mean=True)
    return t_hat.cpu()


def train_zero_shot_model(
    train_city_names: List[str],
    data_root: str = "data",
    epochs: int = 25,
    lr: float = 1e-3,
    hidden_dim: int = 64,
    num_gnn_layers: int = 2,
    graph_type: str = "radius",
    radius_km: float = 5.0,
    knn_k: int = 10,
    loss_type: str = "ztnb",
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = True,
    # --- Validation / early stopping ---
    val_city_names: List[str] | None = None,
    patience: int = 5,
    min_delta: float = 1e-4,
    return_info: bool = False,
) -> tuple:
    """
    Train ZeroShotODModel with optional validation-based early stopping.

    Args:
        train_city_names: Cities to train on.
        val_city_names:   Validation cities for early stopping. If None,
                          trains for exactly `epochs` epochs (pre-specified).
        patience:         Epochs without val CPC improvement before stopping.
        min_delta:        Minimum improvement to count as improvement.
        return_info:      If True, returns (model, scaler, train_info_dict).

    Returns:
        (best_model, scaler) or (best_model, scaler, info)
    """
    import copy
    import numpy as _np

    device = torch.device(device_str)
    if verbose:
        print(f"Loading {len(train_city_names)} source cities onto {device}...")

    train_cities, scaler = load_cities(train_city_names, data_root=data_root)

    # Precompute spatial graphs G^urban for training cities
    city_graphs = []
    for c in train_cities:
        coords = c.lon_lat.numpy()
        if graph_type == "radius":
            ei, ed = build_radius_graph(coords, radius_km=radius_km)
        else:
            ei, ed = build_knn_graph(coords, k=knn_k)
        city_graphs.append((ei, ed))

    # Precompute graphs for validation cities (if provided)
    val_cities_data = []
    val_city_graphs = []
    if val_city_names:
        for name in val_city_names:
            vc = load_city(name, data_root=data_root, feature_scaler=scaler)
            val_cities_data.append(vc)
            coords = vc.lon_lat.numpy()
            if graph_type == "radius":
                ei, ed = build_radius_graph(coords, radius_km=radius_km)
            else:
                ei, ed = build_knn_graph(coords, k=knn_k)
            val_city_graphs.append((ei, ed))

    model = ZeroShotODModel(
        node_in_dim=train_cities[0].node_features.shape[1],
        node_hidden_dim=hidden_dim,
        node_out_dim=hidden_dim,
        num_gnn_layers=num_gnn_layers,
        decoder_hidden_dim=hidden_dim,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_cpc = -float("inf")
    best_epoch = epochs
    best_state = None
    patience_counter = 0
    use_early_stopping = bool(val_city_names)

    val_history = []
    loss_history = []

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
        loss_history.append(loss_val)
        scheduler.step()

        # --- Validation CPC (interzonal) ---
        val_cpc_str = ""
        if use_early_stopping and val_cities_data:
            val_cpcs = []
            model.eval()
            with torch.no_grad():
                for vcd, (vei, ved) in zip(val_cities_data, val_city_graphs):
                    t_hat = infer_zero_shot(model, vcd, vei, ved, device=device)
                    t_hat_np = t_hat.numpy()
                    t_gt_np  = vcd.pair_trips.numpy()
                    dist_km  = _np.expm1(vcd.pair_distance.numpy())
                    o_np = vcd.pair_o_idx.numpy()
                    d_np = vcd.pair_d_idx.numpy()
                    inter = (o_np != d_np) & (dist_km > 0.0)
                    if inter.sum() > 0:
                        from src.training.evaluate import compute_cpc_pair
                        val_cpcs.append(compute_cpc_pair(t_gt_np[inter], t_hat_np[inter]))
            mean_val_cpc = float(_np.mean(val_cpcs)) if val_cpcs else 0.0
            val_history.append(mean_val_cpc)
            val_cpc_str = f" | ValCPC: {mean_val_cpc:.4f}"

            # Best-model tracking
            if mean_val_cpc > best_val_cpc + min_delta:
                best_val_cpc = mean_val_cpc
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

        if verbose and (epoch % 5 == 0 or epoch == 1 or epoch == epochs):
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:02d}/{epochs:02d} | Loss ({loss_type}): {loss_val:.4f} | "
                  f"phi: {model.phi.item():.3f} | alpha: {model.gravity_prior.alpha.item():.3f} | "
                  f"{elapsed:.1f}s{val_cpc_str}")

        # --- Early stopping ---
        if use_early_stopping and patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch} (best epoch {best_epoch}, best val CPC {best_val_cpc:.4f}).")
            break

    # Restore best checkpoint (if early stopping was used and improved)
    if use_early_stopping and best_state is not None:
        model.load_state_dict(best_state)
        if verbose:
            print(f"Restored best model (epoch={best_epoch}, val CPC={best_val_cpc:.4f}).")

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    info = {
        "best_epoch": best_epoch,
        "best_val_cpc": best_val_cpc if use_early_stopping else None,
        "epochs_trained": epoch,
        "stopped_early": use_early_stopping and (patience_counter >= patience),
        "val_cpc_history": val_history,
        "train_loss_history": loss_history,
    }

    if return_info:
        return model, scaler, info
    return model, scaler


