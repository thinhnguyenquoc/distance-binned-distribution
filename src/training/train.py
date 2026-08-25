r"""
Cross-City Training and Transfer Pipeline.

Stage A: Cross-city Training
    Trains ZeroShotODModel on a list of source cities using ZTNB likelihood on all positive observed support (including intrazonal):
        L_train = - 1 / |Omega^+_all| * sum_{(i,j) in Omega^+_all} log P_ZTNB(T_ij; mu_nb_ij, phi)
    City-level losses are averaged within city and optimization proceeds city-by-city, preventing large-support cities from dominating solely through pair count.
    After convergence, freezes parameters -> theta*.

Stage B: Zero-Shot Transfer Evaluation
    Evaluates theta* on held-out target city, evaluating the primary reconstruction estimand on positive interzonal support (Omega_c^+):
        (X^{c*}, G^{urban, c*}, D^{c*}) -> \hat{T}^{ZS} = E[T | T >= 1].
"""

import copy
import time
import datetime
from pathlib import Path
from typing import List, Dict, Optional, Union

import torch
import torch.optim as optim

from src.data.dataset import (
    CityData,
    NODE_FEATURE_COLUMNS,
    get_scaler_fingerprint,
    load_cities,
    load_city,
    validate_feature_scaler,
)
from src.data.urban_graph import build_radius_graph, build_knn_graph
from src.models.zero_shot_model import ZeroShotODModel
from src.loss.ztnb import ztnb_nll, nb_nll
from src.training.evaluate import evaluate_all


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: Union[str, Path],
    model: "ZeroShotODModel",
    scaler: object,
    train_info: dict,
    hyperparams: dict,
    seed: Optional[int] = None,
    run_tag: Optional[str] = None,
) -> Path:
    """
    Persists a trained ZeroShotODModel checkpoint to disk.

    Saved bundle contains:
        - model_state_dict   : weights (best validation checkpoint)
        - scaler_*           : StandardScaler statistics for feature normalization
        - train_info         : best_epoch, best_val_cpc, epochs_trained, histories
        - hyperparams        : architecture + training config needed to reconstruct model
        - seed               : random seed used for this run (None if not set)
        - run_tag            : human-readable label, e.g. "e1_fold1"
        - saved_at           : ISO-8601 UTC timestamp

    Args:
        path:        Full file path to write (created with parents if needed).
        model:       Trained (and eval-mode) ZeroShotODModel instance.
        scaler:      Fitted sklearn StandardScaler from load_cities().
        train_info:  Dict returned by train_zero_shot_model() when return_info=True.
        hyperparams: Dict of architecture / training hyper-parameters.
        seed:        Random seed (optional).
        run_tag:     Short label for this run (optional).

    Returns:
        Resolved Path of the saved file.
    """
    import numpy as _np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    scaler_data: dict = {}
    if scaler is not None and hasattr(scaler, "mean_") and scaler.mean_ is not None:
        validate_feature_scaler(scaler)
        scaler_data = {
            "scaler_mean_":  _np.asarray(scaler.mean_,  dtype=_np.float64),
            "scaler_scale_": _np.asarray(scaler.scale_, dtype=_np.float64),
            "scaler_var_":   _np.asarray(scaler.var_,   dtype=_np.float64),
            "scaler_n_features_in_": int(getattr(scaler, "n_features_in_", len(scaler.mean_))),
            "scaler_fingerprint": get_scaler_fingerprint(scaler),
            "scaler_feature_columns": list(NODE_FEATURE_COLUMNS),
        }
        if hasattr(scaler, "n_samples_seen_"):
            scaler_data["scaler_n_samples_seen_"] = _np.asarray(
                scaler.n_samples_seen_
            ).copy()

    bundle = {
        "model_state_dict": model.state_dict(),
        **scaler_data,
        "train_info":   train_info,
        "hyperparams":  hyperparams,
        "seed":         seed,
        "run_tag":      run_tag,
        "saved_at":     datetime.datetime.utcnow().isoformat() + "Z",
    }

    torch.save(bundle, path)
    return path.resolve()


def load_checkpoint(
    path: Union[str, Path],
    device_str: str = "cpu",
    expected_config: Optional[dict] = None,
) -> tuple:
    """
    Loads a checkpoint saved by save_checkpoint() and reconstructs the model and scaler.

    Args:
        path:       Path to the .pt checkpoint file.
        device_str: Device to map model weights onto ("cpu" or "cuda").
        expected_config: Optional dictionary of hyperparams to validate against the checkpoint.

    Returns:
        (model, scaler, metadata) where:
            model    — ZeroShotODModel in eval mode with frozen weights
            scaler   — Reconstructed sklearn StandardScaler (or None if not saved)
            metadata — Full checkpoint dict (train_info, hyperparams, seed, run_tag, saved_at)
    """
    import numpy as _np

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    bundle = torch.load(path, map_location=torch.device(device_str), weights_only=False)

    hp = bundle["hyperparams"]
    if expected_config is not None:
        for k, v in expected_config.items():
            if k not in hp:
                raise ValueError(f"Checkpoint config missing key '{k}' in {path}. Expected {v}. Checkpoint may be incomplete.")
            if hp[k] != v:
                raise ValueError(f"Checkpoint config mismatch in {path} for key '{k}': expected {v}, got {hp[k]}. Delete the stale checkpoint to retrain.")

    # --- Reconstruct model ---
    hp = bundle["hyperparams"]
    backbone = hp.get("backbone", "gnn")
    
    from src.models.zero_shot_model import ZeroShotODModel, ZeroShotMLPModel
    
    if backbone == "mlp":
        model = ZeroShotMLPModel(
            node_in_dim       = hp["node_in_dim"],
            node_hidden_dim   = hp["hidden_dim"],
            node_out_dim      = hp["hidden_dim"],
            num_gnn_layers    = hp["num_gnn_layers"],
            decoder_hidden_dim= hp["hidden_dim"],
        ).to(torch.device(device_str))
    else:
        model = ZeroShotODModel(
            node_in_dim       = hp["node_in_dim"],
            node_hidden_dim   = hp["hidden_dim"],
            node_out_dim      = hp["hidden_dim"],
            num_gnn_layers    = hp["num_gnn_layers"],
            decoder_hidden_dim= hp["hidden_dim"],
        ).to(torch.device(device_str))
        
    model.load_state_dict(bundle["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # --- Reconstruct scaler ---
    scaler = None
    if "scaler_mean_" in bundle:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.mean_  = bundle["scaler_mean_"]
        scaler.scale_ = bundle["scaler_scale_"]
        scaler.var_   = bundle["scaler_var_"]
        scaler.n_features_in_ = bundle.get("scaler_n_features_in_", len(scaler.mean_))
        if "scaler_n_samples_seen_" in bundle:
            scaler.n_samples_seen_ = bundle["scaler_n_samples_seen_"]
        validate_feature_scaler(scaler)

        expected_columns = bundle.get("scaler_feature_columns")
        if expected_columns is not None and tuple(expected_columns) != NODE_FEATURE_COLUMNS:
            raise ValueError(f"Checkpoint feature schema mismatch in {path}")

        expected_fingerprint = bundle.get("scaler_fingerprint")
        actual_fingerprint = get_scaler_fingerprint(scaler)
        if expected_fingerprint is not None and actual_fingerprint != expected_fingerprint:
            raise ValueError(f"Checkpoint scaler fingerprint mismatch in {path}")

    metadata = {
        "train_info":  bundle.get("train_info"),
        "hyperparams": bundle.get("hyperparams"),
        "seed":        bundle.get("seed"),
        "run_tag":     bundle.get("run_tag"),
        "saved_at":    bundle.get("saved_at"),
        "scaler_provenance": {
            "fingerprint": bundle.get("scaler_fingerprint"),
            "n_features_in": bundle.get("scaler_n_features_in_"),
            "n_samples_seen": bundle.get("scaler_n_samples_seen_"),
            "feature_columns": bundle.get("scaler_feature_columns"),
        },
    }

    return model, scaler, metadata


def train_epoch(
    model: torch.nn.Module,
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

        if not torch.isfinite(loss):
            raise FloatingPointError(f"NaN/Inf loss encountered for {city_data.city_name}. Stopping training to avoid invalid checkpoint.")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(num_cities, 1)


@torch.no_grad()
def infer_zero_shot(
    model: torch.nn.Module,
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
    epochs: int = 200,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    hidden_dim: int = 64,
    num_gnn_layers: int = 2,
    graph_type: str = "radius",
    radius_km: float = 5.0,
    knn_k: int = 10,
    loss_type: str = "ztnb",
    backbone: str = "gnn",
    dropout: float = 0.1,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = True,
    # --- Validation / early stopping ---
    val_city_names: List[str] | None = None,
    patience: int = 15,
    min_delta: float = 1e-4,
    lr_plateau_patience: int = 4,
    lr_plateau_factor: float = 0.5,
    lr_plateau_threshold: float = 1e-4,
    threshold_mode: str = "abs",
    min_lr: float = 1e-5,
    return_info: bool = False,
    seed: int | None = None,
    # --- Checkpoint provenance ---
    fold: int | None = None,
    split_manifest_sha256: str | None = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
    run_tag: Optional[str] = None,
) -> tuple:

    """
    Train ZeroShotODModel with AdamW, ReduceLROnPlateau, and validation-based early stopping.

    Args:
        train_city_names: Cities to train on.
        val_city_names:   Validation cities for early stopping. If None,
                          trains for exactly `epochs` epochs (pre-specified).
        patience:         Epochs without val CPC improvement before stopping.
        min_delta:        Minimum improvement to count as improvement.
        return_info:      If True, returns (model, scaler, train_info_dict).
        seed:             Optional random seed for reproducible weight initialization.
        checkpoint_path:  If provided, saves the trained model to this path as a .pt file.
                          Parent directories are created automatically.
        run_tag:          Short label embedded in the checkpoint (e.g. "e1_fold1_seed2025").

    Returns:
        (best_model, scaler) or (best_model, scaler, info)
    """
    import copy
    import numpy as _np

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        _np.random.seed(seed)

    device = torch.device(device_str)

    if verbose:
        print(f"    [Setup] Precomputing graph structures for {len(train_city_names)} source cities onto {device}...", flush=True)

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

    # Pre-move training tensors onto device to avoid repeated host-to-device transfers per epoch
    train_cities_dev = [
        CityData(
            city_name     = c.city_name,
            n_tracts      = c.n_tracts,
            n_pairs       = c.n_pairs,
            node_features = c.node_features.to(device),
            population    = c.population.to(device),
            lon_lat       = c.lon_lat.to(device),
            pair_o_idx    = c.pair_o_idx.to(device),
            pair_d_idx    = c.pair_d_idx.to(device),
            pair_distance = c.pair_distance.to(device),
            pair_trips    = c.pair_trips.to(device),
            bin_labels    = c.bin_labels.to(device),
        )
        for c in train_cities
    ]
    city_graphs_dev = [(ei.to(device), ed.to(device)) for (ei, ed) in city_graphs]

    # Precompute device-resident structures & masks for validation cities (if provided)
    val_data_on_device = []
    if val_city_names:
        for name in val_city_names:
            vc = load_city(name, data_root=data_root, feature_scaler=scaler)
            coords = vc.lon_lat.numpy()
            if graph_type == "radius":
                ei, ed = build_radius_graph(coords, radius_km=radius_km)
            else:
                ei, ed = build_knn_graph(coords, k=knn_k)

            # Precompute interzonal mask and ground truth on device once
            dist_km = _np.expm1(vc.pair_distance.numpy())
            inter_cpu = (vc.pair_o_idx.numpy() != vc.pair_d_idx.numpy()) & (dist_km > 0.0)
            inter_mask = torch.tensor(inter_cpu, dtype=torch.bool, device=device)
            t_gt_inter = vc.pair_trips.to(device)[inter_mask]

            val_data_on_device.append({
                "x": vc.node_features.to(device),
                "ei": ei.to(device),
                "ed": ed.to(device),
                "p_o": vc.pair_o_idx.to(device),
                "p_d": vc.pair_d_idx.to(device),
                "p_dist": vc.pair_distance.to(device),
                "pop": vc.population.to(device),
                "inter_mask": inter_mask,
                "t_gt_inter": t_gt_inter,
                "t_gt_sum": torch.sum(t_gt_inter),
                "has_inter": bool(inter_cpu.sum() > 0),
            })

    from src.models.zero_shot_model import ZeroShotODModel, ZeroShotMLPModel

    if backbone == "mlp":
        model = ZeroShotMLPModel(
            node_in_dim=train_cities[0].node_features.shape[1],
            node_hidden_dim=hidden_dim,
            node_out_dim=hidden_dim,
            num_gnn_layers=num_gnn_layers,
            decoder_hidden_dim=hidden_dim,
            dropout=dropout,
        ).to(device)
    else:
        model = ZeroShotODModel(
            node_in_dim=train_cities[0].node_features.shape[1],
            node_hidden_dim=hidden_dim,
            node_out_dim=hidden_dim,
            num_gnn_layers=num_gnn_layers,
            decoder_hidden_dim=hidden_dim,
            dropout=dropout,
        ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if val_city_names:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=lr_plateau_factor,
            patience=lr_plateau_patience,
            threshold=lr_plateau_threshold,
            threshold_mode=threshold_mode,
            min_lr=min_lr,
        )
    else:
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
            train_cities=train_cities_dev,
            city_graphs=city_graphs_dev,
            optimizer=optimizer,
            loss_type=loss_type,
            device=device,
        )
        loss_history.append(loss_val)

        # --- Fast GPU-Vectorized Validation CPC (interzonal) ---
        val_cpc_str = ""
        if use_early_stopping and val_data_on_device:
            val_cpcs = []
            model.eval()
            with torch.no_grad():
                for item in val_data_on_device:
                    if not item["has_inter"]:
                        print(f"    [WARNING] Validation city '{item.get('city_name', '?')}' has no interzonal pairs — skipped in CPC computation. Check data integrity.", flush=True)
                        continue
                    t_hat = model(
                        item["x"], item["ei"], item["ed"],
                        item["p_o"], item["p_d"], item["p_dist"],
                        item["pop"], return_conditional_mean=True
                    )
                    t_hat_inter = t_hat[item["inter_mask"]]
                    sum_min = torch.sum(torch.minimum(item["t_gt_inter"], t_hat_inter))
                    sum_total = item["t_gt_sum"] + torch.sum(t_hat_inter)
                    cpc_val = (2.0 * sum_min / sum_total).item() if sum_total > 0 else 0.0
                    val_cpcs.append(cpc_val)

            if not val_cpcs:
                raise RuntimeError(
                    "All validation cities were skipped (no interzonal pairs). "
                    "Cannot compute validation CPC. Check dataset construction."
                )
            mean_val_cpc = float(_np.mean(val_cpcs))
            val_history.append(mean_val_cpc)
            val_cpc_str = f" | ValCPC: {mean_val_cpc:.4f}"

            # Step plateau scheduler on validation metric
            scheduler.step(mean_val_cpc)

            # Best-model tracking
            if mean_val_cpc > best_val_cpc + min_delta:
                best_val_cpc = mean_val_cpc
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
        else:
            scheduler.step()

        if verbose:
            elapsed = time.time() - start_time
            pat_str = f" | Patience: {patience_counter}/{patience}" if use_early_stopping else ""
            curr_lr = optimizer.param_groups[0]["lr"]
            print(
                f"    [Epoch {epoch:03d}/{epochs:03d}] Loss: {loss_val:.4f}{val_cpc_str}{pat_str} | "
                f"lr: {curr_lr:.1e} | phi: {model.phi.item():.3f} | {elapsed:.1f}s",
                flush=True,
            )

        # --- Early stopping ---
        if use_early_stopping and patience_counter >= patience:
            if verbose:
                print(f"    -> Early stopping triggered at epoch {epoch} (best epoch {best_epoch}, best val CPC {best_val_cpc:.4f}).", flush=True)
            break

    # Restore best checkpoint (if early stopping was used and improved)
    if use_early_stopping and best_state is not None:
        model.load_state_dict(best_state)
        if verbose:
            print(f"    -> Restored best model checkpoint (epoch={best_epoch}, val CPC={best_val_cpc:.4f}).", flush=True)

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

    # --- Persist checkpoint to disk if requested ---
    if checkpoint_path is not None:
        # C1: split_manifest_sha256 must be passed explicitly; raise if caller forgot.
        if split_manifest_sha256 is None:
            raise ValueError(
                "split_manifest_sha256 must be provided when saving a checkpoint. "
                "Load the split manifest and pass its SHA256 hash to train_zero_shot_model()."
            )
        hp = {
            "node_in_dim":           train_cities[0].node_features.shape[1],
            "hidden_dim":            hidden_dim,
            "num_gnn_layers":        num_gnn_layers,
            "dropout":               dropout,
            "graph_type":            graph_type,
            "radius_km":             radius_km,
            "knn_k":                 knn_k,
            "loss_type":             loss_type,
            "epochs":                epochs,
            "lr":                    lr,
            "weight_decay":          weight_decay,
            "backbone":              backbone,
            "patience":              patience,
            "min_delta":             min_delta,
            "lr_plateau_patience":   lr_plateau_patience,
            "lr_plateau_factor":     lr_plateau_factor,
            "lr_plateau_threshold":  lr_plateau_threshold,
            "threshold_mode":        threshold_mode,
            "min_lr":                min_lr,
            # Provenance fields (C2, C1)
            "fold":                  fold,
            "split_manifest_sha256": split_manifest_sha256,
            "scaler_fit_scope":      "training_split_only",
            "scaler_weighting":      "per_tract",
            "scaler_fit_cities":     sorted(train_city_names),
            "scaler_fit_n_cities":   len(train_city_names),
            "scaler_fit_n_rows":     int(scaler.n_samples_seen_),
        }
        saved_path = save_checkpoint(
            path=checkpoint_path,
            model=model,
            scaler=scaler,
            train_info=info,
            hyperparams=hp,
            seed=seed,
            run_tag=run_tag,
        )
        if verbose:
            print(f"    -> Checkpoint saved: {saved_path}", flush=True)

    if return_info:
        return model, scaler, info
    return model, scaler
