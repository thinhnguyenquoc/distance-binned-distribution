"""
Master Model Training Module adhering strictly to new_plan.md.

Protocol Invariants:
1. Loss: Log1p-MSE L = Mean( (log(1+T) - log(1+\hat{T}))^2 )
2. Optimizer: AdamW, lr=2e-3, weight_decay=1e-4, grad_clip=5.0, 40 epochs.
3. Batch contract: Strict Full-Batch optimization on Train_f(s).
4. Seeds: [1, 10, 100]. Gravity model optimization is deterministic, yielding std=0.0.
5. Models trained per source city and active fraction:
   - f = 0.30 (main setting for Exp A, C, D)
   - f in {0.10, 0.20, 0.50, 1.00} (for Exp B)
6. Outputs:
   - Checkpoints saved in models/{model_name}_{source_city}_f{fraction}_seed{seed}.pt
   - manifests/gravity_parameters.csv (150 rows: 50 cities x 3 seeds for main fraction f=0.30)
   - manifests/gravity_training_trace.csv (6000 rows: 50 cities x 40 epochs x 3 seeds for main f=0.30)
   - source_city_results.csv (450 rows: 50 cities x 3 models x 3 seeds on 70% held-out for f=0.30)
   - source_city_results_mean.csv (150 rows: 50 cities x 3 models averaged across seeds)
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from implement_new_plan.loss.log1p_mse import log1p_mse_loss
from implement_new_plan.models.od_models import TwoParameterGravity, PairwiseMLP, UrbanGNN
from implement_new_plan.data.dataset import load_raw_city, RawCityData
from implement_new_plan.data.urban_graph import build_radius_graph
from implement_new_plan.training.evaluate import (
    compute_cpc_pair,
    compute_cpc_norm_pair,
    compute_mae_pair,
    compute_mse_pair,
    compute_rmse_pair,
)

CANONICAL_SEEDS = [1, 10, 100]
FRACTIONS = [0.10, 0.20, 0.30, 0.50, 1.00]


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_city_splits(raw_city: RawCityData) -> Dict[str, np.ndarray]:
    """Generates strictly deterministic nested splits on positive support."""
    o = np.asarray(raw_city.pair_o_idx)
    d = np.asarray(raw_city.pair_d_idx)
    dist = np.asarray(raw_city.dist_km)
    trips = np.asarray(raw_city.pair_trips)
    mask = (o != d) & (dist > 0.0) & (trips >= 1.0)

    df_pairs = pd.DataFrame({'origin': o[mask], 'destination': d[mask], 'idx': np.where(mask)[0]})
    df_sorted = df_pairs.sort_values(['origin', 'destination']).reset_index(drop=True)
    n = len(df_sorted)
    rng = np.random.default_rng(42)
    perm = rng.permutation(n)

    idx_arr = df_sorted['idx'].values
    idx_30 = idx_arr[perm[:int(np.floor(0.30 * n))]]
    heldout_30 = idx_arr[perm[int(np.floor(0.30 * n)):]]

    return {
        'support_idx': idx_arr,
        'heldout_30': heldout_30,
        '0.10': idx_arr[perm[:int(np.floor(0.10 * n))]],
        '0.20': idx_arr[perm[:int(np.floor(0.20 * n))]],
        '0.30': idx_30,
        '0.50': idx_arr[perm[:int(np.floor(0.50 * n))]],
        '1.00': idx_arr[perm],
    }


def train_single_gravity(
    pop_raw_o: torch.Tensor,
    pop_raw_d: torch.Tensor,
    dist_raw: torch.Tensor,
    true_flow: torch.Tensor,
    seed: int,
    epochs: int = 40,
    lr: float = 2e-3,
) -> Tuple[TwoParameterGravity, List[Dict]]:
    set_seed(seed)
    model = TwoParameterGravity(init_G=0.0, init_alpha=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    trace = []
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        pred = model(pop_raw_o, pop_raw_d, dist_raw)
        loss = log1p_mse_loss(pred, true_flow)
        assert torch.isfinite(loss), f"Gravity loss is NaN/Inf at epoch {epoch}"
        loss.backward()
        optimizer.step()

        trace.append({
            "epoch": epoch,
            "G": float(model.G.item()),
            "alpha": float(model.alpha.item()),
            "train_loss": float(loss.item()),
        })

    return model, trace


def train_single_mlp(
    x_o: torch.Tensor,
    x_d: torch.Tensor,
    dist_std: torch.Tensor,
    true_flow: torch.Tensor,
    seed: int,
    epochs: int = 40,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    grad_clip: float = 5.0,
) -> PairwiseMLP:
    set_seed(seed)
    model = PairwiseMLP(node_in_dim=x_o.shape[-1], hidden_dim=64, dropout=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        pred = model(x_o, x_d, dist_std)
        loss = log1p_mse_loss(pred, true_flow)
        assert torch.isfinite(loss), f"MLP loss is NaN/Inf at epoch {epoch}"
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

    model.eval()
    return model


def train_single_gnn(
    x_all: torch.Tensor,
    spatial_edge_index: torch.Tensor,
    spatial_edge_dist_raw: torch.Tensor,
    pair_o: torch.Tensor,
    pair_d: torch.Tensor,
    dist_raw: torch.Tensor,
    pop_raw: torch.Tensor,
    true_flow: torch.Tensor,
    seed: int,
    epochs: int = 40,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    grad_clip: float = 5.0,
) -> UrbanGNN:
    set_seed(seed)
    model = UrbanGNN(node_in_dim=x_all.shape[-1], node_hidden_dim=64, node_out_dim=64, decoder_hidden_dim=64, dropout=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        pred = model(
            x=x_all,
            spatial_edge_index=spatial_edge_index,
            spatial_edge_dist_raw=spatial_edge_dist_raw,
            pair_o_idx=pair_o,
            pair_d_idx=pair_d,
            pair_distance_km_raw=dist_raw,
            population_raw=pop_raw,
        )
        loss = log1p_mse_loss(pred, true_flow)
        assert torch.isfinite(loss), f"GNN loss is NaN/Inf at epoch {epoch}"
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

    model.eval()
    return model
