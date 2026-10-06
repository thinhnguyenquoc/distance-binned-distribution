"""
Master Runner & Pipeline Execution for Research Protocol new_plan.md.

Covers:
- Strict Reproducibility & Multi-seed initialization (seeds = [1, 10, 100])
- Full-batch AdamW optimization (40 epochs, lr=2e-3, weight_decay=1e-4, grad_clip=5.0) on Train_f(s)
- Log1p-MSE loss: L = Mean( (log(1+T) - log(1+T_hat))^2 )
- 3 Baseline families: gravity_2param, pairwise_mlp, urban_gnn
- Positive Interzonal Support Omega_t^+: origin != destination, distance_km > 0, true_flow >= 1
- Zero-shot Transfer & Caching: predictions computed once per (source, target, model, seed, fraction)
- Experiment A: f=0.30, K=8, eps=0
- Experiment B: f in {0.10, 0.20, 0.30, 0.50, 1.00}, K=8, eps=0
- Experiment C: Master grid K in {2,4,8,12,20} x eps in {0, 0.01, ..., 0.10} at f=0.30
- Experiment D: Dose-matched Scaled Donor Control at f=0.30, K=8, eps=0
- 4-Tier Statistical Inference (Tier A, Tier B, Tier C, Tier D)
- Exact file outputs matching protocol schema.
"""

import os
import sys
import time
import math
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from implement_new_plan.loss.log1p_mse import log1p_mse_loss
from implement_new_plan.models.od_models import TwoParameterGravity, PairwiseMLP, UrbanGNN
from implement_new_plan.data.dataset import load_raw_city, RawCityData, NODE_FEATURE_COLUMNS
from implement_new_plan.data.urban_graph import build_radius_graph
from implement_new_plan.data.source_scaler import SourceCityFeatureScaler, SingleFeatureScaler, SKEWED_FEATURES
from implement_new_plan.calibration.source_bins import (
    assign_to_source_bins,
    build_target_dbd,
    calibrate_dbd,
    compute_pure_calibration_ratios,
    apply_pure_dbd_calibration,
)
from implement_new_plan.calibration.tv_noise import generate_exact_tv_noise, derive_noise_seed
from implement_new_plan.calibration.dose_matching import (
    compute_source_binned_dbd,
    compute_rms_log_ratio_dose,
    find_dose_matching_lambda,
    reconstruct_scaled_donor,
    fit_crossed_mixed_effects,
)
from implement_new_plan.training.evaluate import (
    compute_cpc_pair,
    compute_cpc_norm_pair,
    compute_mae_pair,
    compute_mse_pair,
    compute_rmse_pair,
    evaluate_calibration_transfer,
)
from implement_new_plan.calibration.statistical_inference import (
    aggregate_seeds,
    compute_target_city_summary,
    compute_global_target_inference,
    compute_source_city_summary,
    fit_crossed_random_effects,
    compute_scarcity_contrasts,
    compute_scarcity_contrast_target_summary,
    compute_scarcity_contrast_inference,
    fit_scarcity_contrast_mixed_effects,
    compute_gap_recovery,
    fit_scarcity_overall_mixed_effects,
)

CANONICAL_SEEDS = [1, 10, 100]
FRACTIONS = [0.10, 0.20, 0.30, 0.50, 1.00]
K_GRID = [2, 4, 8, 12, 20]
EPS_GRID = [0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10]
GLOBAL_NOISE_SEED = 42


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


def load_source_scaler(source_city: str, scalers_df: pd.DataFrame) -> Dict[str, Any]:
    sub = scalers_df[scalers_df['source_city'] == source_city]
    node_means = []
    node_stds = []
    node_transforms = []
    node_imputes = []
    zero_flags = []
    
    for feat in sub[sub['feature_type'] == 'node_feature']['feature_name'].values:
        row = sub[sub['feature_name'] == feat].iloc[0]
        node_means.append(float(row['mean']))
        node_stds.append(float(row['std']))
        node_transforms.append(row['transform'])
        node_imputes.append(float(row['imputation_value']))
        zero_flags.append(bool(row['zero_variance_flag']))
        
    dist_row = sub[sub['feature_type'] == 'pairwise_distance'].iloc[0]
    return {
        'node_means': np.array(node_means, dtype=np.float64),
        'node_stds': np.array(node_stds, dtype=np.float64),
        'node_transforms': node_transforms,
        'node_imputes': np.array(node_imputes, dtype=np.float64),
        'zero_flags': zero_flags,
        'dist_mean': float(dist_row['mean']),
        'dist_std': float(dist_row['std']),
        'dist_impute': float(dist_row['imputation_value']),
    }


def transform_nodes_with_scaler(raw_x: np.ndarray, scaler_dict: Dict[str, Any]) -> np.ndarray:
    x_out = np.zeros_like(raw_x, dtype=np.float32)
    for i in range(raw_x.shape[1]):
        vals = raw_x[:, i].copy().astype(np.float64)
        vals[~np.isfinite(vals)] = scaler_dict['node_imputes'][i]
        if scaler_dict['node_transforms'][i] == 'log1p_zscore':
            vals = np.log1p(np.maximum(0.0, vals))
        if scaler_dict['zero_flags'][i] or scaler_dict['node_stds'][i] < 1e-12:
            x_out[:, i] = 0.0
        else:
            x_out[:, i] = (vals - scaler_dict['node_means'][i]) / scaler_dict['node_stds'][i]
    return x_out


def impute_pop_with_scaler(raw_pop: np.ndarray, scaler_dict: Dict[str, Any]) -> np.ndarray:
    pop_arr = raw_pop.copy().astype(np.float64)
    # total_population is column 0
    pop_impute = scaler_dict['node_imputes'][0]
    pop_arr[~np.isfinite(pop_arr)] = pop_impute
    return np.maximum(0.0, pop_arr).astype(np.float32)


def transform_dist_with_scaler(raw_dist: np.ndarray, scaler_dict: Dict[str, Any]) -> np.ndarray:
    d = raw_dist.copy().astype(np.float64)
    d[~np.isfinite(d)] = scaler_dict['dist_impute']
    d_log = np.log1p(np.maximum(0.0, d))
    if scaler_dict['dist_std'] < 1e-12:
        return np.zeros_like(d, dtype=np.float32)
    return ((d_log - scaler_dict['dist_mean']) / scaler_dict['dist_std']).astype(np.float32)


def train_gravity(
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


def train_mlp(
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


def train_gnn(
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
