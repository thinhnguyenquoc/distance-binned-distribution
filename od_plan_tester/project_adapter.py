"""
Project Adapter: Bridge between od_plan_tester test suite and moving-bin framework.
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Loss & Oracle
from src.loss.ztnb import (
    nb_log_prob,
    nb_log_prob_at_zero,
    ztnb_nll,
    nb_nll,
    compute_conditional_mean,
)

# Models
from src.models.gravity import GravityPrior
from src.models.node_encoder import UrbanGNN
UrbanNodeEncoder = UrbanGNN
from src.models.decoder import PairwiseODDecoder
from src.models.zero_shot_model import ZeroShotODModel

# Data & Graph
from src.data.urban_graph import (
    haversine_distance_matrix,
    build_radius_graph,
    build_knn_graph,
    build_adaptive_radius_graph,
)
from src.data.dataset import CityData, load_city, load_cities, assign_bins
get_distance_bin_indices = assign_bins
from src.data.city_splits import generate_5fold_splits, get_all_cities_sorted_by_size
from src.data.trip_sampler import sample_multinomial_yd, M_GRID
from src.data.yd_extractor import (
    extract_yd_4bin_oracle,
    extract_yd_4bin_real,
    extract_yd_moving_oracle,
    extract_M1_city_oracle_obs,
    compute_distributional_overlap,
    CITY_FIPS_GADM,
)

# Calibration: Primary is calibrate_moving_bins
from src.calibration.bin_calibration import (
    calibrate_moving_bins,
    calibrate_4bin_legacy_ablation,
)

# Evaluation
from src.training.evaluate import (
    compute_cpc_pair,
    compute_cpc_norm_pair,
    compute_rmse_log1p_pair,
    compute_pearson_pair,
    evaluate_all,
    evaluate_moving_and_full,
)

def compute_cpc(t_true, t_pred) -> float:
    t_t = t_true.detach().cpu().numpy() if isinstance(t_true, torch.Tensor) else np.asarray(t_true)
    t_p = t_pred.detach().cpu().numpy() if isinstance(t_pred, torch.Tensor) else np.asarray(t_pred)
    return compute_cpc_pair(t_t, t_p)

def compute_cpc_norm(t_true, t_pred) -> float:
    t_t = t_true.detach().cpu().numpy() if isinstance(t_true, torch.Tensor) else np.asarray(t_true)
    t_p = t_pred.detach().cpu().numpy() if isinstance(t_pred, torch.Tensor) else np.asarray(t_pred)
    return compute_cpc_norm_pair(t_t, t_p)

def compute_rmse_log1p(t_true, t_pred) -> float:
    t_t = t_true.detach().cpu().numpy() if isinstance(t_true, torch.Tensor) else np.asarray(t_true)
    t_p = t_pred.detach().cpu().numpy() if isinstance(t_pred, torch.Tensor) else np.asarray(t_pred)
    return compute_rmse_log1p_pair(t_t, t_p)

def compute_pearson_r(t_true, t_pred) -> float:
    t_t = t_true.detach().cpu().numpy() if isinstance(t_true, torch.Tensor) else np.asarray(t_true)
    t_p = t_pred.detach().cpu().numpy() if isinstance(t_pred, torch.Tensor) else np.asarray(t_pred)
    return compute_pearson_pair(t_t, t_p)

# Training & Experiment
from src.training.train import train_zero_shot_model, infer_zero_shot
from src.experiment.run_experiment import run_target_city_experiments
from src.experiment.compute_qstar import analyze_qstar
from src.experiment.compute_delta_r import analyze_delta_r

__all__ = [
    "nb_log_prob",
    "nb_log_prob_at_zero",
    "ztnb_nll",
    "nb_nll",
    "compute_conditional_mean",
    "GravityPrior",
    "UrbanNodeEncoder",
    "UrbanGNN",
    "PairwiseODDecoder",
    "ZeroShotODModel",
    "haversine_distance_matrix",
    "build_radius_graph",
    "build_knn_graph",
    "build_adaptive_radius_graph",
    "CityData",
    "load_city",
    "load_cities",
    "assign_bins",
    "get_distance_bin_indices",
    "generate_5fold_splits",
    "get_all_cities_sorted_by_size",
    "extract_yd_moving_oracle",
    "extract_M1_city_oracle_obs",
    "extract_yd_4bin_oracle",
    "extract_yd_4bin_real",
    "compute_distributional_overlap",
    "CITY_FIPS_GADM",
    "sample_multinomial_yd",
    "M_GRID",
    "calibrate_moving_bins",
    "calibrate_4bin_legacy_ablation",
    "compute_cpc",
    "compute_cpc_norm",
    "compute_rmse_log1p",
    "compute_pearson_r",
    "evaluate_all",
    "evaluate_moving_and_full",
    "train_zero_shot_model",
    "infer_zero_shot",
    "run_target_city_experiments",

    "analyze_qstar",
    "analyze_delta_r",
]
