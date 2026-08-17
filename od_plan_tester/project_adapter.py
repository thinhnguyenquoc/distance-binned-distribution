"""
Project Adapter: Bridge between od_plan_tester test suite and repository source code.
"""

import sys
from pathlib import Path

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
from src.models.node_encoder import UrbanNodeEncoder
from src.models.decoder import PairwiseODDecoder
from src.models.zero_shot_model import ZeroShotODModel

# Data & Graph
from src.data.urban_graph import (
    haversine_distance_matrix,
    build_radius_graph,
    build_knn_graph,
    build_adaptive_radius_graph,
)
from src.data.dataset import CityData, load_city, load_cities, get_distance_bin_indices
from src.data.city_splits import generate_5fold_splits, get_all_cities_sorted_by_size
from src.data.yd_extractor import extract_yd_oracle, extract_yd_real, CITY_FIPS_GADM
from src.data.trip_sampler import sample_multinomial_yd, M_GRID

# Calibration
from src.calibration.bin_calibration import calibrate_by_distance_bins

# Evaluation
from src.training.evaluate import (
    compute_cpc,
    compute_cpc_norm,
    compute_rmse_log1p,
    compute_pearson_r,
    evaluate_all,
)

# Training & Experiment
from src.training.train import train_zero_shot_model, infer_zero_shot
from src.experiment.run_experiment import run_target_city_experiments, _interpolate_m_star
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
    "PairwiseODDecoder",
    "ZeroShotODModel",
    "haversine_distance_matrix",
    "build_radius_graph",
    "build_knn_graph",
    "build_adaptive_radius_graph",
    "CityData",
    "load_city",
    "load_cities",
    "get_distance_bin_indices",
    "generate_5fold_splits",
    "get_all_cities_sorted_by_size",
    "extract_yd_oracle",
    "extract_yd_real",
    "CITY_FIPS_GADM",
    "sample_multinomial_yd",
    "M_GRID",
    "calibrate_by_distance_bins",
    "compute_cpc",
    "compute_cpc_norm",
    "compute_rmse_log1p",
    "compute_pearson_r",
    "evaluate_all",
    "train_zero_shot_model",
    "infer_zero_shot",
    "run_target_city_experiments",
    "_interpolate_m_star",
    "analyze_qstar",
    "analyze_delta_r",
]
