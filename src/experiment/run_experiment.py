"""
Experiment Runner for Moving-Bin Calibration Framework.

Experimental Conditions per Target City:
    1. M_0:                 Zero-shot baseline (pure spatial transfer)
    2. M_1^{city}:          Primary moving-bin Meta calibration on Omega_c^+ (q=1.0)
    3. M_1^{county}:        County-level calibration
    4. M_1^{subzone}:       Subzone-level calibration

Primary Metric:
    Interzonal CPC (CPC_inter) on Omega_c^+ = {(i,j) in Omega_c : i != j, D_ij > 0}
"""

import numpy as np
import torch
from typing import Dict, Any, List

from src.data.dataset import CityData, load_city
from src.data.urban_graph import build_radius_graph, build_adaptive_radius_graph, build_knn_graph
from src.data.yd_extractor import (
    extract_M1_city_oracle_obs,
)
from src.data.trip_sampler import M_GRID
from src.training.evaluate import evaluate_moving_and_full
from src.training.train import infer_zero_shot




def run_target_city_experiments(
    model: torch.nn.Module,
    city_name: str,
    scaler: object,
    data_root: str = "data",
    graph_type: str = "radius",
    radius_km: float = 5.0,
    knn_k: int = 10,
    device_str: str = "cpu",
    bin_edges: np.ndarray = None,
) -> Dict[str, Any]:
    assert scaler is not None, "StandardScaler must be pre-fitted on source cities."
    if bin_edges is None:
        raise ValueError("bin_edges must be provided from training cities to avoid data leakage.")

    device = torch.device(device_str)
    city_data = load_city(city_name, data_root=data_root, feature_scaler=scaler, fit_scaler=False)
    coords = city_data.lon_lat.numpy()

    if graph_type == "adaptive_radius":
        edge_index, edge_dist, _ = build_adaptive_radius_graph(coords, scale_fraction=0.15)
    elif graph_type == "radius":
        edge_index, edge_dist = build_radius_graph(coords, radius_km=radius_km)
    else:
        edge_index, edge_dist = build_knn_graph(coords, k=knn_k)

    t_true = city_data.pair_trips.numpy().astype(np.float64)
    pair_o = city_data.pair_o_idx.numpy()
    pair_d = city_data.pair_d_idx.numpy()
    pair_dist = city_data.pair_distance.numpy()
    pair_dist_km = np.expm1(pair_dist)
    inter_mask = (pair_o != pair_d) & (pair_dist_km > 0.0)
    n_inter_pairs = int(inter_mask.sum())
    total_inter_trips = float(t_true[inter_mask].sum())
    total_trips = float(t_true.sum())

    # Extract county grouping (GADM 4.1 level-2 point-in-polygon mapping)
    import pandas as pd
    from pathlib import Path
    from src.data.gadm_mapper import get_gadm_gid2_mapping
    
    meta_df = pd.read_csv(Path(data_root) / city_name / "meta.csv")
    assert meta_df["idx"].is_unique, "Mapping invariant failed: meta_df['idx'] has duplicates"
    assert set(pair_o).issubset(set(meta_df["idx"])), "Mapping invariant failed: some pair_o indices are not in meta.csv"
    
    # Get mapping robustly relative to repository root
    repo_root = str(Path(__file__).resolve().parents[2])
    tract_to_county, mapping_stats = get_gadm_gid2_mapping(meta_df, repo_root)
    
    pair_county_idx = np.array([tract_to_county[i] for i in pair_o])
    assert len(pair_county_idx) == len(pair_o), "Mapping invariant failed: length mismatch after county mapping"

    from src.data.yd_extractor import extract_yd_kbins, extract_yd_kbins_grouped
    from src.calibration.bin_calibration import calibrate_kbins, calibrate_kbins_grouped

    # -----------------------------------------------------------------------
    # Condition M0: Pure Zero-Shot Inference
    # -----------------------------------------------------------------------
    t_pred_zs_tensor = infer_zero_shot(model, city_data, edge_index, edge_dist, device=device)
    t_pred_zs = t_pred_zs_tensor.numpy().astype(np.float64)
    m0_metrics = evaluate_moving_and_full(
        city_data.pair_trips, t_pred_zs_tensor, city_data.pair_o_idx, city_data.pair_d_idx, city_data.bin_labels, pair_distance=city_data.pair_distance
    )

    # -----------------------------------------------------------------------
    # Condition M1_city: City-Level Oracle Y_D
    # -----------------------------------------------------------------------
    yd_city = extract_yd_kbins(pair_dist_km, t_true, bin_edges, inter_mask)
    t_pred_city = calibrate_kbins(t_pred_zs, pair_dist_km, inter_mask, yd_city, bin_edges, q=1.0)
    m1_city_metrics = evaluate_moving_and_full(
        city_data.pair_trips, torch.tensor(t_pred_city), city_data.pair_o_idx, city_data.pair_d_idx, city_data.bin_labels, pair_distance=city_data.pair_distance
    )

    # -----------------------------------------------------------------------
    # Condition M1_county: County-Level Oracle Y_D
    # -----------------------------------------------------------------------
    yd_county_dict = extract_yd_kbins_grouped(pair_dist_km, t_true, bin_edges, inter_mask, pair_county_idx)
    t_pred_county = calibrate_kbins_grouped(t_pred_zs, pair_dist_km, inter_mask, yd_county_dict, bin_edges, pair_county_idx, q=1.0)
    m1_county_metrics = evaluate_moving_and_full(
        city_data.pair_trips, torch.tensor(t_pred_county), city_data.pair_o_idx, city_data.pair_d_idx, city_data.bin_labels, pair_distance=city_data.pair_distance
    )

    # -----------------------------------------------------------------------
    # Condition M1_subzone: Tract-Level (Subzone) Oracle Y_D
    # -----------------------------------------------------------------------
    yd_subzone_dict = extract_yd_kbins_grouped(pair_dist_km, t_true, bin_edges, inter_mask, pair_o)
    t_pred_subzone = calibrate_kbins_grouped(t_pred_zs, pair_dist_km, inter_mask, yd_subzone_dict, bin_edges, pair_o, q=1.0)
    m1_subzone_metrics = evaluate_moving_and_full(
        city_data.pair_trips, torch.tensor(t_pred_subzone), city_data.pair_o_idx, city_data.pair_d_idx, city_data.bin_labels, pair_distance=city_data.pair_distance
    )

    rho_c = float(n_inter_pairs) / (float(city_data.n_tracts) * float(city_data.n_tracts - 1)) if city_data.n_tracts > 1 else 0.0
    average_flow = total_inter_trips / n_inter_pairs if n_inter_pairs > 0 else 0.0
    mean_distance = float(np.mean(pair_dist_km[inter_mask])) if n_inter_pairs > 0 else 0.0
    
    return {
        "city": city_name,
        "n_tracts": city_data.n_tracts,
        "n_pairs": city_data.n_pairs,
        "rho_c": rho_c,
        "average_flow": average_flow,
        "mean_distance": mean_distance,
        "n_inter_pairs": n_inter_pairs,
        "total_trips": total_trips,
        "total_inter_trips": total_inter_trips,
        "M0": m0_metrics,
        "M1_city_oracle_obs": m1_city_metrics,
        "M1_county_oracle_obs": m1_county_metrics,
        "M1_subzone_oracle_obs": m1_subzone_metrics,
        "mapping_stats": mapping_stats,
    }
