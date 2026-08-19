"""
Experiment Runner for Moving-Bin Calibration Framework.

Experimental Conditions per Target City:
    1. M_0:                 Zero-shot baseline (pure spatial transfer)
    2. M_1^{real, +}:       Primary moving-bin Meta calibration on Omega_c^+ (q=1.0)
    3. M_1^{oracle, +}:     Oracle moving-bin reference on Omega_c^+ (q=1.0)
    4. M_1^{real, 4bin}:    Ablation deliberately retaining Bin 0 semantic mismatch
    5. M_q^{real, +}:       Soft-calibration curve across q in {0.0, 0.25, 0.5, 0.75, 1.0}
    6. M_m^+:               Multinomial trip sampling curve on Omega_c^+ (S=20 seeds per m)

Primary Metric:
    Interzonal CPC (CPC_inter) on Omega_c^+ = {(i,j) in Omega_c : i != j, D_ij > 0}
"""

import numpy as np
import torch
from typing import Dict, Any, List
from sklearn.isotonic import IsotonicRegression

from src.data.dataset import CityData, load_city
from src.data.urban_graph import build_radius_graph, build_adaptive_radius_graph, build_knn_graph
from src.data.yd_extractor import (
    extract_yd_moving_real,
    extract_yd_moving_oracle,
    extract_yd_4bin_real,
    extract_yd_4bin_oracle,
    compute_distributional_overlap,
)
from src.data.trip_sampler import sample_multinomial_yd, M_GRID
from src.calibration.bin_calibration import calibrate_moving_bins, calibrate_4bin_legacy_ablation
from src.training.evaluate import evaluate_moving_and_full
from src.training.train import infer_zero_shot, ZeroShotODModel


def _interpolate_m_star(
    target_cpc: float,
    m_finite_values: List[float],
    mean_cpcs: List[float],
    oracle_cpc: float,
    total_trips: float,
) -> tuple[float, str]:
    """
    Isotonic monotonic regression inversion to find m* matching target_cpc.
    Guarantees m* <= total_trips so that q* = m* / total_trips <= 1.0 strictly.
    """
    if total_trips <= 0:
        return 0.0, "zero_total_trips"

    # Filter finite values strictly below total_trips
    all_m = []
    all_cpc = []
    for m, cpc in zip(m_finite_values, mean_cpcs):
        if m < total_trips:
            all_m.append(float(m))
            all_cpc.append(float(cpc))

    # Always append total_trips with oracle_cpc as the finite population oracle reference
    all_m.append(float(total_trips))
    all_cpc.append(float(oracle_cpc))

    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    cpc_curve_monotonic = iso.fit_transform(all_m, all_cpc)

    if target_cpc <= cpc_curve_monotonic[0]:
        m_star = float(all_m[0])
        status = "below_min_grid"
    elif target_cpc >= cpc_curve_monotonic[-1]:
        m_star = float(all_m[-1])
        status = "at_oracle_reference"
    else:
        # Leftmost crossing rule for isotonic inversion
        idx = int(np.searchsorted(cpc_curve_monotonic, target_cpc, side="left"))
        if abs(cpc_curve_monotonic[idx] - target_cpc) < 1e-9:
            m_star = float(all_m[idx])
        else:
            prev_cpc = cpc_curve_monotonic[idx - 1]
            next_cpc = cpc_curve_monotonic[idx]
            prev_m = all_m[idx - 1]
            next_m = all_m[idx]
            if next_cpc > prev_cpc:
                frac = (target_cpc - prev_cpc) / (next_cpc - prev_cpc)
                m_star = float(prev_m + frac * (next_m - prev_m))
            else:
                m_star = float(prev_m)
        status = "interpolated" if m_star < total_trips else "at_oracle_reference"

    # Clip to total_trips to strictly enforce q* <= 1.0
    m_star = min(m_star, float(total_trips))
    return m_star, status


def run_target_city_experiments(
    model: ZeroShotODModel,
    city_name: str,
    scaler: object,
    data_root: str = "data",
    meta_prior_dir: str = "meta_prior",
    graph_type: str = "radius",
    radius_km: float = 5.0,
    knn_k: int = 10,
    num_trip_seeds: int = 20,
    m_grid: List[int | float] = M_GRID,
    device_str: str = "cpu",
    bin_edges: np.ndarray = None,
) -> Dict[str, Any]:
    assert scaler is not None, "StandardScaler must be pre-fitted on source cities."
    assert bin_edges is not None, "bin_edges must be provided for K=8 unified calibration."

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
    bin_labels = city_data.bin_labels # Not used for K=8, but kept for evaluation if needed

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
    
    n_short = np.sum((pair_dist_km[inter_mask] > 0) & (pair_dist_km[inter_mask] < 10.0))
    n_long = np.sum(pair_dist_km[inter_mask] >= 100.0)
    short_long_ratio = float(n_short) / float(n_long) if n_long > 0 else 0.0

    return {
        "city": city_name,
        "n_tracts": city_data.n_tracts,
        "n_pairs": city_data.n_pairs,
        "rho_c": rho_c,
        "average_flow": average_flow,
        "mean_distance": mean_distance,
        "short_long_ratio": short_long_ratio,
        "n_inter_pairs": n_inter_pairs,
        "total_trips": total_trips,
        "total_inter_trips": total_inter_trips,
        "M0": m0_metrics,
        "M1_city_oracle_obs": m1_city_metrics,
        "M1_county_oracle_obs": m1_county_metrics,
        "M1_subzone_oracle_obs": m1_subzone_metrics,
        "mapping_stats": mapping_stats,
    }
