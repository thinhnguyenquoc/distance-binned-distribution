"""
Experiment Runner for All 4 Conditions per Target City:
    M_0:         Zero-shot baseline
    M_1^oracle:   Zero-shot + Y_D^oracle (from GT over Omega_c)
    M_1^real:     Zero-shot + Y_D^real (from Meta mobility data across temporal snapshots)
    M_q:          Zero-shot + \tilde{Y}_D^{(m)} (from Multinomial trip sampling across m grid, S=20 seeds)

Monotonic Isotonic Inversion with Asymptotic Infinity Handling:
    m grid includes m in {100, 500, 1k, 5k, 10k, 50k, 100k, inf}.
    m = inf corresponds to full sampling (T_total trips) achieving M1_oracle CPC.
    If target CPC exceeds 100k level, interpolates smoothly towards T_total and flags status.
"""

import numpy as np
import torch
from typing import Dict, Any, List
from sklearn.isotonic import IsotonicRegression

from src.data.dataset import CityData, load_city
from src.data.urban_graph import build_radius_graph, build_adaptive_radius_graph, build_knn_graph
from src.data.yd_extractor import extract_yd_oracle, extract_yd_real
from src.data.trip_sampler import sample_multinomial_yd, M_GRID
from src.calibration.bin_calibration import calibrate_by_distance_bins
from src.training.evaluate import evaluate_all
from src.training.train import infer_zero_shot, ZeroShotODModel


def _interpolate_m_star(
    target_cpc: float,
    m_finite_values: List[float],
    mean_cpcs: List[float],
    oracle_cpc: float,
    total_trips: float,
) -> tuple[float, str]:
    """
    Isotonic monotonic regression inversion to find m* matching target_cpc,
    incorporating the asymptotic oracle limit at m = total_trips (inf).

    Returns:
        (m_star, status)
    """
    # Append total_trips with oracle_cpc as upper asymptote
    all_m = list(m_finite_values)
    all_cpc = list(mean_cpcs)

    if total_trips > all_m[-1]:
        all_m.append(total_trips)
        all_cpc.append(oracle_cpc)

    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    cpc_curve_monotonic = iso.fit_transform(all_m, all_cpc)

    if target_cpc <= cpc_curve_monotonic[0]:
        m_star = float(all_m[0])
        status = "below_min_grid"
    elif target_cpc >= cpc_curve_monotonic[-1]:
        m_star = float(all_m[-1])
        status = "at_oracle_ceiling"
    else:
        m_star = float(np.interp(target_cpc, cpc_curve_monotonic, all_m))
        status = "interpolated" if m_star <= 100000.0 else "extrapolated_towards_total"

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
) -> Dict[str, Any]:
    assert scaler is not None, "StandardScaler must be pre-fitted on source cities."

    device = torch.device(device_str)
    city_data = load_city(city_name, data_root=data_root, feature_scaler=scaler, fit_scaler=False)
    coords = city_data.lon_lat.numpy()

    if graph_type == "adaptive_radius":
        edge_index, edge_dist, _ = build_adaptive_radius_graph(coords, scale_fraction=0.15)
    elif graph_type == "radius":
        edge_index, edge_dist = build_radius_graph(coords, radius_km=radius_km)
    else:
        edge_index, edge_dist = build_knn_graph(coords, k=knn_k)

    t_true = city_data.pair_trips
    bin_labels = city_data.bin_labels

    assert (t_true > 0).all(), f"Violation: found non-positive counts in Omega_c for {city_name}"

    # -----------------------------------------------------------------------
    # Condition M0: Pure Zero-Shot Inference
    # -----------------------------------------------------------------------
    t_pred_zs = infer_zero_shot(model, city_data, edge_index, edge_dist, device=device)
    m0_metrics = evaluate_all(t_true, t_pred_zs)

    # -----------------------------------------------------------------------
    # Condition M1_oracle: Zero-Shot + Y_D^oracle
    # -----------------------------------------------------------------------
    yd_oracle = extract_yd_oracle(t_true, bin_labels)
    t_pred_oracle = calibrate_by_distance_bins(t_pred_zs, bin_labels, yd_oracle)
    m1_oracle_metrics = evaluate_all(t_true, t_pred_oracle)

    # -----------------------------------------------------------------------
    # Condition M1_real: Zero-Shot + Y_D^real (Meta mobility data)
    # -----------------------------------------------------------------------
    yd_real = extract_yd_real(city_name, meta_prior_dir=meta_prior_dir)
    if yd_real is not None:
        t_pred_real = calibrate_by_distance_bins(t_pred_zs, bin_labels, yd_real)
        m1_real_metrics = evaluate_all(t_true, t_pred_real)
    else:
        m1_real_metrics = None

    # -----------------------------------------------------------------------
    # Condition M_q: Multinomial Trip Sampling across m grid (S=20 seeds)
    # -----------------------------------------------------------------------
    mq_results = {}
    m_finite_values = []
    mean_cpcs = []

    for m in m_grid:
        seed_cpcs = []
        seed_rmses = []
        seed_pearsons = []
        for seed in range(num_trip_seeds):
            yd_m = sample_multinomial_yd(t_true, bin_labels, m=m, seed=seed)
            t_pred_m = calibrate_by_distance_bins(t_pred_zs, bin_labels, yd_m)
            metrics_m = evaluate_all(t_true, t_pred_m)
            seed_cpcs.append(metrics_m["cpc"])
            seed_rmses.append(metrics_m["rmse_log1p"])
            seed_pearsons.append(metrics_m["pearson_r"])

        m_key = "inf" if np.isinf(m) else str(int(m))
        cpc_arr = np.array(seed_cpcs)
        rmse_arr = np.array(seed_rmses)
        pearson_arr = np.array(seed_pearsons)

        mq_results[m_key] = {
            "m": m,
            "cpc_mean": float(np.mean(cpc_arr)),
            "cpc_std": float(np.std(cpc_arr)),
            "cpc_median": float(np.median(cpc_arr)),
            "cpc_p25": float(np.percentile(cpc_arr, 25)),
            "cpc_p75": float(np.percentile(cpc_arr, 75)),
            "rmse_mean": float(np.mean(rmse_arr)),
            "rmse_std": float(np.std(rmse_arr)),
            "pearson_mean": float(np.mean(pearson_arr)),
        }

        if not np.isinf(m):
            m_finite_values.append(float(m))
            mean_cpcs.append(float(np.mean(cpc_arr)))

    # -----------------------------------------------------------------------
    # Primary Delta R calculations
    # -----------------------------------------------------------------------
    delta_r_oracle = m1_oracle_metrics["cpc"] - m0_metrics["cpc"]
    delta_r_real = (m1_real_metrics["cpc"] - m0_metrics["cpc"]) if m1_real_metrics is not None else None
    realization_gap = (m1_oracle_metrics["cpc"] - m1_real_metrics["cpc"]) if m1_real_metrics is not None else None

    # -----------------------------------------------------------------------
    # RQ2: Isotonic Interpolation for m* and q*
    # -----------------------------------------------------------------------
    total_trips = float(torch.sum(t_true).item())

    m_star_oracle, oracle_status = _interpolate_m_star(
        m1_oracle_metrics["cpc"],
        m_finite_values,
        mean_cpcs,
        m1_oracle_metrics["cpc"],
        total_trips,
    )
    q_star_oracle = m_star_oracle / total_trips if total_trips > 0 else 0.0

    if m1_real_metrics is not None:
        m_star_real, real_status = _interpolate_m_star(
            m1_real_metrics["cpc"],
            m_finite_values,
            mean_cpcs,
            m1_oracle_metrics["cpc"],
            total_trips,
        )
        q_star_real = m_star_real / total_trips if total_trips > 0 else 0.0
    else:
        m_star_real = None
        q_star_real = None
        real_status = "no_meta_data"

    return {
        "city": city_name,
        "n_tracts": city_data.n_tracts,
        "n_pairs": city_data.n_pairs,
        "total_trips": total_trips,
        "M0": m0_metrics,
        "M1_oracle": m1_oracle_metrics,
        "M1_real": m1_real_metrics,
        "Mq_curve": mq_results,
        "delta_r_oracle": delta_r_oracle,
        "delta_r_real": delta_r_real,
        "realization_gap": realization_gap,
        "m_star_real": m_star_real,
        "q_star_real": q_star_real,
        "m_star_real_status": real_status,
        "m_star_oracle": m_star_oracle,
        "q_star_oracle": q_star_oracle,
        "yd_oracle": yd_oracle.tolist(),
        "yd_real": yd_real.tolist() if yd_real is not None else None,
    }
