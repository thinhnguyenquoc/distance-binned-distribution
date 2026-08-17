"""
Experiment Runner for All 4 Conditions per Target City:
    M_0:       Zero-shot baseline
    M_1^oracle: Zero-shot + Y_D^oracle (from GT over Omega_c)
    M_1^real:   Zero-shot + Y_D^real (from Meta mobility data if available)
    M_q:        Zero-shot + \tilde{Y}_D^{(m)} (from Multinomial trip sampling across m grid)

Computes:
    - Delta R_c = R_c^{YD, real} - R_c^{ZS} (or oracle if real not available)
    - Reference curve R_c(m)
    - q*_c interpolation
"""

import numpy as np
import torch
from typing import Dict, Any, List

from src.data.dataset import CityData, load_city
from src.data.urban_graph import build_knn_graph
from src.data.yd_extractor import extract_yd_oracle, extract_yd_real
from src.data.trip_sampler import sample_multinomial_yd, M_GRID
from src.calibration.bin_calibration import calibrate_by_distance_bins
from src.training.evaluate import evaluate_all
from src.training.train import infer_zero_shot, ZeroShotODModel


def run_target_city_experiments(
    model: ZeroShotODModel,
    city_name: str,
    scaler: object,
    data_root: str = "data",
    meta_prior_dir: str = "meta_prior",
    knn_k: int = 10,
    num_trip_seeds: int = 10,
    m_grid: List[int | float] = M_GRID,
    device_str: str = "cpu",
) -> Dict[str, Any]:
    """
    Runs all 4 experimental conditions on a single held-out target city.

    Returns:
        dictionary containing metrics for M0, M1_oracle, M1_real, Mq curve, and delta R.
    """
    device = torch.device(device_str)
    city_data = load_city(city_name, data_root=data_root, feature_scaler=scaler)
    edge_index, edge_dist = build_knn_graph(city_data.lon_lat.numpy(), k=knn_k)

    t_true = city_data.pair_trips
    bin_labels = city_data.bin_labels

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
    # Condition M_q: Multinomial Trip Sampling across m grid
    # -----------------------------------------------------------------------
    mq_results = {}
    for m in m_grid:
        seed_cpcs = []
        seed_rmses = []
        for seed in range(num_trip_seeds):
            yd_m = sample_multinomial_yd(t_true, bin_labels, m=m, seed=seed)
            t_pred_m = calibrate_by_distance_bins(t_pred_zs, bin_labels, yd_m)
            metrics_m = evaluate_all(t_true, t_pred_m)
            seed_cpcs.append(metrics_m["cpc"])
            seed_rmses.append(metrics_m["rmse_log1p"])

        m_key = "inf" if np.isinf(m) else str(int(m))
        mq_results[m_key] = {
            "m": m,
            "cpc_mean": float(np.mean(seed_cpcs)),
            "cpc_std": float(np.std(seed_cpcs)),
            "rmse_mean": float(np.mean(seed_rmses)),
            "rmse_std": float(np.std(seed_rmses)),
        }

    # -----------------------------------------------------------------------
    # Primary Delta R calculations
    # -----------------------------------------------------------------------
    delta_r_oracle = m1_oracle_metrics["cpc"] - m0_metrics["cpc"]
    delta_r_real = (m1_real_metrics["cpc"] - m0_metrics["cpc"]) if m1_real_metrics is not None else None
    realization_gap = (m1_oracle_metrics["cpc"] - m1_real_metrics["cpc"]) if m1_real_metrics is not None else None

    # -----------------------------------------------------------------------
    # Interpolation for m* and q*
    # -----------------------------------------------------------------------
    total_trips = float(torch.sum(t_true).item())
    target_cpc = m1_real_metrics["cpc"] if m1_real_metrics is not None else m1_oracle_metrics["cpc"]

    # Monotonic 1D interpolation over m grid
    m_finite = [m for m in m_grid if not np.isinf(m)]
    cpc_curve = [mq_results[str(int(m))]["cpc_mean"] for m in m_finite]

    # Find m* where curve crosses target_cpc
    if target_cpc <= cpc_curve[0]:
        m_star = float(m_finite[0])
    elif target_cpc >= cpc_curve[-1]:
        m_star = float(m_finite[-1])
    else:
        m_star = float(np.interp(target_cpc, cpc_curve, m_finite))

    q_star = m_star / total_trips if total_trips > 0 else 0.0

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
        "m_star": m_star,
        "q_star": q_star,
        "yd_oracle": yd_oracle.tolist(),
        "yd_real": yd_real.tolist() if yd_real is not None else None,
    }
