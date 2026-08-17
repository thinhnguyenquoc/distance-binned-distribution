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

    # Always append total_trips with oracle_cpc as the finite population ceiling
    all_m.append(float(total_trips))
    all_cpc.append(float(oracle_cpc))

    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    cpc_curve_monotonic = iso.fit_transform(all_m, all_cpc)

    if target_cpc <= cpc_curve_monotonic[0]:
        m_star = float(all_m[0])
        status = "below_min_grid"
    elif target_cpc >= cpc_curve_monotonic[-1]:
        m_star = float(all_m[-1])
        status = "at_oracle_ceiling"
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
        status = "interpolated" if m_star < total_trips else "at_oracle_ceiling"

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
    pair_o = city_data.pair_o_idx
    pair_d = city_data.pair_d_idx

    inter_mask = (pair_o != pair_d) & (bin_labels > 0)
    n_inter_pairs = int(inter_mask.sum().item())
    total_inter_trips = float(t_true[inter_mask].sum().item())
    total_trips = float(t_true.sum().item())

    # -----------------------------------------------------------------------
    # Condition M0: Pure Zero-Shot Inference
    # -----------------------------------------------------------------------
    t_pred_zs = infer_zero_shot(model, city_data, edge_index, edge_dist, device=device)
    m0_metrics = evaluate_moving_and_full(t_true, t_pred_zs, pair_o, pair_d, bin_labels)

    # -----------------------------------------------------------------------
    # Moving-Bin Target Distributions (Oracle & Real)
    # -----------------------------------------------------------------------
    yd_moving_oracle = extract_yd_moving_oracle(t_true, bin_labels, pair_o, pair_d)
    yd_moving_real = extract_yd_moving_real(city_name, meta_prior_dir=meta_prior_dir)

    # Distributional Overlap on Moving Bins
    if yd_moving_real is not None:
        dist_overlap = compute_distributional_overlap(yd_moving_oracle, yd_moving_real)
    else:
        dist_overlap = None

    # -----------------------------------------------------------------------
    # Condition M1^{oracle, +}: Oracle Moving-Bin Reference (q=1.0)
    # -----------------------------------------------------------------------
    t_pred_oracle_plus = calibrate_moving_bins(
        t_pred_zs, bin_labels, pair_o, pair_d, yd_moving_oracle, q=1.0
    )
    m1_oracle_plus_metrics = evaluate_moving_and_full(t_true, t_pred_oracle_plus, pair_o, pair_d, bin_labels)

    # -----------------------------------------------------------------------
    # Condition M1^{real, +}: Primary Meta Moving-Bin Calibration (q=1.0)
    # -----------------------------------------------------------------------
    if yd_moving_real is not None:
        t_pred_real_plus = calibrate_moving_bins(
            t_pred_zs, bin_labels, pair_o, pair_d, yd_moving_real, q=1.0
        )
        m1_real_plus_metrics = evaluate_moving_and_full(t_true, t_pred_real_plus, pair_o, pair_d, bin_labels)
    else:
        m1_real_plus_metrics = None

    # -----------------------------------------------------------------------
    # Condition M1^{real, 4bin}: Ablation Retaining Bin 0 Semantic Mismatch
    # -----------------------------------------------------------------------
    yd_4bin_real = extract_yd_4bin_real(city_name, meta_prior_dir=meta_prior_dir)
    if yd_4bin_real is not None:
        t_pred_4bin_ablation = calibrate_4bin_legacy_ablation(t_pred_zs, bin_labels, yd_4bin_real)
        m1_4bin_ablation_metrics = evaluate_moving_and_full(t_true, t_pred_4bin_ablation, pair_o, pair_d, bin_labels)
    else:
        m1_4bin_ablation_metrics = None

    # -----------------------------------------------------------------------
    # Condition M_q^{real, +}: Soft Calibration Curve over q in [0, 1]
    # -----------------------------------------------------------------------
    q_curve = {}
    if yd_moving_real is not None:
        for q_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
            t_pred_q = calibrate_moving_bins(t_pred_zs, bin_labels, pair_o, pair_d, yd_moving_real, q=q_val)
            q_metrics = evaluate_moving_and_full(t_true, t_pred_q, pair_o, pair_d, bin_labels)
            q_curve[f"q_{q_val:.2f}"] = {
                "q": q_val,
                "cpc_inter": q_metrics["cpc_inter"],
                "cpc_full": q_metrics["cpc_full"],
            }

    # -----------------------------------------------------------------------
    # Condition M_m^+: Multinomial Sampling on Interzonal Pairs Omega_c^+
    # -----------------------------------------------------------------------
    mq_results = {}
    m_finite_values = []
    mean_cpcs_inter = []

    # Filter GT trips for interzonal sampling
    inter_trips = t_true[inter_mask]
    inter_bins = bin_labels[inter_mask]

    for m in m_grid:
        seed_cpcs = []
        for seed in range(num_trip_seeds):
            # Sample m interzonal trips
            yd_m_4 = sample_multinomial_yd(inter_trips, inter_bins, m=m, seed=seed)
            # moving bins {1, 2, 3}
            yd_m_moving = yd_m_4[1:]
            total_m_moving = np.sum(yd_m_moving)
            if total_m_moving > 0:
                yd_m_moving = yd_m_moving / total_m_moving
            else:
                yd_m_moving = np.array([0.5, 0.4, 0.1])

            t_pred_m = calibrate_moving_bins(t_pred_zs, bin_labels, pair_o, pair_d, yd_m_moving, q=1.0)
            metrics_m = evaluate_moving_and_full(t_true, t_pred_m, pair_o, pair_d, bin_labels)
            seed_cpcs.append(metrics_m["cpc_inter"])

        m_key = "inf" if np.isinf(m) else str(int(m))
        cpc_arr = np.array(seed_cpcs)

        mq_results[m_key] = {
            "m": m,
            "cpc_inter_mean": float(np.mean(cpc_arr)),
            "cpc_inter_std": float(np.std(cpc_arr)),
            "cpc_inter_median": float(np.median(cpc_arr)),
            "cpc_inter_p25": float(np.percentile(cpc_arr, 25)),
            "cpc_inter_p75": float(np.percentile(cpc_arr, 75)),
        }

        if not np.isinf(m):
            m_finite_values.append(float(m))
            mean_cpcs_inter.append(float(np.mean(cpc_arr)))

    # -----------------------------------------------------------------------
    # Primary Delta R Calculations on Interzonal Domain Omega_c^+
    # -----------------------------------------------------------------------
    delta_r_oracle_plus = m1_oracle_plus_metrics["cpc_inter"] - m0_metrics["cpc_inter"]
    delta_r_real_plus = (m1_real_plus_metrics["cpc_inter"] - m0_metrics["cpc_inter"]) if m1_real_plus_metrics else None
    realization_gap_plus = (m1_oracle_plus_metrics["cpc_inter"] - m1_real_plus_metrics["cpc_inter"]) if m1_real_plus_metrics else None

    # Ablation Delta R (4-bin)
    delta_r_4bin_ablation = (m1_4bin_ablation_metrics["cpc_inter"] - m0_metrics["cpc_inter"]) if m1_4bin_ablation_metrics else None

    # -----------------------------------------------------------------------
    # RQ2: Isotonic Interpolation for m* and q* on Interzonal Trips
    # -----------------------------------------------------------------------
    m_star_oracle, oracle_status = _interpolate_m_star(
        m1_oracle_plus_metrics["cpc_inter"],
        m_finite_values,
        mean_cpcs_inter,
        m1_oracle_plus_metrics["cpc_inter"],
        total_inter_trips,
    )
    q_star_oracle = m_star_oracle / total_inter_trips if total_inter_trips > 0 else 0.0

    if m1_real_plus_metrics is not None:
        m_star_real, real_status = _interpolate_m_star(
            m1_real_plus_metrics["cpc_inter"],
            m_finite_values,
            mean_cpcs_inter,
            m1_oracle_plus_metrics["cpc_inter"],
            total_inter_trips,
        )
        q_star_real = m_star_real / total_inter_trips if total_inter_trips > 0 else 0.0
    else:
        m_star_real = None
        q_star_real = None
        real_status = "no_meta_data"

    return {
        "city": city_name,
        "n_tracts": city_data.n_tracts,
        "n_pairs": city_data.n_pairs,
        "n_inter_pairs": n_inter_pairs,
        "total_trips": total_trips,
        "total_inter_trips": total_inter_trips,
        "distributional_overlap": dist_overlap,
        "M0": m0_metrics,
        "M1_real_plus": m1_real_plus_metrics,
        "M1_oracle_plus": m1_oracle_plus_metrics,
        "M1_4bin_ablation": m1_4bin_ablation_metrics,
        "Mq_soft_curve": q_curve,
        "Mm_sampling_curve": mq_results,
        "delta_r_oracle_plus": delta_r_oracle_plus,
        "delta_r_real_plus": delta_r_real_plus,
        "realization_gap_plus": realization_gap_plus,
        "delta_r_4bin_ablation": delta_r_4bin_ablation,
        "m_star_real": m_star_real,
        "q_star_real": q_star_real,
        "m_star_real_status": real_status,
        "m_star_oracle": m_star_oracle,
        "q_star_oracle": q_star_oracle,
        "yd_moving_oracle": yd_moving_oracle.tolist(),
        "yd_moving_real": yd_moving_real.tolist() if yd_moving_real is not None else None,
    }
