"""
Source-Specific Fixed-Width Distance Binning Module.

Protocol Requirements:
1. Distance cap D_cap^(s) is defined as the 99th percentile of OD distances observed ONLY
   in the 30% training split of the source city:
       D_cap^(s) = P_99(d_ij | (i,j) in Omega_s^train)
2. For each resolution K in {2, 4, 8, 12, 20}, bin width is:
       w_s(K) = D_cap^(s) / K
3. Bins are defined as:
       B_1 = [0, w_s)
       B_2 = [w_s, 2*w_s)
       ...
       B_K = [(K-1)*w_s, inf)
4. Bin boundaries are frozen per source city and reused for all target cities,
   models, seeds, and both baseline and calibrated distributions.
"""

from typing import List, Tuple, Dict
import numpy as np
import pandas as pd


class SourceBinDomainError(ValueError):
    """Raised when source distance cap or bin configuration is domain-invalid."""
    pass


class CalibrationDomainError(ValueError):
    """Raised when predictions or target coverage do not satisfy calibration domain invariants."""
    pass


def compute_source_distance_cap(train_distances_km: np.ndarray, quantile: float = 0.99) -> float:
    """Computes D_cap as P_99 of distances observed ONLY in main 30%-reference training split."""
    if len(train_distances_km) == 0:
        raise SourceBinDomainError("Cannot compute distance cap on empty training distances array.")
    d_cap = float(np.quantile(train_distances_km.astype(np.float64), quantile, method="linear"))
    if not np.isfinite(d_cap) or d_cap <= 0.0:
        raise SourceBinDomainError(f"Computed D_cap is non-finite or non-positive: {d_cap}")
    return d_cap


def build_source_bin_edges(d_cap: float, K: int) -> np.ndarray:
    """
    Builds bin edges: [0, w, 2w, ..., (K-1)w, inf].
    Length of returned array is K + 1.
    """
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    w = d_cap / K
    edges = [i * w for i in range(K)]
    edges.append(np.inf)
    return np.array(edges, dtype=np.float64)


def assign_to_source_bins(distances_km: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """
    Assigns distances to bins [0, 1, ..., K-1].
    Bin b is [edges[b], edges[b+1]).
    The last bin K-1 absorbs all distances >= edges[K-1] up to inf.
    """
    K = len(bin_edges) - 1
    # np.digitize: bins[i-1] <= x < bins[i] when right=False
    # For edges [0, w, 2w, ..., inf]:
    # x in [0, w) -> index 1 -> subtract 1 -> 0
    # x >= (K-1)*w -> index K -> subtract 1 -> K-1
    bin_ids = np.digitize(distances_km, bin_edges, right=False) - 1
    return np.clip(bin_ids, 0, K - 1)


def generate_source_bins_manifest(
    source_city: str,
    train_split_seed: int,
    train_distances_km: np.ndarray,
    k_list: List[int] = (2, 4, 8, 12, 20),
) -> List[Dict]:
    """Generates tabular records for source_distance_bins.csv."""
    d_cap = compute_source_distance_cap(train_distances_km)
    records = []
    
    for K in k_list:
        edges = build_source_bin_edges(d_cap, K)
        for b in range(K):
            lo = float(edges[b])
            hi = float(edges[b + 1]) if not np.isinf(edges[b + 1]) else "inf"
            records.append({
                "source_city": source_city,
                "train_split_seed": train_split_seed,
                "K": K,
                "D_cap_p99": round(d_cap, 4),
                "bin_id": b + 1,  # 1-indexed for display
                "lower_km": round(lo, 4),
                "upper_km": hi if hi == "inf" else round(hi, 4)
            })
    return records


def compute_pure_calibration_ratios(
    p_b: np.ndarray,
    q_b: np.ndarray
) -> np.ndarray:
    r"""
    Computes pure piecewise DBD calibration ratios without epsilon smoothing:
        r_b = p_b / q_b  if q_b > 0 and \sum_{b \in B^+} p_b == 1.0
        
    Under neural models (MLP, GNN), Softplus strictly guarantees \hat{T}_{ij} > 0,
    so q_b > 0 for all non-empty distance bins.
    
    Under physics-based Two-Parameter Gravity, tracts with zero population (P_i = 0 or P_j = 0)
    predict \hat{T}_{ij} = 0. When an entire distance bin consists of zero-population tracts,
    q_b = 0 while p_b > 0.
    
    Calibration Policy for q_b = 0, p_b > 0:
    - Bins with q_b = 0 have zero baseline prediction (\hat{T}^{(0)} = 0), so scaling them
      by any finite multiplier produces zero flow (0 * r_b = 0).
    - To maintain mathematical rigor and Exact Volume Preservation (\sum \hat{T}^{(1)} = \sum \hat{T}^{(0)}),
      the target distribution is conditioned on the positive baseline support:
          B^+ = \{ b \in \{1,\ldots,K\} \mid q_b > 0 \}
          P_{\text{covered}} = \sum_{b \in B^+} p_b
      For b \in B^+:
          r_b = \frac{p_b / P_{\text{covered}}}{q_b}
      For b \notin B^+:
          r_b = 1.0 (identity multiplier for unpredicted bins, leaving 0 * 1.0 = 0)
    - If all q_b > 0 on positive p_b (P_{\text{covered}} == 1.0), this collapses bit-exactly to r_b = p_b / q_b.
    """
    if len(p_b) != len(q_b):
        raise ValueError(f"Length mismatch between p_b ({len(p_b)}) and q_b ({len(q_b)})")

    r_b = np.ones_like(q_b, dtype=np.float64)
    pos_mask = (q_b > 0.0)

    if not np.any(pos_mask):
        raise CalibrationDomainError(
            "Baseline predictions have zero mass across all bins (all q_b == 0.0). Cannot calibrate."
        )

    p_covered = float(np.sum(p_b[pos_mask]))
    if not np.isfinite(p_covered) or p_covered <= 0.0:
        raise CalibrationDomainError(
            f"Baseline positive support B^+ covers zero target mass (P_covered = {p_covered:.6e} <= 0). "
            "Cannot perform support-conditioned calibration."
        )

    r_b[pos_mask] = (p_b[pos_mask] / p_covered) / q_b[pos_mask]

    # For b where q_b == 0, r_b remains 1.0 (identity)
    return r_b


def build_target_dbd(
    true_flow: np.ndarray,
    distance_km: np.ndarray,
    bin_edges: np.ndarray,
) -> np.ndarray:
    r"""
    Constructs the normalized Target Distance-Binned Distribution (DBD) vector p on positive support:
        p_b = \sum_{(i,j) \in B_b^{(s,K)}} T_{ij} / \sum_{(i,j) \in \Omega_t^+} T_{ij}

    Strict Protocol Rules:
    - This is the ONLY function that accesses true target OD flows for DBD preparation.
    - Returns strictly the normalized 1D distribution vector p (summing to 1.0).
    - Absolute target volume \sum T_{ij} is NOT retained or returned.
    """
    if len(true_flow) != len(distance_km):
        raise ValueError(f"Length mismatch between true_flow ({len(true_flow)}) and distance_km ({len(distance_km)})")
    
    flow_64 = np.asarray(true_flow, dtype=np.float64)
    total_flow = float(np.sum(flow_64))
    if total_flow <= 0:
        raise ValueError("Cannot build target DBD on zero total flow.")

    K = len(bin_edges) - 1
    bin_ids = assign_to_source_bins(distance_km, bin_edges)

    p = np.zeros(K, dtype=np.float64)
    for b in range(K):
        mask = (bin_ids == b)
        if mask.any():
            p[b] = float(np.sum(flow_64[mask])) / total_flow

    # Verify normalization
    p_sum = float(np.sum(p))
    if abs(p_sum - 1.0) >= 1e-10:
        raise AssertionError(f"Target DBD normalization failed: sum(p) = {p_sum:.12f} != 1.0")
    return p


def apply_pure_dbd_calibration(
    t_pred: np.ndarray,
    bin_assignments: np.ndarray,
    r_b: np.ndarray,
    tol: float = 1e-10
) -> np.ndarray:
    r"""
    Applies pure piecewise DBD calibration to predictions:
        \hat{T}^{(1)}_{ij} = r_{b(ij)} * \hat{T}^{(0)}_{ij}

    Validates exact volume preservation:
        |\sum \hat{T}^{(1)} - \sum \hat{T}^{(0)}| < tol (relative or absolute machine tolerance)
    """
    orig_dtype = t_pred.dtype
    t_pred_64 = t_pred.astype(np.float64, copy=False)
    t_cal_64 = t_pred_64 * r_b[bin_assignments]

    sum_orig = float(np.sum(t_pred_64))
    sum_cal = float(np.sum(t_cal_64))

    if not np.isfinite(sum_orig) or not np.isfinite(sum_cal):
        raise CalibrationDomainError(
            f"Non-finite flow volume encountered: sum_orig={sum_orig}, sum_cal={sum_cal}"
        )

    diff = abs(sum_cal - sum_orig)
    eff_tol = max(tol, 1e-9 * max(abs(sum_orig), 1.0))
    if diff >= eff_tol:
        raise AssertionError(
            f"Exact volume preservation failed: |sum_after - sum_before| = {diff:.6e} >= {eff_tol:.6e}. "
            "Calibration must be exact volume-preserving up to floating-point tolerance."
        )
    return t_cal_64


def calibrate_dbd(
    pred_flow: np.ndarray,
    distance_km: np.ndarray,
    bin_edges: np.ndarray,
    target_dbd_p: np.ndarray,
    tol: float = 1e-10,
) -> np.ndarray:
    r"""
    Post-hoc DBD Calibrator Interface.
    Strictly isolated from true target pair flows and target total flow volume.

    Input Contract:
    - pred_flow: Baseline OD flow predictions on fixed positive support (\hat{T}^{(0)}).
    - distance_km: Physical distance array for the pairs (raw km).
    - bin_edges: Source-defined distance bin edges B^{(s,K)}.
    - target_dbd_p: Normalized target DBD vector p (\sum_b p_b = 1.0).

    Strict Isolation Invariants:
    1. ZERO access to true pair-level flows T_{ij}.
    2. ZERO access to target total flow \sum T_{ij} or R_vol.
    3. Output \hat{T}^{(1)} satisfies exact volume preservation:
           |\sum \hat{T}^{(1)} - \sum \hat{T}^{(0)}| < tol
    """
    if len(pred_flow) != len(distance_km):
        raise ValueError(f"Length mismatch: pred_flow ({len(pred_flow)}) vs distance_km ({len(distance_km)})")

    K = len(bin_edges) - 1
    if len(target_dbd_p) != K:
        raise ValueError(f"target_dbd_p length {len(target_dbd_p)} != K={K}")

    # Verify target DBD normalization
    if abs(float(np.sum(target_dbd_p)) - 1.0) >= 1e-6:
        raise ValueError(f"Input target_dbd_p must sum to 1.0, got sum = {np.sum(target_dbd_p):.6f}")

    pred_flow_64 = pred_flow.astype(np.float64, copy=False)
    total_pred = float(np.sum(pred_flow_64))
    if not np.isfinite(total_pred) or total_pred <= 0.0:
        raise CalibrationDomainError(
            f"Baseline total predicted flow is non-positive or non-finite: {total_pred}. "
            "Cannot perform post-hoc DBD calibration on degenerate baseline."
        )

    # Compute baseline predicted DBD vector q on source bins
    bin_ids = assign_to_source_bins(distance_km, bin_edges)
    q = np.zeros(K, dtype=np.float64)
    for b in range(K):
        mask = (bin_ids == b)
        if mask.any():
            q[b] = float(np.sum(pred_flow_64[mask])) / total_pred

    # Compute piecewise ratios r_b = p_b / q_b
    r_b = compute_pure_calibration_ratios(target_dbd_p, q)

    # Apply calibration with volume preservation check
    pred_after = apply_pure_dbd_calibration(pred_flow_64, bin_ids, r_b, tol=tol)
    return pred_after
