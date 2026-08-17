"""
Comprehensive Evaluation Suite on Interzonal Domain Omega_c^+ and Full Support Omega_c.

Primary Metric:
    Interzonal CPC (CPC_inter) on Omega_c^+ = {(i,j) in Omega_c : i != j, D_ij > 0}:
        Evaluates the displacement flow distribution of moving commuters.

Secondary Metrics:
    1. Scale-Normalized Interzonal CPC (CPC_inter_norm = 1 - TVD):
        Evaluates pure structural flow geometry independent of total flow scale.
    2. Full CPC (CPC_full) on all Omega_c (with intact intrazonal diagonal).
    3. Full Scale-Normalized CPC (CPC_full_norm).
    4. RMSE-log1p on Omega_c^+ and full Omega_c.
    5. Pearson correlation r on Omega_c^+ and full Omega_c.
"""

import math
import numpy as np
import torch


def compute_cpc_pair(t_true: np.ndarray, t_pred: np.ndarray) -> float:
    """Computes standard CPC between two non-negative 1D arrays."""
    sum_min = np.sum(np.minimum(t_true, t_pred))
    sum_total = np.sum(t_true) + np.sum(t_pred)
    if sum_total <= 0:
        return 0.0
    return float(2.0 * sum_min / sum_total)


def compute_cpc_norm_pair(t_true: np.ndarray, t_pred: np.ndarray) -> float:
    """Computes Scale-Normalized CPC (1 - Total Variation Distance)."""
    sum_t = np.sum(t_true)
    sum_p = np.sum(t_pred)
    if sum_t <= 0 or sum_p <= 0:
        return 0.0
    p_t = t_true / sum_t
    p_p = t_pred / sum_p
    return float(np.sum(np.minimum(p_t, p_p)))


def compute_rmse_log1p_pair(t_true: np.ndarray, t_pred: np.ndarray) -> float:
    """Computes RMSE on log1p scale."""
    log_t = np.log1p(np.clip(t_true, 0.0, None))
    log_p = np.log1p(np.clip(t_pred, 0.0, None))
    return float(np.sqrt(np.mean((log_t - log_p) ** 2)))


def compute_pearson_pair(t_true: np.ndarray, t_pred: np.ndarray) -> float:
    """Computes Pearson linear correlation."""
    std_t = np.std(t_true)
    std_p = np.std(t_pred)
    if std_t == 0 or std_p == 0:
        return 0.0
    cov = np.mean((t_true - np.mean(t_true)) * (t_pred - np.mean(t_pred)))
    return float(cov / (std_t * std_p))


def evaluate_moving_and_full(
    t_true: torch.Tensor,
    t_pred: torch.Tensor,
    pair_o_idx: torch.Tensor,
    pair_d_idx: torch.Tensor,
    bin_labels: torch.Tensor,
    pair_distance: torch.Tensor | None = None,
) -> dict[str, float]:
    """
    Computes all locked metrics partitioned by Interzonal Omega_c^+ and Full Support Omega_c.
    """
    t_t = t_true.detach().cpu().numpy().astype(np.float64)
    t_p = t_pred.detach().cpu().numpy().astype(np.float64)
    o_np = pair_o_idx.detach().cpu().numpy()
    d_np = pair_d_idx.detach().cpu().numpy()
    b_np = bin_labels.detach().cpu().numpy()

    if pair_distance is not None:
        p_dist = pair_distance.detach().cpu().numpy()
        dist_km = np.expm1(p_dist) if np.max(p_dist) < 20.0 else p_dist
        inter_mask = (o_np != d_np) & (dist_km > 0.0)
    else:
        inter_mask = (o_np != d_np) & (b_np > 0)

    # 1. Interzonal Domain Omega_c^+ (Primary)
    t_t_inter = t_t[inter_mask]
    t_p_inter = t_p[inter_mask]

    cpc_inter = compute_cpc_pair(t_t_inter, t_p_inter)
    cpc_inter_norm = compute_cpc_norm_pair(t_t_inter, t_p_inter)
    rmse_inter = compute_rmse_log1p_pair(t_t_inter, t_p_inter)
    pearson_inter = compute_pearson_pair(t_t_inter, t_p_inter)

    # 2. Full Matrix Domain Omega_c (Secondary)
    cpc_full = compute_cpc_pair(t_t, t_p)
    cpc_full_norm = compute_cpc_norm_pair(t_t, t_p)
    rmse_full = compute_rmse_log1p_pair(t_t, t_p)
    pearson_full = compute_pearson_pair(t_t, t_p)

    return {
        # Primary interzonal metrics
        "cpc": cpc_inter,                     # primary shorthand
        "cpc_inter": cpc_inter,
        "cpc_inter_norm": cpc_inter_norm,
        "rmse_inter": rmse_inter,
        "pearson_inter": pearson_inter,
        # Secondary full-matrix metrics
        "cpc_full": cpc_full,
        "cpc_full_norm": cpc_full_norm,
        "rmse_full": rmse_full,
        "pearson_full": pearson_full,
    }


def evaluate_all(t_true: torch.Tensor, t_pred: torch.Tensor) -> dict[str, float]:
    """Compatibility helper for raw full-pair evaluation."""
    t_t = t_true.detach().cpu().numpy().astype(np.float64)
    t_p = t_pred.detach().cpu().numpy().astype(np.float64)
    return {
        "cpc": compute_cpc_pair(t_t, t_p),
        "cpc_norm": compute_cpc_norm_pair(t_t, t_p),
        "rmse_log1p": compute_rmse_log1p_pair(t_t, t_p),
        "pearson_r": compute_pearson_pair(t_t, t_p),
    }
