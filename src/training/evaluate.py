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


def compute_spearman_pair(t_true: np.ndarray, t_pred: np.ndarray) -> float:
    """Computes Spearman rank correlation of pairwise flows."""
    if len(t_true) < 2 or np.std(t_true) == 0 or np.std(t_pred) == 0:
        return 0.0
    from scipy import stats
    rho, _ = stats.spearmanr(t_true, t_pred)
    return float(rho) if not np.isnan(rho) else 0.0


def compute_rmse_pair(t_true: np.ndarray, t_pred: np.ndarray) -> float:
    """Computes standard RMSE."""
    return float(np.sqrt(np.mean((t_true - t_pred) ** 2)))

def compute_nrmse_pair(t_true: np.ndarray, t_pred: np.ndarray) -> float:
    """Computes Normalized RMSE (RMSE / mean(true))."""
    mean_t = np.mean(t_true)
    if mean_t <= 0:
        return 0.0
    rmse = compute_rmse_pair(t_true, t_pred)
    return float(rmse / mean_t)

def compute_mae_pair(t_true: np.ndarray, t_pred: np.ndarray) -> float:
    """Computes Mean Absolute Error."""
    return float(np.mean(np.abs(t_true - t_pred)))

def compute_inflow_outflow_cpc(t_true: np.ndarray, t_pred: np.ndarray, o_idx: np.ndarray, d_idx: np.ndarray, n_nodes: int) -> tuple[float, float]:
    """Computes CPC for tract-level inflows and outflows on observed support."""
    outflow_t = np.zeros(n_nodes, dtype=np.float64)
    outflow_p = np.zeros(n_nodes, dtype=np.float64)
    inflow_t = np.zeros(n_nodes, dtype=np.float64)
    inflow_p = np.zeros(n_nodes, dtype=np.float64)
    
    np.add.at(outflow_t, o_idx, t_true)
    np.add.at(outflow_p, o_idx, t_pred)
    np.add.at(inflow_t, d_idx, t_true)
    np.add.at(inflow_p, d_idx, t_pred)
    
    cpc_out = compute_cpc_pair(outflow_t, outflow_p)
    cpc_in = compute_cpc_pair(inflow_t, inflow_p)
    return cpc_in, cpc_out

def evaluate_moving_and_full(
    t_true: torch.Tensor,
    t_pred: torch.Tensor,
    pair_o_idx: torch.Tensor,
    pair_d_idx: torch.Tensor,
    bin_labels: torch.Tensor,
    pair_distance: torch.Tensor | None = None,
) -> dict[str, float]:
    """
    Computes all locked metrics partitioned by Interzonal Omega_c^+ as per partial_od.md.
    No full-matrix CPC or missing pair performance is reported.
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

    # All evaluations only on observed pairs!
    # Primary: Interzonal Domain Omega_c^+
    t_t_inter = t_t[inter_mask]
    t_p_inter = t_p[inter_mask]

    cpc_inter = compute_cpc_pair(t_t_inter, t_p_inter)
    rmse_log1p_inter = compute_rmse_log1p_pair(t_t_inter, t_p_inter)
    rmse_inter = compute_rmse_pair(t_t_inter, t_p_inter)
    nrmse_inter = compute_nrmse_pair(t_t_inter, t_p_inter)
    mae_inter = compute_mae_pair(t_t_inter, t_p_inter)
    spearman_inter = compute_spearman_pair(t_t_inter, t_p_inter)
    
    total_flow_true = np.sum(t_t_inter)
    total_flow_pred = np.sum(t_p_inter)
    rel_error = float(abs(total_flow_pred - total_flow_true) / max(total_flow_true, 1e-9))
    
    # Inflow/Outflow CPC on observed support
    max_node = max(np.max(o_np), np.max(d_np)) + 1 if len(o_np) > 0 else 0
    cpc_inflow, cpc_outflow = compute_inflow_outflow_cpc(t_t_inter, t_p_inter, o_np[inter_mask], d_np[inter_mask], max_node)
    
    result = {
        "cpc": cpc_inter,                     # primary shorthand
        "cpc_inter": cpc_inter,
        "rmse_log1p_inter": rmse_log1p_inter,
        "rmse_inter": rmse_inter,
        "nrmse_inter": nrmse_inter,
        "mae_inter": mae_inter,
        "spearman_inter": spearman_inter,
        "rel_error_total": rel_error,
        "cpc_inflow": cpc_inflow,
        "cpc_outflow": cpc_outflow,
    }
    return result

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
