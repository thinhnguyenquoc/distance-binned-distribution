"""
Evaluation Metrics on Candidate Support Omega_c.

Primary Metric:
    Common Part of Commuters (CPC):
        CPC(T, \hat{T}) = 2 * sum_{ij} min(T_ij, \hat{T}_ij) / (sum T_ij + sum \hat{T}_ij)
        Range: [0, 1], higher is better.

Secondary Metrics:
    RMSE-log1p:
        sqrt( 1/|Omega_c| * sum_{ij} [log(1 + T_ij) - log(1 + \hat{T}_ij)]^2 )
        Well-defined for all counts >= 0.

    Pearson correlation r:
        Linear correlation between T_ij and \hat{T}_ij.
"""

import math
import numpy as np
import torch


def compute_cpc(t_true: torch.Tensor, t_pred: torch.Tensor) -> float:
    """Computes Common Part of Commuters (CPC)."""
    t_t = t_true.detach().cpu().numpy().astype(np.float64)
    t_p = t_pred.detach().cpu().numpy().astype(np.float64)

    sum_min = np.sum(np.minimum(t_t, t_p))
    sum_total = np.sum(t_t) + np.sum(t_p)

    if sum_total == 0:
        return 0.0
    return float(2.0 * sum_min / sum_total)


def compute_rmse_log1p(t_true: torch.Tensor, t_pred: torch.Tensor) -> float:
    """Computes RMSE on log1p scale."""
    t_t = t_true.detach().cpu().numpy().astype(np.float64)
    t_p = t_pred.detach().cpu().numpy().astype(np.float64)

    log_t = np.log1p(np.clip(t_t, 0.0, None))
    log_p = np.log1p(np.clip(t_p, 0.0, None))

    mse = np.mean((log_t - log_p) ** 2)
    return float(np.sqrt(mse))


def compute_pearson_r(t_true: torch.Tensor, t_pred: torch.Tensor) -> float:
    """Computes Pearson correlation coefficient."""
    t_t = t_true.detach().cpu().numpy().astype(np.float64)
    t_p = t_pred.detach().cpu().numpy().astype(np.float64)

    std_t = np.std(t_t)
    std_p = np.std(t_p)

    if std_t == 0 or std_p == 0:
        return 0.0

    cov = np.mean((t_t - np.mean(t_t)) * (t_p - np.mean(t_p)))
    return float(cov / (std_t * std_p))


def evaluate_all(t_true: torch.Tensor, t_pred: torch.Tensor) -> dict[str, float]:
    """Computes all locked evaluation metrics on Omega_c."""
    return {
        "cpc": compute_cpc(t_true, t_pred),
        "rmse_log1p": compute_rmse_log1p(t_true, t_pred),
        "pearson_r": compute_pearson_r(t_true, t_pred),
    }


if __name__ == "__main__":
    t_true = torch.tensor([10.0, 50.0, 100.0, 500.0])
    t_pred = torch.tensor([12.0, 40.0, 110.0, 480.0])
    metrics = evaluate_all(t_true, t_pred)
    print("Test metrics:", metrics)
