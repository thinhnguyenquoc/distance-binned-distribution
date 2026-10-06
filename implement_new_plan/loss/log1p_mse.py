"""
Log1p-MSE Loss Module for OD Flow Prediction Models.

Mathematical formulation:
    L = 1 / |Train_f(s)| * sum_{(i,j) in Train_f(s)} [log(1 + T_ij) - log(1 + \hat{T}_ij)]^2

Strict Invariants:
1. Replaces ZTNB loss across all three baseline model families (TwoParameterGravity, PairwiseMLP, UrbanGNN).
2. Input flows T and \hat{T} are strictly non-negative on positive OD support.
3. Operates stably in float32.
"""

import torch
import torch.nn as nn


def log1p_mse_loss(
    pred_flow: torch.Tensor,
    true_flow: torch.Tensor,
) -> torch.Tensor:
    r"""
    Computes Log1p-MSE Loss:
        L = Mean( (log(1 + pred_flow) - log(1 + true_flow))^2 )
    
    Args:
        pred_flow: Predicted flow intensity \hat{T}_{ij} (torch.Tensor, shape (N,), >= 0).
        true_flow: Ground truth flow T_{ij} (torch.Tensor, shape (N,), >= 0).
        
    Returns:
        Scalar torch.Tensor containing the mean squared log1p error.
    """
    if pred_flow.shape != true_flow.shape:
        raise ValueError(
            f"Shape mismatch in log1p_mse_loss: pred_flow shape {pred_flow.shape} "
            f"!= true_flow shape {true_flow.shape}"
        )
    
    # Numerical safety clamp for pred_flow to ensure non-negative before log1p
    pred_safe = torch.clamp(pred_flow, min=0.0)
    true_safe = torch.clamp(true_flow, min=0.0)
    
    log_pred = torch.log1p(pred_safe)
    log_true = torch.log1p(true_safe)
    
    loss = torch.mean((log_pred - log_true) ** 2)
    return loss


class Log1pMSELoss(nn.Module):
    """PyTorch module wrapper for log1p_mse_loss."""
    def __init__(self):
        super().__init__()

    def forward(self, pred_flow: torch.Tensor, true_flow: torch.Tensor) -> torch.Tensor:
        return log1p_mse_loss(pred_flow, true_flow)
