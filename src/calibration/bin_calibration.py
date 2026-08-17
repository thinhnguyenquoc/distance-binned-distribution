"""
Bin-wise Multiplicative Calibration via KL Projection.

Theoretical Framing:
    \hat{T}^{YD} = argmin_T D_{KL}(T || \hat{T}^{ZS})
    subject to:
        B(T)[k] = Y_D[k] for all k in {0, 1, 2, 3}

Closed-form solution (exact forward I-projection):
    s_k = (Y_D[k] + eps) / (\hat{Y}_D[k] + eps)
    \hat{T}^{cal}_ij = s_{k(i,j)} * \hat{T}^{ZS}_ij

Preserves the learned spatial structure while matching target distance distribution Y_D.
"""

import numpy as np
import torch


def calibrate_by_distance_bins(
    t_pred: torch.Tensor,
    bin_labels: torch.Tensor,
    target_yd: np.ndarray | torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Applies KL-projection bin calibration onto zero-shot predictions.

    Args:
        t_pred:     (E,) predicted flows \hat{T}^{ZS}_ij on Omega_c.
        bin_labels: (E,) bin index (0..3) for each pair.
        target_yd:  (4,) target distance distribution Y_D (sums to 1.0).
        eps:        numerical stability constant.

    Returns:
        t_calibrated: (E,) calibrated flows \hat{T}^{cal}_ij.
    """
    if isinstance(target_yd, np.ndarray):
        target_yd = torch.tensor(target_yd, dtype=torch.float32, device=t_pred.device)
    else:
        target_yd = target_yd.to(device=t_pred.device, dtype=torch.float32)

    total_pred = torch.sum(t_pred)
    if total_pred <= 0:
        return t_pred

    # Step 1: Compute implied model distance-bin distribution \hat{Y}_D
    implied_yd = torch.zeros(4, dtype=torch.float32, device=t_pred.device)
    for k in range(4):
        mask = (bin_labels == k)
        implied_yd[k] = torch.sum(t_pred[mask])
    implied_yd = implied_yd / (total_pred + eps)

    # Step 2: Compute scaling factors s_k
    s = (target_yd + eps) / (implied_yd + eps)  # (4,)

    # Step 3: Apply multiplicative scaling
    t_cal = t_pred * s[bin_labels]

    # Optional: ensure total flow scale is preserved
    # In relative metrics like CPC, scaling by global constant cancels out,
    # but preserving total flow sum keeps magnitudes realistic.
    t_cal = t_cal * (total_pred / (torch.sum(t_cal) + eps))
    return t_cal


if __name__ == "__main__":
    t_pred = torch.tensor([10.0, 20.0, 30.0, 40.0])
    bin_labels = torch.tensor([0, 1, 2, 3])
    target_yd = np.array([0.4, 0.3, 0.2, 0.1])

    t_cal = calibrate_by_distance_bins(t_pred, bin_labels, target_yd)
    print("Zero-shot t_pred:", t_pred)
    print("Calibrated t_cal:", t_cal)
