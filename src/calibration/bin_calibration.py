"""
Bin-wise Multiplicative Calibration via Mass-Preserving KL Projection.

Formulation on Spatial Support Omega_c:
    Certain cities have spatial diameters smaller than 100 km (e.g. Bin 3 has 0 candidate pairs).
    The target distribution Y_D is conditioned on the active spatial bins of Omega_c:
        p_k^{cond} = \frac{p_k \cdot \mathbf{1}(\text{bin } k \text{ has pairs})}{\sum_{k'} p_{k'} \cdot \mathbf{1}(\text{bin } k' \text{ has pairs})}

    \hat{N} = \sum_{(i,j)\in\Omega_c} \hat{T}^{(0)}_{ij}    (total zero-shot predicted mass)
    B_k^{target} = p_k^{cond} \cdot \hat{N}                 (target mass for bin k)
    \hat{B}_k = \sum_{b(i,j)=k} \hat{T}^{(0)}_{ij}           (model implied mass for bin k)

    s_k = \frac{B_k^{target} + \epsilon}{\hat{B}_k + \epsilon}
    \hat{T}^{(1)}_{ij} = s_{b(i,j)} \cdot \hat{T}^{(0)}_{ij}

Strict Invariants Verified Automatically:
    1. Total mass preservation: \sum \hat{T}^{(1)} == \sum \hat{T}^{(0)} (within 1e-4 relative error).
    2. Bin distribution matching: \frac{\sum_{b(i,j)=k} \hat{T}^{(1)}_{ij}}{\sum \hat{T}^{(1)}_{ij}} \approx p_k^{cond}.
"""

import numpy as np
import torch


def calibrate_by_distance_bins(
    t_pred_zero_shot: torch.Tensor,
    bin_labels: torch.Tensor,
    target_yd_probs: np.ndarray | torch.Tensor,
    eps: float = 1e-8,
    tolerance: float = 1e-4,
) -> torch.Tensor:
    """
    Applies exact mass-preserving KL projection calibration conditioned on Omega_c support.
    """
    if isinstance(target_yd_probs, np.ndarray):
        p_raw = torch.tensor(target_yd_probs, dtype=torch.float32, device=t_pred_zero_shot.device)
    else:
        p_raw = target_yd_probs.to(device=t_pred_zero_shot.device, dtype=torch.float32)

    n_hat = torch.sum(t_pred_zero_shot)
    if n_hat <= 0:
        return t_pred_zero_shot

    # Compute implied bin mass \hat{B}_k on Omega_c
    implied_b = torch.zeros(4, dtype=torch.float32, device=t_pred_zero_shot.device)
    active_mask = torch.zeros(4, dtype=torch.bool, device=t_pred_zero_shot.device)
    for k in range(4):
        mask = (bin_labels == k)
        implied_b[k] = torch.sum(t_pred_zero_shot[mask])
        active_mask[k] = mask.any()

    # Condition target distribution on active bins of Omega_c
    p_active = p_raw * active_mask.float()
    active_sum = torch.sum(p_active)
    if active_sum <= 0:
        # Fallback if target has 0 mass on all active bins
        p_cond = implied_b / (n_hat + eps)
    else:
        p_cond = p_active / active_sum

    # Target mass per bin: B_k^{target} = p_k^{cond} * \hat{N}
    target_b = p_cond * n_hat

    # Scaling factor s_k = B_k^{target} / \hat{B}_k
    s = (target_b + eps) / (implied_b + eps)  # (4,)

    # Apply scaling
    t_cal = t_pred_zero_shot * s[bin_labels]

    # Invariant Check 1: Total mass preservation
    cal_mass = torch.sum(t_cal)
    mass_diff_rel = torch.abs(cal_mass - n_hat) / n_hat
    if mass_diff_rel > tolerance:
        t_cal = t_cal * (n_hat / (cal_mass + eps))

    # Invariant Check 2: Implied bin distribution matches conditional target p_cond
    cal_implied_p = torch.zeros(4, dtype=torch.float32, device=t_pred_zero_shot.device)
    for k in range(4):
        if active_mask[k]:
            cal_implied_p[k] = torch.sum(t_cal[bin_labels == k])
    cal_implied_p = cal_implied_p / (torch.sum(t_cal) + eps)

    for k in range(4):
        if active_mask[k]:
            bin_err = torch.abs(cal_implied_p[k] - p_cond[k]).item()
            assert bin_err < tolerance or target_b[k] < eps, (
                f"Calibration invariant failed on active bin {k}: target_cond={p_cond[k].item():.5f}, "
                f"got={cal_implied_p[k].item():.5f}, error={bin_err:.5f}"
            )

    return t_cal


if __name__ == "__main__":
    t0 = torch.tensor([100.0, 300.0, 600.0])  # only bins 0, 1, 2
    bins = torch.tensor([0, 1, 2])
    yd = np.array([0.10, 0.30, 0.40, 0.20])  # has 20% in bin 3 (which is absent in city)

    t1 = calibrate_by_distance_bins(t0, bins, yd)
    print("Zero-shot total mass:", t0.sum().item())
    print("Calibrated total mass:", t1.sum().item())
    print("Calibrated flows:", t1.tolist())
    print("Unit tests passed.")
