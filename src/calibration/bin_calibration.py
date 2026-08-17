r"""
Interzonal Moving-Bin Calibration on Omega_c^+ via Soft KL Projection.

Mathematical Formulation:
    1. Interzonal Domain:
        Omega_c^+ = {(i,j) in Omega_c : i != j, D_ij > 0}
        Intrazonal pairs (i == j, D_ii = 0) are kept intact: \hat{T}_{ii}^{cal} = \hat{T}_{ii}^{ZS}.

    2. Moving-Bin Target Distribution:
        Y_{c, k}^{Meta, +} = Y_{c, k}^{Meta} / sum_{l=1}^3 Y_{c, l}^{Meta}   for k in {1, 2, 3}
        Y_{c, k}^{oracle, +} = sum_{(i,j) in Omega_{c,k}^+} T_{ij}^{GT} / sum_{(i,j) in Omega_c^+} T_{ij}^{GT}

    3. Support Conditioning:
        For cities with diameter < 100 km (where bin 3 has 0 pairs), condition target on active moving bins:
        p_k^{cond, +} = Y_k^+ * 1(k active) / sum_{l active} Y_l^+

    4. Soft Calibration Multipliers (0 <= q <= 1):
        \hat{B}_k^+ = sum_{(i,j) in Omega_{c,k}^+} \hat{T}_{ij}^{ZS}
        \hat{N}^+ = sum_{(i,j) in Omega_c^+} \hat{T}_{ij}^{ZS}
        \hat{Y}_k^{ZS, +} = \hat{B}_k^+ / \hat{N}^+

        w_k(q) = ( p_k^{cond, +} / \hat{Y}_k^{ZS, +} )^q
        s_k = w_k(q) / sum_{l active} [ \hat{Y}_l^{ZS, +} * w_l(q) ]

        \hat{T}_{ij}^{cal} = s_{b(i,j)} * \hat{T}_{ij}^{ZS}   for (i,j) in Omega_c^+

Strict Invariants:
    1. Interzonal mass preservation: \sum_{Omega^+} \hat{T}^{cal} == \sum_{Omega^+} \hat{T}^{ZS}.
    2. Intrazonal identity: \hat{T}_{ii}^{cal} == \hat{T}_{ii}^{ZS}.
    3. At q=1: implied moving-bin proportions match p_k^{cond, +} exactly within 1e-5.
    4. At q=0: \hat{T}^{cal} == \hat{T}^{ZS} (pure zero-shot).
"""

import numpy as np
import torch


def calibrate_moving_bins(
    t_pred_zero_shot: torch.Tensor,
    bin_labels: torch.Tensor,
    pair_o_idx: torch.Tensor,
    pair_d_idx: torch.Tensor,
    target_moving_yd: np.ndarray | torch.Tensor,
    q: float = 1.0,
    eps: float = 1e-8,
    tolerance: float = 1e-5,
) -> torch.Tensor:
    """
    Applies interzonal moving-bin calibration on Omega_c^+ (bins 1, 2, 3).

    Args:
        t_pred_zero_shot: (E,) zero-shot predicted flows on Omega_c.
        bin_labels:       (E,) bin index (0=intrazonal, 1=(0,10), 2=[10,100), 3=100+).
        pair_o_idx:       (E,) origin indices.
        pair_d_idx:       (E,) destination indices.
        target_moving_yd: (3,) normalized moving-bin distribution for bins {1, 2, 3} (sums to 1.0).
        q:                soft calibration parameter in [0, 1]. q=1 is full match, q=0 is zero-shot.
        tolerance:        numerical precision tolerance (default 1e-5).

    Returns:
        t_cal: (E,) calibrated flows with intrazonal preserved and interzonal re-scaled.
    """
    assert 0.0 <= q <= 1.0, f"q must be in [0, 1], got {q}"

    if isinstance(target_moving_yd, np.ndarray):
        p_raw = torch.tensor(target_moving_yd, dtype=torch.float32, device=t_pred_zero_shot.device)
    else:
        p_raw = target_moving_yd.to(device=t_pred_zero_shot.device, dtype=torch.float32)

    # Normalize moving target
    raw_sum = torch.sum(p_raw)
    if raw_sum <= 0:
        return t_pred_zero_shot.clone()
    p_raw = p_raw / raw_sum

    # Mask for interzonal pairs Omega_c^+ (i != j and bin > 0)
    inter_mask = (pair_o_idx != pair_d_idx) & (bin_labels > 0)
    intra_mask = ~inter_mask

    # Clone predictions
    t_cal = t_pred_zero_shot.clone()

    n_inter_hat = torch.sum(t_pred_zero_shot[inter_mask])
    if n_inter_hat <= 0:
        return t_cal

    # Compute implied mass on moving bins {1, 2, 3}
    implied_b = torch.zeros(3, dtype=torch.float32, device=t_pred_zero_shot.device)
    active_mask = torch.zeros(3, dtype=torch.bool, device=t_pred_zero_shot.device)

    for idx, bin_k in enumerate([1, 2, 3]):
        k_mask = inter_mask & (bin_labels == bin_k)
        implied_b[idx] = torch.sum(t_pred_zero_shot[k_mask])
        active_mask[idx] = k_mask.any()

    # Condition target on active moving bins
    p_active = p_raw * active_mask.float()
    active_sum = torch.sum(p_active)
    if active_sum <= 0:
        p_cond = implied_b / n_inter_hat
    else:
        p_cond = p_active / active_sum

    implied_p = implied_b / n_inter_hat

    # Compute soft weights w_k(q) = (p_cond / implied_p)^q
    w = torch.zeros(3, dtype=torch.float32, device=t_pred_zero_shot.device)
    for idx in range(3):
        if active_mask[idx] and implied_p[idx] > 0:
            ratio = p_cond[idx] / implied_p[idx]
            w[idx] = ratio ** q
        else:
            w[idx] = 1.0

    # Normalization to ensure interzonal mass preservation: \sum \hat{T}^{cal} == \sum \hat{T}^{ZS}
    weighted_mass = torch.sum(implied_p * w)
    s = torch.zeros(3, dtype=torch.float32, device=t_pred_zero_shot.device)
    if weighted_mass > 0:
        s = w / weighted_mass

    # Apply scaling to interzonal pairs
    for idx, bin_k in enumerate([1, 2, 3]):
        k_mask = inter_mask & (bin_labels == bin_k)
        if k_mask.any():
            t_cal[k_mask] = t_pred_zero_shot[k_mask] * s[idx]

    # Invariant 1: Interzonal mass preservation within numerical tolerance
    cal_inter_mass = torch.sum(t_cal[inter_mask])
    mass_diff_rel = torch.abs(cal_inter_mass - n_inter_hat) / n_inter_hat
    if mass_diff_rel > tolerance:
        t_cal[inter_mask] = t_cal[inter_mask] * (n_inter_hat / cal_inter_mass)

    # Invariant 2: Intrazonal identity
    assert torch.allclose(t_cal[intra_mask], t_pred_zero_shot[intra_mask], atol=1e-6), "Intrazonal violated!"

    # Invariant 3: If q=1, verify bin matching on active bins within 1e-5
    if abs(q - 1.0) < 1e-4:
        cal_inter_p = torch.zeros(3, dtype=torch.float32, device=t_pred_zero_shot.device)
        total_inter_cal = torch.sum(t_cal[inter_mask])
        for idx, bin_k in enumerate([1, 2, 3]):
            if active_mask[idx]:
                cal_inter_p[idx] = torch.sum(t_cal[inter_mask & (bin_labels == bin_k)])
        if total_inter_cal > 0:
            cal_inter_p = cal_inter_p / total_inter_cal

        for idx in range(3):
            if active_mask[idx]:
                bin_err = torch.abs(cal_inter_p[idx] - p_cond[idx]).item()
                assert bin_err < tolerance, (
                    f"Invariant failed on moving bin {idx+1}: target={p_cond[idx].item():.6f}, "
                    f"got={cal_inter_p[idx].item():.6f}, err={bin_err:.6f}"
                )

    return t_cal


def calibrate_4bin_legacy_ablation(
    t_pred_zero_shot: torch.Tensor,
    bin_labels: torch.Tensor,
    target_4bin_yd: np.ndarray | torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Legacy 4-bin calibration (Ablation M1^{real, 4bin}) deliberately retaining
    the semantic mismatch of Bin 0 to demonstrate its empirical penalty.
    """
    if isinstance(target_4bin_yd, np.ndarray):
        p_raw = torch.tensor(target_4bin_yd, dtype=torch.float32, device=t_pred_zero_shot.device)
    else:
        p_raw = target_4bin_yd.to(device=t_pred_zero_shot.device, dtype=torch.float32)

    n_hat = torch.sum(t_pred_zero_shot)
    if n_hat <= 0:
        return t_pred_zero_shot

    implied_b = torch.zeros(4, dtype=torch.float32, device=t_pred_zero_shot.device)
    active_mask = torch.zeros(4, dtype=torch.bool, device=t_pred_zero_shot.device)
    for k in range(4):
        mask = (bin_labels == k)
        implied_b[k] = torch.sum(t_pred_zero_shot[mask])
        active_mask[k] = mask.any()

    p_active = p_raw * active_mask.float()
    p_cond = p_active / torch.clamp(torch.sum(p_active), min=eps)

    s = (p_cond * n_hat + eps) / (implied_b + eps)
    t_cal = t_pred_zero_shot * s[bin_labels]

    cal_mass = torch.sum(t_cal)
    t_cal = t_cal * (n_hat / (cal_mass + eps))
    return t_cal


if __name__ == "__main__":
    t0 = torch.tensor([50.0, 100.0, 300.0, 600.0])  # pair 0 is intrazonal, 1,2,3 are interzonal
    bins = torch.tensor([0, 1, 2, 3])
    o_idx = torch.tensor([0, 0, 0, 0])
    d_idx = torch.tensor([0, 1, 2, 3])  # pair 0 is (0,0) intrazonal
    target_moving = np.array([0.25, 0.45, 0.30])  # sums to 1.0 for bins 1, 2, 3

    # q=1.0 (Full calibration)
    t_cal_1 = calibrate_moving_bins(t0, bins, o_idx, d_idx, target_moving, q=1.0)
    print("Zero-shot t0:      ", t0.tolist())
    print("Calibrated t_cal(1):", t_cal_1.tolist())
    print("Intrazonal flow 0: ", t_cal_1[0].item(), "== t0[0]:", t0[0].item())
    print("Interzonal mass:   ", t_cal_1[1:].sum().item(), "== t0[1:].sum:", t0[1:].sum().item())

    # q=0.5 (Soft calibration)
    t_cal_half = calibrate_moving_bins(t0, bins, o_idx, d_idx, target_moving, q=0.5)
    print("Soft t_cal(0.5):   ", t_cal_half.tolist())

    # q=0.0 (Zero-shot identity)
    t_cal_0 = calibrate_moving_bins(t0, bins, o_idx, d_idx, target_moving, q=0.0)
    assert torch.allclose(t_cal_0, t0), "q=0 must equal zero-shot!"
    print("q=0 equals zero-shot: PASS")
