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
    pair_distance: torch.Tensor | None = None,
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
        pair_distance:    Optional (E,) pairwise distance tensor (log1p km or km).
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

    # Mask for interzonal pairs Omega_c^+ (i != j and D_ij > 0)
    if pair_distance is not None:
        p_dist = pair_distance.to(device=t_pred_zero_shot.device)
        dist_km = p_dist
        inter_mask = (pair_o_idx != pair_d_idx) & (dist_km > 0.0)
    else:
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
            # Inactive bin (no candidate pairs in this bin) → zero weight.
            # This is consistent with the mathematical spec: inactive bins carry no mass
            # and must not contribute to the scaling normalization.
            w[idx] = 0.0

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


# ---------------------------------------------------------------------------
# E1: Dynamic K-bin calibration (numpy-based, for Oracle Existence Test)
# ---------------------------------------------------------------------------

def calibrate_kbins(
    t0_np: np.ndarray,
    dist_km: np.ndarray,
    inter_mask: np.ndarray,
    yd_target: np.ndarray,
    bin_edges: np.ndarray,
    q: float = 1.0,
    tolerance: float = 1e-5,
) -> np.ndarray:
    r"""
    Closed-form K-bin Moving-Bin calibration for E1.

    Works on numpy arrays (CPU-only). Mirrors calibrate_moving_bins() semantics
    but accepts dynamic bin_edges (K bins, not fixed 3-bin schema).

    Mathematical formulation:
        Y_D_cond_k = Y_D_k * active_k / sum_l(Y_D_l * active_l)
        w_k(q)     = (Y_D_cond_k / Y_hat_k)^q
        s_k        = w_k / sum_l(Y_hat_l * w_l)
        T_cal_ij   = s_{b(ij)} * T0_ij   for (i,j) in Omega_c^+

    Notes on zero-behavior:
        If target Y_D_k == 0, then w_k(q) = 0 for ANY q > 0.
        This forces hard-zero predictions on that bin, making q mapping non-continuous at q=0 if the target contains exact zeros.
        Smoothing/pseudocounts must be applied to Y_D prior to calling this function if a softer response is desired.

    Invariants:
        1. Interzonal mass preservation: sum(T_cal[inter]) == sum(T0[inter]) within tolerance.
        2. Intrazonal identity: T_cal[~inter] == T0[~inter] exactly.
        3. At q=1: bin proportions of T_cal match Y_D_cond within tolerance for active bins.
        4. GT-independence: output is a function of T0 and Y_D only, not T^GT.

    Args:
        t0_np:      (E,) zero-shot predicted flows (numpy float array).
        dist_km:    (E,) pairwise distances in km.
        inter_mask: (E,) boolean mask for Omega_c^+ (interzonal, D>0).
        yd_target:  (K,) target distance distribution summing to 1.0.
        bin_edges:  (K+1,) strictly increasing edges from compute_kbin_edges.
        q:          soft calibration strength in [0, 1]. q=1 = exact match.
        tolerance:  numerical precision for invariant checks.

    Returns:
        t_cal: (E,) calibrated flows; intrazonal unchanged, interzonal rescaled.
    """
    assert 0.0 <= q <= 1.0, f"q must be in [0, 1], got {q}"
    K = len(bin_edges) - 1
    assert len(yd_target) == K, f"yd_target length {len(yd_target)} != K={K}"

    # Normalize input Y_D (defensive)
    yd_sum = float(np.sum(yd_target))
    yd_raw = yd_target / yd_sum if yd_sum > 0 else np.ones(K) / K

    t_cal = t0_np.copy().astype(np.float64)
    inter_T0 = t0_np[inter_mask].astype(np.float64)
    N_hat = inter_T0.sum()

    if N_hat <= 0:
        return t_cal  # no interzonal flow to calibrate

    inter_dist = dist_km[inter_mask]

    # Compute implied distribution Y_hat from zero-shot
    Y_hat = np.zeros(K, dtype=np.float64)
    active = np.zeros(K, dtype=bool)
    for k in range(K):
        lo, hi = float(bin_edges[k]), float(bin_edges[k + 1])
        in_bin = (inter_dist > lo) & (inter_dist <= hi)
        Y_hat[k] = inter_T0[in_bin].sum() / N_hat
        active[k] = bool(in_bin.any())

    # Condition Y_D on active bins only
    yd_active = yd_raw * active.astype(np.float64)
    active_sum = yd_active.sum()
    Y_D_cond = yd_active / active_sum if active_sum > 0 else Y_hat.copy()

    # Soft weights: w_k = (Y_D_cond_k / Y_hat_k)^q
    w = np.ones(K, dtype=np.float64)
    for k in range(K):
        if active[k] and Y_hat[k] > 0:
            w[k] = (Y_D_cond[k] / Y_hat[k]) ** q

    # Normalize: s_k = w_k / sum_l(Y_hat_l * w_l)
    weighted_mass = float((Y_hat * w).sum())
    s = w / weighted_mass if weighted_mass > 0 else np.ones(K)

    # Apply per-bin scaling to interzonal pairs
    idx = np.where(inter_mask)[0]
    for k in range(K):
        lo, hi = float(bin_edges[k]), float(bin_edges[k + 1])
        in_bin = (inter_dist > lo) & (inter_dist <= hi)
        t_cal[idx[in_bin]] = t0_np[idx[in_bin]] * s[k]

    # --- Invariant 1: Interzonal mass preservation ---
    cal_mass = float(t_cal[inter_mask].sum())
    mass_err_rel = abs(cal_mass - N_hat) / max(N_hat, 1e-8)
    if mass_err_rel > tolerance:
        t_cal[inter_mask] = t_cal[inter_mask] * (N_hat / cal_mass)

    # --- Invariant 2: Intrazonal identity ---
    intra_mask = ~inter_mask
    assert np.allclose(t_cal[intra_mask], t0_np[intra_mask], atol=1e-6), \
        "calibrate_kbins: Intrazonal identity violated"

    # --- Invariant 3: At q=1, bin proportions match Y_D_cond ---
    if abs(q - 1.0) < 1e-4:
        total_cal = float(t_cal[inter_mask].sum())
        if total_cal > 0:
            for k in range(K):
                if active[k]:
                    lo, hi = float(bin_edges[k]), float(bin_edges[k + 1])
                    in_bin_cal = (inter_dist > lo) & (inter_dist <= hi)
                    cal_prop = float(t_cal[inter_mask][in_bin_cal].sum()) / total_cal
                    bin_err = abs(cal_prop - Y_D_cond[k])
                    assert bin_err < tolerance, (
                        f"calibrate_kbins bin {k}: target={Y_D_cond[k]:.6f}, "
                        f"got={cal_prop:.6f}, err={bin_err:.6f}"
                    )

    return t_cal


def calibrate_kbins_grouped(
    t0_np: np.ndarray,
    dist_km: np.ndarray,
    inter_mask: np.ndarray,
    yd_target_dict: dict,
    bin_edges: np.ndarray,
    pair_group_idx: np.ndarray,
    q: float = 1.0,
    tolerance: float = 1e-5,
) -> np.ndarray:
    """
    Group-conditioned K-bin calibration (e.g., per-county).
    
    Applies the closed-form K-bin calibration independently for each group
    defined by pair_group_idx (e.g., origin county ID of each pair),
    while preserving the zero-shot predicted outflow of each group.
    
    Args:
        t0_np:          (E,) zero-shot predicted flows.
        dist_km:        (E,) pairwise distances in km.
        inter_mask:     (E,) boolean mask for interzonal pairs Omega_c^+.
        yd_target_dict: Dict mapping group_id -> (K,) target distance distribution.
        bin_edges:      (K+1,) strictly increasing edges.
        pair_group_idx: (E,) group ID for each pair (e.g., origin county ID).
        q:              Soft calibration strength.
        tolerance:      Numerical precision.
        
    Returns:
        t_cal: (E,) calibrated flows.
    """
    t_cal = t0_np.copy().astype(np.float64)
    
    # Intrazonal pairs are not modified
    # We calibrate interzonal pairs group by group
    
    unique_groups = np.unique(pair_group_idx)
    
    for g in unique_groups:
        if g not in yd_target_dict:
            continue
            
        yd_g = yd_target_dict[g]
        
        # Mask for interzonal pairs belonging to group g
        g_mask = (pair_group_idx == g)
        inter_g_mask = inter_mask & g_mask
        
        if not inter_g_mask.any():
            continue
            
        # Extract slices for this group
        t0_g = t0_np[g_mask]
        dist_g = dist_km[g_mask]
        
        # We need a local inter_mask for the group slice
        # inter_g_mask is length E. We need a mask of length len(t0_g)
        # Since t0_g is selected by g_mask, the local inter_mask is simply
        # inter_mask[g_mask]
        local_inter_mask = inter_mask[g_mask]
        
        # Apply city-level calibration logic locally to the group
        # calibrate_kbins requires full E-length arrays if we pass them, 
        # but it works on any size. We pass the local slices.
        t_cal_g = calibrate_kbins(
            t0_np=t0_g,
            dist_km=dist_g,
            inter_mask=local_inter_mask,
            yd_target=yd_g,
            bin_edges=bin_edges,
            q=q,
            tolerance=tolerance
        )
        
        # Assign back to the global array
        t_cal[g_mask] = t_cal_g
        
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
