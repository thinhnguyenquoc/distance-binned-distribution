"""
Positive Interzonal OD Support Module (Omega_t^+).

Protocol Requirements:
1. Core Invariant:
       Omega_t^+ = {(i,j) in Omega_t | origin != destination, distance_km > 0, true_flow >= 1}
   This is the fixed positive interzonal OD support / oracle-support intensity reconstruction setting.

2. Invariant Pipeline Flow:
   ONE TARGET CITY
       -> FIX Omega_t^+
       -> baseline metrics on Omega_t^+
       -> target DBD p on Omega_t^+
       -> predicted DBD q on Omega_t^+
       -> calibration on Omega_t^+
       -> volume preservation check on Omega_t^+
       -> post-calibration metrics on Omega_t^+

   ALL USE EXACTLY THE SAME Omega_t^+.

3. Sanity Checks:
   - assert np.all(origin != destination)
   - assert np.all(distance_km > 0)
   - assert np.all(true_flow >= 1)
   - assert abs(p.sum() - 1.0) < 1e-12
   - assert abs(q.sum() - 1.0) < 1e-12
   - assert len(y_true) == len(y_pred_before) == len(y_pred_after)
"""

from typing import Tuple, Dict
import numpy as np
import pandas as pd
import torch


def get_positive_interzonal_support_mask(
    pair_o_idx: np.ndarray | torch.Tensor,
    pair_d_idx: np.ndarray | torch.Tensor,
    pair_distance_km: np.ndarray | torch.Tensor,
    pair_trips: np.ndarray | torch.Tensor
) -> np.ndarray:
    """
    Constructs the boolean mask for positive interzonal OD support:
        (origin != destination) & (distance_km > 0) & (true_flow >= 1)
    """
    if isinstance(pair_o_idx, torch.Tensor):
        pair_o_idx = pair_o_idx.detach().cpu().numpy()
    if isinstance(pair_d_idx, torch.Tensor):
        pair_d_idx = pair_d_idx.detach().cpu().numpy()
    if isinstance(pair_distance_km, torch.Tensor):
        pair_distance_km = pair_distance_km.detach().cpu().numpy()
    if isinstance(pair_trips, torch.Tensor):
        pair_trips = pair_trips.detach().cpu().numpy()

    mask = (
        (pair_o_idx != pair_d_idx)
        & (pair_distance_km > 0.0)
        & (pair_trips >= 1.0)
    )
    return mask


def validate_positive_interzonal_support(
    origin: np.ndarray,
    destination: np.ndarray,
    distance_km: np.ndarray,
    true_flow: np.ndarray
) -> None:
    """
    Executes mandatory sanity checks verifying the integrity of Omega_t^+.
    """
    if len(true_flow) == 0:
        raise ValueError("Positive support Omega_t^+ cannot be empty.")
    if not np.all(origin != destination):
        raise AssertionError("Support contains intrazonal pairs (origin == destination)!")
    if not np.all(distance_km > 0.0):
        raise AssertionError("Support contains zero or negative distance pairs!")
    if not np.all(true_flow >= 1.0):
        raise AssertionError("Support contains zero-flow pairs (true_flow < 1)!")


def compute_r_vol(
    pred_flow_support: np.ndarray,
    true_flow_support: np.ndarray,
) -> float:
    r"""
    Computes predicted-to-observed total volume ratio (evaluation-only diagnostic):
        R_vol = \sum_{(i,j) \in \Omega_t^+} \hat{T}_{ij} / \sum_{(i,j) \in \Omega_t^+} T_{ij}

    Strict Rules:
    - Purely diagnostic evaluation metric; NEVER used for model fitting, hyperparameter tuning,
      prediction rescaling, DBD calibration, checkpoint selection, donor selection, or G/alpha tuning.
    - Operates strictly on fixed positive interzonal OD support \Omega_t^+.
    """
    true_sum = float(np.sum(true_flow_support))
    if true_sum <= 0:
        raise ValueError("Denominator of R_vol (sum of true flows on positive support) must be > 0.")
    pred_sum = float(np.sum(pred_flow_support))
    return float(pred_sum / true_sum)


def validate_volume_preservation(
    pred_before: np.ndarray,
    pred_after: np.ndarray,
    true_flow: np.ndarray,
    tol: float = 1e-10,
) -> Tuple[float, float]:
    r"""
    Sanity check verifying exact volume preservation of DBD calibration:
        R_vol^{after} == R_vol^{before}  (up to floating-point tolerance 1e-10)
    
    Returns:
        (r_before, r_after)
    """
    r_before = compute_r_vol(pred_before, true_flow)
    r_after = compute_r_vol(pred_after, true_flow)
    diff = abs(r_before - r_after)
    if diff >= tol:
        raise AssertionError(
            f"Volume preservation invariant violated: R_vol_before={r_before:.12f}, "
            f"R_vol_after={r_after:.12f}, abs_diff={diff:.4e} >= tol={tol:.4e}"
        )
    return r_before, r_after


def build_support_audit_record(
    target_city: str,
    true_flow_support: np.ndarray,
    pred_before_support: np.ndarray,
    pred_after_support: np.ndarray
) -> Dict:
    """
    Builds an audit record confirming exact support preservation and volume equality.
    """
    if len(true_flow_support) != len(pred_before_support) or len(true_flow_support) != len(pred_after_support):
        raise AssertionError("Length mismatch across support arrays during audit generation!")

    r_before, r_after = validate_volume_preservation(pred_before_support, pred_after_support, true_flow_support)

    return {
        "target_city": target_city,
        "n_positive_support_pairs": len(true_flow_support),
        "true_total_on_positive_support": round(float(np.sum(true_flow_support)), 4),
        "pred_total_before_on_positive_support": round(float(np.sum(pred_before_support)), 4),
        "pred_total_after_on_positive_support": round(float(np.sum(pred_after_support)), 4),
        "R_vol": round(r_before, 6),
        "support_definition": "i!=j & distance_km>0 & true_flow>=1",
    }
