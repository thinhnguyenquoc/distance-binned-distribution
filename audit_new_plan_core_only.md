# AUDIT: CORE IMPLEMENTATION FOR NEW_PLAN (RESEARCH PROTOCOL)

- **Total files**: 17
- **Scope**: Core modules directly implementing `new_plan.md` with absolute volume preservation, float64 persistence, canonical MixedLM, and zero fallbacks.

## Table of Contents

1. [implement_new_plan/calibration/support.py](#implement-new-plan-calibration-support-py)
2. [implement_new_plan/calibration/source_bins.py](#implement-new-plan-calibration-source-bins-py)
3. [implement_new_plan/calibration/tv_noise.py](#implement-new-plan-calibration-tv-noise-py)
4. [implement_new_plan/calibration/dose_matching.py](#implement-new-plan-calibration-dose-matching-py)
5. [implement_new_plan/calibration/statistical_inference.py](#implement-new-plan-calibration-statistical-inference-py)
6. [implement_new_plan/data/dataset.py](#implement-new-plan-data-dataset-py)
7. [implement_new_plan/data/od_split.py](#implement-new-plan-data-od-split-py)
8. [implement_new_plan/data/source_scaler.py](#implement-new-plan-data-source-scaler-py)
9. [implement_new_plan/data/urban_graph.py](#implement-new-plan-data-urban-graph-py)
10. [implement_new_plan/models/gravity.py](#implement-new-plan-models-gravity-py)
11. [implement_new_plan/models/node_encoder.py](#implement-new-plan-models-node-encoder-py)
12. [implement_new_plan/models/decoder.py](#implement-new-plan-models-decoder-py)
13. [implement_new_plan/models/od_models.py](#implement-new-plan-models-od-models-py)
14. [implement_new_plan/loss/log1p_mse.py](#implement-new-plan-loss-log1p-mse-py)
15. [implement_new_plan/training/evaluate.py](#implement-new-plan-training-evaluate-py)
16. [implement_new_plan/training/model_trainer.py](#implement-new-plan-training-model-trainer-py)
17. [implement_new_plan/experiment/master_protocol_runner.py](#implement-new-plan-experiment-master-protocol-runner-py)

---

<a id="implement-new-plan-calibration-support-py"></a>
## File: `implement_new_plan/calibration/support.py` (148 lines)

```python
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
```

---

<a id="implement-new-plan-calibration-source-bins-py"></a>
## File: `implement_new_plan/calibration/source_bins.py` (277 lines)

```python
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
    
    total_flow = float(np.sum(true_flow))
    if total_flow <= 0:
        raise ValueError("Cannot build target DBD on zero total flow.")

    K = len(bin_edges) - 1
    bin_ids = assign_to_source_bins(distance_km, bin_edges)

    p = np.zeros(K, dtype=np.float64)
    for b in range(K):
        mask = (bin_ids == b)
        if mask.any():
            p[b] = float(np.sum(true_flow[mask])) / total_flow

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
    if diff >= tol:
        raise AssertionError(
            f"Exact volume preservation failed: |sum_after - sum_before| = {diff:.6e} >= {tol:.6e}. "
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
```

---

<a id="implement-new-plan-calibration-tv-noise-py"></a>
## File: `implement_new_plan/calibration/tv_noise.py` (157 lines)

```python
"""
Exact Total Variation (TV) Noise Perturbation Module (Experiment C).

Protocol Requirements:
1. Exact TV Distance: TV(p, p_tilde) = 1/2 * sum_b |p_b - p_tilde_b| == epsilon.
2. Zero-sum centered perturbation:
       u_b ~ N(0, 1),  u_b <- u_b - 1/K * sum_k u_k  => sum_b u_b = 0
3. Scaling:
       alpha = 2 * epsilon / sum_b |u_b|
       p_tilde_b = p_b + alpha * u_b
4. Feasibility check:
       min_b p_tilde_b >= 0
       If violated, reject and resample (up to max_attempts = 10,000).
       NO CLIPPING or RE-NORMALIZATION allowed.
5. Special case:
       If epsilon == 0: p_tilde = p directly.
6. Validation checks:
       |sum_b p_tilde_b - 1| < 1e-12
       min_b p_tilde_b >= 0.0
       |TV(p, p_tilde) - epsilon| < 1e-10
"""

from typing import Tuple, Union, Optional
import hashlib
import numpy as np


def derive_noise_seed(
    global_noise_seed: int,
    source_city: str,
    target_city: str,
    K: int,
    epsilon: float,
    realization_id: int
) -> int:
    r"""
    Derives a deterministic 64-bit integer seed via SHA-256 for reproducible noise generation.
    
    Canonical String Format:
        global_noise_seed|source_city|target_city|K|epsilon|realization_id
    Example:
        42|Boston|Seattle|8|0.060000|7
    
    Seed extraction:
        8 bytes big-endian unsigned integer from SHA-256 digest.
    """
    epsilon_str = f"{epsilon:.6f}"
    canonical_string = f"{global_noise_seed}|{source_city.strip()}|{target_city.strip()}|{K}|{epsilon_str}|{realization_id}"
    digest = hashlib.sha256(canonical_string.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
    # Numpy Generator / default_rng accepts 64-bit unsigned int
    return seed


def compute_tv_distance(p: np.ndarray, p_tilde: np.ndarray) -> float:
    """Computes exact Total Variation distance: 1/2 * sum |p - p_tilde|."""
    return float(0.5 * np.sum(np.abs(p - p_tilde)))


def generate_exact_tv_noise(
    p: np.ndarray,
    epsilon: float,
    rng_or_seed: Union[np.random.Generator, int],
    source_city: str = "UNKNOWN",
    target_city: str = "UNKNOWN",
    K: Optional[int] = None,
    realization_id: int = 0,
    max_attempts: int = 10000,
) -> Tuple[np.ndarray, float, int]:
    r"""
    Generates perturbed distribution p_tilde such that TV(p, p_tilde) == epsilon exactly.

    Protocol Requirements & Hard Invariants:
    1. Zero-sum perturbation direction: u = rng.normal(0, 1, size=K); u = u - u.mean()
    2. Exact scaling: alpha = 2.0 * epsilon / np.abs(u).sum() => p_tilde = p + alpha * u
    3. Strict Rejection sampling:
       If np.any(p_cand < 0.0), reject entire vector u and resample next from RNG stream.
       NO clipping, NO re-normalization, NO per-bin repair.
    4. Exact Numerical Validation:
       - |sum(p_cand) - 1.0| < 1e-12
       - min(p_cand) >= 0.0
       - |TV(p, p_cand) - epsilon| < 1e-10
       If any check fails: continue rejection.
    5. Hard Failure Policy:
       If max_attempts (10,000) reached: RAISE HARD ERROR with full diagnostic context.
       NO silent fallback, NO epsilon reduction, NO algorithm switching.
    """
    p_len = len(p)
    if K is None:
        K = p_len
    assert p_len == K, f"Length of p ({p_len}) does not match K ({K})"

    if isinstance(rng_or_seed, int):
        rng = np.random.default_rng(rng_or_seed)
    else:
        rng = rng_or_seed

    # Special case: epsilon == 0
    if epsilon == 0.0 or abs(epsilon) < 1e-12:
        return p.copy().astype(np.float64), 0.0, 1

    attempts = 0
    while attempts < max_attempts:
        attempts += 1
        
        # 1. Random vector u ~ N(0, 1)
        u = rng.normal(0.0, 1.0, size=K)
        
        # 2. Zero-sum centering: sum_b u_b = 0
        u = u - np.mean(u)
        
        l1_norm = np.sum(np.abs(u))
        if l1_norm < 1e-12:
            continue
            
        # 3. Scale factor alpha = 2 * epsilon / sum |u_b|
        alpha = (2.0 * epsilon) / l1_norm
        p_cand = p + alpha * u
        
        # 4. Strict Rejection Sampling: strictly non-negative, no clipping
        if np.any(p_cand < 0.0):
            continue  # REJECT entire candidate, resample from stream
            
        # 5. Numerical Validation Checks: strictly sum to 1.0, no renormalization
        sum_val = float(np.sum(p_cand))
        sum_err = abs(sum_val - 1.0)
        if sum_err >= 1e-12:
            continue  # REJECT if floating-point sum deviates >= 1e-12
            
        actual_tv = compute_tv_distance(p, p_cand)
        tv_err = abs(actual_tv - epsilon)
        if tv_err >= 1e-10:
            continue
            
        # Passed all strict validation checks cleanly
        return p_cand, actual_tv, attempts

    # 6. Hard Failure Policy: Raise error with required diagnostic log
    pos_mass = p[p > 0]
    min_pos_mass = float(np.min(pos_mass)) if len(pos_mass) > 0 else 0.0
    zero_bins = int(np.sum(p == 0.0))
    
    error_msg = (
        f"HARD FAILURE: Perturbation rejection sampling failed after {max_attempts} attempts!\n"
        f"  source_city: {source_city}\n"
        f"  target_city: {target_city}\n"
        f"  K: {K}\n"
        f"  epsilon: {epsilon:.6f}\n"
        f"  realization_id: {realization_id}\n"
        f"  noise_seed: {rng_or_seed if isinstance(rng_or_seed, int) else 'default_rng'}\n"
        f"  max_attempts: {max_attempts}\n"
        f"  min_positive_mass_of_p: {min_pos_mass:.6e}\n"
        f"  number_of_zero_bins: {zero_bins}\n"
        "Terminating run immediately under hard failure policy (no silent fallbacks)."
    )
    raise RuntimeError(error_msg)

```

---

<a id="implement-new-plan-calibration-dose-matching-py"></a>
## File: `implement_new_plan/calibration/dose_matching.py` (540 lines)

```python
"""
Dose-Matched Scaled Donor DBD Module for Structural Control Experiment (Experiment D).

Core Principle:
    [Source City Dictates Bin Boundaries for Baseline, Target, and Donor]
    For any transfer pair (s, t) and donor city d, all three distributions:
        q_{s,t}  (baseline predicted DBD of target t)
        p_{s,t}  (ground-truth target DBD of target t)
        p_{s,d}  (donor DBD of donor city d)
    MUST be evaluated on the SAME frozen source-specific bin system B^{(s,K)}.

Mathematical Formulation on Effective Baseline Support B^+ = {b : q_b > 0.0}:
1. Target Dose (Effective Support RMS Log-Ratio):
       Dose(p_{s,t}, q_{s,t}) = sqrt( 1/|B^+| * sum_{b in B^+} [ log( (p_{s,t,b}^+ + eps) / (q_{s,t,b} + eps) ) ]^2 )
2. Donor Dose (Effective Support RMS Log-Ratio):
       Dose(p_{s,d}, q_{s,t}) = sqrt( 1/|B^+| * sum_{b in B^+} [ log( (p_{s,d,b}^+ + eps) / (q_{s,t,b} + eps) ) ]^2 )
3. Perturbation in Log-Ratio Space:
       z_{s,d,b} = log( (p_{s,d,b} + eps) / (q_{s,t,b} + eps) )
4. Scaled Perturbation and Reconstruction:
       p_scaled_{s,d,b}(lambda) = q_{s,t,b} * exp(lambda * z_{s,d,b}) / sum_{k in B^+}( q_{s,t,k} * exp(lambda * z_{s,d,k}) )
5. Root Finding for lambda*:
       | Dose(p_scaled_{s,d}(lambda*), q_{s,t}) - Dose(p_{s,t}, q_{s,t}) | < 1e-6
"""

from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp


def compute_source_binned_dbd(
    distances_km: np.ndarray,
    trips: np.ndarray,
    source_bin_edges: np.ndarray
) -> np.ndarray:
    """
    Computes the DBD distribution of ANY city (target or donor) projected onto
    the source city's frozen bin system B^{(s,K)}.

    Args:
        distances_km: OD physical distances (km) of the city.
        trips: OD flow values (predictions or ground-truth).
        source_bin_edges: Pre-frozen bin boundaries [0, w_s, ..., inf] from source s.

    Returns:
        dbd: Normalized probability distribution over K bins (sum(dbd) == 1.0).
    """
    from implement_new_plan.calibration.source_bins import assign_to_source_bins
    K = len(source_bin_edges) - 1
    bin_ids = assign_to_source_bins(distances_km, source_bin_edges)
    
    total_trips = float(np.sum(trips))
    if total_trips <= 0:
        raise ValueError("Cannot compute DBD on zero total trips.")
        
    dbd = np.zeros(K, dtype=np.float64)
    for b in range(K):
        dbd[b] = np.sum(trips[bin_ids == b]) / total_trips
    return dbd


class StructuralControlDomainError(ValueError):
    """Raised when structural control inputs or donor coverage violate domain invariants."""
    pass


def compute_rms_log_ratio_dose(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    r"""
    Computes RMS log-ratio dose on effective baseline support B^+ = {b : q_b > 0.0}:
        Dose = sqrt( 1/|B^+| * sum_{b in B^+} [ log( (p_b^+ + eps) / (q_b + eps) ) ]^2 )
    where p^+ is renormalized on B^+ by P_covered = sum_{b in B^+} p_b.
    """
    pos_mask = (q > 0.0)
    if not np.any(pos_mask):
        raise StructuralControlDomainError(
            "Baseline predictions have zero mass across all bins (B^+ is empty). Cannot compute RMS dose."
        )
    
    p_covered = float(np.sum(p[pos_mask]))
    if not np.isfinite(p_covered) or p_covered <= 0.0:
        raise StructuralControlDomainError(
            f"Distribution has zero or non-finite covered mass on B^+ (P_covered = {p_covered:.6e}). "
            "Cannot normalize distribution on effective baseline support."
        )
    p_plus = p[pos_mask] / p_covered
    q_plus = q[pos_mask]
    
    log_ratio = np.log((p_plus + eps) / (q_plus + eps))
    return float(np.sqrt(np.mean(log_ratio ** 2)))


def reconstruct_scaled_donor(
    p_s_donor: np.ndarray,
    q_s_target: np.ndarray,
    lambda_param: float,
    eps: float = 1e-9
) -> np.ndarray:
    r"""
    Reconstructs normalized scaled donor distribution p_scaled_{s,d}(lambda)
    on source bin system B^{(s,K)} with exact identity at lambda = 0 and strict zero-mass preservation:
    
    1. If lambda == 0: returns q_s_target.copy() bit-exactly.
    2. If lambda > 0: operates strictly on baseline positive support B+ = {b : q_b > 0}:
        z_b = log((p_{s,d,b} + eps) / (q_{s,t,b} + eps))
        log_w_b = log(q_{s,t,b}) + lambda * z_b  (for b in B+)
        log_w_b = log_w_b - max(log_w)
        w_b = exp(log_w_b) (for b in B+), 0.0 (for b not in B+)
        p_scaled = w / sum(w)
    """
    assert lambda_param >= 0.0, f"Lambda must be non-negative, got {lambda_param}"
    if lambda_param == 0.0:
        return q_s_target.copy()

    pos_mask = (q_s_target > 0.0)
    if not np.any(pos_mask):
        raise StructuralControlDomainError(
            "Baseline predictions have zero mass across all bins (B^+ is empty). Cannot reconstruct scaled donor."
        )

    donor_covered = float(np.sum(p_s_donor[pos_mask]))
    if not np.isfinite(donor_covered) or donor_covered <= 0.0:
        raise StructuralControlDomainError(
            f"Donor distribution has zero covered mass on baseline support B^+: {donor_covered:.6e} <= 0.0"
        )

    p_scaled = np.zeros_like(q_s_target, dtype=np.float64)
    z = np.log((p_s_donor[pos_mask] + eps) / (q_s_target[pos_mask] + eps))
    log_w = np.log(q_s_target[pos_mask]) + lambda_param * z
    log_w = log_w - np.max(log_w)
    w = np.exp(log_w)
    sum_w = float(np.sum(w))
    if not np.isfinite(sum_w) or sum_w <= 0.0:
        raise StructuralControlDomainError(
            f"Invalid partition sum in scaled donor reconstruction: sum_w={sum_w}"
        )
    p_scaled[pos_mask] = w / sum_w

    return p_scaled


def find_dose_matching_lambda(
    p_s_target: np.ndarray,
    p_s_donor: np.ndarray,
    q_s_target: np.ndarray,
    tolerance: float = 1e-6,
    eps: float = 1e-9,
    source_city: str = "UNKNOWN",
    target_city: str = "UNKNOWN",
    donor_city: str = "UNKNOWN",
) -> Dict[str, Any]:
    r"""
    Deterministic Brent Root Finding for lambda* in Experiment D.
    
    Protocol Requirements & Hard Invariants:
    1. Fixed Target Dose: D_target = Dose(p_{s,t}, q_{s,t}) with eps_d = 1e-9.
    2. Root function: f(lambda) = Dose(p_scaled(lambda), q) - D_target.
    3. Case D_target < 1e-12: lambda* = 0, p_scaled = q, Dose = 0, no root finding.
    4. Deterministic Lower Bound: lambda_low = 0.0 (f(0) = -D_target <= 0).
    5. Deterministic Upper-Bound Search:
       lambda_high initialized to 1.0, doubled sequentially (1, 2, 4, 8, ..., <= 1024)
       until f(lambda_high) >= 0.
    6. Hard Failure Policy:
       If lambda_high > 1024 and f(lambda_high) < 0: RAISE HARD ERROR.
       NO alternative donor, NO minimization fallback, NO approximate scaling.
    7. Brent Root Finding:
       brentq(f, lambda_low, lambda_high, xtol=1e-12, rtol=1e-12, maxiter=100).
    8. Post-Solution Validation:
       |Dose(p_scaled(lambda*), q) - D_target| < 1e-6.
       |sum(p_scaled) - 1| < 1e-12, min(p_scaled) >= -1e-12.
    """
    assert eps == 1e-9, f"eps must be fixed at 1e-9, got {eps}"

    target_dose = compute_rms_log_ratio_dose(p_s_target, q_s_target, eps=eps)
    donor_raw_dose = compute_rms_log_ratio_dose(p_s_donor, q_s_target, eps=eps)

    # 1. Target Dose < 1e-12: Identity baseline control
    if target_dose < 1e-12:
        return {
            "lambda_star": 0.0,
            "p_scaled_donor": q_s_target.copy(),
            "target_dose": 0.0,
            "donor_raw_dose": donor_raw_dose,
            "donor_scaled_dose": 0.0,
            "dose_error": 0.0,
            "lambda_low": 0.0,
            "lambda_high": 0.0,
            "root_iterations": 0,
            "root_converged": True,
        }

    # Root function
    def f(lam: float) -> float:
        p_cand = reconstruct_scaled_donor(p_s_donor, q_s_target, lam, eps=eps)
        cand_dose = compute_rms_log_ratio_dose(p_cand, q_s_target, eps=eps)
        return cand_dose - target_dose

    # Lower bound
    lambda_low = 0.0
    f_low = f(lambda_low)  # -target_dose <= 0
    assert f_low <= 0.0, f"f(0) must be <= 0, got {f_low}"

    # Deterministic Upper-Bound Search: 1, 2, 4, 8, 16, ..., 1024
    lambda_high = 1.0
    bracket_found = False

    while lambda_high <= 1024.0:
        f_high = f(lambda_high)
        if f_high >= 0.0:
            bracket_found = True
            break
        lambda_high *= 2.0

    # Hard Failure Check: Unable to bracket root up to lambda_high = 1024
    if not bracket_found:
        raise RuntimeError(
            f"HARD FAILURE in Experiment D Dose Matching: Cannot bracket root up to lambda_high=1024!\n"
            f"  source_city: {source_city}\n"
            f"  target_city: {target_city}\n"
            f"  donor_city: {donor_city}\n"
            f"  target_dose: {target_dose:.6e}\n"
            f"  donor_raw_dose: {donor_raw_dose:.6e}\n"
            f"  f(lambda_high=1024): {f(1024.0):.6e} < 0\n"
            "Terminating run immediately under hard failure policy (no minimization or donor swapping allowed)."
        )

    # Brent Root Finding
    brent_res = brentq(
        f,
        lambda_low,
        lambda_high,
        xtol=1e-12,
        rtol=1e-12,
        maxiter=100,
        full_output=True,
    )
    lambda_star = float(brent_res[0])
    root_info = brent_res[1]
    root_iterations = int(root_info.iterations)
    root_converged = bool(root_info.converged)

    if not root_converged:
        raise RuntimeError(
            f"HARD FAILURE: Brent root finding did not converge within 100 iterations for pair "
            f"({source_city} -> {target_city}, donor {donor_city})!"
        )

    # Reconstruct final distribution
    p_scaled_donor = reconstruct_scaled_donor(p_s_donor, q_s_target, lambda_star, eps=eps)
    donor_scaled_dose = compute_rms_log_ratio_dose(p_scaled_donor, q_s_target, eps=eps)
    dose_error = abs(donor_scaled_dose - target_dose)

    # Validation Checks
    if dose_error >= 1e-6:
        raise AssertionError(
            f"HARD FAILURE: Dose error {dose_error:.6e} >= 1e-6 threshold after Brent root finding!"
        )

    sum_err = abs(float(np.sum(p_scaled_donor)) - 1.0)
    min_val = float(np.min(p_scaled_donor))
    if sum_err >= 1e-12 or min_val < -1e-12:
        raise AssertionError(
            f"HARD FAILURE: p_scaled_donor distribution invalid: sum_err={sum_err:.6e}, min_val={min_val:.6e}"
        )

    return {
        "lambda_star": lambda_star,
        "p_scaled_donor": p_scaled_donor,
        "target_dose": target_dose,
        "donor_raw_dose": donor_raw_dose,
        "donor_scaled_dose": donor_scaled_dose,
        "dose_error": dose_error,
        "lambda_low": lambda_low,
        "lambda_high": lambda_high,
        "root_iterations": root_iterations,
        "root_converged": root_converged,
    }


# ===========================================================================
# DETERMINISTIC DONOR SELECTION (EXPERIMENT D PROTOCOL)
# ===========================================================================
DONOR_SELECTION_RULE = "first cyclic city after target excluding source and target"


def choose_donor(
    source_city: str,
    target_city: str,
    canonical_cities: list[str],
) -> str:
    r"""
    Selects the donor city deterministically using canonical cyclic iteration:
    1. Find index of target t in canonical_cities.
    2. Starting from the city immediately following target, iterate cyclically.
    3. Select the first city satisfying d != target_city and d != source_city.
    
    Invariants:
    - Pure function of (source_city, target_city, canonical_cities).
    - No random choice, no model dependency, no noise dependency.
    - Zero post-calibration performance or similarity bias.
    """
    assert target_city in canonical_cities, f"Target city {target_city} not in canonical list"
    assert source_city in canonical_cities, f"Source city {source_city} not in canonical list"
    assert source_city != target_city, f"Source and target cannot be identical: {source_city}"

    t_idx = canonical_cities.index(target_city)
    n_cities = len(canonical_cities)

    for offset in range(1, n_cities):
        candidate = canonical_cities[(t_idx + offset) % n_cities]
        if candidate != target_city and candidate != source_city:
            # Sanity checks
            assert candidate != source_city, "Donor cannot equal source city"
            assert candidate != target_city, "Donor cannot equal target city"
            return candidate

    raise RuntimeError(f"No valid donor city found for source={source_city}, target={target_city}")


def generate_donor_mapping(
    canonical_cities: list[str],
    output_csv_path: Optional[str] = "manifests/donor_mapping.csv",
) -> "pd.DataFrame":
    r"""
    Generates and saves the frozen donor mapping manifest for all 50 x 49 = 2,450 pairs.
    Schema:
        source_city, target_city, donor_city, donor_rule, canonical_order_hash
    """
    import hashlib
    import pandas as pd

    # Compute SHA-256 hash of canonical order
    order_str = ",".join(canonical_cities)
    order_hash = hashlib.sha256(order_str.encode("utf-8")).hexdigest()

    records = []
    for s in canonical_cities:
        for t in canonical_cities:
            if s == t:
                continue
            donor = choose_donor(source_city=s, target_city=t, canonical_cities=canonical_cities)
            records.append({
                "source_city": s,
                "target_city": t,
                "donor_city": donor,
                "donor_rule": DONOR_SELECTION_RULE,
                "canonical_order_hash": order_hash,
            })

    df = pd.DataFrame(records)
    if output_csv_path is not None:
        import os
        from pathlib import Path
        Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv_path, index=False)
        print(f"Saved frozen donor mapping manifest to {output_csv_path} ({len(df)} pairs)")

    return df


# ===========================================================================
# STATISTICAL EVALUATION & PAIRED STRUCTURAL ADVANTAGE (EXPERIMENT D)
# ===========================================================================
def compute_experiment_d_aggregations(
    df_raw: "pd.DataFrame",
    output_dir: Optional[str] = "results",
) -> Tuple["pd.DataFrame", "pd.DataFrame", "pd.DataFrame"]:
    r"""
    Processes Experiment D results according to the paired structural advantage protocol:
    1. Seed Aggregation:
           target_delta_CPC_mean = mean(delta_CPC_target over seeds)
           control_delta_CPC_mean = mean(delta_CPC_donor_control over seeds)
           structural_advantage_delta = target_delta_CPC_mean - control_delta_CPC_mean
    2. Primary Target-Level Summary (H_t):
           H_t = 1/49 * sum_{s != t} structural_advantage_delta_{s,t}
    3. Global Inference:
           - 10,000 bootstrap resamples on the 50 H_t values.
           - Wilcoxon signed-rank test on H_1 ... H_50 vs 0.
           - Crossed mixed-effects model: delta ~ 1 + (1|source) + (1|target).
    """
    import pandas as pd
    from scipy.stats import wilcoxon

    # 1. Seed Aggregation
    df_mean = (
        df_raw.groupby(["source_city", "target_city", "model", "K"])[
            ["delta_CPC_target", "delta_CPC_donor_control"]
        ]
        .mean()
        .reset_index()
        .rename(columns={
            "delta_CPC_target": "target_delta_CPC_mean",
            "delta_CPC_donor_control": "control_delta_CPC_mean",
        })
    )
    df_mean["structural_advantage_delta"] = (
        df_mean["target_delta_CPC_mean"] - df_mean["control_delta_CPC_mean"]
    )

    # 2. Target-Level Summary (H_t)
    target_summary_records = []
    for (target_city, model), group in df_mean.groupby(["target_city", "model"]):
        deltas = group["structural_advantage_delta"].values
        target_summary_records.append({
            "target_city": target_city,
            "model": model,
            "mean_structural_advantage": float(np.mean(deltas)),  # H_t
            "median_structural_advantage": float(np.median(deltas)),
            "n_sources": len(deltas),
            "positive_sources": int(np.sum(deltas > 0)),
        })
    df_target_summary = pd.DataFrame(target_summary_records)

    # 3. Global Inference per Model
    model_data = []
    for model, m_group in df_target_summary.groupby("model"):
        H = m_group["mean_structural_advantage"].values
        n_targets = len(H)
        assert n_targets == 50, f"Expected 50 targets, got {n_targets}"

        # Descriptive metrics
        mean_H = float(np.mean(H))
        median_H = float(np.median(H))
        q75 = float(np.quantile(H, 0.75, method="linear"))
        q25 = float(np.quantile(H, 0.25, method="linear"))
        iqr_H = float(q75 - q25)
        pos_targets = int(np.sum(H > 0))
        pos_frac = float(pos_targets / n_targets)

        # Bootstrap 95% CI (10,000 resamples on 50 H_t values, seed=42)
        rng = np.random.default_rng(42)
        boot_means = [np.mean(rng.choice(H, size=n_targets, replace=True)) for _ in range(10000)]
        ci_low = float(np.percentile(boot_means, 2.5))
        ci_high = float(np.percentile(boot_means, 97.5))

        # Locked Wilcoxon signed-rank test
        if np.all(H == 0) or len(np.unique(H)) <= 1:
            w_stat, w_p = 0.0, 1.0
        else:
            res_w = wilcoxon(x=H, alternative="two-sided", zero_method="wilcox", correction=False, method="auto")
            w_stat, w_p = float(res_w.statistic), float(res_w.pvalue)

        # Crossed mixed-effects model on pair deltas
        pair_data = df_mean[df_mean["model"] == model]
        beta0, se, ci_l, ci_h, var_s, var_t, var_eps = fit_crossed_mixed_effects(
            pair_data, outcome_col="structural_advantage_delta"
        )

        model_data.append({
            "model": model,
            "mean_H": round(mean_H, 6),
            "median_H": round(median_H, 6),
            "IQR_H": round(iqr_H, 6),
            "bootstrap_ci_low": round(ci_low, 6),
            "bootstrap_ci_high": round(ci_high, 6),
            "wilcoxon_stat": round(w_stat, 2),
            "wilcoxon_p_raw": float(w_p),
            "positive_targets": pos_targets,
            "positive_target_fraction": round(pos_frac, 6),
            "n_targets": n_targets,
            "mixed_beta0": round(beta0, 6),
            "mixed_se": round(se, 6),
            "mixed_ci_low": round(ci_l, 6),
            "mixed_ci_high": round(ci_h, 6),
            "source_variance": round(var_s, 6),
            "target_variance": round(var_t, 6),
            "residual_variance": round(var_eps, 6),
        })

    # Holm-Bonferroni correction across 3 models
    p_raws = [d["wilcoxon_p_raw"] for d in model_data]
    order = np.argsort(p_raws)
    p_holm = np.zeros(len(p_raws), dtype=np.float64)
    m_tests = len(p_raws)
    cum_max = 0.0
    for rank, orig_idx in enumerate(order):
        adj = min(1.0, p_raws[orig_idx] * (m_tests - rank))
        cum_max = max(cum_max, adj)
        p_holm[orig_idx] = cum_max

    inference_records = []
    for i, d in enumerate(model_data):
        d["wilcoxon_p_holm"] = float(p_holm[i])
        inference_records.append(d)

    df_inference = pd.DataFrame(inference_records)

    # Save outputs if path provided
    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        df_mean.to_csv(out_path / "structural_control_mean.csv", index=False)
        df_target_summary.to_csv(out_path / "structural_control_target_summary.csv", index=False)
        df_inference.to_csv(out_path / "structural_control_inference.csv", index=False)
        print(f"Saved Experiment D summary files to {out_path}")

    return df_mean, df_target_summary, df_inference


def fit_crossed_mixed_effects(
    df: "pd.DataFrame",
    outcome_col: str = "structural_advantage_delta"
) -> Tuple[float, float, float, float, float, float, float]:
    r"""
    Fits a crossed random effects model strictly adhering to protocol:
        delta ~ 1 + (1|source) + (1|target)
    Using statsmodels MixedLM with groups=_all, re_formula="0", vc_formula crossed, reml=True, lbfgs, maxiter=1000.
    Hard Failure Policy: If optimization fails to converge or errors, RAISE hard error. No silent fallbacks.
    """
    from statsmodels.regression.mixed_linear_model import MixedLM

    data = df.copy()
    data["_all"] = 1

    model = MixedLM.from_formula(
        f"{outcome_col} ~ 1",
        groups=data["_all"],
        re_formula="0",
        vc_formula={
            "source": "0 + C(source_city)",
            "target": "0 + C(target_city)",
        },
        data=data,
    )
    fit = model.fit(method="lbfgs", maxiter=1000, reml=True)
    if not fit.converged:
        raise RuntimeError(
            f"MixedLM failed to converge for outcome '{outcome_col}' after 1000 iterations (Hard Failure Policy)."
        )

    beta0 = float(fit.params["Intercept"])
    se = float(fit.bse["Intercept"])
    ci_l = beta0 - 1.95996 * se
    ci_h = beta0 + 1.95996 * se
    vcomp = fit.vcomp
    var_s = float(vcomp[0]) if len(vcomp) > 0 else 0.0
    var_t = float(vcomp[1]) if len(vcomp) > 1 else 0.0
    var_eps = float(fit.scale)
    return beta0, se, ci_l, ci_h, var_s, var_t, var_eps

```

---

<a id="implement-new-plan-calibration-statistical-inference-py"></a>
## File: `implement_new_plan/calibration/statistical_inference.py` (795 lines)

```python
"""
Statistical Inference Suite for Cross-City Zero-Shot Transfer Experiments.

Architecture:
- Tier A: Seed Aggregation (average 3 seeds per source-target-model tuple).
- Tier B: Target-Level Primary Summary Inference (50 target-city aggregated summaries G_t;
          note: summaries share source models and are not assumed to be fully independent).
- Tier C: Source-Level Transfer Summary Analysis (50 source-city aggregated summaries G_s).
- Tier D: Crossed Random-Effects Robustness Model (Delta_CPC ~ 1 + (1|source) + (1|target),
          primary robustness check accounting for crossed source and target dependence).
- Heatmap Plotting: 50x50 transfer matrix visualization (diagonal = NA).
"""

from typing import Tuple, Dict, Optional, List
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import scipy.linalg as la
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Tier A: Seed Aggregation
# ---------------------------------------------------------------------------
def aggregate_seeds(df_calibration: pd.DataFrame) -> pd.DataFrame:
    """
    Averages metric gains across exactly 3 random seeds for each (source_city, target_city, model, K).
    Enforces strict invariant: each (source, target, model, K) group must contain exactly 3 seed runs.
    From 22,050 seed-level transfer runs overall (7,350 runs per model family),
    returns 2,450 seed-averaged source-target pairs per model family.
    """
    groupby_cols = ["source_city", "target_city", "model", "K"]
    metric_cols = [
        "CPC_before", "CPC_after", "delta_CPC",
        "MAE_before", "MAE_after", "delta_MAE",
        "MSE_before", "MSE_after", "delta_MSE"
    ]
    avail_metrics = [c for c in metric_cols if c in df_calibration.columns]

    # Verify that each pair has exactly 3 seed runs for stochastic models
    counts = df_calibration.groupby(groupby_cols).size()
    if not (counts == 3).all():
        bad_groups = counts[counts != 3]
        raise ValueError(f"Seed aggregation error: {len(bad_groups)} groups do not have exactly 3 seeds. "
                         f"Example: {bad_groups.head()}")

    agg_df = df_calibration.groupby(groupby_cols)[avail_metrics].agg(["mean", "std"]).reset_index()
    flat_cols = []
    for col in agg_df.columns:
        if col[1] == "mean" or col[1] == "":
            flat_cols.append(col[0])
        else:
            flat_cols.append(f"{col[0]}_{col[1]}")
    agg_df.columns = flat_cols
    return agg_df


# ---------------------------------------------------------------------------
# Tier B: Target-Level Analysis (Separating City Summary and Global Inference)
# ---------------------------------------------------------------------------
def compute_target_city_summary(
    df_seed_averaged: pd.DataFrame,
    metric_col: str = "delta_CPC",
) -> pd.DataFrame:
    r"""
    Level 1: Target-City Descriptive Summary.
    Computes descriptive statistics for each target city across its 49 source models.
        G_t = 1/49 * sum_{s != t} Delta_CPC_{s,t}
        
    Schema:
        target_city, model, G_t, median_source_gain, IQR_source_gain, std_source_gain,
        n_sources, positive_sources, positive_source_fraction
        
    Strict Invariant:
    - Exactly 50 rows per model.
    - Zero p-values, zero global bootstrap CIs in individual city rows.
    """
    models = df_seed_averaged["model"].unique()
    all_rows = []

    for m in models:
        m_df = df_seed_averaged[df_seed_averaged["model"] == m]
        for target_city, group in m_df.groupby("target_city"):
            gains = group[metric_col].values
            n_sources = len(gains)
            assert n_sources == 49, f"Target {target_city} has {n_sources} sources, expected 49"

            g_t = float(np.mean(gains))
            med = float(np.median(gains))
            q75, q25 = np.percentile(gains, [75, 25])
            iqr = float(q75 - q25)
            std_val = float(np.std(gains, ddof=1)) if n_sources > 1 else 0.0
            pos = int(np.sum(gains > 0))
            pos_frac = float(pos / n_sources)

            all_rows.append({
                "target_city": target_city,
                "model": m,
                "G_t": round(g_t, 6),
                "median_source_gain": round(med, 6),
                "IQR_source_gain": round(iqr, 6),
                "std_source_gain": round(std_val, 6),
                "n_sources": n_sources,
                "positive_sources": pos,
                "positive_source_fraction": round(pos_frac, 6),
            })

    return pd.DataFrame(all_rows)


def compute_global_target_inference(
    df_target_city_summary: pd.DataFrame,
    n_bootstraps: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    r"""
    Level 2: Global Across-Target Statistical Inference (Tier B).
    Operates on the 50 aggregated G_t values for each model:
        G_t = 1/49 * \sum_{s != t} \bar{\Delta CPC}_{s,t}

    Roles & Scientific Interpretation Rules:
    - Bootstrap (10,000 resamples on 50 G_t):
      Serves strictly as a *target-level descriptive uncertainty summary* reflecting
      empirical variability among target summaries.
      DO NOT describe as "cross-dependence-adjusted CI" or "cluster-robust CI".
      DO NOT describe as removing or accounting for source-target dependence.
    - Wilcoxon signed-rank test (on 50 G_t vs 0):
      Serves strictly as a *target-level summary test* evaluating whether the distribution
      of aggregated target gains systematically deviates from zero.
      DO NOT describe as accounting for dependence due to shared source models.
    - Independence Assumption:
      The 50 target summaries share source models and are NOT independent observations.
      Tier D crossed mixed-effects model (\Delta CPC ~ 1 + (1|source) + (1|target))
      is the primary dependence-aware robustness analysis addressing crossed dependence.
    - Seed:
      Fixed statistical seed (default seed=42) pre-specified and held constant across models.

    Schema:
        model, n_targets, mean_G, median_G, IQR_G, bootstrap_ci_low, bootstrap_ci_high,
        wilcoxon_stat, wilcoxon_p, positive_targets, positive_target_fraction

    Strict Invariants:
    - Exactly ONE row per model.
    - Resampling unit is strictly the 50 target cities (resamples 50 G_t values with replacement).
    - Never bootstrap 2,450 pair rows as independent observations.
    - Wilcoxon test is strictly Wilcoxon(G_1 ... G_50, 0).
    """
    models = df_target_city_summary["model"].unique()
    rng = np.random.default_rng(seed)
    rows = []

    # Compute statistics and raw p-values for all models
    model_data = []
    for m in models:
        m_df = df_target_city_summary[df_target_city_summary["model"] == m]
        g_t = m_df["G_t"].values
        n_targets = len(g_t)
        assert n_targets == 50, f"Expected 50 targets, got {n_targets}"

        mean_G = float(np.mean(g_t))
        median_G = float(np.median(g_t))
        q75 = float(np.quantile(g_t, 0.75, method="linear"))
        q25 = float(np.quantile(g_t, 0.25, method="linear"))
        iqr_G = float(q75 - q25)

        # Bootstrap 95% CI on the 50 G_t values
        boot_means = [np.mean(rng.choice(g_t, size=n_targets, replace=True)) for _ in range(n_bootstraps)]
        ci_low = float(np.percentile(boot_means, 2.5))
        ci_high = float(np.percentile(boot_means, 97.5))

        # Locked Wilcoxon signed-rank test on 50 G_t values vs 0
        if np.all(g_t == 0) or len(np.unique(g_t)) <= 1:
            w_stat, p_val = 0.0, 1.0
        else:
            res_w = stats.wilcoxon(x=g_t, alternative="two-sided", zero_method="wilcox", correction=False, method="auto")
            w_stat, p_val = float(res_w.statistic), float(res_w.pvalue)

        pos_targets = int(np.sum(g_t > 0))
        pos_frac = float(pos_targets / n_targets)

        model_data.append({
            "model": m,
            "n_targets": n_targets,
            "mean_G": round(mean_G, 6),
            "median_G": round(median_G, 6),
            "IQR_G": round(iqr_G, 6),
            "bootstrap_ci_low": round(ci_low, 6),
            "bootstrap_ci_high": round(ci_high, 6),
            "wilcoxon_stat": round(w_stat, 2),
            "wilcoxon_p_raw": float(p_val),
            "positive_targets": pos_targets,
            "positive_target_fraction": round(pos_frac, 6),
        })

    # Holm-Bonferroni correction across models
    p_raws = [d["wilcoxon_p_raw"] for d in model_data]
    order = np.argsort(p_raws)
    p_holm = np.zeros(len(p_raws), dtype=np.float64)
    m_tests = len(p_raws)
    cum_max = 0.0
    for rank, orig_idx in enumerate(order):
        adj = min(1.0, p_raws[orig_idx] * (m_tests - rank))
        cum_max = max(cum_max, adj)
        p_holm[orig_idx] = cum_max

    for i, d in enumerate(model_data):
        d["wilcoxon_p_holm"] = float(p_holm[i])
        rows.append(d)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tier C: Source-Level Transfer Summary Analysis
# ---------------------------------------------------------------------------
def compute_source_city_summary(
    df_seed_averaged: pd.DataFrame,
    metric_col: str = "delta_CPC",
) -> pd.DataFrame:
    r"""
    Tier C: Source-City Descriptive Summary.
    Computes descriptive statistics for each source city across its 49 target cities:
        G_s = 1/49 * \sum_{t != s} \bar{\Delta CPC}_{s,t}

    Roles & Scientific Interpretation Rules:
    - Purely a supporting descriptive diagnostic to evaluate source heterogeneity
      and check whether calibration gains are dominated by specific source models.
    - NOT a separate experiment (no Experiment F) and NOT a confirmatory hypothesis test.
    - No correlation analysis (e.g. held-out performance vs G_s) in main protocol.
    - No causal inferences (e.g. "better source models cause larger DBD gains").

    Schema:
        source_city, model, G_s, median_target_gain, IQR_target_gain, std_target_gain,
        n_targets, positive_targets, positive_target_fraction
    """
    models = df_seed_averaged["model"].unique()
    all_rows = []

    for m in models:
        m_df = df_seed_averaged[df_seed_averaged["model"] == m]
        for source_city, group in m_df.groupby("source_city"):
            gains = group[metric_col].values
            n_targets = len(gains)
            assert n_targets == 49, f"Source {source_city} has {n_targets} targets, expected 49"

            g_s = float(np.mean(gains))
            med = float(np.median(gains))
            q75, q25 = np.percentile(gains, [75, 25])
            iqr = float(q75 - q25)
            std_val = float(np.std(gains, ddof=1)) if n_targets > 1 else 0.0
            pos = int(np.sum(gains > 0))
            pos_frac = float(pos / n_targets)

            all_rows.append({
                "source_city": source_city,
                "model": m,
                "G_s": round(g_s, 6),
                "median_target_gain": round(med, 6),
                "IQR_target_gain": round(iqr, 6),
                "std_target_gain": round(std_val, 6),
                "n_targets": n_targets,
                "positive_targets": pos,
                "positive_target_fraction": round(pos_frac, 6),
            })

    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Tier D: Crossed Source-Target Random-Effects Robustness Model
# ---------------------------------------------------------------------------
def fit_crossed_random_effects(
    df_seed_averaged: pd.DataFrame,
    metric_col: str = "delta_CPC",
    optimizer_maxiter: int = 1000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    r"""
    Fits the canonical crossed source-target random-effects model using statsmodels MixedLM with REML:
        \Delta CPC_{s,t} = \beta_0 + u_s + v_t + \epsilon_{s,t}
        u_s ~ N(0, \sigma_s^2), v_t ~ N(0, \sigma_t^2), \epsilon_{s,t} ~ N(0, \sigma_\epsilon^2)

    Strict Invariants:
    1. Input Data Contract:
       - Exactly 2,450 source-target pairs per model.
       - 50 unique source cities, 50 unique target cities.
       - No diagonal observations (source != target).
       - Zero missing values in metric_col.
    2. Fit separately per model family:
       - gravity_2param, pairwise_mlp, urban_gnn. Never pool across models.
    3. Estimation:
       - statsmodels MixedLM.from_formula with groups=_all, re_formula="0",
         vc_formula={"source": "0 + C(source_city)", "target": "0 + C(target_city)"},
         method="lbfgs", reml=True, maxiter=1000.
    4. Hard Failure Convergence Policy:
       - Raises RuntimeError if optimizer fails to converge. No silent fallback,
         no dropping random effects, no switching optimizers.
    5. Singular / Near-Zero Variance Preservation:
       - Never drops random intercepts even if variance is near boundary.
    6. Outputs:
       - crossed_effects_results.csv (1 row per model)
       - crossed_effects_diagnostics.csv (1 row per model)
    """
    models = df_seed_averaged["model"].unique()
    results_rows = []
    diag_rows = []

    from statsmodels.regression.mixed_linear_model import MixedLM

    for m in models:
        sub = df_seed_averaged[df_seed_averaged["model"] == m].copy()

        # 1. Input data validation
        n_pairs = len(sub)
        if n_pairs != 2450:
            raise ValueError(f"Model {m} has {n_pairs} observations, expected exactly 2450.")
        if sub["source_city"].nunique() != 50:
            raise ValueError(f"Model {m} has {sub['source_city'].nunique()} source cities, expected 50.")
        if sub["target_city"].nunique() != 50:
            raise ValueError(f"Model {m} has {sub['target_city'].nunique()} target cities, expected 50.")
        if (sub["source_city"] == sub["target_city"]).any():
            raise ValueError(f"Model {m} contains diagonal entries (source == target).")
        if sub[metric_col].isna().any():
            raise ValueError(f"Model {m} has missing values in {metric_col}.")

        # 2. Canonical Crossed Random Effects MixedLM
        sub["_all"] = 1
        model = MixedLM.from_formula(
            f"{metric_col} ~ 1",
            groups=sub["_all"],
            re_formula="0",
            vc_formula={
                "source": "0 + C(source_city)",
                "target": "0 + C(target_city)",
            },
            data=sub,
        )

        fit = model.fit(method="lbfgs", maxiter=optimizer_maxiter, reml=True)

        # 3. Strict Convergence Check (Hard Failure Policy)
        if not fit.converged:
            raise RuntimeError(f"Mixed-effects optimizer failed to converge for model {m} after {optimizer_maxiter} iterations.")

        beta0_hat = float(fit.params["Intercept"])
        se_beta0 = float(fit.bse["Intercept"])
        ci_low = float(beta0_hat - 1.95996 * se_beta0)
        ci_high = float(beta0_hat + 1.95996 * se_beta0)

        vcomp = fit.vcomp
        sigma_s2_hat = float(vcomp[0]) if len(vcomp) > 0 else 0.0
        sigma_t2_hat = float(vcomp[1]) if len(vcomp) > 1 else 0.0
        sigma_e2_hat = float(fit.scale)
        log_lik = float(fit.llf)

        # Diagnostics & warnings detection
        singular_warn = bool(se_beta0 < 1e-12 or np.isnan(se_beta0))
        boundary_warn = bool(sigma_s2_hat < 1e-8 or sigma_t2_hat < 1e-8)
        hessian_warn = False
        other_warn = ""
        warnings_list = []
        if boundary_warn:
            warnings_list.append("boundary_variance")
        if singular_warn:
            warnings_list.append("singular_covariance")

        warnings_str = ";".join(warnings_list) if warnings_list else "none"

        results_rows.append({
            "model": m,
            "n_pairs": n_pairs,
            "estimation_method": "REML",
            "optimizer": "lbfgs",
            "converged": True,
            "beta0": round(beta0_hat, 6),
            "se_beta0": round(se_beta0, 6),
            "ci_low": round(ci_low, 6),
            "ci_high": round(ci_high, 6),
            "source_variance": round(sigma_s2_hat, 8),
            "target_variance": round(sigma_t2_hat, 8),
            "residual_variance": round(sigma_e2_hat, 8),
            "log_likelihood": round(log_lik, 4),
            "warnings": warnings_str,
        })

        diag_rows.append({
            "model": m,
            "converged": True,
            "n_iterations": int(fit.iterations if hasattr(fit, "iterations") else 0),
            "singular_warning": singular_warn,
            "boundary_warning": boundary_warn,
            "hessian_warning": hessian_warn,
            "other_warning": other_warn if other_warn else "none",
        })

    return pd.DataFrame(results_rows), pd.DataFrame(diag_rows)


# ---------------------------------------------------------------------------
# Tier Figure: 50x50 Transfer Matrix Heatmap
# ---------------------------------------------------------------------------
def plot_transfer_heatmap(
    df_seed_averaged: pd.DataFrame,
    output_path: Path,
    model_name: str = "GNN",
    metric_col: str = "delta_CPC",
    figsize: Tuple[int, int] = (16, 14)
):
    """
    Plots a 50x50 heatmap with source cities on rows, target cities on columns,
    and seed-averaged delta_CPC on cells (diagonal = NA).
    """
    sub = df_seed_averaged[df_seed_averaged["model"] == model_name]
    matrix = sub.pivot(index="source_city", columns="target_city", values=metric_col)
    
    # Fill diagonal with NaN for visual separation
    for c in matrix.index:
        if c in matrix.columns:
            matrix.loc[c, c] = np.nan

    plt.figure(figsize=figsize)
    data = matrix.values
    masked_data = np.ma.masked_invalid(data)
    
    cmap = plt.cm.coolwarm.copy()
    cmap.set_bad(color="lightgray")
    
    im = plt.imshow(masked_data, cmap=cmap, aspect="auto", interpolation="nearest")
    cbar = plt.colorbar(im)
    cbar.set_label(f"Seed-Averaged {metric_col}", fontsize=12)
    
    plt.title(f"Cross-City Transfer Gain Heatmap (50x50 Matrix, {model_name})", fontsize=14, pad=15)
    plt.xlabel("Target City (Unseen)", fontsize=12)
    plt.ylabel("Source City (30% Training Data)", fontsize=12)
    plt.xticks(ticks=range(len(matrix.columns)), labels=matrix.columns, rotation=90, fontsize=8)
    plt.yticks(ticks=range(len(matrix.index)), labels=matrix.index, rotation=0, fontsize=8)
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved heatmap to: {output_path}")


# ---------------------------------------------------------------------------
# Tier RQ2: Experiment B Adjacent Contrasts & Scarcity Inference
# ---------------------------------------------------------------------------

ADJACENT_CONTRASTS = [
    ("10_vs_20", 0.10, 0.20),
    ("20_vs_30", 0.20, 0.30),
    ("30_vs_50", 0.30, 0.50),
    ("50_vs_100", 0.50, 1.00),
]

def compute_scarcity_contrasts(
    df_scarcity_mean: pd.DataFrame,
    metric_col: str = "delta_CPC_mean",
) -> pd.DataFrame:
    r"""
    Computes 4 adjacent paired contrasts for Experiment B:
        C^{a-b}_{s,t} = g_{s,t,a} - g_{s,t,b}
    where contrast_name in {"10_vs_20", "20_vs_30", "30_vs_50", "50_vs_100"}.
    
    Input: df_scarcity_mean with (source_city, target_city, model, train_fraction, delta_CPC_mean).
    Output: DataFrame with (source_city, target_city, model, contrast_name, contrast_value).
    Total rows = 2450 pairs * 3 models * 4 contrasts = 29,400 rows.
    """
    piv = df_scarcity_mean.pivot(
        index=["source_city", "target_city", "model"],
        columns="train_fraction",
        values=metric_col
    ).reset_index()

    records = []
    for contrast_name, f_a, f_b in ADJACENT_CONTRASTS:
        diff = piv[f_a] - piv[f_b]
        for idx, row in piv.iterrows():
            records.append({
                "source_city": row["source_city"],
                "target_city": row["target_city"],
                "model": row["model"],
                "contrast_name": contrast_name,
                "contrast_value": float(diff.iloc[idx]),
            })
    return pd.DataFrame(records)


def compute_scarcity_contrast_target_summary(
    df_contrasts: pd.DataFrame
) -> pd.DataFrame:
    r"""
    Computes target-level descriptive summary for 4 adjacent contrasts:
        C_t^{a-b} = 1/49 * sum_{s != t} C_{s,t}^{a-b}
    Schema:
        target_city, model, contrast_name, mean_contrast, median_contrast, IQR_contrast, std_contrast, n_sources, positive_sources
    Expected rows: 50 targets * 3 models * 4 contrasts = 600 rows.
    """
    rows = []
    for (m, c_name), group_mc in df_contrasts.groupby(["model", "contrast_name"]):
        for target_city, group in group_mc.groupby("target_city"):
            vals = group["contrast_value"].values
            n_sources = len(vals)
            assert n_sources == 49, f"Target {target_city} has {n_sources} sources, expected 49"

            m_val = float(np.mean(vals))
            med_val = float(np.median(vals))
            q75, q25 = np.percentile(vals, [75, 25])
            iqr_val = float(q75 - q25)
            std_val = float(np.std(vals, ddof=1)) if n_sources > 1 else 0.0
            pos_sources = int(np.sum(vals > 0))

            rows.append({
                "target_city": target_city,
                "model": m,
                "contrast_name": c_name,
                "mean_contrast": round(m_val, 6),
                "median_contrast": round(med_val, 6),
                "IQR_contrast": round(iqr_val, 6),
                "std_contrast": round(std_val, 6),
                "n_sources": n_sources,
                "positive_sources": pos_sources,
            })
    return pd.DataFrame(rows)


def compute_scarcity_contrast_inference(
    df_contrast_target_summary: pd.DataFrame,
    n_bootstraps: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    r"""
    Global across-target statistical inference for 4 adjacent contrasts with Holm correction:
    Operates on 50 aggregated C_t^{a-b} values.
    Schema:
        model, contrast_name, n_targets, mean_contrast, median_contrast, IQR_contrast,
        bootstrap_ci_low, bootstrap_ci_high, wilcoxon_stat, wilcoxon_p_raw, wilcoxon_p_holm,
        positive_targets, positive_target_fraction
    Expected rows: 3 models * 4 contrasts = 12 rows.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for m in df_contrast_target_summary["model"].unique():
        m_df = df_contrast_target_summary[df_contrast_target_summary["model"] == m]
        raw_p_list = []
        model_rows = []

        for c_name in ["10_vs_20", "20_vs_30", "30_vs_50", "50_vs_100"]:
            c_df = m_df[m_df["contrast_name"] == c_name]
            vals = c_df["mean_contrast"].values
            n_targets = len(vals)
            assert n_targets == 50, f"Expected 50 targets for {m}-{c_name}, got {n_targets}"

            mean_c = float(np.mean(vals))
            median_c = float(np.median(vals))
            q75 = float(np.quantile(vals, 0.75, method="linear"))
            q25 = float(np.quantile(vals, 0.25, method="linear"))
            iqr_c = float(q75 - q25)

            boot_means = [np.mean(rng.choice(vals, size=n_targets, replace=True)) for _ in range(n_bootstraps)]
            ci_low = float(np.percentile(boot_means, 2.5))
            ci_high = float(np.percentile(boot_means, 97.5))

            if np.all(vals == 0) or len(np.unique(vals)) <= 1:
                w_stat, p_raw = 0.0, 1.0
            else:
                res_w = stats.wilcoxon(x=vals, alternative="two-sided", zero_method="wilcox", correction=False, method="auto")
                w_stat, p_raw = float(res_w.statistic), float(res_w.pvalue)

            raw_p_list.append(p_raw)
            pos_targets = int(np.sum(vals > 0))
            pos_frac = float(pos_targets / n_targets)

            model_rows.append({
                "model": m,
                "contrast_name": c_name,
                "n_targets": n_targets,
                "mean_contrast": round(mean_c, 6),
                "median_contrast": round(median_c, 6),
                "IQR_contrast": round(iqr_c, 6),
                "bootstrap_ci_low": round(ci_low, 6),
                "bootstrap_ci_high": round(ci_high, 6),
                "wilcoxon_stat": round(w_stat, 2),
                "wilcoxon_p_raw": float(p_raw),
                "positive_targets": pos_targets,
                "positive_target_fraction": round(pos_frac, 6),
            })

        # Holm-Bonferroni correction over the 4 contrasts for this model
        p_raw_arr = np.array(raw_p_list)
        order = np.argsort(p_raw_arr)
        p_holm = np.zeros(len(p_raw_arr), dtype=np.float64)
        m_tests = len(p_raw_arr)  # 4
        cum_max = 0.0
        for rank, orig_idx in enumerate(order):
            adj = min(1.0, p_raw_arr[orig_idx] * (m_tests - rank))
            cum_max = max(cum_max, adj)
            p_holm[orig_idx] = cum_max

        for r_idx, r in enumerate(model_rows):
            r["wilcoxon_p_holm"] = float(p_holm[r_idx])
            rows.append(r)

    return pd.DataFrame(rows)


def fit_scarcity_contrast_mixed_effects(
    df_contrasts: pd.DataFrame
) -> pd.DataFrame:
    r"""
    Fits crossed random-effects models for each adjacent contrast on 2450 pairs:
        C_{s,t} = beta0 + u_s + v_t + epsilon_{s,t}
    Schema:
        model, contrast_name, n_pairs, estimation_method, optimizer, converged, beta0, se_beta0, ci_low, ci_high,
        source_variance, target_variance, residual_variance, log_likelihood, warnings
    Expected rows: 3 models * 4 contrasts = 12 rows.
    """
    results = []
    for (m, c_name), group in df_contrasts.groupby(["model", "contrast_name"]):
        # reuse fit_crossed_random_effects logic
        res_df, _ = fit_crossed_random_effects(group, metric_col="contrast_value")
        row = res_df.iloc[0].to_dict()
        row["contrast_name"] = c_name
        results.append(row)
    return pd.DataFrame(results)


def compute_gap_recovery(
    df_scarcity_results_mean: pd.DataFrame
) -> pd.DataFrame:
    r"""
    Computes GapRecovery for f in {0.10, 0.20, 0.30, 0.50, 1.00}:
        GapRecovery(f) = (CPC_{f+DBD} - CPC_f) / (CPC_100 - CPC_f)
    Strict rules:
    - For f = 1.00: gap_recovery = NaN, gap_recovery_valid = False, invalid_reason = 'full_supervision_reference'
    - For f in {0.10, 0.20, 0.30, 0.50}: valid if denominator > 0, else NaN with reason.
    Total rows = 2450 pairs * 3 models * 5 fractions = 36,750 rows.
    """
    piv_before = df_scarcity_results_mean.pivot(
        index=["source_city", "target_city", "model"],
        columns="train_fraction",
        values="CPC_before_mean"
    )
    piv_after = df_scarcity_results_mean.pivot(
        index=["source_city", "target_city", "model"],
        columns="train_fraction",
        values="CPC_after_mean"
    )

    rows = []
    for (s, t, m), row_before in piv_before.iterrows():
        row_after = piv_after.loc[(s, t, m)]
        cpc_full = float(row_before[1.00])

        for f in [0.10, 0.20, 0.30, 0.50, 1.00]:
            cpc_frac_base = float(row_before[f])
            cpc_frac_cal = float(row_after[f])
            gap_denom = cpc_full - cpc_frac_base

            if f == 1.00:
                gap_rec = np.nan
                valid = False
                inv_reason = "full_supervision_reference"
            elif not np.isfinite(gap_denom):
                gap_rec = np.nan
                valid = False
                inv_reason = "non_finite_denominator"
            elif gap_denom <= 0.0:
                gap_rec = np.nan
                valid = False
                inv_reason = "zero_or_negative_denominator"
            else:
                gap_rec = (cpc_frac_cal - cpc_frac_base) / gap_denom
                valid = True
                inv_reason = "none"

            rows.append({
                "source_city": s,
                "target_city": t,
                "model": m,
                "train_fraction": f,
                "baseline_CPC_fraction": round(cpc_frac_base, 6),
                "calibrated_CPC_fraction": round(cpc_frac_cal, 6),
                "baseline_CPC_full": round(cpc_full, 6),
                "gap_denominator": round(gap_denom, 6),
                "gap_recovery": round(gap_rec, 6) if np.isfinite(gap_rec) else np.nan,
                "gap_recovery_valid": valid,
                "invalid_reason": inv_reason,
            })
    return pd.DataFrame(rows)


def fit_scarcity_overall_mixed_effects(
    df_scarcity_mean: pd.DataFrame
) -> pd.DataFrame:
    r"""
    Fits overall dependence-aware scarcity mixed-effects model across all 5 fractions
    with train_fraction as categorical factor and locked reference level f=1.00:
        \Delta CPC_{s,t,f} = \beta_0 + \sum_{f \in {0.10, 0.20, 0.30, 0.50}} \beta_f * I(frac=f) + u_s + v_t + \epsilon_{s,t,f}
    
    Schema:
        model, n_obs, estimation_method, optimizer, converged, beta0, beta_10, beta_20, beta_30, beta_50,
        se_beta0, se_beta10, se_beta20, se_beta30, se_beta50, p_beta10, p_beta20, p_beta30, p_beta50,
        source_variance, target_variance, residual_variance, log_likelihood, warnings
    Expected rows: exactly 3 rows (1 per model family).
    """
    from statsmodels.regression.mixed_linear_model import MixedLM
    results = []

    for m in df_scarcity_mean["model"].unique():
        sub = df_scarcity_mean[df_scarcity_mean["model"] == m].copy()
        sub["_all"] = 1
        # Create categorical fraction with 1.00 as reference
        sub["frac_cat"] = sub["train_fraction"].astype(str)
        # Using formula with reference level '1.0' or '1.00'
        # Convert to categorical with explicit order: ['1.0', '0.1', '0.2', '0.3', '0.5']
        frac_levels = ['1.0', '0.1', '0.2', '0.3', '0.5']
        sub["frac_cat"] = pd.Categorical(sub["train_fraction"].astype(str), categories=frac_levels, ordered=False)

        model = MixedLM.from_formula(
            "delta_CPC_mean ~ C(frac_cat, Treatment(reference='1.0'))",
            groups=sub["_all"],
            re_formula="0",
            vc_formula={
                "source": "0 + C(source_city)",
                "target": "0 + C(target_city)",
            },
            data=sub,
        )
        fit = model.fit(method="lbfgs", maxiter=1000, reml=True)
        if not fit.converged:
            raise RuntimeError(
                f"MixedLM failed to converge for scarcity overall model on {m} after 1000 iterations (Hard Failure Policy)."
            )
        converged = bool(fit.converged)
        loglik = float(fit.llf)
        params = fit.params
        bse = fit.bse
        pvals = fit.pvalues
        vcomps = fit.vcomp
        scale = float(fit.scale)

        beta0 = float(params.get("Intercept", 0.0))
        se_beta0 = float(bse.get("Intercept", 0.0))

        beta10 = float(params.get("C(frac_cat, Treatment(reference='1.0'))[T.0.1]", 0.0))
        beta20 = float(params.get("C(frac_cat, Treatment(reference='1.0'))[T.0.2]", 0.0))
        beta30 = float(params.get("C(frac_cat, Treatment(reference='1.0'))[T.0.3]", 0.0))
        beta50 = float(params.get("C(frac_cat, Treatment(reference='1.0'))[T.0.5]", 0.0))

        se_beta10 = float(bse.get("C(frac_cat, Treatment(reference='1.0'))[T.0.1]", 0.0))
        se_beta20 = float(bse.get("C(frac_cat, Treatment(reference='1.0'))[T.0.2]", 0.0))
        se_beta30 = float(bse.get("C(frac_cat, Treatment(reference='1.0'))[T.0.3]", 0.0))
        se_beta50 = float(bse.get("C(frac_cat, Treatment(reference='1.0'))[T.0.5]", 0.0))

        p_beta10 = float(pvals.get("C(frac_cat, Treatment(reference='1.0'))[T.0.1]", 1.0))
        p_beta20 = float(pvals.get("C(frac_cat, Treatment(reference='1.0'))[T.0.2]", 1.0))
        p_beta30 = float(pvals.get("C(frac_cat, Treatment(reference='1.0'))[T.0.3]", 1.0))
        p_beta50 = float(pvals.get("C(frac_cat, Treatment(reference='1.0'))[T.0.5]", 1.0))

        var_s = float(vcomps[0]) if len(vcomps) > 0 else 0.0
        var_t = float(vcomps[1]) if len(vcomps) > 1 else 0.0
        warnings = "boundary_variance" if (var_s < 1e-8 or var_t < 1e-8) else "none"

        results.append({
            "model": m,
            "n_obs": len(sub),
            "estimation_method": "REML",
            "optimizer": "lbfgs",
            "converged": converged,
            "beta0": round(beta0, 6),
            "beta_10": round(beta10, 6),
            "beta_20": round(beta20, 6),
            "beta_30": round(beta30, 6),
            "beta_50": round(beta50, 6),
            "se_beta0": round(se_beta0, 6),
            "se_beta10": round(se_beta10, 6),
            "se_beta20": round(se_beta20, 6),
            "se_beta30": round(se_beta30, 6),
            "se_beta50": round(se_beta50, 6),
            "p_beta10": float(p_beta10),
            "p_beta20": float(p_beta20),
            "p_beta30": float(p_beta30),
            "p_beta50": float(p_beta50),
            "source_variance": round(var_s, 8),
            "target_variance": round(var_t, 8),
            "residual_variance": round(scale, 8),
            "log_likelihood": round(loglik, 4),
            "warnings": warnings,
        })

    return pd.DataFrame(results)

```

---

<a id="implement-new-plan-data-dataset-py"></a>
## File: `implement_new_plan/data/dataset.py` (523 lines)

```python
"""
City dataset loader for the distance-binned OD reconstruction study.

Loads a single city's data from the standard directory layout:
    data/{city}/
        meta.csv               — tract metadata (idx, lon, lat, area_km2, city)
        nodes/
            census.csv         — population, income, employment features
            poi.csv            — POI density features
            road.csv           — road network features
        pairs/
            od.csv             — candidate OD pairs with trip_count
            distance.csv       — pairwise distances (km) for same candidate set

Returns a CityData dataclass with:
    node_features  : FloatTensor (N, F)
    pair_o_idx     : LongTensor  (E,)     — origin tract index
    pair_d_idx     : LongTensor  (E,)     — destination tract index
    pair_distance  : FloatTensor (E,)     — distance in km
    pair_trips     : FloatTensor (E,)     — trip count (all >= 1)
    population     : FloatTensor (N,)     — total_population per tract
    lon_lat        : FloatTensor (N, 2)   — centroid coordinates
    city_name      : str
    n_tracts       : int
    n_pairs        : int

Normalization:
    - Node features: StandardScaler fitted on training cities, applied to all.
    - Distances: log(1 + d_km).
    - Trip counts: kept as raw integers (ZTNB operates on counts directly).
"""

from __future__ import annotations

import os
import csv
import hashlib
import dataclasses
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Node feature columns (order must match across all cities)
# ---------------------------------------------------------------------------

# Census features used (subset — well-defined across all 50 cities)
CENSUS_COLS = [
    "total_population", "median_age", "median_income", "per_capita_income",
    "employment_rate", "unemployment_rate", "commute_transit_pct",
    "commute_active_pct", "commute_wfh_pct", "zero_vehicle_pct",
    "avg_vehicles_per_household", "higher_education_pct", "homeownership_rate",
]

# POI features
POI_COLS = [
    "office", "office_density", "industrial", "industrial_density",
    "commercial", "commercial_density", "education_primary",
    "education_primary_density",
]

# Road features
ROAD_COLS = [
    "road_length_total", "road_density", "road_count",
    "motorway_length", "primary_length",
]

NODE_FEATURE_COLUMNS = tuple(CENSUS_COLS + POI_COLS + ROAD_COLS)


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class CityData:
    city_name:      str
    n_tracts:       int
    n_pairs:        int

    # Node-level (N, *)
    node_features:  torch.Tensor   # (N, F) normalized
    population:     torch.Tensor   # (N,)   raw population
    lon_lat:        torch.Tensor   # (N, 2) [lon, lat]

    # Pair-level (E, *)
    pair_o_idx:     torch.LongTensor   # (E,)
    pair_d_idx:     torch.LongTensor   # (E,)
    pair_distance:  torch.Tensor       # (E,) log1p(km)
    pair_trips:     torch.Tensor       # (E,) raw counts, all >= 1
    bin_labels:     torch.LongTensor   # (E,) distance bin index (0-3)

    # Canonical distances for binning; expm1(pair_distance) is NOT bit-identical to this.
    dist_km:        Optional[np.ndarray] = None   # (E,) raw pairwise distance in km


# ---------------------------------------------------------------------------
# Distance bin assignment
# ---------------------------------------------------------------------------
# Bins match Meta mobility categories: 0 km | (0,10) | [10,100) | 100+
BIN_EDGES = [0.0, 1e-9, 10.0, 100.0, float("inf")]
BIN_LABELS = ["zero", "short", "medium", "long"]   # 0, 1, 2, 3

def assign_bins(distance_km: np.ndarray) -> np.ndarray:
    """Assign each pair to a distance bin (0=zero, 1=short, 2=medium, 3=long)."""
    bins = np.zeros(len(distance_km), dtype=np.int64)
    bins[(distance_km > 0)   & (distance_km < 10)]  = 1
    bins[(distance_km >= 10) & (distance_km < 100)] = 2
    bins[distance_km >= 100]                         = 3
    return bins


# ---------------------------------------------------------------------------
# CSV loading helpers
# ---------------------------------------------------------------------------

def _load_csv_columns(path: Path, cols: List[str], key_col: str = "idx") -> np.ndarray:
    """Load specific columns from a CSV, ordered by key_col. Returns float array."""
    data: Dict[int, List[float]] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = int(row[key_col])
            vals = []
            for c in cols:
                v = row.get(c, "0") or "0"
                try:
                    vals.append(float(v))
                except ValueError:
                    vals.append(0.0)
            data[key] = vals
    if not data:
        return np.zeros((0, len(cols)), dtype=np.float32)
    keys = sorted(data.keys())
    n = max(keys) + 1
    if keys != list(range(n)):
        raise ValueError(f"Feature CSV {path} has missing indices. Expected 0 to {n-1}.")
    arr = np.zeros((n, len(cols)), dtype=np.float32)
    for k, v in data.items():
        arr[k] = v
    return arr


def _load_meta(path: Path):
    """Load meta.csv -> idx, lon, lat, population placeholder."""
    idx_list, lons, lats = [], [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx_list.append(int(row["idx"]))
            lons.append(float(row["lon"]))
            lats.append(float(row["lat"]))
    if not idx_list:
        return np.zeros((0, 2), dtype=np.float32)
    n = max(idx_list) + 1
    if sorted(idx_list) != list(range(n)):
        raise ValueError(f"Meta CSV {path} has missing indices. Expected 0 to {n-1}.")
    lon_arr = np.zeros(n, dtype=np.float32)
    lat_arr = np.zeros(n, dtype=np.float32)
    for i, lon, lat in zip(idx_list, lons, lats):
        lon_arr[i] = lon
        lat_arr[i] = lat
    return np.stack([lon_arr, lat_arr], axis=1)   # (N, 2)


def _load_pairs(od_path: Path, dist_path: Path):
    """Load od.csv and distance.csv into aligned arrays."""
    od: Dict[tuple, int] = {}
    with open(od_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trip = int(row["trip_count"])
            if trip > 0:
                od[(int(row["o_idx"]), int(row["d_idx"]))] = trip

    dist_map: Dict[tuple, float] = {}
    with open(dist_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dist_map[(int(row["o_idx"]), int(row["d_idx"]))] = float(row["distance_km"])
            
    od_keys = set(od.keys())
    dist_keys = set(dist_map.keys())
    
    missing_dist = od_keys - dist_keys
    if len(missing_dist) > 0:
        raise ValueError(f"Found {len(missing_dist)} positive OD pairs missing from distance.csv (e.g. {list(missing_dist)[:3]}). Support integrity compromised.")

    # Iterate over distance pairs that have positive OD trips
    origins, dests, trips, dists = [], [], [], []
    for pair in dist_keys:
        trip_count = od.get(pair)
        if trip_count is None:
            # Pair has distance but trip=0 or missing OD, which is fine (zero-trip pairs are ignored in GNN but safe to skip for support)
            continue
        origins.append(pair[0])
        dests.append(pair[1])
        trips.append(trip_count)
        dists.append(dist_map[pair])

    return (
        np.array(origins, dtype=np.int64),
        np.array(dests,   dtype=np.int64),
        np.array(trips,   dtype=np.float32),
        np.array(dists,   dtype=np.float32),
    )


@dataclasses.dataclass
class RawCityData:
    city_name:      str
    n_tracts:       int
    n_pairs:        int
    X_raw:          np.ndarray         # (N, F) unscaled float32
    population:     torch.Tensor       # (N,)   raw population float32
    lon_lat:        torch.Tensor       # (N, 2) [lon, lat] float32
    pair_o_idx:     torch.LongTensor   # (E,)
    pair_d_idx:     torch.LongTensor   # (E,)
    pair_distance:  torch.Tensor       # (E,) log1p(km) float32
    pair_trips:     torch.Tensor       # (E,) raw counts, all >= 1 float32
    bin_labels:     torch.LongTensor   # (E,) distance bin index (0-3)
    dist_km:        np.ndarray         # (E,) raw pairwise distance in km


# Global In-Memory Caches for parsed raw CSV city datasets & normalized CityData instances
_RAW_CITY_CACHE: Dict[tuple[str, str], RawCityData] = {}
_CITY_DATA_CACHE: Dict[tuple[str, str, Optional[str]], CityData] = {}


def get_scaler_fingerprint(scaler: Optional[object]) -> Optional[str]:
    """
    Computes a deterministic content-based fingerprint (SHA-256) of a fitted StandardScaler.
    Prevents cross-fold leakage / normalization contamination caused by Python memory address (id(scaler)) reuse.
    Returns None if scaler is None.
    """
    if scaler is None:
        return None
    if hasattr(scaler, "mean_") and scaler.mean_ is not None:
        m_bytes = np.ascontiguousarray(scaler.mean_, dtype=np.float64).tobytes()
        v_bytes = np.ascontiguousarray(getattr(scaler, "var_", np.zeros_like(scaler.mean_)), dtype=np.float64).tobytes()
        s_bytes = np.ascontiguousarray(getattr(scaler, "scale_", np.ones_like(scaler.mean_)), dtype=np.float64).tobytes()
        return hashlib.sha256(m_bytes + v_bytes + s_bytes).hexdigest()
    return f"unfitted_{id(scaler)}"


def validate_feature_scaler(scaler: object) -> None:
    """Validate that a fitted scaler is safe for the fixed node-feature schema."""
    expected_features = len(NODE_FEATURE_COLUMNS)
    for attribute in ("mean_", "var_", "scale_"):
        if not hasattr(scaler, attribute):
            raise ValueError(f"Feature scaler is not fitted: missing {attribute}")
        values = np.asarray(getattr(scaler, attribute), dtype=np.float64)
        if values.shape != (expected_features,):
            raise ValueError(
                f"Feature scaler {attribute} has shape {values.shape}; "
                f"expected ({expected_features},)"
            )
        if not np.isfinite(values).all():
            raise ValueError(f"Feature scaler {attribute} contains NaN or Inf")

    if np.any(np.asarray(scaler.var_) < 0.0):
        raise ValueError("Feature scaler var_ contains negative values")
    if np.any(np.asarray(scaler.scale_) <= 0.0):
        raise ValueError("Feature scaler scale_ must be strictly positive")

    n_features = getattr(scaler, "n_features_in_", expected_features)
    if int(n_features) != expected_features:
        raise ValueError(
            f"Feature scaler expects {n_features} features; expected {expected_features}"
        )


def clear_city_cache() -> None:
    """Flushes both raw and normalized in-memory city dataset caches."""
    global _RAW_CITY_CACHE, _CITY_DATA_CACHE
    _RAW_CITY_CACHE.clear()
    _CITY_DATA_CACHE.clear()


def load_raw_city(
    city_name: str,
    data_root: str = "data",
    use_cache: bool = True,
) -> RawCityData:
    """
    Load or retrieve unscaled raw city data from disk / in-memory cache.
    """
    cache_key = (city_name, str(Path(data_root).resolve()))
    if use_cache and cache_key in _RAW_CITY_CACHE:
        return _RAW_CITY_CACHE[cache_key]

    base = Path(data_root) / city_name

    # --- Node features ---
    census = _load_csv_columns(base / "nodes" / "census.csv", CENSUS_COLS)
    poi    = _load_csv_columns(base / "nodes" / "poi.csv",    POI_COLS)
    road   = _load_csv_columns(base / "nodes" / "road.csv",   ROAD_COLS)
    X_raw  = np.concatenate([census, poi, road], axis=1)   # (N, F)
    X_raw  = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

    # Population for gravity prior (first census column)
    population = census[:, 0].copy()   # total_population

    # Coordinates
    lon_lat = _load_meta(base / "meta.csv")   # (N, 2)
    n_tracts = X_raw.shape[0]

    # --- Pair data ---
    o_idx, d_idx, trips, dist_km = _load_pairs(
        base / "pairs" / "od.csv",
        base / "pairs" / "distance.csv",
    )
    assert (trips >= 1).all(), f"{city_name}: found zero trip counts in candidate set"

    log_dist = np.log1p(dist_km)
    bin_labels = assign_bins(dist_km)

    raw_data = RawCityData(
        city_name     = city_name,
        n_tracts      = n_tracts,
        n_pairs       = len(o_idx),
        X_raw         = X_raw,
        population    = torch.tensor(population, dtype=torch.float32),
        lon_lat       = torch.tensor(lon_lat,    dtype=torch.float32),
        pair_o_idx    = torch.tensor(o_idx,      dtype=torch.long),
        pair_d_idx    = torch.tensor(d_idx,      dtype=torch.long),
        pair_distance = torch.tensor(log_dist,   dtype=torch.float32),
        pair_trips    = torch.tensor(trips,      dtype=torch.float32),
        bin_labels    = torch.tensor(bin_labels, dtype=torch.long),
        dist_km       = dist_km,
    )

    if use_cache:
        _RAW_CITY_CACHE[cache_key] = raw_data

    return raw_data


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_city(
    city_name: str,
    data_root: str = "data",
    feature_scaler: Optional["StandardScaler"] = None,
    fit_scaler: bool = False,
    use_cache: bool = True,
) -> CityData:
    """
    Load one city's data, optionally applying or fitting a feature scaler.

    Args:
        city_name:      Directory name under data_root.
        data_root:      Root of the data/ directory.
        feature_scaler: Optional fitted sklearn StandardScaler.
                        If None and fit_scaler=True, fits a new one.
        fit_scaler:     If True, fits scaler on this city's data.
        use_cache:      If True, retrieves raw parsed data from in-memory cache.

    Returns:
        CityData instance.
    """
    if feature_scaler is not None and fit_scaler:
        raise ValueError("Pass either feature_scaler or fit_scaler=True, not both")
    if feature_scaler is not None:
        validate_feature_scaler(feature_scaler)

    scaler_key = get_scaler_fingerprint(feature_scaler)
    resolved_root = str(Path(data_root).resolve())
    cache_key = (city_name, resolved_root, scaler_key)

    if use_cache and not fit_scaler and cache_key in _CITY_DATA_CACHE:
        return _CITY_DATA_CACHE[cache_key]

    raw = load_raw_city(city_name, data_root=data_root, use_cache=use_cache)

    # --- Normalize node features ---
    if feature_scaler is not None:
        X_norm = feature_scaler.transform(raw.X_raw)
    elif fit_scaler:
        from sklearn.preprocessing import StandardScaler
        feature_scaler = StandardScaler()
        X_norm = feature_scaler.fit_transform(raw.X_raw)
        scaler_key = get_scaler_fingerprint(feature_scaler)
        cache_key = (city_name, resolved_root, scaler_key)
    else:
        X_norm = raw.X_raw

    # Replace NaN/Inf that may arise from missing features
    X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=0.0, neginf=0.0)

    cd = CityData(
        city_name     = raw.city_name,
        n_tracts      = raw.n_tracts,
        n_pairs       = raw.n_pairs,
        node_features = torch.tensor(X_norm, dtype=torch.float32),
        population    = raw.population,
        lon_lat       = raw.lon_lat,
        pair_o_idx    = raw.pair_o_idx,
        pair_d_idx    = raw.pair_d_idx,
        pair_distance = raw.pair_distance,
        pair_trips    = raw.pair_trips,
        bin_labels    = raw.bin_labels,
        dist_km       = raw.dist_km,
    )

    if use_cache:
        _CITY_DATA_CACHE[cache_key] = cd

    return cd


def load_cities(
    city_names: List[str],
    data_root: str = "data",
    use_cache: bool = True,
) -> tuple[List[CityData], object]:
    """
    Load multiple cities, fitting a single StandardScaler on all training
    node features jointly (to ensure consistent normalization).

    Returns:
        (list of CityData, fitted scaler)
    """
    from sklearn.preprocessing import StandardScaler

    if not city_names:
        raise ValueError("At least one training city is required to fit the feature scaler")
    if len(city_names) != len(set(city_names)):
        raise ValueError("Training city names must be unique when fitting the feature scaler")

    # First pass: collect raw features from memory cache
    raw_list = [load_raw_city(name, data_root=data_root, use_cache=use_cache) for name in city_names]
    all_X = [r.X_raw for r in raw_list]

    scaler = StandardScaler()
    scaler.fit(np.concatenate(all_X, axis=0))
    validate_feature_scaler(scaler)
    scaler_key = get_scaler_fingerprint(scaler)
    resolved_root = str(Path(data_root).resolve())

    # Second pass: construct CityData with fitted scaler and cache into _CITY_DATA_CACHE
    cities = []
    for raw in raw_list:
        cache_key = (raw.city_name, resolved_root, scaler_key)
        if use_cache and cache_key in _CITY_DATA_CACHE:
            cities.append(_CITY_DATA_CACHE[cache_key])
        else:
            cd = CityData(
                city_name     = raw.city_name,
                n_tracts      = raw.n_tracts,
                n_pairs       = raw.n_pairs,
                node_features = torch.tensor(np.nan_to_num(scaler.transform(raw.X_raw), nan=0.0, posinf=0.0, neginf=0.0), dtype=torch.float32),
                population    = raw.population,
                lon_lat       = raw.lon_lat,
                pair_o_idx    = raw.pair_o_idx,
                pair_d_idx    = raw.pair_d_idx,
                pair_distance = raw.pair_distance,
                pair_trips    = raw.pair_trips,
                bin_labels    = raw.bin_labels,
                dist_km       = raw.dist_km,
            )
            if use_cache:
                _CITY_DATA_CACHE[cache_key] = cd
            cities.append(cd)

    return cities, scaler


def preload_all_cities(
    data_root: str = "data",
    city_names: Optional[List[str]] = None,
    build_graphs: bool = True,
    radius_km: float = 5.0,
) -> None:
    """
    Preloads all cities into in-memory cache upfront.
    Optionally computes spatial radius graphs and distance matrices.
    Completely eliminates disk I/O during multi-fold cross-validation.
    """
    from implement_new_plan.data.urban_graph import build_radius_graph
    if city_names is None:
        p = Path(data_root)
        if p.exists():
            city_names = sorted([d.name for d in p.iterdir() if d.is_dir() and (d / "meta.csv").exists()])
        else:
            city_names = []

    for name in city_names:
        raw = load_raw_city(name, data_root=data_root, use_cache=True)
        if build_graphs:
            build_radius_graph(raw.lon_lat, radius_km=radius_km, use_cache=True)


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "data"

    print("Loading Raleigh (small)...")
    cd = load_city("Raleigh", data_root=root)
    print(f"  Tracts: {cd.n_tracts}, Pairs: {cd.n_pairs}")
    print(f"  node_features: {cd.node_features.shape} | dtype: {cd.node_features.dtype}")
    print(f"  pair_trips: min={cd.pair_trips.min():.0f}, max={cd.pair_trips.max():.0f}")
    print(f"  pair_distance: min={cd.pair_distance.min():.3f}, max={cd.pair_distance.max():.3f}")
    print(f"  bin_labels: unique={cd.bin_labels.unique().tolist()}")
    print(f"  bin distribution: { {i: (cd.bin_labels==i).sum().item() for i in range(4)} }")
    print()

    print("Loading Raleigh + Denver jointly (scaler fit)...")
    cities, scaler = load_cities(["Raleigh", "Denver"], data_root=root)
    for c in cities:
        print(f"  {c.city_name}: node_features mean~{c.node_features.mean():.3f} std~{c.node_features.std():.3f}")
    print()

    print("Smoke test passed.")
```

---

<a id="implement-new-plan-data-od-split-py"></a>
## File: `implement_new_plan/data/od_split.py` (135 lines)

```python
"""
OD Split Generator Module: Strict Nested Split Contract for f in {0.10, 0.20, 0.30, 0.50, 1.00}.

Protocol Specifications (§2.2):
1. Candidate Pool:
       Only OD pairs belonging to positive interzonal OD support Omega_s^+:
       (origin != destination) & (distance_km > 0) & (flow >= 1)
   Never split on full matrix, zero-flow pairs, or all possible tract pairs.

2. Deterministic Sort:
       Sort candidate pairs by (origin, destination) before permutation
       to ensure identical splits regardless of disk / CSV row order.

3. Strict Nested Split Contract with split_seed = 42:
       Single permutation: pi_s = Permutation(Omega_s^+)
       Train_10(s)  = pi_s[:floor(0.10 * N_s)]
       Train_20(s)  = pi_s[:floor(0.20 * N_s)]
       Train_30(s)  = pi_s[:floor(0.30 * N_s)]
       Train_50(s)  = pi_s[:floor(0.50 * N_s)]
       Train_100(s) = pi_s[:N_s]

   Invariant:
       Train_10 subset Train_20 subset Train_30 subset Train_50 subset Train_100.

4. Backward Compatibility:
       split == "train" <==> in_train_30 == True
       split == "heldout" for remaining 70% in main setting.

5. Manifest:
       manifests/od_split_manifest.csv
       source_city,origin,destination,split,in_train_10,in_train_20,in_train_30,in_train_50,in_train_100,split_seed
"""

from typing import Dict, List, Tuple
import math
import numpy as np
import pandas as pd
from pathlib import Path

from implement_new_plan.calibration.support import (
    get_positive_interzonal_support_mask,
    validate_positive_interzonal_support,
)


def create_nested_source_splits(
    source_city: str,
    pair_o_idx: np.ndarray,
    pair_d_idx: np.ndarray,
    pair_distance_km: np.ndarray,
    pair_trips: np.ndarray,
    split_seed: int = 42,
) -> pd.DataFrame:
    """
    Creates deterministic nested split on positive interzonal OD support across 5 fractions:
    f in {0.10, 0.20, 0.30, 0.50, 1.00}.
    """
    # 1. Filter to positive interzonal support Omega_s^+
    mask = get_positive_interzonal_support_mask(pair_o_idx, pair_d_idx, pair_distance_km, pair_trips)
    o_supp = pair_o_idx[mask]
    d_supp = pair_d_idx[mask]
    dist_supp = pair_distance_km[mask]
    trips_supp = pair_trips[mask]

    validate_positive_interzonal_support(o_supp, d_supp, dist_supp, trips_supp)

    df_supp = pd.DataFrame({
        "source_city": source_city,
        "origin": o_supp,
        "destination": d_supp,
        "distance_km": dist_supp,
        "flow": trips_supp,
    })

    # 2. Deterministic sort by (origin, destination)
    df_supp = df_supp.sort_values(["origin", "destination"]).reset_index(drop=True)
    n_total = len(df_supp)

    # 3. Single uniform random permutation with seed 42
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(n_total)

    n_train_10 = int(math.floor(0.10 * n_total))
    n_train_20 = int(math.floor(0.20 * n_total))
    n_train_30 = int(math.floor(0.30 * n_total))
    n_train_50 = int(math.floor(0.50 * n_total))

    idx_10 = perm[:n_train_10]
    idx_20 = perm[:n_train_20]
    idx_30 = perm[:n_train_30]
    idx_50 = perm[:n_train_50]
    idx_100 = perm[:n_total]

    # 4. Mandatory Set Inclusion Sanity Checks
    s10 = set(idx_10)
    s20 = set(idx_20)
    s30 = set(idx_30)
    s50 = set(idx_50)
    s100 = set(idx_100)
    assert s10 <= s20 <= s30 <= s50 <= s100, "Nested split set inclusion invariant violated!"
    assert len(s100) == n_total, "Train_100 must contain all positive support pairs!"

    df_supp["in_train_10"] = False
    df_supp["in_train_20"] = False
    df_supp["in_train_30"] = False
    df_supp["in_train_50"] = False
    df_supp["in_train_100"] = True

    df_supp.loc[idx_10, "in_train_10"] = True
    df_supp.loc[idx_20, "in_train_20"] = True
    df_supp.loc[idx_30, "in_train_30"] = True
    df_supp.loc[idx_50, "in_train_50"] = True

    # 5. Backward compatibility for main 30/70 split: split column
    df_supp["split"] = "heldout"
    df_supp.loc[idx_30, "split"] = "train"
    df_supp["split_seed"] = split_seed

    # 6. Verify assertions
    assert (df_supp["split"] == "train").sum() == n_train_30
    assert (df_supp["in_train_30"] == True).sum() == n_train_30

    output_cols = [
        "source_city",
        "origin",
        "destination",
        "split",
        "in_train_10",
        "in_train_20",
        "in_train_30",
        "in_train_50",
        "in_train_100",
        "split_seed"
    ]
    return df_supp[output_cols]
```

---

<a id="implement-new-plan-data-source-scaler-py"></a>
## File: `implement_new_plan/data/source_scaler.py` (310 lines)

```python
"""
Source-Fitted Feature Scaler and Preprocessing Module.

Protocol Requirements:
1. Core Invariant:
       [Fit preprocessing trên source -> freeze -> apply y nguyên sang target]
   Zero-shot transfer NEVER computes or uses any target city distribution statistics.

2. Feature Categorization & Transformation:
   - Non-negative strongly skewed features:
     population, POI counts and densities, road lengths/counts and densities,
     median/per_capita income, and pairwise distance.
     Transform:
         x_log = log(1 + x)
         x_norm = (x_log - mu_s) / sigma_s
   - Symmetric or bounded features:
     Percentages/rates (employment, commute shares, homeownership, education)
     and median age, avg vehicles per household:
     Transform:
         x_norm = (x - mu_s) / sigma_s
   - Distance feature:
     Transform:
         d_log = log(1 + d_km)
         d_norm = (d_log - mu_s^distance) / sigma_s^distance
     where mu_s, sigma_s are estimated ONLY from the 30% training OD split of the source city.

3. Zero-Variance Safety Rule:
   If sigma_s < 10^{-12}:
       x_norm = 0.0, zero_variance_flag = True.
   Never divide by zero. Never compute sigma from target.

4. No Clipping in Main Experiment:
   Target features are normalized using source (mu_s, sigma_s) without clipping,
   preserving genuine out-of-distribution shifts.

5. Missing-Value Handling & Canonical Schema:
   - Schema incompatibility (missing column) raises hard SchemaError.
   - Within-column missing values are imputed strictly using source-city median
     (m_{s,f} = median(x_{s,f}) fitted on valid source values before transform).
   - Target uses the frozen source median value.
   - All transformed features must be finite (assert isfinite).

6. Reproducibility Manifest:
   manifests/source_feature_scalers.csv:
   source_city, feature_name, feature_type, fit_scope, imputation_method, imputation_value, transform, mean, std, zero_variance_flag, n_samples
"""

from typing import Dict, List, Optional, Tuple
import dataclasses
import numpy as np
import pandas as pd
from pathlib import Path

from implement_new_plan.data.dataset import (
    CENSUS_COLS,
    POI_COLS,
    ROAD_COLS,
    NODE_FEATURE_COLUMNS,
    CityData,
    RawCityData,
    load_raw_city,
)

# Strongly skewed features requiring log1p before z-score
SKEWED_FEATURES = set([
    "total_population",
    "median_income",
    "per_capita_income",
    "office",
    "office_density",
    "industrial",
    "industrial_density",
    "commercial",
    "commercial_density",
    "education_primary",
    "education_primary_density",
    "road_length_total",
    "road_density",
    "road_count",
    "motorway_length",
    "primary_length",
    "distance",
])


@dataclasses.dataclass
class SingleFeatureScaler:
    feature_name: str
    feature_type: str  # 'node_feature' or 'pairwise_distance'
    fit_scope: str  # 'all_source_nodes' or 'source_train_od_30pct'
    transform_type: str  # 'log1p_zscore' or 'linear_zscore'
    imputation_method: str  # 'source_median'
    imputation_value: float  # m_{s,f} = median(x_{s,f}) fitted on valid source values
    mean: float
    std: float
    zero_variance_flag: bool
    n_samples: int = 0

    def transform(self, x: np.ndarray) -> np.ndarray:
        x_in = x.copy().astype(np.float64)
        
        # 1. Source-median imputation for any NaN or Inf
        invalid_mask = ~np.isfinite(x_in)
        if np.any(invalid_mask):
            x_in[invalid_mask] = self.imputation_value
        
        # 2. Skewed feature transformation (if applicable)
        if self.transform_type == "log1p_zscore":
            x_in = np.log1p(np.maximum(0.0, x_in))
        
        # 3. Standardization with zero-variance safety rule
        if self.zero_variance_flag or self.std < 1e-12:
            return np.zeros_like(x_in, dtype=np.float32)
        
        x_norm = (x_in - self.mean) / self.std
        return x_norm.astype(np.float32)


class SourceCityFeatureScaler:
    """
    Fitted feature scaler representing a specific source city's observable data.
    Once fitted, it is frozen and applied unchanged to all 49 target cities.
    """
    def __init__(self, source_city: str):
        self.source_city = source_city
        self.node_scalers: Dict[str, SingleFeatureScaler] = {}
        self.distance_scaler: Optional[SingleFeatureScaler] = None
        self.is_fitted: bool = False

    def fit(
        self,
        raw_source_city: RawCityData,
        source_train_indices: np.ndarray,
    ) -> "SourceCityFeatureScaler":
        """
        Fits scalers exclusively on source-side observable data:
        - Node features: fitted on all tracts/nodes of source city.
          Missing values imputed using source median before transform.
        - Distance feature: fitted on the 30% training OD pairs of source city.
          Missing values imputed using source median distance before transform.
        """
        # 1. Fit node features
        X_raw = raw_source_city.X_raw  # (N, 26)
        if X_raw.shape[1] != len(NODE_FEATURE_COLUMNS):
            raise ValueError(
                f"Source city '{self.source_city}' has {X_raw.shape[1]} columns, "
                f"expected {len(NODE_FEATURE_COLUMNS)}"
            )

        for i, col_name in enumerate(NODE_FEATURE_COLUMNS):
            vals = X_raw[:, i].astype(np.float64)
            valid_mask = np.isfinite(vals)
            n_valid = int(np.sum(valid_mask))
            if n_valid == 0:
                raise ValueError(
                    f"Source city '{self.source_city}' has 0 valid values for required feature '{col_name}'"
                )

            # Imputation value: source median of valid entries
            imputation_val = float(np.median(vals[valid_mask]))
            vals_imputed = vals.copy()
            vals_imputed[~valid_mask] = imputation_val

            transform_type = "log1p_zscore" if col_name in SKEWED_FEATURES else "linear_zscore"

            if transform_type == "log1p_zscore":
                vals_t = np.log1p(np.maximum(0.0, vals_imputed))
            else:
                vals_t = vals_imputed

            mean_val = float(np.mean(vals_t))
            std_val = float(np.std(vals_t))
            zero_flag = bool(std_val < 1e-12 or not np.isfinite(std_val))
            if zero_flag:
                std_val = 0.0

            self.node_scalers[col_name] = SingleFeatureScaler(
                feature_name=col_name,
                feature_type="node_feature",
                fit_scope="all_source_nodes",
                transform_type=transform_type,
                imputation_method="source_median",
                imputation_value=imputation_val,
                mean=mean_val,
                std=std_val,
                zero_variance_flag=zero_flag,
                n_samples=X_raw.shape[0],
            )

        # 2. Fit distance feature (from 30% train split)
        train_dists = raw_source_city.dist_km[source_train_indices].astype(np.float64)
        valid_dist_mask = np.isfinite(train_dists)
        if int(np.sum(valid_dist_mask)) == 0:
            raise ValueError(f"Source city '{self.source_city}' has 0 valid distance values in train split")

        d_imputation_val = float(np.median(train_dists[valid_dist_mask]))
        train_dists_imputed = train_dists.copy()
        train_dists_imputed[~valid_dist_mask] = d_imputation_val

        dists_log = np.log1p(np.maximum(0.0, train_dists_imputed))
        d_mean = float(np.mean(dists_log))
        d_std = float(np.std(dists_log))
        d_zero = bool(d_std < 1e-12 or not np.isfinite(d_std))
        if d_zero:
            d_std = 0.0

        self.distance_scaler = SingleFeatureScaler(
            feature_name="distance",
            feature_type="pairwise_distance",
            fit_scope="source_train_od_30pct",
            transform_type="log1p_zscore",
            imputation_method="source_median",
            imputation_value=d_imputation_val,
            mean=d_mean,
            std=d_std,
            zero_variance_flag=d_zero,
            n_samples=len(source_train_indices),
        )

        self.is_fitted = True
        return self

    def transform_node_features(self, X_raw: np.ndarray) -> np.ndarray:
        """Transforms node features using frozen source parameters."""
        if not self.is_fitted:
            raise RuntimeError("Cannot transform before scaler is fitted on source city.")
        if X_raw.shape[1] != len(NODE_FEATURE_COLUMNS):
            raise ValueError(
                f"Feature matrix has {X_raw.shape[1]} columns, expected {len(NODE_FEATURE_COLUMNS)}"
            )

        X_norm = np.zeros_like(X_raw, dtype=np.float32)
        for i, col_name in enumerate(NODE_FEATURE_COLUMNS):
            scaler = self.node_scalers[col_name]
            X_norm[:, i] = scaler.transform(X_raw[:, i])
        
        if not np.isfinite(X_norm).all():
            raise ValueError("Transformed node features contain non-finite values (NaN/Inf).")
        return X_norm

    def impute_raw_population(self, population_raw: np.ndarray) -> np.ndarray:
        """
        Imputes missing values in raw population using the source-fitted median,
        preserving the raw population scale (NO log1p, NO z-score).
        
        Strict Contract:
        - Used for Two-Parameter Gravity and explicit gravity priors.
        - Returns non-negative, finite raw population array.
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot impute population before scaler is fitted on source city.")
        pop_arr = population_raw.copy().astype(np.float64)
        pop_scaler = self.node_scalers.get("total_population")
        if pop_scaler is None:
            raise KeyError("Scaler does not contain 'total_population' scaler.")
        
        invalid_mask = ~np.isfinite(pop_arr)
        if np.any(invalid_mask):
            pop_arr[invalid_mask] = pop_scaler.imputation_value
            
        if not np.isfinite(pop_arr).all():
            raise ValueError("Imputed raw population contains non-finite values.")
        if np.any(pop_arr < 0.0):
            raise ValueError("Imputed raw population contains negative values.")
        return pop_arr.astype(np.float32)

    def transform_distances(self, dist_km: np.ndarray) -> np.ndarray:
        """Transforms pairwise distances using frozen source distance parameters."""
        if not self.is_fitted or self.distance_scaler is None:
            raise RuntimeError("Cannot transform distance before scaler is fitted on source city.")
        d_norm = self.distance_scaler.transform(dist_km)
        if not np.isfinite(d_norm).all():
            raise ValueError("Transformed distance contains non-finite values (NaN/Inf).")
        return d_norm

    def to_manifest_records(self) -> List[Dict]:
        """Generates records for manifests/source_feature_scalers.csv."""
        if not self.is_fitted:
            raise RuntimeError("Scaler must be fitted before exporting records.")
        records = []
        for col in NODE_FEATURE_COLUMNS:
            sc = self.node_scalers[col]
            records.append({
                "source_city": self.source_city,
                "feature_name": sc.feature_name,
                "feature_type": sc.feature_type,
                "fit_scope": sc.fit_scope,
                "imputation_method": sc.imputation_method,
                "imputation_value": round(sc.imputation_value, 6),
                "transform": sc.transform_type,
                "mean": round(sc.mean, 6),
                "std": round(sc.std, 6),
                "zero_variance_flag": sc.zero_variance_flag,
                "n_samples": sc.n_samples,
            })
        if self.distance_scaler:
            records.append({
                "source_city": self.source_city,
                "feature_name": self.distance_scaler.feature_name,
                "feature_type": self.distance_scaler.feature_type,
                "fit_scope": self.distance_scaler.fit_scope,
                "imputation_method": self.distance_scaler.imputation_method,
                "imputation_value": round(self.distance_scaler.imputation_value, 6),
                "transform": self.distance_scaler.transform_type,
                "mean": round(self.distance_scaler.mean, 6),
                "std": round(self.distance_scaler.std, 6),
                "zero_variance_flag": self.distance_scaler.zero_variance_flag,
                "n_samples": self.distance_scaler.n_samples,
            })
        return records
```

---

<a id="implement-new-plan-data-urban-graph-py"></a>
## File: `implement_new_plan/data/urban_graph.py` (248 lines)

```python
"""
Spatial Urban Graph Construction (G^urban).

Constructs the urban spatial graph from tract centroid coordinates (lon, lat).
Crucial requirement: G^urban uses ONLY observable spatial geography, NEVER OD flows.

Supports:
1. k-NN graph: connects each node to its k geographically nearest neighbors.
2. Radius graph: connects nodes within a geographic distance threshold d_max (km).
3. Adaptive Radius graph: radius normalized to the city's empirical spatial diameter / extent.
"""

import math
import numpy as np
import torch


# Global In-Memory Cache for spatial urban graphs & distance matrices
_GRAPH_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
_DISTANCE_MATRIX_CACHE: dict[tuple | int | str, np.ndarray] = {}


def clear_graph_cache() -> None:
    """Flushes the global in-memory spatial urban graph and distance matrix caches."""
    global _GRAPH_CACHE, _DISTANCE_MATRIX_CACHE
    _GRAPH_CACHE.clear()
    _DISTANCE_MATRIX_CACHE.clear()


def clear_distance_matrix_cache() -> None:
    """Flushes the global in-memory pairwise distance matrix cache."""
    global _DISTANCE_MATRIX_CACHE
    _DISTANCE_MATRIX_CACHE.clear()


def haversine_distance_matrix(
    lon_lat: np.ndarray | torch.Tensor,
    use_cache: bool = True,
    cache_key: str | None = None,
) -> np.ndarray:
    """
    Computes pairwise Haversine distances in kilometers with in-memory caching.
    Avoids redundant O(N^2) computation on repeated calls for the same coordinates / city.
    """
    if isinstance(lon_lat, torch.Tensor):
        lon_lat = lon_lat.detach().cpu().numpy()
    else:
        lon_lat = np.asarray(lon_lat, dtype=np.float64)

    import hashlib
    coord_hash = hashlib.sha256(lon_lat.tobytes()).hexdigest()
    key = f"{cache_key}_{coord_hash}" if cache_key else coord_hash
    if use_cache and key in _DISTANCE_MATRIX_CACHE:
        return _DISTANCE_MATRIX_CACHE[key]

    R = 6371.0
    lons = np.radians(lon_lat[:, 0])
    lats = np.radians(lon_lat[:, 1])

    dlon = lons[:, None] - lons[None, :]
    dlat = lats[:, None] - lats[None, :]

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lats[:, None]) * np.cos(lats[None, :]) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    dist_mat = R * c

    if use_cache:
        _DISTANCE_MATRIX_CACHE[key] = dist_mat

    return dist_mat


def build_knn_graph(
    lon_lat: np.ndarray | torch.Tensor,
    k: int = 10,
    include_self_loop: bool = True,
    use_cache: bool = True,
    cache_key: str | None = None,
    dist_mat: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Constructs a k-nearest neighbor spatial graph with in-memory caching."""
    if isinstance(lon_lat, torch.Tensor):
        lon_lat = lon_lat.detach().cpu().numpy()
    else:
        lon_lat = np.asarray(lon_lat)

    import hashlib
    coord_hash = hashlib.sha256(lon_lat.tobytes()).hexdigest()
    base_key = f"{cache_key}_{coord_hash}" if cache_key else coord_hash
    key = (base_key, "knn", k, include_self_loop)
    if use_cache and key in _GRAPH_CACHE:
        return _GRAPH_CACHE[key]

    N = len(lon_lat)
    k = min(k, N - 1)
    if dist_mat is None:
        dist_mat = haversine_distance_matrix(lon_lat, use_cache=use_cache, cache_key=cache_key)

    rows, cols, dists = [], [], []
    for i in range(N):
        indices = np.argsort(dist_mat[i])
        neighbors = indices[1 : k + 1]

        if include_self_loop:
            rows.append(i)
            cols.append(i)
            dists.append(0.0)

        for nbr in neighbors:
            rows.append(i)
            cols.append(nbr)
            dists.append(dist_mat[i, nbr])

    edge_dict = {}
    for r, c, d in zip(rows, cols, dists):
        edge_dict[(r, c)] = d
        edge_dict[(c, r)] = d

    e_rows = [k[0] for k in edge_dict.keys()]
    e_cols = [k[1] for k in edge_dict.keys()]
    e_dists = list(edge_dict.values())

    edge_index = torch.tensor([e_rows, e_cols], dtype=torch.long)
    edge_dist = torch.tensor(e_dists, dtype=torch.float32)

    res = (edge_index, edge_dist)
    if use_cache:
        _GRAPH_CACHE[key] = res
    return res


def build_radius_graph(
    lon_lat: np.ndarray | torch.Tensor,
    radius_km: float = 5.0,
    include_self_loop: bool = True,
    use_cache: bool = True,
    cache_key: str | None = None,
    dist_mat: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Constructs a radius-based spatial graph connecting nodes within radius_km with caching."""
    if isinstance(lon_lat, torch.Tensor):
        lon_lat = lon_lat.detach().cpu().numpy()
    else:
        lon_lat = np.asarray(lon_lat)

    import hashlib
    coord_hash = hashlib.sha256(lon_lat.tobytes()).hexdigest()
    base_key = f"{cache_key}_{coord_hash}" if cache_key else coord_hash
    key = (base_key, "radius", float(radius_km), include_self_loop)
    if use_cache and key in _GRAPH_CACHE:
        return _GRAPH_CACHE[key]

    N = len(lon_lat)
    if dist_mat is None:
        dist_mat = haversine_distance_matrix(lon_lat, use_cache=use_cache, cache_key=cache_key)

    rows, cols, dists = [], [], []
    for i in range(N):
        if include_self_loop:
            rows.append(i)
            cols.append(i)
            dists.append(0.0)

        # Connect strictly within radius_km (excluding self)
        within_radius = np.where((dist_mat[i] <= radius_km) & (dist_mat[i] > 0))[0]
        # For isolated nodes, self-loop guarantees presence in graph.
        # Do NOT auto-connect nearest neighbor outside radius.

        for nbr in within_radius:
            rows.append(i)
            cols.append(nbr)
            dists.append(dist_mat[i, nbr])

    edge_dict = {}
    for r, c, d in zip(rows, cols, dists):
        edge_dict[(r, c)] = d
        edge_dict[(c, r)] = d

    e_rows = [k[0] for k in edge_dict.keys()]
    e_cols = [k[1] for k in edge_dict.keys()]
    e_dists = list(edge_dict.values())

    edge_index = torch.tensor([e_rows, e_cols], dtype=torch.long)
    edge_dist = torch.tensor(e_dists, dtype=torch.float32)

    # Sanity checks on constructed radius graph:
    # 1. Non-self edges must satisfy distance <= radius_km + tolerance
    non_self_mask = edge_index[0] != edge_index[1]
    if torch.any(non_self_mask):
        max_edge_d = float(torch.max(edge_dist[non_self_mask]).item())
        assert max_edge_d <= radius_km + 1e-4, f"Spatial edge distance {max_edge_d:.4f} km exceeds radius {radius_km} km."
    
    # 2. Every node must have self-loop if include_self_loop=True
    if include_self_loop and N > 0:
        self_loop_nodes = set(edge_index[0][~non_self_mask].tolist())
        assert len(self_loop_nodes) == N, f"Missing self-loops: found {len(self_loop_nodes)} nodes with self-loops, expected {N}."

    res = (edge_index, edge_dist)
    if use_cache:
        _GRAPH_CACHE[key] = res
    return res


def compute_spatial_graph_hash(edge_index: torch.Tensor, edge_dist: torch.Tensor) -> str:
    """Computes a deterministic SHA-256 hash of a spatial graph topology."""
    import hashlib
    e_bytes = edge_index.detach().cpu().numpy().tobytes()
    d_bytes = np.ascontiguousarray(edge_dist.detach().cpu().numpy(), dtype=np.float32).tobytes()
    return hashlib.sha256(e_bytes + d_bytes).hexdigest()


def build_adaptive_radius_graph(
    lon_lat: np.ndarray | torch.Tensor,
    scale_fraction: float = 0.15,
    min_radius_km: float = 2.0,
    include_self_loop: bool = True,
    use_cache: bool = True,
    cache_key: str | None = None,
    dist_mat: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """
    Constructs a spatial radius graph where radius_km is normalized to the city's
    empirical spatial diameter (max distance * scale_fraction).
    """
    if dist_mat is None:
        dist_mat = haversine_distance_matrix(lon_lat, use_cache=use_cache, cache_key=cache_key)
    diameter = float(np.max(dist_mat))
    adaptive_radius = max(min_radius_km, diameter * scale_fraction)
    ei, ed = build_radius_graph(
        lon_lat,
        radius_km=adaptive_radius,
        include_self_loop=include_self_loop,
        use_cache=use_cache,
        cache_key=cache_key,
        dist_mat=dist_mat,
    )
    return ei, ed, adaptive_radius


if __name__ == "__main__":
    coords = np.array([
        [-84.3880, 33.7490],
        [-84.3900, 33.7500],
        [-84.4000, 33.7600],
        [-84.5000, 33.8000],
    ])
    ei, ed, r = build_adaptive_radius_graph(coords, scale_fraction=0.2)
    print(f"Adaptive radius: {r:.2f} km | Edges: {ei.shape[1]}")
```

---

<a id="implement-new-plan-models-gravity-py"></a>
## File: `implement_new_plan/models/gravity.py` (65 lines)

```python
"""
Classical 2-parameter Physics Gravity Model Prior.

log T_ij^grav = G + log P_i + log P_j - alpha * log(D_ij)

Parameters:
    G: global scale parameter (learnable scalar)
    alpha: distance decay parameter (learnable scalar, initialized to ~1.0-2.0)

Both G and alpha are global trainable parameters shared across all cities in a fold,
providing the physics prior baseline for cross-city transfer.
"""

import math
import torch
import torch.nn as nn


class GravityPrior(nn.Module):
    def __init__(self, init_G: float = 0.0, init_alpha: float = 1.0):
        super().__init__()
        # Trainable physics parameters
        self.G = nn.Parameter(torch.tensor(init_G, dtype=torch.float32))
        self.log_alpha = nn.Parameter(torch.tensor(math.log(init_alpha), dtype=torch.float32))

    @property
    def alpha(self) -> torch.Tensor:
        return torch.exp(self.log_alpha)  # ensure alpha > 0

    def forward(
        self,
        population_i: torch.Tensor,
        population_j: torch.Tensor,
        distance_km: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes log T_ij^grav for each pair.

        Args:
            population_i: (E,) population of origin tract.
            population_j: (E,) population of destination tract.
            distance_km:  (E,) distance in km (not log).

        Returns:
            log_T_grav: (E,) log-expected gravity flow.
        """
        log_pi = torch.log(torch.clamp(population_i, min=1.0))
        log_pj = torch.log(torch.clamp(population_j, min=1.0))
        # Clamp at 0.1 km to avoid log(0) for intrazonal pairs (D_ii = 0).
        # This floor is an explicit design choice: intrazonal log_d = log(0.1) ≈ -2.3.
        # The model is trained on this behaviour; do not change without a full retrain.
        log_d  = torch.log(torch.clamp(distance_km, min=0.1))

        log_t_grav = self.G + log_pi + log_pj - self.alpha * log_d
        return log_t_grav


if __name__ == "__main__":
    grav = GravityPrior()
    p_i = torch.tensor([1000.0, 5000.0])
    p_j = torch.tensor([2000.0, 10000.0])
    d   = torch.tensor([5.0, 15.0])
    out = grav(p_i, p_j, d)
    print("Gravity prior output log_T:", out)
    print(f"Alpha: {grav.alpha.item():.4f}, G: {grav.G.item():.4f}")
```

---

<a id="implement-new-plan-models-node-encoder-py"></a>
## File: `implement_new_plan/models/node_encoder.py` (190 lines)

```python
"""
Urban Graph Neural Network Node Encoder.

Learns tract representation h_i from urban features X and spatial graph G^urban:
    h_i = GNN_theta(X, G^urban)

Graph structure:
    G^urban is built ONLY from observable spatial geography (k-NN / radius graph).
    No OD data is ever used to construct G^urban.

Architecture:
    Multi-layer Graph Convolution / GAT / GraphConv with residual connections and LayerNorm.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvLayer(nn.Module):
    """
    Message passing layer with edge distance modulation.
    Performs distance-conditioned message passing:
        m_ij = W_msg * [h_j || log(1 + d_ij)]
        h_i' = W_self * h_i + Agg_{j in N(i)}(m_ij)
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.msg_linear = nn.Linear(in_dim + 1, out_dim)
        self.self_linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_dist: torch.Tensor) -> torch.Tensor:
        """
        x: (N, in_dim)
        edge_index: (2, E_graph)
        edge_dist: (E_graph,) distance in km
        """
        src, dst = edge_index[0], edge_index[1]
        
        # Log distance feature
        log_d = torch.log1p(edge_dist).unsqueeze(-1)  # (E_graph, 1)
        
        # Message computation: [h_src, log_d]
        msg_input = torch.cat([x[src], log_d], dim=-1)  # (E_graph, in_dim + 1)
        msg = self.msg_linear(msg_input)  # (E_graph, out_dim)

        # Scatter mean aggregation
        out = torch.zeros(x.size(0), msg.size(1), device=x.device, dtype=x.dtype)
        # Degree count for mean aggregation
        deg = torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)
        
        out.index_add_(0, dst, msg)
        deg.index_add_(0, dst, torch.ones_like(log_d))
        
        out = out / torch.clamp(deg, min=1.0)
        
        # Combine with transformed self features
        h_self = self.self_linear(x)
        out = self.norm(F.relu(out + h_self))
        return out


class UrbanGNN(nn.Module):
    """
    Urban GNN Node Encoder that produces node embeddings h_i in R^d.
    """
    def __init__(
        self,
        in_dim: int = 26,
        hidden_dim: int = 64,
        out_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.layers = nn.ModuleList([
            GraphConvLayer(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        self.output_fc = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_dist: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:          (N, in_dim) normalized node features.
            edge_index: (2, E_graph) spatial graph edges.
            edge_dist:  (E_graph,) geographic distances.

        Returns:
            h: (N, out_dim) node embeddings.
        """
        h = self.input_fc(x)
        for layer in self.layers:
            h_new = layer(h, edge_index, edge_dist)
            h = h + self.dropout(h_new)  # residual connection

        h = self.output_fc(h)
        return h

class MLPLayer(nn.Module):
    """
    A dense layer designed to have the same nominal parameter count as GraphConvLayer.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        # GraphConvLayer has msg_linear (in_dim + 1 -> out_dim) and self_linear (in_dim -> out_dim).
        # We replicate this exactly here.
        self.msg_equivalent = nn.Linear(in_dim + 1, out_dim)
        self.self_linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pad with zeros to match the log(1+d) feature concatenated in GNN message passing
        dummy_dist = torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)
        msg_input = torch.cat([x, dummy_dist], dim=-1)
        
        out = self.msg_equivalent(msg_input) + self.self_linear(x)
        return self.norm(F.relu(out))


class NodeMLP(nn.Module):
    """
    MLP Node Encoder that produces node embeddings h_i in R^d without message passing.
    Architecture matches UrbanGNN but removes the GraphConv aggregation.
    """
    def __init__(
        self,
        in_dim: int = 26,
        hidden_dim: int = 64,
        out_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Use MLPLayer to maintain nominal parameter-count parity with GraphConvLayer.
        self.layers = nn.ModuleList([
            MLPLayer(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        self.output_fc = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_dist: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, in_dim) normalized node features.
            edge_index: Ignored. Kept for signature compatibility with UrbanGNN so that 
                        both models can be dropped into the same training/inference loop 
                        without modifying the call signature.
            edge_dist: Ignored. See above.
        Returns:
            h: (N, out_dim) node embeddings.
        """
        h = self.input_fc(x)
        for layer in self.layers:
            h_new = layer(h)
            h = h + self.dropout(h_new)
        h = self.output_fc(h)
        return h



if __name__ == "__main__":
    gnn = UrbanGNN(in_dim=26, hidden_dim=32, out_dim=32, num_layers=2)
    x = torch.randn(10, 26)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
    edge_dist = torch.tensor([1.0, 2.0, 1.5, 3.0, 0.5])
    h = gnn(x, edge_index, edge_dist)
    print(f"UrbanGNN output shape: {h.shape}")
```

---

<a id="implement-new-plan-models-decoder-py"></a>
## File: `implement_new_plan/models/decoder.py` (92 lines)

```python
r"""
Pairwise OD Decoder with Single Base Magnitude Head (ZTNB).

Input edge representation:
    e_ij = [h_i, h_j, log(1 + D_ij), log(T^{grav}_ij)]

Single prediction head producing base Negative Binomial parameter via
residual-gravity initialization: mu_nb_ij = softplus(log_t_grav + residual_ij).

Exact ZTNB Likelihood & Predictions:
    At training: loss = -log P_ZTNB(T_ij; mu_nb_ij, phi) on positive observations in Omega_c.
    At inference: expected zero-shot prediction is the conditional expectation:
        \hat{T}^{ZS}_ij = E[T_ij | T_ij >= 1] = compute_conditional_mean(mu_nb_ij, log_phi).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PairwiseODDecoder(nn.Module):
    def __init__(
        self,
        node_dim: int = 64,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Input: [h_i, h_j, log_d, log_t_grav] -> dim = 2 * node_dim + 2
        in_dim = 2 * node_dim + 2

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Zero-init final layer so the gravity prior is supplied as a log-scale decoder feature/offset 
        # with a zero-initialized neural residual, yielding softplus(log_t_grav) at initialization.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        h_i: torch.Tensor,
        h_j: torch.Tensor,
        log_distance: torch.Tensor,
        log_t_grav: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            h_i:          (E, node_dim) origin node embeddings.
            h_j:          (E, node_dim) destination node embeddings.
            log_distance: (E,) or (E, 1) log1p(distance_km).
            log_t_grav:   (E,) or (E, 1) log gravity flow.

        Returns:
            mu_nb: (E,) positive base mean parameter mu_nb_ij > 0.
        """
        if log_distance.dim() == 1:
            log_distance = log_distance.unsqueeze(-1)
        if log_t_grav.dim() == 1:
            log_t_grav = log_t_grav.unsqueeze(-1)

        # Concatenate edge representation e_ij
        e_ij = torch.cat([h_i, h_j, log_distance, log_t_grav], dim=-1)

        residual = self.net(e_ij)  # (E, 1), ~0 at init
        # Residual-gravity: gravity prior serves as log-scale feature, GNN learns the deviation.
        # Yields softplus(log_t_grav) at initialization.
        log_mu_nb = log_t_grav + residual
        mu_nb = F.softplus(log_mu_nb.squeeze(-1)) + 1e-4
        return mu_nb


if __name__ == "__main__":
    dec = PairwiseODDecoder(node_dim=32, hidden_dim=64)
    h_i = torch.randn(100, 32)
    h_j = torch.randn(100, 32)
    ld = torch.randn(100)
    ltg = torch.randn(100)
    mu_nb = dec(h_i, h_j, ld, ltg)
    print("Decoder mu_nb output shape:", mu_nb.shape, "min:", mu_nb.min().item(), "max:", mu_nb.max().item())

    # At init, residual ~ 0, so mu_nb should track softplus(log_t_grav) closely
    expected = F.softplus(ltg) + 1e-4
    print("Max deviation from pure gravity at init:", (mu_nb - expected).abs().max().item())
```

---

<a id="implement-new-plan-models-od-models-py"></a>
## File: `implement_new_plan/models/od_models.py` (395 lines)

```python
"""
Three Distinct OD Flow Baseline Families:
1. Two-Parameter Gravity Model (TwoParameterGravity)
2. Pairwise MLP (PairwiseMLP)
3. Urban-GNN (UrbanGNN)

Implementation Invariant:
- These represent THREE DIFFERENT MODELING FAMILIES, not controlled ablations of each other.
- Shared protocol:
    same source city
    same 30% positive OD training pairs
    same held-out/source support
    same target positive support
    same model seeds (1, 10, 100) where applicable
    same evaluation metrics
    same zero-shot transfer protocol
    same DBD calibration pipeline
"""

import math
from typing import Optional, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from implement_new_plan.models.node_encoder import UrbanGNN as GNNNodeEncoder
from implement_new_plan.models.gravity import GravityPrior
from implement_new_plan.models.decoder import PairwiseODDecoder


# ===========================================================================
# MODEL 1 — TWO-PARAMETER GRAVITY MODEL
# ===========================================================================
class TwoParameterGravity(nn.Module):
    r"""
    Two-Parameter Physics-Based Gravity Model:
        \hat{T}_{ij} = \exp(G) * P_i * P_j * D_{ij}^{-\alpha}
    
    Exactly two trainable parameters: \theta_gravity = {G, \alpha}.
    - No neural network, no MLP, no graph, no node embeddings, no message passing.
    - G: Unconstrained global scale parameter (G \in R).
    - \alpha: Distance decay parameter \alpha = \exp(\log \alpha) > 0 (strictly matching baseline GravityPrior).
    - Numerically stable log-space computation:
        \log \hat{T}_{ij} = G + \log P_i + \log P_j - \alpha \log D_{ij}
      for pairs with P_i > 0 and P_j > 0. If P_i == 0 or P_j == 0, \hat{T}_{ij} = 0.
    """
    def __init__(self, init_G: float = 0.0, init_alpha: float = 1.0):
        super().__init__()
        self.G = nn.Parameter(torch.tensor(init_G, dtype=torch.float32))
        # Parameterize alpha = exp(log_alpha) > 0 strictly matching baseline GravityPrior
        if init_alpha <= 0.0:
            raise ValueError(f"init_alpha must be strictly positive, got {init_alpha}")
        self.log_alpha = nn.Parameter(torch.tensor(math.log(init_alpha), dtype=torch.float32))

    @property
    def alpha(self) -> torch.Tensor:
        return torch.exp(self.log_alpha)

    def forward(
        self,
        population_raw_o: torch.Tensor,
        population_raw_d: torch.Tensor,
        distance_km_raw: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Computes gravity flow directly on positive support:
            \hat{T}_{ij} = \exp(G) * P_i * P_j * D_{ij}^{-\alpha}
            
        Strict Contract:
        - population_raw_o: Raw origin tract population P_i (non-negative, finite).
          NO log1p, NO z-score, NO min-max, NO target-normalization.
        - population_raw_d: Raw destination tract population P_j (non-negative, finite).
          NO log1p, NO z-score, NO min-max, NO target-normalization.
        - distance_km_raw: Raw physical Haversine distance in km (> 0).
          NO log1p, NO standardization.
        """
        # 1. Population validity checks
        if not torch.isfinite(population_raw_o).all() or not torch.isfinite(population_raw_d).all():
            raise ValueError("TwoParameterGravity requires finite raw population values (no NaN/Inf).")
        if not (population_raw_o >= 0).all() or not (population_raw_d >= 0).all():
            raise ValueError("TwoParameterGravity requires non-negative raw population values. Found negative population.")
        
        # 2. Distance validity check
        if not torch.isfinite(distance_km_raw).all() or not (distance_km_raw > 0).all():
            raise ValueError("TwoParameterGravity requires strictly positive physical distance values (D_ij > 0).")

        # 3. Handle zero-population pairs explicitly without log(0)
        pos_pop_mask = (population_raw_o > 0) & (population_raw_d > 0)
        
        # Initialize output flow tensor with zeros
        t_hat = torch.zeros_like(distance_km_raw, dtype=torch.float32)

        if torch.any(pos_pop_mask):
            p_o_pos = population_raw_o[pos_pop_mask]
            p_d_pos = population_raw_d[pos_pop_mask]
            d_pos = distance_km_raw[pos_pop_mask]

            alpha_val = self.alpha
            log_flow = self.G + torch.log(p_o_pos) + torch.log(p_d_pos) - alpha_val * torch.log(d_pos)

            # Check for potential floating-point overflow before exp
            if torch.any(log_flow > 88.0):
                max_log = float(torch.max(log_flow).item())
                min_log = float(torch.min(log_flow).item())
                raise OverflowError(
                    f"TwoParameterGravity prediction log_flow exceeded floating point limits (max={max_log:.2f}, min={min_log:.2f}, G={self.G.item():.4f}, alpha={alpha_val.item():.4f})."
                )

            t_hat[pos_pop_mask] = torch.exp(log_flow)

        # 4. Final numerical sanity assertions (no arbitrary clipping)
        if not torch.isfinite(t_hat).all():
            raise ValueError("TwoParameterGravity produced non-finite flow predictions (NaN/Inf).")
        if not (t_hat >= 0).all():
            raise ValueError("TwoParameterGravity produced negative flow predictions.")

        return t_hat


# ===========================================================================
# MODEL 2 — PAIRWISE MLP
# ===========================================================================
class PairwiseMLP(nn.Module):
    r"""
    DeepGravity-Inspired Pairwise MLP (Direct Flow-Intensity Regression).
    
    Adapted to direct OD-flow regression without target origin outflow constraints:
        \hat{T}_{ij} = Softplus(f_MLP([x_i || x_j || d'_ij]))
    
    Critical Separation Invariant:
    - NO G, NO alpha, NO gravity equation, NO gravity residual/prior/initialization.
    - NO graph, NO message passing, NO adjacency matrix.
    - Input: [x_i || x_j || d'_ij] of dimension 2 * node_in_dim + 1.
    """
    def __init__(
        self,
        node_in_dim: int = 26,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Input: origin features (F), destination features (F), source-scaled distance (1)
        in_dim = 2 * node_in_dim + 1

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        x_o: torch.Tensor,
        x_d: torch.Tensor,
        distance_std: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Forward pass predicting flow directly:
            x_ij = [x_o || x_d || distance_std]
            \hat{T}_{ij} = Softplus(net(x_ij)) >= 0
            
        Strict Input Contract:
        - x_o: Origin node features (shape: (..., F)) using canonical NODE_FEATURE_COLUMNS.
        - x_d: Destination node features (shape: (..., F)) using the exact same canonical schema.
        - distance_std: Source-standardized log-distance: (log1p(d_raw) - mu_s) / sigma_s.
        - Dimension check: [x_o || x_d || distance_std] must have exactly 2 * F + 1 dimensions.
        """
        if distance_std.dim() == 1:
            distance_std = distance_std.unsqueeze(-1)

        F_dim = x_o.shape[-1]
        if x_d.shape[-1] != F_dim:
            raise ValueError(
                f"Origin and destination feature dimension mismatch: x_o has {F_dim} dims, "
                f"x_d has {x_d.shape[-1]} dims. Must use identical canonical schema."
            )

        pair_feat = torch.cat([x_o, x_d, distance_std], dim=-1)
        expected_pair_dim = 2 * F_dim + 1
        if pair_feat.shape[-1] != expected_pair_dim:
            raise AssertionError(
                f"Pairwise MLP input dimension mismatch: expected {expected_pair_dim} (2 * {F_dim} + 1), "
                f"got {pair_feat.shape[-1]}"
            )

        z_ij = self.net(pair_feat).squeeze(-1)
        t_hat = F.softplus(z_ij)
        return t_hat


# ===========================================================================
# MODEL 3 — URBAN-GNN
# ===========================================================================
class UrbanGNN(nn.Module):
    r"""
    Spatial Graph Neural Network Model with Message Passing and Trainable Gravity.
    
    Architecture (Inherited baseline implementation):
    1. Spatial radius graph (r=5.0 km) with distance-modulated message passing:
           m_ij = W_msg * [h_j || log(1 + d_ij)]
    2. Node representations h_i, h_j from 2-layer GraphConvLayer.
    3. Jointly trained classical 2-parameter gravity component:
           log T_ij^grav = G + log P_i + log P_j - alpha * log(D_ij)
    4. Neural transfer decoder with residual-gravity combination:
           mu_nb_ij = Softplus(log T_ij^grav + residual_ij)
    """
    def __init__(
        self,
        node_in_dim: int = 26,
        node_hidden_dim: int = 64,
        node_out_dim: int = 64,
        num_gnn_layers: int = 2,
        decoder_hidden_dim: int = 64,
        dropout: float = 0.1,
        init_G: float = 0.0,
        init_alpha: float = 1.0,
    ):
        super().__init__()
        # 1. Spatial GNN Node Encoder (Message Passing)
        self.node_encoder = GNNNodeEncoder(
            in_dim=node_in_dim,
            hidden_dim=node_hidden_dim,
            out_dim=node_out_dim,
            num_layers=num_gnn_layers,
            dropout=dropout,
        )

        # 2. Jointly trainable Gravity component (G, alpha)
        self.gravity = GravityPrior(init_G=init_G, init_alpha=init_alpha)

        # 3. Residual-Gravity Pairwise Decoder
        self.decoder = PairwiseODDecoder(
            node_dim=node_out_dim,
            hidden_dim=decoder_hidden_dim,
            dropout=dropout,
        )

    @property
    def G(self) -> torch.Tensor:
        return self.gravity.G

    @property
    def alpha(self) -> torch.Tensor:
        return self.gravity.alpha

    def forward(
        self,
        x: torch.Tensor,
        spatial_edge_index: torch.Tensor,
        spatial_edge_dist_raw: torch.Tensor,
        pair_o_idx: torch.Tensor,
        pair_d_idx: torch.Tensor,
        pair_distance_km_raw: torch.Tensor,
        population_raw: Optional[torch.Tensor] = None,
        pair_distance_log: Optional[torch.Tensor] = None,
        population: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""
        Forward pass with spatial message passing and residual gravity decoding.
        
        Strict Contract:
        - x: Processed node features (includes source-standardized log1p population).
        - spatial_edge_dist_raw: Raw Haversine physical distance in km (<= 5.0 km radius).
        - pair_distance_km_raw: Raw physical distance D_ij in km (> 0) for explicit gravity prior.
        - population_raw: Raw tract population P_i (non-negative, finite) for explicit gravity prior.
        - pair_distance_log: log(1 + d_raw) used as edge feature in decoder. If None, computed on the fly.
        """
        # Support both population_raw and legacy population keyword argument
        pop_input = population_raw if population_raw is not None else population
        if pop_input is None:
            raise ValueError("UrbanGNN requires population_raw tensor for its gravity prior component.")

        # Population validity check for raw gravity prior
        if not torch.isfinite(pop_input).all():
            raise ValueError("UrbanGNN gravity prior requires finite raw population values.")
        if not (pop_input >= 0).all():
            raise ValueError("UrbanGNN gravity prior requires non-negative raw population values.")

        # 1. Node message passing: internally uses log1p(spatial_edge_dist_raw)
        h = self.node_encoder(x, spatial_edge_index, spatial_edge_dist_raw)
        h_o = h[pair_o_idx]
        h_d = h[pair_d_idx]

        # 2. Gravity prior component: MUST use raw physical distance in km and raw population
        pop_o = pop_input[pair_o_idx]
        pop_d = pop_input[pair_d_idx]
        log_t_grav = self.gravity(pop_o, pop_d, pair_distance_km_raw)

        # 3. Decoder combination: uses log1p(d_raw)
        if pair_distance_log is None:
            pair_distance_log = torch.log1p(pair_distance_km_raw)
            
        t_hat = self.decoder(h_o, h_d, pair_distance_log, log_t_grav)
        return t_hat


# ===========================================================================
# SANITY CHECKS & DISTANCE CONTRACT VERIFICATION
# ===========================================================================
def verify_distance_contract():
    """Validates the strict distance contract across all three model families."""
    print("Running Distance Contract Verification...")
    
    # 1. Two-Parameter Gravity: requires raw physical km and raw population
    grav = TwoParameterGravity()
    population_raw_o = torch.tensor([10000.0, 50000.0])
    population_raw_d = torch.tensor([20000.0, 30000.0])
    distance_km_raw = torch.tensor([5.2, 14.8])
    
    # Assert raw distance and population are valid
    assert torch.all(distance_km_raw > 0)
    assert torch.all(population_raw_o >= 0)
    assert torch.all(population_raw_d >= 0)
    out_grav = grav(population_raw_o, population_raw_d, distance_km_raw)
    assert out_grav.shape == (2,)
    assert torch.all(out_grav > 0)

    # Test zero population handling (P=0 -> flow=0 without log(0) errors)
    out_zero_pop = grav(torch.tensor([0.0, 50000.0]), population_raw_d, distance_km_raw)
    assert out_zero_pop[0].item() == 0.0
    assert out_zero_pop[1].item() > 0.0

    # Test alpha non-negativity property
    assert grav.alpha.item() >= 0.0

    # Test error handling on negative/non-finite population
    try:
        grav(torch.tensor([-1.0, 10.0]), population_raw_d, distance_km_raw)
        assert False, "Should fail on negative population"
    except ValueError:
        pass

    # Test error handling on non-positive distance
    try:
        grav(population_raw_o, population_raw_d, torch.tensor([0.0, 10.0]))
        assert False, "Should fail on non-positive distance"
    except ValueError:
        pass
    
    # 2. Pairwise MLP: requires source-standardized log-distance
    mlp = PairwiseMLP(node_in_dim=26)
    x_o = torch.randn(2, 26)
    x_d = torch.randn(2, 26)
    
    # Simulate source stats
    source_mean = 2.5
    source_std = 0.8
    distance_log = torch.log1p(distance_km_raw)
    distance_std = (distance_log - source_mean) / source_std
    
    # Verify relations
    assert torch.allclose(distance_log, torch.log1p(distance_km_raw))
    assert torch.allclose(distance_std, (distance_log - source_mean) / source_std)
    
    out_mlp = mlp(x_o, x_d, distance_std)
    assert out_mlp.shape == (2,)
    assert torch.all(out_mlp >= 0)
    
    # 3. Urban-GNN: spatial radius edge raw <= 5.0, pair raw km, pair log
    gnn = UrbanGNN(node_in_dim=26)
    x_all = torch.randn(5, 26)
    spatial_edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    spatial_edge_dist_raw = torch.tensor([1.2, 3.4, 4.8])  # <= 5.0 km
    assert torch.all(spatial_edge_dist_raw <= 5.0)
    
    pop_all = torch.tensor([1000.0, 2000.0, 1500.0, 3000.0, 2500.0])
    pair_o = torch.tensor([0, 3])
    pair_d = torch.tensor([1, 4])
    
    out_gnn = gnn(
        x=x_all,
        spatial_edge_index=spatial_edge_index,
        spatial_edge_dist_raw=spatial_edge_dist_raw,
        pair_o_idx=pair_o,
        pair_d_idx=pair_d,
        pair_distance_km_raw=distance_km_raw,
        population_raw=pop_all,
    )
    assert out_gnn.shape == (2,)
    assert torch.all(out_gnn >= 0)
    print("Distance & Raw Population Contract Verification PASSED.")


if __name__ == "__main__":
    verify_distance_contract()
```

---

<a id="implement-new-plan-loss-log1p-mse-py"></a>
## File: `implement_new_plan/loss/log1p_mse.py` (55 lines)

```python
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
```

---

<a id="implement-new-plan-training-evaluate-py"></a>
## File: `implement_new_plan/training/evaluate.py` (241 lines)

```python
"""
Comprehensive Evaluation Suite on Interzonal Domain Omega_c^+ and Full Support Omega_c.

Primary Metric:
    Interzonal CPC (CPC_inter) on Omega_c^+ = {(i,j) in Omega_c : i != j, D_ij > 0}:
        Evaluates the displacement flow distribution of moving commuters.

Secondary Metrics:
    1. Scale-Normalized Interzonal CPC (CPC_inter_norm = 1 - TVD):
        Evaluates pure structural flow geometry independent of total flow scale.
    2. RMSE-log1p on Omega_c^+.
    3. Pearson/Spearman correlation on Omega_c^+.
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
        # NOTE: pair_distance is stored as log1p(km) in CityData. The > 0.0 check is equivalent
        # to distance_km > 0 since log1p is monotone. Do NOT use dist_log1p for metric computation.
        dist_log1p = p_dist
        inter_mask = (o_np != d_np) & (dist_log1p > 0.0)
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
def compute_mse_pair(t_true: np.ndarray, t_pred: np.ndarray) -> float:
    """Computes Mean Squared Error."""
    return float(np.mean((t_true - t_pred) ** 2))


def evaluate_calibration_transfer(
    true_flow: np.ndarray,
    pred_before: np.ndarray,
    pred_after: np.ndarray,
) -> dict[str, float]:
    r"""
    Evaluation Module Interface for Cross-City Zero-Shot Transfer and DBD Calibration.
    Strictly executed AFTER calibration is completed.

    Inputs:
    - true_flow: Target city ground truth flows on fixed positive support (\Omega_t^+).
    - pred_before: Baseline predictions on \Omega_t^+ (\hat{T}^{(0)}).
    - pred_after: Calibrated predictions on \Omega_t^+ (\hat{T}^{(1)}).

    Outputs:
    - Baseline & Calibrated Metrics: CPC, CPC_norm, MAE, MSE, RMSE, R_vol
    - Gains: delta_CPC, delta_MAE, delta_MSE
    - Invariant Checks: Exact volume preservation validation
    """
    if not (len(true_flow) == len(pred_before) == len(pred_after)):
        raise ValueError(
            f"Array length mismatch: true_flow ({len(true_flow)}), "
            f"pred_before ({len(pred_before)}), pred_after ({len(pred_after)})"
        )

    t_true = np.asarray(true_flow, dtype=np.float64)
    t_before = np.asarray(pred_before, dtype=np.float64)
    t_after = np.asarray(pred_after, dtype=np.float64)

    # Baseline metrics
    cpc_before = compute_cpc_pair(t_true, t_before)
    cpc_norm_before = compute_cpc_norm_pair(t_true, t_before)
    mae_before = compute_mae_pair(t_true, t_before)
    mse_before = compute_mse_pair(t_true, t_before)
    rmse_before = compute_rmse_pair(t_true, t_before)

    # Calibrated metrics
    cpc_after = compute_cpc_pair(t_true, t_after)
    cpc_norm_after = compute_cpc_norm_pair(t_true, t_after)
    mae_after = compute_mae_pair(t_true, t_after)
    mse_after = compute_mse_pair(t_true, t_after)
    rmse_after = compute_rmse_pair(t_true, t_after)

    # Diagnostic evaluation-only R_vol
    total_true = float(np.sum(t_true))
    if total_true <= 0:
        raise ValueError("Sum of true flows on positive support must be > 0.")

    r_vol_before = float(np.sum(t_before) / total_true)
    r_vol_after = float(np.sum(t_after) / total_true)

    # Exact volume preservation check
    sum_before_64 = float(np.sum(t_before, dtype=np.float64))
    sum_after_64 = float(np.sum(t_after, dtype=np.float64))
    vol_diff = abs(sum_after_64 - sum_before_64)
    rel_vol_diff = vol_diff / max(sum_before_64, 1.0)
    assert vol_diff < 1e-10 or rel_vol_diff < 1e-10, (
        f"Volume preservation invariant violated in evaluation: "
        f"|sum(after) - sum(before)| = {vol_diff:.4e} (rel={rel_vol_diff:.4e}) >= 1e-10"
    )

    return {
        "CPC_before": cpc_before,
        "CPC_after": cpc_after,
        "delta_CPC": cpc_after - cpc_before,
        "CPC_norm_before": cpc_norm_before,
        "CPC_norm_after": cpc_norm_after,
        "delta_CPC_norm": cpc_norm_after - cpc_norm_before,
        "MAE_before": mae_before,
        "MAE_after": mae_after,
        "delta_MAE": mae_before - mae_after,
        "MSE_before": mse_before,
        "MSE_after": mse_after,
        "delta_MSE": mse_before - mse_after,
        "RMSE_before": rmse_before,
        "RMSE_after": rmse_after,
        "delta_RMSE": rmse_before - rmse_after,
        "R_vol_before": r_vol_before,
        "R_vol_after": r_vol_after,
    }
```

---

<a id="implement-new-plan-training-model-trainer-py"></a>
## File: `implement_new_plan/training/model_trainer.py` (186 lines)

```python
"""
Master Model Training Module adhering strictly to new_plan.md.

Protocol Invariants:
1. Loss: Log1p-MSE L = Mean( (log(1+T) - log(1+\hat{T}))^2 )
2. Optimizer: AdamW, lr=2e-3, weight_decay=1e-4, grad_clip=5.0, 40 epochs.
3. Batch contract: Strict Full-Batch optimization on Train_f(s).
4. Seeds: [1, 10, 100]. Gravity model optimization is deterministic, yielding std=0.0.
5. Models trained per source city and active fraction:
   - f = 0.30 (main setting for Exp A, C, D)
   - f in {0.10, 0.20, 0.50, 1.00} (for Exp B)
6. Outputs:
   - Checkpoints saved in models/{model_name}_{source_city}_f{fraction}_seed{seed}.pt
   - manifests/gravity_parameters.csv (150 rows: 50 cities x 3 seeds for main fraction f=0.30)
   - manifests/gravity_training_trace.csv (6000 rows: 50 cities x 40 epochs x 3 seeds for main f=0.30)
   - source_city_results.csv (450 rows: 50 cities x 3 models x 3 seeds on 70% held-out for f=0.30)
   - source_city_results_mean.csv (150 rows: 50 cities x 3 models averaged across seeds)
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from implement_new_plan.loss.log1p_mse import log1p_mse_loss
from implement_new_plan.models.od_models import TwoParameterGravity, PairwiseMLP, UrbanGNN
from implement_new_plan.data.dataset import load_raw_city, RawCityData
from implement_new_plan.data.urban_graph import build_radius_graph
from implement_new_plan.training.evaluate import (
    compute_cpc_pair,
    compute_cpc_norm_pair,
    compute_mae_pair,
    compute_mse_pair,
    compute_rmse_pair,
)

CANONICAL_SEEDS = [1, 10, 100]
FRACTIONS = [0.10, 0.20, 0.30, 0.50, 1.00]


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_city_splits(raw_city: RawCityData) -> Dict[str, np.ndarray]:
    """Generates strictly deterministic nested splits on positive support."""
    o = np.asarray(raw_city.pair_o_idx)
    d = np.asarray(raw_city.pair_d_idx)
    dist = np.asarray(raw_city.dist_km)
    trips = np.asarray(raw_city.pair_trips)
    mask = (o != d) & (dist > 0.0) & (trips >= 1.0)

    df_pairs = pd.DataFrame({'origin': o[mask], 'destination': d[mask], 'idx': np.where(mask)[0]})
    df_sorted = df_pairs.sort_values(['origin', 'destination']).reset_index(drop=True)
    n = len(df_sorted)
    rng = np.random.default_rng(42)
    perm = rng.permutation(n)

    idx_arr = df_sorted['idx'].values
    idx_30 = idx_arr[perm[:int(np.floor(0.30 * n))]]
    heldout_30 = idx_arr[perm[int(np.floor(0.30 * n)):]]

    return {
        'support_idx': idx_arr,
        'heldout_30': heldout_30,
        '0.10': idx_arr[perm[:int(np.floor(0.10 * n))]],
        '0.20': idx_arr[perm[:int(np.floor(0.20 * n))]],
        '0.30': idx_30,
        '0.50': idx_arr[perm[:int(np.floor(0.50 * n))]],
        '1.00': idx_arr[perm],
    }


def train_single_gravity(
    pop_raw_o: torch.Tensor,
    pop_raw_d: torch.Tensor,
    dist_raw: torch.Tensor,
    true_flow: torch.Tensor,
    seed: int,
    epochs: int = 40,
    lr: float = 2e-3,
) -> Tuple[TwoParameterGravity, List[Dict]]:
    set_seed(seed)
    model = TwoParameterGravity(init_G=0.0, init_alpha=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    trace = []
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        pred = model(pop_raw_o, pop_raw_d, dist_raw)
        loss = log1p_mse_loss(pred, true_flow)
        assert torch.isfinite(loss), f"Gravity loss is NaN/Inf at epoch {epoch}"
        loss.backward()
        optimizer.step()

        trace.append({
            "epoch": epoch,
            "G": float(model.G.item()),
            "alpha": float(model.alpha.item()),
            "train_loss": float(loss.item()),
        })

    return model, trace


def train_single_mlp(
    x_o: torch.Tensor,
    x_d: torch.Tensor,
    dist_std: torch.Tensor,
    true_flow: torch.Tensor,
    seed: int,
    epochs: int = 40,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    grad_clip: float = 5.0,
) -> PairwiseMLP:
    set_seed(seed)
    model = PairwiseMLP(node_in_dim=x_o.shape[-1], hidden_dim=64, dropout=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        pred = model(x_o, x_d, dist_std)
        loss = log1p_mse_loss(pred, true_flow)
        assert torch.isfinite(loss), f"MLP loss is NaN/Inf at epoch {epoch}"
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

    model.eval()
    return model


def train_single_gnn(
    x_all: torch.Tensor,
    spatial_edge_index: torch.Tensor,
    spatial_edge_dist_raw: torch.Tensor,
    pair_o: torch.Tensor,
    pair_d: torch.Tensor,
    dist_raw: torch.Tensor,
    pop_raw: torch.Tensor,
    true_flow: torch.Tensor,
    seed: int,
    epochs: int = 40,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    grad_clip: float = 5.0,
) -> UrbanGNN:
    set_seed(seed)
    model = UrbanGNN(node_in_dim=x_all.shape[-1], node_hidden_dim=64, node_out_dim=64, decoder_hidden_dim=64, dropout=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        pred = model(
            x=x_all,
            spatial_edge_index=spatial_edge_index,
            spatial_edge_dist_raw=spatial_edge_dist_raw,
            pair_o_idx=pair_o,
            pair_d_idx=pair_d,
            pair_distance_km_raw=dist_raw,
            population_raw=pop_raw,
        )
        loss = log1p_mse_loss(pred, true_flow)
        assert torch.isfinite(loss), f"GNN loss is NaN/Inf at epoch {epoch}"
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

    model.eval()
    return model
```

---

<a id="implement-new-plan-experiment-master-protocol-runner-py"></a>
## File: `implement_new_plan/experiment/master_protocol_runner.py` (282 lines)

```python
"""
Master Runner & Pipeline Execution for Research Protocol new_plan.md.

Covers:
- Strict Reproducibility & Multi-seed initialization (seeds = [1, 10, 100])
- Full-batch AdamW optimization (40 epochs, lr=2e-3, weight_decay=1e-4, grad_clip=5.0) on Train_f(s)
- Log1p-MSE loss: L = Mean( (log(1+T) - log(1+T_hat))^2 )
- 3 Baseline families: gravity_2param, pairwise_mlp, urban_gnn
- Positive Interzonal Support Omega_t^+: origin != destination, distance_km > 0, true_flow >= 1
- Zero-shot Transfer & Caching: predictions computed once per (source, target, model, seed, fraction)
- Experiment A: f=0.30, K=8, eps=0
- Experiment B: f in {0.10, 0.20, 0.30, 0.50, 1.00}, K=8, eps=0
- Experiment C: Master grid K in {2,4,8,12,20} x eps in {0, 0.01, ..., 0.10} at f=0.30
- Experiment D: Dose-matched Scaled Donor Control at f=0.30, K=8, eps=0
- 4-Tier Statistical Inference (Tier A, Tier B, Tier C, Tier D)
- Exact file outputs matching protocol schema.
"""

import os
import sys
import time
import math
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from implement_new_plan.loss.log1p_mse import log1p_mse_loss
from implement_new_plan.models.od_models import TwoParameterGravity, PairwiseMLP, UrbanGNN
from implement_new_plan.data.dataset import load_raw_city, RawCityData, NODE_FEATURE_COLUMNS
from implement_new_plan.data.urban_graph import build_radius_graph
from implement_new_plan.data.source_scaler import SourceCityFeatureScaler, SingleFeatureScaler, SKEWED_FEATURES
from implement_new_plan.calibration.source_bins import (
    assign_to_source_bins,
    build_target_dbd,
    calibrate_dbd,
    compute_pure_calibration_ratios,
    apply_pure_dbd_calibration,
)
from implement_new_plan.calibration.tv_noise import generate_exact_tv_noise, derive_noise_seed
from implement_new_plan.calibration.dose_matching import (
    compute_source_binned_dbd,
    compute_rms_log_ratio_dose,
    find_dose_matching_lambda,
    reconstruct_scaled_donor,
    fit_crossed_mixed_effects,
)
from implement_new_plan.training.evaluate import (
    compute_cpc_pair,
    compute_cpc_norm_pair,
    compute_mae_pair,
    compute_mse_pair,
    compute_rmse_pair,
    evaluate_calibration_transfer,
)
from implement_new_plan.calibration.statistical_inference import (
    aggregate_seeds,
    compute_target_city_summary,
    compute_global_target_inference,
    compute_source_city_summary,
    fit_crossed_random_effects,
    compute_scarcity_contrasts,
    compute_scarcity_contrast_target_summary,
    compute_scarcity_contrast_inference,
    fit_scarcity_contrast_mixed_effects,
    compute_gap_recovery,
    fit_scarcity_overall_mixed_effects,
)

CANONICAL_SEEDS = [1, 10, 100]
FRACTIONS = [0.10, 0.20, 0.30, 0.50, 1.00]
K_GRID = [2, 4, 8, 12, 20]
EPS_GRID = [0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10]
GLOBAL_NOISE_SEED = 42


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_city_splits(raw_city: RawCityData) -> Dict[str, np.ndarray]:
    """Generates strictly deterministic nested splits on positive support."""
    o = np.asarray(raw_city.pair_o_idx)
    d = np.asarray(raw_city.pair_d_idx)
    dist = np.asarray(raw_city.dist_km)
    trips = np.asarray(raw_city.pair_trips)
    mask = (o != d) & (dist > 0.0) & (trips >= 1.0)

    df_pairs = pd.DataFrame({'origin': o[mask], 'destination': d[mask], 'idx': np.where(mask)[0]})
    df_sorted = df_pairs.sort_values(['origin', 'destination']).reset_index(drop=True)
    n = len(df_sorted)
    rng = np.random.default_rng(42)
    perm = rng.permutation(n)

    idx_arr = df_sorted['idx'].values
    idx_30 = idx_arr[perm[:int(np.floor(0.30 * n))]]
    heldout_30 = idx_arr[perm[int(np.floor(0.30 * n)):]]

    return {
        'support_idx': idx_arr,
        'heldout_30': heldout_30,
        '0.10': idx_arr[perm[:int(np.floor(0.10 * n))]],
        '0.20': idx_arr[perm[:int(np.floor(0.20 * n))]],
        '0.30': idx_30,
        '0.50': idx_arr[perm[:int(np.floor(0.50 * n))]],
        '1.00': idx_arr[perm],
    }


def load_source_scaler(source_city: str, scalers_df: pd.DataFrame) -> Dict[str, Any]:
    sub = scalers_df[scalers_df['source_city'] == source_city]
    node_means = []
    node_stds = []
    node_transforms = []
    node_imputes = []
    zero_flags = []
    
    for feat in sub[sub['feature_type'] == 'node_feature']['feature_name'].values:
        row = sub[sub['feature_name'] == feat].iloc[0]
        node_means.append(float(row['mean']))
        node_stds.append(float(row['std']))
        node_transforms.append(row['transform'])
        node_imputes.append(float(row['imputation_value']))
        zero_flags.append(bool(row['zero_variance_flag']))
        
    dist_row = sub[sub['feature_type'] == 'pairwise_distance'].iloc[0]
    return {
        'node_means': np.array(node_means, dtype=np.float64),
        'node_stds': np.array(node_stds, dtype=np.float64),
        'node_transforms': node_transforms,
        'node_imputes': np.array(node_imputes, dtype=np.float64),
        'zero_flags': zero_flags,
        'dist_mean': float(dist_row['mean']),
        'dist_std': float(dist_row['std']),
        'dist_impute': float(dist_row['imputation_value']),
    }


def transform_nodes_with_scaler(raw_x: np.ndarray, scaler_dict: Dict[str, Any]) -> np.ndarray:
    x_out = np.zeros_like(raw_x, dtype=np.float32)
    for i in range(raw_x.shape[1]):
        vals = raw_x[:, i].copy().astype(np.float64)
        vals[~np.isfinite(vals)] = scaler_dict['node_imputes'][i]
        if scaler_dict['node_transforms'][i] == 'log1p_zscore':
            vals = np.log1p(np.maximum(0.0, vals))
        if scaler_dict['zero_flags'][i] or scaler_dict['node_stds'][i] < 1e-12:
            x_out[:, i] = 0.0
        else:
            x_out[:, i] = (vals - scaler_dict['node_means'][i]) / scaler_dict['node_stds'][i]
    return x_out


def impute_pop_with_scaler(raw_pop: np.ndarray, scaler_dict: Dict[str, Any]) -> np.ndarray:
    pop_arr = raw_pop.copy().astype(np.float64)
    # total_population is column 0
    pop_impute = scaler_dict['node_imputes'][0]
    pop_arr[~np.isfinite(pop_arr)] = pop_impute
    return np.maximum(0.0, pop_arr).astype(np.float32)


def transform_dist_with_scaler(raw_dist: np.ndarray, scaler_dict: Dict[str, Any]) -> np.ndarray:
    d = raw_dist.copy().astype(np.float64)
    d[~np.isfinite(d)] = scaler_dict['dist_impute']
    d_log = np.log1p(np.maximum(0.0, d))
    if scaler_dict['dist_std'] < 1e-12:
        return np.zeros_like(d, dtype=np.float32)
    return ((d_log - scaler_dict['dist_mean']) / scaler_dict['dist_std']).astype(np.float32)


def train_gravity(
    pop_raw_o: torch.Tensor,
    pop_raw_d: torch.Tensor,
    dist_raw: torch.Tensor,
    true_flow: torch.Tensor,
    seed: int,
    epochs: int = 40,
    lr: float = 2e-3,
) -> Tuple[TwoParameterGravity, List[Dict]]:
    set_seed(seed)
    model = TwoParameterGravity(init_G=0.0, init_alpha=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    trace = []
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        pred = model(pop_raw_o, pop_raw_d, dist_raw)
        loss = log1p_mse_loss(pred, true_flow)
        assert torch.isfinite(loss), f"Gravity loss is NaN/Inf at epoch {epoch}"
        loss.backward()
        optimizer.step()

        trace.append({
            "epoch": epoch,
            "G": float(model.G.item()),
            "alpha": float(model.alpha.item()),
            "train_loss": float(loss.item()),
        })

    return model, trace


def train_mlp(
    x_o: torch.Tensor,
    x_d: torch.Tensor,
    dist_std: torch.Tensor,
    true_flow: torch.Tensor,
    seed: int,
    epochs: int = 40,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    grad_clip: float = 5.0,
) -> PairwiseMLP:
    set_seed(seed)
    model = PairwiseMLP(node_in_dim=x_o.shape[-1], hidden_dim=64, dropout=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        pred = model(x_o, x_d, dist_std)
        loss = log1p_mse_loss(pred, true_flow)
        assert torch.isfinite(loss), f"MLP loss is NaN/Inf at epoch {epoch}"
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

    model.eval()
    return model


def train_gnn(
    x_all: torch.Tensor,
    spatial_edge_index: torch.Tensor,
    spatial_edge_dist_raw: torch.Tensor,
    pair_o: torch.Tensor,
    pair_d: torch.Tensor,
    dist_raw: torch.Tensor,
    pop_raw: torch.Tensor,
    true_flow: torch.Tensor,
    seed: int,
    epochs: int = 40,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    grad_clip: float = 5.0,
) -> UrbanGNN:
    set_seed(seed)
    model = UrbanGNN(node_in_dim=x_all.shape[-1], node_hidden_dim=64, node_out_dim=64, decoder_hidden_dim=64, dropout=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        pred = model(
            x=x_all,
            spatial_edge_index=spatial_edge_index,
            spatial_edge_dist_raw=spatial_edge_dist_raw,
            pair_o_idx=pair_o,
            pair_d_idx=pair_d,
            pair_distance_km_raw=dist_raw,
            population_raw=pop_raw,
        )
        loss = log1p_mse_loss(pred, true_flow)
        assert torch.isfinite(loss), f"GNN loss is NaN/Inf at epoch {epoch}"
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

    model.eval()
    return model
```

---

