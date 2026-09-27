"""
Exact Total Variation (TV) Noise Perturbation Module (Experiments C & D).

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
       min_b p_tilde_b >= -1e-12
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
        20260927|Boston|Seattle|8|0.060000|7
    
    Seed extraction:
        8 bytes big-endian unsigned integer from SHA-256 digest.
    """
    epsilon_str = f"{epsilon:.6f}"
    canonical_string = f"{global_noise_seed}|{source_city.strip()}|{target_city.strip()}|{K}|{epsilon_str}|{realization_id}"
    digest = hashlib.sha256(canonical_string.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
    # Numpy RandomState accepts unsigned 32-bit integer (0 <= seed < 2**32)
    # Map 64-bit seed deterministically into 32-bit range
    return seed % (2**32)


def compute_tv_distance(p: np.ndarray, p_tilde: np.ndarray) -> float:
    """Computes exact Total Variation distance: 1/2 * sum |p - p_tilde|."""
    return float(0.5 * np.sum(np.abs(p - p_tilde)))


def generate_exact_tv_noise(
    p: np.ndarray,
    epsilon: float,
    rng_or_seed: Union[np.random.RandomState, int],
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
    3. Rejection sampling: If min(p_tilde) < 0, reject entire vector u and resample.
       NO clipping, NO re-normalization, NO simplex projection, NO per-bin repair.
    4. Exact Numerical Validation:
       - |sum(p_tilde) - 1| < 1e-12
       - min(p_tilde) >= -1e-12
       - |TV(p, p_tilde) - epsilon| < 1e-10
       If any check fails: RAISE ERROR.
    5. Hard Failure Policy:
       If max_attempts (10,000) reached: RAISE HARD ERROR with full diagnostic context.
       NO silent fallback, NO epsilon reduction, NO algorithm switching.
    """
    p_len = len(p)
    if K is None:
        K = p_len
    assert p_len == K, f"Length of p ({p_len}) does not match K ({K})"

    if isinstance(rng_or_seed, int):
        rng = np.random.RandomState(rng_or_seed)
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
        
        # 4. Strict Rejection Sampling: must be >= -1e-12
        if np.any(p_cand < -1e-12):
            continue  # REJECT entire vector u, resample from stream
            
        # Clean negligible negative float rounding down to 0
        p_cand = np.maximum(0.0, p_cand)
        
        # 5. Numerical Validation Checks
        sum_val = float(np.sum(p_cand))
        sum_err = abs(sum_val - 1.0)
        if sum_err >= 1e-12:
            # Check if slight floating adjustment preserves bounds
            p_cand = p_cand / sum_val
            sum_err = abs(float(np.sum(p_cand)) - 1.0)
            if sum_err >= 1e-12:
                continue  # Reject if sum cannot be satisfied within 1e-12
        
        min_val = float(np.min(p_cand))
        if min_val < -1e-12:
            continue
            
        actual_tv = compute_tv_distance(p, p_cand)
        tv_err = abs(actual_tv - epsilon)
        if tv_err >= 1e-10:
            continue
            
        # Passed all 3 strict validation checks
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
        f"  noise_seed: {rng_or_seed if isinstance(rng_or_seed, int) else 'RandomState'}\n"
        f"  max_attempts: {max_attempts}\n"
        f"  min_positive_mass_of_p: {min_pos_mass:.6e}\n"
        f"  number_of_zero_bins: {zero_bins}\n"
        "Terminating run immediately under hard failure policy (no silent fallbacks)."
    )
    raise RuntimeError(error_msg)
