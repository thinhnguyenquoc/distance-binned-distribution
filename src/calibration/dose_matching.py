"""
Dose-Matched Scaled Donor DBD Module for Structural Control Experiment (Experiment E).

Core Principle:
    [Source City Dictates Bin Boundaries for Baseline, Target, and Donor]
    For any transfer pair (s, t) and donor city d, all three distributions:
        q_{s,t}  (baseline predicted DBD of target t)
        p_{s,t}  (ground-truth target DBD of target t)
        p_{s,d}  (donor DBD of donor city d)
    MUST be evaluated on the SAME frozen source-specific bin system B^{(s,K)}.

Mathematical Formulation:
1. Target Dose (RMS Log-Ratio):
       Dose(p_{s,t}, q_{s,t}) = sqrt( 1/K * sum_{b=1}^K [ log( (p_{s,t,b} + eps) / (q_{s,t,b} + eps) ) ]^2 )
2. Donor Dose (RMS Log-Ratio):
       Dose(p_{s,d}, q_{s,t}) = sqrt( 1/K * sum_{b=1}^K [ log( (p_{s,d,b} + eps) / (q_{s,t,b} + eps) ) ]^2 )
3. Perturbation in Log-Ratio Space:
       z_{s,d,b} = log( (p_{s,d,b} + eps) / (q_{s,t,b} + eps) )
4. Scaled Perturbation and Reconstruction:
       p_scaled_{s,d,b}(lambda) = q_{s,t,b} * exp(lambda * z_{s,d,b}) / sum_k( q_{s,t,k} * exp(lambda * z_{s,d,k}) )
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
    from src.calibration.source_bins import assign_to_source_bins
    K = len(source_bin_edges) - 1
    bin_ids = assign_to_source_bins(distances_km, source_bin_edges)
    
    total_trips = float(np.sum(trips))
    if total_trips <= 0:
        raise ValueError("Cannot compute DBD on zero total trips.")
        
    dbd = np.zeros(K, dtype=np.float64)
    for b in range(K):
        dbd[b] = np.sum(trips[bin_ids == b]) / total_trips
    return dbd


def compute_rms_log_ratio_dose(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    """Computes RMS log-ratio dose between distribution p and baseline distribution q."""
    log_ratio = np.log((p + eps) / (q + eps))
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
    p_scaled = np.zeros_like(q_s_target, dtype=np.float64)
    if not np.any(pos_mask):
        return p_scaled

    z = np.log((p_s_donor[pos_mask] + eps) / (q_s_target[pos_mask] + eps))
    log_w = np.log(q_s_target[pos_mask]) + lambda_param * z
    log_w = log_w - np.max(log_w)
    w = np.exp(log_w)
    sum_w = float(np.sum(w))
    if sum_w > 0:
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
    Deterministic Brent Root Finding for lambda* in Experiment E.
    
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
            f"HARD FAILURE in Experiment E Dose Matching: Cannot bracket root up to lambda_high=1024!\n"
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
# DETERMINISTIC DONOR SELECTION (EXPERIMENT E PROTOCOL)
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
# STATISTICAL EVALUATION & PAIRED STRUCTURAL ADVANTAGE (EXPERIMENT E)
# ===========================================================================
def compute_experiment_e_aggregations(
    df_raw: "pd.DataFrame",
    output_dir: Optional[str] = "results",
) -> Tuple["pd.DataFrame", "pd.DataFrame", "pd.DataFrame"]:
    r"""
    Processes Experiment E results according to the paired structural advantage protocol:
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
    inference_records = []
    for model, m_group in df_target_summary.groupby("model"):
        H = m_group["mean_structural_advantage"].values
        n_targets = len(H)
        assert n_targets == 50, f"Expected 50 targets, got {n_targets}"

        # Descriptive metrics
        mean_H = float(np.mean(H))
        median_H = float(np.median(H))
        q75, q25 = np.percentile(H, [75, 25])
        iqr_H = float(q75 - q25)
        pos_targets = int(np.sum(H > 0))

        # Bootstrap 95% CI (10,000 resamples on 50 H_t values)
        rng = np.random.RandomState(42)
        boot_means = [np.mean(rng.choice(H, size=n_targets, replace=True)) for _ in range(10000)]
        ci_low = float(np.percentile(boot_means, 2.5))
        ci_high = float(np.percentile(boot_means, 97.5))

        # Wilcoxon signed-rank test
        try:
            w_stat, w_p = wilcoxon(H, alternative="greater")
            w_stat, w_p = float(w_stat), float(w_p)
        except Exception:
            w_stat, w_p = float("nan"), float("nan")

        # Crossed mixed-effects model on pair deltas
        pair_data = df_mean[df_mean["model"] == model]
        beta0, se, ci_l, ci_h, var_s, var_t, var_eps = fit_crossed_mixed_effects(
            pair_data, outcome_col="structural_advantage_delta"
        )

        inference_records.append({
            "model": model,
            "mean_H": round(mean_H, 6),
            "median_H": round(median_H, 6),
            "IQR_H": round(iqr_H, 6),
            "bootstrap_ci_low": round(ci_low, 6),
            "bootstrap_ci_high": round(ci_high, 6),
            "wilcoxon_stat": round(w_stat, 2) if np.isfinite(w_stat) else "NA",
            "wilcoxon_p": f"{w_p:.6e}" if np.isfinite(w_p) else "NA",
            "positive_targets": pos_targets,
            "n_targets": n_targets,
            "mixed_beta0": round(beta0, 6),
            "mixed_se": round(se, 6),
            "mixed_ci_low": round(ci_l, 6),
            "mixed_ci_high": round(ci_h, 6),
            "source_variance": round(var_s, 6),
            "target_variance": round(var_t, 6),
            "residual_variance": round(var_eps, 6),
        })

    df_inference = pd.DataFrame(inference_records)

    # Save outputs if path provided
    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        df_mean.to_csv(out_path / "structural_control_mean.csv", index=False)
        df_target_summary.to_csv(out_path / "structural_control_target_summary.csv", index=False)
        df_inference.to_csv(out_path / "structural_control_inference.csv", index=False)
        print(f"Saved Experiment E summary files to {out_path}")

    return df_mean, df_target_summary, df_inference


def fit_crossed_mixed_effects(
    df: "pd.DataFrame",
    outcome_col: str = "structural_advantage_delta"
) -> Tuple[float, float, float, float, float, float, float]:
    r"""
    Fits a crossed random effects model: delta ~ 1 + (1|source) + (1|target).
    Uses statsmodels MixedLM or two-way ANOVA decomposition if formula unavailable.
    """
    y = df[outcome_col].values
    sources = df["source_city"].astype("category")
    targets = df["target_city"].astype("category")
    
    # Try statsmodels MixedLM
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        # Crossed random effects approx via variance components
        model = smf.mixedlm(f"{outcome_col} ~ 1", df, groups=df["source_city"], vc_formula={"target": "0 + C(target_city)"})
        fit = model.fit(reml=True)
        beta0 = float(fit.params["Intercept"])
        se = float(fit.bse["Intercept"])
        ci_l = beta0 - 1.96 * se
        ci_h = beta0 + 1.96 * se
        var_s = float(fit.cov_re.iloc[0, 0]) if hasattr(fit.cov_re, 'iloc') else 0.0
        var_t = float(fit.vcomp[0]) if len(fit.vcomp) > 0 else 0.0
        var_eps = float(fit.scale)
        return beta0, se, ci_l, ci_h, var_s, var_t, var_eps
    except Exception:
        # Fallback to standard two-way random-effects ANOVA estimation
        n = len(y)
        grand_mean = float(np.mean(y))
        s_means = df.groupby("source_city")[outcome_col].mean().values
        t_means = df.groupby("target_city")[outcome_col].mean().values
        var_s = max(0.0, float(np.var(s_means) - np.var(y) / len(t_means)))
        var_t = max(0.0, float(np.var(t_means) - np.var(y) / len(s_means)))
        var_eps = max(1e-8, float(np.var(y) - var_s - var_t))
        se = float(np.sqrt((var_s / len(s_means)) + (var_t / len(t_means)) + (var_eps / n)))
        ci_l = grand_mean - 1.96 * se
        ci_h = grand_mean + 1.96 * se
        return grand_mean, se, ci_l, ci_h, var_s, var_t, var_eps

