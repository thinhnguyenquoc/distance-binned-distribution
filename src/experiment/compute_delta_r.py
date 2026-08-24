"""
Cross-city Statistical Analysis of Moving-Bin Calibration Results (RQ1).
"""

import numpy as np
from scipy import stats
from typing import List, Dict, Any


def _compute_stats(arr: np.ndarray, ddof: int = 1) -> Dict[str, Any]:
    """Compute summary statistics with sample standard deviation (ddof=1) and sample size n."""
    n = int(len(arr))
    std_val = float(np.std(arr, ddof=ddof)) if n > 1 else 0.0
    return {
        "n": n,
        "mean": float(np.mean(arr)),
        "std": std_val,
        "median": float(np.median(arr)),
        "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)) if n > 0 else 0.0,
        "p25": float(np.percentile(arr, 25)) if n > 0 else 0.0,
        "p75": float(np.percentile(arr, 75)) if n > 0 else 0.0,
        "min": float(np.min(arr)) if n > 0 else 0.0,
        "max": float(np.max(arr)) if n > 0 else 0.0,
    }


def _fold_stratified_bootstrap(values: np.ndarray, fold_ids: np.ndarray, n_boot: int = 10000) -> tuple[float, float]:
    """Fold-stratified bootstrap 95% CI for the mean of a given metric."""
    folds = {}
    for i, f in enumerate(fold_ids):
        if f not in folds:
            folds[f] = []
        folds[f].append(values[i])
    
    if len(folds) == 0 or sum(len(v) for v in folds.values()) < 2:
        return 0.0, 0.0
        
    rng = np.random.default_rng(42)
    boot_means = []
    for _ in range(n_boot):
        samp = []
        for f, vals in folds.items():
            if len(vals) > 0:
                samp.extend(rng.choice(vals, size=len(vals), replace=True))
        boot_means.append(np.mean(samp))
    
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def analyze_delta_r(city_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    n_cities = len(city_results)

    # Primary: Interzonal CPC on Omega_c^+
    m0_inter = np.array([r["M0"]["cpc_inter"] for r in city_results])
    
    analysis = {
        "n_cities_evaluated": n_cities,
        "std_definition": "sample_sd_ddof_1",
        "missingness_correlations": {}
    }

    scales = [
        ("city", "M1_city_oracle_obs"),
        ("county", "M1_county_oracle_obs"),
        ("subzone", "M1_subzone_oracle_obs")
    ]

    for scale_name, scale_key in scales:
        m1_inter = np.array([r[scale_key]["cpc_inter"] for r in city_results])
        delta_inter = m1_inter - m0_inter
        
        # Fold-stratified bootstrap
        fold_ids = np.array([r.get("fold", -1) for r in city_results])
        ci_low, ci_high = _fold_stratified_bootstrap(delta_inter, fold_ids)
        
        # Missingness Correlations
        missingness = {}
        for feature in ["rho_c", "n_tracts", "mean_distance", "average_flow", "short_long_ratio"]:
            feature_vals = np.array([r.get(feature, 0.0) for r in city_results])
            if np.std(feature_vals) > 0:
                rho, _ = stats.pearsonr(feature_vals, delta_inter)
                missingness[f"corr_with_{feature}"] = float(rho)
        analysis["missingness_correlations"][scale_name] = missingness

        analysis[scale_name] = {
            "m0_cpc_inter": _compute_stats(m0_inter),
            "m1_cpc_inter": _compute_stats(m1_inter),
            "delta_cpc_inter": {**_compute_stats(delta_inter), "ci_95_lower": ci_low, "ci_95_upper": ci_high},
            "p_improved": float(np.mean(delta_inter > 0)),
        }

        if len(delta_inter) >= 5:
            w_stat, w_p_two = stats.wilcoxon(m1_inter, m0_inter, alternative="two-sided")
            _, w_p_one = stats.wilcoxon(m1_inter, m0_inter, alternative="greater")
            
            # Compute matched-pairs rank-biserial correlation
            diff = m1_inter - m0_inter
            diff = diff[diff != 0]
            ranks = stats.rankdata(np.abs(diff))
            w_plus = np.sum(ranks[diff > 0])
            w_minus = np.sum(ranks[diff < 0])
            r_rb = (w_plus - w_minus) / (w_plus + w_minus) if (w_plus + w_minus) > 0 else 0.0

            analysis[scale_name]["wilcoxon_one_sided_p"] = float(w_p_one)
            analysis[scale_name]["wilcoxon_two_sided_p"] = float(w_p_two)
            analysis[scale_name]["rank_biserial_r"] = float(r_rb)

    return analysis
