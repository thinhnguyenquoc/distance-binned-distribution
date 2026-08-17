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


def analyze_delta_r(city_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    n_cities = len(city_results)

    # Primary: Interzonal CPC on Omega_c^+
    m0_inter = np.array([r["M0"]["cpc_inter"] for r in city_results])
    m1_o_inter = np.array([r["M1_oracle_plus"]["cpc_inter"] for r in city_results])
    delta_o_inter = m1_o_inter - m0_inter

    m0_full = np.array([r["M0"]["cpc_full"] for r in city_results])
    m1_o_full = np.array([r["M1_oracle_plus"]["cpc_full"] for r in city_results])
    delta_o_full = m1_o_full - m0_full

    analysis = {
        "n_cities_evaluated": n_cities,
        "std_definition": "sample_sd_ddof_1",
        "oracle_plus": {
            "m0_cpc_inter": _compute_stats(m0_inter),
            "m1_oracle_cpc_inter": _compute_stats(m1_o_inter),
            "delta_r_inter": _compute_stats(delta_o_inter),
            "p_improved": float(np.mean(delta_o_inter > 0)),
            "m0_cpc_full": _compute_stats(m0_full),
            "m1_oracle_cpc_full": _compute_stats(m1_o_full),
            "delta_r_full": _compute_stats(delta_o_full),
        },
    }

    if len(delta_o_inter) >= 5:
        _, w_p_one = stats.wilcoxon(m1_o_inter, m0_inter, alternative="greater")
        _, w_p_two = stats.wilcoxon(m1_o_inter, m0_inter, alternative="two-sided")
        analysis["oracle_plus"]["wilcoxon_one_sided_p"] = float(w_p_one)
        analysis["oracle_plus"]["wilcoxon_two_sided_p"] = float(w_p_two)

    # Primary Real Moving-Bin Analysis (M1^{real, +})
    real_results = [r for r in city_results if r["M1_real_plus"] is not None]
    if real_results:
        m0_r_inter = np.array([r["M0"]["cpc_inter"] for r in real_results])
        m1_r_inter = np.array([r["M1_real_plus"]["cpc_inter"] for r in real_results])
        delta_r_inter = m1_r_inter - m0_r_inter

        m0_r_full = np.array([r["M0"]["cpc_full"] for r in real_results])
        m1_r_full = np.array([r["M1_real_plus"]["cpc_full"] for r in real_results])
        delta_r_full = m1_r_full - m0_r_full

        gaps_inter = np.array([r["realization_gap_plus"] for r in real_results])
        overlaps = np.array([r["distributional_overlap"] for r in real_results if r["distributional_overlap"] is not None])

        analysis["real_plus"] = {
            "n_cities": len(real_results),
            "m1_real_cpc_inter": _compute_stats(m1_r_inter),
            "delta_r_inter": _compute_stats(delta_r_inter),
            "p_improved": float(np.mean(delta_r_inter > 0)),
            "m1_real_cpc_full": _compute_stats(m1_r_full),
            "delta_r_full": _compute_stats(delta_r_full),
            "distributional_overlap": _compute_stats(overlaps) if len(overlaps) > 0 else None,
            "realization_gap": {
                "mean": float(np.mean(gaps_inter)),
                "std": float(np.std(gaps_inter, ddof=1)) if len(gaps_inter) > 1 else 0.0,
                "median": float(np.median(gaps_inter)),
                "mae": float(np.mean(np.abs(gaps_inter))),
                "min": float(np.min(gaps_inter)),
                "max": float(np.max(gaps_inter)),
            },
        }

        if len(delta_r_inter) >= 5:
            _, w_p_one = stats.wilcoxon(m1_r_inter, m0_r_inter, alternative="greater")
            _, w_p_two = stats.wilcoxon(m1_r_inter, m0_r_inter, alternative="two-sided")
            analysis["real_plus"]["wilcoxon_one_sided_p"] = float(w_p_one)
            analysis["real_plus"]["wilcoxon_two_sided_p"] = float(w_p_two)

    # 4-Bin Legacy Ablation Analysis
    ablation_results = [r for r in city_results if r.get("M1_4bin_ablation") is not None and r.get("M1_real_plus") is not None]
    if ablation_results:
        m1_ab_inter = np.array([r["M1_4bin_ablation"]["cpc_inter"] for r in ablation_results])
        m0_ab_inter = np.array([r["M0"]["cpc_inter"] for r in ablation_results])
        delta_ab_inter = m1_ab_inter - m0_ab_inter

        m1_r_inter = np.array([r["M1_real_plus"]["cpc_inter"] for r in ablation_results])
        delta_r_inter = m1_r_inter - m0_ab_inter

        # Ablation Penalty = Delta R_real+ - Delta R_4bin
        ablation_penalty = delta_r_inter - delta_ab_inter

        m1_ab_full = np.array([r["M1_4bin_ablation"]["cpc_full"] for r in ablation_results])
        m0_ab_full = np.array([r["M0"]["cpc_full"] for r in ablation_results])
        delta_ab_full = m1_ab_full - m0_ab_full

        analysis["4bin_ablation"] = {
            "n_cities": len(ablation_results),
            "m1_4bin_cpc_inter": _compute_stats(m1_ab_inter),
            "delta_r_inter": _compute_stats(delta_ab_inter),
            "p_improved_inter": float(np.mean(delta_ab_inter > 0)),
            "ablation_penalty_inter": _compute_stats(ablation_penalty),
            "m1_4bin_cpc_full": _compute_stats(m1_ab_full),
            "delta_r_full": _compute_stats(delta_ab_full),
            "p_improved_full": float(np.mean(delta_ab_full > 0)),
        }

        if len(delta_ab_inter) >= 5:
            _, w_p_one = stats.wilcoxon(m1_ab_inter, m0_ab_inter, alternative="greater")
            _, w_p_two = stats.wilcoxon(m1_ab_inter, m0_ab_inter, alternative="two-sided")
            analysis["4bin_ablation"]["wilcoxon_one_sided_p_inter"] = float(w_p_one)
            analysis["4bin_ablation"]["wilcoxon_two_sided_p_inter"] = float(w_p_two)

    return analysis
