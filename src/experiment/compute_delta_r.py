"""
Cross-city Statistical Analysis of Delta R (RQ1).

Computes across all 50 held-out cities:
    - Delta R_c = R_c^{YD, real} - R_c^{ZS}  (and Delta R_c^{oracle} = R_c^{YD, oracle} - R_c^{ZS})
    - Mean +- std, median, IQR
    - P(Delta R_c > 0) — fraction of cities improved
    - Paired Wilcoxon signed-rank test and paired t-test for statistical significance
    - Realization gap = R_c^{YD, oracle} - R_c^{YD, real}
"""

import numpy as np
from scipy import stats
from typing import List, Dict, Any


def analyze_delta_r(city_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes statistical metrics for RQ1 across evaluated cities.

    Args:
        city_results: list of dicts from run_target_city_experiments().

    Returns:
        dictionary of statistical summaries and significance tests.
    """
    cities = [r["city"] for r in city_results]
    m0_cpcs = np.array([r["M0"]["cpc"] for r in city_results])
    oracle_cpcs = np.array([r["M1_oracle"]["cpc"] for r in city_results])
    delta_oracle = oracle_cpcs - m0_cpcs

    real_results = [r for r in city_results if r["M1_real"] is not None]
    has_real = len(real_results) > 0

    analysis = {
        "n_cities_evaluated": len(cities),
        "oracle": {
            "m0_cpc_mean": float(np.mean(m0_cpcs)),
            "m0_cpc_median": float(np.median(m0_cpcs)),
            "m1_oracle_cpc_mean": float(np.mean(oracle_cpcs)),
            "m1_oracle_cpc_median": float(np.median(oracle_cpcs)),
            "delta_r_mean": float(np.mean(delta_oracle)),
            "delta_r_std": float(np.std(delta_oracle)),
            "delta_r_median": float(np.median(delta_oracle)),
            "delta_r_iqr": float(np.percentile(delta_oracle, 75) - np.percentile(delta_oracle, 25)),
            "p_improved": float(np.mean(delta_oracle > 0)),
        }
    }

    # Significance test (Wilcoxon signed-rank & paired t-test)
    if len(delta_oracle) >= 5:
        try:
            w_stat, w_p = stats.wilcoxon(oracle_cpcs, m0_cpcs, alternative="greater")
            t_stat, t_p = stats.ttest_rel(oracle_cpcs, m0_cpcs, alternative="greater")
            analysis["oracle"]["wilcoxon_p"] = float(w_p)
            analysis["oracle"]["ttest_p"] = float(t_p)
        except Exception as e:
            analysis["oracle"]["test_error"] = str(e)

    if has_real:
        m0_real_cpcs = np.array([r["M0"]["cpc"] for r in real_results])
        real_cpcs = np.array([r["M1_real"]["cpc"] for r in real_results])
        delta_real = real_cpcs - m0_real_cpcs
        gap = np.array([r["realization_gap"] for r in real_results if r["realization_gap"] is not None])

        analysis["real"] = {
            "n_cities": len(real_results),
            "m1_real_cpc_mean": float(np.mean(real_cpcs)),
            "m1_real_cpc_median": float(np.median(real_cpcs)),
            "delta_r_mean": float(np.mean(delta_real)),
            "delta_r_std": float(np.std(delta_real)),
            "delta_r_median": float(np.median(delta_real)),
            "p_improved": float(np.mean(delta_real > 0)),
            "realization_gap_mean": float(np.mean(gap)) if len(gap) > 0 else None,
            "realization_gap_median": float(np.median(gap)) if len(gap) > 0 else None,
        }
        if len(delta_real) >= 5:
            try:
                w_stat, w_p = stats.wilcoxon(real_cpcs, m0_real_cpcs, alternative="greater")
                analysis["real"]["wilcoxon_p"] = float(w_p)
            except Exception:
                pass

    return analysis
