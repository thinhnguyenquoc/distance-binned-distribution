"""
Cross-city Statistical Analysis of q* and m* (RQ2).

Distinctly computes:
    - Real (Primary RQ2): m*_real, q*_real = m*_real / T_total against Y_D^{Meta}
    - Oracle (Benchmark): m*_oracle, q*_oracle against Y_D^{oracle}
"""

import numpy as np
from typing import List, Dict, Any


def _summary_stats(arr: np.ndarray, ddof: int = 1) -> Dict[str, Any]:
    """Compute summary statistics with sample standard deviation (ddof=1) and sample size n."""
    n = int(len(arr))
    std_val = float(np.std(arr, ddof=ddof)) if n > 1 else 0.0
    return {
        "n": n,
        "mean": float(np.mean(arr)),
        "std": std_val,
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)) if n > 0 else 0.0,
        "p75": float(np.percentile(arr, 75)) if n > 0 else 0.0,
        "min": float(np.min(arr)) if n > 0 else 0.0,
        "max": float(np.max(arr)) if n > 0 else 0.0,
    }


def analyze_qstar(city_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = {"n_cities": len(city_results)}

    # Oracle
    m_oracle = np.array([r["m_star_oracle"] for r in city_results if r["m_star_oracle"] is not None])
    q_oracle = np.array([r["q_star_oracle"] for r in city_results if r["q_star_oracle"] is not None])
    if len(m_oracle) > 0:
        out["oracle"] = {
            "m_star": _summary_stats(m_oracle),
            "q_star": _summary_stats(q_oracle),
        }

    # Real (Primary)
    m_real = np.array([r["m_star_real"] for r in city_results if r["m_star_real"] is not None])
    q_real = np.array([r["q_star_real"] for r in city_results if r["q_star_real"] is not None])
    if len(m_real) > 0:
        out["real"] = {
            "n_cities": len(m_real),
            "m_star": _summary_stats(m_real),
            "q_star": _summary_stats(q_real),
        }

    return out
