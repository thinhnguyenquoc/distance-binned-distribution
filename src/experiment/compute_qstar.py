"""
Cross-city Statistical Analysis of q* and m* (RQ2).

Computes across all 50 held-out cities:
    - m*_c: absolute number of sampled trips required
    - q*_c = m*_c / T_c^{total}: fraction of total trips required
    - Distributions (median, mean +- std, IQR)
    - Relationship between city size (total trips) and q*
"""

import numpy as np
from typing import List, Dict, Any


def analyze_qstar(city_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes statistical distribution of q* and m* across evaluated cities.

    Args:
        city_results: list of dicts from run_target_city_experiments().

    Returns:
        dictionary of summaries for m* and q*.
    """
    m_stars = np.array([r["m_star"] for r in city_results])
    q_stars = np.array([r["q_star"] for r in city_results])
    total_trips = np.array([r["total_trips"] for r in city_results])

    return {
        "n_cities": len(city_results),
        "m_star": {
            "mean": float(np.mean(m_stars)),
            "std": float(np.std(m_stars)),
            "median": float(np.median(m_stars)),
            "p25": float(np.percentile(m_stars, 25)),
            "p75": float(np.percentile(m_stars, 75)),
            "min": float(np.min(m_stars)),
            "max": float(np.max(m_stars)),
        },
        "q_star": {
            "mean": float(np.mean(q_stars)),
            "std": float(np.std(q_stars)),
            "median": float(np.median(q_stars)),
            "p25": float(np.percentile(q_stars, 25)),
            "p75": float(np.percentile(q_stars, 75)),
            "min": float(np.min(q_stars)),
            "max": float(np.max(q_stars)),
        },
    }
