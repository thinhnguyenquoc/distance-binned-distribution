"""
Tests for Isotonic Monotonic Inversion and Observation Equivalence (m*, q*).
(Tests T32 to T36)
"""

import pytest
import numpy as np
from sklearn.isotonic import IsotonicRegression
from od_plan_tester.project_adapter import _interpolate_m_star


@pytest.mark.reference
def test_isotonic_monotonicity_enforcement():
    """T32: Isotonic regression enforces non-decreasing monotonic curve R_c(m)."""
    m_vals = [100.0, 500.0, 1000.0, 5000.0, 10000.0]
    # Noisy non-monotonic raw CPCs
    raw_cpcs = [0.45, 0.42, 0.50, 0.49, 0.55]

    iso = IsotonicRegression(increasing=True)
    mono_cpcs = iso.fit_transform(m_vals, raw_cpcs)

    assert all(mono_cpcs[i] <= mono_cpcs[i + 1] for i in range(len(mono_cpcs) - 1))


@pytest.mark.reference
def test_isotonic_plateau_leftmost_crossing():
    """T33: Plateau handling: leftmost crossing picks minimal sample size m*."""
    m_finite = [100.0, 500.0, 1000.0, 5000.0, 10000.0, 50000.0]
    mean_cpcs = [0.40, 0.50, 0.50, 0.50, 0.60, 0.70]
    oracle_cpc = 0.75
    total_trips = 1000000.0

    # Target CPC is 0.50, which is reached at m = 500.0 (plateau at 500, 1000, 5000)
    m_star, status = _interpolate_m_star(0.50, m_finite, mean_cpcs, oracle_cpc, total_trips)
    assert pytest.approx(500.0, abs=1.0) == m_star


@pytest.mark.reference
def test_isotonic_boundary_below_min_grid():
    """T34: Boundary case: target CPC <= min grid CPC returns m* = m_min with status 'below_min_grid'."""
    m_finite = [100.0, 500.0, 1000.0, 5000.0]
    mean_cpcs = [0.45, 0.50, 0.55, 0.60]
    oracle_cpc = 0.70
    total_trips = 500000.0

    m_star, status = _interpolate_m_star(0.40, m_finite, mean_cpcs, oracle_cpc, total_trips)
    assert m_star == 100.0
    assert status == "below_min_grid"


@pytest.mark.reference
def test_isotonic_boundary_at_oracle_ceiling():
    """T35: Boundary case: target CPC >= oracle CPC returns m* = total_trips with status 'at_oracle_ceiling'."""
    m_finite = [100.0, 500.0, 1000.0, 5000.0]
    mean_cpcs = [0.45, 0.50, 0.55, 0.60]
    oracle_cpc = 0.70
    total_trips = 500000.0

    m_star, status = _interpolate_m_star(0.72, m_finite, mean_cpcs, oracle_cpc, total_trips)
    assert m_star == total_trips
    assert status == "at_oracle_ceiling"


@pytest.mark.contract
def test_qstar_ratio_computation():
    """T36: Observation equivalence ratio q* = m* / T_total is strictly <= 1.0 even when T_inter < 100,000."""
    total_trips = 45000.0  # smaller than 100k grid level
    m_finite = [100.0, 500.0, 1000.0, 5000.0, 10000.0, 50000.0, 100000.0]
    mean_cpcs = [0.40, 0.45, 0.50, 0.55, 0.60, 0.68, 0.70]
    oracle_cpc = 0.68

    m_star, status = _interpolate_m_star(0.68, m_finite, mean_cpcs, oracle_cpc, total_trips)
    assert m_star <= total_trips
    q_star = m_star / total_trips
    assert 0.0 <= q_star <= 1.0
