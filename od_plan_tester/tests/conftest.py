"""
Pytest configuration and shared fixtures for od_plan_tester.
"""

import pytest
import torch
import numpy as np


@pytest.fixture(autouse=True)
def set_random_seeds():
    torch.manual_seed(42)
    np.random.seed(42)


@pytest.fixture
def sample_coordinates():
    """Returns sample lat/lon coordinates for spatial graph tests."""
    return np.array([
        [-84.3880, 33.7490],  # Atlanta core 1
        [-84.3900, 33.7500],  # Atlanta core 2 (~0.25 km)
        [-84.4000, 33.7600],  # Atlanta mid (~1.5 km)
        [-84.4500, 33.8000],  # Atlanta sub (~8 km)
        [-85.0000, 34.2000],  # Distant isolated tract (~70 km)
    ])


@pytest.fixture
def synthetic_od_flows():
    """Returns mock true flow counts and distance bins."""
    torch.manual_seed(42)
    t_true = torch.tensor([12.0, 45.0, 150.0, 8.0, 220.0, 35.0, 90.0, 15.0])
    bins = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    return t_true, bins
