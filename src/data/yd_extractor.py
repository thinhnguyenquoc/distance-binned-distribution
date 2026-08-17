"""
Y_D Extractor for Oracle and Real Meta Mobility Data.

Bins:
    0: 0 km (intrazonal)
    1: (0, 10) km
    2: [10, 100) km
    3: 100+ km

Oracle Y_D:
    Computed directly from ground truth OD flows over candidate support Omega_c:
    Y_D^oracle[k] = sum_{(i,j) in Omega_c and B_k} T_ij / sum_{(i,j) in Omega_c} T_ij

Real Y_D:
    Extracted from meta_prior CSV files (Meta mobility data aggregated to county/city).
"""

import os
import glob
import pandas as pd
import numpy as np
import torch
from pathlib import Path


def extract_yd_oracle(pair_trips: torch.Tensor, bin_labels: torch.Tensor) -> np.ndarray:
    """
    Computes oracle distance-bin distribution from GT flows.

    Args:
        pair_trips: (E,) positive trip counts on Omega_c.
        bin_labels: (E,) bin index (0, 1, 2, 3).

    Returns:
        np.ndarray of shape (4,) normalized such that sum = 1.0.
    """
    yd = np.zeros(4, dtype=np.float64)
    trips_np = pair_trips.detach().cpu().numpy()
    bins_np = bin_labels.detach().cpu().numpy()

    total_flow = float(np.sum(trips_np))
    if total_flow == 0:
        return np.array([0.25, 0.25, 0.25, 0.25])

    for k in range(4):
        mask = (bins_np == k)
        yd[k] = np.sum(trips_np[mask])

    return yd / total_flow


def extract_yd_real(city_name: str, meta_prior_dir: str = "meta_prior") -> np.ndarray | None:
    """
    Extracts real Y_D distribution from Meta mobility datasets for the city.
    Averages over all available temporal snapshots.

    Returns:
        np.ndarray of shape (4,) or None if not available.
    """
    meta_dir = Path(meta_prior_dir)
    if not meta_dir.exists():
        return None

    # Search for matching files for the city
    pattern = str(meta_dir / f"*{city_name}*.csv")
    files = glob.glob(pattern)

    if not files:
        # Try finding csv files containing city name case-insensitively
        all_csvs = list(meta_dir.glob("*.csv"))
        files = [str(f) for f in all_csvs if city_name.lower() in f.name.lower()]

    if not files:
        return None

    # Read and aggregate across files
    collected_bins = []
    for f in files:
        try:
            df = pd.read_csv(f)
            # Expecting columns for distance bins or proportion columns
            # Typical Meta columns: 'length_0', 'length_0_10', 'length_10_100', 'length_100_plus'
            # or 'bin_0', 'bin_1', 'bin_2', 'bin_3'
            cols = [c.lower() for c in df.columns]
            for candidate in [
                ['length_0', 'length_0_10', 'length_10_100', 'length_100_plus'],
                ['bin_0', 'bin_1', 'bin_2', 'bin_3'],
                ['p_0', 'p_0_10', 'p_10_100', 'p_100'],
            ]:
                if all(c in cols for c in candidate):
                    vals = df[[df.columns[cols.index(c)] for c in candidate]].mean().values
                    if np.sum(vals) > 0:
                        collected_bins.append(vals / np.sum(vals))
                    break
        except Exception:
            continue

    if not collected_bins:
        return None

    mean_yd = np.mean(collected_bins, axis=0)
    return mean_yd / np.sum(mean_yd)


if __name__ == "__main__":
    from src.data.dataset import load_city
    cd = load_city("Raleigh", "data")
    yd_oracle = extract_yd_oracle(cd.pair_trips, cd.bin_labels)
    print(f"Raleigh Oracle Y_D: {yd_oracle} (sum={yd_oracle.sum():.4f})")
