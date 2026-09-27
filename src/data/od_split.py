"""
OD Split Generator Module: Fixed 30/70 Uniform Random Split on Positive OD Support.

Protocol Requirements:
1. Candidate Pool:
       Only OD pairs belonging to positive interzonal OD support Omega_s^+:
       (origin != destination) & (distance_km > 0) & (flow >= 1)
   Never split on full matrix, zero-flow pairs, or all possible tract pairs.

2. Uniform Random Split:
       N_train = floor(0.30 * N_support)
       N_eval = N_support - N_train
   Sampling: Uniform random sampling without replacement using split_seed = 42.

3. Deterministic Sorting:
       Sort candidate pairs by (origin, destination) before permutation
       to ensure identical splits regardless of disk / CSV row order.

4. No Stratification / Coverage Heuristics:
       Do NOT stratify by distance, flow magnitude, origin, destination, or tract.
       Do NOT retry to achieve artificial coverage. Accept the split as-is.

5. Invariant Manifest:
       manifests/od_split_manifest.csv
       source_city, origin, destination, split, split_seed
"""

from typing import Dict, List, Tuple
import math
import numpy as np
import pandas as pd
from pathlib import Path

from src.calibration.support import (
    get_positive_interzonal_support_mask,
    validate_positive_interzonal_support,
)


def create_source_split(
    source_city: str,
    pair_o_idx: np.ndarray,
    pair_d_idx: np.ndarray,
    pair_distance_km: np.ndarray,
    pair_trips: np.ndarray,
    split_seed: int = 42,
    train_ratio: float = 0.30,
) -> pd.DataFrame:
    """
    Creates deterministic 30/70 uniform random split on positive interzonal OD support.
    """
    # 1. Filter to positive interzonal support Omega_s^+
    mask = get_positive_interzonal_support_mask(pair_o_idx, pair_d_idx, pair_distance_km, pair_trips)
    o_supp = pair_o_idx[mask]
    d_supp = pair_d_idx[mask]
    dist_supp = pair_distance_km[mask]
    trips_supp = pair_trips[mask]

    validate_positive_interzonal_support(o_supp, d_supp, dist_supp, trips_supp)

    df_supp = pd.DataFrame({
        "source_city": source_city,
        "origin": o_supp,
        "destination": d_supp,
        "distance_km": dist_supp,
        "flow": trips_supp,
    })

    # 2. Deterministic sort by (origin, destination)
    df_supp = df_supp.sort_values(["origin", "destination"]).reset_index(drop=True)
    n_supp = len(df_supp)

    # 3. Uniform random permutation
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(n_supp)

    n_train = int(math.floor(train_ratio * n_supp))
    train_idx = perm[:n_train]
    heldout_idx = perm[n_train:]

    df_supp["split"] = "heldout"
    df_supp.loc[train_idx, "split"] = "train"
    df_supp["split_seed"] = split_seed

    # 4. Mandatory sanity checks
    train_df = df_supp[df_supp["split"] == "train"]
    heldout_df = df_supp[df_supp["split"] == "heldout"]

    assert len(train_df) + len(heldout_df) == n_supp, "Split sum mismatch!"
    assert len(train_df) == n_train, f"Train count {len(train_df)} != {n_train}"
    assert len(heldout_df) == n_supp - n_train, "Heldout count mismatch!"
    assert np.all(train_df["flow"] >= 1.0) and np.all(heldout_df["flow"] >= 1.0)
    assert np.all(train_df["origin"] != train_df["destination"])
    assert np.all(heldout_df["origin"] != heldout_df["destination"])

    train_keys = set(zip(train_df["origin"], train_df["destination"]))
    heldout_keys = set(zip(heldout_df["origin"], heldout_df["destination"]))
    assert train_keys.isdisjoint(heldout_keys), "Overlap found between train and heldout!"

    return df_supp[["source_city", "origin", "destination", "split", "split_seed"]]
