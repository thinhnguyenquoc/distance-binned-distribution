"""
OD Split Generator Module: Strict Nested Split Contract for f in {0.10, 0.20, 0.30, 0.50, 1.00}.

Protocol Specifications (§2.2):
1. Candidate Pool:
       Only OD pairs belonging to positive interzonal OD support Omega_s^+:
       (origin != destination) & (distance_km > 0) & (flow >= 1)
   Never split on full matrix, zero-flow pairs, or all possible tract pairs.

2. Deterministic Sort:
       Sort candidate pairs by (origin, destination) before permutation
       to ensure identical splits regardless of disk / CSV row order.

3. Strict Nested Split Contract with split_seed = 42:
       Single permutation: pi_s = Permutation(Omega_s^+)
       Train_10(s)  = pi_s[:floor(0.10 * N_s)]
       Train_20(s)  = pi_s[:floor(0.20 * N_s)]
       Train_30(s)  = pi_s[:floor(0.30 * N_s)]
       Train_50(s)  = pi_s[:floor(0.50 * N_s)]
       Train_100(s) = pi_s[:N_s]

   Invariant:
       Train_10 subset Train_20 subset Train_30 subset Train_50 subset Train_100.

4. Backward Compatibility:
       split == "train" <==> in_train_30 == True
       split == "heldout" for remaining 70% in main setting.

5. Manifest:
       manifests/od_split_manifest.csv
       source_city,origin,destination,split,in_train_10,in_train_20,in_train_30,in_train_50,in_train_100,split_seed
"""

from typing import Dict, List, Tuple
import math
import numpy as np
import pandas as pd
from pathlib import Path

from implement_new_plan.calibration.support import (
    get_positive_interzonal_support_mask,
    validate_positive_interzonal_support,
)


def create_nested_source_splits(
    source_city: str,
    pair_o_idx: np.ndarray,
    pair_d_idx: np.ndarray,
    pair_distance_km: np.ndarray,
    pair_trips: np.ndarray,
    split_seed: int = 42,
) -> pd.DataFrame:
    """
    Creates deterministic nested split on positive interzonal OD support across 5 fractions:
    f in {0.10, 0.20, 0.30, 0.50, 1.00}.
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
    n_total = len(df_supp)

    # 3. Single uniform random permutation with seed 42
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(n_total)

    n_train_10 = int(math.floor(0.10 * n_total))
    n_train_20 = int(math.floor(0.20 * n_total))
    n_train_30 = int(math.floor(0.30 * n_total))
    n_train_50 = int(math.floor(0.50 * n_total))

    idx_10 = perm[:n_train_10]
    idx_20 = perm[:n_train_20]
    idx_30 = perm[:n_train_30]
    idx_50 = perm[:n_train_50]
    idx_100 = perm[:n_total]

    # 4. Mandatory Set Inclusion Sanity Checks
    s10 = set(idx_10)
    s20 = set(idx_20)
    s30 = set(idx_30)
    s50 = set(idx_50)
    s100 = set(idx_100)
    assert s10 <= s20 <= s30 <= s50 <= s100, "Nested split set inclusion invariant violated!"
    assert len(s100) == n_total, "Train_100 must contain all positive support pairs!"

    df_supp["in_train_10"] = False
    df_supp["in_train_20"] = False
    df_supp["in_train_30"] = False
    df_supp["in_train_50"] = False
    df_supp["in_train_100"] = True

    df_supp.loc[idx_10, "in_train_10"] = True
    df_supp.loc[idx_20, "in_train_20"] = True
    df_supp.loc[idx_30, "in_train_30"] = True
    df_supp.loc[idx_50, "in_train_50"] = True

    # 5. Backward compatibility for main 30/70 split: split column
    df_supp["split"] = "heldout"
    df_supp.loc[idx_30, "split"] = "train"
    df_supp["split_seed"] = split_seed

    # 6. Verify assertions
    assert (df_supp["split"] == "train").sum() == n_train_30
    assert (df_supp["in_train_30"] == True).sum() == n_train_30

    output_cols = [
        "source_city",
        "origin",
        "destination",
        "split",
        "in_train_10",
        "in_train_20",
        "in_train_30",
        "in_train_50",
        "in_train_100",
        "split_seed"
    ]
    return df_supp[output_cols]
