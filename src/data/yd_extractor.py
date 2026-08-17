"""
Y_D Extractor for Moving Bins (Primary) and Full 4-Bin (Ablation).

Primary Moving-Bin Formulation:
    Excludes stay-at-home / immobility Bin 0.
    Normalizes across actual movement/displacement categories {1, 2, 3}:
        Bin 1: (0, 10) km
        Bin 2: [10, 100) km
        Bin 3: 100+ km

    Y_{c, k}^{Meta, +}   = Y_{c, k}^{Meta} / sum_{l=1}^3 Y_{c, l}^{Meta}
    Y_{c, k}^{oracle, +} = sum_{(i,j) in Omega_{c,k}^+} T_{ij}^{GT} / sum_{(i,j) in Omega_c^+} T_{ij}^{GT}

Distributional Overlap Metric (CPC_dist / Overlap):
    Overlap(p, q) = sum_k min(p_k, q_k) = 1 - 0.5 * ||p - q||_1
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
import torch
from pathlib import Path

# Ensure root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# Comprehensive official mapping of 50 US cities to County Names, State, and FIPS
CITY_FIPS_GADM = {
    "Albuquerque": {"state": "NM", "fips": "35001", "gadm_names": ["Bernalillo"]},
    "Arlington": {"state": "TX", "fips": "48439", "gadm_names": ["Tarrant"]},
    "Atlanta": {"state": "GA", "fips": ["13121", "13089"], "gadm_names": ["Fulton", "DeKalb"]},
    "Austin": {"state": "TX", "fips": "48453", "gadm_names": ["Travis"]},
    "Baltimore": {"state": "MD", "fips": "24510", "gadm_names": ["Baltimore City", "Baltimore"]},
    "Boston": {"state": "MA", "fips": "25025", "gadm_names": ["Suffolk"]},
    "Charlotte": {"state": "NC", "fips": "37119", "gadm_names": ["Mecklenburg"]},
    "Chicago": {"state": "IL", "fips": "17031", "gadm_names": ["Cook"]},
    "Colorado_Springs": {"state": "CO", "fips": "08041", "gadm_names": ["El Paso"]},
    "Columbus": {"state": "OH", "fips": "39049", "gadm_names": ["Franklin"]},
    "Dallas": {"state": "TX", "fips": "48113", "gadm_names": ["Dallas"]},
    "Denver": {"state": "CO", "fips": "08031", "gadm_names": ["Denver"]},
    "Detroit": {"state": "MI", "fips": "26163", "gadm_names": ["Wayne"]},
    "El_Paso": {"state": "TX", "fips": "48141", "gadm_names": ["El Paso"]},
    "Fort_Worth": {"state": "TX", "fips": "48439", "gadm_names": ["Tarrant"]},
    "Fresno": {"state": "CA", "fips": "06019", "gadm_names": ["Fresno"]},
    "Houston": {"state": "TX", "fips": "48201", "gadm_names": ["Harris"]},
    "Indianapolis": {"state": "IN", "fips": "18097", "gadm_names": ["Marion"]},
    "Jacksonville": {"state": "FL", "fips": "12031", "gadm_names": ["Duval"]},
    "Kansas_City": {"state": "MO", "fips": "29095", "gadm_names": ["Jackson"]},
    "Las_Vegas": {"state": "NV", "fips": "32003", "gadm_names": ["Clark"]},
    "Long_Beach": {"state": "CA", "fips": "06037", "gadm_names": ["Los Angeles"]},
    "Los_Angeles": {"state": "CA", "fips": "06037", "gadm_names": ["Los Angeles"]},
    "Louisville": {"state": "KY", "fips": "21111", "gadm_names": ["Jefferson"]},
    "Memphis": {"state": "TN", "fips": "47157", "gadm_names": ["Shelby"]},
    "Mesa": {"state": "AZ", "fips": "04013", "gadm_names": ["Maricopa"]},
    "Miami": {"state": "FL", "fips": "12086", "gadm_names": ["Miami-Dade", "Dade"]},
    "Milwaukee": {"state": "WI", "fips": "55079", "gadm_names": ["Milwaukee"]},
    "Minneapolis": {"state": "MN", "fips": "27053", "gadm_names": ["Hennepin"]},
    "Nashville": {"state": "TN", "fips": "47037", "gadm_names": ["Davidson"]},
    "New_York": {"state": "NY", "fips": ["36061", "36047", "36081", "36005", "36085"], "gadm_names": ["New York", "Kings", "Queens", "Bronx", "Richmond"]},
    "Oakland": {"state": "CA", "fips": "06001", "gadm_names": ["Alameda"]},
    "Oklahoma_City": {"state": "OK", "fips": "40109", "gadm_names": ["Oklahoma"]},
    "Omaha": {"state": "NE", "fips": "31055", "gadm_names": ["Douglas"]},
    "Philadelphia": {"state": "PA", "fips": "42101", "gadm_names": ["Philadelphia"]},
    "Phoenix": {"state": "AZ", "fips": "04013", "gadm_names": ["Maricopa"]},
    "Portland": {"state": "OR", "fips": "41051", "gadm_names": ["Multnomah"]},
    "Raleigh": {"state": "NC", "fips": "37183", "gadm_names": ["Wake"]},
    "Sacramento": {"state": "CA", "fips": "06067", "gadm_names": ["Sacramento"]},
    "San_Antonio": {"state": "TX", "fips": "48029", "gadm_names": ["Bexar"]},
    "San_Diego": {"state": "CA", "fips": "06073", "gadm_names": ["San Diego"]},
    "San_Francisco": {"state": "CA", "fips": "06075", "gadm_names": ["San Francisco"]},
    "San_Jose": {"state": "CA", "fips": "06085", "gadm_names": ["Santa Clara"]},
    "Seattle": {"state": "WA", "fips": "53033", "gadm_names": ["King"]},
    "Tampa": {"state": "FL", "fips": "12057", "gadm_names": ["Hillsborough"]},
    "Tucson": {"state": "AZ", "fips": "04019", "gadm_names": ["Pima"]},
    "Tulsa": {"state": "OK", "fips": "40143", "gadm_names": ["Tulsa"]},
    "Virginia_Beach": {"state": "VA", "fips": "51810", "gadm_names": ["Virginia Beach"]},
    "Washington_DC": {"state": "DC", "fips": "11001", "gadm_names": ["District of Columbia"]},
    "Wichita": {"state": "KS", "fips": "20173", "gadm_names": ["Sedgwick"]},
}

META_CAT_TO_BIN = {
    "0": 0,
    "(0, 10)": 1,
    "[10, 100)": 2,
    "100+": 3,
}

_SNAPSHOT_CACHE = None


def _load_snapshot_dataframes(meta_prior_dir: str = "meta_prior") -> list[pd.DataFrame]:
    global _SNAPSHOT_CACHE
    if _SNAPSHOT_CACHE is not None:
        return _SNAPSHOT_CACHE

    meta_dir = Path(meta_prior_dir)
    files = sorted(list(meta_dir.glob("*.csv")))
    snapshots = []
    for f in files:
        try:
            df = pd.read_csv(
                f,
                usecols=["country", "gadm_name", "home_to_ping_distance_category", "distance_category_ping_fraction"],
            )
            us_df = df[df["country"] == "USA"].copy()
            snapshots.append(us_df)
        except Exception:
            continue

    _SNAPSHOT_CACHE = snapshots
    return _SNAPSHOT_CACHE


def extract_yd_4bin_real(city_name: str, meta_prior_dir: str = "meta_prior") -> np.ndarray | None:
    """Extracts raw 4-bin Meta distribution (including Bin 0) for ablation."""
    city_info = CITY_FIPS_GADM.get(city_name, None)
    if city_info is None:
        return None

    counties = city_info["gadm_names"]
    snapshots = _load_snapshot_dataframes(meta_prior_dir=meta_prior_dir)
    if not snapshots:
        return None

    snapshot_distributions = []
    for df in snapshots:
        matched = df[df["gadm_name"].isin(counties)]
        if len(matched) == 0:
            continue

        cat_means = matched.groupby("home_to_ping_distance_category")["distance_category_ping_fraction"].mean()
        yd_snap = np.zeros(4, dtype=np.float64)
        for cat_str, bin_idx in META_CAT_TO_BIN.items():
            if cat_str in cat_means:
                yd_snap[bin_idx] = float(cat_means[cat_str])

        snap_sum = np.sum(yd_snap)
        if snap_sum > 0:
            snapshot_distributions.append(yd_snap / snap_sum)

    if not snapshot_distributions:
        return None

    mean_yd = np.mean(snapshot_distributions, axis=0)
    total = np.sum(mean_yd)
    return mean_yd / total if total > 0 else None


def extract_yd_moving_real(city_name: str, meta_prior_dir: str = "meta_prior") -> np.ndarray | None:
    """
    Primary Meta extractor: extracts the 3 moving bins {1, 2, 3} normalized to sum to 1.0.
    Excludes stay-at-home / immobility Bin 0.
    """
    yd_4 = extract_yd_4bin_real(city_name, meta_prior_dir=meta_prior_dir)
    if yd_4 is None:
        return None

    moving_3 = yd_4[1:].copy()  # bins 1, 2, 3
    total_moving = np.sum(moving_3)
    if total_moving <= 0:
        return None
    return moving_3 / total_moving


def extract_yd_4bin_oracle(pair_trips: torch.Tensor, bin_labels: torch.Tensor) -> np.ndarray:
    """Extracts raw 4-bin oracle distribution from GT flows."""
    yd = np.zeros(4, dtype=np.float64)
    trips_np = pair_trips.detach().cpu().numpy()
    bins_np = bin_labels.detach().cpu().numpy()
    total_flow = float(np.sum(trips_np))
    if total_flow <= 0:
        return np.array([0.25, 0.25, 0.25, 0.25])
    for k in range(4):
        yd[k] = np.sum(trips_np[bins_np == k])
    return yd / total_flow


def extract_yd_moving_oracle(
    pair_trips: torch.Tensor,
    bin_labels: torch.Tensor,
    pair_o_idx: torch.Tensor,
    pair_d_idx: torch.Tensor,
) -> np.ndarray:
    """
    Primary Oracle extractor: computes 3-bin distribution on interzonal pairs Omega_c^+ (bins 1, 2, 3).
    """
    trips_np = pair_trips.detach().cpu().numpy()
    bins_np = bin_labels.detach().cpu().numpy()
    o_np = pair_o_idx.detach().cpu().numpy()
    d_np = pair_d_idx.detach().cpu().numpy()

    inter_mask = (o_np != d_np) & (bins_np > 0)
    inter_trips = trips_np[inter_mask]
    inter_bins = bins_np[inter_mask]

    yd_3 = np.zeros(3, dtype=np.float64)
    total_inter = np.sum(inter_trips)
    if total_inter <= 0:
        return np.array([0.5, 0.4, 0.1])

    for idx, bin_k in enumerate([1, 2, 3]):
        yd_3[idx] = np.sum(inter_trips[inter_bins == bin_k])

    return yd_3 / total_inter


def compute_distributional_overlap(p: np.ndarray, q: np.ndarray) -> float:
    """
    Computes Distributional Overlap (CPC_dist) between two probability vectors:
    Overlap(p, q) = sum_k min(p_k, q_k) = 1 - 0.5 * ||p - q||_1
    """
    return float(np.sum(np.minimum(p, q)))


if __name__ == "__main__":
    from src.data.dataset import load_city

    for city in ["Philadelphia", "Austin", "Raleigh", "Denver", "Seattle"]:
        cd = load_city(city, "data")
        o_3 = extract_yd_moving_oracle(cd.pair_trips, cd.bin_labels, cd.pair_o_idx, cd.pair_d_idx)
        r_3 = extract_yd_moving_real(city, "meta_prior")
        overlap = compute_distributional_overlap(o_3, r_3)
        print(f"{city:<15}: Oracle_moving = {np.round(o_3, 4).tolist()} | Meta_moving = {np.round(r_3, 4).tolist()} | Overlap = {overlap*100:.2f}%")
