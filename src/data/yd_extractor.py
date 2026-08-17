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
    Extracted from Meta mobility datasets in meta_prior/ using official City -> County FIPS & GADM mapping.
    Aggregation protocol:
        1. For each temporal snapshot file f:
           compute the 4-bin distribution p^{(f)}_k across constituent counties.
        2. Average across all temporal snapshots: p_k = (1/F) * sum_f p^{(f)}_k.
        3. Normalize: sum_k p_k = 1.0.
"""

import os
import glob
import pandas as pd
import numpy as np
import torch
from pathlib import Path


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

# Cache for temporal snapshot dataframes
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


def extract_yd_oracle(pair_trips: torch.Tensor, bin_labels: torch.Tensor) -> np.ndarray:
    """
    Computes oracle distance-bin distribution from GT flows over Omega_c.
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
    Protocol:
        1. For each snapshot file: compute mean ping fraction across matching counties for each bin.
        2. Average across all temporal snapshots.
        3. Normalize to sum to 1.0.
    """
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

    # Step 2: Average across temporal snapshots
    mean_yd = np.mean(snapshot_distributions, axis=0)

    # Step 3: Normalize
    total = np.sum(mean_yd)
    if total <= 0:
        return None
    return mean_yd / total


if __name__ == "__main__":
    for test_city in ["Raleigh", "Philadelphia", "Denver", "Chicago", "New_York"]:
        yd_r = extract_yd_real(test_city, "meta_prior")
        fips = CITY_FIPS_GADM[test_city]["fips"]
        state = CITY_FIPS_GADM[test_city]["state"]
        print(f"{test_city:<15} ({state}, FIPS {fips}): Y_D^real = {np.round(yd_r, 4).tolist()} | sum={np.sum(yd_r):.4f}")
