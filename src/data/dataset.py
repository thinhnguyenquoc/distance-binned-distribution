"""
City dataset loader for the distance-binned OD reconstruction study.

Loads a single city's data from the standard directory layout:
    data/{city}/
        meta.csv               — tract metadata (idx, lon, lat, area_km2, city)
        nodes/
            census.csv         — population, income, employment features
            poi.csv            — POI density features
            road.csv           — road network features
        pairs/
            od.csv             — candidate OD pairs with trip_count
            distance.csv       — pairwise distances (km) for same candidate set

Returns a CityData dataclass with:
    node_features  : FloatTensor (N, F)
    pair_o_idx     : LongTensor  (E,)     — origin tract index
    pair_d_idx     : LongTensor  (E,)     — destination tract index
    pair_distance  : FloatTensor (E,)     — distance in km
    pair_trips     : FloatTensor (E,)     — trip count (all >= 1)
    population     : FloatTensor (N,)     — total_population per tract
    lon_lat        : FloatTensor (N, 2)   — centroid coordinates
    city_name      : str
    n_tracts       : int
    n_pairs        : int

Normalization:
    - Node features: StandardScaler fitted on training cities, applied to all.
    - Distances: log(1 + d_km).
    - Trip counts: kept as raw integers (ZTNB operates on counts directly).
"""

from __future__ import annotations

import os
import csv
import dataclasses
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Node feature columns (order must match across all cities)
# ---------------------------------------------------------------------------

# Census features used (subset — well-defined across all 50 cities)
CENSUS_COLS = [
    "total_population", "median_age", "median_income", "per_capita_income",
    "employment_rate", "unemployment_rate", "commute_transit_pct",
    "commute_active_pct", "commute_wfh_pct", "zero_vehicle_pct",
    "avg_vehicles_per_household", "higher_education_pct", "homeownership_rate",
]

# POI features
POI_COLS = [
    "office", "office_density", "industrial", "industrial_density",
    "commercial", "commercial_density", "education_primary",
    "education_primary_density",
]

# Road features
ROAD_COLS = [
    "road_length_total", "road_density", "road_count",
    "motorway_length", "primary_length",
]


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class CityData:
    city_name:      str
    n_tracts:       int
    n_pairs:        int

    # Node-level (N, *)
    node_features:  torch.Tensor   # (N, F) normalized
    population:     torch.Tensor   # (N,)   raw population
    lon_lat:        torch.Tensor   # (N, 2) [lon, lat]

    # Pair-level (E, *)
    pair_o_idx:     torch.LongTensor   # (E,)
    pair_d_idx:     torch.LongTensor   # (E,)
    pair_distance:  torch.Tensor       # (E,) log1p(km)
    pair_trips:     torch.Tensor       # (E,) raw counts, all >= 1
    bin_labels:     torch.LongTensor   # (E,) distance bin index (0-3)


# ---------------------------------------------------------------------------
# Distance bin assignment
# ---------------------------------------------------------------------------
# Bins match Meta mobility categories: 0 km | (0,10) | [10,100) | 100+
BIN_EDGES = [0.0, 1e-9, 10.0, 100.0, float("inf")]
BIN_LABELS = ["zero", "short", "medium", "long"]   # 0, 1, 2, 3

def assign_bins(distance_km: np.ndarray) -> np.ndarray:
    """Assign each pair to a distance bin (0=zero, 1=short, 2=medium, 3=long)."""
    bins = np.zeros(len(distance_km), dtype=np.int64)
    bins[(distance_km > 0)   & (distance_km < 10)]  = 1
    bins[(distance_km >= 10) & (distance_km < 100)] = 2
    bins[distance_km >= 100]                         = 3
    return bins


# ---------------------------------------------------------------------------
# CSV loading helpers
# ---------------------------------------------------------------------------

def _load_csv_columns(path: Path, cols: List[str], key_col: str = "idx") -> np.ndarray:
    """Load specific columns from a CSV, ordered by key_col. Returns float array."""
    data: Dict[int, List[float]] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = int(row[key_col])
            vals = []
            for c in cols:
                v = row.get(c, "0") or "0"
                try:
                    vals.append(float(v))
                except ValueError:
                    vals.append(0.0)
            data[key] = vals
    n = max(data.keys()) + 1
    arr = np.zeros((n, len(cols)), dtype=np.float32)
    for k, v in data.items():
        arr[k] = v
    return arr


def _load_meta(path: Path):
    """Load meta.csv -> idx, lon, lat, population placeholder."""
    idx_list, lons, lats = [], [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx_list.append(int(row["idx"]))
            lons.append(float(row["lon"]))
            lats.append(float(row["lat"]))
    n = max(idx_list) + 1
    lon_arr = np.zeros(n, dtype=np.float32)
    lat_arr = np.zeros(n, dtype=np.float32)
    for i, lon, lat in zip(idx_list, lons, lats):
        lon_arr[i] = lon
        lat_arr[i] = lat
    return np.stack([lon_arr, lat_arr], axis=1)   # (N, 2)


def _load_pairs(od_path: Path, dist_path: Path):
    """Load od.csv and distance.csv into aligned arrays."""
    od: Dict[tuple, int] = {}
    with open(od_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            od[(int(row["o_idx"]), int(row["d_idx"]))] = int(row["trip_count"])

    origins, dests, trips, dists = [], [], [], []
    with open(dist_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            o, d = int(row["o_idx"]), int(row["d_idx"])
            dist_km = float(row["distance_km"])
            trip_count = od.get((o, d), None)
            if trip_count is None:
                # distance file has pair not in OD — skip (Portland has 6 such pairs)
                continue
            origins.append(o)
            dests.append(d)
            trips.append(trip_count)
            dists.append(dist_km)

    return (
        np.array(origins, dtype=np.int64),
        np.array(dests,   dtype=np.int64),
        np.array(trips,   dtype=np.float32),
        np.array(dists,   dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_city(
    city_name: str,
    data_root: str = "data",
    feature_scaler: Optional["StandardScaler"] = None,
    fit_scaler: bool = False,
) -> CityData:
    """
    Load one city's data.

    Args:
        city_name:      Directory name under data_root.
        data_root:      Root of the data/ directory.
        feature_scaler: Optional fitted sklearn StandardScaler.
                        If None and fit_scaler=True, fits a new one.
        fit_scaler:     If True, fits scaler on this city's data (use only for
                        training set aggregate; see load_cities()).

    Returns:
        CityData instance.
    """
    base = Path(data_root) / city_name

    # --- Node features ---
    census = _load_csv_columns(base / "nodes" / "census.csv", CENSUS_COLS)
    poi    = _load_csv_columns(base / "nodes" / "poi.csv",    POI_COLS)
    road   = _load_csv_columns(base / "nodes" / "road.csv",   ROAD_COLS)
    X_raw  = np.concatenate([census, poi, road], axis=1)   # (N, F)

    # Population for gravity prior (first census column)
    population = census[:, 0].copy()   # total_population

    # Coordinates
    lon_lat = _load_meta(base / "meta.csv")   # (N, 2)

    n_tracts = X_raw.shape[0]

    # --- Pair data ---
    o_idx, d_idx, trips, dist_km = _load_pairs(
        base / "pairs" / "od.csv",
        base / "pairs" / "distance.csv",
    )
    assert (trips >= 1).all(), f"{city_name}: found zero trip counts in candidate set"

    # --- Normalize node features ---
    if feature_scaler is not None:
        X_norm = feature_scaler.transform(X_raw)
    elif fit_scaler:
        from sklearn.preprocessing import StandardScaler
        feature_scaler = StandardScaler()
        X_norm = feature_scaler.fit_transform(X_raw)
    else:
        X_norm = X_raw   # raw, use with caution

    # Replace NaN/Inf that may arise from missing features
    X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=0.0, neginf=0.0)

    # --- Distance encoding: log1p ---
    log_dist = np.log1p(dist_km)

    # --- Bin labels ---
    bin_labels = assign_bins(dist_km)

    return CityData(
        city_name     = city_name,
        n_tracts      = n_tracts,
        n_pairs       = len(o_idx),
        node_features = torch.tensor(X_norm,     dtype=torch.float32),
        population    = torch.tensor(population, dtype=torch.float32),
        lon_lat       = torch.tensor(lon_lat,    dtype=torch.float32),
        pair_o_idx    = torch.tensor(o_idx,      dtype=torch.long),
        pair_d_idx    = torch.tensor(d_idx,      dtype=torch.long),
        pair_distance = torch.tensor(log_dist,   dtype=torch.float32),
        pair_trips    = torch.tensor(trips,       dtype=torch.float32),
        bin_labels    = torch.tensor(bin_labels,  dtype=torch.long),
    )


def load_cities(
    city_names: List[str],
    data_root: str = "data",
) -> tuple[List[CityData], object]:
    """
    Load multiple cities, fitting a single StandardScaler on all training
    node features jointly (to ensure consistent normalization).

    Returns:
        (list of CityData, fitted scaler)
    """
    from sklearn.preprocessing import StandardScaler

    # First pass: collect all raw features to fit scaler
    all_X = []
    raw_cities = []
    for name in city_names:
        base = Path(data_root) / name
        census = _load_csv_columns(base / "nodes" / "census.csv", CENSUS_COLS)
        poi    = _load_csv_columns(base / "nodes" / "poi.csv",    POI_COLS)
        road   = _load_csv_columns(base / "nodes" / "road.csv",   ROAD_COLS)
        X_raw  = np.concatenate([census, poi, road], axis=1)
        X_raw  = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
        all_X.append(X_raw)
        raw_cities.append(name)

    scaler = StandardScaler()
    scaler.fit(np.concatenate(all_X, axis=0))

    # Second pass: load with fitted scaler
    cities = [load_city(name, data_root, feature_scaler=scaler) for name in city_names]
    return cities, scaler


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "data"

    print("Loading Raleigh (small)...")
    cd = load_city("Raleigh", data_root=root)
    print(f"  Tracts: {cd.n_tracts}, Pairs: {cd.n_pairs}")
    print(f"  node_features: {cd.node_features.shape} | dtype: {cd.node_features.dtype}")
    print(f"  pair_trips: min={cd.pair_trips.min():.0f}, max={cd.pair_trips.max():.0f}")
    print(f"  pair_distance: min={cd.pair_distance.min():.3f}, max={cd.pair_distance.max():.3f}")
    print(f"  bin_labels: unique={cd.bin_labels.unique().tolist()}")
    print(f"  bin distribution: { {i: (cd.bin_labels==i).sum().item() for i in range(4)} }")
    print()

    print("Loading Raleigh + Denver jointly (scaler fit)...")
    cities, scaler = load_cities(["Raleigh", "Denver"], data_root=root)
    for c in cities:
        print(f"  {c.city_name}: node_features mean~{c.node_features.mean():.3f} std~{c.node_features.std():.3f}")
    print()

    print("Smoke test passed.")
