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
import hashlib
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
    if not data:
        return np.zeros((0, len(cols)), dtype=np.float32)
    keys = sorted(data.keys())
    n = max(keys) + 1
    if keys != list(range(n)):
        raise ValueError(f"Feature CSV {path} has missing indices. Expected 0 to {n-1}.")
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
    if not idx_list:
        return np.zeros((0, 2), dtype=np.float32)
    n = max(idx_list) + 1
    if sorted(idx_list) != list(range(n)):
        raise ValueError(f"Meta CSV {path} has missing indices. Expected 0 to {n-1}.")
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
            trip = int(row["trip_count"])
            if trip > 0:
                od[(int(row["o_idx"]), int(row["d_idx"]))] = trip

    dist_map: Dict[tuple, float] = {}
    with open(dist_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dist_map[(int(row["o_idx"]), int(row["d_idx"]))] = float(row["distance_km"])
            
    od_keys = set(od.keys())
    dist_keys = set(dist_map.keys())
    
    missing_dist = od_keys - dist_keys
    if len(missing_dist) > 0:
        raise ValueError(f"Found {len(missing_dist)} positive OD pairs missing from distance.csv (e.g. {list(missing_dist)[:3]}). Support integrity compromised.")

    # Iterate over distance pairs that have positive OD trips
    origins, dests, trips, dists = [], [], [], []
    for pair in dist_keys:
        trip_count = od.get(pair)
        if trip_count is None:
            # Pair has distance but trip=0 or missing OD, which is fine (zero-trip pairs are ignored in GNN but safe to skip for support)
            continue
        origins.append(pair[0])
        dests.append(pair[1])
        trips.append(trip_count)
        dists.append(dist_map[pair])

    return (
        np.array(origins, dtype=np.int64),
        np.array(dests,   dtype=np.int64),
        np.array(trips,   dtype=np.float32),
        np.array(dists,   dtype=np.float32),
    )


@dataclasses.dataclass
class RawCityData:
    city_name:      str
    n_tracts:       int
    n_pairs:        int
    X_raw:          np.ndarray         # (N, F) unscaled float32
    population:     torch.Tensor       # (N,)   raw population float32
    lon_lat:        torch.Tensor       # (N, 2) [lon, lat] float32
    pair_o_idx:     torch.LongTensor   # (E,)
    pair_d_idx:     torch.LongTensor   # (E,)
    pair_distance:  torch.Tensor       # (E,) log1p(km) float32
    pair_trips:     torch.Tensor       # (E,) raw counts, all >= 1 float32
    bin_labels:     torch.LongTensor   # (E,) distance bin index (0-3)
    dist_km:        np.ndarray         # (E,) raw pairwise distance in km


# Global In-Memory Caches for parsed raw CSV city datasets & normalized CityData instances
_RAW_CITY_CACHE: Dict[tuple[str, str], RawCityData] = {}
_CITY_DATA_CACHE: Dict[tuple[str, str, Optional[str]], CityData] = {}


def get_scaler_fingerprint(scaler: Optional[object]) -> Optional[str]:
    """
    Computes a deterministic content-based fingerprint (SHA-256) of a fitted StandardScaler.
    Prevents cross-fold leakage / normalization contamination caused by Python memory address (id(scaler)) reuse.
    Returns None if scaler is None.
    """
    if scaler is None:
        return None
    if hasattr(scaler, "mean_") and scaler.mean_ is not None:
        m_bytes = np.ascontiguousarray(scaler.mean_, dtype=np.float64).tobytes()
        v_bytes = np.ascontiguousarray(getattr(scaler, "var_", np.zeros_like(scaler.mean_)), dtype=np.float64).tobytes()
        s_bytes = np.ascontiguousarray(getattr(scaler, "scale_", np.ones_like(scaler.mean_)), dtype=np.float64).tobytes()
        return hashlib.sha256(m_bytes + v_bytes + s_bytes).hexdigest()
    return f"unfitted_{id(scaler)}"


def clear_city_cache() -> None:
    """Flushes both raw and normalized in-memory city dataset caches."""
    global _RAW_CITY_CACHE, _CITY_DATA_CACHE
    _RAW_CITY_CACHE.clear()
    _CITY_DATA_CACHE.clear()


def load_raw_city(
    city_name: str,
    data_root: str = "data",
    use_cache: bool = True,
) -> RawCityData:
    """
    Load or retrieve unscaled raw city data from disk / in-memory cache.
    """
    cache_key = (city_name, str(Path(data_root).resolve()))
    if use_cache and cache_key in _RAW_CITY_CACHE:
        return _RAW_CITY_CACHE[cache_key]

    base = Path(data_root) / city_name

    # --- Node features ---
    census = _load_csv_columns(base / "nodes" / "census.csv", CENSUS_COLS)
    poi    = _load_csv_columns(base / "nodes" / "poi.csv",    POI_COLS)
    road   = _load_csv_columns(base / "nodes" / "road.csv",   ROAD_COLS)
    X_raw  = np.concatenate([census, poi, road], axis=1)   # (N, F)
    X_raw  = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

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

    log_dist = np.log1p(dist_km)
    bin_labels = assign_bins(dist_km)

    raw_data = RawCityData(
        city_name     = city_name,
        n_tracts      = n_tracts,
        n_pairs       = len(o_idx),
        X_raw         = X_raw,
        population    = torch.tensor(population, dtype=torch.float32),
        lon_lat       = torch.tensor(lon_lat,    dtype=torch.float32),
        pair_o_idx    = torch.tensor(o_idx,      dtype=torch.long),
        pair_d_idx    = torch.tensor(d_idx,      dtype=torch.long),
        pair_distance = torch.tensor(log_dist,   dtype=torch.float32),
        pair_trips    = torch.tensor(trips,      dtype=torch.float32),
        bin_labels    = torch.tensor(bin_labels, dtype=torch.long),
        dist_km       = dist_km,
    )

    if use_cache:
        _RAW_CITY_CACHE[cache_key] = raw_data

    return raw_data


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_city(
    city_name: str,
    data_root: str = "data",
    feature_scaler: Optional["StandardScaler"] = None,
    fit_scaler: bool = False,
    use_cache: bool = True,
) -> CityData:
    """
    Load one city's data, optionally applying or fitting a feature scaler.

    Args:
        city_name:      Directory name under data_root.
        data_root:      Root of the data/ directory.
        feature_scaler: Optional fitted sklearn StandardScaler.
                        If None and fit_scaler=True, fits a new one.
        fit_scaler:     If True, fits scaler on this city's data.
        use_cache:      If True, retrieves raw parsed data from in-memory cache.

    Returns:
        CityData instance.
    """
    scaler_key = get_scaler_fingerprint(feature_scaler)
    resolved_root = str(Path(data_root).resolve())
    cache_key = (city_name, resolved_root, scaler_key)

    if use_cache and not fit_scaler and cache_key in _CITY_DATA_CACHE:
        return _CITY_DATA_CACHE[cache_key]

    raw = load_raw_city(city_name, data_root=data_root, use_cache=use_cache)

    # --- Normalize node features ---
    if feature_scaler is not None:
        X_norm = feature_scaler.transform(raw.X_raw)
    elif fit_scaler:
        from sklearn.preprocessing import StandardScaler
        feature_scaler = StandardScaler()
        X_norm = feature_scaler.fit_transform(raw.X_raw)
        scaler_key = get_scaler_fingerprint(feature_scaler)
        cache_key = (city_name, resolved_root, scaler_key)
    else:
        X_norm = raw.X_raw

    # Replace NaN/Inf that may arise from missing features
    X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=0.0, neginf=0.0)

    cd = CityData(
        city_name     = raw.city_name,
        n_tracts      = raw.n_tracts,
        n_pairs       = raw.n_pairs,
        node_features = torch.tensor(X_norm, dtype=torch.float32),
        population    = raw.population,
        lon_lat       = raw.lon_lat,
        pair_o_idx    = raw.pair_o_idx,
        pair_d_idx    = raw.pair_d_idx,
        pair_distance = raw.pair_distance,
        pair_trips    = raw.pair_trips,
        bin_labels    = raw.bin_labels,
    )

    if use_cache:
        _CITY_DATA_CACHE[cache_key] = cd

    return cd


def load_cities(
    city_names: List[str],
    data_root: str = "data",
    use_cache: bool = True,
) -> tuple[List[CityData], object]:
    """
    Load multiple cities, fitting a single StandardScaler on all training
    node features jointly (to ensure consistent normalization).

    Returns:
        (list of CityData, fitted scaler)
    """
    from sklearn.preprocessing import StandardScaler

    # First pass: collect raw features from memory cache
    raw_list = [load_raw_city(name, data_root=data_root, use_cache=use_cache) for name in city_names]
    all_X = [r.X_raw for r in raw_list]

    scaler = StandardScaler()
    scaler.fit(np.concatenate(all_X, axis=0))
    scaler_key = get_scaler_fingerprint(scaler)
    resolved_root = str(Path(data_root).resolve())

    # Second pass: construct CityData with fitted scaler and cache into _CITY_DATA_CACHE
    cities = []
    for raw in raw_list:
        cache_key = (raw.city_name, resolved_root, scaler_key)
        if use_cache and cache_key in _CITY_DATA_CACHE:
            cities.append(_CITY_DATA_CACHE[cache_key])
        else:
            cd = CityData(
                city_name     = raw.city_name,
                n_tracts      = raw.n_tracts,
                n_pairs       = raw.n_pairs,
                node_features = torch.tensor(np.nan_to_num(scaler.transform(raw.X_raw), nan=0.0, posinf=0.0, neginf=0.0), dtype=torch.float32),
                population    = raw.population,
                lon_lat       = raw.lon_lat,
                pair_o_idx    = raw.pair_o_idx,
                pair_d_idx    = raw.pair_d_idx,
                pair_distance = raw.pair_distance,
                pair_trips    = raw.pair_trips,
                bin_labels    = raw.bin_labels,
            )
            if use_cache:
                _CITY_DATA_CACHE[cache_key] = cd
            cities.append(cd)

    return cities, scaler


def preload_all_cities(
    data_root: str = "data",
    city_names: Optional[List[str]] = None,
    build_graphs: bool = True,
    radius_km: float = 5.0,
) -> None:
    """
    Preloads all cities into in-memory cache upfront.
    Optionally computes spatial radius graphs and distance matrices.
    Completely eliminates disk I/O during multi-fold cross-validation.
    """
    from src.data.urban_graph import build_radius_graph
    if city_names is None:
        p = Path(data_root)
        if p.exists():
            city_names = sorted([d.name for d in p.iterdir() if d.is_dir() and (d / "meta.csv").exists()])
        else:
            city_names = []

    for name in city_names:
        raw = load_raw_city(name, data_root=data_root, use_cache=True)
        if build_graphs:
            build_radius_graph(raw.lon_lat, radius_km=radius_km, use_cache=True)


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
