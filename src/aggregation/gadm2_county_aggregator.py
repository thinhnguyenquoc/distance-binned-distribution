# File: src/aggregation/gadm2_county_aggregator.py

import os
from pathlib import Path
from typing import Optional, List, Dict, Any
import geopandas as gpd
import pandas as pd
import numpy as np


def find_default_gadm_path() -> str:
    """Find default GADM shapefile path."""
    candidates = [
        "gadm41_USA_shp/gadm41_USA_2.shp",
        "gadm36_USA_2.shp",
        "gadm41_USA_2.shp",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return candidates[0]


def load_gadm(gadm_shp: Optional[str] = None) -> gpd.GeoDataFrame:
    """Load and prepare GADM level-2 GeoDataFrame."""
    if gadm_shp is None:
        gadm_shp = find_default_gadm_path()
    
    print(f"Loading GADM from {gadm_shp}...")
    gadm = gpd.read_file(gadm_shp)
    gadm = gadm[['GID_2', 'NAME_2', 'geometry']].to_crs("EPSG:4326")
    return gadm


def aggregate_city_to_gadm2(
    city_name: str,
    data_root: str = "data",
    gadm_shp: Optional[str] = None,
    gadm_gdf: Optional[gpd.GeoDataFrame] = None,
    output_root: str = "data_gadm2",
) -> Dict[str, Any]:
    """
    Aggregate tract-level OD and distances to GADM level-2 counties.
    
    Returns preflight dict with diagnostics.
    """
    raise RuntimeError(
        "gadm2_county_aggregator.py is officially deprecated and retired from the pipeline.\n"
        "1. It silently drops tracts where centroids don't match exactly without a robust nearest-fallback.\n"
        "2. It averages distances across tracts (mean tract-pair distance), which heavily distorts distance distributions "
        "and is totally invalid for distance-binned Y_D reconstruction.\n"
        "Use src.data.gadm_mapper.get_gadm_gid2_mapping on tract pairs directly instead."
    )
    if gadm_gdf is None:
        gadm_gdf = load_gadm(gadm_shp)
    
    # Load tract metadata
    meta_file = Path(data_root) / city_name / "meta.csv"
    if not meta_file.exists():
        raise FileNotFoundError(f"Meta file not found: {meta_file}")
    
    meta = pd.read_csv(meta_file)
    print(f"  {city_name}: {len(meta)} tracts")
    
    # Create GeoDataFrame with tract centroids
    tract_gdf = gpd.GeoDataFrame(
        meta,
        geometry=gpd.points_from_xy(meta['lon'], meta['lat']),
        crs="EPSG:4326"
    )
    
    # Point-in-polygon: tract -> GID_2
    print(f"  Point-in-polygon spatial join...")
    result = gpd.sjoin(tract_gdf, gadm_gdf[['GID_2', 'geometry']], how='left', predicate='within')
    
    # Handle duplicates if any centroid lies on a boundary
    if result.index.has_duplicates:
        result = result[~result.index.duplicated(keep='first')]
    
    # Map: tract_idx -> GID_2
    if 'idx' in result.columns:
        idx_to_gid2 = dict(zip(result['idx'], result['GID_2']))
    else:
        idx_to_gid2 = dict(enumerate(result['GID_2']))
    
    # Load OD matrix
    od_file = Path(data_root) / city_name / "pairs" / "od.csv"
    if not od_file.exists():
        raise FileNotFoundError(f"OD file not found: {od_file}")
    
    od = pd.read_csv(od_file)
    od['county_o'] = od['o_idx'].map(idx_to_gid2)
    od['county_d'] = od['d_idx'].map(idx_to_gid2)
    
    # Aggregate OD by county pairs
    od_valid = od.dropna(subset=['county_o', 'county_d'])
    od_county = od_valid.groupby(['county_o', 'county_d'])['trip_count'].sum().reset_index()
    od_county.columns = ['county_o', 'county_d', 'trips']
    
    # Load distances
    dist_file = Path(data_root) / city_name / "pairs" / "distance.csv"
    if not dist_file.exists():
        raise FileNotFoundError(f"Distance file not found: {dist_file}")
    
    dist = pd.read_csv(dist_file)
    dist['county_o'] = dist['o_idx'].map(idx_to_gid2)
    dist['county_d'] = dist['d_idx'].map(idx_to_gid2)
    
    # Aggregate distances: mean per county pair
    dist_valid = dist.dropna(subset=['county_o', 'county_d'])
    dist_county = dist_valid.groupby(['county_o', 'county_d'])['distance_km'].agg(['mean', 'count']).reset_index()
    dist_county.columns = ['county_o', 'county_d', 'distance_km', 'pair_count']
    
    # Merge
    county_pairs = od_county.merge(dist_county, on=['county_o', 'county_d'], how='inner')
    
    # Create output directory
    output_dir = Path(output_root) / city_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save
    county_pairs.to_csv(output_dir / "county_pairs.csv", index=False)
    
    # Preflight diagnostics
    valid_gid2 = {v for v in idx_to_gid2.values() if pd.notna(v)}
    n_counties = len(valid_gid2)
    n_tract_unmapped = sum(1 for v in idx_to_gid2.values() if pd.isna(v))
    
    total_trips_tract = int(od['trip_count'].sum())
    total_trips_county = int(county_pairs['trips'].sum())
    conservation_error = (
        abs(total_trips_tract - total_trips_county) / total_trips_tract
        if total_trips_tract > 0 else 0.0
    )
    
    preflight = {
        "city": city_name,
        "n_tracts": len(meta),
        "n_counties": n_counties,
        "n_tract_unmapped": n_tract_unmapped,
        "n_county_pairs": len(county_pairs),
        "total_trips_tract": total_trips_tract,
        "total_trips_county": total_trips_county,
        "trip_conservation_error": conservation_error,
    }
    
    print(f"  -> {n_counties} counties, {len(county_pairs)} county pairs")
    print(f"  -> Trip conservation error: {preflight['trip_conservation_error']:.2e}")
    
    return preflight


def run_all(
    cities: Optional[List[str]] = None,
    data_root: str = "data",
    gadm_shp: Optional[str] = None,
    output_root: str = "data_gadm2",
    results_dir: str = "results",
) -> pd.DataFrame:
    """Run aggregation for all specified or discovered cities."""
    if cities is None:
        data_p = Path(data_root)
        cities = sorted([
            d.name for d in data_p.iterdir()
            if d.is_dir() and (d / "meta.csv").exists()
        ])
    
    print(f"Found {len(cities)} cities to process.")
    
    # Load GADM once for efficiency
    gadm_gdf = load_gadm(gadm_shp)
    
    results = []
    for city in cities:
        print(f"\nProcessing {city}...")
        try:
            r = aggregate_city_to_gadm2(
                city_name=city,
                data_root=data_root,
                gadm_gdf=gadm_gdf,
                output_root=output_root,
            )
            results.append(r)
        except Exception as e:
            print(f"  ERROR {city}: {e}")
            results.append({"city": city, "error": str(e)})
    
    # Preflight report
    report_df = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print(report_df.to_string())
    print("=" * 80)
    
    # Stop gate check
    if 'n_counties' in report_df.columns:
        failed = report_df[report_df['n_counties'].fillna(0) < 2]
        if len(failed) > 0:
            print(f"\nWARNING: {len(failed)} cities with <2 counties (no intercounty OD)")
            print(failed[['city', 'n_counties']])
    
    # Save report
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    report_path = Path(results_dir) / "gadm2_preflight.csv"
    report_df.to_csv(report_path, index=False)
    print(f"\nPreflight report saved to {report_path}")
    
    return report_df


if __name__ == "__main__":
    run_all()
