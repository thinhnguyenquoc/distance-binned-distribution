import pandas as pd
import geopandas as gpd
from pathlib import Path
import os

_GADM_GDF_CACHE = None

def get_gadm_gid2_mapping(meta_df: pd.DataFrame, repo_root: str) -> tuple[dict, dict]:
    """
    Returns a tuple: (mapping_dict, stats_dict).
    mapping_dict maps tract `idx` to GADM `GID_2`.
    stats_dict contains `n_strict_within` and `n_nearest_fallback` for provenance auditing.
    """
    global _GADM_GDF_CACHE
    
    gadm_shp_path = Path(repo_root) / "gadm41_USA_shp" / "gadm41_USA_2.shp"
    if not gadm_shp_path.exists():
        raise FileNotFoundError(f"GADM shapefile not found at {gadm_shp_path}")
        
    if _GADM_GDF_CACHE is None:
        _GADM_GDF_CACHE = gpd.read_file(gadm_shp_path)[['GID_2', 'geometry']].to_crs("EPSG:4326")
        
    gadm = _GADM_GDF_CACHE
    tract_gdf = gpd.GeoDataFrame(
        meta_df, 
        geometry=gpd.points_from_xy(meta_df['lon'], meta_df['lat']), 
        crs="EPSG:4326"
    )
    
    # 1. Strict within
    result = gpd.sjoin(tract_gdf, gadm, how='left', predicate='within')
    if result.index.has_duplicates:
        result = result[~result.index.duplicated(keep='first')]
        
    missing = result['GID_2'].isna()
    n_fallback = 0
    fallback_details = []
    
    if missing.any():
        n_fallback = int(missing.sum())
        
        missing_gdf = tract_gdf[missing].copy()
        
        # Project to EPSG:5070 (NAD83 / Conus Albers) for accurate distance in meters
        gadm_proj = gadm.to_crs("EPSG:5070")
        missing_proj = missing_gdf.to_crs("EPSG:5070")
        
        # sjoin_nearest handles coastal boundary issues
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nearest = gpd.sjoin_nearest(missing_proj, gadm_proj, how='left', distance_col='nearest_distance_m')
            
        if nearest.index.has_duplicates:
            nearest = nearest[~nearest.index.duplicated(keep='first')]
            
        result.loc[missing, 'GID_2'] = nearest['GID_2']
        
        # Enforce 5km threshold and record details
        for row_idx, row in nearest.iterrows():
            idx_val = int(row['idx'])
            dist_m = float(row['nearest_distance_m'])
            gid2 = str(row['GID_2'])
            
            fallback_details.append({
                "tract_idx": idx_val,
                "GID_2": gid2,
                "nearest_distance_m": dist_m
            })
            
            if dist_m > 5000.0:
                raise ValueError(f"Mapping invariant failed: Tract {idx_val} is {dist_m:.2f}m away from nearest GADM polygon, exceeding 5km threshold.")
                
        print(f"  [GADM Mapping] WARNING: {n_fallback} tracts fell outside exact GADM polygons. Using nearest fallback (max dist: {max(d['nearest_distance_m'] for d in fallback_details):.2f}m).")
        
    if result["GID_2"].isna().any():
        raise ValueError("Mapping invariant failed: NaN found in GID_2 even after nearest fallback.")
        
    stats = {
        "n_strict_within": int(len(meta_df) - n_fallback),
        "n_nearest_fallback": n_fallback,
        "fallback_details": fallback_details
    }
        
    return dict(zip(result["idx"], result["GID_2"])), stats
