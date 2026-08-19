import pandas as pd
import geopandas as gpd
from pathlib import Path
import os

_GADM_GDF_CACHE = None

def get_gadm_gid2_mapping(meta_df: pd.DataFrame, repo_root: str) -> dict:
    """
    Returns a dictionary mapping tract `idx` to GADM `GID_2`.
    Uses strict 'within' spatial join first. Falls back to nearest polygon for coastal tracts,
    and prints a warning if fallback is used.
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
    if missing.any():
        num_missing = missing.sum()
        print(f"  [GADM Mapping] WARNING: {num_missing} tracts fell outside exact GADM polygons. Using nearest fallback.")
        
        missing_gdf = tract_gdf[missing]
        # sjoin_nearest handles coastal boundary issues
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nearest = gpd.sjoin_nearest(missing_gdf, gadm, how='left')
        if nearest.index.has_duplicates:
            nearest = nearest[~nearest.index.duplicated(keep='first')]
            
        result.loc[missing, 'GID_2'] = nearest['GID_2']
        
    if result["GID_2"].isna().any():
        raise ValueError("Mapping invariant failed: NaN found in GID_2 even after nearest fallback.")
        
    return dict(zip(result["idx"], result["GID_2"]))
