"""
Source-Fitted Feature Scaler and Preprocessing Module.

Protocol Requirements:
1. Core Invariant:
       [Fit preprocessing trên source -> freeze -> apply y nguyên sang target]
   Zero-shot transfer NEVER computes or uses any target city distribution statistics.

2. Feature Categorization & Transformation:
   - Non-negative strongly skewed features:
     population, POI counts and densities, road lengths/counts and densities,
     median/per_capita income, and pairwise distance.
     Transform:
         x_log = log(1 + x)
         x_norm = (x_log - mu_s) / sigma_s
   - Symmetric or bounded features:
     Percentages/rates (employment, commute shares, homeownership, education)
     and median age, avg vehicles per household:
     Transform:
         x_norm = (x - mu_s) / sigma_s
   - Distance feature:
     Transform:
         d_log = log(1 + d_km)
         d_norm = (d_log - mu_s^distance) / sigma_s^distance
     where mu_s, sigma_s are estimated only from the protocol's frozen reference
     source-training OD split.

3. Zero-Variance Safety Rule:
   If sigma_s < 10^{-12}:
       x_norm = 0.0, zero_variance_flag = True.
   Never divide by zero. Never compute sigma from target.

4. No Clipping in Main Experiment:
   Target features are normalized using source (mu_s, sigma_s) without clipping,
   preserving genuine out-of-distribution shifts.

5. Missing-Value Handling & Canonical Schema:
   - Schema incompatibility (missing column) raises hard SchemaError.
   - Within-column missing values are imputed strictly using source-city median
     (m_{s,f} = median(x_{s,f}) fitted on valid source values before transform).
   - Target uses the frozen source median value.
   - All transformed features must be finite (assert isfinite).

6. Reproducibility Manifest:
   manifests/source_feature_scalers.csv:
   source_city, feature_name, feature_type, fit_scope, imputation_method, imputation_value, transform, mean, std, zero_variance_flag, n_samples
"""

from typing import Dict, List, Optional, Tuple
import dataclasses
import numpy as np
import pandas as pd
from pathlib import Path

from implement_new_plan.data.dataset import (
    CENSUS_COLS,
    POI_COLS,
    ROAD_COLS,
    NODE_FEATURE_COLUMNS,
    CityData,
    RawCityData,
    load_raw_city,
)

# Strongly skewed features requiring log1p before z-score
SKEWED_FEATURES = set([
    "total_population",
    "median_income",
    "per_capita_income",
    "office",
    "office_density",
    "industrial",
    "industrial_density",
    "commercial",
    "commercial_density",
    "education_primary",
    "education_primary_density",
    "road_length_total",
    "road_density",
    "road_count",
    "motorway_length",
    "primary_length",
    "distance",
])


@dataclasses.dataclass
class SingleFeatureScaler:
    feature_name: str
    feature_type: str  # 'node_feature' or 'pairwise_distance'
    fit_scope: str  # 'all_source_nodes' or 'source_train_od_30pct'
    transform_type: str  # 'log1p_zscore' or 'linear_zscore'
    imputation_method: str  # 'source_median'
    imputation_value: float  # m_{s,f} = median(x_{s,f}) fitted on valid source values
    mean: float
    std: float
    zero_variance_flag: bool
    n_samples: int = 0

    def transform(self, x: np.ndarray) -> np.ndarray:
        x_in = x.copy().astype(np.float64)
        
        # 1. Source-median imputation for any NaN or Inf
        invalid_mask = ~np.isfinite(x_in)
        if np.any(invalid_mask):
            x_in[invalid_mask] = self.imputation_value
        
        # 2. Skewed feature transformation (if applicable)
        if self.transform_type == "log1p_zscore":
            x_in = np.log1p(np.maximum(0.0, x_in))
        
        # 3. Standardization with zero-variance safety rule
        if self.zero_variance_flag or self.std < 1e-12:
            return np.zeros_like(x_in, dtype=np.float32)
        
        x_norm = (x_in - self.mean) / self.std
        return x_norm.astype(np.float32)


class SourceCityFeatureScaler:
    """
    Fitted feature scaler representing a specific source city's observable data.
    Once fitted, it is frozen and applied unchanged to all 49 target cities.
    """
    def __init__(self, source_city: str):
        self.source_city = source_city
        self.node_scalers: Dict[str, SingleFeatureScaler] = {}
        self.distance_scaler: Optional[SingleFeatureScaler] = None
        self.is_fitted: bool = False

    def fit(
        self,
        raw_source_city: RawCityData,
        source_train_indices: np.ndarray,
        distance_fit_scope: str = "source_train_od_30pct",
    ) -> "SourceCityFeatureScaler":
        """
        Fits scalers exclusively on source-side observable data:
        - Node features: fitted on all tracts/nodes of source city.
          Missing values imputed using source median before transform.
        - Distance feature: fitted on the caller-supplied frozen reference
          source-training OD pairs.
          Missing values imputed using source median distance before transform.
        """
        # 1. Fit node features
        X_raw = raw_source_city.X_raw  # (N, 26)
        if X_raw.shape[1] != len(NODE_FEATURE_COLUMNS):
            raise ValueError(
                f"Source city '{self.source_city}' has {X_raw.shape[1]} columns, "
                f"expected {len(NODE_FEATURE_COLUMNS)}"
            )

        for i, col_name in enumerate(NODE_FEATURE_COLUMNS):
            vals = X_raw[:, i].astype(np.float64)
            valid_mask = np.isfinite(vals)
            n_valid = int(np.sum(valid_mask))
            if n_valid == 0:
                raise ValueError(
                    f"Source city '{self.source_city}' has 0 valid values for required feature '{col_name}'"
                )

            # Imputation value: source median of valid entries
            imputation_val = float(np.median(vals[valid_mask]))
            vals_imputed = vals.copy()
            vals_imputed[~valid_mask] = imputation_val

            transform_type = "log1p_zscore" if col_name in SKEWED_FEATURES else "linear_zscore"

            if transform_type == "log1p_zscore":
                vals_t = np.log1p(np.maximum(0.0, vals_imputed))
            else:
                vals_t = vals_imputed

            mean_val = float(np.mean(vals_t))
            std_val = float(np.std(vals_t))
            zero_flag = bool(std_val < 1e-12 or not np.isfinite(std_val))
            if zero_flag:
                std_val = 0.0

            self.node_scalers[col_name] = SingleFeatureScaler(
                feature_name=col_name,
                feature_type="node_feature",
                fit_scope="all_source_nodes",
                transform_type=transform_type,
                imputation_method="source_median",
                imputation_value=imputation_val,
                mean=mean_val,
                std=std_val,
                zero_variance_flag=zero_flag,
                n_samples=X_raw.shape[0],
            )

        if not distance_fit_scope or not isinstance(distance_fit_scope, str):
            raise ValueError("distance_fit_scope must be a non-empty string")

        # 2. Fit distance feature from the caller-supplied reference split.
        train_dists = raw_source_city.dist_km[source_train_indices].astype(np.float64)
        valid_dist_mask = np.isfinite(train_dists)
        if int(np.sum(valid_dist_mask)) == 0:
            raise ValueError(f"Source city '{self.source_city}' has 0 valid distance values in train split")

        d_imputation_val = float(np.median(train_dists[valid_dist_mask]))
        train_dists_imputed = train_dists.copy()
        train_dists_imputed[~valid_dist_mask] = d_imputation_val

        dists_log = np.log1p(np.maximum(0.0, train_dists_imputed))
        d_mean = float(np.mean(dists_log))
        d_std = float(np.std(dists_log))
        d_zero = bool(d_std < 1e-12 or not np.isfinite(d_std))
        if d_zero:
            d_std = 0.0

        self.distance_scaler = SingleFeatureScaler(
            feature_name="distance",
            feature_type="pairwise_distance",
            fit_scope=distance_fit_scope,
            transform_type="log1p_zscore",
            imputation_method="source_median",
            imputation_value=d_imputation_val,
            mean=d_mean,
            std=d_std,
            zero_variance_flag=d_zero,
            n_samples=len(source_train_indices),
        )

        self.is_fitted = True
        return self

    def transform_node_features(self, X_raw: np.ndarray) -> np.ndarray:
        """Transforms node features using frozen source parameters."""
        if not self.is_fitted:
            raise RuntimeError("Cannot transform before scaler is fitted on source city.")
        if X_raw.shape[1] != len(NODE_FEATURE_COLUMNS):
            raise ValueError(
                f"Feature matrix has {X_raw.shape[1]} columns, expected {len(NODE_FEATURE_COLUMNS)}"
            )

        X_norm = np.zeros_like(X_raw, dtype=np.float32)
        for i, col_name in enumerate(NODE_FEATURE_COLUMNS):
            scaler = self.node_scalers[col_name]
            X_norm[:, i] = scaler.transform(X_raw[:, i])
        
        if not np.isfinite(X_norm).all():
            raise ValueError("Transformed node features contain non-finite values (NaN/Inf).")
        return X_norm

    def impute_raw_population(self, population_raw: np.ndarray) -> np.ndarray:
        """
        Imputes missing values in raw population using the source-fitted median,
        preserving the raw population scale (NO log1p, NO z-score).
        
        Strict Contract:
        - Used for Two-Parameter Gravity and explicit gravity priors.
        - Returns non-negative, finite raw population array.
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot impute population before scaler is fitted on source city.")
        pop_arr = population_raw.copy().astype(np.float64)
        pop_scaler = self.node_scalers.get("total_population")
        if pop_scaler is None:
            raise KeyError("Scaler does not contain 'total_population' scaler.")
        
        invalid_mask = ~np.isfinite(pop_arr)
        if np.any(invalid_mask):
            pop_arr[invalid_mask] = pop_scaler.imputation_value
            
        if not np.isfinite(pop_arr).all():
            raise ValueError("Imputed raw population contains non-finite values.")
        if np.any(pop_arr < 0.0):
            raise ValueError("Imputed raw population contains negative values.")
        return pop_arr.astype(np.float32)

    def transform_distances(self, dist_km: np.ndarray) -> np.ndarray:
        """Transforms pairwise distances using frozen source distance parameters."""
        if not self.is_fitted or self.distance_scaler is None:
            raise RuntimeError("Cannot transform distance before scaler is fitted on source city.")
        d_norm = self.distance_scaler.transform(dist_km)
        if not np.isfinite(d_norm).all():
            raise ValueError("Transformed distance contains non-finite values (NaN/Inf).")
        return d_norm

    def to_manifest_records(self) -> List[Dict]:
        """Generates records for manifests/source_feature_scalers.csv."""
        if not self.is_fitted:
            raise RuntimeError("Scaler must be fitted before exporting records.")
        records = []
        for col in NODE_FEATURE_COLUMNS:
            sc = self.node_scalers[col]
            records.append({
                "source_city": self.source_city,
                "feature_name": sc.feature_name,
                "feature_type": sc.feature_type,
                "fit_scope": sc.fit_scope,
                "imputation_method": sc.imputation_method,
                "imputation_value": round(sc.imputation_value, 6),
                "transform": sc.transform_type,
                "mean": round(sc.mean, 6),
                "std": round(sc.std, 6),
                "zero_variance_flag": sc.zero_variance_flag,
                "n_samples": sc.n_samples,
            })
        if self.distance_scaler:
            records.append({
                "source_city": self.source_city,
                "feature_name": self.distance_scaler.feature_name,
                "feature_type": self.distance_scaler.feature_type,
                "fit_scope": self.distance_scaler.fit_scope,
                "imputation_method": self.distance_scaler.imputation_method,
                "imputation_value": round(self.distance_scaler.imputation_value, 6),
                "transform": self.distance_scaler.transform_type,
                "mean": round(self.distance_scaler.mean, 6),
                "std": round(self.distance_scaler.std, 6),
                "zero_variance_flag": self.distance_scaler.zero_variance_flag,
                "n_samples": self.distance_scaler.n_samples,
            })
        return records
