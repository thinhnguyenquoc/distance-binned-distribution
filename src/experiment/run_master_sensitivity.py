"""
Master Sensitivity Grid Pipeline (Experiments B, C, D Unified Runner).

Protocol Invariants:
1. One Master Grid: K in {2, 4, 8, 12, 20} x eps in {0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10}.
2. Experiment B is the eps == 0 slice.
3. Experiment C is the K == 8 slice.
4. Experiment D is the full K x eps grid.
5. Unique-case enforcement: each (source, target, model, model_seed, K, epsilon, realization_id) is run ONCE.
6. Baseline prediction is cached and reused across (K, epsilon, realization).
7. Noise realization is independent of model/model_seed and cached by (source, target, K, epsilon, realization_id).
8. Volume preservation check: |sum T_cal - sum T_base| < 1e-10.
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Set, Any
import numpy as np
import pandas as pd

from src.calibration.tv_noise import generate_exact_tv_noise, derive_noise_seed
from src.calibration.source_bins import (
    compute_source_distance_cap,
    build_source_bin_edges,
    assign_to_source_bins,
    compute_pure_calibration_ratios,
    apply_pure_dbd_calibration,
)
from src.training.evaluate import compute_cpc_pair

K_GRID = [2, 4, 8, 12, 20]
EPS_GRID = [0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10]
NUM_REALIZATIONS = 20
GLOBAL_NOISE_SEED = 42

MASTER_OUTPUT_FILE = "noise_robustness_results.csv"
DERIVED_B_FILE = "experiment_b_summary.csv"
DERIVED_C_FILE = "experiment_c_summary.csv"
DERIVED_D_FILE = "experiment_d_summary.csv"


class MasterSensitivityRunner:
    """
    Executes and caches the master sensitivity grid to prevent redundant computation.
    """
    def __init__(
        self,
        output_dir: Path | str = "results",
        global_noise_seed: int = GLOBAL_NOISE_SEED,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.master_csv_path = self.output_dir / MASTER_OUTPUT_FILE
        self.global_noise_seed = global_noise_seed

        # Caches
        self.completed_cases: Set[Tuple] = set()
        self.noise_cache: Dict[Tuple, Tuple[np.ndarray, float, int]] = {}
        self.bin_cache: Dict[Tuple[str, int], np.ndarray] = {}  # (source_city, K) -> bin_edges

        self._load_existing_results()

    def _load_existing_results(self) -> None:
        """Loads completed cases from master output file for resume capability."""
        if self.master_csv_path.exists():
            df = pd.read_csv(self.master_csv_path)
            for _, row in df.iterrows():
                key = (
                    str(row["source_city"]),
                    str(row["target_city"]),
                    str(row["model"]),
                    int(row["model_seed"]),
                    int(row["K"]),
                    round(float(row["epsilon"]), 6),
                    int(row["realization_id"]),
                )
                self.completed_cases.add(key)
            print(f"Loaded {len(self.completed_cases)} existing cases from {self.master_csv_path}")

    def get_noise_realization(
        self,
        p_oracle: np.ndarray,
        source_city: str,
        target_city: str,
        K: int,
        epsilon: float,
        realization_id: int,
    ) -> Tuple[np.ndarray, float, int]:
        """
        Retrieves or generates exact TV noise realization.
        Deterministic and shared across all models and seeds.
        """
        cache_key = (source_city, target_city, K, round(epsilon, 6), realization_id)
        if cache_key in self.noise_cache:
            return self.noise_cache[cache_key]

        if epsilon == 0.0 or abs(epsilon) < 1e-12:
            res = (p_oracle.copy().astype(np.float64), 0.0, 0)
        else:
            noise_seed = derive_noise_seed(
                self.global_noise_seed,
                source_city,
                target_city,
                K,
                epsilon,
                realization_id,
            )
            p_tilde, actual_tv, attempts = generate_exact_tv_noise(
                p=p_oracle,
                epsilon=epsilon,
                rng_or_seed=noise_seed,
                source_city=source_city,
                target_city=target_city,
                K=K,
                realization_id=realization_id,
                max_attempts=10000,
            )
            res = (p_tilde, actual_tv, noise_seed)

        self.noise_cache[cache_key] = res
        return res

    def evaluate_pair_sensitivity(
        self,
        source_city: str,
        target_city: str,
        model_name: str,
        model_seed: int,
        t_true: np.ndarray,
        t_base: np.ndarray,
        d_raw_km: np.ndarray,
        source_train_d_raw: np.ndarray,
        k_values: List[int] = K_GRID,
        eps_values: List[float] = EPS_GRID,
    ) -> List[Dict[str, Any]]:
        """
        Runs sensitivity grid on a single transfer pair with pre-computed baseline prediction.
        Model inference is NEVER called here.
        """
        assert len(t_true) == len(t_base) == len(d_raw_km)
        assert np.all(d_raw_km > 0), "Distance must be strictly positive on support."
        assert np.all(t_true >= 1.0), "Ground truth flow must be >= 1 on positive support."

        # Compute baseline metrics once
        cpc_before = compute_cpc_pair(t_true, t_base)
        mae_before = float(np.mean(np.abs(t_true - t_base)))
        mse_before = float(np.mean((t_true - t_base) ** 2))

        # Compute or fetch D_cap for source city
        d_cap = compute_source_distance_cap(source_train_d_raw, percentile=99.0)

        results = []

        for K in k_values:
            # 1. Source bin edges
            bin_key = (source_city, K)
            if bin_key not in self.bin_cache:
                self.bin_cache[bin_key] = build_source_bin_edges(d_cap, K)
            bin_edges = self.bin_cache[bin_key]

            # 2. Bin assignment using raw physical km
            bin_ids = assign_to_source_bins(d_raw_km, bin_edges)

            # 3. Baseline predicted DBD (q) and True target DBD (p)
            sum_t_base = np.sum(t_base)
            sum_t_true = np.sum(t_true)
            q_base = np.zeros(K, dtype=np.float64)
            p_true = np.zeros(K, dtype=np.float64)

            for b in range(K):
                mask = (bin_ids == b)
                if np.any(mask):
                    q_base[b] = np.sum(t_base[mask])
                    p_true[b] = np.sum(t_true[mask])

            q_base = q_base / sum_t_base
            p_true = p_true / sum_t_true

            # 4. Iterate over epsilon levels
            for eps in eps_values:
                # Realizations: single run for eps=0, 20 runs for eps>0
                realizations = [0] if (eps == 0.0 or abs(eps) < 1e-12) else list(range(NUM_REALIZATIONS))

                for r_id in realizations:
                    case_key = (
                        source_city,
                        target_city,
                        model_name,
                        model_seed,
                        K,
                        round(eps, 6),
                        r_id,
                    )

                    # Resume check
                    if case_key in self.completed_cases:
                        continue

                    # Get noise realization (cached across models/seeds)
                    p_tilde, actual_tv, noise_seed = self.get_noise_realization(
                        p_oracle=p_true,
                        source_city=source_city,
                        target_city=target_city,
                        K=K,
                        epsilon=eps,
                        realization_id=r_id,
                    )

                    # 5. Pure DBD Calibration (No epsilon smoothing)
                    r_ratios = compute_pure_calibration_ratios(p_b=p_tilde, q_b=q_base)
                    t_cal = apply_pure_dbd_calibration(t_base, bin_ids, r_ratios)

                    # Exact Volume Preservation Check (< 1e-10)
                    vol_diff = abs(float(np.sum(t_cal) - sum_t_base))
                    assert vol_diff < 1e-10, f"Volume preservation violated: diff={vol_diff}"

                    # 6. Evaluate metrics on original flow scale
                    cpc_after = compute_cpc_pair(t_true, t_cal)
                    mae_after = float(np.mean(np.abs(t_true - t_cal)))
                    mse_after = float(np.mean((t_true - t_cal) ** 2))

                    delta_cpc = cpc_after - cpc_before
                    delta_mae = mae_before - mae_after
                    delta_mse = mse_before - mse_after

                    record = {
                        "source_city": source_city,
                        "target_city": target_city,
                        "model": model_name,
                        "model_seed": model_seed,
                        "K": K,
                        "epsilon": eps,
                        "realization_id": r_id,
                        "noise_seed": noise_seed,
                        "actual_TV": round(actual_tv, 6),
                        "CPC_before": round(cpc_before, 6),
                        "CPC_after": round(cpc_after, 6),
                        "delta_CPC": round(delta_cpc, 6),
                        "MAE_before": round(mae_before, 4),
                        "MAE_after": round(mae_after, 4),
                        "delta_MAE": round(delta_mae, 4),
                        "MSE_before": round(mse_before, 2),
                        "MSE_after": round(mse_after, 2),
                        "delta_MSE": round(delta_mse, 2),
                    }
                    results.append(record)
                    self.completed_cases.add(case_key)

        # Append to master output file incrementally
        if results:
            df_new = pd.DataFrame(results)
            write_header = not self.master_csv_path.exists()
            df_new.to_csv(self.master_csv_path, mode="a", index=False, header=write_header)

        return results

    def generate_derived_summaries(self) -> None:
        """
        Derives Experiment B, C, and D summaries directly from master results.
        No independent reruns or separate calibrations are performed.
        """
        if not self.master_csv_path.exists():
            print("Master sensitivity file does not exist yet. Run evaluations first.")
            return

        df_master = pd.read_csv(self.master_csv_path)

        # -------------------------------------------------------------
        # Derived View B: Experiment B = Master Grid | epsilon == 0
        # -------------------------------------------------------------
        df_b = df_master[df_master["epsilon"] == 0.0].copy()
        summary_b = (
            df_b.groupby(["model", "K"])[["delta_CPC", "delta_MAE", "delta_MSE"]]
            .agg(["mean", "median", "std"])
            .reset_index()
        )
        summary_b.to_csv(self.output_dir / DERIVED_B_FILE, index=False)
        print(f"Generated Experiment B summary: {self.output_dir / DERIVED_B_FILE} ({len(df_b)} rows)")

        # -------------------------------------------------------------
        # Derived View C: Experiment C = Master Grid | K == 8
        # -------------------------------------------------------------
        df_c = df_master[df_master["K"] == 8].copy()
        summary_c = (
            df_c.groupby(["model", "epsilon"])[["delta_CPC", "delta_MAE", "delta_MSE"]]
            .agg(["mean", "median", "std"])
            .reset_index()
        )
        summary_c.to_csv(self.output_dir / DERIVED_C_FILE, index=False)
        print(f"Generated Experiment C summary: {self.output_dir / DERIVED_C_FILE} ({len(df_c)} rows)")

        # -------------------------------------------------------------
        # Derived View D: Experiment D = Master Grid Interaction (K x epsilon)
        # -------------------------------------------------------------
        summary_d = (
            df_master.groupby(["model", "K", "epsilon"])[["delta_CPC", "delta_MAE", "delta_MSE"]]
            .agg(["mean", "median", "std"])
            .reset_index()
        )
        summary_d.to_csv(self.output_dir / DERIVED_D_FILE, index=False)
        print(f"Generated Experiment D summary: {self.output_dir / DERIVED_D_FILE} ({len(df_master)} rows)")

    def extract_main_experiment_a(self, output_filename: str = "calibration_results.csv") -> pd.DataFrame:
        r"""
        Extracts the Pre-Specified Primary Calibration Result (Experiment A) from the master dataset:
            K == 8  and  epsilon == 0.0

        Strict Invariants:
        1. Pre-specified configuration: K=8, epsilon=0. Never tuned based on Experiment B/C/D.
        2. Applies to all three baseline model families (gravity_2param, pairwise_mlp, urban_gnn).
        3. Output file calibration_results.csv strictly contains rows matching K=8 and epsilon=0.
        4. Validates sanity assertions:
               assert (df['K'] == 8).all()
               assert (df['epsilon'] == 0).all()
        """
        if not self.master_csv_path.exists():
            raise FileNotFoundError(f"Master sensitivity file does not exist: {self.master_csv_path}")

        df_master = pd.read_csv(self.master_csv_path)

        # Filter strictly for pre-specified main configuration: K=8, epsilon=0
        df_main = df_master[(df_master["K"] == 8) & (df_master["epsilon"] == 0.0)].copy()

        # Sanity check assertions
        if not (df_main["K"] == 8).all():
            raise AssertionError("Main calibration results must strictly contain K=8 rows only.")
        if not (df_main["epsilon"] == 0.0).all():
            raise AssertionError("Main calibration results must strictly contain epsilon=0 rows only.")

        # Main calibration results schema
        main_cols = [
            "source_city", "target_city", "model", "model_seed", "K",
            "CPC_before", "CPC_after", "delta_CPC",
            "MAE_before", "MAE_after", "delta_MAE",
            "MSE_before", "MSE_after", "delta_MSE"
        ]
        avail_cols = [c for c in main_cols if c in df_main.columns]
        df_main_out = df_main[avail_cols].copy()
        # Rename model_seed to seed for calibration_results.csv schema standard
        if "model_seed" in df_main_out.columns:
            df_main_out = df_main_out.rename(columns={"model_seed": "seed"})

        out_path = self.output_dir / output_filename
        df_main_out.to_csv(out_path, index=False)
        print(f"Extracted pre-specified Experiment A (K=8, eps=0): {out_path} ({len(df_main_out)} rows)")
        return df_main_out
