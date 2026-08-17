"""
3-City Pilot Experiment.

Trains on Raleigh (small) + Denver (medium) -> Evaluates zero-shot on Philadelphia (large).
Compares:
1. ZTNB (primary conservative model) vs NB (sensitivity model)
2. M0 (Zero-shot) vs M1_oracle vs Mq
3. Verifies numerical stability, phi convergence, and delta R calculation.
"""

import sys
import json
import torch

from src.training.train import train_zero_shot_model
from src.experiment.run_experiment import run_target_city_experiments


def main():
    print("=" * 70)
    print("STARTING 3-CITY PILOT EXPERIMENT")
    print("Source cities: ['Raleigh', 'Denver']")
    print("Target city (held-out): 'Philadelphia'")
    print("=" * 70)

    train_cities = ["Raleigh", "Denver"]
    target_city = "Philadelphia"
    data_root = "data"
    meta_prior_dir = "meta_prior"

    # --- Run 1: ZTNB (Primary) ---
    print("\n[1/2] Training Zero-Shot Model with ZTNB Loss...")
    model_ztnb, scaler_ztnb = train_zero_shot_model(
        train_city_names=train_cities,
        data_root=data_root,
        epochs=20,
        lr=2e-3,
        hidden_dim=48,
        loss_type="ztnb",
        verbose=True,
    )

    print(f"\nEvaluating ZTNB model on held-out target city: {target_city}...")
    res_ztnb = run_target_city_experiments(
        model=model_ztnb,
        city_name=target_city,
        scaler=scaler_ztnb,
        data_root=data_root,
        meta_prior_dir=meta_prior_dir,
        num_trip_seeds=5,
    )

    # --- Run 2: NB (Sensitivity) ---
    print("\n[2/2] Training Zero-Shot Model with NB Loss (Sensitivity Check)...")
    model_nb, scaler_nb = train_zero_shot_model(
        train_city_names=train_cities,
        data_root=data_root,
        epochs=20,
        lr=2e-3,
        hidden_dim=48,
        loss_type="nb",
        verbose=True,
    )

    print(f"\nEvaluating NB model on held-out target city: {target_city}...")
    res_nb = run_target_city_experiments(
        model=model_nb,
        city_name=target_city,
        scaler=scaler_nb,
        data_root=data_root,
        meta_prior_dir=meta_prior_dir,
        num_trip_seeds=5,
    )

    # --- Summary Comparison ---
    print("\n" + "=" * 70)
    print(f"PILOT RESULTS SUMMARY ON HELD-OUT: {target_city}")
    print("=" * 70)
    print(f"{'Condition / Metric':<30} {'ZTNB (Primary)':>18} {'NB (Sensitivity)':>18}")
    print("-" * 70)
    print(f"{'M0 Interzonal CPC':<30} {res_ztnb['M0']['cpc_inter']:>18.4f} {res_nb['M0']['cpc_inter']:>18.4f}")
    print(f"{'M1_oracle+ Interzonal CPC':<30} {res_ztnb['M1_oracle_plus']['cpc_inter']:>18.4f} {res_nb['M1_oracle_plus']['cpc_inter']:>18.4f}")
    print(f"{'Delta R+ (CPC gain)':<30} {res_ztnb['delta_r_oracle_plus']:>18.4f} {res_nb['delta_r_oracle_plus']:>18.4f}")
    print(f"{'M0 RMSE_inter':<30} {res_ztnb['M0']['rmse_inter']:>18.4f} {res_nb['M0']['rmse_inter']:>18.4f}")
    print(f"{'M1_oracle+ RMSE_inter':<30} {res_ztnb['M1_oracle_plus']['rmse_inter']:>18.4f} {res_nb['M1_oracle_plus']['rmse_inter']:>18.4f}")
    print(f"{'m* (oracle interzonal)':<30} {res_ztnb['m_star_oracle']:>18.1f} {res_nb['m_star_oracle']:>18.1f}")
    print(f"{'q* (oracle interzonal)':<30} {res_ztnb['q_star_oracle']:>18.6f} {res_nb['q_star_oracle']:>18.6f}")

    if res_ztnb['M1_real_plus'] is not None:
        print(f"{'M1_real+ Interzonal CPC':<30} {res_ztnb['M1_real_plus']['cpc_inter']:>18.4f} {res_nb['M1_real_plus']['cpc_inter']:>18.4f}")
        print(f"{'Realization gap+':<30} {res_ztnb['realization_gap_plus']:>18.4f} {res_nb['realization_gap_plus']:>18.4f}")
    else:
        print(f"{'M1_real+ (Meta mobility)':<30} {'Not in prior dir':>18} {'Not in prior dir':>18}")

    print("\nM_m Sampling Curve (ZTNB):")
    for k, v in res_ztnb["Mm_sampling_curve"].items():
        print(f"  m = {k:>7}: CPC_inter = {v['cpc_inter_mean']:.4f} +- {v['cpc_inter_std']:.4f} | RMSE_inter = {v['rmse_inter_mean']:.4f}")

    print("\n" + "=" * 70)
    print("3-CITY PILOT COMPLETED SUCCESSFULLY.")
    print("=" * 70)


if __name__ == "__main__":
    main()
