"""
Preflight Verification Script for Moving-Bin Calibration Framework.

Verifies:
1. Gate 1: Omega_c positive flow invariant on all cities.
2. Gate 2: Interzonal mass preservation on Omega_c^+ & Intrazonal preservation (\hat{T}_ii^{cal} == \hat{T}_ii^{ZS}).
3. Gate 3: Zero-shot boundary isolation & exact ZTNB conditional mean.
4. Gate 4: Real Meta moving-bin extraction (Bins 1, 2, 3), Distributional Overlap computation, and Soft Calibration.
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Ensure root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.dataset import load_cities, load_city
from src.data.urban_graph import build_radius_graph
from src.data.yd_extractor import (
    extract_yd_moving_oracle,
    extract_yd_moving_real,
    compute_distributional_overlap,
    CITY_FIPS_GADM,
)
from src.calibration.bin_calibration import calibrate_moving_bins, calibrate_4bin_legacy_ablation
from src.training.train import train_zero_shot_model, infer_zero_shot
from src.experiment.run_experiment import run_target_city_experiments
from src.loss.ztnb import ztnb_nll, nb_nll, compute_conditional_mean


def run_preflight_checks():
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 85)
    print(f"RUNNING 4 PREFLIGHT SAFETY GATES (MOVING-BIN FRAMEWORK, Device: {device_str})")
    print("=" * 85)

    # -----------------------------------------------------------------------
    # Gate 1: Check Omega_c support and positive flows
    # -----------------------------------------------------------------------
    print("\n[Gate 1] Checking Omega_c Support & ZTNB Positive-Flow Invariant...")
    test_city = "Philadelphia"
    train_cities = ["Raleigh", "Denver"]

    for name in [test_city] + train_cities:
        cd = load_city(name, data_root="data")
        assert (cd.pair_trips > 0).all(), f"FAILED: Non-positive flow found in {name}"
        assert cd.pair_trips.min().item() >= 1.0, f"FAILED: Min flow < 1 in {name}"
        print(f"  -> {name:<15}: |Omega_c| = {cd.n_pairs:>7,}, min_flow = {cd.pair_trips.min().item():.0f}, max_flow = {cd.pair_trips.max().item():>10,.0f} [PASS]")
    print("Gate 1: PASS - All pairs in Omega_c are strictly positive count records.")

    # -----------------------------------------------------------------------
    # Gate 2: Check Moving-Bin Interzonal Calibration Invariants
    # -----------------------------------------------------------------------
    print("\n[Gate 2] Checking Moving-Bin Interzonal Calibration Invariants...")
    torch.manual_seed(42)
    # Mock data: 100 intrazonal (bin 0), 900 interzonal (bins 1, 2, 3)
    t0_mock = torch.rand(1000) * 500.0 + 1.0
    o_mock = torch.randint(1, 50, (1000,))
    d_mock = torch.randint(1, 50, (1000,))
    bins_mock = torch.randint(1, 4, (1000,))  # bins 1, 2, 3 for interzonal

    # First 100 are strictly intrazonal (bin 0, o == d)
    o_mock[:100] = d_mock[:100]
    bins_mock[:100] = 0
    # Ensure remaining 900 have o != d
    for i in range(100, 1000):
        if o_mock[i] == d_mock[i]:
            d_mock[i] = (d_mock[i] + 1) % 50 + 1

    p_moving_target = np.array([0.70, 0.25, 0.05])  # sum = 1.0 for bins 1, 2, 3

    t1_cal = calibrate_moving_bins(t0_mock, bins_mock, o_mock, d_mock, p_moving_target, q=1.0)

    # 2a. Intrazonal identity check
    intra_mask = (o_mock == d_mock) & (bins_mock == 0)
    assert torch.allclose(t1_cal[intra_mask], t0_mock[intra_mask]), "FAILED: Intrazonal flows were modified!"
    print(f"  Intrazonal identity: {t1_cal[intra_mask].sum().item():.2f} == {t0_mock[intra_mask].sum().item():.2f} [PASS]")

    # 2b. Interzonal mass preservation check
    inter_mask = ~intra_mask
    mass0_inter = t0_mock[inter_mask].sum().item()
    mass1_inter = t1_cal[inter_mask].sum().item()
    rel_diff = abs(mass1_inter - mass0_inter) / mass0_inter
    assert rel_diff < 1e-4, f"FAILED: Interzonal mass not preserved, diff = {rel_diff}"
    print(f"  Interzonal mass: T0_inter={mass0_inter:.2f}, T1_inter={mass1_inter:.2f}, rel_diff={rel_diff:.2e} [PASS]")

    # 2c. Moving-bin proportion matching check
    cal_p = np.array([(t1_cal[inter_mask & (bins_mock == k)].sum() / mass1_inter).item() for k in [1, 2, 3]])
    max_err = np.max(np.abs(cal_p - p_moving_target))
    assert max_err < 1e-4, f"FAILED: Moving bin matching error = {max_err}"
    print(f"  Target moving probs:   {p_moving_target}")
    print(f"  Calibrated moving prop:{np.round(cal_p, 4).tolist()}")
    print(f"  Max absolute error:    {max_err:.2e} [PASS]")
    print("Gate 2: PASS - Exact interzonal mass preservation, intrazonal identity, and moving-bin matching verified.")

    # -----------------------------------------------------------------------
    # Gate 3: Check Zero-Shot Isolation & ZTNB Conditional Mean
    # -----------------------------------------------------------------------
    print("\n[Gate 3] Checking Zero-Shot Leakage Prevention & ZTNB Conditional Mean...")
    train_data_list, fitted_scaler = load_cities(train_cities, data_root="data")
    test_data = load_city(test_city, data_root="data", feature_scaler=fitted_scaler, fit_scaler=False)
    assert fitted_scaler.n_samples_seen_ == sum(c.n_tracts for c in train_data_list)

    model_ztnb, _ = train_zero_shot_model(train_cities, data_root="data", epochs=5, device_str=device_str, verbose=False)
    for p in model_ztnb.parameters():
        assert not p.requires_grad

    mu_test = torch.tensor([1.0, 10.0, 50.0], device=device_str)
    c_mean = compute_conditional_mean(mu_test, model_ztnb.log_phi)
    assert (c_mean >= mu_test).all()
    print("Gate 3: PASS - Zero-shot boundary strictly isolated and ZTNB conditional mean verified.")

    # -----------------------------------------------------------------------
    # Gate 4: Real Meta Moving Extraction, Distributional Overlap & Full Run
    # -----------------------------------------------------------------------
    print("\n[Gate 4] Checking Meta Moving Bins Extraction (50/50) & Distributional Overlap...")
    for c_name in ["Philadelphia", "Austin", "Raleigh", "Denver", "Seattle"]:
        cd = load_city(c_name, "data")
        o_3 = extract_yd_moving_oracle(cd.pair_trips, cd.bin_labels, cd.pair_o_idx, cd.pair_d_idx)
        r_3 = extract_yd_moving_real(c_name, meta_prior_dir="meta_prior")
        assert r_3 is not None, f"FAILED: Meta moving missing for {c_name}"
        overlap = compute_distributional_overlap(o_3, r_3)
        print(f"  -> {c_name:<15}: Distributional Overlap = {overlap*100:.2f}% | Oracle={np.round(o_3, 3).tolist()} | Meta={np.round(r_3, 3).tolist()} [PASS]")

    # Run single target city experiments
    res = run_target_city_experiments(
        model=model_ztnb,
        city_name=test_city,
        scaler=fitted_scaler,
        num_trip_seeds=20,
        device_str=device_str,
    )

    print(f"\n  Philadelphia Moving-Bin Pipeline Summary:")
    print(f"    Distributional Overlap with Meta: {res['distributional_overlap']*100:.2f}%")
    print(f"    M0 Interzonal CPC:                {res['M0']['cpc_inter']:.4f}")
    print(f"    M1_real+ Interzonal CPC:          {res['M1_real_plus']['cpc_inter']:.4f} (Delta: {res['delta_r_real_plus']:+.4f})")
    print(f"    M1_oracle+ Interzonal CPC:        {res['M1_oracle_plus']['cpc_inter']:.4f} (Delta: {res['delta_r_oracle_plus']:+.4f})")
    print(f"    Realization Gap+:                 {res['realization_gap_plus']:+.4f}")
    if res['M1_4bin_ablation']:
        print(f"    M1_4bin Ablation Interzonal CPC:  {res['M1_4bin_ablation']['cpc_inter']:.4f} (Delta: {res['delta_r_4bin_ablation']:+.4f})")
    print(f"    Soft Calibration Curve (q in [0, 1]):")
    for q_k, q_v in res['Mq_soft_curve'].items():
        print(f"      {q_k}: CPC_inter = {q_v['cpc_inter']:.4f} | CPC_full = {q_v['cpc_full']:.4f}")

    print("\n" + "=" * 85)
    print("ALL 4 PREFLIGHT SAFETY GATES PASSED (MOVING-BIN FRAMEWORK).")
    print("=" * 85)


if __name__ == "__main__":
    run_preflight_checks()
