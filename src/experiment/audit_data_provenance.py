"""
Data Provenance Audit: End-to-End Verification

This script picks representative cities, loads checkpoints fresh,
recomputes all key quantities from scratch, and cross-checks against
every CSV/JSON file used in the frozen reports.

Goal: Ensure NO obsolete/stale data is used in any report.
"""

import sys
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dataset import load_city
from src.data.urban_graph import build_radius_graph
from src.data.yd_extractor import compute_kbin_edges, extract_yd_kbins
from src.data.city_splits import load_splits_manifest_v2
from src.training.train import load_checkpoint, infer_zero_shot
from src.calibration.bin_calibration import calibrate_kbins

# ─── Configuration ───────────────────────────────────────────────────────
SEEDS = [1, 10, 100]
K = 8
Q = 1.0
DATA_ROOT = "data"
MANIFEST = "results/e1/splits_manifest_v2.json"
SAMPLE_CITIES_PER_FOLD = 2  # check 2 cities per fold = 10 cities total
ATOL_CPC = 1e-6  # tolerance for CPC match
ATOL_WEIGHT = 1e-8  # tolerance for weight match

def compute_cpc(t_true, t_pred):
    denom = t_true.sum() + t_pred.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * np.minimum(t_true, t_pred).sum() / denom)


def main():
    splits = load_splits_manifest_v2(MANIFEST, data_root=DATA_ROOT)

    # Load reference CSVs
    k_raw = pd.read_csv("results/k_sensitivity_v1/k_sensitivity_raw.csv")
    k_raw_k8 = k_raw[k_raw["K"] == K].copy()

    placebo_csv_path = Path("results/unified_placebo_v1/unified_placebo_per_city.csv")
    has_placebo = placebo_csv_path.exists()
    if has_placebo:
        placebo_df = pd.read_csv(placebo_csv_path)

    calib_audit_path = Path("results/audit/calibration_weight_audit_per_city.csv")
    has_calib = calib_audit_path.exists()
    if has_calib:
        calib_df = pd.read_csv(calib_audit_path)

    noise_per_city_path = Path("results/noise_robustness_fine_v1/noise_per_city.csv")
    has_noise = noise_per_city_path.exists()
    if has_noise:
        noise_df = pd.read_csv(noise_per_city_path)

    total_checks = 0
    total_pass = 0
    total_fail = 0
    failures = []

    print("=" * 80)
    print("DATA PROVENANCE AUDIT: End-to-End Verification")
    print("=" * 80)

    for fold_id in range(1, 6):
        split = splits[fold_id]
        train_cities = split["train"]
        test_cities = split["test"]

        # Compute bin edges from training cities (fresh)
        bin_edges, _ = compute_kbin_edges(train_cities, K=K, data_root=DATA_ROOT)

        # Pick sample cities
        sample_cities = test_cities[:SAMPLE_CITIES_PER_FOLD]

        for city in sample_cities:
            print(f"\n--- Fold {fold_id}, City: {city} ---")

            for seed in SEEDS:
                ckpt_path = Path(f"results/checkpoints/5fold_fold{fold_id}_seed{seed}.pt")
                if not ckpt_path.exists():
                    print(f"  [SKIP] Checkpoint not found: {ckpt_path}")
                    continue

                model, scaler, _ = load_checkpoint(ckpt_path, device_str="cpu")
                model.eval()

                cd = load_city(city, data_root=DATA_ROOT, feature_scaler=scaler, fit_scaler=False)
                ei, ed = build_radius_graph(cd.lon_lat, radius_km=5.0, include_self_loop=True, cache_key=f"{city}_tracts")

                dist_km = np.expm1(cd.pair_distance.numpy())
                inter_mask = (cd.pair_o_idx.numpy() != cd.pair_d_idx.numpy()) & (dist_km > 0.0)
                t_gt = cd.pair_trips.numpy().astype(np.float64)

                with torch.no_grad():
                    t0_tensor = infer_zero_shot(model, cd, ei, ed, device="cpu")
                t0 = t0_tensor.numpy().astype(np.float64)

                t0_inter = t0[inter_mask]
                t_gt_inter = t_gt[inter_mask]
                dist_inter = dist_km[inter_mask]

                # ── Check 1: M0 CPC ──
                cpc0_fresh = compute_cpc(t_gt_inter, t0_inter)

                ref_row = k_raw_k8[(k_raw_k8["city"] == city) &
                                   (k_raw_k8["fold"] == fold_id) &
                                   (k_raw_k8["seed"] == seed)]
                if len(ref_row) == 1:
                    cpc0_ref = ref_row["m0_cpc_inter"].values[0]
                    match = abs(cpc0_fresh - cpc0_ref) < ATOL_CPC
                    total_checks += 1
                    if match:
                        total_pass += 1
                    else:
                        total_fail += 1
                        failures.append(f"M0 CPC mismatch: {city}/fold{fold_id}/seed{seed}: fresh={cpc0_fresh:.8f} vs csv={cpc0_ref:.8f}")
                    print(f"  Seed {seed}: M0 CPC fresh={cpc0_fresh:.8f} vs k_sensitivity_raw={cpc0_ref:.8f} -> {'PASS' if match else 'FAIL'}")
                else:
                    print(f"  Seed {seed}: [WARN] No matching row in k_sensitivity_raw.csv")

                # ── Check 2: Calibrated M1 CPC ──
                yd_target = extract_yd_kbins(dist_km, t_gt, bin_edges, inter_mask)
                t1 = calibrate_kbins(
                    t0, dist_km, inter_mask, yd_target, bin_edges, q=Q
                )
                t1_inter = t1[inter_mask]
                cpc1_fresh = compute_cpc(t_gt_inter, t1_inter)

                if len(ref_row) == 1:
                    cpc1_ref = ref_row["m1_cpc_inter"].values[0]
                    match = abs(cpc1_fresh - cpc1_ref) < ATOL_CPC
                    total_checks += 1
                    if match:
                        total_pass += 1
                    else:
                        total_fail += 1
                        failures.append(f"M1 CPC mismatch: {city}/fold{fold_id}/seed{seed}: fresh={cpc1_fresh:.8f} vs csv={cpc1_ref:.8f}")
                    print(f"  Seed {seed}: M1 CPC fresh={cpc1_fresh:.8f} vs k_sensitivity_raw={cpc1_ref:.8f} -> {'PASS' if match else 'FAIL'}")

                # ── Check 3: Delta CPC ──
                delta_fresh = cpc1_fresh - cpc0_fresh
                if len(ref_row) == 1:
                    delta_ref = ref_row["delta_cpc"].values[0]
                    match = abs(delta_fresh - delta_ref) < ATOL_CPC
                    total_checks += 1
                    if match:
                        total_pass += 1
                    else:
                        total_fail += 1
                        failures.append(f"Delta CPC mismatch: {city}/fold{fold_id}/seed{seed}: fresh={delta_fresh:.8f} vs csv={delta_ref:.8f}")
                    print(f"  Seed {seed}: Delta CPC fresh={delta_fresh:.8f} vs k_sensitivity_raw={delta_ref:.8f} -> {'PASS' if match else 'FAIL'}")

                # ── Check 4: Calibration weight invariants ──
                # Recompute w_min and w_max from scratch
                N_hat = t0_inter.sum()
                Y_hat = np.zeros(K, dtype=np.float64)
                active_bins = np.zeros(K, dtype=bool)
                dist_inter = dist_km[inter_mask]
                for k in range(K):
                    lo, hi = float(bin_edges[k]), float(bin_edges[k+1])
                    in_bin = (dist_inter > lo) & (dist_inter <= hi)
                    Y_hat[k] = t0_inter[in_bin].sum() / N_hat if N_hat > 0 else 0
                    active_bins[k] = bool(in_bin.any())

                yd_active = yd_target * active_bins.astype(np.float64)
                active_sum = yd_active.sum()
                Y_D_cond = yd_active / active_sum if active_sum > 0 else Y_hat.copy()

                w_fresh = np.ones(K, dtype=np.float64)
                for k in range(K):
                    if active_bins[k] and Y_hat[k] > 0:
                        w_fresh[k] = Y_D_cond[k] / Y_hat[k]

                wmin_fresh = float(np.min(w_fresh[active_bins]))
                wmax_fresh = float(np.max(w_fresh[active_bins]))

                if has_calib:
                    calib_row = calib_df[(calib_df["city"] == city) &
                                        (calib_df["fold"] == fold_id) &
                                        (calib_df["seed"] == seed)]
                    if len(calib_row) == 1:
                        wmin_ref = calib_row["w_raw_min"].values[0]
                        wmax_ref = calib_row["w_raw_max"].values[0]
                        match_wmin = abs(wmin_fresh - wmin_ref) < ATOL_WEIGHT
                        match_wmax = abs(wmax_fresh - wmax_ref) < ATOL_WEIGHT
                        total_checks += 2
                        if match_wmin:
                            total_pass += 1
                        else:
                            total_fail += 1
                            failures.append(f"w_min mismatch: {city}/fold{fold_id}/seed{seed}: fresh={wmin_fresh:.8f} vs csv={wmin_ref:.8f}")
                        if match_wmax:
                            total_pass += 1
                        else:
                            total_fail += 1
                            failures.append(f"w_max mismatch: {city}/fold{fold_id}/seed{seed}: fresh={wmax_fresh:.8f} vs csv={wmax_ref:.8f}")
                        print(f"  Seed {seed}: w_min fresh={wmin_fresh:.6f} vs audit={wmin_ref:.6f} -> {'PASS' if match_wmin else 'FAIL'}")
                        print(f"  Seed {seed}: w_max fresh={wmax_fresh:.6f} vs audit={wmax_ref:.6f} -> {'PASS' if match_wmax else 'FAIL'}")

                # Check: w_min < 1 and w_max > 1
                total_checks += 2
                if wmin_fresh < 1.0:
                    total_pass += 1
                else:
                    total_fail += 1
                    failures.append(f"w_min >= 1: {city}/fold{fold_id}/seed{seed}: w_min={wmin_fresh}")
                if wmax_fresh > 1.0:
                    total_pass += 1
                else:
                    total_fail += 1
                    failures.append(f"w_max <= 1: {city}/fold{fold_id}/seed{seed}: w_max={wmax_fresh}")

                # Check flow conservation
                mass_err_fresh = abs(t1_inter.sum() - t0_inter.sum()) / t0_inter.sum()
                total_checks += 1
                if mass_err_fresh < 1e-12:
                    total_pass += 1
                else:
                    total_fail += 1
                    failures.append(f"Flow conservation fail: {city}/fold{fold_id}/seed{seed}: relative_err={mass_err_fresh:.2e}")
                print(f"  Seed {seed}: w_min={wmin_fresh:.4f}<1={'OK' if wmin_fresh<1 else 'FAIL'}, w_max={wmax_fresh:.4f}>1={'OK' if wmax_fresh>1 else 'FAIL'}, mass_err={mass_err_fresh:.2e}")

            # ── Check 5: Seed-averaged placebo cross-check ──
            if has_placebo:
                p_row = placebo_df[(placebo_df["city"] == city) & (placebo_df["fold"] == fold_id)]
                if len(p_row) == 1:
                    # Recompute seed-averaged target delta from k_sensitivity_raw
                    k8_city = k_raw_k8[(k_raw_k8["city"] == city) & (k_raw_k8["fold"] == fold_id)]
                    if len(k8_city) == 3:
                        delta_avg_k8 = k8_city["delta_cpc"].mean()
                        delta_placebo = p_row["d_cpc_target"].values[0]
                        match = abs(delta_avg_k8 - delta_placebo) < ATOL_CPC
                        total_checks += 1
                        if match:
                            total_pass += 1
                        else:
                            total_fail += 1
                            failures.append(f"Target delta cross-source mismatch: {city}: k_sensitivity_avg={delta_avg_k8:.8f} vs placebo_csv={delta_placebo:.8f}")
                        print(f"  Cross-source target delta: k_sensitivity_avg={delta_avg_k8:.8f} vs placebo_csv={delta_placebo:.8f} -> {'PASS' if match else 'FAIL'}")

    # ── Check 6: Population-level statistics ──
    print("\n" + "=" * 80)
    print("POPULATION-LEVEL CROSS-CHECKS")
    print("=" * 80)

    # K=8 seed-averaged city means
    city_means = k_raw_k8.groupby(["fold", "city"])["delta_cpc"].mean()
    pop_mean = city_means.mean()
    print(f"\nK=8 population mean Delta CPC (from k_sensitivity_raw.csv): {pop_mean:.8f}")
    print(f"Expected (manuscript): 0.00353949")
    match = abs(pop_mean - 0.003539) < 1e-4
    total_checks += 1
    if match:
        total_pass += 1
    else:
        total_fail += 1
        failures.append(f"Population mean mismatch: {pop_mean:.8f} vs expected 0.003539")
    print(f"  -> {'PASS' if match else 'FAIL'}")

    # M0 and M1 population means
    m0_mean = k_raw_k8.groupby(["fold", "city"])["m0_cpc_inter"].mean().mean()
    m1_mean = k_raw_k8.groupby(["fold", "city"])["m1_cpc_inter"].mean().mean()
    print(f"\nM0 CPC mean: {m0_mean:.5f} (expected: 0.71281)")
    print(f"M1 CPC mean: {m1_mean:.5f} (expected: 0.71635)")

    match_m0 = abs(m0_mean - 0.71281) < 1e-3
    match_m1 = abs(m1_mean - 0.71635) < 1e-3
    total_checks += 2
    if match_m0: total_pass += 1
    else:
        total_fail += 1
        failures.append(f"M0 mean mismatch: {m0_mean}")
    if match_m1: total_pass += 1
    else:
        total_fail += 1
        failures.append(f"M1 mean mismatch: {m1_mean}")

    # Win rate
    win_count = (city_means > 0).sum()
    print(f"\nWin rate: {win_count}/50 (expected: 45/50)")
    match_win = (win_count == 45)
    total_checks += 1
    if match_win: total_pass += 1
    else:
        total_fail += 1
        failures.append(f"Win rate mismatch: {win_count}/50 vs 45/50")

    # If placebo exists, check population means
    if has_placebo:
        print("\nUnified placebo population means:")
        for col, label, expected in [
            ("d_cpc_target", "Oracle Target", 0.003539),
            ("d_cpc_matched", "Dose-Matched Donors", -0.000091),
            ("d_cpc_raw_train", "Raw Training Donors", -0.035148),
            ("d_cpc_train_mean", "Raw Train-Mean", -0.017735),
            ("d_cpc_perm", "Permuted", -0.006964),
        ]:
            if col in placebo_df.columns:
                val = placebo_df[col].mean()
                match = abs(val - expected) < 1e-4
                total_checks += 1
                if match: total_pass += 1
                else:
                    total_fail += 1
                    failures.append(f"Placebo {label} mismatch: {val:.6f} vs {expected:.6f}")
                print(f"  {label}: {val:.6f} (expected: {expected:.6f}) -> {'PASS' if match else 'FAIL'}")

    # ── Final Report ──
    print("\n" + "=" * 80)
    print("FINAL AUDIT REPORT")
    print("=" * 80)
    print(f"Total checks: {total_checks}")
    print(f"PASS: {total_pass}")
    print(f"FAIL: {total_fail}")
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  [FAIL] {f}")
    else:
        print("\n[SUCCESS] ALL CHECKS PASSED - No obsolete data detected.")
    print("=" * 80)

    return total_fail


if __name__ == "__main__":
    n_fail = main()
    sys.exit(0 if n_fail == 0 else 1)
