"""
Unit and Contract Tests for Noise Robustness Module (run_noise_robustness.py).
Executable directly with standard python (no pytest dependency required).
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.experiment.run_noise_robustness import (
    holm_correction,
    get_stable_seed,
    generate_nested_noisy_yd,
    fold_stratified_bootstrap,
    fast_cal_metrics,
    generate_summary,
)


class TestNoiseRobustness(unittest.TestCase):

    def test_get_stable_seed_determinism(self):
        """Verify sha256 stable seed is strictly deterministic."""
        s1 = get_stable_seed(20260822, 1, "Atlanta", 42)
        s2 = get_stable_seed(20260822, 1, "Atlanta", 42)
        s3 = get_stable_seed(20260822, 1, "Atlanta", 43)
        self.assertEqual(s1, s2)
        self.assertNotEqual(s1, s3)
        self.assertTrue(0 <= s1 < 2**32)

    def test_generate_nested_noisy_yd(self):
        """Verify nested noisy Y_D preserves active simplex and exact TV distances."""
        p_active = np.array([0.30, 0.25, 0.20, 0.15, 0.07, 0.03])
        p_active = p_active / p_active.sum()
        epsilons = [0.0, 0.05, 0.10, 0.20]
        seed = 42

        noisy_dict = generate_nested_noisy_yd(p_active, epsilons, seed)
        
        self.assertEqual(set(noisy_dict.keys()), set(epsilons))
        
        # Oracle eps=0.0 exact match
        np.testing.assert_allclose(noisy_dict[0.0], p_active, atol=1e-10)
        
        # Check properties for each epsilon
        for eps in epsilons:
            p_eps = noisy_dict[eps]
            self.assertTrue(np.all(p_eps >= 0.0), f"Negative probability found for eps={eps}")
            self.assertAlmostEqual(float(np.sum(p_eps)), 1.0, places=7, msg=f"Probabilities do not sum to 1 for eps={eps}")
            
            achieved_tv = 0.5 * float(np.sum(np.abs(p_eps - p_active)))
            self.assertAlmostEqual(achieved_tv, eps, places=6, msg=f"TV distance mismatch for eps={eps}")

    def test_holm_correction(self):
        """Verify Holm-Bonferroni step-down multiple testing correction."""
        raw_p = [0.01, 0.04, 0.03]
        adj_p = holm_correction(raw_p)
        
        # Sorted p: 0.01 (x3=0.03), 0.03 (x2=0.06), 0.04 (x1=0.04 -> max 0.06)
        expected = np.array([0.03, 0.06, 0.06])
        np.testing.assert_allclose(adj_p, expected, atol=1e-6)
        self.assertTrue(np.all(adj_p <= 1.0))

    def test_fast_cal_metrics(self):
        """Verify fast vectorized M1 calibration mathematics and mass preservation."""
        N = 10000
        K = 8
        np.random.seed(123)
        t_true = np.random.exponential(10.0, size=N)
        t0 = np.random.exponential(9.0, size=N)
        dist = np.random.uniform(0.1, 100.0, size=N)
        bin_edges = np.linspace(0.1, 100.0, K+1)
        bin_idx = np.clip(np.digitize(dist, bin_edges[1:-1], right=True), 0, K - 1).astype(np.int32)
        
        N_hat = float(t0.sum())
        counts = np.bincount(bin_idx, weights=t0, minlength=K)
        Y_hat = counts / N_hat
        pair_counts = np.bincount(bin_idx, minlength=K)
        active = pair_counts > 0
        
        yd_target = np.random.dirichlet(np.ones(K))
        cpc_m0 = 0.60
        
        inv_N = 1.0 / N
        sum_denom = float(t_true.sum()) + N_hat
        inv_sum_denom = 2.0 / sum_denom if sum_denom > 0 else 0.0
        
        t_cal_buf = np.empty(N, dtype=np.float64)
        diff_buf = np.empty(N, dtype=np.float64)
        
        cpc, mae, rmse, spr, tv_ach, js_div, stats = fast_cal_metrics(
            yd_target, 0.0, True, N_hat, K, active, Y_hat, t0, bin_idx, t_true, cpc_m0, yd_target,
            inv_sum_denom, inv_N, t_cal_buf, diff_buf
        )
        
        # Check calibrated mass is preserved exactly
        self.assertAlmostEqual(float(t_cal_buf.sum()), N_hat, places=6)
        self.assertTrue(0.0 <= cpc <= 1.0)
        self.assertTrue(mae >= 0.0)
        self.assertTrue(rmse >= 0.0)
        self.assertAlmostEqual(tv_ach, 0.0, places=8)

    def test_fold_stratified_bootstrap(self):
        """Verify 2D vectorized fold-stratified bootstrap confidence intervals."""
        rows = []
        for f in [2, 3, 4, 5]:
            for c in range(10):
                rows.append({
                    "fold": f,
                    "target_city": f"city_{f}_{c}",
                    "epsilon": 0.05,
                    "delta_cpc_mean": 0.04 + 0.002 * c
                })
        df = pd.DataFrame(rows)
        ci_lo, ci_hi = fold_stratified_bootstrap(df, "delta_cpc_mean", 0.05, [2, 3, 4, 5], n_boot=1000)
        self.assertTrue(0.035 <= ci_lo <= ci_hi <= 0.065)

    def test_generate_summary_artifacts(self):
        """Verify summary reports and visual figures export cleanly."""
        output_dir = "results/test_noise_summary"
        os.makedirs(output_dir, exist_ok=True)
        epsilons = [0.0, 0.05, 0.10, 0.20]
        nonzero_eps = [0.05, 0.10, 0.20]
        
        rows = []
        for f in [2, 3, 4, 5]:
            for c in range(10):
                city = f"City_{f}_{c}"
                for eps in epsilons:
                    rows.append({
                        "fold": f,
                        "target_city": city,
                        "epsilon": eps,
                        "delta_cpc_mean": 0.05 - 0.15 * eps,
                        "degradation_mean": 0.15 * eps,
                        "prob_positive": 1.0 if eps < 0.1 else 0.5,
                        "cpc_m1_inter": 0.65 - 0.1 * eps,
                        "w_max": 2.0,
                        "w_gt_2": 0.1
                    })
        df = pd.DataFrame(rows)
        generate_summary(df, output_dir, epsilons, nonzero_eps)
        
        self.assertTrue((Path(output_dir) / "noise_summary.json").exists())
        self.assertTrue((Path(output_dir) / "noise_summary.md").exists())
        self.assertTrue((Path(output_dir) / "fig_noise_dose_response.png").exists())
        self.assertTrue((Path(output_dir) / "fig_noise_harm_rate.png").exists())
        self.assertTrue((Path(output_dir) / "fig_noise_by_city.png").exists())


if __name__ == "__main__":
    unittest.main()
