"""
Unit and Contract Tests for Finite-Sample Y_D Robustness and Audit Module.
"""

import math
import unittest
from pathlib import Path

from audit import AuditFailure, _finite, _require, audit_results
from src.experiment.run_finite_sample_yd_robustness import _stable_seed, _holm, SAMPLE_SIZES, MODEL_SEEDS


class TestFiniteSampleYDRobustness(unittest.TestCase):

    def test_stable_seed_determinism(self):
        """Verify sha256 stable seed generation is strictly deterministic and non-negative."""
        s1 = _stable_seed(1, "Austin", 0, 0)
        s2 = _stable_seed(1, "Austin", 0, 0)
        s3 = _stable_seed(1, "Austin", 1, 0)
        self.assertEqual(s1, s2)
        self.assertNotEqual(s1, s3)
        self.assertTrue(0 <= s1 < 2**32)

    def test_holm_correction(self):
        """Verify step-down Holm correction bounds and monotonic adjustment."""
        raw_p = [0.001, 0.02, 0.04, 0.03]
        adj_p = _holm(raw_p)
        self.assertEqual(len(adj_p), 4)
        for p in adj_p:
            self.assertTrue(0.0 <= p <= 1.0)
        self.assertTrue(adj_p[0] <= adj_p[1])

    def test_finite_validation_guards(self):
        """Verify _finite rejects non-finite values and boolean types."""
        self.assertEqual(_finite(0.5, "test"), 0.5)
        self.assertEqual(_finite("0.25", "test"), 0.25)
        with self.assertRaises(AuditFailure):
            _finite(True, "test")
        with self.assertRaises(AuditFailure):
            _finite(float("nan"), "test")
        with self.assertRaises(AuditFailure):
            _finite(float("inf"), "test")
        with self.assertRaises(AuditFailure):
            _finite("invalid_str", "test")

    def test_audit_results_on_frozen_artifacts(self):
        """Verify full audit passes on the frozen results artifact directory."""
        results_dir = Path("results/finite_sample_yd_robustness_v1")
        if not results_dir.exists():
            self.skipTest("results/finite_sample_yd_robustness_v1 does not exist")

        report = audit_results(results_dir)
        self.assertEqual(report["status"], "PASS")
        for gate_name, gate_status in report["gates"].items():
            self.assertEqual(gate_status, "PASS", f"Gate {gate_name} did not pass")
        self.assertEqual(report["n_cities"], 50)
        self.assertEqual(report["sample_sizes"], SAMPLE_SIZES)
        self.assertTrue(report["clean_gain"] > 0.0)
        self.assertIn("sampling_provenance", report["limitations"])


if __name__ == "__main__":
    unittest.main()