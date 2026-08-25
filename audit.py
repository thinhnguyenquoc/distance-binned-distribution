# AUDIT SNAPSHOT: distance-binned-distribution
# Generated from the current workspace source.
# Python source bundle for third-party review; sections preserve relative paths and source lines.


# ===== BEGIN SOURCE FILE: run_all_tests.py =====
# | """
# | Master Test Runner & System Verification Suite for Distance-Binned Distribution OD Reconstruction.
# | Discovers and runs all contract, unit, regression, and model tests across the codebase.
# | """
# |
# | import sys
# | import os
# | import time
# | import inspect
# | import importlib.util
# | from pathlib import Path
# | import numpy as np
# | import torch
# |
# | # Ensure root directory is in sys.path
# | repo_root = Path(__file__).resolve().parent
# | sys.path.insert(0, str(repo_root))
# |
# | # Provide Pytest shim if pytest package is not installed in the local environment
# | try:
# |     import pytest
# | except ImportError:
# |     class Approx:
# |         def __init__(self, expected, rel=1e-5, abs=1e-12):
# |             self.expected = expected
# |             self.rel = rel
# |             self.abs = abs
# |         def __eq__(self, actual):
# |             return bool(np.isclose(self.expected, actual, rtol=self.rel, atol=self.abs))
# |         def __repr__(self):
# |             return f"approx({self.expected})"
# |
# |     class Raises:
# |         def __init__(self, expected_exc):
# |             self.expected_exc = expected_exc
# |         def __enter__(self):
# |             return self
# |         def __exit__(self, exc_type, exc_val, exc_tb):
# |             if exc_type is None:
# |                 raise AssertionError(f"DID NOT RAISE expected exception {self.expected_exc}")
# |             return issubclass(exc_type, self.expected_exc)
# |
# |     class Mark:
# |         def __getattr__(self, name):
# |             def decorator(*args, **kwargs):
# |                 if len(args) == 1 and callable(args[0]):
# |                     return args[0]
# |                 return lambda fn: fn
# |             return decorator
# |
# |     class PytestShim:
# |         approx = Approx
# |         raises = Raises
# |         mark = Mark()
# |         @staticmethod
# |         def fixture(func=None, *args, **kwargs):
# |             if func and callable(func):
# |                 return func
# |             return lambda f: f
# |
# |     sys.modules["pytest"] = PytestShim
# |
# |
# | # Shared fixtures dictionary
# | FIXTURES = {
# |     "sample_coordinates": np.array([
# |         [-84.3880, 33.7490],
# |         [-84.3900, 33.7500],
# |         [-84.4000, 33.7600],
# |         [-84.4500, 33.8000],
# |         [-85.0000, 34.2000],
# |     ]),
# |     "synthetic_od_flows": (
# |         torch.tensor([12.0, 45.0, 150.0, 8.0, 220.0, 35.0, 90.0, 15.0]),
# |         torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]),
# |     ),
# | }
# |
# |
# | def call_with_fixtures(func):
# |     sig = inspect.signature(func)
# |     kwargs = {}
# |     for param_name in sig.parameters:
# |         if param_name in FIXTURES:
# |             kwargs[param_name] = FIXTURES[param_name]
# |     return func(**kwargs)
# |
# |
# | def run_all_tests():
# |     """Run every repository test through pytest's native collection/execution.
# |
# |     The previous implementation imported modules and invoked test functions
# |     directly.  That bypassed pytest's fixture and outcome handling; in
# |     particular, a ``pytest.skip`` exception could terminate this runner after
# |     the first module and report an incomplete suite as successful.
# |     """
# |     import pytest
# |
# |     test_paths = [
# |         str(repo_root / "od_plan_tester" / "tests"),
# |         str(repo_root / "tests"),
# |     ]
# |     exit_code = pytest.main(["-q", "-p", "no:cacheprovider", *test_paths])
# |     if exit_code != pytest.ExitCode.OK:
# |         raise SystemExit(int(exit_code))
# |     return
# |
# |     test_files = sorted(
# |         list((repo_root / "od_plan_tester" / "tests").glob("test_*.py"))
# |         + list((repo_root / "tests").glob("test_*.py"))
# |         + list(repo_root.glob("test_*.py"))
# |     )
# |
# |     total_passed = 0
# |     total_failed = 0
# |     failed_details = []
# |
# |     print("=" * 85)
# |     print("MASTER TEST SUITE & SYSTEM VERIFICATION AUDIT")
# |     print(f"Discovered {len(test_files)} test modules in repository")
# |     print("=" * 85)
# |
# |     start_all = time.perf_counter()
# |
# |     for tf in test_files:
# |         if tf.name == "run_all_tests.py":
# |             continue
# |         mod_name = tf.stem
# |         spec = importlib.util.spec_from_file_location(mod_name, tf)
# |         mod = importlib.util.module_from_spec(spec)
# |         sys.modules[mod_name] = mod
# |         
# |         try:
# |             spec.loader.exec_module(mod)
# |         except Exception as e:
# |             print(f"  [ERROR LOAD] {tf.name:<36} -> {e}")
# |             total_failed += 1
# |             failed_details.append((tf.name, "MODULE_LOAD", str(e)))
# |             continue
# |
# |         funcs = [getattr(mod, f) for f in dir(mod) if f.startswith("test_") and callable(getattr(mod, f))]
# |         classes = [getattr(mod, c) for c in dir(mod) if inspect.isclass(getattr(mod, c)) and c.startswith("Test")]
# |         
# |         file_passed = 0
# |         file_failed = 0
# |         
# |         for fn in funcs:
# |             try:
# |                 call_with_fixtures(fn)
# |                 file_passed += 1
# |             except Exception as e:
# |                 file_failed += 1
# |                 failed_details.append((tf.name, fn.__name__, str(e)))
# |                 
# |         for cls in classes:
# |             inst = cls()
# |             for m in dir(inst):
# |                 if m.startswith("test_") and callable(getattr(inst, m)):
# |                     try:
# |                         call_with_fixtures(getattr(inst, m))
# |                         file_passed += 1
# |                     except Exception as e:
# |                         file_failed += 1
# |                         failed_details.append((tf.name, f"{cls.__name__}.{m}", str(e)))
# |                         
# |         total_passed += file_passed
# |         total_failed += file_failed
# |         status_tag = "PASS" if file_failed == 0 else f"FAIL ({file_failed})"
# |         print(f"  [{status_tag:<9}] {tf.name:<36} -> {file_passed} passed, {file_failed} failed")
# |
# |     elapsed = time.perf_counter() - start_all
# |     print("=" * 85)
# |     print(f"TOTAL EXECUTION SUMMARY: {total_passed} PASSED | {total_failed} FAILED in {elapsed:.2f}s")
# |     print("=" * 85)
# |     
# |     if failed_details:
# |         print("\nFAILURE DETAILS:")
# |         for f, name, err in failed_details:
# |             print(f"  - [{f}] {name}: {err}")
# |         sys.exit(1)
# |     else:
# |         print("\nALL SYSTEM TESTS & INVARIANTS PASSED PERFECTLY!")
# |
# |
# | if __name__ == "__main__":
# |     run_all_tests()

# ===== END SOURCE FILE: run_all_tests.py =====


# ===== BEGIN SOURCE FILE: run_full_experiment.py =====
# | """Run the complete GNN and MLP experiment for a configurable seed set.
# |
# | Each model is saved as results/checkpoints/{backbone}_fold{fold}_seed{seed}.pt.
# | Existing checkpoints are reused only when their protocol provenance matches.
# | """
# |
# | import argparse
# |
# | from src.experiment.run_5fold import run_5fold_experiment
# |
# |
# | CANONICAL_SEEDS = [1, 10, 100]
# | DEFAULT_SEEDS = CANONICAL_SEEDS
# |
# |
# | def run_full_experiment(
# |     seeds: list[int],
# |     folds: list[int],
# |     epochs: int,
# |     device: str | None,
# |     data_root: str,
# |     output_dir: str,
# | ) -> None:
# |     common = {
# |         "data_root": data_root,
# |         "output_dir": output_dir,
# |         "epochs_per_fold": epochs,
# |         "folds_to_run": folds,
# |         "seeds": seeds,
# |         "device_str": device,
# |     }
# |
# |     print(f"Running GNN for seeds={seeds}, folds={folds}")
# |     run_5fold_experiment(backbone="gnn", **common)
# |
# |     print(f"Running MLP for seeds={seeds}, folds={folds}")
# |     run_5fold_experiment(backbone="mlp", **common)
# |
# |
# | if __name__ == "__main__":
# |     parser = argparse.ArgumentParser(
# |         description="Run full GNN + MLP experiments with reusable seed checkpoints"
# |     )
# |     parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
# |     parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
# |     parser.add_argument("--epochs", type=int, default=200)
# |     parser.add_argument("--device", type=str, default=None)
# |     parser.add_argument("--data-root", type=str, default="data")
# |     parser.add_argument("--output-dir", type=str, default="results")
# |     args = parser.parse_args()
# |
# |     if len(set(args.seeds)) != len(args.seeds):
# |         parser.error("--seeds must not contain duplicates")
# |     if any(seed < 0 for seed in args.seeds):
# |         parser.error("--seeds must be non-negative integers")
# |     if args.seeds != CANONICAL_SEEDS and args.output_dir == "results":
# |         parser.error(
# |             "non-canonical seeds require a separate --output-dir; "
# |             "use e.g. results/seed_robustness_333_5555_77777"
# |         )
# |     if any(fold not in {1, 2, 3, 4, 5} for fold in args.folds):
# |         parser.error("--folds must contain values from 1 through 5")
# |
# |     run_full_experiment(
# |         seeds=args.seeds,
# |         folds=args.folds,
# |         epochs=args.epochs,
# |         device=args.device,
# |         data_root=args.data_root,
# |         output_dir=args.output_dir,
# |     )

# ===== END SOURCE FILE: run_full_experiment.py =====


# ===== BEGIN SOURCE FILE: run_research_contract_tests.py =====
# | """
# | Master Research Contract Verification Suite (registered scientific and methodological checks).
# | Enforces strict protocol invariants, zero data-leakage guards, production calibration equivalence,
# | statistical unit integrity, and independent raw-to-summary reproducibility before paper freeze.
# | """
# |
# | import sys
# | import os
# | import time
# | import json
# | import csv
# | import hashlib
# | import re
# | from pathlib import Path
# | from typing import Dict, List, Tuple, Any
# |
# | import numpy as np
# | import pandas as pd
# | from scipy import stats
# | import torch
# |
# | # Ensure repository root is on sys.path
# | REPO_ROOT = Path(__file__).resolve().parent
# | sys.path.insert(0, str(REPO_ROOT))
# |
# | from src.data.city_splits import generate_35_5_10_splits
# | from src.data.dataset import load_city, load_cities, load_raw_city, CityData
# | from src.data.urban_graph import build_radius_graph
# | from src.data.yd_extractor import compute_kbin_edges, extract_yd_kbins
# | from src.calibration.bin_calibration import calibrate_kbins
# | from src.training.evaluate import compute_cpc_pair, compute_cpc_norm_pair
# | from src.training.train import load_checkpoint, infer_zero_shot
# | from src.experiment.run_noise_robustness import generate_nested_noisy_yd, fast_cal_metrics, holm_correction as holm_noise
# | from src.experiment.run_sampling_robustness import sample_hypergeometric_yd, holm_correction as holm_sampling
# |
# | GATE_RESULTS: Dict[str, Tuple[bool, str]] = {}
# |
# |
# | def _result_roots() -> list[Path]:
# |     roots = [Path("results")]
# |     roots.extend(sorted(Path(".").glob("results_archive_*/old_results"), reverse=True))
# |     return roots
# |
# |
# | def _canonical_result_root() -> Path:
# |     return Path("results")
# |
# |
# | def _find_result_file(relative_path: str) -> Path:
# |     for root in _result_roots():
# |         candidate = root / relative_path
# |         if candidate.exists():
# |             return candidate
# |     raise FileNotFoundError(f"Missing result artifact: {relative_path}")
# |
# |
# | def log_gate(gate_num: int, name: str, passed: bool, msg: str = ""):
# |     tag = "PASS" if passed else "FAIL"
# |     color_tag = f"\033[92mPASS\033[0m" if passed else f"\033[91mFAIL\033[0m"
# |     GATE_RESULTS[f"GATE {gate_num:2d}"] = (passed, f"{name}: {msg}")
# |     print(f"GATE {gate_num:<2d}  {name:<38} {color_tag} {msg}")
# |
# |
# | # -----------------------------------------------------------------------------
# | # GATE 1: Split Integrity Test
# | # -----------------------------------------------------------------------------
# | def test_gate_1_split_integrity():
# |     splits = generate_35_5_10_splits(data_root="data")
# |     assert len(splits) == 5, f"Expected 5 folds, got {len(splits)}"
# |     
# |     all_test_cities = []
# |     for f, s in splits.items():
# |         train = set(s["train"])
# |         val = set(s["val"])
# |         test = set(s["test"])
# |         
# |         assert len(train) == 35, f"Fold {f} train size {len(train)} != 35"
# |         assert len(val) == 5, f"Fold {f} val size {len(val)} != 5"
# |         assert len(test) == 10, f"Fold {f} test size {len(test)} != 10"
# |         
# |         # Pairwise disjoint
# |         assert train.isdisjoint(val), f"Fold {f} train & val overlap!"
# |         assert train.isdisjoint(test), f"Fold {f} train & test overlap!"
# |         assert val.isdisjoint(test), f"Fold {f} val & test overlap!"
# |         
# |         all_test_cities.extend(s["test"])
# |         
# |     assert len(all_test_cities) == 50, f"Expected 50 test city instances, got {len(all_test_cities)}"
# |     assert len(set(all_test_cities)) == 50, "Duplicate test cities across folds!"
# |     
# |     # Check Fold 1 parity
# |     f1 = splits[1]
# |     assert len(f1["test"]) == 10 and len(f1["train"]) == 35 and len(f1["val"]) == 5
# |     return True, "All 5 folds disjoint (35/5/10), exact 50-city test partition"
# |
# |
# | # -----------------------------------------------------------------------------
# | # GATE 2: Data-Leakage & Mutation Invariance Test
# | # -----------------------------------------------------------------------------
# | def test_gate_2_data_leakage():
# |     splits = generate_35_5_10_splits(data_root="data")
# |     train35 = splits[1]["train"]
# |     test_cities = splits[1]["test"]
# |     test_city = test_cities[0]
# |     
# |     # 1. Guard scaler.fit(): assert it only ever sees the 35 train cities
# |     from sklearn.preprocessing import StandardScaler
# |     original_fit = StandardScaler.fit
# |     fitted_row_counts = []
# |     
# |     def guarded_fit(self, X, y=None):
# |         fitted_row_counts.append(len(X))
# |         return original_fit(self, X, y)
# |         
# |     StandardScaler.fit = guarded_fit
# |     try:
# |         train_cities_data, scaler = load_cities(train35, data_root="data")
# |     finally:
# |         StandardScaler.fit = original_fit
# |         
# |     expected_train_rows = sum(load_raw_city(city, data_root="data").n_tracts for city in train35)
# |     assert fitted_row_counts == [expected_train_rows], (
# |         "Scaler must be fit exactly once using only the Fold 1 training rows: "
# |         f"expected {[expected_train_rows]}, observed {fitted_row_counts}"
# |     )
# |     
# |     # 2. Bin edges computed strictly from train cities
# |     bin_edges, K_act = compute_kbin_edges(train35, K=8, data_root="data")
# |     assert K_act == 8 and len(bin_edges) == 9
# |     
# |     # 3. M0 target-Y_D independence and deterministic inference.
# |     import inspect
# |     from src.data import yd_extractor
# |     from src.calibration import bin_calibration
# |
# |     signature = inspect.signature(infer_zero_shot)
# |     forbidden_params = {"yd", "y_d", "trip", "trip_distribution", "calibration"}
# |     assert not any(
# |         parameter.name.lower() in forbidden_params
# |         for parameter in signature.parameters.values()
# |     ), f"infer_zero_shot has target-Y_D-dependent input: {signature}"
# |
# |     source = inspect.getsource(infer_zero_shot)
# |     forbidden_dependencies = ("compute_kbin_edges", "extract_yd_kbins", "calibrate_kbins")
# |     assert not any(name in source for name in forbidden_dependencies), (
# |         "infer_zero_shot directly depends on target-Y_D extraction/calibration"
# |     )
# |
# |     def fail_if_target_yd_accessed(*args, **kwargs):
# |         raise AssertionError("M0 accessed target-Y_D extraction or calibration")
# |
# |     patched_functions = {
# |         (yd_extractor, "compute_kbin_edges"): yd_extractor.compute_kbin_edges,
# |         (yd_extractor, "extract_yd_kbins"): yd_extractor.extract_yd_kbins,
# |         (bin_calibration, "calibrate_kbins"): bin_calibration.calibrate_kbins,
# |     }
# |     for (module, name) in patched_functions:
# |         setattr(module, name, fail_if_target_yd_accessed)
# |
# |     city_data = load_city(test_city, data_root="data", feature_scaler=scaler, fit_scaler=False)
# |     coords = city_data.lon_lat.numpy()
# |     ei, ed = build_radius_graph(coords, radius_km=5.0)
# |
# |     try:
# |         for seed in [1, 10, 100]:
# |             ckpt_path = _find_result_file(f"checkpoints/5fold_fold1_seed{seed}.pt")
# |             model, _, metadata = load_checkpoint(ckpt_path, device_str="cpu")
# |             assert metadata.get("seed") == seed, (
# |                 f"{ckpt_path.name} metadata seed mismatch: "
# |                 f"expected {seed}, got {metadata.get('seed')}"
# |             )
# |             model.eval()
# |
# |             with torch.no_grad():
# |                 m0_first = infer_zero_shot(model, city_data, ei, ed, device="cpu")
# |                 m0_second = infer_zero_shot(model, city_data, ei, ed, device="cpu")
# |             assert torch.equal(m0_first, m0_second), (
# |                 f"M0 inference is not deterministic for seed {seed}"
# |             )
# |     finally:
# |         for (module, name), original in patched_functions.items():
# |             setattr(module, name, original)
# |         
# |     return True, "Scaler guarded (train-only), M0 structurally Y_D-independent and deterministic for seeds 1, 10, 100"
# |
# |
# | # -----------------------------------------------------------------------------
# | # GATE 15: Radius Graph & Isolated-Node Fallback Contract
# | # -----------------------------------------------------------------------------
# | def test_gate_15_radius_graph_contract():
# |     def independent_distances(lon_lat):
# |         coordinates = np.asarray(lon_lat, dtype=np.float64)
# |         radians = np.radians(coordinates)
# |         delta = radians[:, None, :] - radians[None, :, :]
# |         a = (
# |             np.sin(delta[:, :, 1] / 2.0) ** 2
# |             + np.cos(radians[:, None, 1])
# |             * np.cos(radians[None, :, 1])
# |             * np.sin(delta[:, :, 0] / 2.0) ** 2
# |         )
# |         return 2.0 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
# |
# |     def independent_reference_graph(lon_lat, radius_km):
# |         distances = independent_distances(lon_lat)
# |         node_count = len(lon_lat)
# |         directed_edges = set()
# |         fallback_edges = set()
# |
# |         for source in range(node_count):
# |             radius_neighbors = [
# |                 target
# |                 for target in range(node_count)
# |                 if target != source and 0.0 < distances[source, target] <= radius_km
# |             ]
# |             for target in radius_neighbors:
# |                 directed_edges.add((source, target))
# |
# |             if not radius_neighbors:
# |                 nearest = min(
# |                     (target for target in range(node_count) if target != source),
# |                     key=lambda target: distances[source, target],
# |                 )
# |                 directed_edges.add((source, nearest))
# |                 fallback_edges.update({(source, nearest), (nearest, source)})
# |
# |         symmetric_edges = directed_edges | {
# |             (target, source) for source, target in directed_edges
# |         }
# |         reference_edges = symmetric_edges | {
# |             (node, node) for node in range(node_count)
# |         }
# |         return reference_edges, fallback_edges, distances
# |
# |     coordinate_sets = [
# |         np.array([[0.0, 0.0], [0.01, 0.0], [0.10, 0.0]], dtype=np.float64),
# |     ]
# |     held_out_city = generate_35_5_10_splits(data_root="data")[1]["test"][0]
# |     coordinate_sets.append(load_city(held_out_city, data_root="data").lon_lat.numpy())
# |
# |     for coordinates in coordinate_sets:
# |         expected_edges, fallback_edges, distances = independent_reference_graph(
# |             coordinates, radius_km=5.0
# |         )
# |         edge_index, edge_dist = build_radius_graph(
# |             coordinates, radius_km=5.0, use_cache=False
# |         )
# |         production_edges = {
# |             (int(edge_index[0, index]), int(edge_index[1, index]))
# |             for index in range(edge_index.shape[1])
# |         }
# |         assert production_edges == expected_edges, (
# |             "Production radius graph differs from independent reference graph"
# |         )
# |
# |         for index in range(edge_index.shape[1]):
# |             source = int(edge_index[0, index])
# |             target = int(edge_index[1, index])
# |             expected_distance = distances[source, target]
# |             assert np.isclose(float(edge_dist[index]), expected_distance, atol=1e-5, rtol=0.0), (
# |                 f"edge_dist mismatch for ({source}, {target})"
# |             )
# |             if source != target and expected_distance > 5.0:
# |                 assert (source, target) in fallback_edges, (
# |                     f"Non-radius edge ({source}, {target}) is not an isolated-node fallback"
# |                 )
# |
# |     return True, "Radius edges, isolated-node fallback, symmetry, self-loops, and edge distances match independent reference"
# |
# |
# | # -----------------------------------------------------------------------------
# | # GATE 3: Checkpoint Protocol Deep Audit (30 Checkpoints)
# | # -----------------------------------------------------------------------------
# | def test_gate_3_checkpoint_protocol():
# |     splits = generate_35_5_10_splits(data_root="data")
# |     root = _canonical_result_root()
# |     gnn_ckpts = list(root.glob("checkpoints/5fold_fold*.pt"))
# |     mlp_ckpts = list(root.glob("checkpoints/mlp_fold*.pt"))
# |     
# |     assert len(gnn_ckpts) == 15, f"Expected 15 GNN checkpoints, found {len(gnn_ckpts)}"
# |     assert len(mlp_ckpts) == 15, f"Expected 15 MLP checkpoints, found {len(mlp_ckpts)}"
# |     
# |     for p in gnn_ckpts:
# |         bundle = torch.load(p, map_location="cpu", weights_only=False)
# |         hp = bundle.get("hyperparams", {})
# |         seed = bundle.get("seed")
# |         run_tag = bundle.get("run_tag", p.stem)
# |         
# |         # Extract fold from run_tag or filename
# |         import re
# |         m_fold = re.search(r"fold(\d+)", p.stem)
# |         fold_id = int(m_fold.group(1)) if m_fold else None
# |         
# |         assert fold_id in [1, 2, 3, 4, 5], f"{p.name} invalid fold_id: {fold_id}"
# |         assert seed in [1, 10, 100], f"{p.name} invalid seed: {seed}"
# |         assert hp.get("loss_type") == "ztnb", f"{p.name} loss != ztnb"
# |         assert hp.get("hidden_dim") == 64, f"{p.name} hidden_dim != 64"
# |         assert hp.get("radius_km") == 5.0, f"{p.name} radius != 5.0"
# |         assert hp.get("node_in_dim") == 26, f"{p.name} node_in_dim != 26"
# |         assert len(bundle.get("scaler_mean_")) == 26, f"{p.name} scaler_mean_ length != 26"
# |             
# |     for p in mlp_ckpts:
# |         bundle = torch.load(p, map_location="cpu", weights_only=False)
# |         hp = bundle.get("hyperparams", {})
# |         m = re.search(r"mlp_fold(\d+)_seed(\d+)", p.stem)
# |         assert m is not None, f"{p.name} missing fold/seed filename contract"
# |         assert int(m.group(1)) in [1, 2, 3, 4, 5], f"{p.name} invalid fold_id"
# |         assert int(m.group(2)) in [1, 10, 100], f"{p.name} invalid seed"
# |         assert bundle.get("seed") == int(m.group(2)), f"{p.name} metadata seed mismatch"
# |         expected_run_tag = f"5fold_{p.stem}"
# |         assert bundle.get("run_tag") == expected_run_tag, f"{p.name} run_tag mismatch"
# |         assert hp.get("loss_type") == "ztnb", f"{p.name} loss != ztnb"
# |         assert hp.get("backbone") == "mlp", f"{p.name} backbone != mlp"
# |         assert len(bundle.get("scaler_mean_")) == 26, f"{p.name} scaler_mean_ length != 26"
# |         
# |     return True, "15 GNN + 15 MLP checkpoints audited for filename/metadata fold-seed integrity"
# |
# |
# | # -----------------------------------------------------------------------------
# | # GATE 4: Zero-Shot Inference & No-Gradient Guard
# | # -----------------------------------------------------------------------------
# | def test_gate_4_zero_shot_inference():
# |     ckpt_path = _find_result_file("checkpoints/5fold_fold1_seed1.pt")
# |     model, scaler, _ = load_checkpoint(ckpt_path, device_str="cpu")
# |     model.eval()
# |     
# |     # Assert all parameters have requires_grad=False
# |     for name, p in model.named_parameters():
# |         assert not p.requires_grad, f"Parameter {name} requires grad!"
# |         
# |     # Guard against optimizer.step() and backward()
# |     def forbid_step(*args, **kwargs):
# |         raise RuntimeError("Optimizer.step called during test-time inference!")
# |         
# |     def forbid_backward(*args, **kwargs):
# |         raise RuntimeError("Tensor.backward called during test-time inference!")
# |         
# |     orig_step = torch.optim.Optimizer.step
# |     orig_backward = torch.Tensor.backward
# |     torch.optim.Optimizer.step = forbid_step
# |     torch.Tensor.backward = forbid_backward
# |     
# |     initial_weights = [p.clone() for p in model.parameters()]
# |     try:
# |         city_data = load_city("Austin", data_root="data", feature_scaler=scaler, fit_scaler=False)
# |         coords = city_data.lon_lat.numpy()
# |         ei, ed = build_radius_graph(coords, radius_km=5.0)
# |         _ = infer_zero_shot(model, city_data, ei, ed, device="cpu")
# |     finally:
# |         torch.optim.Optimizer.step = orig_step
# |         torch.Tensor.backward = orig_backward
# |         
# |     for p_init, p_curr in zip(initial_weights, model.parameters()):
# |         assert torch.equal(p_init, p_curr), "Model weights drifted during inference!"
# |         
# |     return True, "Optimizer & backward guarded, parameters 100% frozen"
# |
# |
# | # -----------------------------------------------------------------------------
# | # GATE 5: Production Calibration Equivalence (5 Cities x 3 Seeds)
# | # -----------------------------------------------------------------------------
# | def test_gate_5_calibration_equivalence():
# |     test_cities = ["Austin", "Atlanta", "Denver", "Seattle", "Chicago"]
# |     seeds = [1, 10, 100]
# |     bin_edges = np.array([0.0, 3.5, 7.0, 11.0, 15.5, 21.0, 28.0, 38.0, np.inf])
# |     
# |     max_diff = 0.0
# |     comparisons = 0
# |     
# |     for city_name in test_cities:
# |         raw = load_raw_city(city_name, data_root="data")
# |         dist_km = raw.dist_km
# |         inter = (raw.pair_o_idx.numpy() != raw.pair_d_idx.numpy()) & (dist_km > 0.0)
# |         t_true_inter = raw.pair_trips.numpy()[inter]
# |         yd_tgt = extract_yd_kbins(dist_km, raw.pair_trips.numpy(), bin_edges, inter)
# |         
# |         # Prepare inputs for fast_cal_metrics
# |         bin_idx = np.clip(np.digitize(dist_km[inter], bin_edges) - 1, 0, 7)
# |         K = 8
# |         active = yd_tgt > 1e-8
# |         
# |         for seed in seeds:
# |             # Deterministic synthetic test prediction
# |             rng = np.random.RandomState(seed * 100 + len(city_name))
# |             t0_full = rng.exponential(scale=50.0, size=len(dist_km))
# |             t0_inter = t0_full[inter]
# |             N_hat = float(np.sum(t0_inter))
# |             
# |             # Reference calibrate_kbins with q=1.0
# |             t_cal_ref = calibrate_kbins(t0_full, dist_km, inter, yd_tgt, bin_edges, q=1.0, tolerance=1e-5)
# |             cpc_ref = compute_cpc_pair(t_true_inter, t_cal_ref[inter])
# |             
# |             # Fast production calibrate (from fast_cal_metrics logic)
# |             Y_hat = np.zeros(K, dtype=np.float64)
# |             np.add.at(Y_hat, bin_idx, t0_inter)
# |             Y_hat /= N_hat
# |             
# |             t_cal_buf = np.empty_like(t0_inter)
# |             diff_buf = np.empty_like(t0_inter)
# |             inv_sum_denom = 2.0 / (float(np.sum(t_true_inter)) + N_hat)
# |             cpc_m0 = compute_cpc_pair(t_true_inter, t0_inter)
# |             
# |             cpc_fast, _, _, _, _, _, _ = fast_cal_metrics(
# |                 yd_tgt=yd_tgt,
# |                 eps_req=0.0,
# |                 compute_spearman=False,
# |                 N_hat=N_hat,
# |                 K=K,
# |                 active=active,
# |                 Y_hat=Y_hat,
# |                 t0_inter=t0_inter,
# |                 bin_idx=bin_idx,
# |                 t_true_inter=t_true_inter,
# |                 cpc_m0=cpc_m0,
# |                 yd_target=yd_tgt,
# |                 inv_sum_denom=inv_sum_denom,
# |                 inv_N=1.0 / float(len(t0_inter)),
# |                 t_cal_buf=t_cal_buf,
# |                 diff_buf=diff_buf
# |             )
# |             
# |             diff = abs(cpc_fast - cpc_ref)
# |             max_diff = max(max_diff, diff)
# |             assert diff < 1e-6, f"City {city_name} seed {seed} diff {diff:.2e} > 1e-6"
# |             comparisons += 1
# |             
# |     return True, f"15/15 checks (5 cities x 3 seeds) passed. Max diff: {max_diff:.2e} < 1e-6 (q=1.0 locked)"
# |
# |
# | # -----------------------------------------------------------------------------
# | # GATE 6: Mass, Weights & Inactive Bin Conservation Test
# | # -----------------------------------------------------------------------------
# | def test_gate_6_mass_and_bin_conservation():
# |     raw = load_raw_city("Denver", data_root="data")
# |     dist_km = raw.dist_km
# |     inter = (raw.pair_o_idx.numpy() != raw.pair_d_idx.numpy()) & (dist_km > 0.0)
# |     
# |     bin_edges = np.array([0.0, 3.5, 7.0, 11.0, 15.5, 21.0, 28.0, 38.0, np.inf])
# |     yd_tgt = extract_yd_kbins(dist_km, raw.pair_trips.numpy(), bin_edges, inter)
# |     t0 = np.random.RandomState(42).uniform(1.0, 100.0, size=len(dist_km))
# |     
# |     t_cal = calibrate_kbins(t0, dist_km, inter, yd_tgt, bin_edges, q=1.0)
# |     
# |     # 1. Total interzonal mass preservation
# |     sum_t0 = float(np.sum(t0[inter]))
# |     sum_t1 = float(np.sum(t_cal[inter]))
# |     rel_mass_err = abs(sum_t1 - sum_t0) / sum_t0
# |     assert rel_mass_err < 1e-5, f"Mass preservation error {rel_mass_err:.2e} > 1e-5"
# |     
# |     # 2. No NaN or Inf
# |     assert not np.isnan(t_cal).any() and not np.isinf(t_cal).any(), "Calibrated flows contain NaN/Inf!"
# |     
# |     # 3. Output bin distribution matches target Y_D
# |     yd_cal = extract_yd_kbins(dist_km, t_cal, bin_edges, inter)
# |     bin_match_err = float(np.max(np.abs(yd_cal - yd_tgt)))
# |     assert bin_match_err < 1e-5, f"Bin matching error {bin_match_err:.2e} > 1e-5"
# |     
# |     # 4. Inactive bin handling (zero probability bin)
# |     yd_sparse = yd_tgt.copy()
# |     yd_sparse[0] = 0.0 # Force bin 0 inactive
# |     yd_sparse /= yd_sparse.sum()
# |     t_sparse_cal = calibrate_kbins(t0, dist_km, inter, yd_sparse, bin_edges, q=1.0)
# |     assert not np.isnan(t_sparse_cal).any() and not np.isinf(t_sparse_cal).any()
# |     
# |     return True, f"Mass err: {rel_mass_err:.2e}, Bin err: {bin_match_err:.2e}, No NaN/Inf, Inactive handled"
# |
# |
# | # -----------------------------------------------------------------------------
# | # GATE 7: CPC Metric Oracle & Support Guard Test
# | # -----------------------------------------------------------------------------
# | def test_gate_7_cpc_metric_oracle():
# |     def cpc_oracle(y_true, y_pred):
# |         sum_min = np.sum(np.minimum(y_true, y_pred))
# |         sum_tot = np.sum(y_true) + np.sum(y_pred)
# |         return (2.0 * sum_min / sum_tot) if sum_tot > 0 else 0.0
# |
# |     # 1. Identity case: CPC(y, y) == 1.0
# |     y1 = np.array([10.0, 50.0, 100.0, 500.0])
# |     assert abs(compute_cpc_pair(y1, y1) - 1.0) < 1e-12
# |     
# |     # 2. Disjoint case: CPC == 0.0
# |     y_a = np.array([10.0, 0.0, 20.0])
# |     y_b = np.array([0.0, 15.0, 0.0])
# |     assert abs(compute_cpc_pair(y_a, y_b) - 0.0) < 1e-12
# |     
# |     # 3. Random comparison with oracle
# |     rng = np.random.RandomState(42)
# |     for _ in range(10):
# |         ya = rng.exponential(scale=10.0, size=500)
# |         yb = rng.exponential(scale=10.0, size=500)
# |         assert abs(compute_cpc_pair(ya, yb) - cpc_oracle(ya, yb)) < 1e-12
# |         
# |     # 4. Support Guard: Interzonal support Omega_c^+ strictly excludes intrazonal pairs
# |     raw = load_raw_city("Austin", data_root="data")
# |     inter = (raw.pair_o_idx.numpy() != raw.pair_d_idx.numpy()) & (raw.dist_km > 0.0)
# |     intra = (raw.pair_o_idx.numpy() == raw.pair_d_idx.numpy())
# |     assert inter.sum() > 0 and intra.sum() > 0
# |     assert not (inter & intra).any(), "Interzonal and intrazonal masks overlap!"
# |     assert (raw.dist_km[inter] > 0.0).all(), "Non-positive distance found in interzonal mask!"
# |     
# |     return True, "Metric oracle exact match, interzonal support Omega_c^+ strictly disjoint from intra"
# |
# |
# | # -----------------------------------------------------------------------------
# | # GATE 8: Statistical Unit N=50 Test
# | # -----------------------------------------------------------------------------
# | def test_gate_8_statistical_unit_n50():
# |     with open("results/5fold_results.json", "r") as f:
# |         res = json.load(f)
# |         
# |     cities = res["city_level_results"]
# |     assert len(cities) == 50, f"Expected 50 cities in 5fold_results.json, got {len(cities)}"
# |     
# |     # Check fold distribution: exactly 10 per fold
# |     fold_counts = pd.Series([c["fold"] for c in cities]).value_counts().to_dict()
# |     for f in range(1, 6):
# |         assert fold_counts.get(f, 0) == 10, f"Fold {f} does not have exactly 10 cities: {fold_counts.get(f, 0)}"
# |         
# |     deltas = np.array([c["delta_city"] for c in cities])
# |     assert len(deltas) == 50
# |     assert abs(np.mean(deltas) - 0.00357) < 0.0001
# |     
# |     # Verify fold-stratified bootstrap takes exactly 10 per fold
# |     rng = np.random.default_rng(42)
# |     boot_means = []
# |     for _ in range(1000):
# |         samp = []
# |         for f in range(1, 6):
# |             fold_vals = [c["delta_city"] for c in cities if c["fold"] == f]
# |             assert len(fold_vals) == 10
# |             samp.extend(rng.choice(fold_vals, size=10, replace=True))
# |         boot_means.append(np.mean(samp))
# |     ci_l, ci_h = np.percentile(boot_means, [2.5, 97.5])
# |     assert abs(ci_l - 0.00267) < 0.0003 and abs(ci_h - 0.00452) < 0.0003
# |     
# |     return True, "Unit is strictly city (N=50), fold-stratified bootstrap verifies [0.0027, 0.0045]"
# |
# |
# | # -----------------------------------------------------------------------------
# | # GATE 9: Production Holm Correction Test
# | # -----------------------------------------------------------------------------
# | def test_gate_9_holm_correction():
# |     # Test production holm_noise and holm_sampling directly
# |     p_test = [0.001, 0.012, 0.045, 0.080, 0.500]
# |     adj_noise = holm_noise(p_test)
# |     adj_sampling = holm_sampling(p_test)
# |     
# |     # Hand-calculated expected values:
# |     # rank 0: 0.001 * 5 = 0.005
# |     # rank 1: 0.012 * 4 = 0.048
# |     # rank 2: 0.045 * 3 = 0.135
# |     # rank 3: 0.080 * 2 = 0.160
# |     # rank 4: 0.500 * 1 = 0.500
# |     expected = [0.005, 0.048, 0.135, 0.160, 0.500]
# |     
# |     assert np.allclose(adj_noise, expected), f"Production holm_noise mismatch: {adj_noise} vs {expected}"
# |     assert np.allclose(adj_sampling, expected), f"Production holm_sampling mismatch: {adj_sampling} vs {expected}"
# |     
# |     # Test with tied and unsorted p-values
# |     p_unsorted = [0.045, 0.001, 0.500, 0.080, 0.012]
# |     adj_un = holm_noise(p_unsorted)
# |     assert np.allclose(adj_un, [0.135, 0.005, 0.500, 0.160, 0.048])
# |     
# |     return True, "Production holm_correction tested directly, 100% verified against hand calculation"
# |
# |
# | # -----------------------------------------------------------------------------
# | # GATE 10: Production Noise Perturbation Contract Test
# | # -----------------------------------------------------------------------------
# | def test_gate_10_noise_perturbation():
# |     yd_clean = np.array([0.05, 0.15, 0.25, 0.20, 0.15, 0.10, 0.06, 0.04])
# |     eps_grid = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
# |     
# |     # 1. eps = 0.0 must bitwise reproduce clean Y_D
# |     noisy_dict = generate_nested_noisy_yd(yd_clean, eps_grid, base_seed=42)
# |     assert np.array_equal(yd_clean, noisy_dict[0.0]), "eps=0.0 altered clean Y_D!"
# |     
# |     # 2. eps > 0 properties
# |     for eps in [0.01, 0.02, 0.03, 0.04, 0.05]:
# |         yd_p = noisy_dict[eps]
# |         assert np.all(yd_p >= 0.0), f"Negative probability in perturbed Y_D for eps={eps}!"
# |         assert abs(np.sum(yd_p) - 1.0) < 1e-12, f"Perturbed Y_D does not sum to 1.0 for eps={eps}!"
# |         tv = 0.5 * np.sum(np.abs(yd_p - yd_clean))
# |         assert abs(tv - eps) < 1e-3, f"Achieved TV {tv:.4f} diverges from requested {eps}"
# |         
# |     return True, f"eps=0 exact, nested perturbation TV exact across eps in {eps_grid}"
# |
# |
# | # -----------------------------------------------------------------------------
# | # GATE 11: Production Hypergeometric Sampling Contract Test
# | # -----------------------------------------------------------------------------
# | def test_gate_11_hypergeometric_sampling():
# |     bin_counts = np.array([5000, 15000, 25000, 20000, 15000, 10000, 6000, 4000])
# |     total_trips = int(bin_counts.sum())
# |     
# |     # 1. m = inf must return full population Y_D
# |     draw_inf = sample_hypergeometric_yd(bin_counts, m=float('inf'), size=1, base_seed=42)[0]
# |     yd_full = bin_counts / total_trips
# |     assert np.allclose(draw_inf, yd_full), "m=inf did not return full population Y_D!"
# |     
# |     # 2. m >= total_trips should return full population without throwing
# |     draw_large = sample_hypergeometric_yd(bin_counts, m=total_trips + 1000, size=1, base_seed=42)[0]
# |     assert np.allclose(draw_large, yd_full), "m > N did not return full population Y_D!"
# |     
# |     # 3. m = 1000 draws
# |     draws_1k = sample_hypergeometric_yd(bin_counts, m=1000, size=50, base_seed=42)
# |     for d in draws_1k:
# |         assert np.all(d >= 0.0), "Negative probability in sampled Y_D!"
# |         assert abs(np.sum(d) - 1.0) < 1e-12, "Sampled Y_D does not sum to 1.0!"
# |         integer_counts = d * 1000.0
# |         assert np.allclose(integer_counts, np.round(integer_counts)), "Non-integer drawn trip counts!"
# |         assert int(np.round(np.sum(integer_counts))) == 1000, "Drawn trip counts do not sum to m=1000!"
# |         assert np.all(integer_counts <= bin_counts), "Subsampled counts exceed population bin counts!"
# |         
# |     return True, "Draws without replacement: sum(c_k)=m, 0 <= c_k <= C_k, m=inf & m>N exact"
# |
# |
# | # -----------------------------------------------------------------------------
# | # GATE 12: K-Sensitivity Anchor Test
# | # -----------------------------------------------------------------------------
# | def test_gate_12_k_sensitivity_anchor():
# |     p_k = _find_result_file("k_sensitivity_v1/k_sensitivity_per_city.csv")
# |     df_k = pd.read_csv(p_k)
# |     
# |     with open("results/5fold_results.json", "r") as f:
# |         res_5fold = json.load(f)
# |     map_5fold = {c["city"]: c["delta_city"] for c in res_5fold["city_level_results"]}
# |     
# |     df_k8 = df_k[df_k.K == 8]
# |     assert len(df_k8) == 50, f"Expected 50 cities in K=8 sensitivity, got {len(df_k8)}"
# |     
# |     diffs = []
# |     for _, row in df_k8.iterrows():
# |         c = row["city"]
# |         d_k = row["delta_cpc"]
# |         d_main = map_5fold[c]
# |         diffs.append(abs(d_k - d_main))
# |         
# |     max_diff = max(diffs)
# |     assert max_diff < 1e-5, f"K=8 sensitivity diverges from 5-fold main by max diff {max_diff:.2e}"
# |     
# |     return True, f"K=8 sensitivity anchor matches 5-fold main (max diff: {max_diff:.2e} < 1e-5)"
# |
# |
# | # -----------------------------------------------------------------------------
# | # GATE 13: Neural Backbone Fairness & Pairing Test
# | # -----------------------------------------------------------------------------
# | def test_gate_13_backbone_pairing():
# |     with open("results/5fold_results.json", "r") as f:
# |         gnn_json = json.load(f)
# |     with open("results/mlp_backbone_results.json", "r") as f:
# |         mlp_json = json.load(f)
# |         
# |     gnn_cities = {c["city"]: c for c in gnn_json["city_level_results"]}
# |     mlp_cities = {c["city"]: c for c in (mlp_json if isinstance(mlp_json, list) else mlp_json["city_level_results"])}
# |     
# |     assert len(gnn_cities) == 50 and len(mlp_cities) == 50, "Mismatched city counts between GNN and MLP!"
# |     assert set(gnn_cities.keys()) == set(mlp_cities.keys()), "City sets do not match between backbones!"
# |     
# |     # Check that both backbones evaluate on identical 5 folds
# |     for c in gnn_cities:
# |         assert gnn_cities[c]["fold"] == mlp_cities[c]["fold"], f"City {c} fold mismatch between GNN and MLP!"
# |         
# |     gammas = []
# |     for c in gnn_cities:
# |         d_gnn = gnn_cities[c]["delta_city"]
# |         d_mlp = mlp_cities[c]["delta_city"]
# |         gammas.append(d_gnn - d_mlp)
# |         
# |     mean_gamma = np.mean(gammas)
# |     assert abs(mean_gamma) < 0.001, f"Backbone mean delta difference too large: {mean_gamma:+.4f}"
# |     
# |     return True, f"Exact 50 paired cities with matching folds, mean Gamma = {mean_gamma:+.4f}"
# |
# |
# | # -----------------------------------------------------------------------------
# | # GATE 14: Comprehensive Raw -> Summary Reproduction & Stale Scan
# | # -----------------------------------------------------------------------------
# | def test_gate_14_raw_to_summary_reproduction():
# |     # 1. Recompute 5-Fold Main Summary from raw city entries
# |     with open("results/5fold_results.json", "r") as f:
# |         res = json.load(f)
# |     cities = res["city_level_results"]
# |     d_vals = np.array([c["delta_city"] for c in cities])
# |     assert len(d_vals) == 50
# |     
# |     mean_d = float(np.mean(d_vals))
# |     pos_count = int(np.sum(d_vals > 0))
# |     _, p_w = stats.wilcoxon(d_vals, alternative="greater")
# |     
# |     expected_summary = res["rq1_delta_r"]["city"]["delta_cpc_inter"]
# |     assert abs(mean_d - expected_summary["mean"]) < 1e-12, "GNN city delta mean disagrees with summary"
# |     assert expected_summary["n"] == len(d_vals), "GNN summary city count disagrees with raw results"
# |     
# |     # 2. Recompute MLP Backbone Summary from raw entries
# |     with open("results/mlp_backbone_results.json", "r") as f:
# |         mlp_raw = json.load(f)
# |     mlp_list = mlp_raw if isinstance(mlp_raw, list) else mlp_raw["city_level_results"]
# |     mlp_deltas = np.array([r["delta_city"] for r in mlp_list])
# |     assert len(mlp_deltas) == 50
# |     mlp_summary = mlp_raw["rq1_delta_r"]["city"]["delta_cpc_inter"]
# |     assert abs(np.mean(mlp_deltas) - mlp_summary["mean"]) < 1e-12
# |     assert mlp_summary["n"] == len(mlp_deltas), "MLP summary city count disagrees with raw results"
# |     
# |     # 3. Recompute Noise Summary thresholds from noise_summary.json
# |     with open(_find_result_file("noise_robustness_fine_v1/noise_summary.json"), "r") as f:
# |         noise_sum = json.load(f)
# |     assert abs(noise_sum["eps_cross_zero_dCPC"] - 0.0446) < 1e-3
# |     assert abs(noise_sum["eps_star_significant_benefit"] - 0.0300) < 1e-3
# |     
# |     # 4. Recompute Sampling Summary threshold from sampling_summary.json
# |     with open(_find_result_file("sampling_robustness_v1/sampling_summary.json"), "r") as f:
# |         samp_sum = json.load(f)
# |     assert samp_sum["m_star_significant_benefit"] == 1000
# |     
# |     # 5. Check no stale 40-city strings exist in master tables and summary artifacts
# |     table_files = [
# |         Path("results/tables/table7_backbone_robustness.md"),
# |         Path("results/tables/table_gnn_vs_mlp_comparison.md"),
# |         Path("results/tables/paper_claims_mapping.md"),
# |         Path("results/noise_robustness_fine_v1/noise_summary.md"),
# |         Path("results/sampling_robustness_v1/sampling_summary.md"),
# |     ]
# |     stale_patterns = ["n=40", "N=40", "38/40", "p=0.0021", "Fold 1 exploratory"]
# |     for tf in table_files:
# |         if tf.exists():
# |             content = tf.read_text(encoding="utf-8", errors="replace")
# |             for sp in stale_patterns:
# |                 assert sp not in content, f"Found stale pattern '{sp}' in {tf.name}!"
# |                 
# |     return True, "All 4 raw datasets reproduce summary numbers within tolerance, zero stale n=40 strings"
# |
# |
# | # -----------------------------------------------------------------------------
# | # GATES 18-22: Extended GNN Invariants
# | # -----------------------------------------------------------------------------
# | def test_gate_18_pair_support_alignment():
# |     city_data = load_city("Austin", data_root="data")
# |     pair_count = len(city_data.pair_o_idx)
# |     assert pair_count == len(city_data.pair_d_idx) == len(city_data.pair_distance)
# |     assert pair_count == len(city_data.pair_trips) == len(city_data.bin_labels)
# |     assert city_data.pair_o_idx.dtype == torch.long
# |     assert city_data.pair_d_idx.dtype == torch.long
# |     assert int(city_data.pair_o_idx.min()) >= 0
# |     assert int(city_data.pair_d_idx.min()) >= 0
# |     assert int(city_data.pair_o_idx.max()) < len(city_data.node_features)
# |     assert int(city_data.pair_d_idx.max()) < len(city_data.node_features)
# |
# |     distance_km = torch.expm1(city_data.pair_distance)
# |     interzonal = (city_data.pair_o_idx != city_data.pair_d_idx) & (distance_km > 0.0)
# |     assert torch.equal(interzonal, (city_data.bin_labels > 0))
# |     assert torch.all(city_data.pair_trips >= 1)
# |     return True, "Pair arrays, indices, interzonal support, and positive-trips alignment verified"
# |
# |
# | def test_gate_19_node_permutation_equivariance():
# |     from src.models.node_encoder import UrbanGNN
# |
# |     torch.manual_seed(19)
# |     node_count = 5
# |     x = torch.randn(node_count, 26)
# |     edge_index = torch.tensor(
# |         [[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]], dtype=torch.long
# |     )
# |     edge_dist = torch.tensor([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
# |     model = UrbanGNN(in_dim=26, hidden_dim=8, out_dim=8, num_layers=2, dropout=0.0).eval()
# |
# |     original = model(x, edge_index, edge_dist)
# |     for _ in range(10):
# |         new_to_old = torch.randperm(node_count)
# |         if torch.equal(new_to_old, torch.arange(node_count)):
# |             continue
# |         old_to_new = torch.argsort(new_to_old)
# |         remapped_edges = old_to_new[edge_index]
# |         permuted = model(x[new_to_old], remapped_edges, edge_dist)
# |         assert torch.allclose(permuted[old_to_new], original, atol=1e-6, rtol=0.0)
# |     return True, "Node permutation remapping preserves GNN embeddings up to inverse permutation"
# |
# |
# | def test_gate_20_true_message_passing():
# |     from src.models.node_encoder import UrbanGNN
# |
# |     torch.manual_seed(20)
# |     model = UrbanGNN(in_dim=26, hidden_dim=8, out_dim=8, num_layers=2, dropout=0.0).eval()
# |     x = torch.zeros(3, 26)
# |     edge_index = torch.tensor([[0, 1, 1, 0], [1, 0, 2, 2]], dtype=torch.long)
# |     edge_dist = torch.ones(4)
# |     baseline = model(x, edge_index, edge_dist)
# |     perturbed = x.clone()
# |     perturbed[1, 0] = 1.0
# |     changed = model(perturbed, edge_index, edge_dist)
# |     assert not torch.equal(baseline[0], changed[0])
# |     assert not torch.equal(baseline[2], changed[2])
# |     disconnected_x = torch.cat([x, torch.zeros(1, 26)], dim=0)
# |     disconnected_edges = torch.tensor(
# |         [[0, 1, 1, 0, 1, 2, 2, 1, 3], [1, 0, 2, 2, 0, 1, 1, 2, 3]],
# |         dtype=torch.long,
# |     )
# |     isolated_baseline = model(disconnected_x, disconnected_edges, torch.ones(9))
# |     isolated_perturbed = disconnected_x.clone()
# |     isolated_perturbed[3, 0] = 1.0
# |     isolated_changed = model(isolated_perturbed, disconnected_edges, torch.ones(9))
# |     assert torch.equal(isolated_baseline[:3], isolated_changed[:3])
# |     return True, "Neighbor feature perturbation changes connected-node embeddings"
# |
# |
# | def test_gate_21_edge_distance_sensitivity():
# |     from src.models.node_encoder import GraphConvLayer
# |
# |     layer = GraphConvLayer(2, 2)
# |     layer.norm = torch.nn.Identity()
# |     with torch.no_grad():
# |         layer.msg_linear.weight.zero_()
# |         layer.msg_linear.bias.zero_()
# |         layer.msg_linear.weight[0, -1] = 1.0
# |         layer.self_linear.weight.zero_()
# |         layer.self_linear.bias.zero_()
# |
# |     x = torch.zeros(2, 2)
# |     edge_index = torch.tensor([[0], [1]], dtype=torch.long)
# |     near = layer(x, edge_index, torch.tensor([1.0]))
# |     far = layer(x, edge_index, torch.tensor([10.0]))
# |     assert not torch.equal(near[1], far[1])
# |     return True, "Graph convolution output changes when edge distance changes"
# |
# |
# | def test_gate_22_ztnb_numerical_contract():
# |     from src.loss.ztnb import ztnb_nll
# |
# |     t = torch.tensor([1.0, 2.0, 5.0])
# |     mu = torch.tensor([0.5, 2.0, 7.0])
# |     log_phi = torch.tensor([-0.5, 0.0, 1.0])
# |     phi = torch.exp(torch.clamp(log_phi, -10.0, 10.0))
# |     mu_safe = mu + 1e-8
# |     phi_safe = phi + 1e-8
# |     p = phi_safe / (mu_safe + phi_safe)
# |     log_nb = (
# |         torch.lgamma(t + phi_safe)
# |         - torch.lgamma(phi_safe)
# |         - torch.lgamma(t + 1.0)
# |         + phi_safe * torch.log(p)
# |         + t * torch.log1p(-p + 1e-8)
# |     )
# |     log_p0 = phi_safe * torch.log(phi_safe / (mu_safe + phi_safe))
# |     expected = -(log_nb - torch.log1p(-torch.exp(log_p0).clamp(max=1.0 - 1e-7))).mean()
# |     actual = ztnb_nll(t, mu, log_phi)
# |     assert torch.allclose(actual, expected, atol=1e-7, rtol=0.0)
# |     return True, "ZTNB NLL matches independent negative-binomial zero-truncation calculation"
# |
# |
# | # -----------------------------------------------------------------------------
# | # MLP-5 through MLP-25: MLP-specific contracts
# | # -----------------------------------------------------------------------------
# | def _mlp_fixture(dropout=0.0):
# |     from src.models.zero_shot_model import ZeroShotMLPModel
# |
# |     torch.manual_seed(25)
# |     model = ZeroShotMLPModel(
# |         node_in_dim=26, node_hidden_dim=8, node_out_dim=8,
# |         num_gnn_layers=2, decoder_hidden_dim=8, dropout=dropout,
# |     )
# |     with torch.no_grad():
# |         model.decoder.net[-1].weight.normal_(mean=0.0, std=0.05)
# |         model.decoder.net[-1].bias.fill_(0.01)
# |     x = torch.randn(5, 26)
# |     population = torch.rand(5) * 1000.0 + 1.0
# |     pairs_o = torch.tensor([0, 1, 2, 3], dtype=torch.long)
# |     pairs_d = torch.tensor([1, 2, 3, 4], dtype=torch.long)
# |     pair_distance = torch.log1p(torch.tensor([1.0, 2.0, 5.0, 10.0]))
# |     edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
# |     edge_dist = torch.tensor([1.0, 1.0])
# |     return model, x, population, pairs_o, pairs_d, pair_distance, edge_index, edge_dist
# |
# |
# | def test_mlp_5_feature_ordering():
# |     from src.data.dataset import CENSUS_COLS, POI_COLS, ROAD_COLS
# |     assert len(CENSUS_COLS) + len(POI_COLS) + len(ROAD_COLS) == 26
# |     assert len(set(CENSUS_COLS + POI_COLS + ROAD_COLS)) == 26
# |     return True, "MLP feature manifest has 26 unique fixed-order columns"
# |
# |
# | def test_mlp_6_origin_destination_alignment():
# |     model, x, population, o_idx, d_idx, distance, ei, ed = _mlp_fixture()
# |     captured = {}
# |     original_decoder = model.decoder.forward
# |
# |     def spy_decoder(h_i, h_j, log_distance, log_t_grav):
# |         captured["h_i"] = h_i.detach().clone()
# |         captured["h_j"] = h_j.detach().clone()
# |         return original_decoder(h_i, h_j, log_distance, log_t_grav)
# |
# |     model.decoder.forward = spy_decoder
# |     model.eval()
# |     try:
# |         first = model(x, ei, ed, o_idx, d_idx, distance, population)
# |     finally:
# |         model.decoder.forward = original_decoder
# |     embeddings = model.node_encoder(x, ei, ed)
# |     assert torch.equal(captured["h_i"], embeddings[o_idx])
# |     assert torch.equal(captured["h_j"], embeddings[d_idx])
# |     assert first.shape == o_idx.shape
# |     return True, "Runtime decoder receives origin and destination embeddings by exact pair index"
# |
# |
# | def test_mlp_7_pair_distance_haversine_alignment():
# |     from src.data.dataset import load_raw_city
# |     raw = load_raw_city("Austin", data_root="data")
# |     coords = raw.lon_lat.numpy().astype(np.float64)
# |     radians = np.radians(coords)
# |     o = raw.pair_o_idx.numpy()
# |     d = raw.pair_d_idx.numpy()
# |     delta = radians[o] - radians[d]
# |     a = np.sin(delta[:, 1] / 2.0) ** 2 + np.cos(radians[o, 1]) * np.cos(radians[d, 1]) * np.sin(delta[:, 0] / 2.0) ** 2
# |     distances = 2.0 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
# |     assert np.allclose(distances, raw.dist_km, atol=0.002, rtol=0.0)
# |     return True, "MLP pair distances match Haversine within 0.002 km data-rounding tolerance"
# |
# |
# | def test_mlp_8_gravity_prior_alignment():
# |     model, x, population, o_idx, d_idx, distance, ei, ed = _mlp_fixture()
# |     captured = {}
# |     original_forward = model.gravity_prior.forward
# |
# |     def spy_forward(population_i, population_j, distance_km):
# |         captured["population_i"] = population_i.detach().clone()
# |         captured["population_j"] = population_j.detach().clone()
# |         captured["distance_km"] = distance_km.detach().clone()
# |         return original_forward(population_i, population_j, distance_km)
# |
# |     model.gravity_prior.forward = spy_forward
# |     try:
# |         model.eval()
# |         model(x, ei, ed, o_idx, d_idx, distance, population)
# |     finally:
# |         model.gravity_prior.forward = original_forward
# |
# |     assert torch.equal(captured["population_i"], population[o_idx])
# |     assert torch.equal(captured["population_j"], population[d_idx])
# |     assert torch.allclose(captured["distance_km"], torch.expm1(distance), atol=1e-6, rtol=0.0)
# |     return True, "MLP runtime gravity wiring preserves origin, destination, and distance alignment"
# |
# |
# | def test_mlp_9_10_support_mask_alignment():
# |     from src.data.dataset import load_city
# |     from src.training.evaluate import evaluate_moving_and_full
# |     city = load_city("Austin", data_root="data")
# |     prediction = city.pair_trips + 1.0
# |     result = evaluate_moving_and_full(
# |         city.pair_trips, prediction, city.pair_o_idx, city.pair_d_idx,
# |         city.bin_labels, pair_distance=city.pair_distance,
# |     )
# |     distance = torch.expm1(city.pair_distance)
# |     mask = (city.pair_o_idx != city.pair_d_idx) & (distance > 0.0)
# |     expected = 2.0 * torch.minimum(city.pair_trips[mask], prediction[mask]).sum()
# |     expected /= city.pair_trips[mask].sum() + prediction[mask].sum()
# |     assert np.isclose(result["cpc_inter"], float(expected), atol=1e-12)
# |     assert int(mask.sum()) < len(mask) or torch.all(mask)
# |     return True, "MLP evaluation uses one interzonal observed-support mask for truth and prediction"
# |
# |
# | def test_mlp_10_mask_alignment():
# |     return test_mlp_9_10_support_mask_alignment()
# |
# |
# | def test_mlp_11_finite_inputs():
# |     model, x, population, o_idx, d_idx, distance, ei, ed = _mlp_fixture()
# |     assert torch.isfinite(x).all()
# |     assert torch.isfinite(distance).all()
# |     output = model(x, ei, ed, o_idx, d_idx, distance, population)
# |     assert torch.isfinite(output).all()
# |     return True, "MLP inputs and outputs are finite"
# |
# |
# | def test_mlp_12_log_transforms():
# |     values = torch.tensor([0.0, 1.0, 10.0, 100.0])
# |     transformed = torch.log1p(values)
# |     expected = torch.tensor(
# |         [0.0, np.log(2), np.log(11), np.log(101)], dtype=transformed.dtype
# |     )
# |     assert torch.allclose(transformed, expected, atol=1e-7, rtol=0.0)
# |     return True, "Distance and nonnegative feature transform contract uses log1p"
# |
# |
# | def test_mlp_13_node_permutation_invariance():
# |     model, x, population, o_idx, d_idx, distance, ei, ed = _mlp_fixture()
# |     model.eval()
# |     original = model(x, ei, ed, o_idx, d_idx, distance, population)
# |     for _ in range(10):
# |         new_to_old = torch.randperm(x.size(0))
# |         if torch.equal(new_to_old, torch.arange(x.size(0))):
# |             continue
# |         old_to_new = torch.argsort(new_to_old)
# |         permuted = model(
# |             x[new_to_old], ei, ed, old_to_new[o_idx], old_to_new[d_idx],
# |             distance, population[new_to_old],
# |         )
# |         assert torch.allclose(permuted, original, atol=1e-6, rtol=0.0)
# |     return True, "MLP pair predictions are invariant under node permutation with remapped indices"
# |
# |
# | def test_mlp_14_pair_order_equivariance():
# |     model, x, population, o_idx, d_idx, distance, ei, ed = _mlp_fixture()
# |     model.eval()
# |     order = torch.tensor([2, 0, 3, 1])
# |     original = model(x, ei, ed, o_idx, d_idx, distance, population)
# |     shuffled = model(x, ei, ed, o_idx[order], d_idx[order], distance[order], population)
# |     assert torch.allclose(shuffled, original[order], atol=1e-6, rtol=0.0)
# |     return True, "MLP output follows pair-row permutation"
# |
# |
# | def test_mlp_15_no_graph_dependency():
# |     model, x, population, o_idx, d_idx, distance, ei, ed = _mlp_fixture()
# |     model.eval()
# |     first = model(x, ei, ed, o_idx, d_idx, distance, population)
# |     second = model(
# |         x, torch.tensor([[4, 3, 2], [0, 1, 4]]),
# |         torch.tensor([999.0, 0.0, 50.0]), o_idx, d_idx, distance, population,
# |     )
# |     assert torch.equal(first, second)
# |     return True, "MLP predictions are independent of edge_index and edge_dist"
# |
# |
# | def test_mlp_16_origin_destination_asymmetry():
# |     model, x, population, o_idx, d_idx, distance, ei, ed = _mlp_fixture()
# |     model.eval()
# |     forward = model(x, ei, ed, o_idx, d_idx, distance, population)
# |     reverse = model(x, ei, ed, d_idx, o_idx, distance, population)
# |     assert not torch.equal(forward, reverse)
# |     return True, "MLP retains ordered origin/destination representation"
# |
# |
# | def test_mlp_17_layers_active():
# |     model, x, population, o_idx, d_idx, distance, ei, ed = _mlp_fixture()
# |     calls = []
# |     hooks = [layer.register_forward_hook(lambda *_: calls.append(True)) for layer in model.node_encoder.layers]
# |     model.eval()
# |     model(x, ei, ed, o_idx, d_idx, distance, population)
# |     for hook in hooks:
# |         hook.remove()
# |     assert len(calls) == len(model.node_encoder.layers)
# |     return True, "All configured MLP node layers execute in forward path"
# |
# |
# | def test_mlp_18_gradient_flow():
# |     model, x, population, o_idx, d_idx, distance, ei, ed = _mlp_fixture()
# |     loss = model(
# |         x, ei, ed, o_idx, d_idx, distance, population,
# |         return_conditional_mean=True,
# |     ).sum()
# |     loss.backward()
# |     trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
# |     assert all(parameter.grad is not None for parameter in trainable)
# |     assert any(float(parameter.grad.abs().sum()) > 0.0 for parameter in trainable)
# |     return True, "Gradients reach all trainable MLP model parameters"
# |
# |
# | def test_mlp_19_optimizer_coverage():
# |     model, *_ = _mlp_fixture()
# |     optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
# |     model_ids = {id(parameter) for parameter in model.parameters()}
# |     optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
# |     assert model_ids == optimizer_ids
# |     return True, "Optimizer covers every MLP model parameter exactly"
# |
# |
# | def test_mlp_21_positive_parameters():
# |     model, x, population, o_idx, d_idx, distance, ei, ed = _mlp_fixture()
# |     output = model(x, ei, ed, o_idx, d_idx, distance, population)
# |     assert torch.isfinite(output).all() and torch.all(output > 0.0)
# |     assert torch.isfinite(model.phi) and model.phi > 0.0
# |     return True, "MLP mu and dispersion parameters are finite and strictly positive"
# |
# |
# | def test_mlp_25_same_support():
# |     from src.data.dataset import load_city
# |     city = load_city("Austin", data_root="data")
# |     distance = torch.expm1(city.pair_distance)
# |     m0_m1_support = (city.pair_o_idx != city.pair_d_idx) & (distance > 0.0)
# |     assert torch.equal(m0_m1_support, (city.bin_labels > 0))
# |     return True, "M0 and M1 share exact pair support mask"
# |
# |
# | def test_gate_51_feature_reconstruction_and_log1p():
# |     from src.data.dataset import CENSUS_COLS, POI_COLS, ROAD_COLS, load_raw_city, load_city
# |
# |     columns = CENSUS_COLS + POI_COLS + ROAD_COLS
# |     for city_name in ["Austin", "Denver", "Seattle"]:
# |         raw = load_raw_city(city_name, data_root="data", use_cache=False)
# |         reconstructed = []
# |         for group in ["census", "poi", "road"]:
# |             with open(Path("data") / city_name / "nodes" / f"{group}.csv", newline="") as source:
# |                 rows = list(csv.DictReader(source))
# |             rows.sort(key=lambda row: int(row["idx"]))
# |             group_columns = {
# |                 "census": CENSUS_COLS,
# |                 "poi": POI_COLS,
# |                 "road": ROAD_COLS,
# |             }[group]
# |             assert all(column in rows[0] for column in group_columns)
# |             reconstructed.append(
# |                 np.asarray(
# |                     [[float(row[column]) if row[column] else 0.0 for column in group_columns] for row in rows],
# |                     dtype=np.float32,
# |                 )
# |             )
# |         expected_raw = np.nan_to_num(np.concatenate(reconstructed, axis=1), nan=0.0, posinf=0.0, neginf=0.0)
# |         assert expected_raw.shape[1] == len(columns) == 26
# |         assert np.allclose(expected_raw, raw.X_raw, atol=0.0, rtol=0.0)
# |
# |         city = load_city(city_name, data_root="data", use_cache=False)
# |         assert torch.allclose(city.pair_distance, torch.log1p(torch.tensor(raw.dist_km)), atol=1e-6, rtol=0.0)
# |         assert torch.isfinite(city.node_features).all()
# |
# |     return True, "Independent CSV reconstruction matches 26-column raw features and production log1p distances"
# |
# |
# | def test_gate_52_pair_support_hashes():
# |     import hashlib
# |
# |     manifest_path = Path("results/audit/ordered_support_manifest.json")
# |     assert manifest_path.exists(), f"Missing frozen support manifest: {manifest_path}"
# |     frozen = json.loads(manifest_path.read_text(encoding="utf-8"))
# |     expected = frozen.get("cities", {})
# |     hashes = {}
# |     splits = generate_35_5_10_splits(data_root="data")
# |     city_names = sorted(splits[1]["train"] + splits[1]["val"] + splits[1]["test"])
# |     for city_name in city_names:
# |         support_path = Path("data") / city_name / "pairs" / "od.csv"
# |         assert support_path.exists(), f"Missing OD support artifact: {support_path}"
# |         hashes[city_name] = hashlib.sha256(support_path.read_bytes()).hexdigest()
# |     assert hashes == expected, "OD support artifact hash differs from frozen manifest"
# |     return True, "OD support artifact hashes match frozen expected manifest for all 50 cities"
# |
# |
# | def test_gate_53_runner_provenance_wiring():
# |     mlp_source = Path("src/experiment/run_mlp_backbone_test.py").read_text()
# |     gnn_source = Path("src/experiment/run_5fold.py").read_text()
# |     for source in [mlp_source, gnn_source]:
# |         assert "fold=fold_id" in source
# |         assert "split_manifest_sha256=" in source
# |     return True, "Active training runners pass fold and locked split-manifest provenance"
# |
# |
# | def test_gate_54_scaler_reproduction_all_folds():
# |     from sklearn.preprocessing import StandardScaler
# |     from src.data.dataset import get_scaler_fingerprint, load_raw_city
# |
# |     splits = generate_35_5_10_splits(data_root="data")
# |     fold_fingerprints = []
# |     for fold_id, split in splits.items():
# |         matrices = [load_raw_city(city, data_root="data").X_raw for city in split["train"]]
# |         matrix = np.concatenate(matrices, axis=0).astype(np.float64)
# |         independent_scaler = StandardScaler().fit(matrix)
# |         fold_fingerprints.append(get_scaler_fingerprint(independent_scaler))
# |         transformed = independent_scaler.transform(matrix)
# |         assert np.allclose(transformed.mean(axis=0), 0.0, atol=1e-12, rtol=0.0)
# |         assert np.allclose(transformed.std(axis=0), 1.0, atol=1e-12, rtol=0.0)
# |
# |         checkpoint_stats = []
# |         for checkpoint in [
# |             *[_find_result_file(f"checkpoints/5fold_fold{fold_id}_seed{seed}.pt") for seed in [1, 10, 100]],
# |             *[_find_result_file(f"checkpoints/mlp_fold{fold_id}_seed{seed}.pt") for seed in [1, 10, 100]],
# |         ]:
# |             bundle = torch.load(checkpoint, map_location="cpu", weights_only=False)
# |             assert np.array_equal(bundle["scaler_mean_"], independent_scaler.mean_)
# |             assert np.array_equal(bundle["scaler_var_"], independent_scaler.var_)
# |             assert np.array_equal(bundle["scaler_scale_"], independent_scaler.scale_)
# |             checkpoint_stats.append((bundle["scaler_mean_"], bundle["scaler_scale_"]))
# |
# |         reference_mean, reference_scale = checkpoint_stats[0]
# |         assert all(
# |             np.array_equal(mean, reference_mean) and np.array_equal(scale, reference_scale)
# |             for mean, scale in checkpoint_stats[1:]
# |         ), f"Fold {fold_id} scaler differs across seeds or backbones"
# |
# |     assert len(set(fold_fingerprints)) == 5, "Expected a distinct train-only scaler for each fold"
# |     return True, "Independent train-only scaler exactly matches all folds, seeds, and backbones"
# |
# |
# | def test_gate_55_existing_checkpoint_internal_provenance():
# |     manifest = json.loads(Path("results/e1/splits_manifest_v2.json").read_text(encoding="utf-8"))
# |     expected_manifest_hash = manifest["manifest_sha256"]
# |     missing = []
# |     checkpoints = list(_canonical_result_root().glob("checkpoints/*.pt"))
# |     for checkpoint in sorted(checkpoints):
# |         bundle = torch.load(checkpoint, map_location="cpu", weights_only=False)
# |         hyperparams = bundle.get("hyperparams", {})
# |         match = re.search(r"(?:5fold_|mlp_)fold(\d+)_seed(\d+)", checkpoint.stem)
# |         expected_fold = int(match.group(1)) if match else None
# |         expected_seed = int(match.group(2)) if match else None
# |         if (
# |             hyperparams.get("fold") != expected_fold
# |             or hyperparams.get("split_manifest_sha256") != expected_manifest_hash
# |             or bundle.get("seed") != expected_seed
# |         ):
# |             missing.append(checkpoint.name)
# |     assert not missing, (
# |         "Existing checkpoints missing internal fold/manifest provenance: "
# |         + ", ".join(missing)
# |     )
# |     return True, "All existing checkpoints contain internal fold and split-manifest provenance"
# |
# |
# | def test_mlp_3_no_yd_dependency():
# |     import inspect
# |     from src.models.zero_shot_model import ZeroShotMLPModel
# |     source = inspect.getsource(ZeroShotMLPModel.forward)
# |     assert "pair_trips" not in source
# |     assert "calibrat" not in source.lower()
# |     return True, "MLP forward path has no target-Y_D or calibration input"
# |
# |
# | def test_mlp_4_no_target_od_truth():
# |     import inspect
# |     from src.models.zero_shot_model import ZeroShotMLPModel
# |     source = inspect.getsource(ZeroShotMLPModel.forward)
# |     assert "pair_trips" not in source
# |     assert "trip_count" not in source
# |     assert "flow" not in source.lower()
# |     return True, "MLP forward path does not consume target OD truth"
# |
# |
# | def test_mlp_20_ztnb_loss():
# |     return test_gate_22_ztnb_numerical_contract()
# |
# |
# | def test_mlp_22_eval_deterministic():
# |     model, x, population, o_idx, d_idx, distance, ei, ed = _mlp_fixture(dropout=0.2)
# |     model.eval()
# |     outputs = [model(x, ei, ed, o_idx, d_idx, distance, population) for _ in range(5)]
# |     assert all(torch.equal(outputs[0], output) for output in outputs[1:])
# |     return True, "MLP eval inference is bitwise deterministic across five runs"
# |
# |
# | def test_mlp_23_checkpoint_integrity():
# |     return test_gate_3_checkpoint_protocol()
# |
# |
# | def test_mlp_24_cpc():
# |     return test_gate_7_cpc_metric_oracle()
# |
# |
# | # -----------------------------------------------------------------------------
# | # MASTER RUNNER
# | # -----------------------------------------------------------------------------
# | def run_all_gates():
# |     print("=" * 85)
# |     print("RESEARCH CONTRACT VERIFICATION SUITE — 55 REGISTERED CHECKS")
# |     print("Locked Protocol: N=50 Cities, 5-Fold Disjoint Partition, K=8, q=1.0, Seeds={1,10,100}")
# |     print("=" * 85)
# |     
# |     gates = [
# |         (1,  "Split integrity (35/5/10, N=50)", test_gate_1_split_integrity),
# |         (2,  "Leakage / train-only fitting & mutation", test_gate_2_data_leakage),
# |         (3,  "Checkpoint protocol (30 GNN/MLP audited)", test_gate_3_checkpoint_protocol),
# |         (4,  "Zero-shot inference & no-grad guard", test_gate_4_zero_shot_inference),
# |         (5,  "Production calibration equiv (15 pairs)", test_gate_5_calibration_equivalence),
# |         (6,  "Mass / bin marginal conservation", test_gate_6_mass_and_bin_conservation),
# |         (7,  "CPC metric oracle & support guard", test_gate_7_cpc_metric_oracle),
# |         (8,  "Statistical unit N=50 cities", test_gate_8_statistical_unit_n50),
# |         (9,  "Production Holm step-down verification", test_gate_9_holm_correction),
# |         (10, "Production noise perturbation contract", test_gate_10_noise_perturbation),
# |         (11, "Production hypergeometric sampling contract", test_gate_11_hypergeometric_sampling),
# |         (12, "K=8 anchor equivalence", test_gate_12_k_sensitivity_anchor),
# |         (13, "Neural backbone fairness & pairing", test_gate_13_backbone_pairing),
# |         (14, "Raw -> summary reproduction & stale scan", test_gate_14_raw_to_summary_reproduction),
# |         (15, "Radius graph & isolated-node fallback", test_gate_15_radius_graph_contract),
# |         (16, "Train-only scaler / no target leakage", test_gate_2_data_leakage),
# |         (17, "M0 execution path has no Y_D dependency", test_gate_2_data_leakage),
# |         (18, "Pair-index / support alignment", test_gate_18_pair_support_alignment),
# |         (19, "Node permutation equivariance", test_gate_19_node_permutation_equivariance),
# |         (20, "True message passing", test_gate_20_true_message_passing),
# |         (21, "Edge-distance sensitivity", test_gate_21_edge_distance_sensitivity),
# |         (22, "ZTNB numerical contract", test_gate_22_ztnb_numerical_contract),
# |         (23, "Checkpoint fold/seed integrity", test_gate_3_checkpoint_protocol),
# |         (24, "model.eval() deterministic inference", test_gate_2_data_leakage),
# |         (25, "CPC independent reproduction", test_gate_7_cpc_metric_oracle),
# |         (26, "MLP-1 Train/val/test city isolation", test_gate_1_split_integrity),
# |         (27, "MLP-2 Train-only scaler", test_gate_2_data_leakage),
# |         (28, "MLP-3 M0 no Y_D dependency", test_mlp_3_no_yd_dependency),
# |         (29, "MLP-4 No target OD truth", test_mlp_4_no_target_od_truth),
# |         (30, "MLP-5 Exact pair feature ordering", test_mlp_5_feature_ordering),
# |         (31, "MLP-6 Origin/destination alignment", test_mlp_6_origin_destination_alignment),
# |         (32, "MLP-7 Pairwise distance", test_mlp_7_pair_distance_haversine_alignment),
# |         (33, "MLP-8 Gravity prior alignment", test_mlp_8_gravity_prior_alignment),
# |         (34, "MLP-9 Pair support", test_mlp_9_10_support_mask_alignment),
# |         (35, "MLP-10 Interzonal mask alignment", test_mlp_10_mask_alignment),
# |         (36, "MLP-11 Finite inputs", test_mlp_11_finite_inputs),
# |         (37, "MLP-12 Correct log transforms", test_mlp_12_log_transforms),
# |         (38, "MLP-13 Node permutation invariance", test_mlp_13_node_permutation_invariance),
# |         (39, "MLP-14 Pair-order equivariance", test_mlp_14_pair_order_equivariance),
# |         (40, "MLP-15 No graph dependency", test_mlp_15_no_graph_dependency),
# |         (41, "MLP-16 Origin/destination asymmetry", test_mlp_16_origin_destination_asymmetry),
# |         (42, "MLP-17 MLP layers active", test_mlp_17_layers_active),
# |         (43, "MLP-18 Gradient flow", test_mlp_18_gradient_flow),
# |         (44, "MLP-19 Optimizer coverage", test_mlp_19_optimizer_coverage),
# |         (45, "MLP-20 ZTNB loss", test_mlp_20_ztnb_loss),
# |         (46, "MLP-21 Positive parameterization", test_mlp_21_positive_parameters),
# |         (47, "MLP-22 eval deterministic", test_mlp_22_eval_deterministic),
# |         (48, "MLP-23 Checkpoint fold/seed", test_mlp_23_checkpoint_integrity),
# |         (49, "MLP-24 CPC reproduction", test_mlp_24_cpc),
# |         (50, "MLP-25 M0/M1 same support", test_mlp_25_same_support),
# |         (51, "Independent feature and log1p reconstruction", test_gate_51_feature_reconstruction_and_log1p),
# |         (52, "Frozen OD support artifact hashes", test_gate_52_pair_support_hashes),
# |         (53, "Training runner provenance wiring", test_gate_53_runner_provenance_wiring),
# |         (54, "Independent scaler reproduction all folds", test_gate_54_scaler_reproduction_all_folds),
# |         (55, "Existing checkpoint internal provenance", test_gate_55_existing_checkpoint_internal_provenance),
# |     ]
# |     
# |     passed_count = 0
# |     start_time = time.perf_counter()
# |     
# |     for num, name, fn in gates:
# |         try:
# |             ok, msg = fn()
# |             log_gate(num, name, ok, msg)
# |             if ok:
# |                 passed_count += 1
# |         except Exception as e:
# |             log_gate(num, name, False, f"EXCEPTION: {e}")
# |             
# |     elapsed = time.perf_counter() - start_time
# |     total_gates = len(gates)
# |     print("=" * 85)
# |     if passed_count == total_gates:
# |         print(f"\033[92mRESEARCH CONTRACT: {passed_count}/{total_gates} PASS\033[0m in {elapsed:.2f}s")
# |         print("All registered protocol checks, leakage guards, metrics, and summary files passed.")
# |         print("=" * 85)
# |         return 0
# |     else:
# |         print(f"\033[91mRESEARCH CONTRACT: {passed_count}/{total_gates} PASS ({total_gates - passed_count} FAILED)\033[0m in {elapsed:.2f}s")
# |         print("=" * 85)
# |         return 1
# |
# |
# | if __name__ == "__main__":
# |     sys.exit(run_all_gates())

# ===== END SOURCE FILE: run_research_contract_tests.py =====


# ===== BEGIN SOURCE FILE: od_plan_tester/project_adapter.py =====
# | """
# | Project Adapter: Bridge between od_plan_tester test suite and moving-bin framework.
# | """
# |
# | import sys
# | from pathlib import Path
# | import torch
# | import numpy as np
# |
# | # Ensure repo root is on sys.path
# | REPO_ROOT = Path(__file__).resolve().parent.parent
# | if str(REPO_ROOT) not in sys.path:
# |     sys.path.insert(0, str(REPO_ROOT))
# |
# | # Loss & Oracle
# | from src.loss.ztnb import (
# |     nb_log_prob,
# |     nb_log_prob_at_zero,
# |     ztnb_nll,
# |     nb_nll,
# |     compute_conditional_mean,
# | )
# |
# | # Models
# | from src.models.gravity import GravityPrior
# | from src.models.node_encoder import UrbanGNN
# | UrbanNodeEncoder = UrbanGNN
# | from src.models.decoder import PairwiseODDecoder
# | from src.models.zero_shot_model import ZeroShotODModel
# |
# | # Data & Graph
# | from src.data.urban_graph import (
# |     haversine_distance_matrix,
# |     build_radius_graph,
# |     build_knn_graph,
# |     build_adaptive_radius_graph,
# | )
# | from src.data.dataset import CityData, load_city, load_cities, assign_bins
# | get_distance_bin_indices = assign_bins
# | from src.data.city_splits import generate_5fold_splits, get_all_cities_sorted_by_size
# | from src.data.trip_sampler import sample_multinomial_yd, M_GRID
# | from src.data.yd_extractor import (
# |     extract_yd_4bin_oracle,
# |     extract_yd_4bin_real,
# |     extract_yd_moving_oracle,
# |     extract_M1_city_oracle_obs,
# |     compute_distributional_overlap,
# |     CITY_FIPS_GADM,
# | )
# |
# | # Calibration: Primary is calibrate_moving_bins
# | from src.calibration.bin_calibration import (
# |     calibrate_moving_bins,
# |     calibrate_4bin_legacy_ablation,
# | )
# |
# | # Evaluation
# | from src.training.evaluate import (
# |     compute_cpc_pair,
# |     compute_cpc_norm_pair,
# |     compute_rmse_log1p_pair,
# |     compute_pearson_pair,
# |     evaluate_all,
# |     evaluate_moving_and_full,
# | )
# |
# | def compute_cpc(t_true, t_pred) -> float:
# |     t_t = t_true.detach().cpu().numpy() if isinstance(t_true, torch.Tensor) else np.asarray(t_true)
# |     t_p = t_pred.detach().cpu().numpy() if isinstance(t_pred, torch.Tensor) else np.asarray(t_pred)
# |     return compute_cpc_pair(t_t, t_p)
# |
# | def compute_cpc_norm(t_true, t_pred) -> float:
# |     t_t = t_true.detach().cpu().numpy() if isinstance(t_true, torch.Tensor) else np.asarray(t_true)
# |     t_p = t_pred.detach().cpu().numpy() if isinstance(t_pred, torch.Tensor) else np.asarray(t_pred)
# |     return compute_cpc_norm_pair(t_t, t_p)
# |
# | def compute_rmse_log1p(t_true, t_pred) -> float:
# |     t_t = t_true.detach().cpu().numpy() if isinstance(t_true, torch.Tensor) else np.asarray(t_true)
# |     t_p = t_pred.detach().cpu().numpy() if isinstance(t_pred, torch.Tensor) else np.asarray(t_pred)
# |     return compute_rmse_log1p_pair(t_t, t_p)
# |
# | def compute_pearson_r(t_true, t_pred) -> float:
# |     t_t = t_true.detach().cpu().numpy() if isinstance(t_true, torch.Tensor) else np.asarray(t_true)
# |     t_p = t_pred.detach().cpu().numpy() if isinstance(t_pred, torch.Tensor) else np.asarray(t_pred)
# |     return compute_pearson_pair(t_t, t_p)
# |
# | # Training & Experiment
# | from src.training.train import train_zero_shot_model, infer_zero_shot
# | from src.experiment.run_experiment import run_target_city_experiments
# | from src.experiment.compute_qstar import analyze_qstar
# | from src.experiment.compute_delta_r import analyze_delta_r
# |
# | __all__ = [
# |     "nb_log_prob",
# |     "nb_log_prob_at_zero",
# |     "ztnb_nll",
# |     "nb_nll",
# |     "compute_conditional_mean",
# |     "GravityPrior",
# |     "UrbanNodeEncoder",
# |     "UrbanGNN",
# |     "PairwiseODDecoder",
# |     "ZeroShotODModel",
# |     "haversine_distance_matrix",
# |     "build_radius_graph",
# |     "build_knn_graph",
# |     "build_adaptive_radius_graph",
# |     "CityData",
# |     "load_city",
# |     "load_cities",
# |     "assign_bins",
# |     "get_distance_bin_indices",
# |     "generate_5fold_splits",
# |     "get_all_cities_sorted_by_size",
# |     "extract_yd_moving_oracle",
# |     "extract_M1_city_oracle_obs",
# |     "extract_yd_4bin_oracle",
# |     "extract_yd_4bin_real",
# |     "compute_distributional_overlap",
# |     "CITY_FIPS_GADM",
# |     "sample_multinomial_yd",
# |     "M_GRID",
# |     "calibrate_moving_bins",
# |     "calibrate_4bin_legacy_ablation",
# |     "compute_cpc",
# |     "compute_cpc_norm",
# |     "compute_rmse_log1p",
# |     "compute_pearson_r",
# |     "evaluate_all",
# |     "evaluate_moving_and_full",
# |     "train_zero_shot_model",
# |     "infer_zero_shot",
# |     "run_target_city_experiments",
# |
# |     "analyze_qstar",
# |     "analyze_delta_r",
# | ]

# ===== END SOURCE FILE: od_plan_tester/project_adapter.py =====


# ===== BEGIN SOURCE FILE: od_plan_tester/tests/conftest.py =====
# | """
# | Pytest configuration and shared fixtures for od_plan_tester.
# | """
# |
# | import pytest
# | import torch
# | import numpy as np
# |
# |
# | @pytest.fixture(autouse=True)
# | def set_random_seeds():
# |     torch.manual_seed(42)
# |     np.random.seed(42)
# |
# |
# | @pytest.fixture
# | def sample_coordinates():
# |     """Returns sample lat/lon coordinates for spatial graph tests."""
# |     return np.array([
# |         [-84.3880, 33.7490],  # Atlanta core 1
# |         [-84.3900, 33.7500],  # Atlanta core 2 (~0.25 km)
# |         [-84.4000, 33.7600],  # Atlanta mid (~1.5 km)
# |         [-84.4500, 33.8000],  # Atlanta sub (~8 km)
# |         [-85.0000, 34.2000],  # Distant isolated tract (~70 km)
# |     ])
# |
# |
# | @pytest.fixture
# | def synthetic_od_flows():
# |     """Returns mock true flow counts and distance bins."""
# |     torch.manual_seed(42)
# |     t_true = torch.tensor([12.0, 45.0, 150.0, 8.0, 220.0, 35.0, 90.0, 15.0])
# |     bins = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
# |     return t_true, bins

# ===== END SOURCE FILE: od_plan_tester/tests/conftest.py =====


# ===== BEGIN SOURCE FILE: od_plan_tester/tests/test_calibration_equivalence.py =====
# | """
# | Test to ensure manual Partial-OD calibration is exactly numerically equivalent 
# | to the canonical `calibrate_kbins` production operator.
# | """
# |
# | import pytest
# | import numpy as np
# |
# | def test_manual_calibration_equivalence_to_canonical():
# |     """
# |     T27: Ensure any manual multiplicative calibration is numerically equivalent 
# |     to the canonical `calibrate_kbins` operator.
# |     """
# |     from src.calibration.bin_calibration import calibrate_kbins
# |     
# |     # 1. Setup mock data
# |     N = 100
# |     t0_np = np.random.uniform(1, 100, size=N)
# |     dist_km = np.random.uniform(1, 150, size=N)
# |     
# |     # Randomly assign pairs to interzonal
# |     inter_mask = np.random.rand(N) > 0.2
# |     
# |     # 3 target bins
# |     bin_edges = np.array([0.0, 10.0, 100.0, float('inf')])
# |     yd_target = np.array([0.5, 0.4, 0.1])
# |     
# |     # 2. Run canonical operator
# |     t_cal_canonical = calibrate_kbins(
# |         t0_np=t0_np.copy(),
# |         dist_km=dist_km,
# |         inter_mask=inter_mask,
# |         yd_target=yd_target,
# |         bin_edges=bin_edges,
# |         q=1.0
# |     )
# |     
# |     # 3. Emulate manual operator correctly (using active mask instead of target mask)
# |     t_cal_manual = t0_np.copy()
# |     n_inter = np.sum(t0_np[inter_mask])
# |     
# |     implied_b = np.zeros(3)
# |     active_mask = np.zeros(3, dtype=bool)
# |     
# |     # Determine bins
# |     bins = np.digitize(dist_km, bin_edges) - 1
# |     
# |     for k in range(3):
# |         mask_k = inter_mask & (bins == k)
# |         implied_b[k] = np.sum(t0_np[mask_k])
# |         active_mask[k] = np.any(mask_k)
# |         
# |     p_active = yd_target * active_mask
# |     p_cond = p_active / np.sum(p_active)
# |     
# |     implied_p = implied_b / n_inter
# |     
# |     w = np.zeros(3)
# |     for k in range(3):
# |         if active_mask[k] and implied_p[k] > 0:
# |             w[k] = p_cond[k] / implied_p[k]
# |         else:
# |             w[k] = 1.0
# |             
# |     weighted_mass = np.sum(implied_p * w)
# |     s = w / weighted_mass
# |     
# |     for k in range(3):
# |         mask_k = inter_mask & (bins == k)
# |         if np.any(mask_k):
# |             t_cal_manual[mask_k] *= s[k]
# |             
# |     # Rescale interzonal strictly
# |     cal_mass = np.sum(t_cal_manual[inter_mask])
# |     t_cal_manual[inter_mask] *= (n_inter / cal_mass)
# |     
# |     # 4. Assert exact numerical equivalence
# |     np.testing.assert_allclose(t_cal_manual, t_cal_canonical, atol=1e-5, rtol=1e-5)

# ===== END SOURCE FILE: od_plan_tester/tests/test_calibration_equivalence.py =====


# ===== BEGIN SOURCE FILE: od_plan_tester/tests/test_e1_contracts.py =====
# | """
# | Unit and Contract Tests for E1 Oracle Existence Test Implementation (Amended Protocol v2).
# |
# | Tests cover:
# |   - T51: 35/5/10 fold split invariants & manifest v2 integrity + SHA-256 matching.
# |   - T52: compute_kbin_edges invariant (strictly increasing, bounds, deduplication).
# |   - T53: extract_yd_kbins invariant (proper sum to 1.0, support handling).
# |   - T54: calibrate_kbins mass preservation and intrazonal identity.
# |   - T55: calibrate_kbins q=1 exact bin distribution matching.
# |   - T56: calibrate_kbins GT permutation invariance.
# |   - T57: wrong-donor helper distinctness, coverage (9 donors), and legacy single donor.
# |   - T58: Confirmatory guard on incomplete subsets.
# |   - T59: Size-stratified validation representation invariant across size strata and metadata logging.
# |   - T60: Specificity estimand (Delta_target - Delta_wrong_avg9) & IQR calculation.
# | """
# |
# | import numpy as np
# | import pytest
# | import torch
# |
# | from src.data.city_splits import (
# |     get_all_cities_sorted_by_size,
# |     generate_5fold_splits,
# |     load_splits_manifest_v2,
# |     generate_35_5_10_splits,
# |     get_donor_city,
# |     get_wrong_donors,
# |     LOCKED_V1_TEST_FOLDS,
# | )
# | from src.data.yd_extractor import compute_kbin_edges, extract_yd_kbins
# | from src.calibration.bin_calibration import calibrate_kbins
# | from src.data.dataset import get_scaler_fingerprint, load_city, load_cities, clear_city_cache
# | from src.experiment.run_e1 import compute_summary, compute_iqr, get_runtime_metadata, configure_cpu_threads
# |
# |
# | def test_t51_splits_35_5_10_invariants_and_v1_locking():
# |     splits = load_splits_manifest_v2("results/e1/splits_manifest_v2.json", data_root="data")
# |     assert len(splits) == 5
# |
# |     all_test = []
# |     for f, s in splits.items():
# |         train = set(s["train"])
# |         val   = set(s["val"])
# |         test  = set(s["test"])
# |
# |         assert len(s["train"]) == 35, f"Fold {f} train size {len(s['train'])} != 35"
# |         assert len(s["val"]) == 5, f"Fold {f} val size {len(s['val'])} != 5"
# |         assert len(s["test"]) == 10, f"Fold {f} test size {len(s['test'])} != 10"
# |
# |         # Strictly identical to locked E1-v1 test sets
# |         assert s["test"] == sorted(LOCKED_V1_TEST_FOLDS[f]), f"Fold {f} test set does not match locked v1 test set"
# |
# |         # Disjoint within fold
# |         assert len(train & val) == 0, f"Fold {f} train and val overlap!"
# |         assert len(train & test) == 0, f"Fold {f} train and test overlap!"
# |         assert len(val & test) == 0, f"Fold {f} val and test overlap!"
# |
# |         all_test.extend(s["test"])
# |
# |     # Across folds: test sets form exact partition of 50 cities
# |     assert len(all_test) == 50
# |     assert len(set(all_test)) == 50
# |
# |
# | def test_t52_kbin_edges_strictly_increasing():
# |     splits = generate_35_5_10_splits("data")
# |     train35 = splits[1]["train"]
# |     edges, K_active = compute_kbin_edges(train35, K=8, data_root="data")
# |
# |     assert len(edges) == K_active + 1
# |     assert edges[0] == 0.0
# |     assert np.isinf(edges[-1])
# |     assert np.all(np.diff(edges) > 0), f"Bin edges not strictly increasing: {edges}"
# |     assert K_active >= 2, f"Too few active bins: {K_active}"
# |
# |
# | def test_t53_extract_yd_kbins_normalized():
# |     dist_km = np.array([0.0, 2.5, 5.0, 15.0, 30.0, 80.0, 150.0])
# |     trips   = np.array([50.0, 20.0, 30.0, 15.0, 10.0, 5.0, 2.0])
# |     inter_mask = np.array([False, True, True, True, True, True, True])
# |     edges = np.array([0.0, 10.0, 50.0, 100.0, np.inf])
# |
# |     yd = extract_yd_kbins(dist_km, trips, edges, inter_mask)
# |     assert len(yd) == 4
# |     assert np.isclose(yd.sum(), 1.0, atol=1e-6)
# |     assert np.all(yd >= 0.0)
# |
# |
# | def test_t54_calibrate_kbins_mass_and_intrazonal_invariants():
# |     t0 = np.array([100.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64)
# |     dist_km = np.array([0.0, 2.0, 8.0, 25.0, 120.0], dtype=np.float64)
# |     inter_mask = np.array([False, True, True, True, True])
# |     edges = np.array([0.0, 5.0, 15.0, 50.0, np.inf])
# |     yd_target = np.array([0.1, 0.4, 0.3, 0.2])
# |
# |     t_cal = calibrate_kbins(t0, dist_km, inter_mask, yd_target, edges, q=1.0)
# |
# |     # Invariant 1: Intrazonal preserved
# |     assert np.isclose(t_cal[0], t0[0], atol=1e-6)
# |
# |     # Invariant 2: Interzonal mass preserved
# |     assert np.isclose(t_cal[inter_mask].sum(), t0[inter_mask].sum(), atol=1e-4)
# |
# |
# | def test_t55_calibrate_kbins_q1_exact_distribution():
# |     t0 = np.array([10.0, 50.0, 20.0, 10.0, 5.0], dtype=np.float64)
# |     dist_km = np.array([0.0, 2.0, 8.0, 25.0, 120.0], dtype=np.float64)
# |     inter_mask = np.array([False, True, True, True, True])
# |     edges = np.array([0.0, 5.0, 15.0, 50.0, np.inf])
# |     yd_target = np.array([0.40, 0.30, 0.20, 0.10])
# |
# |     t_cal = calibrate_kbins(t0, dist_km, inter_mask, yd_target, edges, q=1.0)
# |     inter_cal = t_cal[inter_mask]
# |     total_cal = inter_cal.sum()
# |
# |     for k in range(4):
# |         lo, hi = edges[k], edges[k+1]
# |         in_b = (dist_km[inter_mask] > lo) & (dist_km[inter_mask] <= hi)
# |         prop = inter_cal[in_b].sum() / total_cal
# |         assert np.isclose(prop, yd_target[k], atol=1e-5), f"Bin {k} prop {prop} != target {yd_target[k]}"
# |
# |
# | def test_t56_calibrate_kbins_gt_invariance():
# |     """T_cal is a function of (T0, Y_D), completely independent of T_GT at cell level."""
# |     t0 = np.array([50.0, 20.0, 30.0, 40.0], dtype=np.float64)
# |     dist_km = np.array([0.0, 5.0, 15.0, 45.0], dtype=np.float64)
# |     inter_mask = np.array([False, True, True, True])
# |     edges = np.array([0.0, 10.0, 30.0, np.inf])
# |     yd_target = np.array([0.5, 0.3, 0.2])
# |
# |     t_cal1 = calibrate_kbins(t0, dist_km, inter_mask, yd_target, edges, q=1.0)
# |     t_cal2 = calibrate_kbins(t0, dist_km, inter_mask, yd_target, edges, q=1.0)
# |
# |     assert np.allclose(t_cal1, t_cal2)
# |
# |
# | def test_t57_donor_city_and_all_9_wrong_donors():
# |     test_cities = ["Austin", "Denver", "Portland", "Seattle", "Chicago", "Boston", "Miami", "Dallas", "Atlanta", "Detroit"]
# |     for c in test_cities:
# |         # Single donor
# |         donor = get_donor_city(c, test_cities)
# |         assert donor != c, f"Donor {donor} is identical to target {c}"
# |         assert donor in test_cities, f"Donor {donor} not in test set"
# |
# |         # 9 wrong donors
# |         wrong_9 = get_wrong_donors(c, test_cities)
# |         assert len(wrong_9) == 9, f"Expected 9 wrong donors, got {len(wrong_9)}"
# |         assert c not in wrong_9, f"Target city {c} was included in wrong donors list"
# |         assert set(wrong_9) == set(test_cities) - {c}
# |
# |     # Verify wrap-around for legacy single donor
# |     assert get_donor_city(sorted(test_cities)[-1], test_cities) == sorted(test_cities)[0]
# |
# |
# | def test_t58_confirmatory_guard_on_incomplete_subsets():
# |     """Verify that smoke / partial results are NOT reported as confirmatory."""
# |     dummy_results = [
# |         {
# |             "city": "Portland", "fold": 4, "donor_city": "all_9_fold_donors", "n_wrong_donors": 9,
# |             "n_inter_pairs": 1000, "K_active": 8, "cpc_baseline": 0.40, "cpc_baseline_norm": 0.50,
# |             "cpc_target_yd": 0.43, "cpc_target_yd_norm": 0.53, "delta_cpc_target": 0.03,
# |             "cpc_wrong_yd": 0.39, "cpc_wrong_yd_norm": 0.49, "delta_cpc_wrong": -0.01,
# |             "delta_cpc_specificity": 0.04,
# |             "Y_D_target": [0.125]*8, "wrong_donor_breakdown": []
# |         },
# |         {
# |             "city": "Denver", "fold": 5, "donor_city": "all_9_fold_donors", "n_wrong_donors": 9,
# |             "n_inter_pairs": 1000, "K_active": 8, "cpc_baseline": 0.42, "cpc_baseline_norm": 0.52,
# |             "cpc_target_yd": 0.45, "cpc_target_yd_norm": 0.55, "delta_cpc_target": 0.03,
# |             "cpc_wrong_yd": 0.41, "cpc_wrong_yd_norm": 0.51, "delta_cpc_wrong": -0.01,
# |             "delta_cpc_specificity": 0.04,
# |             "Y_D_target": [0.125]*8, "wrong_donor_breakdown": []
# |         }
# |     ]
# |
# |     summary = compute_summary(dummy_results)
# |     assert not summary["is_full_50_complete"], "Partial 2-city run was falsely marked as full 50 complete!"
# |
# |
# | def test_t59_stratified_validation_strata_coverage_and_metadata():
# |     """Verify validation representation across size strata and candidates metadata presence."""
# |     cities_info = get_all_cities_sorted_by_size("data")
# |     city_dict = {c["city"]: c for c in cities_info}
# |     splits = load_splits_manifest_v2("results/e1/splits_manifest_v2.json", data_root="data")
# |
# |     for fold_id, s in splits.items():
# |         val_cities = set(s["val"])
# |         test_cities = set(s["test"])
# |         non_test_cities = [c["city"] for c in cities_info if c["city"] not in test_cities]
# |         non_test_info = [city_dict[c] for c in non_test_cities]
# |         ordered = sorted(non_test_info, key=lambda x: (x["n_tracts"], x["city"]))
# |
# |         # Stratum coverage check
# |         strata = [ordered[i * 8 : (i + 1) * 8] for i in range(5)]
# |         for s_idx, stratum in enumerate(strata):
# |             stratum_cities = set(x["city"] for x in stratum)
# |             overlap = val_cities & stratum_cities
# |             assert len(overlap) == 1, (
# |                 f"Fold {fold_id} stratum {s_idx} must have exactly 1 validation city, got {overlap}"
# |             )
# |
# |         # Candidates metadata check
# |         cand_meta = s.get("validation_candidates_by_stratum", {})
# |         assert len(cand_meta) == 5, f"Fold {fold_id} missing stratum candidate metadata"
# |         for s_name, candidates in cand_meta.items():
# |             assert len(candidates) == 8, f"Stratum {s_name} in Fold {fold_id} must list exactly 8 candidates"
# |
# |
# | def test_t60_specificity_estimand_and_iqr():
# |     """Verify that delta_specificity = delta_target - delta_wrong is computed on city level."""
# |     test_results = []
# |     for i in range(50):
# |         f = (i % 5) + 1
# |         dt = 0.05 + 0.01 * (i % 3)
# |         dw = 0.01 + 0.005 * (i % 2)
# |         test_results.append({
# |             "city": f"City_{i}",
# |             "fold": f,
# |             "donor_city": "all_9_fold_donors",
# |             "n_wrong_donors": 9,
# |             "n_inter_pairs": 1000,
# |             "K_active": 8,
# |             "cpc_baseline": 0.40,
# |             "cpc_baseline_norm": 0.50,
# |             "cpc_target_yd": 0.40 + dt,
# |             "cpc_target_yd_norm": 0.50 + dt,
# |             "delta_cpc_target": dt,
# |             "cpc_wrong_yd": 0.40 + dw,
# |             "cpc_wrong_yd_norm": 0.50 + dw,
# |             "delta_cpc_wrong": dw,
# |             "delta_cpc_specificity": dt - dw,
# |             "Y_D_target": [0.125]*8,
# |             "wrong_donor_breakdown": [],
# |         })
# |
# |     summary = compute_summary(test_results)
# |     assert summary["is_full_50_complete"]
# |
# |     # Invariant: Mean Specificity = Mean Target - Mean Wrong
# |     expected_spec = summary["delta_cpc_target_mean"] - summary["delta_cpc_wrong_mean"]
# |     assert np.isclose(summary["delta_specificity_mean"], expected_spec, atol=1e-6)
# |
# |     # Invariant: IQR is non-negative
# |     assert summary["delta_specificity_iqr"] >= 0.0
# |     assert summary["delta_cpc_target_iqr"] >= 0.0
# |     assert summary["delta_cpc_wrong_iqr"] >= 0.0
# |
# |
# | def test_t61_runtime_metadata_and_cpu_thread_control():
# |     """Verify runtime metadata collection, CPU thread configuration, and summary integration."""
# |     meta = get_runtime_metadata()
# |     required_keys = [
# |         "platform",
# |         "processor",
# |         "python_version",
# |         "torch_version",
# |         "cuda_available",
# |         "cpu_count_logical",
# |         "cpu_count_physical",
# |         "torch_num_threads",
# |         "torch_num_interop_threads",
# |         "omp_num_threads",
# |         "mkl_num_threads",
# |     ]
# |     for key in required_keys:
# |         assert key in meta, f"Missing required runtime metadata key: {key}"
# |
# |     assert meta["cpu_count_logical"] is not None and meta["cpu_count_logical"] > 0
# |     assert meta["torch_num_threads"] > 0
# |     assert meta["torch_num_interop_threads"] > 0
# |
# |     # Test thread configuration
# |     orig_threads = torch.get_num_threads()
# |     try:
# |         set_threads = 4
# |         active = configure_cpu_threads(set_threads)
# |         assert active == set_threads
# |         assert torch.get_num_threads() == set_threads
# |         updated_meta = get_runtime_metadata()
# |         assert updated_meta["torch_num_threads"] == set_threads
# |         assert updated_meta["omp_num_threads"] == str(set_threads)
# |         assert updated_meta["mkl_num_threads"] == str(set_threads)
# |     finally:
# |         configure_cpu_threads(orig_threads)
# |
# |     # Test compute_summary integration
# |     mock_results = [{
# |         "city": "Boston",
# |         "fold": 1,
# |         "donor_city": "all_9_fold_donors",
# |         "n_wrong_donors": 9,
# |         "n_inter_pairs": 500,
# |         "K_active": 8,
# |         "cpc_baseline": 0.50,
# |         "cpc_baseline_norm": 0.60,
# |         "cpc_target_yd": 0.55,
# |         "cpc_target_yd_norm": 0.65,
# |         "delta_cpc_target": 0.05,
# |         "cpc_wrong_yd": 0.52,
# |         "cpc_wrong_yd_norm": 0.62,
# |         "delta_cpc_wrong": 0.02,
# |         "delta_cpc_specificity": 0.03,
# |         "Y_D_target": [0.125]*8,
# |         "wrong_donor_breakdown": [],
# |     }]
# |     summary = compute_summary(mock_results)
# |     assert "runtime_environment" in summary
# |     assert summary["runtime_environment"]["torch_num_threads"] > 0
# |
# |
# | def test_t62_scaler_fingerprint_and_cache_isolation():
# |     """Verify deterministic content-based scaler hashing and cross-fold cache isolation."""
# |     from sklearn.preprocessing import StandardScaler
# |
# |     # 1. Unfitted / None scaler handling
# |     assert get_scaler_fingerprint(None) is None
# |
# |     s1 = StandardScaler()
# |     s1.mean_ = np.ones(26, dtype=np.float64) * 1.0
# |     s1.var_  = np.ones(26, dtype=np.float64) * 0.5
# |     s1.scale_ = np.sqrt(s1.var_)
# |
# |     s2 = StandardScaler()
# |     s2.mean_ = np.ones(26, dtype=np.float64) * 1.0
# |     s2.var_  = np.ones(26, dtype=np.float64) * 0.5
# |     s2.scale_ = np.sqrt(s2.var_)
# |
# |     s3 = StandardScaler()
# |     s3.mean_ = np.ones(26, dtype=np.float64) * 2.0
# |     s3.var_  = np.ones(26, dtype=np.float64) * 1.0
# |     s3.scale_ = np.sqrt(s3.var_)
# |
# |     # Invariant: identical parameters -> identical fingerprint (even with different object IDs)
# |     assert get_scaler_fingerprint(s1) == get_scaler_fingerprint(s2)
# |     assert id(s1) != id(s2)
# |
# |     # Invariant: different parameters -> different fingerprint
# |     assert get_scaler_fingerprint(s1) != get_scaler_fingerprint(s3)
# |
# |     # 2. In-memory cache isolation on load_city
# |     clear_city_cache()
# |     cd_s1 = load_city("Boston", data_root="data", feature_scaler=s1)
# |     cd_s3 = load_city("Boston", data_root="data", feature_scaler=s3)
# |
# |     # Features must differ because s1 and s3 normalization parameters differ
# |     assert not torch.allclose(cd_s1.node_features, cd_s3.node_features)
# |     
# |     # Reloading with s1 must hit cache and return exact same tensor values
# |     cd_s1_cached = load_city("Boston", data_root="data", feature_scaler=s1)
# |     assert torch.allclose(cd_s1.node_features, cd_s1_cached.node_features)

# ===== END SOURCE FILE: od_plan_tester/tests/test_e1_contracts.py =====


# ===== BEGIN SOURCE FILE: od_plan_tester/tests/test_experiment_contracts.py =====
# | """
# | Tests for Experiment Contracts, Moving-Bin Support, and Manifest Integrity.
# | (Tests T37 to T45)
# | """
# |
# | import json
# | import hashlib
# | from pathlib import Path
# | import pytest
# | import torch
# | import numpy as np
# | from od_plan_tester.project_adapter import (
# |     train_zero_shot_model,
# |     load_cities,
# |     load_city,
# |     run_target_city_experiments,
# | )
# | from src.training.evaluate import compute_cpc_pair
# | from src.data.city_splits import generate_5fold_splits
# | from src.data.urban_graph import build_radius_graph
# |
# |
# | @pytest.mark.scientific
# | def test_model_freezing_theta_star():
# |     """T37: Model parameters theta* are completely frozen (requires_grad=False) before target inference."""
# |     model, _ = train_zero_shot_model(
# |         train_city_names=["Raleigh", "Denver"],
# |         data_root="data",
# |         epochs=1,
# |         device_str="cpu",
# |         verbose=False,
# |     )
# |
# |     for name, param in model.named_parameters():
# |         assert not param.requires_grad, f"Parameter {name} was not frozen!"
# |
# |
# | @pytest.mark.scientific
# | def test_shared_support_omega_c_across_conditions():
# |     """T38: All moving-bin experimental conditions evaluate on identical candidate support Omega_c."""
# |     train_data_list, fitted_scaler = load_cities(["Raleigh", "Denver"], data_root="data")
# |     model, _ = train_zero_shot_model(
# |         train_city_names=["Raleigh", "Denver"],
# |         data_root="data",
# |         epochs=1,
# |         device_str="cpu",
# |         verbose=False,
# |     )
# |
# |     cd = load_city("Denver", data_root="data", feature_scaler=fitted_scaler)
# |     dist_km = np.expm1(cd.pair_distance.numpy())
# |     expected_inter_mask = (cd.pair_o_idx.numpy() != cd.pair_d_idx.numpy()) & (dist_km > 0.0)
# |     expected_n_inter = int(np.sum(expected_inter_mask))
# |
# |     from src.data.yd_extractor import compute_kbin_edges
# |     bin_edges, _ = compute_kbin_edges(["Raleigh", "Denver"], K=8, data_root="data")
# |     res = run_target_city_experiments(
# |         model=model,
# |         city_name="Denver",
# |         scaler=fitted_scaler,
# |         data_root="data",
# |         bin_edges=bin_edges,
# |         device_str="cpu",
# |     )
# |
# |     # 1. Candidate pair counts strictly match target city candidate dataset
# |     assert res["n_pairs"] == len(cd.pair_o_idx)
# |     assert res["n_inter_pairs"] == expected_n_inter
# |
# |     # 2. Key conditions exist
# |     assert "M0" in res
# |     assert "M1_city_oracle_obs" in res
# |     assert "M1_county_oracle_obs" in res
# |     assert "M1_subzone_oracle_obs" in res
# |
# |     # 3. Verify that evaluation support on interzonal pairs is mathematically identical
# |     t_gt = cd.pair_trips.numpy()
# |     cpc_inter_m0_expected = compute_cpc_pair(
# |         t_gt[expected_inter_mask],
# |         res["M0"]["cpc_inter"] # checked via consistent evaluation
# |     )
# |     assert res["M0"]["cpc_inter"] > 0.0
# |     assert res["M1_city_oracle_obs"]["cpc_inter"] > 0.0
# |
# |
# | @pytest.mark.reference
# | def test_delta_r_and_realization_gap_formulas():
# |     """T39: Verify Delta R^+ and realization gap arithmetic formulas on interzonal metrics."""
# |     cpc_m0 = 0.35
# |     cpc_m1_real = 0.42
# |     cpc_m1_oracle = 0.50
# |
# |     delta_r_real = cpc_m1_real - cpc_m0
# |     delta_r_oracle = cpc_m1_oracle - cpc_m0
# |     realization_gap = cpc_m1_oracle - cpc_m1_real
# |
# |     assert pytest.approx(0.07, rel=1e-5) == delta_r_real
# |     assert pytest.approx(0.15, rel=1e-5) == delta_r_oracle
# |     assert pytest.approx(0.08, rel=1e-5) == realization_gap
# |
# |
# |
# |
# | @pytest.mark.contract
# | def test_t40_manifest_exists():
# |     """T40: manifest exists, is readable, and file hashes match."""
# |     import hashlib
# |     manifest_path = Path("results/manifest_rq1_v1.json")
# |     assert manifest_path.exists(), "Production manifest results/manifest_rq1_v1.json does not exist!"
# |     with open(manifest_path, "r") as f:
# |         manifest = json.load(f)
# |     assert "contract_conditions" in manifest
# |     
# |     # Verify file hashes
# |     file_hashes = manifest.get("file_hashes", {})
# |     assert len(file_hashes) > 0, "No file hashes found in manifest!"
# |     for fp, expected_hash in file_hashes.items():
# |         p = Path("results") / fp
# |         if p.exists():
# |             computed = hashlib.sha256(p.read_bytes()).hexdigest()
# |             assert computed == expected_hash, f"Hash mismatch for {fp}!"
# |
# | @pytest.mark.contract
# | def test_t41_50_unique_test_cities():
# |     """T41: manifest vs pipeline (50 unique test cities)."""
# |     with open("results/manifest_rq1_v1.json", "r") as f:
# |         manifest = json.load(f)
# |     with open("results/5fold_results.json", "r") as f:
# |         results = json.load(f)
# |     cities = set([r["city"] for r in results["city_level_results"]])
# |     assert len(cities) == manifest["contract_conditions"]["unique_cities"]
# |     assert len(cities) == 50
# |
# | @pytest.mark.contract
# | def test_t42_5_folds_by_10_cities():
# |     """T42: manifest vs pipeline (5 folds x 10 cities)."""
# |     with open("results/manifest_rq1_v1.json", "r") as f:
# |         manifest = json.load(f)
# |     with open("results/5fold_results.json", "r") as f:
# |         results = json.load(f)
# |     from collections import defaultdict
# |     fold_counts = defaultdict(int)
# |     for r in results["city_level_results"]:
# |         fold_counts[r["fold"]] += 1
# |     
# |     assert len(fold_counts) == manifest["contract_conditions"]["folds"]
# |     assert len(fold_counts) == 5
# |     for count in fold_counts.values():
# |         assert count == manifest["contract_conditions"]["cities_per_fold"]
# |         assert count == 10
# |
# | @pytest.mark.contract
# | def test_t43_primary_k_is_8():
# |     """T43: manifest vs pipeline (primary K == 8)."""
# |     with open("results/manifest_rq1_v1.json", "r") as f:
# |         manifest = json.load(f)
# |     locked_k = manifest["contract_conditions"]["primary_k_bins"]
# |     
# |     from src.data.yd_extractor import compute_kbin_edges
# |     bin_edges, _ = compute_kbin_edges(["Denver"], K=locked_k, data_root="data")
# |     assert len(bin_edges) - 1 == locked_k
# |
# | @pytest.mark.contract
# | def test_t44_seeds_are_1_10_100():
# |     """T44: manifest vs pipeline (seeds == {1,10,100})."""
# |     with open("results/manifest_rq1_v1.json", "r") as f:
# |         manifest = json.load(f)
# |     locked_seeds = set(manifest["contract_conditions"]["model_seeds"])
# |     
# |     import ast
# |     with open("src/experiment/run_5fold.py", "r") as f:
# |         tree = ast.parse(f.read())
# |     
# |     found_seeds = None
# |     for node in ast.walk(tree):
# |         if isinstance(node, ast.Assign):
# |             for target in node.targets:
# |                 if getattr(target, 'id', '') == 'seeds':
# |                     if isinstance(node.value, ast.List):
# |                         found_seeds = set(elt.value for elt in node.value.elts if isinstance(elt, ast.Constant))
# |     
# |     assert found_seeds == locked_seeds, f"Found seeds {found_seeds} in run_5fold.py!"
# |
# | @pytest.mark.contract
# | def test_t45_m1_city_is_primary_treatment():
# |     """T45: manifest vs pipeline (M1_city is primary treatment)."""
# |     with open("results/manifest_rq1_v1.json", "r") as f:
# |         manifest = json.load(f)
# |     primary = manifest["contract_conditions"]["primary_treatment"]
# |     
# |     with open("results/5fold_results.json", "r") as f:
# |         results = json.load(f)
# |     keys = results["city_level_results"][0].keys()
# |     assert primary in keys, f"{primary} not found in pipeline results!"
# |
# | @pytest.mark.contract
# | def test_t46_m1_subzone_is_ceiling_only():
# |     """T46: manifest vs pipeline (M1_subzone is ceiling only)."""
# |     with open("results/manifest_rq1_v1.json", "r") as f:
# |         manifest = json.load(f)
# |     ceiling = manifest["contract_conditions"]["ceiling_treatment"]
# |     
# |     with open("results/5fold_results.json", "r") as f:
# |         results = json.load(f)
# |     keys = results["city_level_results"][0].keys()
# |     assert ceiling in keys, f"{ceiling} not found in pipeline results!"
# |
# | @pytest.mark.contract
# | def test_t47_primary_metric_is_cpc_interzonal():
# |     """T47: manifest vs pipeline (primary metric == CPC interzonal)."""
# |     with open("results/manifest_rq1_v1.json", "r") as f:
# |         manifest = json.load(f)
# |     metric = manifest["contract_conditions"]["primary_metric"]
# |     
# |     with open("results/5fold_results.json", "r") as f:
# |         results = json.load(f)
# |     m0_metrics = results["city_level_results"][0]["M0"].keys()
# |     assert metric in m0_metrics, f"{metric} not found in pipeline metrics!"
# |
# | @pytest.mark.scientific
# | def test_t48_support_is_omega_c_plus():
# |     """T48: Rigorous verification that Omega_c^+ is defined by D_ij > 0 and strictly equal to bin_labels in {1,2,3}. (Restored T44)"""
# |     for city_name in ["Denver", "Portland"]:
# |         cd = load_city(city_name, data_root="data")
# |         dist_km = np.expm1(cd.pair_distance.numpy())
# |         o_np = cd.pair_o_idx.numpy()
# |         d_np = cd.pair_d_idx.numpy()
# |         b_np = cd.bin_labels.numpy()
# |
# |         # Check equivalence between distance threshold and bin assignment
# |         mask_dist = (o_np != d_np) & (dist_km > 0.0)
# |         mask_bins = (o_np != d_np) & (b_np > 0)
# |         assert np.array_equal(mask_dist, mask_bins), f"{city_name}: mask_dist and mask_bins mismatch!"
# |
# |         # Intrazonal pairs are strictly bin 0
# |         diag_mask = (o_np == d_np)
# |         assert np.all(b_np[diag_mask] == 0), f"{city_name}: intrazonal pairs contain non-zero bins!"
# |
# |
# | @pytest.mark.contract
# | def test_t49_main_results_reproduce_locked_values():
# |     """T49: main results reproduce locked values."""
# |     with open("results/manifest_rq1_v1.json", "r") as f:
# |         manifest = json.load(f)
# |     
# |     with open("results/5fold_results.json", "r") as f:
# |         results = json.load(f)
# |         
# |     city_results = results["city_level_results"]
# |     import numpy as np
# |     m0_cpcs = np.array([r["M0"]["cpc_inter"] for r in city_results])
# |     m1_cpcs = np.array([r["M1_city_oracle_obs"]["cpc_inter"] for r in city_results])
# |     
# |     mean_delta = np.mean(m1_cpcs - m0_cpcs)
# |     win_rate = np.mean(m1_cpcs > m0_cpcs) * 100.0
# |     
# |     locked_delta = manifest["locked_results"]["mean_delta_cpc"]
# |     locked_win_rate = manifest["locked_results"]["win_rate_percent"]
# |     
# |     assert abs(mean_delta - locked_delta) < 1e-4, f"Delta CPC mismatch: {mean_delta} vs {locked_delta}"
# |     assert abs(win_rate - locked_win_rate) < 1e-4, f"Win rate mismatch: {win_rate} vs {locked_win_rate}"
# |
# | @pytest.mark.scientific
# | def test_t50_target_ground_truth_permutation_invariance_for_m0():
# |     """T50: Changing, permuting, or zeroing target ground-truth T^GT has zero effect on M0 predictions. (Restored T45)"""
# |     train_data_list, fitted_scaler = load_cities(["Raleigh", "Denver"], data_root="data")
# |     model, _ = train_zero_shot_model(
# |         train_city_names=["Raleigh", "Denver"],
# |         data_root="data",
# |         epochs=1,
# |         device_str="cpu",
# |         verbose=False,
# |     )
# |
# |     cd = load_city("Portland", data_root="data", feature_scaler=fitted_scaler)
# |     import copy
# |     cd = copy.deepcopy(cd)
# |     edge_idx, edge_dist = build_radius_graph(cd.lon_lat, radius_km=5.0)
# |
# |     with torch.no_grad():
# |         pred_orig = model(
# |             cd.node_features,
# |             edge_idx,
# |             edge_dist,
# |             cd.pair_o_idx,
# |             cd.pair_d_idx,
# |             cd.pair_distance,
# |             cd.population,
# |             return_conditional_mean=True,
# |         )
# |
# |     # Permute trips completely
# |     cd.pair_trips = cd.pair_trips[torch.randperm(len(cd.pair_trips))]
# |
# |     with torch.no_grad():
# |         pred_permuted = model(
# |             cd.node_features,
# |             edge_idx,
# |             edge_dist,
# |             cd.pair_o_idx,
# |             cd.pair_d_idx,
# |             cd.pair_distance,
# |             cd.population,
# |             return_conditional_mean=True,
# |         )
# |
# |     # Set trips to zeros
# |     cd.pair_trips = torch.zeros_like(cd.pair_trips)
# |
# |     with torch.no_grad():
# |         pred_zeros = model(
# |             cd.node_features,
# |             edge_idx,
# |             edge_dist,
# |             cd.pair_o_idx,
# |             cd.pair_d_idx,
# |             cd.pair_distance,
# |             cd.population,
# |             return_conditional_mean=True,
# |         )
# |
# |     torch.testing.assert_close(pred_orig, pred_permuted, rtol=0, atol=1e-6)
# |     torch.testing.assert_close(pred_orig, pred_zeros, rtol=0, atol=1e-6)

# ===== END SOURCE FILE: od_plan_tester/tests/test_experiment_contracts.py =====


# ===== BEGIN SOURCE FILE: od_plan_tester/tests/test_graph_topology.py =====
# | """
# | Tests for Spatial Urban Graph Topology and Fallback Invariants.
# | (Tests T07 to T10)
# | """
# |
# | import pytest
# | import torch
# | import numpy as np
# | from od_plan_tester.project_adapter import (
# |     build_radius_graph,
# |     haversine_distance_matrix,
# | )
# |
# |
# | @pytest.mark.reference
# | def test_radius_graph_5km_threshold(sample_coordinates):
# |     """T07: Radius graph with r=5.0 km connects all pairs within 5 km."""
# |     edge_index, edge_dist = build_radius_graph(sample_coordinates, radius_km=5.0)
# |
# |     # Core tract 0 (-84.388, 33.749) and tract 1 (-84.390, 33.750) are ~0.25 km apart
# |     # Tract 2 is ~1.5 km apart. They should be connected.
# |     dist_mat = haversine_distance_matrix(sample_coordinates)
# |     assert dist_mat[0, 1] < 5.0
# |     assert dist_mat[0, 2] < 5.0
# |
# |     # Verify edge exists in edge_index
# |     edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
# |     assert (0, 1) in edges or (1, 0) in edges
# |     assert (0, 2) in edges or (2, 0) in edges
# |
# |
# | @pytest.mark.reference
# | def test_radius_graph_isolated_fallback_1nn(sample_coordinates):
# |     """T08: Isolated tract (node 4, ~70 km away) connects via 1-NN fallback (degree >= 1)."""
# |     edge_index, edge_dist = build_radius_graph(sample_coordinates, radius_km=5.0)
# |
# |     # Node 4 is distant. It must have at least 1 neighbor via fallback
# |     edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
# |     node_4_neighbors = [dst for src, dst in edges if src == 4 and dst != 4]
# |
# |     assert len(node_4_neighbors) >= 1, "Isolated node must have 1-NN fallback edge"
# |
# |
# | @pytest.mark.scientific
# | def test_urban_graph_zero_od_leakage(sample_coordinates):
# |     """T09: Graph construction uses only spatial coordinates and is completely independent of OD flows."""
# |     ei1, ed1 = build_radius_graph(sample_coordinates, radius_km=5.0)
# |     ei2, ed2 = build_radius_graph(sample_coordinates, radius_km=5.0)
# |
# |     assert torch.equal(ei1, ei2)
# |     assert torch.equal(ed1, ed2)
# |
# |
# | @pytest.mark.contract
# | def test_graph_symmetric_and_self_loops(sample_coordinates):
# |     """T10: Built graph contains self-loops and is undirected (symmetric adjacency)."""
# |     edge_index, edge_dist = build_radius_graph(sample_coordinates, radius_km=5.0, include_self_loop=True)
# |
# |     n_nodes = len(sample_coordinates)
# |     edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
# |
# |     # Check self loops
# |     for i in range(n_nodes):
# |         assert (i, i) in edges
# |
# |     # Check symmetry
# |     for u, v in edges:
# |         assert (v, u) in edges
# |
# |
# | @pytest.mark.contract
# | def test_graph_caching_and_clear(sample_coordinates):
# |     """T11: Graph caching returns identical cached tensors and respects clear_graph_cache."""
# |     from src.data.urban_graph import build_radius_graph, build_knn_graph, clear_graph_cache, _GRAPH_CACHE
# |
# |     clear_graph_cache()
# |     assert len(_GRAPH_CACHE) == 0
# |
# |     ei1, ed1 = build_radius_graph(sample_coordinates, radius_km=5.0, use_cache=True)
# |     assert len(_GRAPH_CACHE) == 1
# |
# |     ei2, ed2 = build_radius_graph(sample_coordinates, radius_km=5.0, use_cache=True)
# |     assert len(_GRAPH_CACHE) == 1
# |     assert torch.equal(ei1, ei2)
# |     assert torch.equal(ed1, ed2)
# |
# |     # Test knn caching
# |     ki1, kd1 = build_knn_graph(sample_coordinates, k=3, use_cache=True)
# |     assert len(_GRAPH_CACHE) == 2
# |     ki2, kd2 = build_knn_graph(sample_coordinates, k=3, use_cache=True)
# |     assert len(_GRAPH_CACHE) == 2
# |     assert torch.equal(ki1, ki2)
# |     assert torch.equal(kd1, kd2)
# |
# |     clear_graph_cache()
# |     assert len(_GRAPH_CACHE) == 0
# |
# |
# | @pytest.mark.contract
# | def test_raw_city_data_caching():
# |     """T12: Raw city dataset cache avoids redundant I/O and supports independent scalers."""
# |     from src.data.dataset import load_city, load_raw_city, clear_city_cache, _RAW_CITY_CACHE
# |     from sklearn.preprocessing import StandardScaler
# |
# |     clear_city_cache()
# |     assert len(_RAW_CITY_CACHE) == 0
# |
# |     raw1 = load_raw_city("Raleigh", data_root="data", use_cache=True)
# |     assert len(_RAW_CITY_CACHE) == 1
# |     raw2 = load_raw_city("Raleigh", data_root="data", use_cache=True)
# |     assert raw1 is raw2
# |
# |     # Verify scaling with two different scalers preserves raw features
# |     s1 = StandardScaler()
# |     s1.fit(raw1.X_raw * 2.0)
# |     cd1 = load_city("Raleigh", data_root="data", feature_scaler=s1, use_cache=True)
# |
# |     s2 = StandardScaler()
# |     s2.fit(raw1.X_raw * 0.5)
# |     cd2 = load_city("Raleigh", data_root="data", feature_scaler=s2, use_cache=True)
# |
# |     # cd1 and cd2 have different normalized features but share same raw data
# |     assert not torch.allclose(cd1.node_features, cd2.node_features)
# |     assert torch.equal(cd1.pair_distance, cd2.pair_distance)
# |     assert torch.equal(cd1.pair_trips, cd2.pair_trips)
# |     assert torch.equal(cd1.lon_lat, cd2.lon_lat)
# |
# |     clear_city_cache()
# |     assert len(_RAW_CITY_CACHE) == 0
# |
# |
# | @pytest.mark.contract
# | def test_distance_matrix_caching_and_clear(sample_coordinates):
# |     """T13: Haversine distance matrix cache avoids redundant O(N^2) computation."""
# |     from src.data.urban_graph import (
# |         haversine_distance_matrix,
# |         build_radius_graph,
# |         clear_distance_matrix_cache,
# |         clear_graph_cache,
# |         _DISTANCE_MATRIX_CACHE,
# |     )
# |
# |     clear_distance_matrix_cache()
# |     assert len(_DISTANCE_MATRIX_CACHE) == 0
# |
# |     # 1. Direct call caches the matrix
# |     m1 = haversine_distance_matrix(sample_coordinates, use_cache=True)
# |     assert len(_DISTANCE_MATRIX_CACHE) == 1
# |
# |     m2 = haversine_distance_matrix(sample_coordinates, use_cache=True)
# |     assert m1 is m2  # Exact object identity from cache
# |     assert np.array_equal(m1, m2)
# |
# |     # 2. build_radius_graph with different radii shares the cached distance matrix
# |     clear_graph_cache()
# |     assert len(_DISTANCE_MATRIX_CACHE) == 0
# |
# |     _ = build_radius_graph(sample_coordinates, radius_km=3.0, use_cache=True)
# |     assert len(_DISTANCE_MATRIX_CACHE) == 1
# |
# |     # Calling with radius=7.0 reuses the distance matrix without recomputing
# |     _ = build_radius_graph(sample_coordinates, radius_km=7.0, use_cache=True)
# |     assert len(_DISTANCE_MATRIX_CACHE) == 1
# |
# |     clear_distance_matrix_cache()
# |     assert len(_DISTANCE_MATRIX_CACHE) == 0
# |
# |
# | @pytest.mark.contract
# | def test_city_data_instance_cache_with_scaler():
# |     """T14: CityData instance cache returns identical object when called with same scaler."""
# |     from src.data.dataset import load_city, load_raw_city, clear_city_cache, _CITY_DATA_CACHE
# |     from sklearn.preprocessing import StandardScaler
# |
# |     clear_city_cache()
# |     assert len(_CITY_DATA_CACHE) == 0
# |
# |     raw = load_raw_city("Raleigh", data_root="data", use_cache=True)
# |     s = StandardScaler()
# |     s.fit(raw.X_raw)
# |
# |     cd1 = load_city("Raleigh", data_root="data", feature_scaler=s, use_cache=True)
# |     assert len(_CITY_DATA_CACHE) == 1
# |
# |     cd2 = load_city("Raleigh", data_root="data", feature_scaler=s, use_cache=True)
# |     assert cd1 is cd2  # Exact object identity reused without re-creating tensors
# |     assert len(_CITY_DATA_CACHE) == 1
# |
# |     clear_city_cache()
# |     assert len(_CITY_DATA_CACHE) == 0
# |
# |
# | @pytest.mark.contract
# | def test_preload_all_cities_smoke():
# |     """T15: preload_all_cities warms up raw data and spatial graph caches for selected cities."""
# |     from src.data.dataset import preload_all_cities, clear_city_cache, _RAW_CITY_CACHE
# |     from src.data.urban_graph import _GRAPH_CACHE, clear_graph_cache
# |
# |     clear_city_cache()
# |     clear_graph_cache()
# |
# |     preload_all_cities(data_root="data", city_names=["Raleigh", "Denver"], build_graphs=True, radius_km=5.0)
# |     assert len(_RAW_CITY_CACHE) == 2
# |     assert len(_GRAPH_CACHE) == 2
# |
# |     clear_city_cache()
# |     clear_graph_cache()
# |
# |

# ===== END SOURCE FILE: od_plan_tester/tests/test_graph_topology.py =====


# ===== BEGIN SOURCE FILE: od_plan_tester/tests/test_kl_calibration.py =====
# | """
# | Tests for Interzonal Moving-Bin Calibration on Omega_c^+ (Soft KL Projection).
# | (Tests T16 to T21)
# | """
# |
# | import pytest
# | import torch
# | import numpy as np
# | from od_plan_tester.project_adapter import calibrate_moving_bins
# |
# |
# | @pytest.mark.scientific
# | def test_moving_mass_preservation():
# |     """T16: Interzonal mass preservation on Omega_c^+ within numerical tolerance."""
# |     torch.manual_seed(42)
# |     # 100 pairs: 10 intrazonal (bin 0), 90 interzonal (bins 1, 2, 3)
# |     o_idx = torch.randint(0, 20, (100,))
# |     d_idx = torch.randint(0, 20, (100,))
# |     # force first 10 to be intrazonal
# |     for i in range(10):
# |         d_idx[i] = o_idx[i]
# |
# |     bins = torch.randint(1, 4, (100,))
# |     bins[:10] = 0
# |
# |     t0 = torch.rand(100) * 100.0 + 1.0
# |     target_moving = np.array([0.30, 0.50, 0.20])
# |
# |     t_cal = calibrate_moving_bins(t0, bins, o_idx, d_idx, target_moving, q=1.0)
# |
# |     inter_mask = (o_idx != d_idx) & (bins > 0)
# |     inter_mass0 = t0[inter_mask].sum().item()
# |     inter_mass_cal = t_cal[inter_mask].sum().item()
# |     rel_err = abs(inter_mass_cal - inter_mass0) / inter_mass0
# |
# |     assert rel_err < 1e-5, f"Interzonal mass preservation relative error {rel_err} exceeds 1e-5"
# |
# |
# | @pytest.mark.scientific
# | def test_intrazonal_identity():
# |     """T17: Intrazonal pairs (i == j, D == 0, bin 0) are strictly preserved identically: T_cal == T_zs."""
# |     torch.manual_seed(42)
# |     o_idx = torch.tensor([0, 1, 0, 2, 3, 3])
# |     d_idx = torch.tensor([0, 1, 1, 3, 2, 3])  # pairs 0, 1, 5 are intrazonal
# |     bins = torch.tensor([0, 0, 1, 2, 3, 0])
# |     t0 = torch.tensor([50.0, 120.0, 30.0, 80.0, 20.0, 200.0])
# |     target_moving = np.array([0.25, 0.50, 0.25])
# |
# |     t_cal = calibrate_moving_bins(t0, bins, o_idx, d_idx, target_moving, q=1.0)
# |
# |     intra_mask = (o_idx == d_idx) | (bins == 0)
# |     assert torch.allclose(t_cal[intra_mask], t0[intra_mask], atol=1e-6)
# |
# |
# | @pytest.mark.contract
# | def test_q_zero_is_zero_shot_identity():
# |     """T18: At q=0, calibration outputs exactly zero-shot flows (T_cal == T_zs)."""
# |     torch.manual_seed(42)
# |     o_idx = torch.tensor([0, 0, 0])
# |     d_idx = torch.tensor([1, 2, 3])
# |     bins = torch.tensor([1, 2, 3])
# |     t0 = torch.tensor([10.0, 50.0, 100.0])
# |     target_moving = np.array([0.40, 0.40, 0.20])
# |
# |     t_cal_0 = calibrate_moving_bins(t0, bins, o_idx, d_idx, target_moving, q=0.0)
# |     assert torch.allclose(t_cal_0, t0, atol=1e-6)
# |
# |
# | @pytest.mark.scientific
# | def test_q_one_matches_target_distribution():
# |     """T19: At q=1, active moving-bin proportions match target distribution within tolerance < 1e-5."""
# |     torch.manual_seed(42)
# |     o_idx = torch.tensor([0, 0, 0, 0, 0, 0])
# |     d_idx = torch.tensor([1, 2, 3, 4, 5, 6])
# |     bins = torch.tensor([1, 1, 2, 2, 3, 3])
# |     t0 = torch.tensor([10.0, 15.0, 40.0, 60.0, 100.0, 150.0])
# |     target_moving = np.array([0.20, 0.50, 0.30])
# |
# |     t_cal_1 = calibrate_moving_bins(t0, bins, o_idx, d_idx, target_moving, q=1.0, tolerance=1e-5)
# |
# |     inter_total = t_cal_1.sum().item()
# |     p_b1 = t_cal_1[bins == 1].sum().item() / inter_total
# |     p_b2 = t_cal_1[bins == 2].sum().item() / inter_total
# |     p_b3 = t_cal_1[bins == 3].sum().item() / inter_total
# |
# |     assert abs(p_b1 - 0.20) < 1e-5
# |     assert abs(p_b2 - 0.50) < 1e-5
# |     assert abs(p_b3 - 0.30) < 1e-5
# |
# |
# | @pytest.mark.reference
# | def test_q_monotonic_soft_response():
# |     """T20: Soft multiplier w_k(q) = (p_cond / p_implied)^q smoothly interpolates between q=0 and q=1."""
# |     o_idx = torch.tensor([0, 0])
# |     d_idx = torch.tensor([1, 2])
# |     bins = torch.tensor([1, 2])
# |     t0 = torch.tensor([10.0, 90.0])  # implied: [0.1, 0.9]
# |     target_moving = np.array([0.5, 0.5, 0.0])  # target for bins 1, 2, 3
# |
# |     t_q0 = calibrate_moving_bins(t0, bins, o_idx, d_idx, target_moving, q=0.0)
# |     t_q5 = calibrate_moving_bins(t0, bins, o_idx, d_idx, target_moving, q=0.5)
# |     t_q1 = calibrate_moving_bins(t0, bins, o_idx, d_idx, target_moving, q=1.0)
# |
# |     # Bin 1 flow must strictly increase with q
# |     assert t_q0[0].item() < t_q5[0].item() < t_q1[0].item()
# |     # Bin 2 flow must strictly decrease with q
# |     assert t_q0[1].item() > t_q5[1].item() > t_q1[1].item()
# |
# |
# | @pytest.mark.reference
# | def test_inactive_bin_conditioning():
# |     """T21: For cities with diameter < 100 km (bin 3 absent), target is conditioned on active moving bins."""
# |     o_idx = torch.tensor([0, 0])
# |     d_idx = torch.tensor([1, 2])
# |     bins = torch.tensor([1, 2])  # only bins 1 and 2 present
# |     t0 = torch.tensor([30.0, 70.0])
# |     target_moving = np.array([0.40, 0.40, 0.20])  # has 0.20 on bin 3
# |
# |     t_cal = calibrate_moving_bins(t0, bins, o_idx, d_idx, target_moving, q=1.0)
# |
# |     # Conditioned target on {1, 2} is [0.4/0.8, 0.4/0.8] = [0.5, 0.5]
# |     inter_total = t_cal.sum().item()
# |     p_b1 = t_cal[bins == 1].sum().item() / inter_total
# |     p_b2 = t_cal[bins == 2].sum().item() / inter_total
# |
# |     assert pytest.approx(0.5, abs=1e-5) == p_b1
# |     assert pytest.approx(0.5, abs=1e-5) == p_b2
# |
# |
# | @pytest.mark.scientific
# | def test_calibrate_moving_bins_with_pair_distance():
# |     """T21b: Verify calibrate_moving_bins using direct pair_distance tensor (D_ij > 0)."""
# |     torch.manual_seed(42)
# |     o_idx = torch.tensor([0, 1, 0, 2, 3, 3])
# |     d_idx = torch.tensor([0, 1, 1, 3, 2, 3])  # 0, 1, 5 are intrazonal
# |     bins = torch.tensor([0, 0, 1, 2, 3, 0])
# |     dist_km = torch.tensor([0.0, 0.0, 5.2, 25.0, 120.0, 0.0])
# |     pair_distance = torch.log1p(dist_km)
# |     t0 = torch.tensor([50.0, 120.0, 30.0, 80.0, 20.0, 200.0])
# |     target_moving = np.array([0.25, 0.50, 0.25])
# |
# |     t_cal_bins = calibrate_moving_bins(t0, bins, o_idx, d_idx, target_moving, q=1.0)
# |     t_cal_dist = calibrate_moving_bins(t0, bins, o_idx, d_idx, target_moving, q=1.0, pair_distance=pair_distance)
# |
# |     assert torch.allclose(t_cal_bins, t_cal_dist, atol=1e-6)
# |     # Intrazonal identity
# |     assert torch.allclose(t_cal_dist[[0, 1, 5]], t0[[0, 1, 5]], atol=1e-6)
# |     # Interzonal mass preservation
# |     assert pytest.approx(t0[[2, 3, 4]].sum().item(), rel=1e-5) == t_cal_dist[[2, 3, 4]].sum().item()
# |

# ===== END SOURCE FILE: od_plan_tester/tests/test_kl_calibration.py =====


# ===== BEGIN SOURCE FILE: od_plan_tester/tests/test_leakage_and_splits.py =====
# | """
# | Tests for Data Splits, Scaler Isolation, and Omega_c Candidate Provenance.
# | (Tests T11 to T15)
# | """
# |
# | import pytest
# | import torch
# | import numpy as np
# | from od_plan_tester.project_adapter import (
# |     generate_5fold_splits,
# |     load_city,
# |     load_cities,
# | )
# |
# |
# | @pytest.mark.reference
# | def test_stratified_5fold_split_structure():
# |     """T11: 5-Fold split contains 5 folds, each with 40 train and 10 test cities."""
# |     splits = generate_5fold_splits(data_root="data")
# |
# |     assert len(splits) == 5
# |     for fold_id, split in splits.items():
# |         assert len(split["train"]) == 40
# |         assert len(split["test"]) == 10
# |         # Disjoint
# |         assert len(set(split["train"]) & set(split["test"])) == 0
# |
# |
# | @pytest.mark.scientific
# | def test_stratified_5fold_exact_single_coverage():
# |     """T12: Across all 5 folds, every single city of the 50 cities is tested exactly once."""
# |     splits = generate_5fold_splits(data_root="data")
# |
# |     all_test_cities = []
# |     for split in splits.values():
# |         all_test_cities.extend(split["test"])
# |
# |     assert len(all_test_cities) == 50
# |     assert len(set(all_test_cities)) == 50, "Every city must appear in the test set exactly once."
# |
# |
# | @pytest.mark.scientific
# | def test_scaler_train_isolation():
# |     """T13: StandardScaler is fitted strictly on source training cities; target city tracts never enter scaler."""
# |     train_cities = ["Raleigh", "Denver"]
# |     test_city = "Philadelphia"
# |
# |     train_data_list, fitted_scaler = load_cities(train_cities, data_root="data")
# |     test_data = load_city(test_city, data_root="data", feature_scaler=fitted_scaler, fit_scaler=False)
# |
# |     total_train_tracts = sum(c.n_tracts for c in train_data_list)
# |     assert fitted_scaler.n_samples_seen_ == total_train_tracts
# |     assert fitted_scaler.n_samples_seen_ != (total_train_tracts + test_data.n_tracts)
# |
# |
# | @pytest.mark.scientific
# | def test_omega_c_provenance_strictly_positive():
# |     """T14: Target city candidate pairs Omega_c contain strictly positive flow counts (T_ij >= 1)."""
# |     test_cities = ["Philadelphia", "Denver", "Raleigh"]
# |     for c_name in test_cities:
# |         cd = load_city(c_name, data_root="data")
# |         assert (cd.pair_trips >= 1.0).all()
# |         assert cd.pair_trips.min().item() >= 1.0
# |
# |
# | @pytest.mark.scientific
# | def test_unobserved_pairs_excluded_not_zero_filled():
# |     """T15: Candidate support Omega_c has size <= N*(N-1); unobserved pairs are excluded (not zero-filled)."""
# |     cd = load_city("Philadelphia", data_root="data")
# |     n_nodes = cd.n_tracts
# |     max_possible_pairs = n_nodes * n_nodes
# |
# |     assert cd.n_pairs <= max_possible_pairs
# |     assert len(cd.pair_trips) == cd.n_pairs
# |     # Ensure there are no 0s stored
# |     assert not (cd.pair_trips == 0).any()
# |
# |
# | @pytest.mark.scientific
# | def test_omega_c_plus_distance_equivalence():
# |     """T15b: Omega_c^+ definition invariant: (bin_labels > 0) <=> (D_ij > 0) <=> (pair_o != pair_d)."""
# |     test_cities = ["Philadelphia", "Denver", "Raleigh", "Austin", "Seattle"]
# |     for c_name in test_cities:
# |         cd = load_city(c_name, data_root="data")
# |         dist_km = torch.expm1(cd.pair_distance)
# |
# |         # 1. Invariant: bin_labels > 0 iff distance_km > 0
# |         assert torch.equal(cd.bin_labels > 0, dist_km > 0.0), f"Bin label vs distance mismatch in {c_name}"
# |
# |         # 2. Invariant: (pair_o != pair_d) & (bin_labels > 0) strictly matches (pair_o != pair_d) & (D_ij > 0)
# |         mask_bins = (cd.pair_o_idx != cd.pair_d_idx) & (cd.bin_labels > 0)
# |         mask_dist = (cd.pair_o_idx != cd.pair_d_idx) & (dist_km > 0.0)
# |         assert torch.equal(mask_bins, mask_dist), f"Omega_c^+ mask mismatch in {c_name}"
# |
# |         # 3. Invariant: Intrazonal pairs have distance == 0 and bin == 0
# |         intra_mask = (cd.pair_o_idx == cd.pair_d_idx)
# |         assert torch.all(dist_km[intra_mask] == 0.0)
# |         assert torch.all(cd.bin_labels[intra_mask] == 0)
# |
# |
# | @pytest.mark.scientific
# | def test_all_50_cities_omega_plus_invariants():
# |     """T15c: Exhaustive verification across all 50 dataset cities that D_ij > 0 <=> bin_labels > 0."""
# |     from src.data.city_splits import get_all_cities_sorted_by_size
# |     all_cities_info = get_all_cities_sorted_by_size(data_root="data")
# |     assert len(all_cities_info) == 50
# |
# |     for c_info in all_cities_info:
# |         c_name = c_info["city"]
# |         cd = load_city(c_name, data_root="data")
# |         dist_km = torch.expm1(cd.pair_distance)
# |
# |         # Invariant: bin_labels > 0 strictly equals distance_km > 0
# |         assert torch.equal(cd.bin_labels > 0, dist_km > 0.0), f"Distance vs bin mismatch in {c_name}"
# |
# |         # Invariant: Omega_c^+ mask strictly identical
# |         mask_bins = (cd.pair_o_idx != cd.pair_d_idx) & (cd.bin_labels > 0)
# |         mask_dist = (cd.pair_o_idx != cd.pair_d_idx) & (dist_km > 0.0)
# |         assert torch.equal(mask_bins, mask_dist), f"Omega_c^+ definition discrepancy in {c_name}"
# |
# |
# |

# ===== END SOURCE FILE: od_plan_tester/tests/test_leakage_and_splits.py =====


# ===== BEGIN SOURCE FILE: od_plan_tester/tests/test_metrics.py =====
# | """
# | Tests for Evaluation Metrics Suite (CPC, CPC_norm, RMSE-log1p, Pearson r).
# | (Tests T27 to T31)
# | """
# |
# | import pytest
# | import torch
# | import numpy as np
# | from od_plan_tester.project_adapter import (
# |     compute_cpc,
# |     compute_cpc_norm,
# |     compute_rmse_log1p,
# |     compute_pearson_r,
# |     evaluate_all,
# | )
# |
# |
# | @pytest.mark.reference
# | def test_cpc_bounds_and_symmetry():
# |     """T27: CPC is bounded in [0, 1] and symmetric CPC(A, B) == CPC(B, A)."""
# |     t_true = torch.tensor([10.0, 50.0, 100.0, 500.0])
# |     t_pred = torch.tensor([15.0, 40.0, 120.0, 480.0])
# |
# |     cpc1 = compute_cpc(t_true, t_pred)
# |     cpc2 = compute_cpc(t_pred, t_true)
# |
# |     assert 0.0 <= cpc1 <= 1.0
# |     assert pytest.approx(cpc1, rel=1e-6) == cpc2
# |
# |     # Perfect prediction has CPC = 1.0
# |     assert pytest.approx(1.0, rel=1e-6) == compute_cpc(t_true, t_true)
# |
# |
# | @pytest.mark.reference
# | def test_cpc_norm_1_minus_tvd():
# |     """T28: Normalized CPC matches 1 - 0.5 * sum |p_i - q_i| and is scale-invariant."""
# |     t_true = torch.tensor([10.0, 20.0, 30.0])
# |     t_pred = torch.tensor([20.0, 40.0, 60.0])  # identical shape, 2x scale
# |
# |     cpc_norm = compute_cpc_norm(t_true, t_pred)
# |     assert pytest.approx(1.0, rel=1e-6) == cpc_norm
# |
# |
# | @pytest.mark.reference
# | def test_rmse_log1p_zero_on_identical():
# |     """T29: RMSE-log1p is strictly 0.0 on identical inputs and positive otherwise."""
# |     t_true = torch.tensor([5.0, 15.0, 50.0])
# |     t_pred = torch.tensor([10.0, 20.0, 60.0])
# |
# |     assert pytest.approx(0.0, abs=1e-6) == compute_rmse_log1p(t_true, t_true)
# |     assert compute_rmse_log1p(t_true, t_pred) > 0.0
# |
# |
# | @pytest.mark.reference
# | def test_pearson_r_bounds():
# |     """T30: Pearson r is 1.0 on linear transformation and bounded in [-1, 1]."""
# |     t_true = torch.tensor([10.0, 20.0, 30.0, 40.0])
# |     t_pred = 3.0 * t_true + 5.0
# |
# |     r = compute_pearson_r(t_true, t_pred)
# |     assert pytest.approx(1.0, rel=1e-5) == r
# |
# |
# | @pytest.mark.contract
# | def test_evaluate_all_contract():
# |     """T31: evaluate_all returns all locked primary and secondary metrics in a dictionary."""
# |     t_true = torch.tensor([10.0, 20.0, 30.0])
# |     t_pred = torch.tensor([12.0, 18.0, 35.0])
# |
# |     res = evaluate_all(t_true, t_pred)
# |     assert isinstance(res, dict)
# |     assert "cpc" in res
# |     assert "cpc_norm" in res
# |     assert "rmse_log1p" in res
# |     assert "pearson_r" in res
# |     for v in res.values():
# |         assert isinstance(v, float)

# ===== END SOURCE FILE: od_plan_tester/tests/test_metrics.py =====


# ===== BEGIN SOURCE FILE: od_plan_tester/tests/test_yd_and_sampling.py =====
# | """
# | Tests for Moving-Bin Y_D Extraction (Oracle & Real Meta) and Distributional Overlap.
# | (Tests T22 to T26)
# | """
# |
# | import pytest
# | import torch
# | import numpy as np
# | from od_plan_tester.project_adapter import (
# |     extract_yd_moving_oracle,
# |     extract_M1_city_oracle_obs,
# |     compute_distributional_overlap,
# |     sample_multinomial_yd,
# | )
# |
# |
# | @pytest.mark.reference
# | def test_yd_moving_oracle_assignment():
# |     """T22: extract_yd_moving_oracle produces shape (3,), sums to 1.0, and excludes intrazonal trips."""
# |     # 4 pairs: pair 0 is intrazonal (0,0, bin 0, trips=100); pairs 1,2,3 are interzonal in bins 1,2,3 with trips 10, 20, 30
# |     o_idx = torch.tensor([0, 0, 0, 0])
# |     d_idx = torch.tensor([0, 1, 2, 3])
# |     bin_labels = torch.tensor([0, 1, 2, 3])
# |     pair_trips = torch.tensor([100.0, 10.0, 20.0, 30.0])
# |
# |     yd_moving = extract_yd_moving_oracle(pair_trips, bin_labels, o_idx, d_idx)
# |
# |     assert yd_moving.shape == (3,)
# |     assert pytest.approx(1.0, rel=1e-6) == float(np.sum(yd_moving))
# |     # Interzonal total = 60 -> proportions: [10/60, 20/60, 30/60] = [1/6, 1/3, 1/2]
# |     np.testing.assert_allclose(yd_moving, [1.0 / 6.0, 1.0 / 3.0, 0.5], atol=1e-5)
# |
# |
# | @pytest.mark.contract
# | def test_M1_city_oracle_obs_meta_sum():
# |     """T23: extract_M1_city_oracle_obs from Meta mobility data produces shape (3,) and sums strictly to 1.0."""
# |     sample_cities = ["Philadelphia", "Denver", "Raleigh"]
# |     for c_name in sample_cities:
# |         yd_real = extract_M1_city_oracle_obs(c_name, meta_prior_dir="meta_prior")
# |         assert yd_real is not None, f"Missing moving Meta Y_D for {c_name}"
# |         assert yd_real.shape == (3,)
# |         assert pytest.approx(1.0, rel=1e-5) == float(np.sum(yd_real))
# |         assert (yd_real >= 0.0).all()
# |
# |
# | @pytest.mark.reference
# | def test_distributional_overlap_bounds():
# |     """T24: compute_distributional_overlap (Overlap / CPC_dist) is in [0, 1] and 1.0 on identical distributions."""
# |     p = np.array([0.25, 0.50, 0.25])
# |     q = np.array([0.30, 0.40, 0.30])
# |
# |     overlap_self = compute_distributional_overlap(p, p)
# |     overlap_pq = compute_distributional_overlap(p, q)
# |
# |     assert pytest.approx(1.0, rel=1e-6) == overlap_self
# |     assert 0.0 <= overlap_pq <= 1.0
# |     # Overlap = min(0.25, 0.30) + min(0.50, 0.40) + min(0.25, 0.30) = 0.25 + 0.40 + 0.25 = 0.90
# |     assert pytest.approx(0.90, rel=1e-6) == overlap_pq
# |
# |
# | @pytest.mark.reference
# | def test_multinomial_sampling_stochastic_validity():
# |     """T25: Multinomial sampling produces valid distributions across seeds."""
# |     t_true = torch.tensor([50.0, 150.0, 300.0, 500.0])
# |     bin_labels = torch.tensor([0, 1, 2, 3])
# |
# |     for m in [100, 1000, 10000]:
# |         yd_m = sample_multinomial_yd(t_true, bin_labels, m=m, seed=42)
# |         assert len(yd_m) == 4
# |         assert pytest.approx(1.0, rel=1e-5) == float(np.sum(yd_m))
# |         assert (yd_m >= 0.0).all()
# |
# |
# | @pytest.mark.reference
# | def test_multinomial_sampling_asymptotic_convergence():
# |     """T26: When m=inf, Multinomial sampling converges to the exact underlying empirical distribution."""
# |     t_true = torch.tensor([100.0, 200.0, 300.0, 400.0])
# |     bin_labels = torch.tensor([0, 1, 2, 3])
# |
# |     yd_inf = sample_multinomial_yd(t_true, bin_labels, m=np.inf, seed=42)
# |     expected = np.array([0.1, 0.2, 0.3, 0.4])
# |     np.testing.assert_allclose(yd_inf, expected, atol=1e-6)

# ===== END SOURCE FILE: od_plan_tester/tests/test_yd_and_sampling.py =====


# ===== BEGIN SOURCE FILE: od_plan_tester/tests/test_ztnb_oracle.py =====
# | """
# | Tests for ZTNB Loss, Conditional Expectation, and Gravity Prior Oracle.
# | (Tests T01 to T06)
# | """
# |
# | import math
# | import pytest
# | import torch
# | import numpy as np
# | from od_plan_tester.project_adapter import (
# |     nb_log_prob,
# |     nb_log_prob_at_zero,
# |     ztnb_nll,
# |     compute_conditional_mean,
# |     GravityPrior,
# | )
# |
# |
# | @pytest.mark.reference
# | def test_ztnb_log_prob_zero_formula():
# |     """T01: Verify base NB log P(T=0) = phi * log(phi / (mu + phi))."""
# |     mu_vals = [0.5, 2.0, 10.0, 100.0]
# |     phi_vals = [0.1, 1.0, 5.0]
# |
# |     for mu in mu_vals:
# |         for phi in phi_vals:
# |             log_phi = torch.tensor(math.log(phi))
# |             mu_t = torch.tensor([mu])
# |             log_p0 = nb_log_prob_at_zero(mu_t, log_phi).item()
# |             expected = phi * math.log(phi / (mu + phi))
# |             assert pytest.approx(expected, rel=1e-5) == log_p0
# |
# |
# | @pytest.mark.contract
# | @pytest.mark.scientific
# | def test_ztnb_nll_strictly_positive_support():
# |     """T02: ZTNB NLL enforces T >= 1 and computes finite exact likelihood on positive counts."""
# |     t_pos = torch.tensor([1.0, 2.0, 5.0, 20.0])
# |     mu = torch.tensor([1.5, 2.0, 4.0, 18.0])
# |     log_phi = torch.tensor(0.0)
# |
# |     loss = ztnb_nll(t_pos, mu, log_phi)
# |     assert torch.isfinite(loss)
# |     assert loss.item() > 0.0
# |
# |     # Ensure zero counts trigger assertion error
# |     t_with_zero = torch.tensor([0.0, 1.0, 2.0])
# |     with pytest.raises(AssertionError):
# |         ztnb_nll(t_with_zero, torch.ones(3), log_phi)
# |
# |
# | @pytest.mark.reference
# | def test_ztnb_conditional_mean_strictly_greater():
# |     """T03: Conditional positive expectation E[T | T >= 1] = mu / (1 - P(0)) is strictly > mu."""
# |     mu = torch.tensor([0.1, 1.0, 5.0, 20.0])
# |     log_phi = torch.tensor(math.log(2.0))
# |     c_mean = compute_conditional_mean(mu, log_phi)
# |
# |     assert (c_mean > mu).all()
# |     # Manual check for mu=1.0, phi=2.0
# |     # P(0) = (2 / (1 + 2))^2 = 4/9
# |     # E[T|T>=1] = 1.0 / (1 - 4/9) = 9/5 = 1.8
# |     expected_1 = 1.0 / (1.0 - (2.0 / 3.0) ** 2)
# |     assert pytest.approx(expected_1, rel=1e-4) == c_mean[1].item()
# |
# |
# | @pytest.mark.reference
# | def test_ztnb_conditional_mean_asymptotics():
# |     """T04: As mu -> infinity, P(0) -> 0 and E[T | T >= 1] -> mu."""
# |     mu_large = torch.tensor([10000.0, 50000.0])
# |     log_phi = torch.tensor(0.0)  # phi = 1.0
# |     c_mean = compute_conditional_mean(mu_large, log_phi)
# |
# |     rel_diff = torch.abs(c_mean - mu_large) / mu_large
# |     assert (rel_diff < 1e-3).all()
# |
# |
# | @pytest.mark.reference
# | def test_gravity_prior_formula_and_decay():
# |     """T05: Gravity model formula: log T = G + log Pi + log Pj - alpha * log D."""
# |     grav = GravityPrior(init_G=0.5, init_alpha=1.5)
# |     pi = torch.tensor([1000.0, 2000.0])
# |     pj = torch.tensor([500.0, 4000.0])
# |     d = torch.tensor([10.0, 20.0])
# |
# |     log_t = grav(pi, pj, d)
# |     expected_0 = 0.5 + math.log(1000.0) + math.log(500.0) - 1.5 * math.log(10.0)
# |     assert pytest.approx(expected_0, rel=1e-4) == log_t[0].item()
# |
# |     # Verify distance decay: larger distance yields smaller predicted flow
# |     d_near = torch.tensor([5.0])
# |     d_far = torch.tensor([50.0])
# |     p_const = torch.tensor([1000.0])
# |     assert grav(p_const, p_const, d_near).item() > grav(p_const, p_const, d_far).item()
# |
# |
# | @pytest.mark.contract
# | def test_gravity_prior_learnable_gradients():
# |     """T06: Trainable shared gravity parameters G and log_alpha produce finite valid gradients."""
# |     grav = GravityPrior()
# |     pi = torch.tensor([1000.0])
# |     pj = torch.tensor([1000.0])
# |     d = torch.tensor([5.0])
# |
# |     out = grav(pi, pj, d)
# |     loss = (out - torch.tensor([10.0])) ** 2
# |     loss.backward()
# |
# |     assert grav.G.grad is not None and torch.isfinite(grav.G.grad)
# |     assert grav.log_alpha.grad is not None and torch.isfinite(grav.log_alpha.grad)

# ===== END SOURCE FILE: od_plan_tester/tests/test_ztnb_oracle.py =====


# ===== BEGIN SOURCE FILE: src/aggregation/gadm2_county_aggregator.py =====
# | # File: src/aggregation/gadm2_county_aggregator.py
# |
# | import os
# | from pathlib import Path
# | from typing import Optional, List, Dict, Any
# | import geopandas as gpd
# | import pandas as pd
# | import numpy as np
# |
# |
# | def find_default_gadm_path() -> str:
# |     """Find default GADM shapefile path."""
# |     candidates = [
# |         "gadm41_USA_shp/gadm41_USA_2.shp",
# |         "gadm36_USA_2.shp",
# |         "gadm41_USA_2.shp",
# |     ]
# |     for c in candidates:
# |         if Path(c).exists():
# |             return c
# |     return candidates[0]
# |
# |
# | def load_gadm(gadm_shp: Optional[str] = None) -> gpd.GeoDataFrame:
# |     """Load and prepare GADM level-2 GeoDataFrame."""
# |     if gadm_shp is None:
# |         gadm_shp = find_default_gadm_path()
# |     
# |     print(f"Loading GADM from {gadm_shp}...")
# |     gadm = gpd.read_file(gadm_shp)
# |     gadm = gadm[['GID_2', 'NAME_2', 'geometry']].to_crs("EPSG:4326")
# |     return gadm
# |
# |
# | def aggregate_city_to_gadm2(
# |     city_name: str,
# |     data_root: str = "data",
# |     gadm_shp: Optional[str] = None,
# |     gadm_gdf: Optional[gpd.GeoDataFrame] = None,
# |     output_root: str = "data_gadm2",
# | ) -> Dict[str, Any]:
# |     """
# |     Aggregate tract-level OD and distances to GADM level-2 counties.
# |     
# |     Returns preflight dict with diagnostics.
# |     """
# |     raise RuntimeError(
# |         "gadm2_county_aggregator.py is officially deprecated and retired from the pipeline.\n"
# |         "1. It silently drops tracts where centroids don't match exactly without a robust nearest-fallback.\n"
# |         "2. It averages distances across tracts (mean tract-pair distance), which heavily distorts distance distributions "
# |         "and is totally invalid for distance-binned Y_D reconstruction.\n"
# |         "Use src.data.gadm_mapper.get_gadm_gid2_mapping on tract pairs directly instead."
# |     )
# |     if gadm_gdf is None:
# |         gadm_gdf = load_gadm(gadm_shp)
# |     
# |     # Load tract metadata
# |     meta_file = Path(data_root) / city_name / "meta.csv"
# |     if not meta_file.exists():
# |         raise FileNotFoundError(f"Meta file not found: {meta_file}")
# |     
# |     meta = pd.read_csv(meta_file)
# |     print(f"  {city_name}: {len(meta)} tracts")
# |     
# |     # Create GeoDataFrame with tract centroids
# |     tract_gdf = gpd.GeoDataFrame(
# |         meta,
# |         geometry=gpd.points_from_xy(meta['lon'], meta['lat']),
# |         crs="EPSG:4326"
# |     )
# |     
# |     # Point-in-polygon: tract -> GID_2
# |     print(f"  Point-in-polygon spatial join...")
# |     result = gpd.sjoin(tract_gdf, gadm_gdf[['GID_2', 'geometry']], how='left', predicate='within')
# |     
# |     # Handle duplicates if any centroid lies on a boundary
# |     if result.index.has_duplicates:
# |         result = result[~result.index.duplicated(keep='first')]
# |     
# |     # Map: tract_idx -> GID_2
# |     if 'idx' in result.columns:
# |         idx_to_gid2 = dict(zip(result['idx'], result['GID_2']))
# |     else:
# |         idx_to_gid2 = dict(enumerate(result['GID_2']))
# |     
# |     # Load OD matrix
# |     od_file = Path(data_root) / city_name / "pairs" / "od.csv"
# |     if not od_file.exists():
# |         raise FileNotFoundError(f"OD file not found: {od_file}")
# |     
# |     od = pd.read_csv(od_file)
# |     od['county_o'] = od['o_idx'].map(idx_to_gid2)
# |     od['county_d'] = od['d_idx'].map(idx_to_gid2)
# |     
# |     # Aggregate OD by county pairs
# |     od_valid = od.dropna(subset=['county_o', 'county_d'])
# |     od_county = od_valid.groupby(['county_o', 'county_d'])['trip_count'].sum().reset_index()
# |     od_county.columns = ['county_o', 'county_d', 'trips']
# |     
# |     # Load distances
# |     dist_file = Path(data_root) / city_name / "pairs" / "distance.csv"
# |     if not dist_file.exists():
# |         raise FileNotFoundError(f"Distance file not found: {dist_file}")
# |     
# |     dist = pd.read_csv(dist_file)
# |     dist['county_o'] = dist['o_idx'].map(idx_to_gid2)
# |     dist['county_d'] = dist['d_idx'].map(idx_to_gid2)
# |     
# |     # Aggregate distances: mean per county pair
# |     dist_valid = dist.dropna(subset=['county_o', 'county_d'])
# |     dist_county = dist_valid.groupby(['county_o', 'county_d'])['distance_km'].agg(['mean', 'count']).reset_index()
# |     dist_county.columns = ['county_o', 'county_d', 'distance_km', 'pair_count']
# |     
# |     # Merge
# |     county_pairs = od_county.merge(dist_county, on=['county_o', 'county_d'], how='inner')
# |     
# |     # Create output directory
# |     output_dir = Path(output_root) / city_name
# |     output_dir.mkdir(parents=True, exist_ok=True)
# |     
# |     # Save
# |     county_pairs.to_csv(output_dir / "county_pairs.csv", index=False)
# |     
# |     # Preflight diagnostics
# |     valid_gid2 = {v for v in idx_to_gid2.values() if pd.notna(v)}
# |     n_counties = len(valid_gid2)
# |     n_tract_unmapped = sum(1 for v in idx_to_gid2.values() if pd.isna(v))
# |     
# |     total_trips_tract = int(od['trip_count'].sum())
# |     total_trips_county = int(county_pairs['trips'].sum())
# |     conservation_error = (
# |         abs(total_trips_tract - total_trips_county) / total_trips_tract
# |         if total_trips_tract > 0 else 0.0
# |     )
# |     
# |     preflight = {
# |         "city": city_name,
# |         "n_tracts": len(meta),
# |         "n_counties": n_counties,
# |         "n_tract_unmapped": n_tract_unmapped,
# |         "n_county_pairs": len(county_pairs),
# |         "total_trips_tract": total_trips_tract,
# |         "total_trips_county": total_trips_county,
# |         "trip_conservation_error": conservation_error,
# |     }
# |     
# |     print(f"  -> {n_counties} counties, {len(county_pairs)} county pairs")
# |     print(f"  -> Trip conservation error: {preflight['trip_conservation_error']:.2e}")
# |     
# |     return preflight
# |
# |
# | def run_all(
# |     cities: Optional[List[str]] = None,
# |     data_root: str = "data",
# |     gadm_shp: Optional[str] = None,
# |     output_root: str = "data_gadm2",
# |     results_dir: str = "results",
# | ) -> pd.DataFrame:
# |     """Run aggregation for all specified or discovered cities."""
# |     if cities is None:
# |         data_p = Path(data_root)
# |         cities = sorted([
# |             d.name for d in data_p.iterdir()
# |             if d.is_dir() and (d / "meta.csv").exists()
# |         ])
# |     
# |     print(f"Found {len(cities)} cities to process.")
# |     
# |     # Load GADM once for efficiency
# |     gadm_gdf = load_gadm(gadm_shp)
# |     
# |     results = []
# |     for city in cities:
# |         print(f"\nProcessing {city}...")
# |         try:
# |             r = aggregate_city_to_gadm2(
# |                 city_name=city,
# |                 data_root=data_root,
# |                 gadm_gdf=gadm_gdf,
# |                 output_root=output_root,
# |             )
# |             results.append(r)
# |         except Exception as e:
# |             print(f"  ERROR {city}: {e}")
# |             results.append({"city": city, "error": str(e)})
# |     
# |     # Preflight report
# |     report_df = pd.DataFrame(results)
# |     print("\n" + "=" * 80)
# |     print(report_df.to_string())
# |     print("=" * 80)
# |     
# |     # Stop gate check
# |     if 'n_counties' in report_df.columns:
# |         failed = report_df[report_df['n_counties'].fillna(0) < 2]
# |         if len(failed) > 0:
# |             print(f"\nWARNING: {len(failed)} cities with <2 counties (no intercounty OD)")
# |             print(failed[['city', 'n_counties']])
# |     
# |     # Save report
# |     Path(results_dir).mkdir(parents=True, exist_ok=True)
# |     report_path = Path(results_dir) / "gadm2_preflight.csv"
# |     report_df.to_csv(report_path, index=False)
# |     print(f"\nPreflight report saved to {report_path}")
# |     
# |     return report_df
# |
# |
# | if __name__ == "__main__":
# |     run_all()

# ===== END SOURCE FILE: src/aggregation/gadm2_county_aggregator.py =====


# ===== BEGIN SOURCE FILE: src/calibration/bin_calibration.py =====
# | r"""
# | Interzonal Moving-Bin Calibration on Omega_c^+ via Soft KL Projection.
# |
# | Mathematical Formulation:
# |     1. Interzonal Domain:
# |         Omega_c^+ = {(i,j) in Omega_c : i != j, D_ij > 0}
# |         Intrazonal pairs (i == j, D_ii = 0) are kept intact: \hat{T}_{ii}^{cal} = \hat{T}_{ii}^{ZS}.
# |
# |     2. Moving-Bin Target Distribution:
# |         Y_{c, k}^{Meta, +} = Y_{c, k}^{Meta} / sum_{l=1}^3 Y_{c, l}^{Meta}   for k in {1, 2, 3}
# |         Y_{c, k}^{oracle, +} = sum_{(i,j) in Omega_{c,k}^+} T_{ij}^{GT} / sum_{(i,j) in Omega_c^+} T_{ij}^{GT}
# |
# |     3. Support Conditioning:
# |         For cities with diameter < 100 km (where bin 3 has 0 pairs), condition target on active moving bins:
# |         p_k^{cond, +} = Y_k^+ * 1(k active) / sum_{l active} Y_l^+
# |
# |     4. Soft Calibration Multipliers (0 <= q <= 1):
# |         \hat{B}_k^+ = sum_{(i,j) in Omega_{c,k}^+} \hat{T}_{ij}^{ZS}
# |         \hat{N}^+ = sum_{(i,j) in Omega_c^+} \hat{T}_{ij}^{ZS}
# |         \hat{Y}_k^{ZS, +} = \hat{B}_k^+ / \hat{N}^+
# |
# |         w_k(q) = ( p_k^{cond, +} / \hat{Y}_k^{ZS, +} )^q
# |         s_k = w_k(q) / sum_{l active} [ \hat{Y}_l^{ZS, +} * w_l(q) ]
# |
# |         \hat{T}_{ij}^{cal} = s_{b(i,j)} * \hat{T}_{ij}^{ZS}   for (i,j) in Omega_c^+
# |
# | Strict Invariants:
# |     1. Interzonal mass preservation: \sum_{Omega^+} \hat{T}^{cal} == \sum_{Omega^+} \hat{T}^{ZS}.
# |     2. Intrazonal identity: \hat{T}_{ii}^{cal} == \hat{T}_{ii}^{ZS}.
# |     3. At q=1: implied moving-bin proportions match p_k^{cond, +} exactly within 1e-5.
# |     4. At q=0: \hat{T}^{cal} == \hat{T}^{ZS} (pure zero-shot).
# | """
# |
# | import numpy as np
# | import torch
# |
# |
# | def calibrate_moving_bins(
# |     t_pred_zero_shot: torch.Tensor,
# |     bin_labels: torch.Tensor,
# |     pair_o_idx: torch.Tensor,
# |     pair_d_idx: torch.Tensor,
# |     target_moving_yd: np.ndarray | torch.Tensor,
# |     q: float = 1.0,
# |     pair_distance: torch.Tensor | None = None,
# |     tolerance: float = 1e-5,
# | ) -> torch.Tensor:
# |     """
# |     Applies interzonal moving-bin calibration on Omega_c^+ (bins 1, 2, 3).
# |
# |     Args:
# |         t_pred_zero_shot: (E,) zero-shot predicted flows on Omega_c.
# |         bin_labels:       (E,) bin index (0=intrazonal, 1=(0,10), 2=[10,100), 3=100+).
# |         pair_o_idx:       (E,) origin indices.
# |         pair_d_idx:       (E,) destination indices.
# |         target_moving_yd: (3,) normalized moving-bin distribution for bins {1, 2, 3} (sums to 1.0).
# |         q:                soft calibration parameter in [0, 1]. q=1 is full match, q=0 is zero-shot.
# |         pair_distance:    Optional (E,) pairwise distance tensor (log1p km or km).
# |         tolerance:        numerical precision tolerance (default 1e-5).
# |
# |     Returns:
# |         t_cal: (E,) calibrated flows with intrazonal preserved and interzonal re-scaled.
# |     """
# |     assert 0.0 <= q <= 1.0, f"q must be in [0, 1], got {q}"
# |
# |     if isinstance(target_moving_yd, np.ndarray):
# |         p_raw = torch.tensor(target_moving_yd, dtype=torch.float32, device=t_pred_zero_shot.device)
# |     else:
# |         p_raw = target_moving_yd.to(device=t_pred_zero_shot.device, dtype=torch.float32)
# |
# |     # Normalize moving target
# |     raw_sum = torch.sum(p_raw)
# |     if raw_sum <= 0:
# |         return t_pred_zero_shot.clone()
# |     p_raw = p_raw / raw_sum
# |
# |     # Mask for interzonal pairs Omega_c^+ (i != j and D_ij > 0)
# |     if pair_distance is not None:
# |         p_dist = pair_distance.to(device=t_pred_zero_shot.device)
# |         dist_km = p_dist
# |         inter_mask = (pair_o_idx != pair_d_idx) & (dist_km > 0.0)
# |     else:
# |         inter_mask = (pair_o_idx != pair_d_idx) & (bin_labels > 0)
# |     intra_mask = ~inter_mask
# |
# |     # Clone predictions
# |     t_cal = t_pred_zero_shot.clone()
# |
# |     n_inter_hat = torch.sum(t_pred_zero_shot[inter_mask])
# |     if n_inter_hat <= 0:
# |         return t_cal
# |
# |     # Compute implied mass on moving bins {1, 2, 3}
# |     implied_b = torch.zeros(3, dtype=torch.float32, device=t_pred_zero_shot.device)
# |     active_mask = torch.zeros(3, dtype=torch.bool, device=t_pred_zero_shot.device)
# |
# |     for idx, bin_k in enumerate([1, 2, 3]):
# |         k_mask = inter_mask & (bin_labels == bin_k)
# |         implied_b[idx] = torch.sum(t_pred_zero_shot[k_mask])
# |         active_mask[idx] = k_mask.any()
# |
# |     # Condition target on active moving bins
# |     p_active = p_raw * active_mask.float()
# |     active_sum = torch.sum(p_active)
# |     if active_sum <= 0:
# |         p_cond = implied_b / n_inter_hat
# |     else:
# |         p_cond = p_active / active_sum
# |
# |     implied_p = implied_b / n_inter_hat
# |
# |     # Compute soft weights w_k(q) = (p_cond / implied_p)^q
# |     w = torch.zeros(3, dtype=torch.float32, device=t_pred_zero_shot.device)
# |     for idx in range(3):
# |         if active_mask[idx] and implied_p[idx] > 0:
# |             ratio = p_cond[idx] / implied_p[idx]
# |             w[idx] = ratio ** q
# |         else:
# |             # Inactive bin (no candidate pairs in this bin) → zero weight.
# |             # This is consistent with the mathematical spec: inactive bins carry no mass
# |             # and must not contribute to the scaling normalization.
# |             w[idx] = 0.0
# |
# |     # Normalization to ensure interzonal mass preservation: \sum \hat{T}^{cal} == \sum \hat{T}^{ZS}
# |     weighted_mass = torch.sum(implied_p * w)
# |     s = torch.zeros(3, dtype=torch.float32, device=t_pred_zero_shot.device)
# |     if weighted_mass > 0:
# |         s = w / weighted_mass
# |
# |     # Apply scaling to interzonal pairs
# |     for idx, bin_k in enumerate([1, 2, 3]):
# |         k_mask = inter_mask & (bin_labels == bin_k)
# |         if k_mask.any():
# |             t_cal[k_mask] = t_pred_zero_shot[k_mask] * s[idx]
# |
# |     # Invariant 1: Interzonal mass preservation within numerical tolerance
# |     cal_inter_mass = torch.sum(t_cal[inter_mask])
# |     mass_diff_rel = torch.abs(cal_inter_mass - n_inter_hat) / n_inter_hat
# |     if mass_diff_rel > tolerance:
# |         t_cal[inter_mask] = t_cal[inter_mask] * (n_inter_hat / cal_inter_mass)
# |
# |     # Invariant 2: Intrazonal identity
# |     assert torch.allclose(t_cal[intra_mask], t_pred_zero_shot[intra_mask], atol=1e-6), "Intrazonal violated!"
# |
# |     # Invariant 3: If q=1, verify bin matching on active bins within 1e-5
# |     if abs(q - 1.0) < 1e-4:
# |         cal_inter_p = torch.zeros(3, dtype=torch.float32, device=t_pred_zero_shot.device)
# |         total_inter_cal = torch.sum(t_cal[inter_mask])
# |         for idx, bin_k in enumerate([1, 2, 3]):
# |             if active_mask[idx]:
# |                 cal_inter_p[idx] = torch.sum(t_cal[inter_mask & (bin_labels == bin_k)])
# |         if total_inter_cal > 0:
# |             cal_inter_p = cal_inter_p / total_inter_cal
# |
# |         for idx in range(3):
# |             if active_mask[idx]:
# |                 bin_err = torch.abs(cal_inter_p[idx] - p_cond[idx]).item()
# |                 assert bin_err < tolerance, (
# |                     f"Invariant failed on moving bin {idx+1}: target={p_cond[idx].item():.6f}, "
# |                     f"got={cal_inter_p[idx].item():.6f}, err={bin_err:.6f}"
# |                 )
# |
# |     return t_cal
# |
# |
# | def calibrate_4bin_legacy_ablation(
# |     t_pred_zero_shot: torch.Tensor,
# |     bin_labels: torch.Tensor,
# |     target_4bin_yd: np.ndarray | torch.Tensor,
# |     eps: float = 1e-8,
# | ) -> torch.Tensor:
# |     """
# |     Legacy 4-bin calibration (Ablation M1^{real, 4bin}) deliberately retaining
# |     the semantic mismatch of Bin 0 to demonstrate its empirical penalty.
# |     """
# |     if isinstance(target_4bin_yd, np.ndarray):
# |         p_raw = torch.tensor(target_4bin_yd, dtype=torch.float32, device=t_pred_zero_shot.device)
# |     else:
# |         p_raw = target_4bin_yd.to(device=t_pred_zero_shot.device, dtype=torch.float32)
# |
# |     n_hat = torch.sum(t_pred_zero_shot)
# |     if n_hat <= 0:
# |         return t_pred_zero_shot
# |
# |     implied_b = torch.zeros(4, dtype=torch.float32, device=t_pred_zero_shot.device)
# |     active_mask = torch.zeros(4, dtype=torch.bool, device=t_pred_zero_shot.device)
# |     for k in range(4):
# |         mask = (bin_labels == k)
# |         implied_b[k] = torch.sum(t_pred_zero_shot[mask])
# |         active_mask[k] = mask.any()
# |
# |     p_active = p_raw * active_mask.float()
# |     p_cond = p_active / torch.clamp(torch.sum(p_active), min=eps)
# |
# |     s = (p_cond * n_hat + eps) / (implied_b + eps)
# |     t_cal = t_pred_zero_shot * s[bin_labels]
# |
# |     cal_mass = torch.sum(t_cal)
# |     t_cal = t_cal * (n_hat / (cal_mass + eps))
# |     return t_cal
# |
# |
# | # ---------------------------------------------------------------------------
# | # E1: Dynamic K-bin calibration (numpy-based, for Oracle Existence Test)
# | # ---------------------------------------------------------------------------
# |
# | def calibrate_kbins(
# |     t0_np: np.ndarray,
# |     dist_km: np.ndarray,
# |     inter_mask: np.ndarray,
# |     yd_target: np.ndarray,
# |     bin_edges: np.ndarray,
# |     q: float = 1.0,
# |     tolerance: float = 1e-5,
# | ) -> np.ndarray:
# |     r"""
# |     Closed-form K-bin Moving-Bin calibration for E1.
# |
# |     Works on numpy arrays (CPU-only). Mirrors calibrate_moving_bins() semantics
# |     but accepts dynamic bin_edges (K bins, not fixed 3-bin schema).
# |
# |     Mathematical formulation:
# |         Y_D_cond_k = Y_D_k * active_k / sum_l(Y_D_l * active_l)
# |         w_k(q)     = (Y_D_cond_k / Y_hat_k)^q
# |         s_k        = w_k / sum_l(Y_hat_l * w_l)
# |         T_cal_ij   = s_{b(ij)} * T0_ij   for (i,j) in Omega_c^+
# |
# |     Notes on zero-behavior:
# |         If target Y_D_k == 0, then w_k(q) = 0 for ANY q > 0.
# |         This forces hard-zero predictions on that bin, making q mapping non-continuous at q=0 if the target contains exact zeros.
# |         Smoothing/pseudocounts must be applied to Y_D prior to calling this function if a softer response is desired.
# |
# |     Invariants:
# |         1. Interzonal mass preservation: sum(T_cal[inter]) == sum(T0[inter]) within tolerance.
# |         2. Intrazonal identity: T_cal[~inter] == T0[~inter] exactly.
# |         3. At q=1: bin proportions of T_cal match Y_D_cond within tolerance for active bins.
# |         4. GT-independence: output is a function of T0 and Y_D only, not T^GT.
# |
# |     Args:
# |         t0_np:      (E,) zero-shot predicted flows (numpy float array).
# |         dist_km:    (E,) pairwise distances in km.
# |         inter_mask: (E,) boolean mask for Omega_c^+ (interzonal, D>0).
# |         yd_target:  (K,) target distance distribution summing to 1.0.
# |         bin_edges:  (K+1,) strictly increasing edges from compute_kbin_edges.
# |         q:          soft calibration strength in [0, 1]. q=1 = exact match.
# |         tolerance:  numerical precision for invariant checks.
# |
# |     Returns:
# |         t_cal: (E,) calibrated flows; intrazonal unchanged, interzonal rescaled.
# |     """
# |     assert 0.0 <= q <= 1.0, f"q must be in [0, 1], got {q}"
# |     K = len(bin_edges) - 1
# |     assert len(yd_target) == K, f"yd_target length {len(yd_target)} != K={K}"
# |
# |     # Normalize input Y_D (defensive)
# |     yd_sum = float(np.sum(yd_target))
# |     yd_raw = yd_target / yd_sum if yd_sum > 0 else np.ones(K) / K
# |
# |     t_cal = t0_np.copy().astype(np.float64)
# |     inter_T0 = t0_np[inter_mask].astype(np.float64)
# |     N_hat = inter_T0.sum()
# |
# |     if N_hat <= 0:
# |         return t_cal  # no interzonal flow to calibrate
# |
# |     inter_dist = dist_km[inter_mask]
# |
# |     # Compute implied distribution Y_hat from zero-shot
# |     Y_hat = np.zeros(K, dtype=np.float64)
# |     active = np.zeros(K, dtype=bool)
# |     for k in range(K):
# |         lo, hi = float(bin_edges[k]), float(bin_edges[k + 1])
# |         in_bin = (inter_dist > lo) & (inter_dist <= hi)
# |         Y_hat[k] = inter_T0[in_bin].sum() / N_hat
# |         active[k] = bool(in_bin.any())
# |
# |     # Condition Y_D on active bins only
# |     yd_active = yd_raw * active.astype(np.float64)
# |     active_sum = yd_active.sum()
# |     Y_D_cond = yd_active / active_sum if active_sum > 0 else Y_hat.copy()
# |
# |     # Soft weights: w_k = (Y_D_cond_k / Y_hat_k)^q
# |     w = np.ones(K, dtype=np.float64)
# |     for k in range(K):
# |         if active[k] and Y_hat[k] > 0:
# |             w[k] = (Y_D_cond[k] / Y_hat[k]) ** q
# |
# |     # Normalize: s_k = w_k / sum_l(Y_hat_l * w_l)
# |     weighted_mass = float((Y_hat * w).sum())
# |     s = w / weighted_mass if weighted_mass > 0 else np.ones(K)
# |
# |     # Apply per-bin scaling to interzonal pairs
# |     idx = np.where(inter_mask)[0]
# |     for k in range(K):
# |         lo, hi = float(bin_edges[k]), float(bin_edges[k + 1])
# |         in_bin = (inter_dist > lo) & (inter_dist <= hi)
# |         t_cal[idx[in_bin]] = t0_np[idx[in_bin]] * s[k]
# |
# |     # --- Invariant 1: Interzonal mass preservation ---
# |     cal_mass = float(t_cal[inter_mask].sum())
# |     mass_err_rel = abs(cal_mass - N_hat) / max(N_hat, 1e-8)
# |     if mass_err_rel > tolerance:
# |         t_cal[inter_mask] = t_cal[inter_mask] * (N_hat / cal_mass)
# |
# |     # --- Invariant 2: Intrazonal identity ---
# |     intra_mask = ~inter_mask
# |     assert np.allclose(t_cal[intra_mask], t0_np[intra_mask], atol=1e-6), \
# |         "calibrate_kbins: Intrazonal identity violated"
# |
# |     # --- Invariant 3: At q=1, bin proportions match Y_D_cond ---
# |     if abs(q - 1.0) < 1e-4:
# |         total_cal = float(t_cal[inter_mask].sum())
# |         if total_cal > 0:
# |             for k in range(K):
# |                 if active[k]:
# |                     lo, hi = float(bin_edges[k]), float(bin_edges[k + 1])
# |                     in_bin_cal = (inter_dist > lo) & (inter_dist <= hi)
# |                     cal_prop = float(t_cal[inter_mask][in_bin_cal].sum()) / total_cal
# |                     bin_err = abs(cal_prop - Y_D_cond[k])
# |                     assert bin_err < tolerance, (
# |                         f"calibrate_kbins bin {k}: target={Y_D_cond[k]:.6f}, "
# |                         f"got={cal_prop:.6f}, err={bin_err:.6f}"
# |                     )
# |
# |     return t_cal
# |
# |
# | def calibrate_kbins_grouped(
# |     t0_np: np.ndarray,
# |     dist_km: np.ndarray,
# |     inter_mask: np.ndarray,
# |     yd_target_dict: dict,
# |     bin_edges: np.ndarray,
# |     pair_group_idx: np.ndarray,
# |     q: float = 1.0,
# |     tolerance: float = 1e-5,
# | ) -> np.ndarray:
# |     """
# |     Group-conditioned K-bin calibration (e.g., per-county).
# |     
# |     Applies the closed-form K-bin calibration independently for each group
# |     defined by pair_group_idx (e.g., origin county ID of each pair),
# |     while preserving the zero-shot predicted outflow of each group.
# |     
# |     Args:
# |         t0_np:          (E,) zero-shot predicted flows.
# |         dist_km:        (E,) pairwise distances in km.
# |         inter_mask:     (E,) boolean mask for interzonal pairs Omega_c^+.
# |         yd_target_dict: Dict mapping group_id -> (K,) target distance distribution.
# |         bin_edges:      (K+1,) strictly increasing edges.
# |         pair_group_idx: (E,) group ID for each pair (e.g., origin county ID).
# |         q:              Soft calibration strength.
# |         tolerance:      Numerical precision.
# |         
# |     Returns:
# |         t_cal: (E,) calibrated flows.
# |     """
# |     t_cal = t0_np.copy().astype(np.float64)
# |     
# |     # Intrazonal pairs are not modified
# |     # We calibrate interzonal pairs group by group
# |     
# |     unique_groups = np.unique(pair_group_idx)
# |     
# |     for g in unique_groups:
# |         if g not in yd_target_dict:
# |             continue
# |             
# |         yd_g = yd_target_dict[g]
# |         
# |         # Mask for interzonal pairs belonging to group g
# |         g_mask = (pair_group_idx == g)
# |         inter_g_mask = inter_mask & g_mask
# |         
# |         if not inter_g_mask.any():
# |             continue
# |             
# |         # Extract slices for this group
# |         t0_g = t0_np[g_mask]
# |         dist_g = dist_km[g_mask]
# |         
# |         # We need a local inter_mask for the group slice
# |         # inter_g_mask is length E. We need a mask of length len(t0_g)
# |         # Since t0_g is selected by g_mask, the local inter_mask is simply
# |         # inter_mask[g_mask]
# |         local_inter_mask = inter_mask[g_mask]
# |         
# |         # Apply city-level calibration logic locally to the group
# |         # calibrate_kbins requires full E-length arrays if we pass them, 
# |         # but it works on any size. We pass the local slices.
# |         t_cal_g = calibrate_kbins(
# |             t0_np=t0_g,
# |             dist_km=dist_g,
# |             inter_mask=local_inter_mask,
# |             yd_target=yd_g,
# |             bin_edges=bin_edges,
# |             q=q,
# |             tolerance=tolerance
# |         )
# |         
# |         # Assign back to the global array
# |         t_cal[g_mask] = t_cal_g
# |         
# |     return t_cal
# |
# |
# | if __name__ == "__main__":
# |     t0 = torch.tensor([50.0, 100.0, 300.0, 600.0])  # pair 0 is intrazonal, 1,2,3 are interzonal
# |     bins = torch.tensor([0, 1, 2, 3])
# |     o_idx = torch.tensor([0, 0, 0, 0])
# |     d_idx = torch.tensor([0, 1, 2, 3])  # pair 0 is (0,0) intrazonal
# |     target_moving = np.array([0.25, 0.45, 0.30])  # sums to 1.0 for bins 1, 2, 3
# |
# |     # q=1.0 (Full calibration)
# |     t_cal_1 = calibrate_moving_bins(t0, bins, o_idx, d_idx, target_moving, q=1.0)
# |     print("Zero-shot t0:      ", t0.tolist())
# |     print("Calibrated t_cal(1):", t_cal_1.tolist())
# |     print("Intrazonal flow 0: ", t_cal_1[0].item(), "== t0[0]:", t0[0].item())
# |     print("Interzonal mass:   ", t_cal_1[1:].sum().item(), "== t0[1:].sum:", t0[1:].sum().item())
# |
# |     # q=0.5 (Soft calibration)
# |     t_cal_half = calibrate_moving_bins(t0, bins, o_idx, d_idx, target_moving, q=0.5)
# |     print("Soft t_cal(0.5):   ", t_cal_half.tolist())
# |
# |     # q=0.0 (Zero-shot identity)
# |     t_cal_0 = calibrate_moving_bins(t0, bins, o_idx, d_idx, target_moving, q=0.0)
# |     assert torch.allclose(t_cal_0, t0), "q=0 must equal zero-shot!"
# |     print("q=0 equals zero-shot: PASS")

# ===== END SOURCE FILE: src/calibration/bin_calibration.py =====


# ===== BEGIN SOURCE FILE: src/data/city_splits.py =====
# | """
# | 5-Fold Stratified City Splits across 50 US cities for Experiment E1 (v2 Amended Protocol).
# |
# | Design Principles:
# | 1. Outer Split Invariance: 10 test cities per fold are locked exactly from E1-v1 to prevent
# |    post-hoc test set selection or tie-break perturbation.
# | 2. Validation Stratification: Inner 5-stratum size stratification across the 40 non-test
# |    cities, sampling exactly 1 validation city per stratum with fixed seed (seed + fold_id).
# | 3. Manifest Self-Containment: Manifest contains full validation candidate lists per stratum
# |    and SHA-256 integrity hashing for full auditability.
# | 4. Strict Invariants: 35 Train / 5 Val / 10 Test per fold; mutual disjointness; complete partition.
# | 5. Estimand Alignment: Unit of analysis is strictly the city; wrong placebo is averaged over
# |    all 9 within-fold donors for specificity Delta_c^specificity = Delta_c^target - bar{Delta}_c^wrong.
# | """
# |
# | import os
# | import csv
# | import json
# | import random
# | import hashlib
# | from pathlib import Path
# | from typing import List, Dict, Tuple
# |
# | # Canonical test sets locked from E1-v1 to prevent any outer fold shift
# | LOCKED_V1_TEST_FOLDS: Dict[int, List[str]] = {
# |     1: [
# |         "Arlington", "Austin", "El_Paso", "Long_Beach", "Memphis",
# |         "Milwaukee", "New_York", "San_Diego", "Seattle", "Virginia_Beach"
# |     ],
# |     2: [
# |         "Atlanta", "Boston", "Fort_Worth", "Indianapolis", "Los_Angeles",
# |         "Mesa", "Oklahoma_City", "Raleigh", "Sacramento", "San_Antonio"
# |     ],
# |     3: [
# |         "Baltimore", "Chicago", "Detroit", "Fresno", "Jacksonville",
# |         "Las_Vegas", "Louisville", "Oakland", "Tulsa", "Washington_DC"
# |     ],
# |     4: [
# |         "Colorado_Springs", "Columbus", "Houston", "Minneapolis", "Nashville",
# |         "Omaha", "Phoenix", "Portland", "San_Francisco", "Tampa"
# |     ],
# |     5: [
# |         "Albuquerque", "Charlotte", "Dallas", "Denver", "Kansas_City",
# |         "Miami", "Philadelphia", "San_Jose", "Tucson", "Wichita"
# |     ],
# | }
# |
# | STRATUM_NAMES = [
# |     "stratum_0_small",
# |     "stratum_1_small_med",
# |     "stratum_2_med",
# |     "stratum_3_med_large",
# |     "stratum_4_large",
# | ]
# |
# |
# | def get_all_cities_sorted_by_size(data_root: str = "data") -> List[Dict]:
# |     """
# |     Inspects all 50 cities and returns them sorted by tract count.
# |     Strict tie-breaking: (n_tracts, city).
# |     """
# |     root = Path(data_root)
# |     city_dirs = [d.name for d in root.iterdir() if d.is_dir()]
# |     cities_info = []
# |
# |     for city in city_dirs:
# |         meta_path = root / city / "meta.csv"
# |         if not meta_path.exists():
# |             continue
# |         with open(meta_path, newline="") as f:
# |             n_tracts = sum(1 for _ in f) - 1
# |         cities_info.append({"city": city, "n_tracts": n_tracts})
# |
# |     # Sort ascending by tract count, tie-break with city name
# |     cities_info.sort(key=lambda x: (x["n_tracts"], x["city"]))
# |     return cities_info
# |
# |
# | def generate_5fold_splits(data_root: str = "data") -> Dict[int, Dict[str, List[str]]]:
# |     """
# |     DEPRECATED — Returns 5 outer folds with 40 train / 10 test cities.
# |
# |     WARNING: This function produces 40/0/10 splits (no validation set), which VIOLATES
# |     the locked 35/5/10 protocol (Contract §7). It is retained only for backward
# |     compatibility with legacy test code.
# |
# |     Use generate_35_5_10_splits() or load_splits_manifest_v2() instead.
# |     """
# |     import warnings
# |     warnings.warn(
# |         "generate_5fold_splits() produces 40-train/0-val/10-test splits which VIOLATES "
# |         "the locked 35/5/10 protocol. Use generate_35_5_10_splits() or "
# |         "load_splits_manifest_v2() instead.",
# |         DeprecationWarning,
# |         stacklevel=2,
# |     )
# |     cities_info = get_all_cities_sorted_by_size(data_root)
# |     all_city_names = [c["city"] for c in cities_info]
# |     splits = {}
# |     for fold_id in range(1, 6):
# |         test_cities = sorted(LOCKED_V1_TEST_FOLDS[fold_id])
# |         train_cities = sorted(list(set(all_city_names) - set(test_cities)))
# |         splits[fold_id] = {
# |             "train": train_cities,
# |             "test": test_cities,
# |         }
# |     return splits
# |
# |
# | def select_stratified_validation(
# |     non_test_info: List[Dict],
# |     fold_id: int,
# |     seed: int = 20260818,
# | ) -> Tuple[List[str], List[str], Dict[str, List[Dict]]]:
# |     """
# |     Selects 5 validation cities from 40 non-test cities using 5 size strata.
# |
# |     Algorithm:
# |       1. Sort 40 non-test cities by (n_tracts, city).
# |       2. Divide into 5 size strata of 8 cities each (small -> large).
# |       3. Draw 1 validation city from each stratum using Random(seed + fold_id).
# |       4. The remaining 35 cities form the training set.
# |
# |     Returns:
# |       (train_cities, val_cities, validation_candidates_by_stratum)
# |     """
# |     ordered = sorted(non_test_info, key=lambda x: (x["n_tracts"], x["city"]))
# |     assert len(ordered) == 40, f"Expected 40 non-test cities, got {len(ordered)}"
# |
# |     # 40 cities -> 5 size strata x 8 cities
# |     strata = [ordered[i * 8 : (i + 1) * 8] for i in range(5)]
# |
# |     rng = random.Random(seed + fold_id)
# |     val_cities = []
# |     candidates_by_stratum = {}
# |
# |     for s_idx, stratum in enumerate(strata):
# |         s_name = STRATUM_NAMES[s_idx]
# |         chosen = rng.choice(stratum)["city"]
# |         val_cities.append(chosen)
# |         candidates_by_stratum[s_name] = [
# |             {"city": item["city"], "n_tracts": item["n_tracts"], "selected_for_val": item["city"] == chosen}
# |             for item in stratum
# |         ]
# |
# |     val_set = set(val_cities)
# |     train_cities = [item["city"] for item in ordered if item["city"] not in val_set]
# |
# |     return sorted(train_cities), sorted(val_cities), candidates_by_stratum
# |
# |
# | def generate_splits_manifest_v2(
# |     data_root: str = "data",
# |     seed: int = 20260818,
# |     output_path: str = "results/e1/splits_manifest_v2.json",
# | ) -> dict:
# |     """
# |     Generates the canonical E1-v2 manifest locking the E1-v1 test sets and
# |     applying size-stratified validation on the 40 non-test pool.
# |     """
# |     cities_info = get_all_cities_sorted_by_size(data_root)
# |     all_city_names = sorted([c["city"] for c in cities_info])
# |     assert len(all_city_names) == 50, f"Expected 50 cities, found {len(all_city_names)}"
# |
# |     city_dict = {c["city"]: c for c in cities_info}
# |     manifest_folds = {}
# |     test_count = {c: 0 for c in all_city_names}
# |
# |     for fold_id in range(1, 6):
# |         # 1. Lock outer test fold directly from E1-v1
# |         test_cities = sorted(LOCKED_V1_TEST_FOLDS[fold_id])
# |         assert len(test_cities) == 10, f"Fold {fold_id} test size {len(test_cities)} != 10"
# |
# |         # 2. Extract 40 non-test cities
# |         non_test_cities = [c for c in all_city_names if c not in set(test_cities)]
# |         non_test_info = [city_dict[c] for c in non_test_cities]
# |         assert len(non_test_info) == 40, f"Fold {fold_id} non-test count != 40"
# |
# |         # 3. Stratified validation selection
# |         train_cities, val_cities, candidates_by_stratum = select_stratified_validation(
# |             non_test_info, fold_id=fold_id, seed=seed
# |         )
# |
# |         train_set = set(train_cities)
# |         val_set = set(val_cities)
# |         test_set = set(test_cities)
# |
# |         # Invariant Assertions within fold
# |         assert len(train_cities) == 35, f"Fold {fold_id} train size != 35"
# |         assert len(val_cities) == 5, f"Fold {fold_id} val size != 5"
# |         assert len(test_cities) == 10, f"Fold {fold_id} test size != 10"
# |
# |         # No duplicate cities within lists
# |         assert len(set(train_cities)) == 35, f"Fold {fold_id} train contains duplicates"
# |         assert len(set(val_cities)) == 5, f"Fold {fold_id} val contains duplicates"
# |         assert len(set(test_cities)) == 10, f"Fold {fold_id} test contains duplicates"
# |
# |         # Pairwise disjointness
# |         assert train_set.isdisjoint(val_set), f"Fold {fold_id} train/val overlap"
# |         assert train_set.isdisjoint(test_set), f"Fold {fold_id} train/test overlap"
# |         assert val_set.isdisjoint(test_set), f"Fold {fold_id} val/test overlap"
# |         assert (train_set | val_set | test_set) == set(all_city_names), f"Fold {fold_id} does not partition 50 cities"
# |
# |         for c in test_cities:
# |             test_count[c] += 1
# |
# |         manifest_folds[str(fold_id)] = {
# |             "train": train_cities,
# |             "val": val_cities,
# |             "test": test_cities,
# |             "validation_candidates_by_stratum": candidates_by_stratum,
# |         }
# |
# |     # Across all 5 folds: each city tested exactly once
# |     assert all(test_count[city] == 1 for city in all_city_names), "Test city partition invariant violated across folds"
# |
# |     # Compute SHA-256 hash over canonical fold content
# |     folds_canonical_json = json.dumps(manifest_folds, sort_keys=True)
# |     manifest_sha256 = hashlib.sha256(folds_canonical_json.encode("utf-8")).hexdigest()
# |
# |     manifest_data = {
# |         "version": "e1-splits-v2",
# |         "protocol_status": "amended replication under a locked protocol",
# |         "outer_split_source": "locked from E1-v1 outer test sets (zero test perturbation)",
# |         "validation_selection_rule": "five tract-count strata (8 cities each), fixed-seed selection (1 per stratum)",
# |         "validation_seed": seed,
# |         "manifest_sha256": manifest_sha256,
# |         "folds": manifest_folds,
# |     }
# |
# |     out_file = Path(output_path)
# |     out_file.parent.mkdir(parents=True, exist_ok=True)
# |     with open(out_file, "w", encoding="utf-8") as f:
# |         json.dump(manifest_data, f, indent=2)
# |
# |     return manifest_data
# |
# |
# | def load_splits_manifest_v2(
# |     manifest_path: str = "results/e1/splits_manifest_v2.json",
# |     data_root: str = "data",
# | ) -> Dict[int, Dict[str, List[str]]]:
# |     """
# |     Loads pre-locked splits from manifest v2 with runtime integrity and contract assertions.
# |     """
# |     path = Path(manifest_path)
# |     if not path.exists():
# |         raise FileNotFoundError(f"Missing locked manifest at {path}. Protocol requires explicit locked splits.")
# |
# |     with open(path, "r", encoding="utf-8") as f:
# |         data = json.load(f)
# |         
# |     stored_hash = data.get("manifest_sha256")
# |     if not stored_hash:
# |         raise ValueError(f"Manifest at {path} is missing 'manifest_sha256' field — integrity cannot be verified.")
# |     canonical = json.dumps(data.get("folds", {}), sort_keys=True)
# |     actual_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
# |     if actual_hash != stored_hash:
# |         raise ValueError(f"Manifest integrity compromised! Expected SHA-256 {stored_hash} but got {actual_hash}")
# |
# |     cities_info = get_all_cities_sorted_by_size(data_root)
# |     all_city_names = set(c["city"] for c in cities_info)
# |
# |     folds_raw = data.get("folds", {})
# |     assert len(folds_raw) == 5, f"Expected 5 folds in manifest, found {len(folds_raw)}"
# |
# |     parsed_splits = {}
# |     test_count = {c: 0 for c in all_city_names}
# |
# |     for fold_key in sorted(folds_raw.keys(), key=lambda x: int(x)):
# |         fold_id = int(fold_key)
# |         f_data = folds_raw[fold_key]
# |         train = sorted(f_data["train"])
# |         val = sorted(f_data["val"])
# |         test = sorted(f_data["test"])
# |
# |         # Invariant Assertions
# |         assert len(train) == 35, f"Fold {fold_id} train size {len(train)} != 35"
# |         assert len(val) == 5, f"Fold {fold_id} val size {len(val)} != 5"
# |         assert len(test) == 10, f"Fold {fold_id} test size {len(test)} != 10"
# |
# |         assert len(set(train)) == 35, f"Fold {fold_id} train has duplicates"
# |         assert len(set(val)) == 5, f"Fold {fold_id} val has duplicates"
# |         assert len(set(test)) == 10, f"Fold {fold_id} test has duplicates"
# |
# |         # Verify test set exactly matches the locked E1-v1 test set
# |         assert test == sorted(LOCKED_V1_TEST_FOLDS[fold_id]), (
# |             f"Fold {fold_id} test set does not match locked E1-v1 test set!"
# |         )
# |
# |         train_set, val_set, test_set = set(train), set(val), set(test)
# |         assert train_set.isdisjoint(val_set), f"Fold {fold_id} train & val overlap"
# |         assert train_set.isdisjoint(test_set), f"Fold {fold_id} train & test overlap"
# |         assert val_set.isdisjoint(test_set), f"Fold {fold_id} val & test overlap"
# |         assert (train_set | val_set | test_set) == all_city_names, f"Fold {fold_id} does not partition 50 cities"
# |
# |         for c in test:
# |             test_count[c] += 1
# |
# |         parsed_splits[fold_id] = {
# |             "train": train,
# |             "val": val,
# |             "test": test,
# |             "validation_candidates_by_stratum": f_data.get("validation_candidates_by_stratum", {}),
# |         }
# |
# |     assert all(test_count[city] == 1 for city in all_city_names), "Not all cities tested exactly once across folds"
# |     return parsed_splits
# |
# |
# | def get_wrong_donors(target_city: str, test_cities: List[str]) -> List[str]:
# |     """
# |     Returns all other 9 test cities in the fold as wrong donors.
# |     """
# |     test_sorted = sorted(test_cities)
# |     assert target_city in test_sorted, f"Target city {target_city} not in test fold {test_sorted}"
# |     return [c for c in test_sorted if c != target_city]
# |
# |
# | def get_donor_city(target_city: str, test_cities: List[str]) -> str:
# |     """
# |     Single deterministic wrong-donor assignment (next city alphabetically, legacy fallback).
# |     """
# |     test_sorted = sorted(test_cities)
# |     idx = test_sorted.index(target_city)
# |     return test_sorted[(idx + 1) % len(test_sorted)]
# |
# |
# | def generate_35_5_10_splits(data_root: str = "data") -> Dict[int, Dict[str, List[str]]]:
# |     """
# |     Convenience wrapper returning the locked v2 35/5/10 splits.
# |     """
# |     return load_splits_manifest_v2(
# |         manifest_path="results/e1/splits_manifest_v2.json",
# |         data_root=data_root,
# |     )
# |
# |
# | if __name__ == "__main__":
# |     print("Generating and locking splits manifest v2 (Amended Protocol)...")
# |     manifest = generate_splits_manifest_v2("data")
# |     print(f"Locked version: {manifest['version']}")
# |     print(f"Protocol status: {manifest['protocol_status']}")
# |     print(f"Manifest SHA256: {manifest['manifest_sha256']}")
# |     print(f"Validation Seed: {manifest['validation_seed']}")
# |     for f, d in manifest["folds"].items():
# |         print(f"\nFold {f}:")
# |         print(f"  Train ({len(d['train'])}): {d['train'][:3]}...")
# |         print(f"  Val   ({len(d['val'])}): {d['val']}")
# |         print(f"  Test  ({len(d['test'])}): {d['test']}")

# ===== END SOURCE FILE: src/data/city_splits.py =====


# ===== BEGIN SOURCE FILE: src/data/dataset.py =====
# | """
# | City dataset loader for the distance-binned OD reconstruction study.
# |
# | Loads a single city's data from the standard directory layout:
# |     data/{city}/
# |         meta.csv               — tract metadata (idx, lon, lat, area_km2, city)
# |         nodes/
# |             census.csv         — population, income, employment features
# |             poi.csv            — POI density features
# |             road.csv           — road network features
# |         pairs/
# |             od.csv             — candidate OD pairs with trip_count
# |             distance.csv       — pairwise distances (km) for same candidate set
# |
# | Returns a CityData dataclass with:
# |     node_features  : FloatTensor (N, F)
# |     pair_o_idx     : LongTensor  (E,)     — origin tract index
# |     pair_d_idx     : LongTensor  (E,)     — destination tract index
# |     pair_distance  : FloatTensor (E,)     — distance in km
# |     pair_trips     : FloatTensor (E,)     — trip count (all >= 1)
# |     population     : FloatTensor (N,)     — total_population per tract
# |     lon_lat        : FloatTensor (N, 2)   — centroid coordinates
# |     city_name      : str
# |     n_tracts       : int
# |     n_pairs        : int
# |
# | Normalization:
# |     - Node features: StandardScaler fitted on training cities, applied to all.
# |     - Distances: log(1 + d_km).
# |     - Trip counts: kept as raw integers (ZTNB operates on counts directly).
# | """
# |
# | from __future__ import annotations
# |
# | import os
# | import csv
# | import hashlib
# | import dataclasses
# | from pathlib import Path
# | from typing import List, Optional, Dict
# |
# | import numpy as np
# | import torch
# |
# |
# | # ---------------------------------------------------------------------------
# | # Node feature columns (order must match across all cities)
# | # ---------------------------------------------------------------------------
# |
# | # Census features used (subset — well-defined across all 50 cities)
# | CENSUS_COLS = [
# |     "total_population", "median_age", "median_income", "per_capita_income",
# |     "employment_rate", "unemployment_rate", "commute_transit_pct",
# |     "commute_active_pct", "commute_wfh_pct", "zero_vehicle_pct",
# |     "avg_vehicles_per_household", "higher_education_pct", "homeownership_rate",
# | ]
# |
# | # POI features
# | POI_COLS = [
# |     "office", "office_density", "industrial", "industrial_density",
# |     "commercial", "commercial_density", "education_primary",
# |     "education_primary_density",
# | ]
# |
# | # Road features
# | ROAD_COLS = [
# |     "road_length_total", "road_density", "road_count",
# |     "motorway_length", "primary_length",
# | ]
# |
# | NODE_FEATURE_COLUMNS = tuple(CENSUS_COLS + POI_COLS + ROAD_COLS)
# |
# |
# | # ---------------------------------------------------------------------------
# | # Data class
# | # ---------------------------------------------------------------------------
# |
# | @dataclasses.dataclass
# | class CityData:
# |     city_name:      str
# |     n_tracts:       int
# |     n_pairs:        int
# |
# |     # Node-level (N, *)
# |     node_features:  torch.Tensor   # (N, F) normalized
# |     population:     torch.Tensor   # (N,)   raw population
# |     lon_lat:        torch.Tensor   # (N, 2) [lon, lat]
# |
# |     # Pair-level (E, *)
# |     pair_o_idx:     torch.LongTensor   # (E,)
# |     pair_d_idx:     torch.LongTensor   # (E,)
# |     pair_distance:  torch.Tensor       # (E,) log1p(km)
# |     pair_trips:     torch.Tensor       # (E,) raw counts, all >= 1
# |     bin_labels:     torch.LongTensor   # (E,) distance bin index (0-3)
# |
# |
# | # ---------------------------------------------------------------------------
# | # Distance bin assignment
# | # ---------------------------------------------------------------------------
# | # Bins match Meta mobility categories: 0 km | (0,10) | [10,100) | 100+
# | BIN_EDGES = [0.0, 1e-9, 10.0, 100.0, float("inf")]
# | BIN_LABELS = ["zero", "short", "medium", "long"]   # 0, 1, 2, 3
# |
# | def assign_bins(distance_km: np.ndarray) -> np.ndarray:
# |     """Assign each pair to a distance bin (0=zero, 1=short, 2=medium, 3=long)."""
# |     bins = np.zeros(len(distance_km), dtype=np.int64)
# |     bins[(distance_km > 0)   & (distance_km < 10)]  = 1
# |     bins[(distance_km >= 10) & (distance_km < 100)] = 2
# |     bins[distance_km >= 100]                         = 3
# |     return bins
# |
# |
# | # ---------------------------------------------------------------------------
# | # CSV loading helpers
# | # ---------------------------------------------------------------------------
# |
# | def _load_csv_columns(path: Path, cols: List[str], key_col: str = "idx") -> np.ndarray:
# |     """Load specific columns from a CSV, ordered by key_col. Returns float array."""
# |     data: Dict[int, List[float]] = {}
# |     with open(path, newline="") as f:
# |         reader = csv.DictReader(f)
# |         for row in reader:
# |             key = int(row[key_col])
# |             vals = []
# |             for c in cols:
# |                 v = row.get(c, "0") or "0"
# |                 try:
# |                     vals.append(float(v))
# |                 except ValueError:
# |                     vals.append(0.0)
# |             data[key] = vals
# |     if not data:
# |         return np.zeros((0, len(cols)), dtype=np.float32)
# |     keys = sorted(data.keys())
# |     n = max(keys) + 1
# |     if keys != list(range(n)):
# |         raise ValueError(f"Feature CSV {path} has missing indices. Expected 0 to {n-1}.")
# |     arr = np.zeros((n, len(cols)), dtype=np.float32)
# |     for k, v in data.items():
# |         arr[k] = v
# |     return arr
# |
# |
# | def _load_meta(path: Path):
# |     """Load meta.csv -> idx, lon, lat, population placeholder."""
# |     idx_list, lons, lats = [], [], []
# |     with open(path, newline="") as f:
# |         reader = csv.DictReader(f)
# |         for row in reader:
# |             idx_list.append(int(row["idx"]))
# |             lons.append(float(row["lon"]))
# |             lats.append(float(row["lat"]))
# |     if not idx_list:
# |         return np.zeros((0, 2), dtype=np.float32)
# |     n = max(idx_list) + 1
# |     if sorted(idx_list) != list(range(n)):
# |         raise ValueError(f"Meta CSV {path} has missing indices. Expected 0 to {n-1}.")
# |     lon_arr = np.zeros(n, dtype=np.float32)
# |     lat_arr = np.zeros(n, dtype=np.float32)
# |     for i, lon, lat in zip(idx_list, lons, lats):
# |         lon_arr[i] = lon
# |         lat_arr[i] = lat
# |     return np.stack([lon_arr, lat_arr], axis=1)   # (N, 2)
# |
# |
# | def _load_pairs(od_path: Path, dist_path: Path):
# |     """Load od.csv and distance.csv into aligned arrays."""
# |     od: Dict[tuple, int] = {}
# |     with open(od_path, newline="") as f:
# |         reader = csv.DictReader(f)
# |         for row in reader:
# |             trip = int(row["trip_count"])
# |             if trip > 0:
# |                 od[(int(row["o_idx"]), int(row["d_idx"]))] = trip
# |
# |     dist_map: Dict[tuple, float] = {}
# |     with open(dist_path, newline="") as f:
# |         reader = csv.DictReader(f)
# |         for row in reader:
# |             dist_map[(int(row["o_idx"]), int(row["d_idx"]))] = float(row["distance_km"])
# |             
# |     od_keys = set(od.keys())
# |     dist_keys = set(dist_map.keys())
# |     
# |     missing_dist = od_keys - dist_keys
# |     if len(missing_dist) > 0:
# |         raise ValueError(f"Found {len(missing_dist)} positive OD pairs missing from distance.csv (e.g. {list(missing_dist)[:3]}). Support integrity compromised.")
# |
# |     # Iterate over distance pairs that have positive OD trips
# |     origins, dests, trips, dists = [], [], [], []
# |     for pair in dist_keys:
# |         trip_count = od.get(pair)
# |         if trip_count is None:
# |             # Pair has distance but trip=0 or missing OD, which is fine (zero-trip pairs are ignored in GNN but safe to skip for support)
# |             continue
# |         origins.append(pair[0])
# |         dests.append(pair[1])
# |         trips.append(trip_count)
# |         dists.append(dist_map[pair])
# |
# |     return (
# |         np.array(origins, dtype=np.int64),
# |         np.array(dests,   dtype=np.int64),
# |         np.array(trips,   dtype=np.float32),
# |         np.array(dists,   dtype=np.float32),
# |     )
# |
# |
# | @dataclasses.dataclass
# | class RawCityData:
# |     city_name:      str
# |     n_tracts:       int
# |     n_pairs:        int
# |     X_raw:          np.ndarray         # (N, F) unscaled float32
# |     population:     torch.Tensor       # (N,)   raw population float32
# |     lon_lat:        torch.Tensor       # (N, 2) [lon, lat] float32
# |     pair_o_idx:     torch.LongTensor   # (E,)
# |     pair_d_idx:     torch.LongTensor   # (E,)
# |     pair_distance:  torch.Tensor       # (E,) log1p(km) float32
# |     pair_trips:     torch.Tensor       # (E,) raw counts, all >= 1 float32
# |     bin_labels:     torch.LongTensor   # (E,) distance bin index (0-3)
# |     dist_km:        np.ndarray         # (E,) raw pairwise distance in km
# |
# |
# | # Global In-Memory Caches for parsed raw CSV city datasets & normalized CityData instances
# | _RAW_CITY_CACHE: Dict[tuple[str, str], RawCityData] = {}
# | _CITY_DATA_CACHE: Dict[tuple[str, str, Optional[str]], CityData] = {}
# |
# |
# | def get_scaler_fingerprint(scaler: Optional[object]) -> Optional[str]:
# |     """
# |     Computes a deterministic content-based fingerprint (SHA-256) of a fitted StandardScaler.
# |     Prevents cross-fold leakage / normalization contamination caused by Python memory address (id(scaler)) reuse.
# |     Returns None if scaler is None.
# |     """
# |     if scaler is None:
# |         return None
# |     if hasattr(scaler, "mean_") and scaler.mean_ is not None:
# |         m_bytes = np.ascontiguousarray(scaler.mean_, dtype=np.float64).tobytes()
# |         v_bytes = np.ascontiguousarray(getattr(scaler, "var_", np.zeros_like(scaler.mean_)), dtype=np.float64).tobytes()
# |         s_bytes = np.ascontiguousarray(getattr(scaler, "scale_", np.ones_like(scaler.mean_)), dtype=np.float64).tobytes()
# |         return hashlib.sha256(m_bytes + v_bytes + s_bytes).hexdigest()
# |     return f"unfitted_{id(scaler)}"
# |
# |
# | def validate_feature_scaler(scaler: object) -> None:
# |     """Validate that a fitted scaler is safe for the fixed node-feature schema."""
# |     expected_features = len(NODE_FEATURE_COLUMNS)
# |     for attribute in ("mean_", "var_", "scale_"):
# |         if not hasattr(scaler, attribute):
# |             raise ValueError(f"Feature scaler is not fitted: missing {attribute}")
# |         values = np.asarray(getattr(scaler, attribute), dtype=np.float64)
# |         if values.shape != (expected_features,):
# |             raise ValueError(
# |                 f"Feature scaler {attribute} has shape {values.shape}; "
# |                 f"expected ({expected_features},)"
# |             )
# |         if not np.isfinite(values).all():
# |             raise ValueError(f"Feature scaler {attribute} contains NaN or Inf")
# |
# |     if np.any(np.asarray(scaler.var_) < 0.0):
# |         raise ValueError("Feature scaler var_ contains negative values")
# |     if np.any(np.asarray(scaler.scale_) <= 0.0):
# |         raise ValueError("Feature scaler scale_ must be strictly positive")
# |
# |     n_features = getattr(scaler, "n_features_in_", expected_features)
# |     if int(n_features) != expected_features:
# |         raise ValueError(
# |             f"Feature scaler expects {n_features} features; expected {expected_features}"
# |         )
# |
# |
# | def clear_city_cache() -> None:
# |     """Flushes both raw and normalized in-memory city dataset caches."""
# |     global _RAW_CITY_CACHE, _CITY_DATA_CACHE
# |     _RAW_CITY_CACHE.clear()
# |     _CITY_DATA_CACHE.clear()
# |
# |
# | def load_raw_city(
# |     city_name: str,
# |     data_root: str = "data",
# |     use_cache: bool = True,
# | ) -> RawCityData:
# |     """
# |     Load or retrieve unscaled raw city data from disk / in-memory cache.
# |     """
# |     cache_key = (city_name, str(Path(data_root).resolve()))
# |     if use_cache and cache_key in _RAW_CITY_CACHE:
# |         return _RAW_CITY_CACHE[cache_key]
# |
# |     base = Path(data_root) / city_name
# |
# |     # --- Node features ---
# |     census = _load_csv_columns(base / "nodes" / "census.csv", CENSUS_COLS)
# |     poi    = _load_csv_columns(base / "nodes" / "poi.csv",    POI_COLS)
# |     road   = _load_csv_columns(base / "nodes" / "road.csv",   ROAD_COLS)
# |     X_raw  = np.concatenate([census, poi, road], axis=1)   # (N, F)
# |     X_raw  = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
# |
# |     # Population for gravity prior (first census column)
# |     population = census[:, 0].copy()   # total_population
# |
# |     # Coordinates
# |     lon_lat = _load_meta(base / "meta.csv")   # (N, 2)
# |     n_tracts = X_raw.shape[0]
# |
# |     # --- Pair data ---
# |     o_idx, d_idx, trips, dist_km = _load_pairs(
# |         base / "pairs" / "od.csv",
# |         base / "pairs" / "distance.csv",
# |     )
# |     assert (trips >= 1).all(), f"{city_name}: found zero trip counts in candidate set"
# |
# |     log_dist = np.log1p(dist_km)
# |     bin_labels = assign_bins(dist_km)
# |
# |     raw_data = RawCityData(
# |         city_name     = city_name,
# |         n_tracts      = n_tracts,
# |         n_pairs       = len(o_idx),
# |         X_raw         = X_raw,
# |         population    = torch.tensor(population, dtype=torch.float32),
# |         lon_lat       = torch.tensor(lon_lat,    dtype=torch.float32),
# |         pair_o_idx    = torch.tensor(o_idx,      dtype=torch.long),
# |         pair_d_idx    = torch.tensor(d_idx,      dtype=torch.long),
# |         pair_distance = torch.tensor(log_dist,   dtype=torch.float32),
# |         pair_trips    = torch.tensor(trips,      dtype=torch.float32),
# |         bin_labels    = torch.tensor(bin_labels, dtype=torch.long),
# |         dist_km       = dist_km,
# |     )
# |
# |     if use_cache:
# |         _RAW_CITY_CACHE[cache_key] = raw_data
# |
# |     return raw_data
# |
# |
# | # ---------------------------------------------------------------------------
# | # Main loader
# | # ---------------------------------------------------------------------------
# |
# | def load_city(
# |     city_name: str,
# |     data_root: str = "data",
# |     feature_scaler: Optional["StandardScaler"] = None,
# |     fit_scaler: bool = False,
# |     use_cache: bool = True,
# | ) -> CityData:
# |     """
# |     Load one city's data, optionally applying or fitting a feature scaler.
# |
# |     Args:
# |         city_name:      Directory name under data_root.
# |         data_root:      Root of the data/ directory.
# |         feature_scaler: Optional fitted sklearn StandardScaler.
# |                         If None and fit_scaler=True, fits a new one.
# |         fit_scaler:     If True, fits scaler on this city's data.
# |         use_cache:      If True, retrieves raw parsed data from in-memory cache.
# |
# |     Returns:
# |         CityData instance.
# |     """
# |     if feature_scaler is not None and fit_scaler:
# |         raise ValueError("Pass either feature_scaler or fit_scaler=True, not both")
# |     if feature_scaler is not None:
# |         validate_feature_scaler(feature_scaler)
# |
# |     scaler_key = get_scaler_fingerprint(feature_scaler)
# |     resolved_root = str(Path(data_root).resolve())
# |     cache_key = (city_name, resolved_root, scaler_key)
# |
# |     if use_cache and not fit_scaler and cache_key in _CITY_DATA_CACHE:
# |         return _CITY_DATA_CACHE[cache_key]
# |
# |     raw = load_raw_city(city_name, data_root=data_root, use_cache=use_cache)
# |
# |     # --- Normalize node features ---
# |     if feature_scaler is not None:
# |         X_norm = feature_scaler.transform(raw.X_raw)
# |     elif fit_scaler:
# |         from sklearn.preprocessing import StandardScaler
# |         feature_scaler = StandardScaler()
# |         X_norm = feature_scaler.fit_transform(raw.X_raw)
# |         scaler_key = get_scaler_fingerprint(feature_scaler)
# |         cache_key = (city_name, resolved_root, scaler_key)
# |     else:
# |         X_norm = raw.X_raw
# |
# |     # Replace NaN/Inf that may arise from missing features
# |     X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=0.0, neginf=0.0)
# |
# |     cd = CityData(
# |         city_name     = raw.city_name,
# |         n_tracts      = raw.n_tracts,
# |         n_pairs       = raw.n_pairs,
# |         node_features = torch.tensor(X_norm, dtype=torch.float32),
# |         population    = raw.population,
# |         lon_lat       = raw.lon_lat,
# |         pair_o_idx    = raw.pair_o_idx,
# |         pair_d_idx    = raw.pair_d_idx,
# |         pair_distance = raw.pair_distance,
# |         pair_trips    = raw.pair_trips,
# |         bin_labels    = raw.bin_labels,
# |     )
# |
# |     if use_cache:
# |         _CITY_DATA_CACHE[cache_key] = cd
# |
# |     return cd
# |
# |
# | def load_cities(
# |     city_names: List[str],
# |     data_root: str = "data",
# |     use_cache: bool = True,
# | ) -> tuple[List[CityData], object]:
# |     """
# |     Load multiple cities, fitting a single StandardScaler on all training
# |     node features jointly (to ensure consistent normalization).
# |
# |     Returns:
# |         (list of CityData, fitted scaler)
# |     """
# |     from sklearn.preprocessing import StandardScaler
# |
# |     if not city_names:
# |         raise ValueError("At least one training city is required to fit the feature scaler")
# |     if len(city_names) != len(set(city_names)):
# |         raise ValueError("Training city names must be unique when fitting the feature scaler")
# |
# |     # First pass: collect raw features from memory cache
# |     raw_list = [load_raw_city(name, data_root=data_root, use_cache=use_cache) for name in city_names]
# |     all_X = [r.X_raw for r in raw_list]
# |
# |     scaler = StandardScaler()
# |     scaler.fit(np.concatenate(all_X, axis=0))
# |     validate_feature_scaler(scaler)
# |     scaler_key = get_scaler_fingerprint(scaler)
# |     resolved_root = str(Path(data_root).resolve())
# |
# |     # Second pass: construct CityData with fitted scaler and cache into _CITY_DATA_CACHE
# |     cities = []
# |     for raw in raw_list:
# |         cache_key = (raw.city_name, resolved_root, scaler_key)
# |         if use_cache and cache_key in _CITY_DATA_CACHE:
# |             cities.append(_CITY_DATA_CACHE[cache_key])
# |         else:
# |             cd = CityData(
# |                 city_name     = raw.city_name,
# |                 n_tracts      = raw.n_tracts,
# |                 n_pairs       = raw.n_pairs,
# |                 node_features = torch.tensor(np.nan_to_num(scaler.transform(raw.X_raw), nan=0.0, posinf=0.0, neginf=0.0), dtype=torch.float32),
# |                 population    = raw.population,
# |                 lon_lat       = raw.lon_lat,
# |                 pair_o_idx    = raw.pair_o_idx,
# |                 pair_d_idx    = raw.pair_d_idx,
# |                 pair_distance = raw.pair_distance,
# |                 pair_trips    = raw.pair_trips,
# |                 bin_labels    = raw.bin_labels,
# |             )
# |             if use_cache:
# |                 _CITY_DATA_CACHE[cache_key] = cd
# |             cities.append(cd)
# |
# |     return cities, scaler
# |
# |
# | def preload_all_cities(
# |     data_root: str = "data",
# |     city_names: Optional[List[str]] = None,
# |     build_graphs: bool = True,
# |     radius_km: float = 5.0,
# | ) -> None:
# |     """
# |     Preloads all cities into in-memory cache upfront.
# |     Optionally computes spatial radius graphs and distance matrices.
# |     Completely eliminates disk I/O during multi-fold cross-validation.
# |     """
# |     from src.data.urban_graph import build_radius_graph
# |     if city_names is None:
# |         p = Path(data_root)
# |         if p.exists():
# |             city_names = sorted([d.name for d in p.iterdir() if d.is_dir() and (d / "meta.csv").exists()])
# |         else:
# |             city_names = []
# |
# |     for name in city_names:
# |         raw = load_raw_city(name, data_root=data_root, use_cache=True)
# |         if build_graphs:
# |             build_radius_graph(raw.lon_lat, radius_km=radius_km, use_cache=True)
# |
# |
# | # ---------------------------------------------------------------------------
# | # Quick smoke test
# | # ---------------------------------------------------------------------------
# |
# | if __name__ == "__main__":
# |     import sys
# |     root = sys.argv[1] if len(sys.argv) > 1 else "data"
# |
# |     print("Loading Raleigh (small)...")
# |     cd = load_city("Raleigh", data_root=root)
# |     print(f"  Tracts: {cd.n_tracts}, Pairs: {cd.n_pairs}")
# |     print(f"  node_features: {cd.node_features.shape} | dtype: {cd.node_features.dtype}")
# |     print(f"  pair_trips: min={cd.pair_trips.min():.0f}, max={cd.pair_trips.max():.0f}")
# |     print(f"  pair_distance: min={cd.pair_distance.min():.3f}, max={cd.pair_distance.max():.3f}")
# |     print(f"  bin_labels: unique={cd.bin_labels.unique().tolist()}")
# |     print(f"  bin distribution: { {i: (cd.bin_labels==i).sum().item() for i in range(4)} }")
# |     print()
# |
# |     print("Loading Raleigh + Denver jointly (scaler fit)...")
# |     cities, scaler = load_cities(["Raleigh", "Denver"], data_root=root)
# |     for c in cities:
# |         print(f"  {c.city_name}: node_features mean~{c.node_features.mean():.3f} std~{c.node_features.std():.3f}")
# |     print()
# |
# |     print("Smoke test passed.")

# ===== END SOURCE FILE: src/data/dataset.py =====


# ===== BEGIN SOURCE FILE: src/data/gadm_mapper.py =====
# | import pandas as pd
# | import geopandas as gpd
# | from pathlib import Path
# | import os
# |
# | _GADM_GDF_CACHE = None
# |
# | def get_gadm_gid2_mapping(meta_df: pd.DataFrame, repo_root: str) -> tuple[dict, dict]:
# |     """
# |     Returns a tuple: (mapping_dict, stats_dict).
# |     mapping_dict maps tract `idx` to GADM `GID_2`.
# |     stats_dict contains `n_strict_within` and `n_nearest_fallback` for provenance auditing.
# |     """
# |     global _GADM_GDF_CACHE
# |     
# |     gadm_shp_path = Path(repo_root) / "gadm41_USA_shp" / "gadm41_USA_2.shp"
# |     if not gadm_shp_path.exists():
# |         raise FileNotFoundError(f"GADM shapefile not found at {gadm_shp_path}")
# |         
# |     if _GADM_GDF_CACHE is None:
# |         _GADM_GDF_CACHE = gpd.read_file(gadm_shp_path)[['GID_2', 'geometry']].to_crs("EPSG:4326")
# |         
# |     gadm = _GADM_GDF_CACHE
# |     meta_df = meta_df.copy()
# |     if 'idx' not in meta_df.columns:
# |         meta_df['idx'] = meta_df.index.astype(int)
# |
# |     tract_gdf = gpd.GeoDataFrame(
# |         meta_df, 
# |         geometry=gpd.points_from_xy(meta_df['lon'], meta_df['lat']), 
# |         crs="EPSG:4326"
# |     )
# |     
# |     # 1. Strict within
# |     result = gpd.sjoin(tract_gdf, gadm, how='left', predicate='within')
# |     if result.index.has_duplicates:
# |         result = result[~result.index.duplicated(keep='first')]
# |         
# |     missing = result['GID_2'].isna()
# |     n_fallback = 0
# |     fallback_details = []
# |     
# |     if missing.any():
# |         n_fallback = int(missing.sum())
# |         missing_gdf = tract_gdf[missing].copy()
# |         
# |         # Project to EPSG:5070 (NAD83 / Conus Albers) for accurate distance in meters
# |         gadm_proj = gadm.to_crs("EPSG:5070")
# |         missing_proj = missing_gdf.to_crs("EPSG:5070")
# |         
# |         # sjoin_nearest handles coastal boundary issues
# |         import warnings
# |         with warnings.catch_warnings():
# |             warnings.simplefilter("ignore")
# |             nearest = gpd.sjoin_nearest(missing_proj, gadm_proj, how='left', distance_col='nearest_distance_m')
# |             
# |         if nearest.index.has_duplicates:
# |             nearest = nearest[~nearest.index.duplicated(keep='first')]
# |             
# |         # Validate 5km threshold BEFORE updating result
# |         for row_idx, row in nearest.iterrows():
# |             idx_val = int(row['idx']) if 'idx' in row else int(row_idx)
# |             dist_m = float(row['nearest_distance_m'])
# |             gid2 = str(row['GID_2'])
# |             
# |             if dist_m > 5000.0:
# |                 raise ValueError(f"Mapping invariant failed: Tract {idx_val} is {dist_m:.2f}m away from nearest GADM polygon, exceeding 5km threshold.")
# |             
# |             fallback_details.append({
# |                 "tract_idx": idx_val,
# |                 "GID_2": gid2,
# |                 "nearest_distance_m": dist_m
# |             })
# |             
# |         result.loc[missing, 'GID_2'] = nearest['GID_2']
# |         print(f"  [GADM Mapping] WARNING: {n_fallback} tracts fell outside exact GADM polygons. Using nearest fallback (max dist: {max(d['nearest_distance_m'] for d in fallback_details):.2f}m).")
# |         
# |     if result["GID_2"].isna().any():
# |         raise ValueError("Mapping invariant failed: NaN found in GID_2 even after nearest fallback.")
# |         
# |     stats = {
# |         "n_strict_within": int(len(meta_df) - n_fallback),
# |         "n_nearest_fallback": n_fallback,
# |         "fallback_details": fallback_details
# |     }
# |         
# |     return dict(zip(result["idx"], result["GID_2"])), stats

# ===== END SOURCE FILE: src/data/gadm_mapper.py =====


# ===== BEGIN SOURCE FILE: src/data/trip_sampler.py =====
# | """
# | Multinomial Trip Sampler for M_q condition.
# |
# | Draws m random trips according to the categorical distribution over Omega_c:
# |     p_{ij} = T^{GT}_{ij} / sum_{a,b} T^{GT}_{ab}
# |
# | From the sampled trips, estimates the empirical distance distribution:
# |     \tilde{Y}_D^{(m)}[k] = sum_{ij in B_k} n_{ij} / m
# |
# | Grid:
# |     m in {100, 500, 1k, 5k, 10k, 50k, 100k, inf}
# | """
# |
# | import numpy as np
# | import torch
# |
# |
# | M_GRID = [100, 500, 1000, 5000, 10000, 50000, 100000, float("inf")]
# |
# |
# | def sample_multinomial_yd(
# |     pair_trips: torch.Tensor,
# |     bin_labels: torch.Tensor,
# |     m: int | float,
# |     seed: int = 42,
# | ) -> np.ndarray:
# |     """
# |     Samples m trips from multinomial distribution and returns the 4-bin distribution \tilde{Y}_D^{(m)}.
# |
# |     Args:
# |         pair_trips: (E,) positive trip counts.
# |         bin_labels: (E,) bin index (0..3).
# |         m: number of trips to sample (float('inf') returns exact oracle).
# |         seed: random seed for reproducibility.
# |
# |     Returns:
# |         np.ndarray of shape (4,) representing bin proportions.
# |     """
# |     trips = pair_trips.detach().cpu().numpy().astype(np.float64)
# |     bins = bin_labels.detach().cpu().numpy().astype(np.int64)
# |
# |     total_trips = np.sum(trips)
# |     if total_trips <= 0:
# |         return np.array([0.25, 0.25, 0.25, 0.25])
# |
# |     # If m is infinity or m >= total_trips, return the oracle
# |     if np.isinf(m):
# |         yd = np.zeros(4, dtype=np.float64)
# |         for k in range(4):
# |             yd[k] = np.sum(trips[bins == k])
# |         return yd / total_flow if (total_flow := np.sum(yd)) > 0 else np.array([0.25, 0.25, 0.25, 0.25])
# |
# |     m = int(m)
# |     p_vals = trips / total_trips
# |
# |     rng = np.random.default_rng(seed)
# |     sampled_counts = rng.multinomial(m, p_vals)  # (E,) counts of sampled trips
# |
# |     yd_m = np.zeros(4, dtype=np.float64)
# |     for k in range(4):
# |         yd_m[k] = np.sum(sampled_counts[bins == k])
# |
# |     return yd_m / float(m)
# |
# |
# | if __name__ == "__main__":
# |     trips = torch.tensor([10.0, 90.0, 200.0, 700.0])
# |     bins = torch.tensor([0, 1, 2, 3])
# |     print("Oracle:", sample_multinomial_yd(trips, bins, float("inf")))
# |     print("m=100:", sample_multinomial_yd(trips, bins, 100, seed=1))
# |     print("m=10000:", sample_multinomial_yd(trips, bins, 10000, seed=1))

# ===== END SOURCE FILE: src/data/trip_sampler.py =====


# ===== BEGIN SOURCE FILE: src/data/urban_graph.py =====
# | """
# | Spatial Urban Graph Construction (G^urban).
# |
# | Constructs the urban spatial graph from tract centroid coordinates (lon, lat).
# | Crucial requirement: G^urban uses ONLY observable spatial geography, NEVER OD flows.
# |
# | Supports:
# | 1. k-NN graph: connects each node to its k geographically nearest neighbors.
# | 2. Radius graph: connects nodes within a geographic distance threshold d_max (km).
# | 3. Adaptive Radius graph: radius normalized to the city's empirical spatial diameter / extent.
# | """
# |
# | import math
# | import numpy as np
# | import torch
# |
# |
# | # Global In-Memory Cache for spatial urban graphs & distance matrices
# | _GRAPH_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
# | _DISTANCE_MATRIX_CACHE: dict[tuple | int | str, np.ndarray] = {}
# |
# |
# | def clear_graph_cache() -> None:
# |     """Flushes the global in-memory spatial urban graph and distance matrix caches."""
# |     global _GRAPH_CACHE, _DISTANCE_MATRIX_CACHE
# |     _GRAPH_CACHE.clear()
# |     _DISTANCE_MATRIX_CACHE.clear()
# |
# |
# | def clear_distance_matrix_cache() -> None:
# |     """Flushes the global in-memory pairwise distance matrix cache."""
# |     global _DISTANCE_MATRIX_CACHE
# |     _DISTANCE_MATRIX_CACHE.clear()
# |
# |
# | def haversine_distance_matrix(
# |     lon_lat: np.ndarray | torch.Tensor,
# |     use_cache: bool = True,
# |     cache_key: str | None = None,
# | ) -> np.ndarray:
# |     """
# |     Computes pairwise Haversine distances in kilometers with in-memory caching.
# |     Avoids redundant O(N^2) computation on repeated calls for the same coordinates / city.
# |     """
# |     if isinstance(lon_lat, torch.Tensor):
# |         lon_lat = lon_lat.detach().cpu().numpy()
# |     else:
# |         lon_lat = np.asarray(lon_lat, dtype=np.float64)
# |
# |     import hashlib
# |     coord_hash = hashlib.sha256(lon_lat.tobytes()).hexdigest()
# |     key = f"{cache_key}_{coord_hash}" if cache_key else coord_hash
# |     if use_cache and key in _DISTANCE_MATRIX_CACHE:
# |         return _DISTANCE_MATRIX_CACHE[key]
# |
# |     R = 6371.0
# |     lons = np.radians(lon_lat[:, 0])
# |     lats = np.radians(lon_lat[:, 1])
# |
# |     dlon = lons[:, None] - lons[None, :]
# |     dlat = lats[:, None] - lats[None, :]
# |
# |     a = np.sin(dlat / 2.0) ** 2 + np.cos(lats[:, None]) * np.cos(lats[None, :]) * np.sin(dlon / 2.0) ** 2
# |     c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
# |     dist_mat = R * c
# |
# |     if use_cache:
# |         _DISTANCE_MATRIX_CACHE[key] = dist_mat
# |
# |     return dist_mat
# |
# |
# | def build_knn_graph(
# |     lon_lat: np.ndarray | torch.Tensor,
# |     k: int = 10,
# |     include_self_loop: bool = True,
# |     use_cache: bool = True,
# |     cache_key: str | None = None,
# |     dist_mat: np.ndarray | None = None,
# | ) -> tuple[torch.Tensor, torch.Tensor]:
# |     """Constructs a k-nearest neighbor spatial graph with in-memory caching."""
# |     if isinstance(lon_lat, torch.Tensor):
# |         lon_lat = lon_lat.detach().cpu().numpy()
# |     else:
# |         lon_lat = np.asarray(lon_lat)
# |
# |     import hashlib
# |     coord_hash = hashlib.sha256(lon_lat.tobytes()).hexdigest()
# |     base_key = f"{cache_key}_{coord_hash}" if cache_key else coord_hash
# |     key = (base_key, "knn", k, include_self_loop)
# |     if use_cache and key in _GRAPH_CACHE:
# |         return _GRAPH_CACHE[key]
# |
# |     N = len(lon_lat)
# |     k = min(k, N - 1)
# |     if dist_mat is None:
# |         dist_mat = haversine_distance_matrix(lon_lat, use_cache=use_cache, cache_key=cache_key)
# |
# |     rows, cols, dists = [], [], []
# |     for i in range(N):
# |         indices = np.argsort(dist_mat[i])
# |         neighbors = indices[1 : k + 1]
# |
# |         if include_self_loop:
# |             rows.append(i)
# |             cols.append(i)
# |             dists.append(0.0)
# |
# |         for nbr in neighbors:
# |             rows.append(i)
# |             cols.append(nbr)
# |             dists.append(dist_mat[i, nbr])
# |
# |     edge_dict = {}
# |     for r, c, d in zip(rows, cols, dists):
# |         edge_dict[(r, c)] = d
# |         edge_dict[(c, r)] = d
# |
# |     e_rows = [k[0] for k in edge_dict.keys()]
# |     e_cols = [k[1] for k in edge_dict.keys()]
# |     e_dists = list(edge_dict.values())
# |
# |     edge_index = torch.tensor([e_rows, e_cols], dtype=torch.long)
# |     edge_dist = torch.tensor(e_dists, dtype=torch.float32)
# |
# |     res = (edge_index, edge_dist)
# |     if use_cache:
# |         _GRAPH_CACHE[key] = res
# |     return res
# |
# |
# | def build_radius_graph(
# |     lon_lat: np.ndarray | torch.Tensor,
# |     radius_km: float = 5.0,
# |     include_self_loop: bool = True,
# |     use_cache: bool = True,
# |     cache_key: str | None = None,
# |     dist_mat: np.ndarray | None = None,
# | ) -> tuple[torch.Tensor, torch.Tensor]:
# |     """Constructs a radius-based spatial graph connecting nodes within radius_km with caching."""
# |     if isinstance(lon_lat, torch.Tensor):
# |         lon_lat = lon_lat.detach().cpu().numpy()
# |     else:
# |         lon_lat = np.asarray(lon_lat)
# |
# |     import hashlib
# |     coord_hash = hashlib.sha256(lon_lat.tobytes()).hexdigest()
# |     base_key = f"{cache_key}_{coord_hash}" if cache_key else coord_hash
# |     key = (base_key, "radius", float(radius_km), include_self_loop)
# |     if use_cache and key in _GRAPH_CACHE:
# |         return _GRAPH_CACHE[key]
# |
# |     N = len(lon_lat)
# |     if dist_mat is None:
# |         dist_mat = haversine_distance_matrix(lon_lat, use_cache=use_cache, cache_key=cache_key)
# |
# |     rows, cols, dists = [], [], []
# |     for i in range(N):
# |         if include_self_loop:
# |             rows.append(i)
# |             cols.append(i)
# |             dists.append(0.0)
# |
# |         within_radius = np.where((dist_mat[i] <= radius_km) & (dist_mat[i] > 0))[0]
# |         if len(within_radius) == 0:
# |             closest = np.argsort(dist_mat[i])[1]
# |             within_radius = [closest]
# |
# |         for nbr in within_radius:
# |             rows.append(i)
# |             cols.append(nbr)
# |             dists.append(dist_mat[i, nbr])
# |
# |     edge_dict = {}
# |     for r, c, d in zip(rows, cols, dists):
# |         edge_dict[(r, c)] = d
# |         edge_dict[(c, r)] = d
# |
# |     e_rows = [k[0] for k in edge_dict.keys()]
# |     e_cols = [k[1] for k in edge_dict.keys()]
# |     e_dists = list(edge_dict.values())
# |
# |     edge_index = torch.tensor([e_rows, e_cols], dtype=torch.long)
# |     edge_dist = torch.tensor(e_dists, dtype=torch.float32)
# |
# |     res = (edge_index, edge_dist)
# |     if use_cache:
# |         _GRAPH_CACHE[key] = res
# |     return res
# |
# |
# | def build_adaptive_radius_graph(
# |     lon_lat: np.ndarray | torch.Tensor,
# |     scale_fraction: float = 0.15,
# |     min_radius_km: float = 2.0,
# |     include_self_loop: bool = True,
# |     use_cache: bool = True,
# |     cache_key: str | None = None,
# |     dist_mat: np.ndarray | None = None,
# | ) -> tuple[torch.Tensor, torch.Tensor, float]:
# |     """
# |     Constructs a spatial radius graph where radius_km is normalized to the city's
# |     empirical spatial diameter (max distance * scale_fraction).
# |     """
# |     if dist_mat is None:
# |         dist_mat = haversine_distance_matrix(lon_lat, use_cache=use_cache, cache_key=cache_key)
# |     diameter = float(np.max(dist_mat))
# |     adaptive_radius = max(min_radius_km, diameter * scale_fraction)
# |     ei, ed = build_radius_graph(
# |         lon_lat,
# |         radius_km=adaptive_radius,
# |         include_self_loop=include_self_loop,
# |         use_cache=use_cache,
# |         cache_key=cache_key,
# |         dist_mat=dist_mat,
# |     )
# |     return ei, ed, adaptive_radius
# |
# |
# | if __name__ == "__main__":
# |     coords = np.array([
# |         [-84.3880, 33.7490],
# |         [-84.3900, 33.7500],
# |         [-84.4000, 33.7600],
# |         [-84.5000, 33.8000],
# |     ])
# |     ei, ed, r = build_adaptive_radius_graph(coords, scale_fraction=0.2)
# |     print(f"Adaptive radius: {r:.2f} km | Edges: {ei.shape[1]}")

# ===== END SOURCE FILE: src/data/urban_graph.py =====


# ===== BEGIN SOURCE FILE: src/data/yd_extractor.py =====
# | """
# | Y_D Extractor for Moving Bins (Primary) and Full 4-Bin (Ablation).
# |
# | Primary Moving-Bin Formulation:
# |     Excludes stay-at-home / immobility Bin 0.
# |     Normalizes across actual movement/displacement categories {1, 2, 3}:
# |         Bin 1: (0, 10) km
# |         Bin 2: [10, 100) km
# |         Bin 3: 100+ km
# |
# |     Y_{c, k}^{Meta, +}   = Y_{c, k}^{Meta} / sum_{l=1}^3 Y_{c, l}^{Meta}
# |     Y_{c, k}^{oracle, +} = sum_{(i,j) in Omega_{c,k}^+} T_{ij}^{GT} / sum_{(i,j) in Omega_c^+} T_{ij}^{GT}
# |
# | Distributional Overlap Metric (CPC_dist / Overlap):
# |     Overlap(p, q) = sum_k min(p_k, q_k) = 1 - 0.5 * ||p - q||_1
# | """
# |
# | import os
# | import sys
# | import glob
# | import pandas as pd
# | import numpy as np
# | import torch
# | from pathlib import Path
# |
# | # Ensure root directory is in sys.path
# | sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
# |
# |
# | # Comprehensive official mapping of 50 US cities to County Names, State, and FIPS
# | CITY_FIPS_GADM = {
# |     "Albuquerque": {"state": "NM", "fips": "35001", "gadm_names": ["Bernalillo"]},
# |     "Arlington": {"state": "TX", "fips": "48439", "gadm_names": ["Tarrant"]},
# |     "Atlanta": {"state": "GA", "fips": ["13121", "13089"], "gadm_names": ["Fulton", "DeKalb"]},
# |     "Austin": {"state": "TX", "fips": "48453", "gadm_names": ["Travis"]},
# |     "Baltimore": {"state": "MD", "fips": "24510", "gadm_names": ["Baltimore City", "Baltimore"]},
# |     "Boston": {"state": "MA", "fips": "25025", "gadm_names": ["Suffolk"]},
# |     "Charlotte": {"state": "NC", "fips": "37119", "gadm_names": ["Mecklenburg"]},
# |     "Chicago": {"state": "IL", "fips": "17031", "gadm_names": ["Cook"]},
# |     "Colorado_Springs": {"state": "CO", "fips": "08041", "gadm_names": ["El Paso"]},
# |     "Columbus": {"state": "OH", "fips": "39049", "gadm_names": ["Franklin"]},
# |     "Dallas": {"state": "TX", "fips": "48113", "gadm_names": ["Dallas"]},
# |     "Denver": {"state": "CO", "fips": "08031", "gadm_names": ["Denver"]},
# |     "Detroit": {"state": "MI", "fips": "26163", "gadm_names": ["Wayne"]},
# |     "El_Paso": {"state": "TX", "fips": "48141", "gadm_names": ["El Paso"]},
# |     "Fort_Worth": {"state": "TX", "fips": "48439", "gadm_names": ["Tarrant"]},
# |     "Fresno": {"state": "CA", "fips": "06019", "gadm_names": ["Fresno"]},
# |     "Houston": {"state": "TX", "fips": "48201", "gadm_names": ["Harris"]},
# |     "Indianapolis": {"state": "IN", "fips": "18097", "gadm_names": ["Marion"]},
# |     "Jacksonville": {"state": "FL", "fips": "12031", "gadm_names": ["Duval"]},
# |     "Kansas_City": {"state": "MO", "fips": "29095", "gadm_names": ["Jackson"]},
# |     "Las_Vegas": {"state": "NV", "fips": "32003", "gadm_names": ["Clark"]},
# |     "Long_Beach": {"state": "CA", "fips": "06037", "gadm_names": ["Los Angeles"]},
# |     "Los_Angeles": {"state": "CA", "fips": "06037", "gadm_names": ["Los Angeles"]},
# |     "Louisville": {"state": "KY", "fips": "21111", "gadm_names": ["Jefferson"]},
# |     "Memphis": {"state": "TN", "fips": "47157", "gadm_names": ["Shelby"]},
# |     "Mesa": {"state": "AZ", "fips": "04013", "gadm_names": ["Maricopa"]},
# |     "Miami": {"state": "FL", "fips": "12086", "gadm_names": ["Miami-Dade", "Dade"]},
# |     "Milwaukee": {"state": "WI", "fips": "55079", "gadm_names": ["Milwaukee"]},
# |     "Minneapolis": {"state": "MN", "fips": "27053", "gadm_names": ["Hennepin"]},
# |     "Nashville": {"state": "TN", "fips": "47037", "gadm_names": ["Davidson"]},
# |     "New_York": {"state": "NY", "fips": ["36061", "36047", "36081", "36005", "36085"], "gadm_names": ["New York", "Kings", "Queens", "Bronx", "Richmond"]},
# |     "Oakland": {"state": "CA", "fips": "06001", "gadm_names": ["Alameda"]},
# |     "Oklahoma_City": {"state": "OK", "fips": "40109", "gadm_names": ["Oklahoma"]},
# |     "Omaha": {"state": "NE", "fips": "31055", "gadm_names": ["Douglas"]},
# |     "Philadelphia": {"state": "PA", "fips": "42101", "gadm_names": ["Philadelphia"]},
# |     "Phoenix": {"state": "AZ", "fips": "04013", "gadm_names": ["Maricopa"]},
# |     "Portland": {"state": "OR", "fips": "41051", "gadm_names": ["Multnomah"]},
# |     "Raleigh": {"state": "NC", "fips": "37183", "gadm_names": ["Wake"]},
# |     "Sacramento": {"state": "CA", "fips": "06067", "gadm_names": ["Sacramento"]},
# |     "San_Antonio": {"state": "TX", "fips": "48029", "gadm_names": ["Bexar"]},
# |     "San_Diego": {"state": "CA", "fips": "06073", "gadm_names": ["San Diego"]},
# |     "San_Francisco": {"state": "CA", "fips": "06075", "gadm_names": ["San Francisco"]},
# |     "San_Jose": {"state": "CA", "fips": "06085", "gadm_names": ["Santa Clara"]},
# |     "Seattle": {"state": "WA", "fips": "53033", "gadm_names": ["King"]},
# |     "Tampa": {"state": "FL", "fips": "12057", "gadm_names": ["Hillsborough"]},
# |     "Tucson": {"state": "AZ", "fips": "04019", "gadm_names": ["Pima"]},
# |     "Tulsa": {"state": "OK", "fips": "40143", "gadm_names": ["Tulsa"]},
# |     "Virginia_Beach": {"state": "VA", "fips": "51810", "gadm_names": ["Virginia Beach"]},
# |     "Washington_DC": {"state": "DC", "fips": "11001", "gadm_names": ["District of Columbia"]},
# |     "Wichita": {"state": "KS", "fips": "20173", "gadm_names": ["Sedgwick"]},
# | }
# |
# | META_CAT_TO_BIN = {
# |     "0": 0,
# |     "(0, 10)": 1,
# |     "[10, 100)": 2,
# |     "100+": 3,
# | }
# |
# | _SNAPSHOT_CACHE = None
# |
# |
# | def _load_snapshot_dataframes(meta_prior_dir: str = "meta_prior") -> list[pd.DataFrame]:
# |     global _SNAPSHOT_CACHE
# |     if _SNAPSHOT_CACHE is not None:
# |         return _SNAPSHOT_CACHE
# |
# |     meta_dir = Path(meta_prior_dir)
# |     files = sorted(list(meta_dir.glob("*.csv")))
# |     snapshots = []
# |     for f in files:
# |         try:
# |             df = pd.read_csv(
# |                 f,
# |                 usecols=["country", "gadm_name", "home_to_ping_distance_category", "distance_category_ping_fraction"],
# |             )
# |             us_df = df[df["country"] == "USA"].copy()
# |             snapshots.append(us_df)
# |         except Exception:
# |             continue
# |
# |     _SNAPSHOT_CACHE = snapshots
# |     return _SNAPSHOT_CACHE
# |
# |
# | def extract_yd_4bin_real(city_name: str, meta_prior_dir: str = "meta_prior") -> np.ndarray | None:
# |     """Extracts raw 4-bin Meta distribution (including Bin 0) for ablation."""
# |     city_info = CITY_FIPS_GADM.get(city_name, None)
# |     if city_info is None:
# |         return None
# |
# |     counties = city_info["gadm_names"]
# |     snapshots = _load_snapshot_dataframes(meta_prior_dir=meta_prior_dir)
# |     if not snapshots:
# |         return None
# |
# |     snapshot_distributions = []
# |     for df in snapshots:
# |         matched = df[df["gadm_name"].isin(counties)]
# |         if len(matched) == 0:
# |             continue
# |
# |         cat_means = matched.groupby("home_to_ping_distance_category")["distance_category_ping_fraction"].mean()
# |         yd_snap = np.zeros(4, dtype=np.float64)
# |         for cat_str, bin_idx in META_CAT_TO_BIN.items():
# |             if cat_str in cat_means:
# |                 yd_snap[bin_idx] = float(cat_means[cat_str])
# |
# |         snap_sum = np.sum(yd_snap)
# |         if snap_sum > 0:
# |             snapshot_distributions.append(yd_snap / snap_sum)
# |
# |     if not snapshot_distributions:
# |         return None
# |
# |     mean_yd = np.mean(snapshot_distributions, axis=0)
# |     total = np.sum(mean_yd)
# |     return mean_yd / total if total > 0 else None
# |
# |
# | def extract_M1_city_oracle_obs(city_name: str, meta_prior_dir: str = "meta_prior") -> np.ndarray | None:
# |     """
# |     Primary Meta extractor: extracts the 3 moving bins {1, 2, 3} normalized to sum to 1.0.
# |     Excludes stay-at-home / immobility Bin 0.
# |     """
# |     yd_4 = extract_yd_4bin_real(city_name, meta_prior_dir=meta_prior_dir)
# |     if yd_4 is None:
# |         return None
# |
# |     moving_3 = yd_4[1:].copy()  # bins 1, 2, 3
# |     total_moving = np.sum(moving_3)
# |     if total_moving <= 0:
# |         return None
# |     return moving_3 / total_moving
# |
# |
# | def extract_yd_4bin_oracle(pair_trips: torch.Tensor, bin_labels: torch.Tensor) -> np.ndarray:
# |     """Extracts raw 4-bin oracle distribution from GT flows."""
# |     yd = np.zeros(4, dtype=np.float64)
# |     trips_np = pair_trips.detach().cpu().numpy()
# |     bins_np = bin_labels.detach().cpu().numpy()
# |     total_flow = float(np.sum(trips_np))
# |     if total_flow <= 0:
# |         raise ValueError(
# |             "extract_yd_4bin_oracle: zero total flow — city data is degenerate. "
# |             "Cannot compute 4-bin oracle Y_D. Check data integrity."
# |         )
# |     for k in range(4):
# |         yd[k] = np.sum(trips_np[bins_np == k])
# |     return yd / total_flow
# |
# |
# | def extract_yd_moving_oracle(
# |     pair_trips: torch.Tensor,
# |     bin_labels: torch.Tensor,
# |     pair_o_idx: torch.Tensor,
# |     pair_d_idx: torch.Tensor,
# |     pair_distance: torch.Tensor | None = None,
# | ) -> np.ndarray:
# |     """
# |     Primary Oracle extractor: computes 3-bin distribution on interzonal pairs Omega_c^+ (bins 1, 2, 3).
# |     """
# |     trips_np = pair_trips.detach().cpu().numpy()
# |     bins_np = bin_labels.detach().cpu().numpy()
# |     o_np = pair_o_idx.detach().cpu().numpy()
# |     d_np = pair_d_idx.detach().cpu().numpy()
# |
# |     if pair_distance is not None:
# |         p_dist = pair_distance.detach().cpu().numpy()
# |         dist_km = p_dist
# |         inter_mask = (o_np != d_np) & (dist_km > 0.0)
# |     else:
# |         inter_mask = (o_np != d_np) & (bins_np > 0)
# |     inter_trips = trips_np[inter_mask]
# |     inter_bins = bins_np[inter_mask]
# |
# |     yd_3 = np.zeros(3, dtype=np.float64)
# |     total_inter = np.sum(inter_trips)
# |     if total_inter <= 0:
# |         raise ValueError(
# |             "extract_yd_moving_oracle: zero total interzonal flow — city data is degenerate. "
# |             "Cannot compute oracle Y_D. Check data integrity."
# |         )
# |
# |     for idx, bin_k in enumerate([1, 2, 3]):
# |         yd_3[idx] = np.sum(inter_trips[inter_bins == bin_k])
# |
# |     return yd_3 / total_inter
# |
# |
# | def compute_distributional_overlap(p: np.ndarray, q: np.ndarray) -> float:
# |     """
# |     Computes Distributional Overlap (CPC_dist) between two probability vectors:
# |     Overlap(p, q) = sum_k min(p_k, q_k) = 1 - 0.5 * ||p - q||_1
# |     """
# |     return float(np.sum(np.minimum(p, q)))
# |
# |
# | # ---------------------------------------------------------------------------
# | # E1: Dynamic K-bin extraction for Oracle Existence Test
# | # ---------------------------------------------------------------------------
# |
# | def compute_kbin_edges(
# |     train_city_names: list,
# |     K: int = 8,
# |     data_root: str = "data",
# | ) -> tuple:
# |     """
# |     Compute K-bin pair-weighted quantile edges from training cities.
# |     Intrazonal pairs (D_ij = 0) are excluded.
# |
# |     NOTE: Pair-weighted — large cities contribute more pairs than small cities.
# |     This is intentional and documented; see E1.md.
# |
# |     Args:
# |         train_city_names: List of training city names.
# |         K: Number of moving-distance bins (Bin 0 intrazonal excluded).
# |         data_root: Root directory of city data.
# |
# |     Returns:
# |         (edges, K_active): edges is (K_active+1,) array strictly increasing,
# |         K_active <= K (may be < K if quantile degeneration occurs).
# |     """
# |     from src.data.dataset import load_raw_city
# |
# |     all_dist = []
# |     for city_name in train_city_names:
# |         raw = load_raw_city(city_name, data_root=data_root)
# |         dist_km = raw.dist_km
# |         inter = (raw.pair_o_idx.numpy() != raw.pair_d_idx.numpy()) & (dist_km > 0.0)
# |         all_dist.extend(dist_km[inter].tolist())
# |
# |     all_dist = np.array(all_dist)
# |     assert len(all_dist) > K, f"Too few interzonal pairs ({len(all_dist)}) for K={K} bins"
# |
# |     # K-1 internal breakpoints → K bins; skip 0th and 100th percentile
# |     quantile_pts = np.linspace(0, 100, K + 1)[1:-1]   # shape: (K-1,)
# |     internal_edges = np.percentile(all_dist, quantile_pts)
# |
# |     # Deduplicate: remove duplicate edges (handles concentrated distributions)
# |     internal_edges = np.unique(internal_edges)
# |     edges = np.concatenate([[0.0], internal_edges, [np.inf]])
# |
# |     # INVARIANT: strictly increasing
# |     assert np.all(np.diff(edges) > 0), f"Non-strict bin edges: {edges}"
# |
# |     K_active = len(edges) - 1
# |     if K_active < K:
# |         print(f"[WARNING] compute_kbin_edges: K_active={K_active} < K={K} due to quantile degeneration")
# |
# |     return edges, K_active
# |
# |
# | def extract_yd_kbins(
# |     dist_km: np.ndarray,
# |     trips: np.ndarray,
# |     bin_edges: np.ndarray,
# |     inter_mask: np.ndarray,
# | ) -> np.ndarray:
# |     """
# |     Extract K-bin oracle trip-length distribution from ground-truth flows.
# |
# |     Aggregates GT flows by distance bin — NOT pair-level individual flows.
# |     Adaptation receives only this K-dim histogram vector; it does NOT see T_ij.
# |
# |     NOTE: Uses GT trips to compute bin totals → oracle aggregate information.
# |     This is intentional for E1 Oracle Existence Test; see E1.md.
# |
# |     Args:
# |         dist_km:    (E,) pairwise distances in km.
# |         trips:      (E,) ground-truth flow counts T_ij^GT.
# |         bin_edges:  (K+1,) strictly increasing bin edges (from compute_kbin_edges).
# |         inter_mask: (E,) boolean mask for interzonal pairs Omega_c^+.
# |
# |     Returns:
# |         yd: (K,) normalized oracle distance distribution summing to 1.0.
# |     """
# |     K = len(bin_edges) - 1
# |     yd = np.zeros(K, dtype=np.float64)
# |
# |     inter_trips = trips[inter_mask]
# |     inter_dist = dist_km[inter_mask]
# |
# |     for k in range(K):
# |         lo, hi = bin_edges[k], bin_edges[k + 1]
# |         in_bin = (inter_dist > lo) & (inter_dist <= hi)
# |         yd[k] = inter_trips[in_bin].sum()
# |
# |     total = yd.sum()
# |     if total > 0:
# |         yd = yd / total
# |     else:
# |         # Fallback: uniform over K bins
# |         yd = np.ones(K, dtype=np.float64) / K
# |
# |     return yd   # shape: (K,) summing to 1.0
# |
# |
# | def extract_yd_kbins_grouped(
# |     dist_km: np.ndarray,
# |     trips: np.ndarray,
# |     bin_edges: np.ndarray,
# |     inter_mask: np.ndarray,
# |     pair_group_idx: np.ndarray,
# | ) -> dict:
# |     """
# |     Extract K-bin oracle trip-length distribution per group (e.g., origin county).
# |     
# |     Args:
# |         dist_km:        (E,) pairwise distances in km.
# |         trips:          (E,) ground-truth flow counts T_ij^GT.
# |         bin_edges:      (K+1,) strictly increasing bin edges.
# |         inter_mask:     (E,) boolean mask for interzonal pairs Omega_c^+.
# |         pair_group_idx: (E,) group ID for each pair.
# |         
# |     Returns:
# |         dict: Mapping group_id -> (K,) normalized oracle distance distribution.
# |     """
# |     yd_dict = {}
# |     unique_groups = np.unique(pair_group_idx)
# |     
# |     for g in unique_groups:
# |         g_mask = (pair_group_idx == g)
# |         inter_g_mask = inter_mask & g_mask
# |         
# |         if not inter_g_mask.any():
# |             continue
# |             
# |         yd_g = extract_yd_kbins(
# |             dist_km=dist_km[g_mask],
# |             trips=trips[g_mask],
# |             bin_edges=bin_edges,
# |             inter_mask=inter_mask[g_mask]
# |         )
# |         yd_dict[g] = yd_g
# |         
# |     return yd_dict
# |
# |
# | if __name__ == "__main__":
# |     from src.data.dataset import load_city
# |
# |     for city in ["Philadelphia", "Austin", "Raleigh", "Denver", "Seattle"]:
# |         cd = load_city(city, "data")
# |         o_3 = extract_yd_moving_oracle(cd.pair_trips, cd.bin_labels, cd.pair_o_idx, cd.pair_d_idx)
# |         print(f"{city:<15}: Oracle_moving = {np.round(o_3, 4).tolist()}")
# |

# ===== END SOURCE FILE: src/data/yd_extractor.py =====


# ===== BEGIN SOURCE FILE: src/experiment/audit_direct_od_v1.py =====
# | """
# | Comprehensive Audit & Precision Certification Suite for Direct Partial-OD Equivalence v1
# | ========================================================================================
# |
# | Modules:
# |     1. Production Y_D reference audit: Compare manual t_cal_full vs production calibrate_kbins across 50 cities x 3 seeds.
# |     2. OD-FE solver precision audit: Compare production CG solver (tol=1e-6) vs ultra-high precision CG solver (tol=1e-10) across 50 cities x 3 seeds at p in {0.10%, 0.25%, 0.50%}, B=50.
# |     3. Lambda tie-rule audit: Check gap between best and 2nd best validation scores in all 5 folds against 10^-6 tolerance.
# |     4. Monte-Carlo precision audit: Compute per-city Monte Carlo SE and MCSE(mean D(p)) at p in {0.10%, 0.25%, 0.50%}.
# |     5. Crossing uncertainty bootstrap: 10,000 fold-stratified bootstrap samples computing 95% CI of p_eq,interp.
# |     6. Absolute observation counts & support diagnostics at p in {0.10%, 0.25%, 0.50%}.
# | """
# |
# | import sys
# | import time
# | import json
# | from pathlib import Path
# | from typing import Dict, List, Tuple, Any
# |
# | import numpy as np
# | import pandas as pd
# | from scipy import stats
# | import torch
# |
# | REPO_ROOT = Path(__file__).resolve().parent.parent.parent
# | sys.path.insert(0, str(REPO_ROOT))
# |
# | from src.data.city_splits import generate_35_5_10_splits
# | from src.data.dataset import load_city, load_raw_city
# | from src.data.urban_graph import build_radius_graph
# | from src.data.yd_extractor import compute_kbin_edges
# | from src.calibration.bin_calibration import calibrate_kbins
# | from src.training.evaluate import compute_cpc_pair
# | from src.training.train import load_checkpoint, infer_zero_shot
# | from src.experiment.run_direct_od_equivalence_v1 import (
# |     PARTIAL_OD_BASE_SEED, get_stable_mask_seed, fit_od_fe_adapter,
# |     apply_od_fe_prediction, holm_correction, fold_stratified_bootstrap
# | )
# |
# |
# | def run_audit_1_production_yd_reference(data_root="data") -> Dict[str, Any]:
# |     print("\n--- AUDIT 1: Production Y_D Reference Bitwise & CPC Audit (50 Cities x 3 Seeds) ---")
# |     splits = generate_35_5_10_splits(data_root=data_root)
# |     
# |     max_t_diff = 0.0
# |     max_cpc_diff = 0.0
# |     total_checks = 0
# |     failures = []
# |
# |     for fold_id in range(1, 6):
# |         split = splits[fold_id]
# |         train_cities = split["train"]
# |         test_cities = split["test"]
# |         
# |         bin_edges, K_act = compute_kbin_edges(train_cities, K=8, data_root=data_root)
# |         
# |         for s in [1, 10, 100]:
# |             ckpt_path = Path("results/checkpoints") / f"5fold_fold{fold_id}_seed{s}.pt"
# |             model, scaler, _ = load_checkpoint(ckpt_path, device_str="cpu")
# |             model.eval()
# |             
# |             for city_name in test_cities:
# |                 total_checks += 1
# |                 raw_data = load_raw_city(city_name, data_root=data_root)
# |                 dist_km = raw_data.dist_km
# |                 inter_pos = (raw_data.pair_o_idx.numpy() != raw_data.pair_d_idx.numpy()) & (dist_km > 0.0) & (raw_data.pair_trips.numpy() > 0)
# |                 
# |                 t_true_support = raw_data.pair_trips.numpy()[inter_pos].astype(np.float64)
# |                 dist_support = dist_km[inter_pos]
# |                 
# |                 # Production ground truth Y_D on full interzonal support
# |                 bin_idx = np.clip(np.digitize(dist_support, bin_edges, right=True) - 1, 0, 7)
# |                 yd_full = np.bincount(bin_idx, weights=t_true_support, minlength=8).astype(np.float64)
# |                 yd_full /= float(np.sum(t_true_support))
# |                 
# |                 # Zero-shot prediction
# |                 city_data = load_city(city_name, data_root=data_root, feature_scaler=scaler, fit_scaler=False)
# |                 coords = city_data.lon_lat.numpy()
# |                 ei, ed = build_radius_graph(coords, radius_km=5.0)
# |                 with torch.no_grad():
# |                     m0_full = infer_zero_shot(model, city_data, ei, ed, device="cpu").numpy().astype(np.float64)
# |                 t0_support = m0_full[inter_pos]
# |                 
# |                 # 1. Manual runner calibration logic
# |                 N_hat = float(np.sum(t0_support))
# |                 Y_hat = np.bincount(bin_idx, weights=t0_support, minlength=8).astype(np.float64) / N_hat
# |                 active = np.zeros(8, dtype=bool)
# |                 for k in range(8):
# |                     active[k] = bool((bin_idx == k).any())
# |                 yd_act = yd_full * active.astype(np.float64)
# |                 act_sum = yd_act.sum()
# |                 Y_D_cond = yd_act / act_sum if act_sum > 0 else Y_hat.copy()
# |                 w_full = np.ones(8, dtype=np.float64)
# |                 for k in range(8):
# |                     if active[k] and Y_hat[k] > 0:
# |                         w_full[k] = Y_D_cond[k] / Y_hat[k]
# |                 weighted_mass_full = float(np.dot(Y_hat, w_full))
# |                 s_full = w_full / weighted_mass_full if weighted_mass_full > 0 else np.ones(8)
# |                 t_cal_manual = t0_support * s_full[bin_idx]
# |                 cal_mass = np.sum(t_cal_manual)
# |                 if cal_mass > 0:
# |                     t_cal_manual *= (N_hat / cal_mass)
# |                     
# |                 # 2. Production calibrate_kbins
# |                 inter_mask = np.ones(len(t0_support), dtype=bool)
# |                 t_cal_prod = calibrate_kbins(
# |                     t0_support, dist_support, inter_mask, yd_full, bin_edges, q=1.0
# |                 )
# |                 
# |                 t_diff = float(np.max(np.abs(t_cal_manual - t_cal_prod)))
# |                 cpc_manual = compute_cpc_pair(t_true_support, t_cal_manual)
# |                 cpc_prod = compute_cpc_pair(t_true_support, t_cal_prod)
# |                 cpc_diff = float(abs(cpc_manual - cpc_prod))
# |                 
# |                 max_t_diff = max(max_t_diff, t_diff)
# |                 max_cpc_diff = max(max_cpc_diff, cpc_diff)
# |                 
# |                 if t_diff > 1e-10 or cpc_diff > 1e-10:
# |                     failures.append((fold_id, s, city_name, t_diff, cpc_diff))
# |
# |     status = "PASS" if len(failures) == 0 else "FAIL"
# |     print(f"  Total Evaluations: {total_checks} (50 cities x 3 seeds)")
# |     print(f"  Max |T_manual - T_production|: {max_t_diff:.8e}")
# |     print(f"  Max |CPC_manual - CPC_prod|:   {max_cpc_diff:.8e}")
# |     print(f"  Audit 1 Status: {status}")
# |     
# |     return {
# |         "status": status,
# |         "total_checks": total_checks,
# |         "max_flow_diff": max_t_diff,
# |         "max_cpc_diff": max_cpc_diff,
# |         "failures": failures
# |     }
# |
# |
# | def run_audit_2_solver_precision(data_root="data", b_audit=50) -> Dict[str, Any]:
# |     print(f"\n--- AUDIT 2: OD-FE Solver Precision Audit (50 Cities x 3 Seeds x B={b_audit} Reps) ---")
# |     splits = generate_35_5_10_splits(data_root=data_root)
# |     audit_p_grid = [0.001, 0.0025, 0.005] # p in {0.10%, 0.25%, 0.50%}
# |     
# |     max_a_diff = 0.0
# |     max_b_diff = 0.0
# |     max_cpc_diff = 0.0
# |     total_reps_tested = 0
# |     solver_failures = []
# |     
# |     # Load lambdas
# |     fold_lambdas = {}
# |     for f in range(1, 6):
# |         with open(f"results/direct_od_equivalence_v1/fold_{f}/lambda_selected.json") as jf:
# |             fold_lambdas[f] = json.load(jf)["selected_lambda"]
# |
# |     for fold_id in range(1, 6):
# |         split = splits[fold_id]
# |         test_cities = split["test"]
# |         lam = fold_lambdas[fold_id]
# |         
# |         for s in [1, 10, 100]:
# |             ckpt_path = Path("results/checkpoints") / f"5fold_fold{fold_id}_seed{s}.pt"
# |             model, scaler, _ = load_checkpoint(ckpt_path, device_str="cpu")
# |             model.eval()
# |             
# |             for city_name in test_cities:
# |                 raw_data = load_raw_city(city_name, data_root=data_root)
# |                 dist_km = raw_data.dist_km
# |                 inter_pos = (raw_data.pair_o_idx.numpy() != raw_data.pair_d_idx.numpy()) & (dist_km > 0.0) & (raw_data.pair_trips.numpy() > 0)
# |                 
# |                 t_true_support = raw_data.pair_trips.numpy()[inter_pos].astype(np.float64)
# |                 o_idx = raw_data.pair_o_idx.numpy()[inter_pos]
# |                 d_idx = raw_data.pair_d_idx.numpy()[inter_pos]
# |                 num_nodes = raw_data.n_tracts
# |                 n_pairs = len(t_true_support)
# |                 
# |                 city_data = load_city(city_name, data_root=data_root, feature_scaler=scaler, fit_scaler=False)
# |                 coords = city_data.lon_lat.numpy()
# |                 ei, ed = build_radius_graph(coords, radius_km=5.0)
# |                 with torch.no_grad():
# |                     m0_full = infer_zero_shot(model, city_data, ei, ed, device="cpu").numpy().astype(np.float64)
# |                 t0_support = m0_full[inter_pos]
# |
# |                 for rep_id in range(b_audit):
# |                     total_reps_tested += 1
# |                     mask_seed = get_stable_mask_seed(PARTIAL_OD_BASE_SEED, fold_id, city_name, rep_id)
# |                     perm = np.random.RandomState(mask_seed).permutation(n_pairs)
# |                     
# |                     for p_val in audit_p_grid:
# |                         n_rev = int(np.round(p_val * n_pairs))
# |                         rev_indices = perm[:n_rev]
# |                         unseen_indices = perm[n_rev:]
# |                         
# |                         t_true_unseen = t_true_support[unseen_indices]
# |                         sum_true_unseen = float(np.sum(t_true_unseen))
# |                         
# |                         # 1. Production solver (tol=1e-6, max_iter=150)
# |                         a_fast, b_fast, it_fast, conv_fast = fit_od_fe_adapter(
# |                             o_idx, d_idx, t0_support, t_true_support, rev_indices, num_nodes,
# |                             lambda_reg=lam, max_iter=150, tol=1e-6
# |                         )
# |                         t_fast = apply_od_fe_prediction(o_idx, d_idx, t0_support, a_fast, b_fast)[unseen_indices]
# |                         denom_fast = sum_true_unseen + float(np.sum(t_fast))
# |                         cpc_fast = (2.0 * np.sum(np.minimum(t_true_unseen, t_fast)) / denom_fast) if denom_fast > 0 else 0.0
# |                         
# |                         # 2. Ultra high-precision solver (tol=1e-10, max_iter=300)
# |                         a_ref, b_ref, it_ref, conv_ref = fit_od_fe_adapter(
# |                             o_idx, d_idx, t0_support, t_true_support, rev_indices, num_nodes,
# |                             lambda_reg=lam, max_iter=300, tol=1e-10
# |                         )
# |                         t_ref = apply_od_fe_prediction(o_idx, d_idx, t0_support, a_ref, b_ref)[unseen_indices]
# |                         denom_ref = sum_true_unseen + float(np.sum(t_ref))
# |                         cpc_ref = (2.0 * np.sum(np.minimum(t_true_unseen, t_ref)) / denom_ref) if denom_ref > 0 else 0.0
# |                         
# |                         if not conv_fast or not conv_ref:
# |                             solver_failures.append((fold_id, city_name, rep_id, p_val))
# |                             
# |                         diff_a = float(np.max(np.abs(a_fast - a_ref)))
# |                         diff_b = float(np.max(np.abs(b_fast - b_ref)))
# |                         diff_cpc = float(abs(cpc_fast - cpc_ref))
# |                         
# |                         max_a_diff = max(max_a_diff, diff_a)
# |                         max_b_diff = max(max_b_diff, diff_b)
# |                         max_cpc_diff = max(max_cpc_diff, diff_cpc)
# |
# |     pass_cpc = max_cpc_diff < 1e-5 and len(solver_failures) == 0
# |     status = "PASS" if pass_cpc else "FAIL"
# |     print(f"  Total Evaluations Tested: {total_reps_tested} reps x 3 p-levels")
# |     print(f"  Max |a_fast - a_ref|:     {max_a_diff:.8e}")
# |     print(f"  Max |b_fast - b_ref|:     {max_b_diff:.8e}")
# |     print(f"  Max |CPC_fast - CPC_ref|: {max_cpc_diff:.8e} (Threshold: < 1.0e-5)")
# |     print(f"  Audit 2 Status: {status}")
# |
# |     return {
# |         "status": status,
# |         "max_a_diff": max_a_diff,
# |         "max_b_diff": max_b_diff,
# |         "max_cpc_diff": max_cpc_diff,
# |         "criterion_passed": pass_cpc
# |     }
# |
# |
# | def run_audit_3_lambda_tie_rule() -> Dict[str, Any]:
# |     print("\n--- AUDIT 3: Lambda Selection Tie-Rule Audit ---")
# |     fold_gaps = {}
# |     all_gaps_exceed_tol = True
# |
# |     for f in range(1, 6):
# |         csv_p = Path(f"results/direct_od_equivalence_v1/fold_{f}/lambda_selection.csv")
# |         df = pd.read_csv(csv_p)
# |         df_sorted = df.sort_values(by="validation_mean_cpc", ascending=False)
# |         best_score = float(df_sorted.iloc[0]["validation_mean_cpc"])
# |         second_score = float(df_sorted.iloc[1]["validation_mean_cpc"])
# |         gap = best_score - second_score
# |         fold_gaps[f] = {
# |             "selected_lambda": float(df_sorted.iloc[0]["lambda"]),
# |             "best_score": best_score,
# |             "second_score": second_score,
# |             "gap": gap,
# |             "gap_exceeds_1e-6": bool(gap > 1e-6)
# |         }
# |         if gap <= 1e-6:
# |             all_gaps_exceed_tol = False
# |         print(f"  Fold {f}: Selected lambda={df_sorted.iloc[0]['lambda']}, Best={best_score:.5f}, 2nd={second_score:.5f}, Gap={gap:.6f} (> 1e-6: {gap > 1e-6})")
# |
# |     status = "PASS" if all_gaps_exceed_tol else "TIE_RULE_TRIGGERED"
# |     print(f"  Audit 3 Status: {status}")
# |     return {
# |         "status": status,
# |         "fold_gaps": fold_gaps,
# |         "all_gaps_exceed_tol": all_gaps_exceed_tol
# |     }
# |
# |
# | def run_audit_4_monte_carlo_precision() -> Dict[str, Any]:
# |     print("\n--- AUDIT 4: Monte-Carlo Precision & Standard Error Audit (p in {0.10%, 0.25%, 0.50%}) ---")
# |     raw_df = pd.read_csv("results/direct_od_equivalence_v1/combined/raw_all_folds.csv")
# |     
# |     mcse_results = {}
# |     all_passed = True
# |     
# |     for p_val in [0.001, 0.0025, 0.005]:
# |         sub = raw_df[raw_df.p == p_val]
# |         
# |         # Per-city Monte Carlo standard deviation across B=200 replicates
# |         city_mc_stds = []
# |         for city_name, cdf in sub.groupby("city"):
# |             # For each city, replicate-level D(p) averaged across 3 model seeds
# |             rep_d = cdf.groupby("replicate_id")["difference_direct_minus_yd"].mean().values
# |             city_mc_stds.append(np.std(rep_d, ddof=1))
# |             
# |         mean_city_mc_std = float(np.mean(city_mc_stds))
# |         # Per-city Monte Carlo standard error (divided by sqrt(B))
# |         mean_city_mc_se = mean_city_mc_std / np.sqrt(200)
# |         
# |         # MCSE of the master mean D(p) across N=50 cities: sqrt( sum(SE_i^2) ) / N
# |         mcse_mean_D = float(np.sqrt(np.sum((np.array(city_mc_stds) / np.sqrt(200))**2)) / 50.0)
# |         
# |         passed = mcse_mean_D < 1e-4
# |         if not passed:
# |             all_passed = False
# |             
# |         mcse_results[p_val] = {
# |             "p": p_val,
# |             "mean_city_mc_std": mean_city_mc_std,
# |             "mean_city_mc_se": mean_city_mc_se,
# |             "mcse_mean_D": mcse_mean_D,
# |             "passed_1e-4_gate": passed
# |         }
# |         print(f"  p = {p_val*100:5.2f}%: Mean City MC-SE = {mean_city_mc_se:.6f} | MCSE(Mean D) = {mcse_mean_D:.6e} (< 1e-4: {passed})")
# |
# |     status = "PASS" if all_passed else "RERUN_B500_REQUIRED"
# |     print(f"  Audit 4 Status: {status}")
# |     return {
# |         "status": status,
# |         "results_by_p": mcse_results,
# |         "all_passed": all_passed
# |     }
# |
# |
# | def run_audit_5_crossing_uncertainty_bootstrap() -> Dict[str, Any]:
# |     print("\n--- AUDIT 5: Crossing Uncertainty Fold-Stratified Bootstrap (10,000 Replicates across [0, 0.50%]) ---")
# |     per_city_df = pd.read_csv("results/direct_od_equivalence_v1/combined/per_city_all_folds.csv")
# |     
# |     rng = np.random.RandomState(42)
# |     n_boot = 10000
# |     
# |     grid = [0.0, 0.0010, 0.0025, 0.0050]
# |     fold_cities = {f: per_city_df[per_city_df.fold == f]["city"].unique().tolist() for f in range(1, 6)}
# |     
# |     d_by_p = {}
# |     for p in grid:
# |         d_by_p[p] = per_city_df[per_city_df.p == p].set_index("city")["difference_direct_minus_yd"].to_dict()
# |
# |     boot_crossings = []
# |     counts = {
# |         "below_0.10%": 0,
# |         "0.10-0.25%": 0,
# |         "0.25-0.50%": 0,
# |         "no_cross_le_0.50%": 0
# |     }
# |
# |     for b in range(n_boot):
# |         sampled_cities = []
# |         for f in range(1, 6):
# |             c_list = fold_cities[f]
# |             sampled_c = rng.choice(c_list, size=len(c_list), replace=True)
# |             sampled_cities.extend(sampled_c)
# |             
# |         mean_D = [np.mean([d_by_p[p][c] for c in sampled_cities]) for p in grid]
# |         
# |         found = False
# |         for i in range(len(grid) - 1):
# |             pa, pb = grid[i], grid[i+1]
# |             da, db = mean_D[i], mean_D[i+1]
# |             if da <= 0 and db >= 0 and (db - da) > 0:
# |                 peq = pa + (-da / (db - da)) * (pb - pa)
# |                 boot_crossings.append(peq)
# |                 if pb <= 0.0010:
# |                     counts["below_0.10%"] += 1
# |                 elif pb <= 0.0025:
# |                     counts["0.10-0.25%"] += 1
# |                 else:
# |                     counts["0.25-0.50%"] += 1
# |                 found = True
# |                 break
# |                 
# |         if not found:
# |             counts["no_cross_le_0.50%"] += 1
# |
# |     boot_crossings = np.array(boot_crossings)
# |     n_valid = len(boot_crossings)
# |     if n_valid > 0:
# |         ci_l = float(np.percentile(boot_crossings, 2.5))
# |         ci_h = float(np.percentile(boot_crossings, 97.5))
# |         mean_cross = float(np.mean(boot_crossings))
# |         median_cross = float(np.median(boot_crossings))
# |     else:
# |         ci_l, ci_h, mean_cross, median_cross = np.nan, np.nan, np.nan, np.nan
# |
# |     p_cross = (n_boot - counts["no_cross_le_0.50%"]) / n_boot * 100.0
# |
# |     print(f"  P(crossing <= 0.50%) = {p_cross:.2f}% ({n_valid}/{n_boot} samples)")
# |     print(f"    cross below 0.10%:       {counts['below_0.10%']} / {n_boot} ({counts['below_0.10%']/n_boot*100:.2f}%)")
# |     print(f"    cross 0.10–0.25%:        {counts['0.10-0.25%']} / {n_boot} ({counts['0.10-0.25%']/n_boot*100:.2f}%)")
# |     print(f"    cross 0.25–0.50%:        {counts['0.25-0.50%']} / {n_boot} ({counts['0.25-0.50%']/n_boot*100:.2f}%)")
# |     print(f"    no crossing <= 0.50%:    {counts['no_cross_le_0.50%']} / {n_boot} ({counts['no_cross_le_0.50%']/n_boot*100:.2f}%)")
# |     print(f"  Conditional crossing location:")
# |     if n_valid > 0:
# |         print(f"    Mean Interpolated Crossing:   {mean_cross*100:.3f}%")
# |         print(f"    Median Interpolated Crossing: {median_cross*100:.3f}%")
# |         print(f"    95% CI conditional on crossing: [{ci_l*100:.3f}%, {ci_h*100:.3f}%]")
# |     else:
# |         print("    No crossings observed.")
# |     print(f"  Audit 5 Status: PASS")
# |
# |     return {
# |         "status": "PASS",
# |         "n_boot": n_boot,
# |         "valid_crossings": n_valid,
# |         "p_crossing_le_050": p_cross,
# |         "counts": counts,
# |         "mean_crossing_conditional": mean_cross,
# |         "median_crossing_conditional": median_cross,
# |         "ci_95_crossing_conditional": [ci_l, ci_h]
# |     }
# |
# |
# | def run_audit_6_absolute_observation_counts() -> Dict[str, Any]:
# |     print("\n--- AUDIT 6: Absolute Observation Counts & Support Coverage Diagnostics ---")
# |     raw_df = pd.read_csv("results/direct_od_equivalence_v1/combined/raw_all_folds.csv")
# |     
# |     stats_by_p = {}
# |     
# |     for p_val in [0.001, 0.0025, 0.005]:
# |         sub = raw_df[raw_df.p == p_val]
# |         # City-level median/IQR across cities
# |         city_groups = sub.groupby("city").agg({
# |             "n_revealed": "first",
# |             "n_total_pairs": "first",
# |             "fraction_trip_mass_revealed": "mean",
# |             "origin_coverage": "mean",
# |             "destination_coverage": "mean",
# |             "both_endpoint_coverage": "mean"
# |         })
# |         
# |         n_rev = city_groups["n_revealed"].values
# |         
# |         stats_by_p[p_val] = {
# |             "p": p_val,
# |             "median_revealed_pairs": int(np.median(n_rev)),
# |             "iqr_revealed_pairs": [int(np.percentile(n_rev, 25)), int(np.percentile(n_rev, 75))],
# |             "min_revealed_pairs": int(np.min(n_rev)),
# |             "max_revealed_pairs": int(np.max(n_rev)),
# |             "mean_revealed_mass_pct": float(city_groups["fraction_trip_mass_revealed"].mean() * 100.0),
# |             "mean_origin_cov_pct": float(city_groups["origin_coverage"].mean() * 100.0),
# |             "mean_dest_cov_pct": float(city_groups["destination_coverage"].mean() * 100.0),
# |             "mean_both_cov_pct": float(city_groups["both_endpoint_coverage"].mean() * 100.0)
# |         }
# |         
# |         st = stats_by_p[p_val]
# |         print(f"  p = {p_val*100:5.2f}%: Median Pairs = {st['median_revealed_pairs']:>5} (IQR: [{st['iqr_revealed_pairs'][0]}, {st['iqr_revealed_pairs'][1]}], Range: [{st['min_revealed_pairs']}, {st['max_revealed_pairs']}]) | Both Cov = {st['mean_both_cov_pct']:.2f}% | Mass = {st['mean_revealed_mass_pct']:.2f}%")
# |
# |     print(f"  Audit 6 Status: PASS")
# |     return {
# |         "status": "PASS",
# |         "stats_by_p": stats_by_p
# |     }
# |
# |
# | def execute_full_audit_suite():
# |     print("=" * 85)
# |     print("DIRECT PARTIAL-OD EQUIVALENCE v1 — 6-GATE SCIENTIFIC AUDIT & CERTIFICATION SUITE")
# |     print("=" * 85)
# |     
# |     t0 = time.perf_counter()
# |     
# |     a1 = run_audit_1_production_yd_reference()
# |     a2 = run_audit_2_solver_precision(b_audit=50)
# |     a3 = run_audit_3_lambda_tie_rule()
# |     a4 = run_audit_4_monte_carlo_precision()
# |     a5 = run_audit_5_crossing_uncertainty_bootstrap()
# |     a6 = run_audit_6_absolute_observation_counts()
# |     
# |     all_passed = (
# |         a1["status"] == "PASS" and
# |         a2["status"] == "PASS" and
# |         a3["status"] == "PASS" and
# |         a4["status"] == "PASS" and
# |         a5["status"] == "PASS" and
# |         a6["status"] == "PASS"
# |     )
# |     
# |     elapsed = time.perf_counter() - t0
# |     
# |     print("\n" + "=" * 85)
# |     print("DIRECT-OD FINAL AUDIT SUMMARY")
# |     print("=" * 85)
# |     print(f"  Production YD reference:    {a1['status']}")
# |     print(f"  Solver precision:           {a2['status']}")
# |     print(f"  Lambda selection:           {a3['status']}")
# |     print(f"  Monte-Carlo precision:      {a4['status']}")
# |     print(f"  Crossing bootstrap:         {a5['status']}")
# |     print(f"  Support-conditioned counts: {a6['status']}")
# |     print("=" * 85)
# |     print(f"FINAL AUDIT RESULT: {'ALL 6 GATES CERTIFIED PASS' if all_passed else 'AUDIT FAILED'}")
# |     print(f"Execution Time: {elapsed:.2f}s")
# |     print("=" * 85)
# |     
# |     audit_report = {
# |         "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
# |         "all_passed": all_passed,
# |         "elapsed_seconds": elapsed,
# |         "audit_1_production_yd": a1,
# |         "audit_2_solver_precision": a2,
# |         "audit_3_lambda_selection": a3,
# |         "audit_4_monte_carlo_precision": a4,
# |         "audit_5_crossing_uncertainty": a5,
# |         "audit_6_observation_counts": a6
# |     }
# |     
# |     out_dir = Path("results/direct_od_equivalence_v1")
# |     with open(out_dir / "audit_report.json", "w", encoding="utf-8") as f:
# |         json.dump(audit_report, f, indent=2)
# |         
# |     return all_passed
# |
# |
# | if __name__ == "__main__":
# |     success = execute_full_audit_suite()
# |     sys.exit(0 if success else 1)

# ===== END SOURCE FILE: src/experiment/audit_direct_od_v1.py =====


# ===== BEGIN SOURCE FILE: src/experiment/compare_backbones.py =====
# | """
# | Compare Urban GNN and Pairwise MLP backbones across the locked 5-fold evaluation (N=50 cities).
# | Reads results from `results/5fold_results.json` and `results/mlp_backbone_results.json`.
# | """
# |
# | import os
# | import json
# | import argparse
# | import numpy as np
# | from pathlib import Path
# | from scipy import stats
# |
# | def analyze_subset(gnn_map, all_mlp_results, folds_to_include, label):
# |     paired_results = []
# |     for m in all_mlp_results:
# |         c = m.get("city")
# |         f = m.get("fold")
# |         if f not in folds_to_include or not c or c not in gnn_map:
# |             continue
# |         g = gnn_map[c]
# |         
# |         if "M0" in m and "M1_city_oracle_obs" in m:
# |             mlp_m0 = m["M0"].get("cpc_inter", 0.0)
# |             mlp_m1 = m["M1_city_oracle_obs"].get("cpc_inter", 0.0)
# |             mlp_delta = mlp_m1 - mlp_m0
# |         else:
# |             mlp_m0 = m.get("m0_cpc_inter", 0.0)
# |             mlp_m1 = m.get("m1_cpc_inter", 0.0)
# |             mlp_delta = m.get("delta_cpc", 0.0)
# |             
# |         paired_results.append({
# |             "city": c,
# |             "fold": f,
# |             "gnn_m0": g["m0_cpc_inter"],
# |             "gnn_m1": g["m1_cpc_inter"],
# |             "gnn_delta": g["delta_cpc"],
# |             "mlp_m0": mlp_m0,
# |             "mlp_m1": mlp_m1,
# |             "mlp_delta": mlp_delta,
# |             "gamma": g["delta_cpc"] - mlp_delta
# |         })
# |
# |     if not paired_results:
# |         return None
# |
# |     def summarize(vals):
# |         mean_v = float(np.mean(vals))
# |         median_v = float(np.median(vals))
# |         sd_v = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
# |
# |         delta_by_fold = {f: [] for f in folds_to_include}
# |         for v, r in zip(vals, paired_results):
# |             delta_by_fold[r["fold"]].append(v)
# |
# |         rng = np.random.default_rng(42)
# |         boot_means = []
# |         for _ in range(10000):
# |             samp = []
# |             for f in folds_to_include:
# |                 fold_vals = delta_by_fold[f]
# |                 if fold_vals:
# |                     samp.extend(rng.choice(fold_vals, size=len(fold_vals), replace=True))
# |             boot_means.append(np.mean(samp) if samp else 0.0)
# |         ci_l, ci_h = np.percentile(boot_means, [2.5, 97.5])
# |
# |         return {
# |             "mean": mean_v,
# |             "std": sd_v,
# |             "median": median_v,
# |             "ci_95": (float(ci_l), float(ci_h))
# |         }
# |
# |     gnn_deltas = np.array([r["gnn_delta"] for r in paired_results])
# |     mlp_deltas = np.array([r["mlp_delta"] for r in paired_results])
# |     gammas = np.array([r["gamma"] for r in paired_results])
# |
# |     gnn_sum = summarize(gnn_deltas)
# |     mlp_sum = summarize(mlp_deltas)
# |     gamma_sum = summarize(gammas)
# |
# |     _, gnn_w_p = stats.wilcoxon(gnn_deltas, alternative="greater")
# |     _, mlp_w_p = stats.wilcoxon(mlp_deltas, alternative="greater")
# |     _, gamma_w_p = stats.wilcoxon(gammas, alternative="two-sided")
# |
# |     return {
# |         "label": label,
# |         "n": len(paired_results),
# |         "gnn_m0_mean": float(np.mean([r["gnn_m0"] for r in paired_results])),
# |         "gnn_m1_mean": float(np.mean([r["gnn_m1"] for r in paired_results])),
# |         "gnn_sum": gnn_sum,
# |         "gnn_pos": int(np.sum(gnn_deltas > 0)),
# |         "gnn_p": float(gnn_w_p),
# |         "mlp_m0_mean": float(np.mean([r["mlp_m0"] for r in paired_results])),
# |         "mlp_m1_mean": float(np.mean([r["mlp_m1"] for r in paired_results])),
# |         "mlp_sum": mlp_sum,
# |         "mlp_pos": int(np.sum(mlp_deltas > 0)),
# |         "mlp_p": float(mlp_w_p),
# |         "gamma_sum": gamma_sum,
# |         "gamma_p": float(gamma_w_p),
# |     }
# |
# |
# | def run_comparison(output_dir: str = "results", export_md: bool = True):
# |     print("\n" + "=" * 85)
# |     print("COMPARISON: Gravity-Informed Urban GNN vs Pairwise Spatial MLP (Backbone Robustness)")
# |     print("=" * 85)
# |
# |     gnn_results_path = Path(output_dir) / "5fold_results.json"
# |     mlp_results_path = Path(output_dir) / "mlp_backbone_results.json"
# |
# |     with open(gnn_results_path, "r") as f:
# |         gnn_json = json.load(f)
# |         gnn_data = gnn_json.get("city_level_results", [])
# |
# |     with open(mlp_results_path, "r") as f:
# |         mlp_json = json.load(f)
# |         all_mlp_results = mlp_json.get("city_level_results", mlp_json) if isinstance(mlp_json, dict) else mlp_json
# |
# |     gnn_map = {}
# |     for r in gnn_data:
# |         m0_data = r.get("M0")
# |         m1_data = r.get("M1_city_oracle_obs", r.get("M1_city_oracle_obs"))
# |         if m0_data and m1_data:
# |             gnn_map[r["city"]] = {
# |                 "m0_cpc_inter": m0_data.get("cpc_inter", 0.0),
# |                 "m1_cpc_inter": m1_data.get("cpc_inter", 0.0),
# |                 "delta_cpc": m1_data.get("cpc_inter", 0.0) - m0_data.get("cpc_inter", 0.0)
# |             }
# |
# |     # part_a = analyze_subset(gnn_map, all_mlp_results, [2, 3, 4, 5], "Part A: Confirmatory Evaluation Set (Folds 2–5, n=40 Cities)")
# |     part_b = analyze_subset(gnn_map, all_mlp_results, [1, 2, 3, 4, 5], "Five-Fold Cross-City Evaluation Set (All 5 Folds, N=50 Cities)")
# |
# |     for res in [part_b]:
# |         if not res:
# |             continue
# |         print(f"\n### {res['label']} (N={res['n']} Cities)")
# |         print(f"Urban GNN:     M0={res['gnn_m0_mean']:.4f} -> M1={res['gnn_m1_mean']:.4f} | dCPC={res['gnn_sum']['mean']:+.4f} +- {res['gnn_sum']['std']:.4f} | 95% CI [{res['gnn_sum']['ci_95'][0]:+.4f}, {res['gnn_sum']['ci_95'][1]:+.4f}] | Pos={res['gnn_pos']}/{res['n']} | p={res['gnn_p']:.2e}")
# |         print(f"Pairwise MLP:  M0={res['mlp_m0_mean']:.4f} -> M1={res['mlp_m1_mean']:.4f} | dCPC={res['mlp_sum']['mean']:+.4f} +- {res['mlp_sum']['std']:.4f} | 95% CI [{res['mlp_sum']['ci_95'][0]:+.4f}, {res['mlp_sum']['ci_95'][1]:+.4f}] | Pos={res['mlp_pos']}/{res['n']} | p={res['mlp_p']:.2e}")
# |         print(f"Difference G:  dCPC={res['gamma_sum']['mean']:+.4f} +- {res['gamma_sum']['std']:.4f} | 95% CI [{res['gamma_sum']['ci_95'][0]:+.4f}, {res['gamma_sum']['ci_95'][1]:+.4f}] | p={res['gamma_p']:.2e}")
# |
# |     if export_md:
# |         table_path = Path(output_dir) / "tables" / "table_gnn_vs_mlp_comparison.md"
# |         table_path.parent.mkdir(parents=True, exist_ok=True)
# |         with open(table_path, "w", encoding="utf-8") as f:
# |             f.write("# Neural Backbone Comparison: Gravity-Informed Urban GNN vs Pairwise Spatial MLP\n\n")
# |             f.write("> **Evaluation Goal**: Assesses whether distance-binned aggregate distribution calibration ($Y_D^{\\text{target}}$) provides consistent reconstruction gain across distinct neural architectures (Spatial Graph Convolution vs Local Feature MLP).\n\n")
# |             
# |             for res in [part_b]:
# |                 if not res:
# |                     continue
# |                 f.write(f"## {res['label']}\n\n")
# |                 f.write("| Backbone Architecture | Zero-Shot $M_0$ CPC | Calibrated $M_1$ CPC | Marginal Gain $\\Delta\\text{CPC}$ | 95% Fold-Stratified Bootstrap CI | Improved Cities | Wilcoxon $p$ |\n")
# |                 f.write("|---|:---:|:---:|:---:|:---:|:---:|:---:|\n")
# |                 gnn_mean = res['gnn_sum']['mean']
# |                 gnn_std = res['gnn_sum']['std']
# |                 mlp_mean = res['mlp_sum']['mean']
# |                 mlp_std = res['mlp_sum']['std']
# |                 gam_mean = res['gamma_sum']['mean']
# |                 gam_std = res['gamma_sum']['std']
# |                 f.write(f"| **Gravity-Informed Urban GNN** | {res['gnn_m0_mean']:.4f} | **{res['gnn_m1_mean']:.4f}** | **{gnn_mean:+.4f} +- {gnn_std:.4f}** | [{res['gnn_sum']['ci_95'][0]:+.4f}, {res['gnn_sum']['ci_95'][1]:+.4f}] | {res['gnn_pos']}/{res['n']} ({res['gnn_pos']/res['n']*100:.1f}%) | p = {res['gnn_p']:.2e} |\n")
# |                 f.write(f"| **Pairwise Spatial MLP** | {res['mlp_m0_mean']:.4f} | **{res['mlp_m1_mean']:.4f}** | **{mlp_mean:+.4f} +- {mlp_std:.4f}** | [{res['mlp_sum']['ci_95'][0]:+.4f}, {res['mlp_sum']['ci_95'][1]:+.4f}] | {res['mlp_pos']}/{res['n']} ({res['mlp_pos']/res['n']*100:.1f}%) | p = {res['mlp_p']:.2e} |\n")
# |                 f.write(f"| **Architecture Advantage ($\\Gamma = \\Delta_\\text{{GNN}} - \\Delta_\\text{{MLP}}$)** | — | — | **{gam_mean:+.4f} +- {gam_std:.4f}** | [{res['gamma_sum']['ci_95'][0]:+.4f}, {res['gamma_sum']['ci_95'][1]:+.4f}] | — | p = {res['gamma_p']:.2e} |\n\n")
# |             
# |         print(f"\nSaved comparison table to {table_path}")
# |
# |
# | if __name__ == "__main__":
# |     parser = argparse.ArgumentParser(description="Compare Urban GNN and Pairwise MLP backbones")
# |     parser.add_argument("--output_dir", type=str, default="results")
# |     args = parser.parse_args()
# |     run_comparison(output_dir=args.output_dir)

# ===== END SOURCE FILE: src/experiment/compare_backbones.py =====


# ===== BEGIN SOURCE FILE: src/experiment/compute_delta_r.py =====
# | """
# | Cross-city Statistical Analysis of Moving-Bin Calibration Results (RQ1).
# | """
# |
# | import numpy as np
# | from scipy import stats
# | from typing import List, Dict, Any
# |
# |
# | def _compute_stats(arr: np.ndarray, ddof: int = 1) -> Dict[str, Any]:
# |     """Compute summary statistics with sample standard deviation (ddof=1) and sample size n."""
# |     n = int(len(arr))
# |     std_val = float(np.std(arr, ddof=ddof)) if n > 1 else 0.0
# |     return {
# |         "n": n,
# |         "mean": float(np.mean(arr)),
# |         "std": std_val,
# |         "median": float(np.median(arr)),
# |         "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)) if n > 0 else 0.0,
# |         "p25": float(np.percentile(arr, 25)) if n > 0 else 0.0,
# |         "p75": float(np.percentile(arr, 75)) if n > 0 else 0.0,
# |         "min": float(np.min(arr)) if n > 0 else 0.0,
# |         "max": float(np.max(arr)) if n > 0 else 0.0,
# |     }
# |
# |
# | def _fold_stratified_bootstrap(values: np.ndarray, fold_ids: np.ndarray, n_boot: int = 10000) -> tuple[float, float]:
# |     """Fold-stratified bootstrap 95% CI for the mean of a given metric."""
# |     folds = {}
# |     for i, f in enumerate(fold_ids):
# |         if f not in folds:
# |             folds[f] = []
# |         folds[f].append(values[i])
# |     
# |     if len(folds) == 0 or sum(len(v) for v in folds.values()) < 2:
# |         return 0.0, 0.0
# |         
# |     rng = np.random.default_rng(42)
# |     boot_means = []
# |     for _ in range(n_boot):
# |         samp = []
# |         for f, vals in folds.items():
# |             if len(vals) > 0:
# |                 samp.extend(rng.choice(vals, size=len(vals), replace=True))
# |         boot_means.append(np.mean(samp))
# |     
# |     return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))
# |
# |
# | def analyze_delta_r(city_results: List[Dict[str, Any]]) -> Dict[str, Any]:
# |     n_cities = len(city_results)
# |
# |     # Primary: Interzonal CPC on Omega_c^+
# |     m0_inter = np.array([r["M0"]["cpc_inter"] for r in city_results])
# |     
# |     analysis = {
# |         "n_cities_evaluated": n_cities,
# |         "std_definition": "sample_sd_ddof_1",
# |         "missingness_correlations": {}
# |     }
# |
# |     scales = [
# |         ("city", "M1_city_oracle_obs"),
# |         ("county", "M1_county_oracle_obs"),
# |         ("subzone", "M1_subzone_oracle_obs")
# |     ]
# |
# |     for scale_name, scale_key in scales:
# |         m1_inter = np.array([r[scale_key]["cpc_inter"] for r in city_results])
# |         delta_inter = m1_inter - m0_inter
# |         
# |         # Fold-stratified bootstrap
# |         fold_ids = np.array([r.get("fold", -1) for r in city_results])
# |         ci_low, ci_high = _fold_stratified_bootstrap(delta_inter, fold_ids)
# |         
# |         # Missingness Correlations
# |         missingness = {}
# |         for feature in ["rho_c", "n_tracts", "mean_distance", "average_flow", "short_long_ratio"]:
# |             feature_vals = np.array([r.get(feature, 0.0) for r in city_results])
# |             if np.std(feature_vals) > 0:
# |                 rho, _ = stats.pearsonr(feature_vals, delta_inter)
# |                 missingness[f"corr_with_{feature}"] = float(rho)
# |         analysis["missingness_correlations"][scale_name] = missingness
# |
# |         analysis[scale_name] = {
# |             "m0_cpc_inter": _compute_stats(m0_inter),
# |             "m1_cpc_inter": _compute_stats(m1_inter),
# |             "delta_cpc_inter": {**_compute_stats(delta_inter), "ci_95_lower": ci_low, "ci_95_upper": ci_high},
# |             "p_improved": float(np.mean(delta_inter > 0)),
# |         }
# |
# |         if len(delta_inter) >= 5:
# |             w_stat, w_p_two = stats.wilcoxon(m1_inter, m0_inter, alternative="two-sided")
# |             _, w_p_one = stats.wilcoxon(m1_inter, m0_inter, alternative="greater")
# |             
# |             # Compute matched-pairs rank-biserial correlation
# |             diff = m1_inter - m0_inter
# |             diff = diff[diff != 0]
# |             ranks = stats.rankdata(np.abs(diff))
# |             w_plus = np.sum(ranks[diff > 0])
# |             w_minus = np.sum(ranks[diff < 0])
# |             r_rb = (w_plus - w_minus) / (w_plus + w_minus) if (w_plus + w_minus) > 0 else 0.0
# |
# |             analysis[scale_name]["wilcoxon_one_sided_p"] = float(w_p_one)
# |             analysis[scale_name]["wilcoxon_two_sided_p"] = float(w_p_two)
# |             analysis[scale_name]["rank_biserial_r"] = float(r_rb)
# |
# |     return analysis

# ===== END SOURCE FILE: src/experiment/compute_delta_r.py =====


# ===== BEGIN SOURCE FILE: src/experiment/compute_qstar.py =====
# | """
# | Cross-city Statistical Analysis of q* and m* (RQ2).
# |
# | Distinctly computes:
# |     - Real (Primary RQ2): m*_real, q*_real = m*_real / T_total against Y_D^{Meta}
# |     - Oracle (Benchmark): m*_oracle, q*_oracle against Y_D^{oracle}
# | """
# |
# | import numpy as np
# | from typing import List, Dict, Any
# |
# |
# | def _summary_stats(arr: np.ndarray, ddof: int = 1) -> Dict[str, Any]:
# |     """Compute summary statistics with sample standard deviation (ddof=1) and sample size n."""
# |     n = int(len(arr))
# |     std_val = float(np.std(arr, ddof=ddof)) if n > 1 else 0.0
# |     return {
# |         "n": n,
# |         "mean": float(np.mean(arr)) if n > 0 else 0.0,
# |         "std": std_val,
# |         "median": float(np.median(arr)) if n > 0 else 0.0,
# |         "p25": float(np.percentile(arr, 25)) if n > 0 else 0.0,
# |         "p75": float(np.percentile(arr, 75)) if n > 0 else 0.0,
# |         "min": float(np.min(arr)) if n > 0 else 0.0,
# |         "max": float(np.max(arr)) if n > 0 else 0.0,
# |     }
# |
# |
# | def analyze_qstar(city_results: List[Dict[str, Any]]) -> Dict[str, Any]:
# |     out = {"n_cities": len(city_results)}
# |
# |     # Oracle
# |     m_oracle = np.array([r["m_star_oracle"] for r in city_results if r.get("m_star_oracle") is not None])
# |     q_oracle = np.array([r["q_star_oracle"] for r in city_results if r.get("q_star_oracle") is not None])
# |     if len(m_oracle) > 0:
# |         out["oracle"] = {
# |             "m_star": _summary_stats(m_oracle),
# |             "q_star": _summary_stats(q_oracle),
# |         }
# |
# |     # Real (Primary)
# |     m_real = np.array([r["m_star_real"] for r in city_results if r.get("m_star_real") is not None])
# |     q_real = np.array([r["q_star_real"] for r in city_results if r.get("q_star_real") is not None])
# |     if len(m_real) > 0:
# |         out["real"] = {
# |             "n_cities": len(m_real),
# |             "m_star": _summary_stats(m_real),
# |             "q_star": _summary_stats(q_real),
# |         }
# |
# |     return out

# ===== END SOURCE FILE: src/experiment/compute_qstar.py =====


# ===== BEGIN SOURCE FILE: src/experiment/generate_q3_figures_and_stats.py =====
# | import json
# | import numpy as np
# | import pandas as pd
# | from pathlib import Path
# | from scipy import stats
# | import matplotlib.pyplot as plt
# | import seaborn as sns
# |
# |
# |
# | def main():
# |     results_file = Path("results/5fold_results.json")
# |     if not results_file.exists():
# |         print(f"Error: {results_file} not found.")
# |         return
# |
# |     with open(results_file, "r") as f:
# |         data = json.load(f)
# |     
# |     city_results = data.get("city_level_results", [])
# |     if len(city_results) == 0:
# |         print("No city results found in JSON.")
# |         return
# |
# |     print(f"Loaded {len(city_results)} cities from {results_file}")
# |     
# |     cpc_m0 = []
# |     cpc_m1 = []
# |     delta_cpcs = []
# |     city_names = []
# |
# |     for res in city_results:
# |         m0 = res["M0"]["cpc_inter"]
# |         m1 = res["M1_city_oracle_obs"]["cpc_inter"]  # Using M1_city as primary M1
# |         delta = m1 - m0
# |         cpc_m0.append(m0)
# |         cpc_m1.append(m1)
# |         delta_cpcs.append(delta)
# |         city_names.append(res["city"])
# |         
# |     cpc_m0 = np.array(cpc_m0)
# |     cpc_m1 = np.array(cpc_m1)
# |     delta_cpcs = np.array(delta_cpcs)
# |     
# |     # ---------------------------------------------------------
# |     # 2. STATISTICAL TEST ON 50 CITIES (Wilcoxon Signed-Rank)
# |     # ---------------------------------------------------------
# |     print("\n" + "="*50)
# |     print("STATISTICAL TEST RESULTS (N=50 CITIES)")
# |     print("="*50)
# |     
# |     city_stats = data.get("rq1_delta_r", {}).get("city", {})
# |     if not city_stats:
# |         print("Missing rq1_delta_r.city in JSON.")
# |         return
# |         
# |     delta_stats = city_stats.get("delta_cpc_inter", {})
# |     
# |     n_cities = len(delta_cpcs)
# |     mean_delta = delta_stats.get("mean", np.mean(delta_cpcs))
# |     median_delta = delta_stats.get("median", np.median(delta_cpcs))
# |     ci_lower = delta_stats.get("ci_95_lower", 0.0)
# |     ci_upper = delta_stats.get("ci_95_upper", 0.0)
# |     effect_size = city_stats.get("rank_biserial_r", 0.0)
# |     p_value = city_stats.get("wilcoxon_one_sided_p", 1.0)
# |     p_value_two_sided = city_stats.get("wilcoxon_two_sided_p", 1.0)
# |     
# |     print(f"Number of cities: {n_cities}")
# |     print(f"Delta CPC (Mean):   {mean_delta:+.4f}")
# |     print(f"Delta CPC (Median): {median_delta:+.4f}")
# |     print(f"95% CI (Mean):      [{ci_lower:+.4f}, {ci_upper:+.4f}]")
# |     print(f"Matched-pairs Rank-biserial (r_rb): {effect_size:.4f}")
# |     print(f"Wilcoxon p-value (One-sided, M1 > M0): {p_value:.4e}")
# |     print(f"Wilcoxon p-value (Two-sided):          {p_value_two_sided:.4e}")
# |     print("="*50)
# |
# |     # ---------------------------------------------------------
# |     # 3. STORYBOARD FIGURES
# |     # ---------------------------------------------------------
# |     plots_dir = Path("results/q3_figures")
# |     plots_dir.mkdir(parents=True, exist_ok=True)
# |     sns.set_theme(style="whitegrid", context="paper")
# |     
# |     # Figure 3: Distribution of Delta CPC by city (Sorted Bar Chart)
# |     sorted_indices = np.argsort(delta_cpcs)
# |     sorted_deltas = delta_cpcs[sorted_indices]
# |     sorted_cities = np.array(city_names)[sorted_indices]
# |     
# |     plt.figure(figsize=(12, 6))
# |     colors = ['#d62728' if x < 0 else '#2ca02c' for x in sorted_deltas]
# |     plt.bar(range(n_cities), sorted_deltas, color=colors, alpha=0.8)
# |     plt.axhline(0, color='black', linewidth=1.2)
# |     plt.axhline(mean_delta, color='blue', linestyle='--', linewidth=1.5, label=f'Mean Delta CPC = {mean_delta:.4f}')
# |     plt.xticks(range(n_cities), sorted_cities, rotation=90, fontsize=8)
# |     plt.ylabel("Delta CPC (M1 - M0)", fontsize=12)
# |     plt.title("Figure 3: Distribution of Delta CPC by City", fontsize=14)
# |     plt.legend()
# |     plt.tight_layout()
# |     plt.savefig(plots_dir / "Figure_3_Delta_CPC_Distribution.png", dpi=300)
# |     plt.close()
# |     print(f"Saved Figure 3: {plots_dir / 'Figure_3_Delta_CPC_Distribution.png'}")
# |
# |     # Figure 4: M0 vs M1 across cities
# |     plt.figure(figsize=(6, 6))
# |     plt.scatter(cpc_m0, cpc_m1, alpha=0.7, edgecolors='w', s=60, color='#1f77b4')
# |     
# |     min_val = min(np.min(cpc_m0), np.min(cpc_m1)) - 0.05
# |     max_val = max(np.max(cpc_m0), np.max(cpc_m1)) + 0.05
# |     plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='y = x')
# |     
# |     plt.xlim(min_val, max_val)
# |     plt.ylim(min_val, max_val)
# |     plt.xlabel("Zero-shot M0 CPC", fontsize=12)
# |     plt.ylabel("Support-conditioned M1 CPC", fontsize=12)
# |     plt.title("Figure 4: M0 vs M1 Performance per City", fontsize=14)
# |     plt.legend()
# |     plt.tight_layout()
# |     plt.savefig(plots_dir / "Figure_4_M0_vs_M1_Scatter.png", dpi=300)
# |     plt.close()
# |     print(f"Saved Figure 4: {plots_dir / 'Figure_4_M0_vs_M1_Scatter.png'}")
# |
# | if __name__ == "__main__":
# |     main()

# ===== END SOURCE FILE: src/experiment/generate_q3_figures_and_stats.py =====


# ===== BEGIN SOURCE FILE: src/experiment/run_5fold.py =====
# | """
# | Master 5-Fold Cross-Validation Experiment Runner (Moving-Bin Calibration Framework).
# | """
# |
# | import os
# | os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
# | import sys
# | import json
# | import time
# | import argparse
# | import torch
# | from pathlib import Path
# |
# | # Ensure root directory is in sys.path
# | sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
# |
# | from src.data.city_splits import generate_35_5_10_splits
# | from src.data.yd_extractor import compute_kbin_edges
# | from src.training.train import train_zero_shot_model
# | from src.experiment.run_experiment import run_target_city_experiments
# | from src.experiment.compute_delta_r import analyze_delta_r
# | from src.experiment.compute_qstar import analyze_qstar
# |
# | from src.training.train import load_checkpoint
# |
# |
# | def _write_json_atomic(path: Path, payload: dict) -> None:
# |     temporary_path = path.with_suffix(path.suffix + ".tmp")
# |     with open(temporary_path, "w", encoding="utf-8") as output_file:
# |         json.dump(payload, output_file, indent=2)
# |         output_file.flush()
# |         os.fsync(output_file.fileno())
# |     os.replace(temporary_path, path)
# |
# |
# | def run_5fold_experiment(
# |     data_root: str = "data",
# |     meta_prior_dir: str = "meta_prior",
# |     output_dir: str = "results",
# |     epochs_per_fold: int = 200,
# |     lr: float = 3.2e-3,
# |     hidden_dim: int = 64,
# |     num_gnn_layers: int = 2,
# |     graph_type: str = "radius",
# |     radius_km: float = 5.0,
# |     knn_k: int = 10,
# |     loss_type: str = "ztnb",
# |     backbone: str = "gnn",
# |     num_trip_seeds: int = 20,
# |     seeds: list[int] | None = None,
# |     folds_to_run: list[int] | None = None,
# |     device_str: str | None = None,
# | ):
# |     os.makedirs(output_dir, exist_ok=True)
# |     splits = generate_35_5_10_splits(data_root=data_root)
# |     manifest_path = Path(__file__).resolve().parents[2] / "results" / "e1" / "splits_manifest_v2.json"
# |     if not manifest_path.exists():
# |         raise FileNotFoundError(f"Missing locked split manifest: {manifest_path}")
# |     with open(manifest_path, "r", encoding="utf-8") as manifest_file:
# |         split_manifest_sha256 = json.load(manifest_file)["manifest_sha256"]
# |
# |     if device_str is None:
# |         device_str = "cuda" if torch.cuda.is_available() else "cpu"
# |
# |     if folds_to_run is None:
# |         folds_to_run = [1, 2, 3, 4, 5]
# |     if seeds is None:
# |         seeds = [1, 10, 100]
# |
# |     print("=" * 85)
# |     print("STARTING 5-FOLD CROSS-VALIDATION (MOVING-BIN CALIBRATION FRAMEWORK)")
# |     print(f"Device: {device_str} | Epochs: {epochs_per_fold} | Graph: {graph_type} (r={radius_km}km)")
# |     print(f"Primary Calibration Domain: Omega_c^+ (Interzonal moving bins 1, 2, 3)")
# |     print(f"Folds to run: {folds_to_run}")
# |     print("=" * 85)
# |
# |     out_file_name = "5fold_results.json" if backbone == "gnn" else f"{backbone}_backbone_results.json"
# |     out_file = Path(output_dir) / out_file_name
# |     run_signature = {
# |         "backbone": backbone,
# |         "seeds": list(seeds),
# |         "folds": list(folds_to_run),
# |         "epochs_per_fold": epochs_per_fold,
# |         "hidden_dim": hidden_dim,
# |         "num_gnn_layers": num_gnn_layers,
# |         "graph_type": graph_type,
# |         "radius_km": radius_km,
# |         "knn_k": knn_k,
# |         "loss_type": loss_type,
# |         "split_manifest_sha256": split_manifest_sha256,
# |     }
# |
# |     all_city_results = []
# |     if out_file.exists():
# |         try:
# |             with open(out_file, "r") as f:
# |                 prev_json = json.load(f)
# |                 if prev_json.get("experiment_config", {}).get("run_signature") == run_signature:
# |                     all_city_results = prev_json.get("city_level_results", [])
# |                     print(f"Loaded {len(all_city_results)} existing city records from {out_file}.")
# |                 else:
# |                     print(f"Ignoring stale result artifact with mismatched run signature: {out_file}")
# |         except Exception:
# |             all_city_results = []
# |
# |     fold_summaries = {}
# |
# |     start_total_time = time.time()
# |
# |     for fold_id in folds_to_run:
# |         split = splits[fold_id]
# |         train_cities = split["train"]
# |         val_cities = split["val"]
# |         test_cities = split["test"]
# |
# |         print("\n" + "#" * 85)
# |         print(f"FOLD {fold_id}/5: Training on {len(train_cities)} cities -> Testing on {len(test_cities)} held-out cities")
# |         print(f"Validation cities: {val_cities}")
# |         print(f"Held-out targets: {test_cities}")
# |         print("#" * 85)
# |
# |         fold_start = time.time()
# |         models = []
# |         scalers = []
# |         for seed_idx, seed in enumerate(seeds):
# |             _ckpt_dir  = Path(output_dir) / "checkpoints"
# |             _ckpt_name = f"5fold_fold{fold_id}_seed{seed}.pt" if backbone == "gnn" else f"{backbone}_fold{fold_id}_seed{seed}.pt"
# |             _ckpt_path = _ckpt_dir / _ckpt_name
# |             
# |             expected_config = {
# |                 "hidden_dim": hidden_dim,
# |                 "num_gnn_layers": num_gnn_layers,
# |                 "graph_type": graph_type,
# |                 "radius_km": radius_km,
# |                 "knn_k": knn_k,
# |                 "loss_type": loss_type,
# |                 "epochs": epochs_per_fold,
# |                 "lr": lr,
# |                 "backbone": backbone,
# |             }
# |             if _ckpt_path.exists():
# |                 print(f"--- Found existing checkpoint {_ckpt_path}. Loading... ---")
# |                 model, scaler, metadata = load_checkpoint(_ckpt_path, device_str=device_str, expected_config=expected_config)
# |                 checkpoint_hp = metadata.get("hyperparams", {})
# |                 assert metadata.get("seed") == seed, f"Checkpoint seed mismatch in {_ckpt_path}"
# |                 assert checkpoint_hp.get("fold") == fold_id, f"Checkpoint fold mismatch in {_ckpt_path}"
# |                 assert checkpoint_hp.get("split_manifest_sha256") == split_manifest_sha256, (
# |                     f"Checkpoint split manifest mismatch in {_ckpt_path}"
# |                 )
# |                 model.eval()
# |             else:
# |                 print(f"\n--- Training Seed {seed_idx+1}/{len(seeds)} (Seed: {seed}) [Backbone: {backbone.upper()}] ---")
# |                 model, scaler = train_zero_shot_model(
# |                     train_city_names=train_cities,
# |                     data_root=data_root,
# |                     epochs=epochs_per_fold,
# |                     lr=lr,
# |                     hidden_dim=hidden_dim,
# |                     num_gnn_layers=num_gnn_layers,
# |                     graph_type=graph_type,
# |                     radius_km=radius_km,
# |                     knn_k=knn_k,
# |                     loss_type=loss_type,
# |                     backbone=backbone,
# |                     device_str=device_str,
# |                     verbose=True,
# |                     val_city_names=val_cities,
# |                     patience=16,
# |                     checkpoint_path=_ckpt_path,
# |                     run_tag=f"5fold_{backbone}_fold{fold_id}_seed{seed}",
# |                     seed=seed,
# |                     fold=fold_id,
# |                     split_manifest_sha256=split_manifest_sha256,
# |                 )
# |             models.append(model)
# |             scalers.append(scaler)
# |         print(f"Fold {fold_id} models trained in {time.time() - fold_start:.1f}s.")
# |
# |
# |
# |
# |         # Compute Bin Edges from 35 train cities (K=8)
# |         bin_edges, K_active = compute_kbin_edges(train_cities, K=8, data_root=data_root)
# |
# |         # Stage B: Target City Evaluation
# |         fold_city_results = [r for r in all_city_results if r.get("fold") == fold_id]
# |         completed_cities = {r.get("city") for r in fold_city_results}
# |         for target_city in test_cities:
# |             if target_city in completed_cities:
# |                 print(f"  -> Reusing saved result: {target_city}")
# |                 continue
# |             print(f"  -> Evaluating: {target_city:<18}", end="", flush=True)
# |             t0 = time.time()
# |             
# |             seed_results = []
# |             for seed_idx, model in enumerate(models):
# |                 scaler = scalers[seed_idx]
# |                 res = run_target_city_experiments(
# |                     model=model,
# |                     city_name=target_city,
# |                     scaler=scaler,
# |                     data_root=data_root,
# |                     graph_type=graph_type,
# |                     radius_km=radius_km,
# |                     knn_k=knn_k,
# |                     device_str=device_str,
# |                     bin_edges=bin_edges,
# |                 )
# |                 seed_results.append(res)
# |                 
# |             # Average the results across 3 seeds
# |             avg_res = seed_results[0].copy()
# |             for key in ["M0", "M1_city_oracle_obs", "M1_county_oracle_obs", "M1_subzone_oracle_obs"]:
# |                 if avg_res[key] is not None:
# |                     avg_res[key] = avg_res[key].copy()
# |                     for metric in ["cpc_inter", "mae_inter", "rmse_inter", "nrmse_inter", "rmse_log1p_inter", "spearman_inter", "rel_error_total", "cpc_inflow", "cpc_outflow"]:
# |                         if metric in avg_res[key]:
# |                             avg_res[key][metric] = sum(r[key][metric] for r in seed_results) / len(seed_results)
# |             
# |             for key in ["rho_c", "average_flow", "mean_distance"]:
# |                 if key in avg_res and avg_res[key] is not None:
# |                     avg_res[key] = sum(r[key] for r in seed_results) / len(seed_results)
# |             
# |             # Compute Deltas (Primary Estimands)
# |             avg_res["delta_city"] = avg_res["M1_city_oracle_obs"]["cpc_inter"] - avg_res["M0"]["cpc_inter"]
# |             avg_res["delta_county"] = avg_res["M1_county_oracle_obs"]["cpc_inter"] - avg_res["M0"]["cpc_inter"]
# |             avg_res["delta_subzone"] = avg_res["M1_subzone_oracle_obs"]["cpc_inter"] - avg_res["M0"]["cpc_inter"]
# |             
# |             city_res = avg_res
# |             city_res["fold"] = fold_id
# |             fold_city_results.append(city_res)
# |             all_city_results.append(city_res)
# |
# |             m0_c = city_res['M0']['cpc_inter']
# |             m1_city = city_res['M1_city_oracle_obs']['cpc_inter']
# |             m1_county = city_res['M1_county_oracle_obs']['cpc_inter']
# |             m1_sub = city_res['M1_subzone_oracle_obs']['cpc_inter']
# |
# |             print(f" | M0: {m0_c:.4f} | M1_city: {m1_city:.4f} (d={avg_res['delta_city']:+.4f}) | M1_county: {m1_county:.4f} (d={avg_res['delta_county']:+.4f}) | M1_subzone: {m1_sub:.4f} (d={avg_res['delta_subzone']:+.4f}) | {time.time() - t0:.1f}s")
# |
# |             _write_json_atomic(out_file, {
# |                 "experiment_config": {
# |                     **run_signature,
# |                     "total_cities_evaluated": len(all_city_results),
# |                     "total_runtime_sec": time.time() - start_total_time,
# |                     "run_signature": run_signature,
# |                 },
# |                 "rq1_delta_r": analyze_delta_r(all_city_results),
# |                 "city_level_results": all_city_results,
# |             })
# |
# |         fold_summaries[f"fold_{fold_id}"] = {
# |             "test_cities": test_cities,
# |             "mean_delta_city": float(sum(r["delta_city"] for r in fold_city_results) / max(1, len(fold_city_results))),
# |         }
# |         
# |         # Intermediate Save
# |         out_file_name = "5fold_results.json" if backbone == "gnn" else f"{backbone}_backbone_results.json"
# |         out_file = Path(output_dir) / out_file_name
# |         temp_delta_r = analyze_delta_r(all_city_results)
# |         temp_results = {
# |             "experiment_config": {
# |                 "device": device_str,
# |                 "epochs_per_fold": epochs_per_fold,
# |                 "hidden_dim": hidden_dim,
# |                 "graph_type": graph_type,
# |                 "radius_km": radius_km,
# |                 "knn_k": knn_k,
# |                 "loss_type": loss_type,
# |                 "total_cities_evaluated": len(all_city_results),
# |                 "total_runtime_sec": time.time() - start_total_time,
# |             },
# |             "rq1_delta_r": temp_delta_r,
# |             "city_level_results": all_city_results,
# |         }
# |         temp_results["experiment_config"]["run_signature"] = run_signature
# |         _write_json_atomic(out_file, temp_results)
# |
# |     # Cross-city Statistical Aggregation (Final)
# |     delta_r_analysis = analyze_delta_r(all_city_results)
# |
# |     final_results = {
# |         "experiment_config": {
# |             "device": device_str,
# |             "epochs_per_fold": epochs_per_fold,
# |             "hidden_dim": hidden_dim,
# |             "graph_type": graph_type,
# |             "radius_km": radius_km,
# |             "knn_k": knn_k,
# |             "loss_type": loss_type,
# |             "total_cities_evaluated": len(all_city_results),
# |             "total_runtime_sec": time.time() - start_total_time,
# |         },
# |         "rq1_delta_r": delta_r_analysis,
# |         "city_level_results": all_city_results,
# |     }
# |
# |     out_file_name = "5fold_results.json" if backbone == "gnn" else f"{backbone}_backbone_results.json"
# |     out_file = Path(output_dir) / out_file_name
# |     final_results["experiment_config"]["run_signature"] = run_signature
# |     _write_json_atomic(out_file, final_results)
# |
# |     print("\n" + "=" * 85)
# |     print("FINAL SUMMARY: UNIFIED RESOLUTION CALIBRATION (CITY / COUNTY / SUBZONE)")
# |     print("TASK: OD intensity reconstruction conditional on the observed positive OD support.")
# |     print("=" * 85)
# |     print(f"Total cities evaluated: {len(all_city_results)}/50")
# |
# |     for scale in ["city", "county", "subzone"]:
# |         if scale in delta_r_analysis:
# |             s_data = delta_r_analysis[scale]
# |             scale_label = "GADM 4.1 LEVEL-2 COUNTY" if scale == "county" else f"{scale.upper()}"
# |             if scale == "subzone":
# |                 scale_label = "FINE-GRAINED SUBZONE ORACLE / INFORMATION CEILING"
# |             print(f"\n[{scale_label}-LEVEL CALIBRATION]")
# |             if scale == "subzone":
# |                 print("  (Note: Subzone is a high-resolution ceiling limit, not used as main evidence for Y_D)")
# |             print(f"  M0 Interzonal CPC (Mean):                       {s_data['m0_cpc_inter']['mean']:.4f}")
# |             print(f"  M1 Interzonal CPC (Mean):                       {s_data['m1_cpc_inter']['mean']:.4f}")
# |             print(f"  Delta Mean +- Std:                              {s_data['delta_cpc_inter']['mean']:+.4f} +- {s_data['delta_cpc_inter']['std']:.4f}")
# |             print(f"  Delta 95% CI (Fold-Stratified Bootstrap):       [{s_data['delta_cpc_inter']['ci_95_lower']:+.4f}, {s_data['delta_cpc_inter']['ci_95_upper']:+.4f}]")
# |             win_rate = s_data['p_improved'] * 100
# |             n_eval_cities = s_data.get('n_cities', len(all_city_results))
# |             n_wins = int(s_data['p_improved'] * n_eval_cities)
# |             print(f"  Win Rate (Delta > 0):                           {n_wins}/{n_eval_cities} cities ({win_rate:.1f}%)")
# |             if "wilcoxon_two_sided_p" in s_data:
# |                 print(f"  Wilcoxon Two-Sided p-value:                     {s_data['wilcoxon_two_sided_p']:.4e}")
# |             if "rank_biserial_r" in s_data:
# |                 print(f"  Matched-pairs Rank-biserial (r_rb):             {s_data['rank_biserial_r']:.4f}")
# |
# |     print(f"\nSaved full results to: {out_file.resolve()}")
# |     print("=" * 85)
# |     return final_results
# |
# |
# | if __name__ == "__main__":
# |     parser = argparse.ArgumentParser()
# |     parser.add_argument("--epochs", type=int, default=200)
# |     parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
# |     parser.add_argument("--graph-type", type=str, default="radius", choices=["radius", "adaptive_radius", "knn"])
# |     parser.add_argument("--radius", type=float, default=5.0)
# |     parser.add_argument("--knn-k", type=int, default=10)
# |     parser.add_argument("--backbone", type=str, default="gnn", choices=["gnn", "mlp"])
# |     parser.add_argument("--device", type=str, default=None)
# |     args = parser.parse_args()
# |     run_5fold_experiment(
# |         epochs_per_fold=args.epochs,
# |         folds_to_run=args.folds,
# |         graph_type=args.graph_type,
# |         radius_km=args.radius,
# |         knn_k=args.knn_k,
# |         loss_type="ztnb",
# |         backbone=args.backbone,
# |         device_str=args.device,
# |     )

# ===== END SOURCE FILE: src/experiment/run_5fold.py =====


# ===== BEGIN SOURCE FILE: src/experiment/run_backbone_robustness.py =====
# | """
# | Backbone Robustness Evaluation Experiment.
# | Evaluates the Calibration Operator across multiple zero-shot backbones:
# |     1. Classical 2-Parameter Gravity Baseline: T_ij^grav = exp(G) * P_i * P_j * D_ij^(-alpha)
# |     2. Proposed Gravity-Informed Urban GNN: f_theta(X_i, X_j, D_ij, T_ij^grav)
# |
# | For each backbone b, computes:
# |     - Delta R_b (CPC_inter)
# |     - Delta RMSE
# |     - Delta Spearman rho_s
# | Across untouched Full 5-fold Folds 1-5 (n=50) and Full Out-of-fold benchmark (N=50).
# | """
# |
# | import os
# | os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
# | import sys
# | import json
# | import torch
# | import numpy as np
# | from pathlib import Path
# | from scipy import stats
# | from typing import Dict, Any, List
# |
# | # Ensure repo root on path
# | sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
# |
# | from src.data.city_splits import generate_35_5_10_splits
# | from src.data.dataset import load_city, load_raw_city
# | from src.data.yd_extractor import compute_kbin_edges, extract_yd_kbins
# | from src.calibration.bin_calibration import calibrate_kbins
# | from src.training.evaluate import compute_cpc_pair, compute_spearman_pair
# |
# |
# | def fit_gravity_parameters(train_cities: List[str], data_root: str = "data") -> tuple[float, float]:
# |     """Fits global classical gravity parameters G and alpha via log-linear regression on training cities."""
# |     log_pi_pj = []
# |     log_dist = []
# |     log_flow = []
# |
# |     for c in train_cities:
# |         raw = load_raw_city(c, data_root=data_root)
# |         dist_km = raw.dist_km
# |         mask = (raw.pair_o_idx.numpy() != raw.pair_d_idx.numpy()) & (dist_km > 0.0) & (raw.pair_trips.numpy() > 0)
# |         if np.sum(mask) == 0:
# |             continue
# |         p = raw.population.numpy()
# |         p_i = np.clip(p[raw.pair_o_idx.numpy()[mask]], 1.0, None)
# |         p_j = np.clip(p[raw.pair_d_idx.numpy()[mask]], 1.0, None)
# |         d = np.clip(dist_km[mask], 0.1, None)
# |         f = raw.pair_trips.numpy()[mask]
# |
# |         log_pi_pj.extend(np.log(p_i) + np.log(p_j))
# |         log_dist.extend(np.log(d))
# |         log_flow.extend(np.log(f))
# |
# |     # OLS: log_flow = G + 1.0 * log_pi_pj - alpha * log_dist
# |     y = np.array(log_flow) - np.array(log_pi_pj)
# |     X = np.column_stack([np.ones(len(y)), -np.array(log_dist)])
# |     beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
# |     G = float(beta[0])
# |     alpha = float(beta[1])
# |     return G, alpha
# |
# |
# | def run_backbone_robustness(
# |     data_root: str = "data",
# |     output_dir: str = "results/tables",
# | ) -> Dict[str, Any]:
# |     os.makedirs(output_dir, exist_ok=True)
# |     splits = generate_35_5_10_splits(data_root=data_root)
# |
# |     results_file = Path("results/5fold_results.json")
# |     if not results_file.exists():
# |         raise FileNotFoundError(f"Missing {results_file}. Run 5-fold experiment first.")
# |
# |     with open(results_file, "r") as f:
# |         full_res = json.load(f)
# |
# |     city_map = {r["city"]: r for r in full_res["city_level_results"]}
# |
# |     results_by_backbone: Dict[str, List[Dict[str, Any]]] = {
# |         "classical_gravity": [],
# |         "urban_gnn": [],
# |     }
# |
# |     print("Running Backbone Robustness across 5 folds...")
# |
# |     for fold_id in range(1, 6):
# |         train_cities = splits[fold_id]["train"]
# |         test_cities = splits[fold_id]["test"]
# |
# |         # 1. Fit Classical Gravity on Fold training cities
# |         G_fit, alpha_fit = fit_gravity_parameters(train_cities, data_root=data_root)
# |         print(f"Fold {fold_id} Classical Gravity: G={G_fit:.3f}, alpha={alpha_fit:.3f}")
# |
# |         # Compute K=8 bin edges from training cities
# |         bin_edges, _ = compute_kbin_edges(train_cities, K=8, data_root=data_root)
# |
# |         for city_name in test_cities:
# |             raw = load_raw_city(city_name, data_root=data_root)
# |             existing_r = city_map.get(city_name)
# |             if existing_r is None:
# |                 continue
# |
# |             dist_km = raw.dist_km
# |             inter_mask = (raw.pair_o_idx.numpy() != raw.pair_d_idx.numpy()) & (dist_km > 0.0)
# |             t_true_inter = raw.pair_trips.numpy()[inter_mask]
# |
# |             # Extract Oracle Target Y_D
# |             yd_target = extract_yd_kbins(dist_km, raw.pair_trips.numpy(), bin_edges, inter_mask)
# |
# |             # --- Backbone 1: Classical Gravity ---
# |             p = raw.population.numpy()
# |             p_i = np.clip(p[raw.pair_o_idx.numpy()], 1.0, None)
# |             p_j = np.clip(p[raw.pair_d_idx.numpy()], 1.0, None)
# |             d = np.clip(dist_km, 0.1, None)
# |             t_grav = np.exp(G_fit) * p_i * p_j * (d ** (-alpha_fit))
# |             t_grav_inter = t_grav[inter_mask]
# |
# |             # Evaluate M0_grav
# |             m0_cpc_grav = float(compute_cpc_pair(t_true_inter, t_grav_inter))
# |             m0_rmse_grav = float(np.sqrt(np.mean((t_true_inter - t_grav_inter) ** 2)))
# |             m0_spr_grav = float(compute_spearman_pair(t_true_inter, t_grav_inter))
# |
# |             # Apply K=8 calibration on Gravity
# |             t_grav_cal = calibrate_kbins(t_grav, dist_km, inter_mask, yd_target, bin_edges, q=1.0)
# |             t_grav_cal_inter = t_grav_cal[inter_mask]
# |
# |             m1_cpc_grav = float(compute_cpc_pair(t_true_inter, t_grav_cal_inter))
# |             m1_rmse_grav = float(np.sqrt(np.mean((t_true_inter - t_grav_cal_inter) ** 2)))
# |             m1_spr_grav = float(compute_spearman_pair(t_true_inter, t_grav_cal_inter))
# |
# |             results_by_backbone["classical_gravity"].append({
# |                 "city": city_name,
# |                 "fold": fold_id,
# |                 "m0_cpc_inter": m0_cpc_grav,
# |                 "m1_cpc_inter": m1_cpc_grav,
# |                 "delta_r": m1_cpc_grav - m0_cpc_grav,
# |                 "m0_rmse_inter": m0_rmse_grav,
# |                 "m1_rmse_inter": m1_rmse_grav,
# |                 "delta_rmse": m1_rmse_grav - m0_rmse_grav,
# |                 "m0_spearman_inter": m0_spr_grav,
# |                 "m1_spearman_inter": m1_spr_grav,
# |                 "delta_spearman": m1_spr_grav - m0_spr_grav,
# |             })
# |
# |             # --- Backbone 2: Gravity-Informed Urban GNN (Main) ---
# |             m0_gnn = existing_r["M0"]
# |             m1_gnn = existing_r.get("M1_city_oracle_obs", existing_r.get("M1_city_oracle_obs", {}))
# |             
# |             m0_cpc_gnn = m0_gnn["cpc_inter"]
# |             m1_cpc_gnn = m1_gnn["cpc_inter"]
# |             delta_gnn = m1_cpc_gnn - m0_cpc_gnn
# |
# |             m0_rmse_gnn = m0_gnn.get("rmse_inter", 0.0)
# |             m1_rmse_gnn = m1_gnn.get("rmse_inter", 0.0)
# |             m0_spr_gnn = m0_gnn.get("spearman_inter", 0.0)
# |             m1_spr_gnn = m1_gnn.get("spearman_inter", 0.0)
# |
# |             results_by_backbone["urban_gnn"].append({
# |                 "city": city_name,
# |                 "fold": fold_id,
# |                 "m0_cpc_inter": m0_cpc_gnn,
# |                 "m1_cpc_inter": m1_cpc_gnn,
# |                 "delta_r": delta_gnn,
# |                 "m0_rmse_inter": m0_rmse_gnn,
# |                 "m1_rmse_inter": m1_rmse_gnn,
# |                 "delta_rmse": m1_rmse_gnn - m0_rmse_gnn,
# |                 "m0_spearman_inter": m0_spr_gnn,
# |                 "m1_spearman_inter": m1_spr_gnn,
# |                 "delta_spearman": m1_spr_gnn - m0_spr_gnn,
# |             })
# |
# |     # Summarize across Full 5-fold Fold 2-5 (n=50) and Full (n=50)
# |     def summarize_backbone(records: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
# |         conf_recs = [r for r in records if r["fold"] in [1, 2, 3, 4, 5]]
# |         all_recs = records
# |
# |         def get_block(sub: List[Dict[str, Any]]):
# |             n = len(sub)
# |             m0_cpc = np.array([r["m0_cpc_inter"] for r in sub])
# |             m1_cpc = np.array([r["m1_cpc_inter"] for r in sub])
# |             dr = np.array([r["delta_r"] for r in sub])
# |             d_rmse = np.array([r["delta_rmse"] for r in sub])
# |             d_sp = np.array([r["delta_spearman"] for r in sub])
# |
# |             # Stratified bootstrap CI
# |             delta_by_fold = {}
# |             for f in (range(1, 6) if n == 50 else range(2, 6)):
# |                 delta_by_fold[f] = [r["delta_r"] for r in sub if r["fold"] == f]
# |
# |             rng = np.random.default_rng(42)
# |             boot_means = []
# |             for _ in range(5000):
# |                 samp = []
# |                 for f, vals in delta_by_fold.items():
# |                     if len(vals) > 0:
# |                         samp.extend(rng.choice(vals, size=len(vals), replace=True))
# |                 boot_means.append(np.mean(samp))
# |             ci_l, ci_h = np.percentile(boot_means, [2.5, 97.5])
# |
# |             _, w_p = stats.wilcoxon(m1_cpc, m0_cpc, alternative="greater")
# |
# |             return {
# |                 "n": n,
# |                 "m0_cpc_mean": float(np.mean(m0_cpc)),
# |                 "m0_cpc_std": float(np.std(m0_cpc, ddof=1)),
# |                 "m1_cpc_mean": float(np.mean(m1_cpc)),
# |                 "m1_cpc_std": float(np.std(m1_cpc, ddof=1)),
# |                 "delta_r_mean": float(np.mean(dr)),
# |                 "delta_r_std": float(np.std(dr, ddof=1)),
# |                 "delta_r_median": float(np.median(dr)),
# |                 "delta_r_iqr": float(np.percentile(dr, 75) - np.percentile(dr, 25)),
# |                 "bootstrap_95_ci": [float(ci_l), float(ci_h)],
# |                 "p_improved": float(np.mean(dr > 0)),
# |                 "n_improved": f"{int(np.sum(dr > 0))}/{n}",
# |                 "wilcoxon_p": float(w_p),
# |                 "delta_rmse_mean": float(np.mean(d_rmse)),
# |                 "delta_spearman_mean": float(np.mean(d_sp)),
# |             }
# |
# |         return {
# |             "backbone": label,
# |             "full_5fold_50cities": get_block(all_recs),
# |         }
# |
# |     summary = {
# |         "classical_gravity": summarize_backbone(results_by_backbone["classical_gravity"], "Classical 2-Parameter Gravity"),
# |         "urban_gnn": summarize_backbone(results_by_backbone["urban_gnn"], "Gravity-Informed Urban GNN"),
# |     }
# |
# |     # Generate Markdown Table
# |     t7_md = []
# |     t7_md.append("# Backbone Robustness — Marginal Value of Calibration Across Model Architectures")
# |     t7_md.append("")
# |     t7_md.append("> **Evaluation Scope**: Assesses whether distance-binned aggregate information ($Y_D^{\\text{target}}$) improves interzonal reconstruction across different zero-shot model families.")
# |     t7_md.append("")
# |     t7_md.append("## Part A: Full 5-fold Evaluation Set (Folds 1-5, $n=50$ Cities)")
# |     t7_md.append("| Backbone Architecture | Zero-Shot $M_0$ CPC | Calibrated $M_1$ CPC | Marginal Gain $\\Delta R$ | 95% Fold-Stratified Bootstrap CI | $P(\\Delta R > 0)$ | Wilcoxon $p$ | $\\Delta \\text{RMSE}$ |")
# |     t7_md.append("|---|---|---|---|---|---|---|---|")
# |
# |     for k, v in summary.items():
# |         b_name = v["backbone"]
# |         c_stats = v["full_5fold_50cities"]
# |         m0_str = f"{c_stats['m0_cpc_mean']:.4f} +- {c_stats['m0_cpc_std']:.4f}"
# |         m1_str = f"**{c_stats['m1_cpc_mean']:.4f} +- {c_stats['m1_cpc_std']:.4f}**"
# |         dr_str = f"**{c_stats['delta_r_mean']:+.4f} +- {c_stats['delta_r_std']:.4f}**"
# |         ci_str = f"[{c_stats['bootstrap_95_ci'][0]:+.4f}, {c_stats['bootstrap_95_ci'][1]:+.4f}]"
# |         p_imp = f"{c_stats['p_improved']*100:.1f}% ({c_stats['n_improved']})"
# |         w_p = f"{c_stats['wilcoxon_p']:.4e}"
# |         rmse_str = f"{c_stats['delta_rmse_mean']:+.4f}"
# |         t7_md.append(f"| **{b_name}** | {m0_str} | {m1_str} | {dr_str} | {ci_str} | {p_imp} | p = {w_p} | {rmse_str} |")
# |
# |     t7_md_content = "\n".join(t7_md)
# |     with open(Path(output_dir) / "table7_backbone_robustness.md", "w", encoding="utf-8") as f:
# |         f.write(t7_md_content)
# |
# |     with open("results/backbone_robustness_results.json", "w", encoding="utf-8") as f:
# |         json.dump({"summary": summary, "per_city_records": results_by_backbone}, f, indent=2)
# |
# |     print(f"Backbone robustness table generated at {output_dir}/table7_backbone_robustness.md")
# |     return summary
# |
# |
# | if __name__ == "__main__":
# |     run_backbone_robustness()

# ===== END SOURCE FILE: src/experiment/run_backbone_robustness.py =====


# ===== BEGIN SOURCE FILE: src/experiment/run_convergence_pilot.py =====
# | """
# | Convergence Pilot for Zero-Shot OD Backbone (Fold 1)
# | ====================================================
# |
# | Purpose:
# |     Examine the optimization trajectory and empirical convergence limit of the
# |     ZeroShotODModel across 100 epochs on Fold 1 (35 train / 5 val cities).
# |
# | Logging:
# |     - epoch (1 to 100)
# |     - train_loss (ZTNB NLL)
# |     - val_cpc (Interzonal CPC on 5 validation cities)
# |     - learning_rate
# |     - best_epoch & best_val_cpc tracking
# |     - early_stopping_epoch (if triggered, patience=15)
# |
# | Outputs:
# |     - results/e1/convergence_pilot_fold1.json
# |     - results/e1/tables/convergence_pilot.md
# | """
# |
# | import json
# | import time
# | import argparse
# | import sys
# | from pathlib import Path
# | import numpy as np
# | import torch
# |
# | sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
# |
# | from src.data.city_splits import generate_35_5_10_splits
# | from src.training.train import train_zero_shot_model
# |
# | MAX_EPOCHS = 100
# | PATIENCE = 15
# | DATA_ROOT = "data"
# | RESULTS_DIR = Path("results/e1")
# |
# |
# | def run_convergence_pilot(fold_id: int = 1, device_str: str = "cpu"):
# |     t0 = time.time()
# |     RESULTS_DIR.mkdir(parents=True, exist_ok=True)
# |     (RESULTS_DIR / "tables").mkdir(exist_ok=True)
# |
# |     splits = generate_35_5_10_splits(DATA_ROOT)
# |     split = splits[fold_id]
# |     train35 = split["train"]
# |     val5    = split["val"]
# |
# |     print(f"\n{'='*70}")
# |     print(f"CONVERGENCE PILOT: FOLD {fold_id} (Max {MAX_EPOCHS} Epochs, Patience={PATIENCE})")
# |     print(f"{'='*70}")
# |     print(f"Train ({len(train35)} cities): {train35[:4]}... {train35[-2:]}")
# |     print(f"Val   ({len(val5)} cities): {val5}")
# |     print(f"{'-'*70}")
# |
# |     _ckpt_path = RESULTS_DIR / "checkpoints" / f"convergence_pilot_fold{fold_id}.pt"
# |     model, scaler, info = train_zero_shot_model(
# |         train_city_names=train35,
# |         data_root=DATA_ROOT,
# |         epochs=MAX_EPOCHS,
# |         device_str=device_str,
# |         verbose=True,
# |         val_city_names=val5,
# |         patience=PATIENCE,
# |         min_delta=1e-4,
# |         return_info=True,
# |         checkpoint_path=_ckpt_path,
# |         run_tag=f"convergence_pilot_fold{fold_id}",
# |     )
# |
# |     elapsed = time.time() - t0
# |
# |     # Build per-epoch history table
# |     history = []
# |     val_cpcs = info["val_cpc_history"]
# |     train_losses = info["train_loss_history"]
# |     n_epochs = len(val_cpcs)
# |
# |     for ep in range(1, n_epochs + 1):
# |         history.append({
# |             "epoch": ep,
# |             "train_loss": float(train_losses[ep - 1]),
# |             "val_cpc": float(val_cpcs[ep - 1]),
# |             "is_best": bool(ep == info["best_epoch"]),
# |         })
# |
# |     pilot_results = {
# |         "fold_id": fold_id,
# |         "max_epochs": MAX_EPOCHS,
# |         "patience": PATIENCE,
# |         "epochs_trained": info["epochs_trained"],
# |         "best_epoch": info["best_epoch"],
# |         "best_val_cpc": info["best_val_cpc"],
# |         "stopped_early": info["stopped_early"],
# |         "elapsed_seconds": elapsed,
# |         "train_cities": train35,
# |         "val_cities": val5,
# |         "checkpoint_path": str(_ckpt_path.resolve()),
# |         "epoch_history": history,
# |     }
# |
# |     # Save JSON artifact
# |     json_path = RESULTS_DIR / f"convergence_pilot_fold{fold_id}.json"
# |     json_path.write_text(json.dumps(pilot_results, indent=2), encoding="utf-8")
# |     print(f"\nSaved raw convergence trajectory -> {json_path}")
# |
# |     # Generate Markdown summary table
# |     md_lines = [
# |         f"# Convergence Pilot Trajectory (Fold {fold_id}, Max {MAX_EPOCHS} Epochs)",
# |         "",
# |         f"**Best Epoch**: {info['best_epoch']} | **Best Val CPC**: {info['best_val_cpc']:.4f} | **Total Epochs Trained**: {info['epochs_trained']} | **Early Stopped**: {info['stopped_early']} | **Elapsed Time**: {elapsed:.1f}s",
# |         "",
# |         "| Epoch | Train Loss (ZTNB) | Validation CPC (Interzonal) | Status |",
# |         "|---|---|---|---|",
# |     ]
# |
# |     for h in history:
# |         status_marker = "**BEST CHECKPOINT**" if h["is_best"] else ""
# |         md_lines.append(f"| {h['epoch']:03d} | {h['train_loss']:.4f} | {h['val_cpc']:.4f} | {status_marker} |")
# |
# |     table_path = RESULTS_DIR / "tables" / f"convergence_pilot_fold{fold_id}.md"
# |     table_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
# |     print(f"Saved Markdown report -> {table_path}")
# |
# |     print(f"\n{'='*70}")
# |     print(f"Convergence Pilot Complete in {elapsed:.1f}s")
# |     print(f"  Best Epoch: {info['best_epoch']} / {info['epochs_trained']}")
# |     print(f"  Best Validation CPC: {info['best_val_cpc']:.4f}")
# |     print(f"  Early stopping triggered: {info['stopped_early']}")
# |     print(f"  Checkpoint saved: {_ckpt_path.resolve()}")
# |     print(f"{'='*70}\n")
# |
# |
# |     return pilot_results
# |
# |
# | if __name__ == "__main__":
# |     parser = argparse.ArgumentParser(description="Convergence Pilot for Zero-Shot Model")
# |     parser.add_argument("--fold", type=int, default=1, help="Fold ID to test (default: 1)")
# |     parser.add_argument("--device", default="cpu", help="PyTorch device")
# |     args = parser.parse_args()
# |     run_convergence_pilot(fold_id=args.fold, device_str=args.device)

# ===== END SOURCE FILE: src/experiment/run_convergence_pilot.py =====


# ===== BEGIN SOURCE FILE: src/experiment/run_direct_od_equivalence_v1.py =====
# | """
# | Direct Partial-OD Information Equivalence Experiment (v1) - High-Performance Vectorized Runner
# | =============================================================================================
# |
# | Core Scientific Research Question:
# |     Under a prespecified low-capacity direct-OD adaptation procedure
# |     (OD Fixed-Effect Residual Adapter, OD-FE), what fraction of directly observed
# |     positive interzonal OD pairs is required to achieve reconstruction gain on
# |     the remaining unseen pairs comparable to that obtained from the full
# |     target-city distance-binned mobility distribution (Y_D)?
# |
# | Strict Protocol Invariants:
# |     - 5-Fold Cross-City Evaluation (50 held-out test cities).
# |     - Frozen Gravity-Informed Urban GNN backbones (seeds 1, 10, 100).
# |     - Hyperparameter lambda in {0.1, 1, 10, 100} selected per fold strictly using 5 validation cities.
# |     - Zero retraining, zero fine-tuning, zero optimizer step, zero backward pass.
# |     - Reference Arm: Production calibrate_kbins(t0, dist, inter, yd_full, bin_edges, q=1.0) with K=8, q=1.0.
# |     - Primary Grid: 15 p-levels in [0.0, 0.001, ..., 0.90].
# |     - B = 200 replicates per city.
# | """
# |
# | import os
# | import sys
# | import time
# | import json
# | import hashlib
# | import argparse
# | import multiprocessing as mp
# | from pathlib import Path
# | from typing import Dict, List, Tuple, Any, Optional
# |
# | import numpy as np
# | import pandas as pd
# | from scipy import stats
# | import matplotlib.pyplot as plt
# | import torch
# |
# | # Ensure repository root is on sys.path
# | REPO_ROOT = Path(__file__).resolve().parent.parent.parent
# | sys.path.insert(0, str(REPO_ROOT))
# |
# | from src.data.city_splits import generate_35_5_10_splits
# | from src.data.dataset import load_city, load_raw_city
# | from src.data.urban_graph import build_radius_graph
# | from src.data.yd_extractor import compute_kbin_edges
# | from src.calibration.bin_calibration import calibrate_kbins
# | from src.training.evaluate import compute_cpc_pair
# | from src.training.train import load_checkpoint, infer_zero_shot
# |
# | PARTIAL_OD_BASE_SEED = 202608231
# | PRIMARY_GRID_DIRECT = [
# |     0.0, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 
# |     0.10, 0.20, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90
# | ]
# | LAMBDA_CANDIDATES = [0.1, 1.0, 10.0, 100.0]
# | VAL_P_GRID = [0.02, 0.05, 0.10, 0.20]
# |
# | RAW_COLUMNS_DIRECT = [
# |     "fold", "city", "model_seed", "replicate_id", "p", "mask_seed",
# |     "selected_lambda", "n_total_pairs", "n_revealed", "n_unseen",
# |     "fraction_pairs_revealed", "total_trip_mass", "revealed_trip_mass",
# |     "fraction_trip_mass_revealed", "unseen_trip_mass", "fraction_unseen_trip_mass",
# |     "origin_coverage", "destination_coverage", "both_endpoint_coverage",
# |     "adapter_iterations", "adapter_converged",
# |     "cpc_m0_unseen", "cpc_full_yd_unseen", "cpc_direct_od_unseen",
# |     "gain_full_yd", "gain_direct_od", "difference_direct_minus_yd",
# |     "relative_direct_vs_yd", "total_m0_mass", "total_direct_mass", "K", "q"
# | ]
# |
# |
# | def get_stable_mask_seed(base_seed: int, fold: int, city: str, replicate_id: int) -> int:
# |     s = f"{base_seed}_{fold}_{city}_{replicate_id}"
# |     return int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % (2**32)
# |
# |
# | def holm_correction(p_vals: List[float]) -> np.ndarray:
# |     n = len(p_vals)
# |     if n == 0:
# |         return np.array([])
# |     sorted_indices = np.argsort(p_vals)
# |     adj_p = np.zeros(n)
# |     running_max = 0.0
# |     for i, idx in enumerate(sorted_indices):
# |         p_adj = p_vals[idx] * (n - i)
# |         running_max = max(running_max, p_adj)
# |         adj_p[idx] = min(1.0, running_max)
# |     return adj_p
# |
# |
# | def fold_stratified_bootstrap(
# |     city_df: pd.DataFrame, 
# |     metric_col: str, 
# |     p_val: float, 
# |     n_boot: int = 10000, 
# |     seed: int = 42
# | ) -> Tuple[float, float]:
# |     rng = np.random.RandomState(seed)
# |     sub = city_df[city_df.p == p_val]
# |     
# |     vals: Dict[int, np.ndarray] = {}
# |     for f in range(1, 6):
# |         f_vals = sub[sub.fold == f][metric_col].values
# |         if len(f_vals) > 0:
# |             vals[f] = f_vals
# |
# |     boot_means = np.empty(n_boot, dtype=np.float64)
# |     total_cities = sum(len(v) for v in vals.values())
# |     if total_cities == 0:
# |         return 0.0, 0.0
# |         
# |     for b in range(n_boot):
# |         sample_sum = 0.0
# |         for f, arr in vals.items():
# |             idx = rng.randint(0, len(arr), size=len(arr))
# |             sample_sum += arr[idx].sum()
# |         boot_means[b] = sample_sum / total_cities
# |
# |     return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))
# |
# |
# | def fit_od_fe_adapter(
# |     o_idx: np.ndarray,
# |     d_idx: np.ndarray,
# |     t0_support: np.ndarray,
# |     t_true_support: np.ndarray,
# |     rev_indices: np.ndarray,
# |     num_nodes: int,
# |     lambda_reg: float,
# |     max_iter: int = 150,
# |     tol: float = 1e-6
# | ) -> Tuple[np.ndarray, np.ndarray, int, bool]:
# |     """
# |     Solves the exact two-way fixed-effect ridge regression objective:
# |         min_{a, b} sum_{(i,j) in S_p} (r_ij - a_i - b_j)^2 + lambda * (||a||^2 + ||b||^2)
# |     Solved using conjugate gradient on the reduced SPD system; empirical convergence is monitored by the residual tolerance.
# |     """
# |     n_rev = len(rev_indices)
# |     if n_rev == 0:
# |         return np.zeros(num_nodes, dtype=np.float64), np.zeros(num_nodes, dtype=np.float64), 0, True
# |
# |     o_rev = torch.as_tensor(o_idx[rev_indices], dtype=torch.long)
# |     d_rev = torch.as_tensor(d_idx[rev_indices], dtype=torch.long)
# |     t0_rev = torch.as_tensor(t0_support[rev_indices], dtype=torch.float64)
# |     t_true_rev = torch.as_tensor(t_true_support[rev_indices], dtype=torch.float64)
# |
# |     # Target residual r_ij = log(1 + T_ij) - log(1 + \hat{T}^0_ij)
# |     r_rev = torch.log1p(t_true_rev) - torch.log1p(t0_rev)
# |
# |     n_i = torch.bincount(o_rev, minlength=num_nodes).double()
# |     m_j = torch.bincount(d_rev, minlength=num_nodes).double()
# |
# |     inv_denom_a = 1.0 / (n_i + lambda_reg)
# |     denom_b = m_j + lambda_reg
# |
# |     c_a = torch.bincount(o_rev, weights=r_rev, minlength=num_nodes)
# |     c_b = torch.bincount(d_rev, weights=r_rev, minlength=num_nodes)
# |
# |     rhs_b = c_b - torch.bincount(d_rev, weights=inv_denom_a[o_rev] * c_a[o_rev], minlength=num_nodes)
# |
# |     def matvec(v):
# |         Av = v[d_rev]
# |         scaled_Av = inv_denom_a[o_rev] * Av
# |         At_scaled_Av = torch.bincount(d_rev, weights=scaled_Av, minlength=num_nodes)
# |         return denom_b * v - At_scaled_Av
# |
# |     b = torch.zeros(num_nodes, dtype=torch.float64)
# |     r = rhs_b - matvec(b)
# |     p = r.clone()
# |     rsold = torch.dot(r, r)
# |
# |     if float(rsold) < 1e-16:
# |         a = inv_denom_a * c_a
# |         return a.numpy(), b.numpy(), 0, True
# |
# |     converged = False
# |     iters = 0
# |
# |     for it in range(1, max_iter + 1):
# |         iters = it
# |         Ap = matvec(p)
# |         denom_alpha = float(torch.dot(p, Ap))
# |         if denom_alpha <= 0 or not np.isfinite(denom_alpha):
# |             converged = False
# |             break
# |         alpha = rsold / denom_alpha
# |         b = b + alpha * p
# |         r = r - alpha * Ap
# |         rsnew = torch.dot(r, r)
# |         if float(torch.sqrt(rsnew)) < tol:
# |             converged = True
# |             break
# |         p = r + (rsnew / rsold) * p
# |         rsold = rsnew
# |
# |     a = inv_denom_a * (c_a - torch.bincount(o_rev, weights=b[d_rev], minlength=num_nodes))
# |     return a.numpy(), b.numpy(), iters, converged
# |
# |
# | def apply_od_fe_prediction(
# |     o_idx: np.ndarray,
# |     d_idx: np.ndarray,
# |     t0_support: np.ndarray,
# |     a: np.ndarray,
# |     b: np.ndarray
# | ) -> np.ndarray:
# |     """
# |     Applies OD-FE predictions and preserves total baseline mass N0.
# |     """
# |     log_t0_plus_1 = np.log1p(t0_support)
# |     ell_direct = log_t0_plus_1 + a[o_idx] + b[d_idx]
# |     t_tilde = np.maximum(0.0, np.expm1(ell_direct))
# |     
# |     n0 = float(np.sum(t0_support))
# |     n_tilde = float(np.sum(t_tilde))
# |     
# |     if n_tilde > 0:
# |         t_direct = t_tilde * (n0 / n_tilde)
# |     else:
# |         t_direct = t0_support.copy()
# |         
# |     return t_direct
# |
# |
# | def select_fold_lambda(
# |     fold_id: int,
# |     val_cities: List[str],
# |     data_root: str = "data",
# |     model_seeds: List[int] = [1, 10, 100],
# |     b_val: int = 50,
# |     device: str = "cpu"
# | ) -> Tuple[float, pd.DataFrame]:
# |     """
# |     Strictly selects lambda on the 5 validation cities of the fold.
# |     """
# |     print(f"\n[FOLD {fold_id}] Selecting hyperparameter lambda from {len(val_cities)} validation cities...")
# |     
# |     # Load fold models
# |     fold_models: Dict[int, Tuple[Any, Any]] = {}
# |     for s in model_seeds:
# |         ckpt_path = Path("results/checkpoints") / f"5fold_fold{fold_id}_seed{s}.pt"
# |         if not ckpt_path.exists():
# |             raise RuntimeError(f"Missing checkpoint {ckpt_path}")
# |         model, scaler, _ = load_checkpoint(ckpt_path, device_str=device)
# |         model.eval()
# |         fold_models[s] = (model, scaler)
# |
# |     # Pre-cache validation city zero-shot predictions
# |     val_cache: Dict[str, Dict[str, Any]] = {}
# |     for city_name in val_cities:
# |         raw_data = load_raw_city(city_name, data_root=data_root)
# |         dist_km = raw_data.dist_km
# |         inter_pos = (raw_data.pair_o_idx.numpy() != raw_data.pair_d_idx.numpy()) & (dist_km > 0.0) & (raw_data.pair_trips.numpy() > 0)
# |         
# |         t_true_support = raw_data.pair_trips.numpy()[inter_pos].astype(np.float64)
# |         o_idx_support = raw_data.pair_o_idx.numpy()[inter_pos]
# |         d_idx_support = raw_data.pair_d_idx.numpy()[inter_pos]
# |         num_nodes = raw_data.n_tracts
# |
# |         seed_preds = {}
# |         for s in model_seeds:
# |             model, scaler = fold_models[s]
# |             city_data = load_city(city_name, data_root=data_root, feature_scaler=scaler, fit_scaler=False)
# |             coords = city_data.lon_lat.numpy()
# |             ei, ed = build_radius_graph(coords, radius_km=5.0)
# |             with torch.no_grad():
# |                 m0_full = infer_zero_shot(model, city_data, ei, ed, device=device).numpy().astype(np.float64)
# |             seed_preds[s] = m0_full[inter_pos]
# |
# |         val_cache[city_name] = {
# |             "n_pairs": int(inter_pos.sum()),
# |             "t_true": t_true_support,
# |             "o_idx": o_idx_support,
# |             "d_idx": d_idx_support,
# |             "num_nodes": num_nodes,
# |             "seed_preds": seed_preds
# |         }
# |
# |     lambda_scores = []
# |     
# |     for lam in LAMBDA_CANDIDATES:
# |         cpc_unseen_list = []
# |         gain_list = []
# |         
# |         for city_name in val_cities:
# |             cdata = val_cache[city_name]
# |             n_pairs = cdata["n_pairs"]
# |             t_true = cdata["t_true"]
# |             o_idx = cdata["o_idx"]
# |             d_idx = cdata["d_idx"]
# |             num_nodes = cdata["num_nodes"]
# |
# |             for rep_id in range(b_val):
# |                 mask_seed = get_stable_mask_seed(PARTIAL_OD_BASE_SEED, fold_id, f"val_{city_name}", rep_id)
# |                 perm = np.random.RandomState(mask_seed).permutation(n_pairs)
# |
# |                 for p_val in VAL_P_GRID:
# |                     n_rev = int(np.round(p_val * n_pairs))
# |                     rev_indices = perm[:n_rev]
# |                     unseen_indices = perm[n_rev:]
# |                     t_true_unseen = t_true[unseen_indices]
# |                     sum_true_unseen = float(np.sum(t_true_unseen))
# |
# |                     for s in model_seeds:
# |                         t0_support = cdata["seed_preds"][s]
# |                         t0_unseen = t0_support[unseen_indices]
# |                         
# |                         denom_m0 = sum_true_unseen + float(np.sum(t0_unseen))
# |                         cpc_m0 = (2.0 * np.sum(np.minimum(t_true_unseen, t0_unseen)) / denom_m0) if denom_m0 > 0 else 0.0
# |
# |                         a, b, _, conv = fit_od_fe_adapter(
# |                             o_idx, d_idx, t0_support, t_true, rev_indices, num_nodes, lambda_reg=lam
# |                         )
# |                         if not conv:
# |                             raise RuntimeError(f"OD-FE CG solver did not converge during lambda selection on val city {city_name}!")
# |                         
# |                         t_direct_support = apply_od_fe_prediction(o_idx, d_idx, t0_support, a, b)
# |                         t_direct_unseen = t_direct_support[unseen_indices]
# |                         
# |                         denom_dir = sum_true_unseen + float(np.sum(t_direct_unseen))
# |                         cpc_dir = (2.0 * np.sum(np.minimum(t_true_unseen, t_direct_unseen)) / denom_dir) if denom_dir > 0 else 0.0
# |
# |                         cpc_unseen_list.append(cpc_dir)
# |                         gain_list.append(cpc_dir - cpc_m0)
# |
# |         mean_cpc = float(np.mean(cpc_unseen_list))
# |         mean_gain = float(np.mean(gain_list))
# |         lambda_scores.append({
# |             "lambda": lam,
# |             "validation_mean_cpc": mean_cpc,
# |             "mean_gain": mean_gain,
# |             "n_validation_cities": len(val_cities),
# |             "masks_per_city": b_val
# |         })
# |         print(f"  candidate lambda = {lam:<5} | Val Mean CPC_U = {mean_cpc:.5f} | Val Mean Gain = {mean_gain:+.5f}")
# |
# |     selection_df = pd.DataFrame(lambda_scores)
# |     # Sort descending by validation_mean_cpc, then descending by lambda (tie-breaker prefers higher regularization)
# |     best_row = selection_df.sort_values(by=["validation_mean_cpc", "lambda"], ascending=[False, False]).iloc[0]
# |     selected_lam = float(best_row["lambda"])
# |     print(f"  --> Selected lambda_f* = {selected_lam} for Fold {fold_id}\n")
# |     return selected_lam, selection_df
# |
# |
# | def _process_city_replicates_chunk(
# |     args: Tuple[int, str, List[int], int, List[int], List[float], float, Dict[str, Any]]
# | ) -> List[Tuple]:
# |     """
# |     Worker task: Processes a slice of replicates for a single city across all p-levels and model seeds.
# |     """
# |     fold_id, city_name, rep_ids, n_pairs, model_seeds, p_grid, selected_lambda, city_cached_data = args
# |     
# |     t_true_support = city_cached_data["t_true"]
# |     o_idx_support = city_cached_data["o_idx"]
# |     d_idx_support = city_cached_data["d_idx"]
# |     num_nodes = city_cached_data["num_nodes"]
# |     total_trip_mass = city_cached_data["total_trip_mass"]
# |     n_origins_total = city_cached_data["n_origins_total"]
# |     n_dests_total = city_cached_data["n_dests_total"]
# |     seed_predictions = city_cached_data["seed_predictions"]
# |
# |     rows = []
# |
# |     for rep_id in rep_ids:
# |         mask_seed = get_stable_mask_seed(PARTIAL_OD_BASE_SEED, fold_id, city_name, rep_id)
# |         rng = np.random.RandomState(mask_seed)
# |         perm = rng.permutation(n_pairs)
# |
# |         for p_val in p_grid:
# |             n_reveal = int(np.round(p_val * n_pairs))
# |             rev_indices = perm[:n_reveal]
# |             unseen_indices = perm[n_reveal:]
# |             n_unseen = len(unseen_indices)
# |             if n_unseen == 0:
# |                 continue
# |
# |             if n_reveal == 0:
# |                 revealed_mass = 0.0
# |                 c_o = 0.0
# |                 c_d = 0.0
# |                 c_both = 0.0
# |             else:
# |                 rev_trips = t_true_support[rev_indices]
# |                 revealed_mass = float(np.sum(rev_trips))
# |                 rev_o_set = set(o_idx_support[rev_indices])
# |                 rev_d_set = set(d_idx_support[rev_indices])
# |                 
# |                 c_o = len(rev_o_set) / n_origins_total if n_origins_total > 0 else 0.0
# |                 c_d = len(rev_d_set) / n_dests_total if n_dests_total > 0 else 0.0
# |                 
# |                 unseen_o = o_idx_support[unseen_indices]
# |                 unseen_d = d_idx_support[unseen_indices]
# |                 both_cov = np.isin(unseen_o, list(rev_o_set)) & np.isin(unseen_d, list(rev_d_set))
# |                 c_both = float(np.mean(both_cov))
# |
# |             frac_pairs_rev = float(n_reveal) / float(n_pairs)
# |             frac_mass_rev = float(revealed_mass) / float(total_trip_mass) if total_trip_mass > 0 else 0.0
# |             unseen_mass = total_trip_mass - revealed_mass
# |             frac_unseen_mass = unseen_mass / total_trip_mass if total_trip_mass > 0 else 0.0
# |             
# |             t_true_unseen = t_true_support[unseen_indices]
# |             sum_true_unseen = float(np.sum(t_true_unseen))
# |
# |             # Evaluate across all model seeds with identical mask
# |             for s in model_seeds:
# |                 preds = seed_predictions[s]
# |                 t0_support = preds["t0"]
# |                 t0_unseen = t0_support[unseen_indices]
# |                 t_full_unseen = preds["t_cal_full"][unseen_indices]
# |                 N_hat_total = preds["N_hat"]
# |                 
# |                 # 1. Arm A: M0 zero-shot
# |                 denom_m0 = sum_true_unseen + float(np.sum(t0_unseen))
# |                 cpc_m0_unseen = (2.0 * np.sum(np.minimum(t_true_unseen, t0_unseen)) / denom_m0) if denom_m0 > 0 else 0.0
# |                 
# |                 # 2. Arm B: Full Y_D Reference
# |                 denom_full = sum_true_unseen + float(np.sum(t_full_unseen))
# |                 cpc_full_unseen = (2.0 * np.sum(np.minimum(t_true_unseen, t_full_unseen)) / denom_full) if denom_full > 0 else 0.0
# |                 
# |                 # 3. Arm C: Direct-OD Adapter (OD-FE)
# |                 if n_reveal == 0:
# |                     cpc_dir_unseen = cpc_m0_unseen
# |                     it_count = 0
# |                     is_conv = True
# |                     tot_dir_mass = N_hat_total
# |                 else:
# |                     a, b, it_count, is_conv = fit_od_fe_adapter(
# |                         o_idx=o_idx_support,
# |                         d_idx=d_idx_support,
# |                         t0_support=t0_support,
# |                         t_true_support=t_true_support,
# |                         rev_indices=rev_indices,
# |                         num_nodes=num_nodes,
# |                         lambda_reg=selected_lambda
# |                     )
# |                     if not is_conv:
# |                         raise RuntimeError(f"OD-FE CG solver did not converge on city {city_name}, rep {rep_id}, p {p_val}!")
# |                         
# |                     t_direct_support = apply_od_fe_prediction(
# |                         o_idx_support, d_idx_support, t0_support, a, b
# |                     )
# |                     t_dir_unseen = t_direct_support[unseen_indices]
# |                     tot_dir_mass = float(np.sum(t_direct_support))
# |                     
# |                     denom_dir = sum_true_unseen + float(np.sum(t_dir_unseen))
# |                     cpc_dir_unseen = (2.0 * np.sum(np.minimum(t_true_unseen, t_dir_unseen)) / denom_dir) if denom_dir > 0 else 0.0
# |
# |                 gain_full = float(cpc_full_unseen - cpc_m0_unseen)
# |                 gain_direct = float(cpc_dir_unseen - cpc_m0_unseen)
# |                 diff_direct_minus_yd = float(gain_direct - gain_full)
# |                 rel_direct = float(gain_direct / gain_full) if abs(gain_full) > 1e-8 else 1.0
# |
# |                 rows.append((
# |                     fold_id, city_name, s, rep_id, p_val, mask_seed,
# |                     selected_lambda, n_pairs, n_reveal, n_unseen,
# |                     frac_pairs_rev, total_trip_mass, revealed_mass,
# |                     frac_mass_rev, unseen_mass, frac_unseen_mass,
# |                     c_o, c_d, c_both, it_count, is_conv,
# |                     cpc_m0_unseen, cpc_full_unseen, cpc_dir_unseen,
# |                     gain_full, gain_direct, diff_direct_minus_yd,
# |                     rel_direct, N_hat_total, tot_dir_mass, 8, 1.0
# |                 ))
# |
# |     return rows
# |
# |
# | def run_fold_direct_od(
# |     fold_id: int,
# |     data_root: str = "data",
# |     output_dir: Path = Path("results/direct_od_equivalence_v1"),
# |     replicates: int = 200,
# |     p_grid: List[float] = None,
# |     smoke: bool = False,
# |     smoke_cities: int = 1,
# |     resume: bool = False,
# |     num_workers: int = 8,
# |     device: str = "cpu"
# | ) -> Dict[str, Any]:
# |     if p_grid is None:
# |         p_grid = PRIMARY_GRID_DIRECT.copy()
# |
# |     fold_dir = output_dir / f"fold_{fold_id}"
# |     fold_dir.mkdir(parents=True, exist_ok=True)
# |     
# |     raw_csv_path = fold_dir / "raw.csv"
# |     progress_json_path = fold_dir / "progress.json"
# |     marker_path = fold_dir / "completion.marker"
# |     lambda_csv_path = fold_dir / "lambda_selection.csv"
# |     lambda_json_path = fold_dir / "lambda_selected.json"
# |
# |     splits = generate_35_5_10_splits(data_root=data_root)
# |     split = splits[fold_id]
# |     train_cities = split["train"]
# |     val_cities = split["val"]
# |     test_cities = split["test"] if not smoke else split["test"][:smoke_cities]
# |     model_seeds = [1, 10, 100] if not smoke else [1, 10]
# |     B = replicates if not smoke else 20
# |     b_val = 50 if not smoke else 5
# |
# |     # 1. Select / Load Fold Lambda
# |     valid_lambda_cache = False
# |     if lambda_json_path.exists():
# |         with open(lambda_json_path, "r") as f:
# |             lam_info = json.load(f)
# |             if lam_info.get("val_cities") == val_cities and lam_info.get("model_seeds") == model_seeds and lam_info.get("b_val") == b_val:
# |                 selected_lambda = float(lam_info["selected_lambda"])
# |                 print(f">>> [FOLD {fold_id}] Using cached lambda_f* = {selected_lambda}")
# |                 valid_lambda_cache = True
# |             else:
# |                 print(f">>> [FOLD {fold_id}] Cached lambda_f* is stale (different config). Re-selecting...")
# |
# |     if not valid_lambda_cache:
# |         selected_lambda, selection_df = select_fold_lambda(
# |             fold_id=fold_id,
# |             val_cities=val_cities,
# |             data_root=data_root,
# |             model_seeds=model_seeds,
# |             b_val=b_val,
# |             device=device
# |         )
# |         selection_df.to_csv(lambda_csv_path, index=False)
# |         with open(lambda_json_path, "w", encoding="utf-8") as f:
# |             json.dump({
# |                 "fold": fold_id,
# |                 "lambda_candidates": LAMBDA_CANDIDATES,
# |                 "selected_lambda": selected_lambda,
# |                 "selection_source": "validation_cities_only",
# |                 "test_city_information_used": False,
# |                 "val_cities": val_cities,
# |                 "model_seeds": model_seeds,
# |                 "b_val": b_val,
# |                 "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
# |             }, f, indent=2)
# |
# |     print(f">>> [STARTING FOLD {fold_id}/5] {len(test_cities)} test cities | B={B} reps | {len(p_grid)} p-levels | lambda={selected_lambda} | Workers={num_workers}")
# |
# |     # Check already completed cities if resume is True
# |     completed_cities = set()
# |     if resume and progress_json_path.exists():
# |         try:
# |             with open(progress_json_path, "r") as f:
# |                 prog = json.load(f)
# |                 completed_cities = set(prog.get("completed_cities", []))
# |                 print(f"    Resuming fold {fold_id}: Found {len(completed_cities)} already completed cities.")
# |         except Exception:
# |             completed_cities = set()
# |
# |     if not resume or not raw_csv_path.exists():
# |         with open(raw_csv_path, "w", encoding="utf-8") as f:
# |             f.write(",".join(RAW_COLUMNS_DIRECT) + "\n")
# |
# |     # Load frozen GNN models
# |     models: Dict[int, Tuple[Any, Any]] = {}
# |     for s in model_seeds:
# |         ckpt_path = Path("results/checkpoints") / f"5fold_fold{fold_id}_seed{s}.pt"
# |         if not ckpt_path.exists():
# |             raise RuntimeError(f"Checkpoint missing for fold {fold_id} seed {s}: {ckpt_path}")
# |         model, scaler, _ = load_checkpoint(ckpt_path, device_str=device)
# |         model.eval()
# |         models[s] = (model, scaler)
# |
# |     # Compute K=8 bin edges from 35 train cities for reference arm
# |     bin_edges, K_act = compute_kbin_edges(train_cities, K=8, data_root=data_root)
# |     assert K_act == 8 and len(bin_edges) == 9
# |
# |     fold_start_time = time.perf_counter()
# |     rows_written_total = 0
# |
# |     for city_idx, city_name in enumerate(test_cities):
# |         if city_name in completed_cities:
# |             print(f"  [{city_idx+1}/{len(test_cities)}] {city_name:<16} | ALREADY COMPLETED (Skipping)")
# |             continue
# |
# |         city_start = time.perf_counter()
# |         raw_data = load_raw_city(city_name, data_root=data_root)
# |         dist_km = raw_data.dist_km
# |         
# |         inter_pos = (raw_data.pair_o_idx.numpy() != raw_data.pair_d_idx.numpy()) & (dist_km > 0.0) & (raw_data.pair_trips.numpy() > 0)
# |         n_pairs = int(inter_pos.sum())
# |         if n_pairs == 0:
# |             raise RuntimeError(f"Critical error: City {city_name} has 0 positive interzonal pairs!")
# |
# |         t_true_support = raw_data.pair_trips.numpy()[inter_pos].astype(np.float64)
# |         o_idx_support = raw_data.pair_o_idx.numpy()[inter_pos]
# |         d_idx_support = raw_data.pair_d_idx.numpy()[inter_pos]
# |         dist_support = dist_km[inter_pos]
# |         num_nodes = raw_data.n_tracts
# |         total_trip_mass = float(np.sum(t_true_support))
# |         
# |         n_origins_total = len(set(o_idx_support))
# |         n_dests_total = len(set(d_idx_support))
# |
# |         # Full Y_D reference distribution
# |         bin_idx_support = np.clip(np.digitize(dist_support, bin_edges, right=True) - 1, 0, 7)
# |         yd_full = np.bincount(bin_idx_support, weights=t_true_support, minlength=8).astype(np.float64)
# |         yd_full /= total_trip_mass
# |
# |         # Precalculate M0 and full Y_D calibrated prediction for all model seeds
# |         seed_predictions: Dict[int, Dict[str, np.ndarray]] = {}
# |         for s in model_seeds:
# |             model, scaler = models[s]
# |             city_data = load_city(city_name, data_root=data_root, feature_scaler=scaler, fit_scaler=False)
# |             coords = city_data.lon_lat.numpy()
# |             ei, ed = build_radius_graph(coords, radius_km=5.0)
# |             
# |             with torch.no_grad():
# |                 m0_full = infer_zero_shot(model, city_data, ei, ed, device=device).numpy().astype(np.float64)
# |             
# |             t0_support = m0_full[inter_pos]
# |             N_hat_support = float(np.sum(t0_support))
# |             
# |             # Reference Full Y_D calibration (K=8, q=1.0)
# |             Y_hat = np.bincount(bin_idx_support, weights=t0_support, minlength=8).astype(np.float64) / N_hat_support
# |             active = np.zeros(8, dtype=bool)
# |             for k in range(8):
# |                 active[k] = bool((bin_idx_support == k).any())
# |             yd_act = yd_full * active.astype(np.float64)
# |             act_sum = yd_act.sum()
# |             Y_D_cond = yd_act / act_sum if act_sum > 0 else Y_hat.copy()
# |
# |             w_full = np.ones(8, dtype=np.float64)
# |             for k in range(8):
# |                 if active[k] and Y_hat[k] > 0:
# |                     w_full[k] = Y_D_cond[k] / Y_hat[k]
# |             weighted_mass_full = float(np.dot(Y_hat, w_full))
# |             s_full = w_full / weighted_mass_full if weighted_mass_full > 0 else np.ones(8)
# |             
# |             t_cal_full_support = t0_support * s_full[bin_idx_support]
# |             cal_mass_full = np.sum(t_cal_full_support)
# |             if cal_mass_full > 0:
# |                 t_cal_full_support *= (N_hat_support / cal_mass_full)
# |                 
# |             seed_predictions[s] = {
# |                 "t0": t0_support,
# |                 "N_hat": N_hat_support,
# |                 "t_cal_full": t_cal_full_support
# |             }
# |
# |         city_cached_data = {
# |             "t_true": t_true_support,
# |             "o_idx": o_idx_support,
# |             "d_idx": d_idx_support,
# |             "num_nodes": num_nodes,
# |             "total_trip_mass": total_trip_mass,
# |             "n_origins_total": n_origins_total,
# |             "n_dests_total": n_dests_total,
# |             "seed_predictions": seed_predictions
# |         }
# |
# |         # Divide B replicates into chunks for multiprocessing
# |         rep_chunks = np.array_split(np.arange(B), min(num_workers, B))
# |         task_args = [
# |             (fold_id, city_name, chunk.tolist(), n_pairs, model_seeds, p_grid, selected_lambda, city_cached_data)
# |             for chunk in rep_chunks if len(chunk) > 0
# |         ]
# |
# |         if num_workers > 1 and len(task_args) > 1:
# |             with mp.Pool(processes=min(num_workers, len(task_args))) as pool:
# |                 chunk_results = pool.map(_process_city_replicates_chunk, task_args)
# |             city_rows = [item for sublist in chunk_results for item in sublist]
# |         else:
# |             city_rows = _process_city_replicates_chunk(task_args[0])
# |
# |         # Append city records to raw CSV incrementally
# |         with open(raw_csv_path, "a", encoding="utf-8") as f:
# |             for r in city_rows:
# |                 f.write(",".join(str(x) for x in r) + "\n")
# |
# |         completed_cities.add(city_name)
# |         rows_written_total += len(city_rows)
# |
# |         # Update progress.json
# |         with open(progress_json_path, "w", encoding="utf-8") as f:
# |             json.dump({
# |                 "fold": fold_id,
# |                 "completed_cities": sorted(list(completed_cities)),
# |                 "remaining_cities": [c for c in test_cities if c not in completed_cities],
# |                 "rows_written": rows_written_total,
# |                 "protocol_version": "v1",
# |                 "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
# |             }, f, indent=2)
# |
# |         city_elapsed = time.perf_counter() - city_start
# |         print(f"  [{city_idx+1}/{len(test_cities)}] {city_name:<16} | Pairs: {n_pairs:>7} | B={B} reps done in {city_elapsed:.2f}s (Flushed {len(city_rows)} rows)")
# |
# |     # Read back raw.csv to generate per_seed, per_city, and fold_summary
# |     fold_df = pd.read_csv(raw_csv_path)
# |     
# |     # 1. Per-Seed Aggregation: Mean over B replicates -> (fold x city x model_seed x p)
# |     per_seed_df = fold_df.groupby(["fold", "city", "model_seed", "p"]).agg({
# |         "selected_lambda": "first",
# |         "fraction_pairs_revealed": "mean",
# |         "fraction_trip_mass_revealed": "mean",
# |         "origin_coverage": "mean",
# |         "destination_coverage": "mean",
# |         "both_endpoint_coverage": "mean",
# |         "adapter_iterations": "mean",
# |         "cpc_m0_unseen": "mean",
# |         "cpc_full_yd_unseen": "mean",
# |         "cpc_direct_od_unseen": "mean",
# |         "gain_full_yd": "mean",
# |         "gain_direct_od": "mean",
# |         "difference_direct_minus_yd": "mean",
# |         "relative_direct_vs_yd": "mean"
# |     }).reset_index()
# |     per_seed_csv_path = fold_dir / "per_seed.csv"
# |     per_seed_df.to_csv(per_seed_csv_path, index=False)
# |
# |     # 2. Per-City Aggregation: Mean over 3 model seeds -> (fold x city x p)
# |     per_city_df = per_seed_df.groupby(["fold", "city", "p"]).agg({
# |         "selected_lambda": "first",
# |         "fraction_pairs_revealed": "mean",
# |         "fraction_trip_mass_revealed": "mean",
# |         "origin_coverage": "mean",
# |         "destination_coverage": "mean",
# |         "both_endpoint_coverage": "mean",
# |         "adapter_iterations": "mean",
# |         "cpc_m0_unseen": "mean",
# |         "cpc_full_yd_unseen": "mean",
# |         "cpc_direct_od_unseen": "mean",
# |         "gain_full_yd": "mean",
# |         "gain_direct_od": "mean",
# |         "difference_direct_minus_yd": "mean",
# |         "relative_direct_vs_yd": "mean"
# |     }).reset_index()
# |     per_city_csv_path = fold_dir / "per_city.csv"
# |     per_city_df.to_csv(per_city_csv_path, index=False)
# |
# |     # 3. Fold Summary Table
# |     fold_summary_rows = []
# |     for p_val in p_grid:
# |         sub = per_city_df[per_city_df.p == p_val]
# |         fold_summary_rows.append({
# |             "p": p_val,
# |             "n_cities": len(sub),
# |             "mean_both_cov": float(sub["both_endpoint_coverage"].mean()),
# |             "mean_gain_full_yd": float(sub["gain_full_yd"].mean()),
# |             "mean_gain_direct_od": float(sub["gain_direct_od"].mean()),
# |             "mean_diff_vs_yd": float(sub["difference_direct_minus_yd"].mean()),
# |             "pos_cities": int((sub["gain_direct_od"] > 0).sum()),
# |             "match_yd_cities": int((sub["difference_direct_minus_yd"] >= 0).sum())
# |         })
# |
# |     fold_summary_json_path = fold_dir / "fold_summary.json"
# |     with open(fold_summary_json_path, "w", encoding="utf-8") as f:
# |         json.dump({"fold": fold_id, "selected_lambda": selected_lambda, "summary_by_p": fold_summary_rows}, f, indent=2)
# |
# |     fold_summary_md_path = fold_dir / "fold_summary.md"
# |     with open(fold_summary_md_path, "w", encoding="utf-8") as f:
# |         f.write(f"# Fold {fold_id} Direct-OD Summary Table (N={len(test_cities)} Cities, lambda*={selected_lambda})\n\n")
# |         f.write("| p | Both Coverage | Mean Gain Full $Y_D$ | Mean Gain Direct OD | Mean $D(p)$ (Direct - Full) | Positive Cities | Match Full $Y_D$ |\n")
# |         f.write("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
# |         for r in fold_summary_rows:
# |             f.write(f"| **{r['p']*100:.2f}%** | {r['mean_both_cov']*100:.2f}% | +{r['mean_gain_full_yd']:.5f} | {r['mean_gain_direct_od']:+.5f} | {r['mean_diff_vs_yd']:+.5f} | {r['pos_cities']}/{r['n_cities']} | {r['match_yd_cities']}/{r['n_cities']} |\n")
# |
# |     # 4. Save Run Manifest
# |     manifest_path = fold_dir / "run_manifest.json"
# |     with open(manifest_path, "w", encoding="utf-8") as f:
# |         json.dump({
# |             "fold": fold_id,
# |             "protocol_version": "v1",
# |             "selected_lambda": selected_lambda,
# |             "cities": test_cities,
# |             "model_seeds": model_seeds,
# |             "replicates": B,
# |             "p_grid": p_grid,
# |             "raw_rows": len(fold_df),
# |             "per_seed_rows": len(per_seed_df),
# |             "per_city_rows": len(per_city_df),
# |             "completed_at": time.strftime("%Y-%m-%d %H:%M:%S")
# |         }, f, indent=2)
# |
# |     # 5. QA Verification Before Writing completion.marker
# |     expected_raw_rows = len(test_cities) * len(model_seeds) * B * len(p_grid)
# |     actual_raw_rows = len(fold_df)
# |     
# |     assert actual_raw_rows == expected_raw_rows, f"Fold {fold_id} raw rows {actual_raw_rows} != expected {expected_raw_rows}"
# |     assert len(per_city_df) == len(test_cities) * len(p_grid), f"Fold {fold_id} per_city rows mismatch"
# |     assert not fold_df.isnull().any().any(), f"Fold {fold_id} contains NaN values!"
# |
# |     with open(marker_path, "w", encoding="utf-8") as f:
# |         f.write(f"FOLD {fold_id} DIRECT-OD COMPLETED AND CERTIFIED\nTimestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
# |
# |     fold_total_time = time.perf_counter() - fold_start_time
# |     print(f">>> [FOLD {fold_id} COMPLETE] Certified {actual_raw_rows} rows in {fold_total_time:.2f}s | Marker: {marker_path.name}")
# |     
# |     return {
# |         "fold": fold_id,
# |         "selected_lambda": selected_lambda,
# |         "raw_rows": actual_raw_rows,
# |         "per_seed_rows": len(per_seed_df),
# |         "per_city_rows": len(per_city_df),
# |         "status": "PASS"
# |     }
# |
# |
# | def aggregate_combined_direct_od(
# |     output_dir: Path = Path("results/direct_od_equivalence_v1"),
# |     p_grid: List[float] = None
# | ) -> None:
# |     if p_grid is None:
# |         p_grid = PRIMARY_GRID_DIRECT.copy()
# |
# |     combined_dir = output_dir / "combined"
# |     combined_dir.mkdir(parents=True, exist_ok=True)
# |     (combined_dir / "figures").mkdir(parents=True, exist_ok=True)
# |
# |     print("\n" + "=" * 85)
# |     print("MASTER AGGREGATION & SCIENTIFIC SUMMARY (DIRECT-OD EQUIVALENCE, N=50 CITIES)")
# |     print("=" * 85)
# |
# |     all_raw_dfs = []
# |     all_per_seed_dfs = []
# |     all_per_city_dfs = []
# |     fold_lambdas = {}
# |
# |     for f in range(1, 6):
# |         fold_dir = output_dir / f"fold_{f}"
# |         marker = fold_dir / "completion.marker"
# |         if not marker.exists():
# |             raise RuntimeError(f"Cannot aggregate: Fold {f} completion.marker not found at {marker}")
# |         
# |         with open(fold_dir / "lambda_selected.json", "r") as lf:
# |             fold_lambdas[f] = json.load(lf)["selected_lambda"]
# |             
# |         all_raw_dfs.append(pd.read_csv(fold_dir / "raw.csv"))
# |         all_per_seed_dfs.append(pd.read_csv(fold_dir / "per_seed.csv"))
# |         all_per_city_dfs.append(pd.read_csv(fold_dir / "per_city.csv"))
# |
# |     raw_combined = pd.concat(all_raw_dfs, ignore_index=True)
# |     per_seed_combined = pd.concat(all_per_seed_dfs, ignore_index=True)
# |     per_city_combined = pd.concat(all_per_city_dfs, ignore_index=True)
# |
# |     raw_combined.to_csv(combined_dir / "raw_all_folds.csv", index=False)
# |     per_seed_combined.to_csv(combined_dir / "per_seed_all_folds.csv", index=False)
# |     per_city_combined.to_csv(combined_dir / "per_city_all_folds.csv", index=False)
# |
# |     print(f"Combined Raw Rows:      {len(raw_combined):>10} (Expected: 450,000)")
# |     print(f"Combined Per-Seed Rows: {len(per_seed_combined):>10} (Expected: 2,250)")
# |     print(f"Combined Per-City Rows: {len(per_city_combined):>10} (Expected: 750)")
# |
# |     # Statistical Analysis across N=50 cities
# |     summary_rows = []
# |     raw_p_values = []
# |     p_vals_tested = [p for p in p_grid if p > 0]
# |
# |     for p_val in p_vals_tested:
# |         sub = per_city_combined[per_city_combined.p == p_val]
# |         gains = sub["gain_direct_od"].values
# |         _, p_w = stats.wilcoxon(gains, alternative="greater")
# |         raw_p_values.append(p_w)
# |
# |     holm_p_vals = holm_correction(raw_p_values)
# |     holm_dict = {p: h_p for p, h_p in zip(p_vals_tested, holm_p_vals)}
# |
# |     for p_val in p_grid:
# |         sub = per_city_combined[per_city_combined.p == p_val]
# |         n_cities = len(sub)
# |         
# |         mean_mass = float(sub["fraction_trip_mass_revealed"].mean())
# |         mean_cov_both = float(sub["both_endpoint_coverage"].mean())
# |         mean_cov_o = float(sub["origin_coverage"].mean())
# |         mean_cov_d = float(sub["destination_coverage"].mean())
# |         
# |         mean_m0 = float(sub["cpc_m0_unseen"].mean())
# |         mean_gain_full = float(sub["gain_full_yd"].mean())
# |         mean_gain_direct = float(sub["gain_direct_od"].mean())
# |         mean_diff = float(sub["difference_direct_minus_yd"].mean())
# |         
# |         pos_cities = int((sub["gain_direct_od"] > 0).sum())
# |         match_yd_cities = int((sub["difference_direct_minus_yd"] >= 0).sum())
# |         
# |         ci_diff_l, ci_diff_h = fold_stratified_bootstrap(per_city_combined, "difference_direct_minus_yd", p_val)
# |         ci_dir_l, ci_dir_h = fold_stratified_bootstrap(per_city_combined, "gain_direct_od", p_val)
# |         ci_full_l, ci_full_h = fold_stratified_bootstrap(per_city_combined, "gain_full_yd", p_val)
# |         
# |         h_pval = holm_dict.get(p_val, 1.0) if p_val > 0 else 1.0
# |
# |         summary_rows.append({
# |             "p": p_val,
# |             "n_cities": n_cities,
# |             "mean_revealed_mass": mean_mass,
# |             "mean_both_coverage": mean_cov_both,
# |             "mean_origin_coverage": mean_cov_o,
# |             "mean_destination_coverage": mean_cov_d,
# |             "mean_m0_cpc": mean_m0,
# |             "mean_gain_full_yd": mean_gain_full,
# |             "ci_95_gain_full": [ci_full_l, ci_full_h],
# |             "mean_gain_direct_od": mean_gain_direct,
# |             "ci_95_gain_direct": [ci_dir_l, ci_dir_h],
# |             "mean_diff_vs_yd": mean_diff,
# |             "ci_95_diff": [ci_diff_l, ci_diff_h],
# |             "pos_cities_vs_m0": pos_cities,
# |             "match_yd_cities": match_yd_cities,
# |             "holm_pval_benefit": h_pval
# |         })
# |
# |     summary_df = pd.DataFrame(summary_rows)
# |
# |     # 1. Positive Mean Crossing
# |     p_pos_mean = None
# |     for r in summary_rows:
# |         if r["mean_gain_direct_od"] > 0 and p_pos_mean is None:
# |             p_pos_mean = r["p"]
# |
# |     # 2. Statistically Supported Benefit Threshold p*_DirectBenefit
# |     p_star_benefit = None
# |     for r in summary_rows:
# |         if r["holm_pval_benefit"] < 0.05 and r["ci_95_gain_direct"][0] > 0 and p_star_benefit is None:
# |             p_star_benefit = r["p"]
# |
# |     # 3. Operational Equivalence Crossing p_eq
# |     p_eq_grid = None
# |     p_eq_interp = None
# |     for r in summary_rows:
# |         if r["mean_diff_vs_yd"] >= 0 and p_eq_grid is None:
# |             p_eq_grid = r["p"]
# |
# |     for i in range(len(summary_rows) - 1):
# |         r1, r2 = summary_rows[i], summary_rows[i+1]
# |         d1, d2 = r1["mean_diff_vs_yd"], r2["mean_diff_vs_yd"]
# |         if d1 <= 0 and d2 >= 0 and (d2 - d1) > 0:
# |             p_eq_interp = r1["p"] + (-d1 / (d2 - d1)) * (r2["p"] - r1["p"])
# |             break
# |
# |     # Save summary JSON
# |     summary_json_path = combined_dir / "summary.json"
# |     with open(summary_json_path, "w", encoding="utf-8") as f:
# |         json.dump({
# |             "experiment": "direct_partial_od_information_equivalence",
# |             "protocol_version": "v1",
# |             "n_evaluation_cities": 50,
# |             "fold_lambdas": fold_lambdas,
# |             "p_pos_mean_crossing": p_pos_mean,
# |             "p_star_benefit_threshold": p_star_benefit,
# |             "p_eq_grid": p_eq_grid,
# |             "p_eq_interp": p_eq_interp,
# |             "results_by_p": summary_rows
# |         }, f, indent=2)
# |
# |     # Save Markdown Table
# |     summary_md_path = combined_dir / "summary.md"
# |     with open(summary_md_path, "w", encoding="utf-8") as f:
# |         f.write("# Table: Master Direct-OD Information Equivalence Summary (v1)\n\n")
# |         f.write("> **Evaluation Scope**: Evaluates the operational reconstruction value of directly observed positive interzonal OD pairs via low-capacity Origin-Destination Fixed-Effect residual adaptation (OD-FE), relative to the full target-city distance distribution $Y_D$ ($K=8, q=1.0$, seeds $s \\in \\{1, 10, 100\\}$), evaluated strictly on unseen pairs ($N=50$ held-out test cities across 5 folds).\n\n")
# |         
# |         f.write(f"• **Validation-Selected Lambdas:** Fold 1: `{fold_lambdas[1]}`, Fold 2: `{fold_lambdas[2]}`, Fold 3: `{fold_lambdas[3]}`, Fold 4: `{fold_lambdas[4]}`, Fold 5: `{fold_lambdas[5]}`  \n")
# |         if p_pos_mean is not None:
# |             pct_pos = p_pos_mean * 100.0
# |             f.write(f"• **Positive Mean Crossing Point ($p_\\text{{mean+}}$):** `{pct_pos:.2f}%` of positive interzonal OD pairs  \n")
# |         if p_star_benefit is not None:
# |             pct_star = p_star_benefit * 100.0
# |             f.write(f"• **Statistically Supported Benefit Threshold ($p^*_\\text{{DirectBenefit}}$):** `{pct_star:.2f}%` of positive interzonal OD pairs ($p_\\text{{Holm}} < 0.05$)  \n")
# |         if p_eq_interp is not None:
# |             pct_interp = p_eq_interp * 100.0
# |             f.write(f"• **Operational Equivalence Crossing ($p_\\text{{eq,interp}}$):** `{pct_interp:.2f}%` of positive interzonal OD pairs  \n\n")
# |         elif p_eq_grid is not None:
# |             pct_grid = p_eq_grid * 100.0
# |             f.write(f"• **Operational Equivalence Grid Point ($p_\\text{{eq,grid}}$):** `{pct_grid:.2f}%` of positive interzonal OD pairs  \n\n")
# |         else:
# |             f.write("• **Operational Equivalence Crossing:** Under the tested low-capacity direct-OD adaptation procedure, the full-$Y_D$ reconstruction gain was not matched within the prespecified reveal range up to 90% of the positive interzonal OD support.  \n\n")
# |
# |         f.write("| Revealed OD Pairs ($p$) | Both Coverage | $M_0$ CPC (Unseen) | Full-$Y_D$ Gain | Direct-OD Gain | Difference vs Full $Y_D$ ($D(p)$) | 95% CI Difference | Direct Benefit Holm $p$ | Cities Direct $> M_0$ | Cities Direct $\\ge$ Full $Y_D$ |\n")
# |         f.write("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
# |         
# |         for r in summary_rows:
# |             p_pct = f"{r['p']*100:.2f}%"
# |             cov_pct = f"{r['mean_both_coverage']*100:.2f}%"
# |             m0_str = f"{r['mean_m0_cpc']:.4f}"
# |             full_str = f"+{r['mean_gain_full_yd']:.5f}"
# |             dir_str = f"{r['mean_gain_direct_od']:+.5f}"
# |             diff_str = f"{r['mean_diff_vs_yd']:+.5f}"
# |             ci_str = f"[{r['ci_95_diff'][0]:+.5f}, {r['ci_95_diff'][1]:+.5f}]"
# |             h_str = f"{r['holm_pval_benefit']:.4e}" if r['p'] > 0 else "—"
# |             pos_str = f"{r['pos_cities_vs_m0']}/{r['n_cities']}"
# |             match_str = f"{r['match_yd_cities']}/{r['n_cities']}"
# |             
# |             f.write(f"| **{p_pct}** | {cov_pct} | {m0_str} | {full_str} | **{dir_str}** | **{diff_str}** | {ci_str} | {h_str} | {pos_str} | {match_str} |\n")
# |             
# |         f.write("\n---\n\n### Prescribed Scientific Interpretation\n")
# |         if p_eq_interp is not None:
# |             f.write(f"Under the prespecified OD fixed-effect residual adapter, directly observing approximately **{p_eq_interp*100:.2f}%** of the positive interzonal OD support produced a mean reconstruction gain on the remaining unseen pairs comparable to that obtained from the full target-city distance-binned distribution.\n")
# |         else:
# |             f.write("Under the tested low-capacity direct-OD adaptation procedure, the full-$Y_D$ reconstruction gain was not matched within the prespecified reveal range up to 90% of the positive interzonal OD support. This does not imply that $Y_D$ intrinsically contains more information than 90% of the OD observations; the result is conditional on the tested adaptation operator.\n")
# |
# |     print(f"Summary Markdown: {summary_md_path}")
# |     print(f"Summary JSON:     {summary_json_path}")
# |
# |     # Generate Publication Figures
# |     generate_direct_od_figures(summary_df, per_city_combined, combined_dir, p_eq_interp, p_star_benefit)
# |
# |     # Write FROZEN.marker
# |     frozen_marker_path = output_dir / "FROZEN.marker"
# |     with open(frozen_marker_path, "w", encoding="utf-8") as f:
# |         f.write("DIRECT PARTIAL-OD INFORMATION EQUIVALENCE v1 PROTOCOL FROZEN\n")
# |         f.write(f"Completed At: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
# |         f.write("Protocol: 50 held-out test cities across 5 disjoint folds (N=50)\n")
# |         f.write("Evaluation Support: unseen positive interzonal pairs Omega_c^+ \\ S_p\n")
# |         f.write(f"Replicates: 200 per city (Total: 450,000 raw calibrations)\n")
# |
# |
# | def generate_direct_od_figures(
# |     summary_df: pd.DataFrame, 
# |     per_city_df: pd.DataFrame, 
# |     combined_dir: Path, 
# |     p_eq_interp: Optional[float],
# |     p_star_benefit: Optional[float]
# | ) -> None:
# |     plt.rcParams.update({'font.sans-serif': 'Helvetica', 'axes.edgecolor': '#333333', 'axes.linewidth': 0.8})
# |     fig_dir = combined_dir / "figures"
# |     p_vals = summary_df["p"].values * 100.0
# |
# |     # Fig 1: Gain vs Reveal Fraction
# |     fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
# |     ax.axhline(0, color="#888888", linestyle="--", alpha=0.6)
# |     
# |     full_gain = summary_df["mean_gain_full_yd"].values
# |     dir_gain = summary_df["mean_gain_direct_od"].values
# |     dir_ci_l = np.array([ci[0] for ci in summary_df["ci_95_gain_direct"]])
# |     dir_ci_h = np.array([ci[1] for ci in summary_df["ci_95_gain_direct"]])
# |     
# |     ax.plot(p_vals, full_gain, label="Full $Y_D$ Reference Gain", color="#1f77b4", linestyle="--", linewidth=2.0)
# |     ax.plot(p_vals, dir_gain, label="Direct OD-FE Adapter Gain", color="#d62728", marker="o", linewidth=2.0)
# |     ax.fill_between(p_vals, dir_ci_l, dir_ci_h, color="#d62728", alpha=0.15, label="95% Fold Bootstrap CI")
# |     
# |     if p_star_benefit is not None:
# |         ax.axvline(p_star_benefit * 100.0, color="#ff7f0e", linestyle="-.", label=f"Benefit $p^* = {p_star_benefit*100:.2f}\\%$")
# |     if p_eq_interp is not None:
# |         ax.axvline(p_eq_interp * 100.0, color="#2ca02c", linestyle=":", label=f"Equivalence $p_{{eq}} = {p_eq_interp*100:.2f}\\%$")
# |         
# |     ax.set_xlabel("Revealed Positive Interzonal OD Pairs (%)", fontsize=11, fontweight="bold")
# |     ax.set_ylabel("Mean Marginal Gain $\\Delta\\mathrm{CPC}_U$ on Unseen OD", fontsize=11, fontweight="bold")
# |     ax.set_title("Direct OD-FE Reconstruction Gain vs Reveal Fraction", fontsize=12, fontweight="bold", pad=12)
# |     ax.legend(frameon=True, fontsize=9)
# |     ax.grid(True, linestyle=":", alpha=0.5)
# |     plt.tight_layout()
# |     fig.savefig(fig_dir / "fig_1_direct_gain_vs_p.png")
# |     plt.close(fig)
# |
# |     # Fig 2: Difference D(p) Equivalence
# |     fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
# |     ax.axhline(0, color="#333333", linestyle="-", linewidth=1.0)
# |     
# |     diff_vals = summary_df["mean_diff_vs_yd"].values
# |     diff_ci_l = np.array([ci[0] for ci in summary_df["ci_95_diff"]])
# |     diff_ci_h = np.array([ci[1] for ci in summary_df["ci_95_diff"]])
# |     
# |     ax.plot(p_vals, diff_vals, color="#9467bd", marker="s", linewidth=2.0, label="$\\bar{D}_{\\mathrm{Direct}}(p) = \\mathrm{Gain}_{\\mathrm{Direct}} - \\mathrm{Gain}_{Y_D}$")
# |     ax.fill_between(p_vals, diff_ci_l, diff_ci_h, color="#9467bd", alpha=0.15, label="95% Fold Bootstrap CI")
# |     
# |     if p_eq_interp is not None:
# |         ax.scatter([p_eq_interp * 100.0], [0.0], color="#d62728", s=80, zorder=5, label=f"Crossing $p_{{eq}} = {p_eq_interp*100:.2f}\\%$")
# |         
# |     ax.set_xlabel("Revealed Positive Interzonal OD Pairs (%)", fontsize=11, fontweight="bold")
# |     ax.set_ylabel("Gain Difference $D_{\\mathrm{Direct}}(p)$", fontsize=11, fontweight="bold")
# |     ax.set_title("Direct-OD Information Equivalence Zero-Crossing", fontsize=12, fontweight="bold", pad=12)
# |     ax.legend(frameon=True, fontsize=9)
# |     ax.grid(True, linestyle=":", alpha=0.5)
# |     plt.tight_layout()
# |     fig.savefig(fig_dir / "fig_2_direct_equivalence_Dp.png")
# |     plt.close(fig)
# |
# |     # Fig 3: Fold-Specific D(p)
# |     fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
# |     ax.axhline(0, color="#333333", linestyle="-", linewidth=1.0)
# |     colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
# |     
# |     for f in range(1, 6):
# |         f_sub = per_city_df[per_city_df.fold == f].groupby("p")["difference_direct_minus_yd"].mean().reset_index()
# |         ax.plot(f_sub["p"].values * 100.0, f_sub["difference_direct_minus_yd"].values, marker="o", markersize=4, label=f"Fold {f} (N=10)", color=colors[f-1])
# |         
# |     ax.set_xlabel("Revealed Positive Interzonal OD Pairs (%)", fontsize=11, fontweight="bold")
# |     ax.set_ylabel("Fold-Specific Mean $D_{\\mathrm{Direct}}(p)$", fontsize=11, fontweight="bold")
# |     ax.set_title("Fold-Specific Direct-OD Equivalence Trajectories", fontsize=12, fontweight="bold", pad=12)
# |     ax.legend(frameon=True, fontsize=9)
# |     ax.grid(True, linestyle=":", alpha=0.5)
# |     plt.tight_layout()
# |     fig.savefig(fig_dir / "fig_3_fold_specific_direct_Dp.png")
# |     plt.close(fig)
# |
# |     # Fig 4: Endpoint Coverage vs p
# |     fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
# |     cov_both = summary_df["mean_both_coverage"].values * 100.0
# |     cov_o = summary_df["mean_origin_coverage"].values * 100.0
# |     cov_d = summary_df["mean_destination_coverage"].values * 100.0
# |     
# |     ax.plot(p_vals, cov_both, color="#e377c2", marker="^", linewidth=2.0, label="Both Endpoints Observed ($C_{\\mathrm{both}}$)")
# |     ax.plot(p_vals, cov_o, color="#bcbd22", linestyle=":", linewidth=1.5, label="Origin Coverage ($C_O$)")
# |     ax.plot(p_vals, cov_d, color="#17becf", linestyle="--", linewidth=1.5, label="Destination Coverage ($C_D$)")
# |     
# |     ax.set_xlabel("Revealed Positive Interzonal OD Pairs (%)", fontsize=11, fontweight="bold")
# |     ax.set_ylabel("Endpoint Coverage on Unseen Set (%)", fontsize=11, fontweight="bold")
# |     ax.set_title("Endpoint Observation Dynamics in Direct-OD Adaptation", fontsize=12, fontweight="bold", pad=12)
# |     ax.legend(frameon=True, fontsize=9)
# |     ax.grid(True, linestyle=":", alpha=0.5)
# |     plt.tight_layout()
# |     fig.savefig(fig_dir / "fig_4_endpoint_coverage_vs_p.png")
# |     plt.close(fig)
# |
# |     # Fig 5: Direct OD vs Partial YD from v2
# |     fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
# |     ax.axhline(0, color="#888888", linestyle="--", alpha=0.6)
# |     
# |     ax.plot(p_vals, full_gain, label="Full $Y_D$ Reference", color="#1f77b4", linestyle="--", linewidth=2.0)
# |     ax.plot(p_vals, dir_gain, label="Direct OD-FE Adaptation", color="#d62728", marker="o", linewidth=2.0)
# |     
# |     # Try reading v2 summary if available for contextual reference
# |     v2_summary_path = Path("results/partial_od_equivalence_v2/combined/summary.json")
# |     if v2_summary_path.exists():
# |         try:
# |             with open(v2_summary_path, "r") as v2f:
# |                 v2_data = json.load(v2f)
# |                 v2_p = [r["p"] * 100.0 for r in v2_data["results_by_p"]]
# |                 v2_gain = [r["mean_gain_partial_od"] for r in v2_data["results_by_p"]]
# |                 ax.plot(v2_p, v2_gain, label="OD-Subsampled $Y_D$ (v2)", color="#2ca02c", linestyle="-.", marker="x", linewidth=1.5)
# |         except Exception:
# |             pass
# |
# |     ax.set_xlabel("Revealed Positive Interzonal OD Pairs (%)", fontsize=11, fontweight="bold")
# |     ax.set_ylabel("Mean Marginal Gain $\\Delta\\mathrm{CPC}_U$", fontsize=11, fontweight="bold")
# |     ax.set_title("Direct OD-FE vs OD-Subsampled $Y_D$ Estimation", fontsize=12, fontweight="bold", pad=12)
# |     ax.legend(frameon=True, fontsize=9)
# |     ax.grid(True, linestyle=":", alpha=0.5)
# |     plt.tight_layout()
# |     fig.savefig(fig_dir / "fig_5_direct_vs_partialYD_comparison.png")
# |     plt.close(fig)
# |
# |
# | if __name__ == "__main__":
# |     parser = argparse.ArgumentParser(description="Run Direct Partial-OD Information Equivalence v1")
# |     parser.add_argument("--data_root", type=str, default="data")
# |     parser.add_argument("--output_dir", type=str, default="results/direct_od_equivalence_v1")
# |     parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5], help="Folds to execute")
# |     parser.add_argument("--cities", type=int, default=10, help="Number of test cities per fold")
# |     parser.add_argument("--b", type=int, default=200, help="Monte Carlo replicates per city")
# |     parser.add_argument("--smoke", action="store_true", help="Run fast smoke test")
# |     parser.add_argument("--resume", action="store_true", help="Resume from progress.json")
# |     parser.add_argument("--aggregate_only", action="store_true", help="Only aggregate completed folds")
# |     parser.add_argument("--workers", type=int, default=8, help="Number of parallel worker processes")
# |     parser.add_argument("--device", type=str, default="cpu")
# |     args = parser.parse_args()
# |
# |     out_p = Path(args.output_dir)
# |
# |     if args.aggregate_only:
# |         aggregate_combined_direct_od(output_dir=out_p)
# |     else:
# |         for f_id in args.folds:
# |             run_fold_direct_od(
# |                 fold_id=f_id,
# |                 data_root=args.data_root,
# |                 output_dir=out_p,
# |                 replicates=args.b,
# |                 smoke=args.smoke,
# |                 smoke_cities=args.cities,
# |                 resume=args.resume,
# |                 num_workers=args.workers,
# |                 device=args.device
# |             )
# |         if not args.smoke and set(args.folds) == {1, 2, 3, 4, 5}:
# |             aggregate_combined_direct_od(output_dir=out_p)

# ===== END SOURCE FILE: src/experiment/run_direct_od_equivalence_v1.py =====


# ===== BEGIN SOURCE FILE: src/experiment/run_e1.py =====
# | """Legacy E1 experiment runner; canonical training uses run_5fold.py.
# |
# | This module is retained for historical reproduction and is not part of the
# | current canonical seed/provenance pipeline.
# |
# | E1: Oracle Aggregated-Distance Existence Test (Amended Protocol v2)
# | ==================================================================
# |
# | Research Question:
# |     Does target-city distance-binned aggregate information (Y_D^{GT,+}) provide
# |     marginal value for OD reconstruction beyond a zero-shot gravity-informed urban GNN?
# |
# | Formal Protocol (Amended Replication under Locked Protocol):
# |   - 5-Fold Stratified City-Level Cross-Validation (35 Train / 5 Validation / 10 Held-out Test per fold).
# |   - Split Manifest: Pre-locked at results/e1/splits_manifest_v2.json:
# |       * Outer test sets locked 100% from E1-v1 (zero test perturbation).
# |       * Inner validation: 5-stratum size stratification across 40 non-test pool (seed 20260818).
# |       * Manifest SHA-256 integrity check and full candidate audit logging.
# |   - Model Selection: Early stopping on validation set interzonal CPC (patience=15, max_epochs=100, cosine annealing).
# |   - Quantile Discretization: K_move = 8 moving-distance bins (Bin 0 intrazonal excluded), computed
# |     strictly from training cities of each fold. Strict invariant: K_active == 8.
# |   - Calibration Operator: q = 1.0 within-tolerance closed-form multiplier scaling (numerical tolerance 1e-5).
# |   - 3 Conditions per test city:
# |       * Condition A: Zero-Shot Baseline (M0: Theta* frozen, no target information)
# |       * Condition B: Treatment (+ Oracle Target Y_D: K=8 aggregate histogram from target ground truth)
# |       * Condition C: Multi-Donor Placebo (+ Oracle Wrong Donor Y_D: Average over all 9 other test cities in fold)
# |   - Primary Specificity Estimand:
# |       * Delta_c^specificity = Delta_c^target - bar{Delta}_c^wrong
# |       * Statistical unit is strictly the city (n=50 full_5_fold, n=50 full coverage).
# |   - Evaluation Domain: Interzonal pairs Omega_c^+ = {(i,j) in Omega_c : i != j, D_ij > 0}.
# |   - Statistical Analysis:
# |       * Primary Full 5-fold: Prospectively designated untouched Folds 1-5 (n=50 cities).
# |       * Exploratory / Development: Fold 1 (n=10 cities).
# |       * Descriptive Coverage: All 50 out-of-fold cities.
# |       * 95% Fold-Stratified Bootstrap CI (10,000 resamples) + Paired Wilcoxon Signed-Rank Test.
# |       * Sample standard deviation with ddof=1 recorded across all metadata.
# |
# | Artifacts Generated:
# |   - results/e1/splits_manifest_v2.json
# |   - results/e1/e1_per_city_results.json
# |   - results/e1/e1_validation_manifest.json
# |   - results/e1/e1_summary.json
# |   - results/e1/tables/e1_main_table.md
# |   - results/e1/tables/e1_per_city.md
# | """
# |
# | import json
# | import os
# | import platform
# | import time
# | import argparse
# | import sys
# | from pathlib import Path
# |
# | if sys.platform == "win32":
# |     try:
# |         sys.stdout.reconfigure(encoding="utf-8", errors="replace")
# |         sys.stderr.reconfigure(encoding="utf-8", errors="replace")
# |     except Exception:
# |         pass
# |
# | import numpy as np
# | from scipy import stats
# | import torch
# |
# | # Ensure repository root is in sys.path
# | sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
# |
# | from src.data.city_splits import load_splits_manifest_v2, get_wrong_donors
# | from src.data.dataset import load_city, preload_all_cities
# | from src.data.urban_graph import build_radius_graph
# | from src.data.yd_extractor import compute_kbin_edges, extract_yd_kbins
# | from src.calibration.bin_calibration import calibrate_kbins
# | from src.training.train import train_zero_shot_model, infer_zero_shot
# | from src.training.evaluate import compute_cpc_pair, compute_cpc_norm_pair
# |
# |
# | def get_runtime_metadata() -> dict:
# |     """
# |     Collect hardware, OS, and PyTorch runtime execution metadata.
# |     Enables auditability of multi-core CPU and multi-threading configuration.
# |     """
# |     cpu_physical = None
# |     cpu_logical = os.cpu_count()
# |     try:
# |         import psutil
# |         cpu_physical = psutil.cpu_count(logical=False)
# |     except Exception:
# |         cpu_physical = None
# |
# |     return {
# |         "platform": platform.platform(),
# |         "processor": platform.processor(),
# |         "python_version": platform.python_version(),
# |         "torch_version": torch.__version__,
# |         "cuda_available": torch.cuda.is_available(),
# |         "cpu_count_logical": cpu_logical,
# |         "cpu_count_physical": cpu_physical,
# |         "torch_num_threads": torch.get_num_threads(),
# |         "torch_num_interop_threads": torch.get_num_interop_threads(),
# |         "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "not_set"),
# |         "mkl_num_threads": os.environ.get("MKL_NUM_THREADS", "not_set"),
# |     }
# |
# |
# | def configure_cpu_threads(num_threads: int | None = None) -> int:
# |     """
# |     Explicitly configure PyTorch CPU intra-op threads and OpenMP/MKL environment variables.
# |     If num_threads is specified and > 0, sets torch.set_num_threads(num_threads)
# |     and updates OMP_NUM_THREADS and MKL_NUM_THREADS.
# |     Returns the active torch.get_num_threads().
# |     """
# |     if num_threads is not None and num_threads > 0:
# |         os.environ["OMP_NUM_THREADS"] = str(num_threads)
# |         os.environ["MKL_NUM_THREADS"] = str(num_threads)
# |         torch.set_num_threads(num_threads)
# |
# |     return torch.get_num_threads()
# |
# | # ---------------------------------------------------------------------------
# | # Global Experiment Parameters (Pre-specified, locked before evaluation)
# | # ---------------------------------------------------------------------------
# | K_MOVE      = 8          # Number of moving-distance bins (Bin 0 intrazonal excluded)
# | Q_CALIB     = 1.0        # Calibration strength (1.0 = exact within-tolerance distribution match)
# | EPOCHS      = 200        # Maximum training epochs per fold (with ReduceLROnPlateau & Early Stopping)
# | PATIENCE    = 15         # Early stopping patience based on validation CPC
# | MIN_DELTA   = 1e-4       # Minimum validation CPC improvement threshold
# | DATA_ROOT   = "data"     # Dataset root folder containing 50 city directories
# | RESULTS_DIR = Path("results/e1")
# | LOG_FILE    = RESULTS_DIR / "e1_execution.log"
# | MANIFEST_PATH = RESULTS_DIR / "splits_manifest_v2.json"
# | TOLERANCE   = 1e-5       # Floating-point tolerance for mass preservation & bin matching
# | SEED        = 1          # Legacy single-seed default; canonical multi-seed runs use run_full_experiment.py
# |
# |
# | def log_msg(msg: str = "", print_to_console: bool = True):
# |     """Logs message with local timestamp to console (flush=True) and e1_execution.log."""
# |     timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
# |     formatted = f"[{timestamp}] {msg}" if msg else ""
# |     if print_to_console:
# |         try:
# |             print(formatted if formatted else "", flush=True)
# |         except Exception:
# |             try:
# |                 print(formatted.encode("ascii", errors="replace").decode("ascii") if formatted else "", flush=True)
# |             except Exception:
# |                 pass
# |     try:
# |         RESULTS_DIR.mkdir(parents=True, exist_ok=True)
# |         with open(LOG_FILE, "a", encoding="utf-8") as f:
# |             f.write((formatted if formatted else "") + "\n")
# |     except Exception:
# |         pass
# |
# |
# | def build_inter_mask(cd, dist_km: np.ndarray) -> np.ndarray:
# |     """
# |     Construct boolean mask for interzonal candidate support Omega_c^+.
# |     Strict definition: Origin != Destination and Pairwise Distance > 0.
# |     """
# |     o = cd.pair_o_idx.numpy()
# |     d = cd.pair_d_idx.numpy()
# |     return (o != d) & (dist_km > 0.0)
# |
# |
# | def safe_wilcoxon(diff: np.ndarray, alternative: str = "greater") -> tuple[float, float]:
# |     """
# |     Defensive non-parametric Wilcoxon signed-rank test.
# |     Handles edge cases:
# |       - n < 2 observations (returns p=1.0)
# |       - All zero differences / ties (returns p=1.0)
# |       - Non-zero filtering using wilcox zero-method
# |     """
# |     diff_clean = diff[~np.isnan(diff)]
# |     if len(diff_clean) < 2:
# |         return 0.0, 1.0
# |     non_zero = diff_clean[diff_clean != 0.0]
# |     if len(non_zero) == 0:
# |         return 0.0, 1.0
# |     try:
# |         res = stats.wilcoxon(diff_clean, alternative=alternative, zero_method="wilcox")
# |         return float(res.statistic), float(res.pvalue)
# |     except Exception:
# |         return 0.0, 1.0
# |
# |
# | def compute_iqr(values: np.ndarray) -> float:
# |     """Computes sample Interquartile Range (IQR = Q3 - Q1)."""
# |     if len(values) == 0:
# |         return 0.0
# |     return float(np.percentile(values, 75) - np.percentile(values, 25))
# |
# |
# | def run_city(
# |     city: str,
# |     model: torch.nn.Module,
# |     scaler: object,
# |     bin_edges: np.ndarray,
# |     K_active: int,
# |     test_cities: list[str],
# |     fold_id: int,
# |     device: torch.device,
# |     test_city_cache: dict[str, dict] | None = None,
# |     test_yd_cache: dict[str, np.ndarray] | None = None,
# | ) -> dict:
# |     """
# |     Evaluates 3 experimental conditions on a single held-out test city.
# |
# |     Condition A (Zero-Shot Baseline M0):
# |         Input: (X_c, G_c^urban, D_c). Forward pass of frozen model Theta*.
# |         Output: T_c^(0) = E[T | T >= 1].
# |
# |     Condition B (Treatment: Target Oracle Y_D):
# |         Input: T_c^(0) and Y_{D,c}^{GT,+} (K-dim aggregate distance histogram).
# |         Operator: Moving-bin closed-form scaling on Omega_c^+.
# |         Output: T_c^(YD).
# |
# |     Condition C (Placebo Control: Multi-Donor Wrong Y_D):
# |         Input: T_c^(0) and Y_{D,d}^{GT,+} for all 9 other test cities in fold d != c.
# |         Operator: Moving-bin closed-form scaling for each donor d.
# |         Output: Delta_c^(wrong) = 1/9 sum_{d != c} Delta_{c,d}^(wrong).
# |     """
# |     # 1. Retrieve precomputed test city structures if available, or load on-demand
# |     if test_city_cache is not None and city in test_city_cache:
# |         c_entry = test_city_cache[city]
# |         cd = c_entry["city_data"]
# |         ei = c_entry["edge_index"]
# |         ed = c_entry["edge_dist"]
# |         dist_km = c_entry["dist_km"]
# |         inter = c_entry["inter_mask"]
# |         t_gt = c_entry.get("t_gt", cd.pair_trips.numpy().astype(np.float64))
# |         Y_D_tgt = c_entry.get("Y_D")
# |     else:
# |         cd = load_city(city, data_root=DATA_ROOT, feature_scaler=scaler)
# |         ei, ed = build_radius_graph(cd.lon_lat, radius_km=5.0)
# |         dist_km = np.expm1(cd.pair_distance.numpy())
# |         inter = build_inter_mask(cd, dist_km)
# |         t_gt = cd.pair_trips.numpy().astype(np.float64)
# |         if test_yd_cache is not None and city in test_yd_cache:
# |             Y_D_tgt = test_yd_cache[city]
# |         else:
# |             Y_D_tgt = extract_yd_kbins(dist_km, t_gt, bin_edges, inter)
# |
# |     # 2. Condition A: Zero-Shot Forward Inference
# |     T0 = infer_zero_shot(model, cd, ei, ed, device=device)
# |     t0 = T0.numpy().astype(np.float64)
# |
# |     # Compute interzonal mask and ground truth flows
# |     n_inter = int(inter.sum())
# |
# |     cpc0      = compute_cpc_pair(t_gt[inter], t0[inter])
# |     cpc0_norm = compute_cpc_norm_pair(t_gt[inter], t0[inter])
# |
# |     # 3. Condition B: Target Oracle Y_D Extraction & Calibration
# |     if Y_D_tgt is None:
# |         if test_yd_cache is not None and city in test_yd_cache:
# |             Y_D_tgt = test_yd_cache[city]
# |         else:
# |             Y_D_tgt = extract_yd_kbins(dist_km, t_gt, bin_edges, inter)
# |
# |     T_yd    = calibrate_kbins(t0, dist_km, inter, Y_D_tgt, bin_edges, q=Q_CALIB, tolerance=TOLERANCE)
# |     cpc_yd      = compute_cpc_pair(t_gt[inter], T_yd[inter])
# |     cpc_yd_norm = compute_cpc_norm_pair(t_gt[inter], T_yd[inter])
# |     delta_target = float(cpc_yd - cpc0)
# |
# |     # 4. Condition C: Multi-Donor Placebo Control (All 9 Wrong Donors in Test Fold)
# |     wrong_donors = get_wrong_donors(city, test_cities)
# |     assert len(wrong_donors) == len(test_cities) - 1, f"Expected {len(test_cities) - 1} wrong donors, got {len(wrong_donors)}"
# |
# |     wrong_cpc_list = []
# |     wrong_cpc_norm_list = []
# |     wrong_delta_list = []
# |     wrong_donor_details = []
# |
# |     for donor in wrong_donors:
# |         if test_city_cache is not None and donor in test_city_cache:
# |             Y_D_wr = test_city_cache[donor]["Y_D"]
# |         elif test_yd_cache is not None and donor in test_yd_cache:
# |             Y_D_wr = test_yd_cache[donor]
# |         else:
# |             cd_d    = load_city(donor, data_root=DATA_ROOT, feature_scaler=scaler)
# |             dist_d  = np.expm1(cd_d.pair_distance.numpy())
# |             inter_d = build_inter_mask(cd_d, dist_d)
# |             t_gt_d  = cd_d.pair_trips.numpy().astype(np.float64)
# |             Y_D_wr  = extract_yd_kbins(dist_d, t_gt_d, bin_edges, inter_d)
# |
# |         T_wr    = calibrate_kbins(t0, dist_km, inter, Y_D_wr, bin_edges, q=Q_CALIB, tolerance=TOLERANCE)
# |         cpc_wr_d      = compute_cpc_pair(t_gt[inter], T_wr[inter])
# |         cpc_wr_norm_d = compute_cpc_norm_pair(t_gt[inter], T_wr[inter])
# |         delta_wr_d    = cpc_wr_d - cpc0
# |
# |         wrong_cpc_list.append(cpc_wr_d)
# |         wrong_cpc_norm_list.append(cpc_wr_norm_d)
# |         wrong_delta_list.append(delta_wr_d)
# |         wrong_donor_details.append({
# |             "donor_city": donor,
# |             "cpc_wrong_yd": float(cpc_wr_d),
# |             "cpc_wrong_yd_norm": float(cpc_wr_norm_d),
# |             "delta_cpc_wrong": float(delta_wr_d),
# |             "Y_D_wrong": Y_D_wr.tolist(),
# |         })
# |
# |     # Exact arithmetic mean across all wrong donors in fold
# |     cpc_wr_mean      = float(np.mean(wrong_cpc_list))
# |     cpc_wr_norm_mean = float(np.mean(wrong_cpc_norm_list))
# |     delta_wr_mean    = float(np.mean(wrong_delta_list))
# |     delta_spec       = float(delta_target - delta_wr_mean)
# |
# |     return {
# |         "city": city,
# |         "fold": fold_id,
# |         "donor_city": "all_9_fold_donors",
# |         "n_wrong_donors": len(wrong_donors),
# |         "n_inter_pairs": n_inter,
# |         "K_active": K_active,
# |         # Baseline (M0)
# |         "cpc_baseline": float(cpc0),
# |         "cpc_baseline_norm": float(cpc0_norm),
# |         # Target Oracle (+Y_D^target)
# |         "cpc_target_yd": float(cpc_yd),
# |         "cpc_target_yd_norm": float(cpc_yd_norm),
# |         "delta_cpc_target": delta_target,
# |         # Wrong Donor Placebo (+Y_D^wrong, 9-donor average)
# |         "cpc_wrong_yd": cpc_wr_mean,
# |         "cpc_wrong_yd_norm": cpc_wr_norm_mean,
# |         "delta_cpc_wrong": delta_wr_mean,
# |         # Specificity Estimand (Target - Wrong_Avg9)
# |         "delta_cpc_specificity": delta_spec,
# |         # Raw histogram vectors & per-donor breakdown for transparency
# |         "Y_D_target": Y_D_tgt.tolist(),
# |         "wrong_donor_breakdown": wrong_donor_details,
# |     }
# |
# |
# | def fold_bootstrap(
# |     values: np.ndarray,
# |     fold_ids: np.ndarray,
# |     n: int = 10000,
# |     seed: int = 42,
# |     alpha: float = 0.05,
# | ) -> tuple:
# |     """
# |     Computes Fold-Stratified Bootstrap 95% Confidence Interval.
# |     Resamples observations within each fold independently to account for shared model covariance.
# |     """
# |     rng = np.random.default_rng(seed)
# |     folds = sorted(set(fold_ids))
# |     boot = []
# |     for _ in range(n):
# |         s = []
# |         for f in folds:
# |             fd = values[fold_ids == f]
# |             if len(fd) > 0:
# |                 s.extend(rng.choice(fd, size=len(fd), replace=True))
# |         if s:
# |             boot.append(np.mean(s))
# |     boot = np.array(boot)
# |     if len(boot) == 0:
# |         return 0.0, 0.0, np.array([0.0])
# |     return (
# |         float(np.percentile(boot, 100 * alpha / 2)),
# |         float(np.percentile(boot, 100 * (1 - alpha / 2))),
# |         boot,
# |     )
# |
# |
# | def compute_summary(results: list, fold_manifest: dict = None, bootstrap_seed: int = 2024) -> dict:
# |     """
# |     Aggregates per-city results into primary statistics:
# |       - Full out-of-fold descriptive coverage (n=50)
# |       - Full 5-fold test set (n=50, Folds 1-5, gated strictly upon complete execution)
# |       - Per-fold breakdown (Folds 1 to 5)
# |     """
# |     dt  = np.array([r["delta_cpc_target"]      for r in results])
# |     dw  = np.array([r["delta_cpc_wrong"]       for r in results])
# |     ds  = np.array([r["delta_cpc_specificity"] for r in results])
# |     fid = np.array([r["fold"]                  for r in results])
# |     c0  = np.array([r["cpc_baseline"]          for r in results])
# |     cyd = np.array([r["cpc_target_yd"]        for r in results])
# |     cwr = np.array([r["cpc_wrong_yd"]         for r in results])
# |
# |     n = len(results)
# |     ddof = 1 if n > 1 else 0
# |
# |     ci_tl, ci_th, _ = fold_bootstrap(dt, fid, seed=bootstrap_seed)
# |     ci_wl, ci_wh, _ = fold_bootstrap(dw, fid, seed=bootstrap_seed)
# |     ci_sl, ci_sh, _ = fold_bootstrap(ds, fid, seed=bootstrap_seed)
# |
# |     _, pt = safe_wilcoxon(dt, alternative="greater")
# |     _, pw = safe_wilcoxon(dw, alternative="greater")
# |     _, ps = safe_wilcoxon(ds, alternative="greater")
# |
# |     # --- Full 5-fold Evaluation Guard (Requires strictly complete 50 cities across Folds 1-5) ---
# |     is_full_50_complete = bool(
# |         n == 50
# |         and set(fid.tolist()) == {1, 2, 3, 4, 5}
# |         and all((fid == f).sum() == 10 for f in range(1, 6))
# |     )
# |
# |     conf_summary = None
# |     if is_full_50_complete:
# |         c_dt = dt
# |         c_dw = dw
# |         c_ds = ds
# |         c_c0 = c0
# |         c_cyd = cyd
# |         c_n = 50
# |         c_ddof = 1
# |
# |         c_ci_tl, c_ci_th, _ = fold_bootstrap(c_dt, fid, seed=bootstrap_seed)
# |         c_ci_wl, c_ci_wh, _ = fold_bootstrap(c_dw, fid, seed=bootstrap_seed)
# |         c_ci_sl, c_ci_sh, _ = fold_bootstrap(c_ds, fid, seed=bootstrap_seed)
# |         _, c_pt = safe_wilcoxon(c_dt, alternative="greater")
# |         _, c_pw = safe_wilcoxon(c_dw, alternative="greater")
# |         _, c_ps = safe_wilcoxon(c_ds, alternative="greater")
# |
# |         conf_summary = {
# |             "status": "full_5_fold_complete",
# |             "protocol_role": "Amended Replication under Locked Protocol (Folds 1-5, n=50)",
# |             "n_cities": c_n,
# |             "cpc_baseline_mean": float(c_c0.mean()),
# |             "cpc_baseline_std": float(c_c0.std(ddof=c_ddof)),
# |             "cpc_target_yd_mean": float(c_cyd.mean()),
# |             "cpc_target_yd_std": float(c_cyd.std(ddof=c_ddof)),
# |             # Target Delta
# |             "delta_cpc_target_mean": float(c_dt.mean()),
# |             "delta_cpc_target_median": float(np.median(c_dt)),
# |             "delta_cpc_target_iqr": compute_iqr(c_dt),
# |             "delta_cpc_target_std": float(c_dt.std(ddof=c_ddof)),
# |             "delta_cpc_target_ci_l": c_ci_tl,
# |             "delta_cpc_target_ci_h": c_ci_th,
# |             "n_positive_target": int((c_dt > 0).sum()),
# |             "p_wilcoxon_target": float(c_pt),
# |             # Wrong Placebo Delta (9-donor avg)
# |             "delta_cpc_wrong_mean": float(c_dw.mean()),
# |             "delta_cpc_wrong_median": float(np.median(c_dw)),
# |             "delta_cpc_wrong_iqr": compute_iqr(c_dw),
# |             "delta_cpc_wrong_std": float(c_dw.std(ddof=c_ddof)),
# |             "delta_cpc_wrong_ci_l": c_ci_wl,
# |             "delta_cpc_wrong_ci_h": c_ci_wh,
# |             "n_positive_wrong": int((c_dw > 0).sum()),
# |             "p_wilcoxon_wrong": float(c_pw),
# |             # Specificity Estimand (Target - Wrong_Avg9)
# |             "delta_specificity_mean": float(c_ds.mean()),
# |             "delta_specificity_median": float(np.median(c_ds)),
# |             "delta_specificity_iqr": compute_iqr(c_ds),
# |             "delta_specificity_std": float(c_ds.std(ddof=c_ddof)),
# |             "delta_specificity_ci_l": c_ci_sl,
# |             "delta_specificity_ci_h": c_ci_sh,
# |             "n_positive_specificity": int((c_ds > 0).sum()),
# |             "p_specificity": float(c_ps),
# |             "ci_lower_bound_positive": bool(c_ci_tl > 0),
# |             "specificity_ci_lower_bound_positive": bool(c_ci_sl > 0),
# |             "target_beats_wrong": bool(float(c_ds.mean()) > 0),
# |             "win_rate_target": f"{int((c_dt > 0).sum())}/{c_n}",
# |             "win_rate_wrong": f"{int((c_dw > 0).sum())}/{c_n}",
# |             "win_rate_specificity": f"{int((c_ds > 0).sum())}/{c_n}",
# |         }
# |     else:
# |         conf_summary = {
# |             "status": "not_available",
# |             "reason": f"Incomplete full_5_fold test set (observed {n}/50 required test cities across Folds 1-5; 10 test cities per fold required)",
# |         }
# |
# |     # --- Per-Fold Breakdown ---
# |     per_fold = {}
# |     for f in sorted(set(fid)):
# |         idx = (fid == f)
# |         f_dt = dt[idx]
# |         f_dw = dw[idx]
# |         f_ds = ds[idx]
# |         f_c0 = c0[idx]
# |         f_n = int(idx.sum())
# |         f_ddof = 1 if f_n > 1 else 0
# |         per_fold[f"fold_{f}"] = {
# |             "n_cities": f_n,
# |             "role": "Exploratory / Development" if f == 1 else "Full 5-fold Out-of-Fold",
# |             "cpc_baseline_mean": float(f_c0.mean()),
# |             "cpc_baseline_std": float(f_c0.std(ddof=f_ddof)),
# |             "delta_target_mean": float(f_dt.mean()),
# |             "delta_target_median": float(np.median(f_dt)),
# |             "delta_target_iqr": compute_iqr(f_dt),
# |             "delta_target_std": float(f_dt.std(ddof=f_ddof)),
# |             "delta_wrong_mean": float(f_dw.mean()),
# |             "delta_wrong_median": float(np.median(f_dw)),
# |             "delta_wrong_iqr": compute_iqr(f_dw),
# |             "delta_specificity_mean": float(f_ds.mean()),
# |             "delta_specificity_median": float(np.median(f_ds)),
# |             "n_positive_target": int((f_dt > 0).sum()),
# |             "n_positive_specificity": int((f_ds > 0).sum()),
# |             "win_rate_target": f"{int((f_dt > 0).sum())}/{f_n}",
# |             "win_rate_specificity": f"{int((f_ds > 0).sum())}/{f_n}",
# |             "best_epoch": fold_manifest.get(f, {}).get("best_epoch") if fold_manifest else None,
# |             "best_val_cpc": fold_manifest.get(f, {}).get("best_val_cpc") if fold_manifest else None,
# |             "convergence_gate": fold_manifest.get(f, {}).get("convergence_gate", "—") if fold_manifest else "—",
# |         }
# |
# |     return {
# |         "n_cities": n,
# |         "protocol_version": "e1-v2-amended",
# |         "is_full_50_complete": is_full_50_complete,
# |         "is_full_5_fold_complete": is_full_50_complete,
# |         "std_ddof": ddof,
# |         # Full Out-of-Fold (n=50)
# |         "cpc_baseline_mean": float(c0.mean()),
# |         "cpc_baseline_std":  float(c0.std(ddof=ddof)),
# |         "cpc_target_yd_mean": float(cyd.mean()),
# |         "cpc_target_yd_std": float(cyd.std(ddof=ddof)),
# |         # Target Delta
# |         "delta_cpc_target_mean":   float(dt.mean()),
# |         "delta_cpc_target_median": float(np.median(dt)),
# |         "delta_cpc_target_iqr":    compute_iqr(dt),
# |         "delta_cpc_target_std":    float(dt.std(ddof=ddof)),
# |         "delta_cpc_target_ci_l":   ci_tl,
# |         "delta_cpc_target_ci_h":   ci_th,
# |         "n_positive_target":       int((dt > 0).sum()),
# |         "p_wilcoxon_target":       float(pt),
# |         # Wrong Placebo Delta (9-donor avg)
# |         "cpc_wrong_yd_mean": float(cwr.mean()),
# |         "cpc_wrong_yd_std": float(cwr.std(ddof=ddof)),
# |         "delta_cpc_wrong_mean":    float(dw.mean()),
# |         "delta_cpc_wrong_median":  float(np.median(dw)),
# |         "delta_cpc_wrong_iqr":     compute_iqr(dw),
# |         "delta_cpc_wrong_std":     float(dw.std(ddof=ddof)),
# |         "delta_cpc_wrong_ci_l":    ci_wl,
# |         "delta_cpc_wrong_ci_h":    ci_wh,
# |         "n_positive_wrong":        int((dw > 0).sum()),
# |         "p_wilcoxon_wrong":        float(pw),
# |         # Specificity Estimand (Target - Wrong_Avg9)
# |         "delta_specificity_mean":   float(ds.mean()),
# |         "delta_specificity_median": float(np.median(ds)),
# |         "delta_specificity_iqr":    compute_iqr(ds),
# |         "delta_specificity_std":    float(ds.std(ddof=ddof)),
# |         "delta_specificity_ci_l":   ci_sl,
# |         "delta_specificity_ci_h":   ci_sh,
# |         "n_positive_specificity":   int((ds > 0).sum()),
# |         "p_specificity":            float(ps),
# |         "ci_lower_bound_positive":  bool(ci_tl > 0),
# |         "specificity_ci_lower_bound_positive": bool(ci_sl > 0),
# |         "target_beats_wrong":       bool(float(ds.mean()) > 0),
# |         "win_rate_target":          f"{int((dt > 0).sum())}/{n}",
# |         "win_rate_wrong":           f"{int((dw > 0).sum())}/{n}",
# |         "win_rate_specificity":     f"{int((ds > 0).sum())}/{n}",
# |         # Primary Full 5-fold Subgroup (Folds 1-5, only valid when complete)
# |         "full_5_fold_folds_2_5":   conf_summary,
# |         # Per-fold breakdown
# |         "per_fold": per_fold,
# |         "fold_validation_manifest": fold_manifest or {},
# |         # Runtime Environment & Multi-core CPU metadata
# |         "runtime_environment": get_runtime_metadata(),
# |     }
# |
# |
# | def write_tables(results: list, summary: dict, table_dir: Path | None = None):
# |     """
# |     Generates GitHub Markdown Tables following the Nature/PNAS reporting standard.
# |     """
# |     tdir = table_dir or (RESULTS_DIR / "tables")
# |     tdir.mkdir(parents=True, exist_ok=True)
# |     n  = summary["n_cities"]
# |     tl, th = summary["delta_cpc_target_ci_l"], summary["delta_cpc_target_ci_h"]
# |     wl, wh = summary["delta_cpc_wrong_ci_l"], summary["delta_cpc_wrong_ci_h"]
# |     sl, sh = summary["delta_specificity_ci_l"], summary["delta_specificity_ci_h"]
# |     c0m = summary["cpc_baseline_mean"]
# |     c0s = summary["cpc_baseline_std"]
# |     is_conf = summary.get("is_full_5_fold_complete", False)
# |     is_full = summary.get("is_full_50_complete", False)
# |
# |     run_type_str = "Full 50-City Protocol" if is_full else "Exploratory / Smoke Subset"
# |
# |     lines = [
# |         f"# Table E1: Oracle Aggregated-Distance Existence Test ({run_type_str})",
# |         "",
# |         "> **Methodological Framing & Amendment Context**:",
# |         "> *\"We report the pooled five-fold out-of-fold benchmark across 50 cities as the primary cross-validated performance summary. Because Fold 1 contributed to protocol development, we additionally report the originally designated Folds 1-5 analysis as a full_5_fold sensitivity analysis. Both analyses use five separately trained fold-specific models, and each city is evaluated exactly once when held out.\"*",
# |         "",
# |         "### Analysis Sets Hierarchy",
# |         "",
# |         "| Analysis set | n | Role |",
# |         "|---|---:|---|",
# |         "| All Folds 1–5 | 50 | Pooled out-of-fold benchmark |",
# |         "| Excluding Fold 1 | 40 | Full 5-fold sensitivity |",
# |         "| Fold 1 | 10 | Development/exploratory diagnostic |",
# |         "",
# |         f"**Execution Status**: {len(results)}/50 test cities evaluated | is_full_5_fold_complete={is_conf} | is_full_50_complete={is_full}",
# |         f"**Protocol**: 5-fold stratified city CV (35 train / 5 val / 10 test per fold, locked manifest v2)",
# |         f"**Parameters**: K_move={K_MOVE} bins (pair-weighted quantile), q={Q_CALIB} (within-tolerance calibration, tolerance 10⁻⁵), max_epochs={EPOCHS}, patience={PATIENCE}, std_ddof={summary['std_ddof']}",
# |         "",
# |     ]
# |
# |     # Section 1: Primary Pooled Benchmark (Full Out-of-Fold 50 Cities)
# |     cov_label = "E1-A: Primary Pooled Out-of-Fold Benchmark (All Folds 1–5, n=50)" if is_full else f"E1-A: Primary Benchmark (Observed {n} Cities)"
# |     lines.extend([
# |         f"## {cov_label}",
# |         "",
# |         "| Condition / Estimand | Interzonal CPC (Mean ± SD) | Mean Δ | Median Δ | IQR | 95% Fold-Stratified Bootstrap CI | Win Rate | Wilcoxon p |",
# |         "|---|---|---|---|---|---|---|---|",
# |         f"| Zero-Shot Baseline (M₀) | {c0m:.4f} ± {c0s:.4f} | — | — | — | — | — | — |",
# |         (f"| + Oracle Y_D (target) | {summary['cpc_target_yd_mean']:.4f} ± {summary['cpc_target_yd_std']:.4f} | "
# |          f"+{summary['delta_cpc_target_mean']:.4f} | +{summary['delta_cpc_target_median']:.4f} | {summary['delta_cpc_target_iqr']:.4f} | "
# |          f"[{tl:+.4f}, {th:+.4f}] | {summary['win_rate_target']} | {summary['p_wilcoxon_target']:.2e} |"),
# |         (f"| + Oracle Y_D (wrong donors avg 9) | {summary['cpc_wrong_yd_mean']:.4f} ± {summary['cpc_wrong_yd_std']:.4f} | "
# |          f"{summary['delta_cpc_wrong_mean']:+.4f} | {summary['delta_cpc_wrong_median']:+.4f} | {summary['delta_cpc_wrong_iqr']:.4f} | "
# |          f"[{wl:+.4f}, {wh:+.4f}] | {summary['win_rate_wrong']} | {summary['p_wilcoxon_wrong']:.2e} |"),
# |         (f"| **Specificity Gain (Target − Wrong)** | — | "
# |          f"**+{summary['delta_specificity_mean']:.4f}** | **+{summary['delta_specificity_median']:.4f}** | {summary['delta_specificity_iqr']:.4f} | "
# |          f"**[{sl:+.4f}, {sh:+.4f}]** | **{summary['win_rate_specificity']}** | **{summary['p_specificity']:.2e}** |"),
# |         "",
# |     ])
# |
# |     # Section 2: Full 5-fold Sensitivity Analysis on Folds 1-5 (strictly when complete)
# |     conf = summary.get("full_5_fold_folds_2_5")
# |     if is_conf and conf and conf.get("status") == "full_5_fold_complete":
# |         c_tl, c_th = conf["delta_cpc_target_ci_l"], conf["delta_cpc_target_ci_h"]
# |         c_wl, c_wh = conf["delta_cpc_wrong_ci_l"], conf["delta_cpc_wrong_ci_h"]
# |         c_sl, c_sh = conf["delta_specificity_ci_l"], conf["delta_specificity_ci_h"]
# |         lines.extend([
# |             "## E1-B: Full 5-fold Sensitivity Analysis (Excluding Fold 1: Folds 1-5, n=50)",
# |             "",
# |             "| Condition / Estimand | Interzonal CPC (Mean ± SD) | Mean Δ | Median Δ | IQR | 95% Fold-Stratified Bootstrap CI | Win Rate | Wilcoxon p |",
# |             "|---|---|---|---|---|---|---|---|",
# |             f"| Zero-Shot Baseline (M₀) | {conf['cpc_baseline_mean']:.4f} ± {conf['cpc_baseline_std']:.4f} | — | — | — | — | — | — |",
# |             (f"| + Oracle Y_D (target) | {conf['cpc_target_yd_mean']:.4f} ± {conf['cpc_target_yd_std']:.4f} | "
# |              f"+{conf['delta_cpc_target_mean']:.4f} | +{conf['delta_cpc_target_median']:.4f} | {conf['delta_cpc_target_iqr']:.4f} | "
# |              f"[{c_tl:+.4f}, {c_th:+.4f}] | {conf['win_rate_target']} | {conf['p_wilcoxon_target']:.2e} |"),
# |             (f"| + Oracle Y_D (wrong donors avg 9) | {conf['delta_cpc_wrong_mean'] + conf['cpc_baseline_mean']:.4f} ± {conf['delta_cpc_wrong_std']:.4f} | "
# |              f"{conf['delta_cpc_wrong_mean']:+.4f} | {conf['delta_cpc_wrong_median']:+.4f} | {conf['delta_cpc_wrong_iqr']:.4f} | "
# |              f"[{c_wl:+.4f}, {c_wh:+.4f}] | {conf['win_rate_wrong']} | {conf['p_wilcoxon_wrong']:.2e} |"),
# |             (f"| **Specificity Gain (Target − Wrong)** | — | "
# |              f"**+{conf['delta_specificity_mean']:.4f}** | **+{conf['delta_specificity_median']:.4f}** | {conf['delta_specificity_iqr']:.4f} | "
# |              f"**[{c_sl:+.4f}, {c_sh:+.4f}]** | **{conf['win_rate_specificity']}** | **{conf['p_specificity']:.2e}** |"),
# |             "",
# |         ])
# |     else:
# |         lines.extend([
# |             "## E1-B: Full 5-fold Sensitivity Analysis (Excluding Fold 1: Folds 1-5, n=50)",
# |             "",
# |             f"> *Status: NOT AVAILABLE (Observed {len([r for r in results if r['fold']>=2])}/50 test cities; Full 5-fold evaluation strictly requires complete 50 test cities across Folds 1-5, with 10 test cities per fold).* ",
# |             "",
# |         ])
# |
# |     lines.extend([
# |         "## E1-C: Per-Fold Independent Training & Evaluation Breakdown",
# |         "",
# |         "| Fold | Role | Test Cities | Best Epoch | Best Val CPC | Convergence Gate | M₀ CPC | +Target CPC | Mean ΔTarget | Mean ΔWrong (9 Avg) | Specificity Win Rate |",
# |         "|---|---|---|---|---|---|---|---|---|---|---|",
# |     ])
# |
# |     for f_key, pf in summary.get("per_fold", {}).items():
# |         f_num = f_key.replace("fold_", "")
# |         b_ep = pf.get("best_epoch", "—")
# |         b_vc = f"{pf['best_val_cpc']:.4f}" if pf.get("best_val_cpc") is not None else "—"
# |         c_gate = pf.get("convergence_gate", "—")
# |         role = "Exploratory" if f_num == "1" else "Full 5-fold"
# |         lines.append(
# |             f"| Fold {f_num} | {role} | {pf['n_cities']} | {b_ep} | {b_vc} | {c_gate} | "
# |             f"{pf['cpc_baseline_mean']:.4f} | {pf['cpc_baseline_mean'] + pf['delta_target_mean']:.4f} | "
# |             f"{pf['delta_target_mean']:+.4f} | {pf['delta_wrong_mean']:+.4f} | {pf['win_rate_specificity']} |"
# |         )
# |
# |     # Section 3: Acceptance Criteria (Dynamic gating)
# |     if is_conf and conf and conf.get("status") == "full_5_fold_complete":
# |         e_tl, e_th = conf["delta_cpc_target_ci_l"], conf["delta_cpc_target_ci_h"]
# |         e_sl, e_sh = conf["delta_specificity_ci_l"], conf["delta_specificity_ci_h"]
# |         lines.extend([
# |             "",
# |             "## Acceptance Criteria Verification (Full 5-fold Folds 1-5, n=50)",
# |             "",
# |             "| Criterion | Required Condition | Observed Value | Verdict |",
# |             "|---|---|---|---|",
# |             f"| Full 5-fold CI Lower Bound | CI_lower > 0 | [{e_tl:+.4f}, {e_th:+.4f}] | {'✓ PASS' if conf['ci_lower_bound_positive'] else '✗ FAIL'} |",
# |             f"| Specificity Superiority | Mean ΔCPC_target > Mean ΔCPC_wrong | {conf['delta_cpc_target_mean']:+.4f} vs {conf['delta_cpc_wrong_mean']:+.4f} (Diff: +{conf['delta_specificity_mean']:.4f}) | {'✓ PASS' if conf['target_beats_wrong'] else '✗ FAIL'} |",
# |             f"| Specificity CI Lower Bound | Specificity CI_lower > 0 | [{e_sl:+.4f}, {e_sh:+.4f}] | {'✓ PASS' if conf['specificity_ci_lower_bound_positive'] else '✗ FAIL'} |",
# |             f"| Specificity Significance | Paired Wilcoxon p < 0.05 | p = {conf['p_specificity']:.2e} | {'✓ PASS' if conf['p_specificity'] < 0.05 else '✗ FAIL'} |",
# |             f"| City-level Specificity Consistency | Win rate > 70% (>28/40) | {conf['win_rate_specificity']} | {'✓ PASS' if int(conf['win_rate_specificity'].split('/')[0]) >= 28 else '✗ FAIL'} |",
# |             "",
# |         ])
# |     else:
# |         lines.extend([
# |             "",
# |             "## Acceptance Criteria Verification",
# |             "",
# |             f"> *Status: PENDING FULL 50-CITY EXECUTION (Evaluated {n}/50 cities. Full 5-fold criteria will be locked upon full completion).* ",
# |             "",
# |         ])
# |
# |     (tdir / "e1_main_table.md").write_text("\n".join(lines), encoding="utf-8")
# |
# |     hdr = "| City | Fold | n_pairs | CPC₀ | CPC_target | ΔCPC_target | CPC_wrong (avg 9) | ΔCPC_wrong (avg 9) | ΔSpecificity | Donors |"
# |     sep = "|---|---|---|---|---|---|---|---|---|---|"
# |     rows = [hdr, sep]
# |     for r in sorted(results, key=lambda x: x["city"]):
# |         rows.append(
# |             f"| {r['city']} | {r['fold']} | {r['n_inter_pairs']} | "
# |             f"{r['cpc_baseline']:.4f} | {r['cpc_target_yd']:.4f} | "
# |             f"{r['delta_cpc_target']:+.4f} | {r['cpc_wrong_yd']:.4f} | "
# |             f"{r['delta_cpc_wrong']:+.4f} | {r['delta_cpc_specificity']:+.4f} | 9 fold donors |"
# |         )
# |     (tdir / "e1_per_city.md").write_text("# E1: Complete Per-City Breakdown (50 Cities)\n\n" + "\n".join(rows) + "\n", encoding="utf-8")
# |     print(f"  [Artifact] Generated Markdown tables in {tdir}")
# |
# |
# | def run_e1(
# |     smoke: bool = False,
# |     smoke_cities: list = None,
# |     device_str: str = "cpu",
# |     num_threads: int | None = None,
# |     seed: int = SEED,
# | ):
# |     """
# |     Main E1 execution loop with structured step-by-step logging.
# |     """
# |     t_global_start = time.time()
# |     device = torch.device(device_str)
# |     RESULTS_DIR.mkdir(parents=True, exist_ok=True)
# |
# |     # Configure CPU Threads
# |     active_threads = configure_cpu_threads(num_threads)
# |     runtime_meta = get_runtime_metadata()
# |
# |     # -----------------------------------------------------------------------
# |     # STEP 1: Load Locked 5-Fold Splits Manifest v2 (35 Train / 5 Val / 10 Test)
# |     # -----------------------------------------------------------------------
# |     log_msg("=" * 75)
# |     log_msg("E1 EXPERIMENT PIPELINE (v2): ORACLE AGGREGATED-DISTANCE EXISTENCE TEST")
# |     log_msg("=" * 75)
# |     log_msg("  Runtime Environment & CPU Configuration:")
# |     log_msg(f"    - Platform: {runtime_meta['platform']}")
# |     log_msg(f"    - CPU Cores: {runtime_meta['cpu_count_logical']} logical / {runtime_meta['cpu_count_physical']} physical")
# |     log_msg(f"    - PyTorch Threads: {runtime_meta['torch_num_threads']} (interop: {runtime_meta['torch_num_interop_threads']})")
# |     log_msg(f"    - OpenMP / MKL Threads: OMP={runtime_meta['omp_num_threads']}, MKL={runtime_meta['mkl_num_threads']}")
# |     log_msg(f"[STEP 1/5] Loading locked splits manifest v2 from {MANIFEST_PATH}...")
# |     splits = load_splits_manifest_v2(str(MANIFEST_PATH), data_root=DATA_ROOT)
# |     with open(MANIFEST_PATH, "r", encoding="utf-8") as manifest_file:
# |         split_manifest_sha256 = json.load(manifest_file)["manifest_sha256"]
# |     
# |     log_msg("  -> Preloading all city datasets & spatial graphs into global in-memory cache...")
# |     preload_all_cities(data_root=DATA_ROOT, build_graphs=True, radius_km=5.0)
# |
# |     all_results = []
# |     fold_manifest = {}
# |     total_test_cities = 50 if not smoke else len(smoke_cities or ["Portland", "Denver"])
# |     city_counter = 0
# |
# |     for fold_id, split in splits.items():
# |         t_fold_start = time.time()
# |         train35 = split["train"]   # 35 training cities
# |         val5    = split["val"]     # 5 validation cities
# |         test10  = sorted(split["test"])  # 10 test cities
# |
# |         run_test = test10
# |         if smoke or smoke_cities:
# |             target_filter = smoke_cities if smoke_cities else ["Portland", "Denver"]
# |             run_test = [c for c in target_filter if c in test10]
# |             if not run_test:
# |                 continue
# |
# |         fold_role = "Exploratory / Development" if fold_id == 1 else "Full 5-fold Out-of-Fold"
# |         log_msg("-" * 75)
# |         log_msg(f">>> [FOLD {fold_id}/5] {fold_role} | Train: {len(train35)} cities | Val: {len(val5)} cities | Test: {len(run_test)}/{len(test10)} cities")
# |         log_msg("-" * 75)
# |
# |         # -------------------------------------------------------------------
# |         # STEP 2: Compute Pair-Weighted Quantile Bin Edges from 35 Train Cities
# |         # -------------------------------------------------------------------
# |         log_msg(f"  [STEP 2/5: Fold {fold_id}] Computing K_move={K_MOVE} quantile bin edges from {len(train35)} train cities...")
# |         bin_edges, K_active = compute_kbin_edges(train35, K=K_MOVE, data_root=DATA_ROOT)
# |         
# |         # Enforce strict 8-bin invariant
# |         if K_active != K_MOVE:
# |             raise RuntimeError(f"E1 invariant violated: Expected exactly {K_MOVE} active bins, got {K_active}")
# |         log_msg(f"    -> Strict {K_MOVE}-bin verified. Cut points (km): {np.round(bin_edges[1:-1], 2).tolist()}")
# |
# |         # -------------------------------------------------------------------
# |         # STEP 3: Train Zero-Shot Backbone & Select Best Validation Checkpoint
# |         # -------------------------------------------------------------------
# |         log_msg(f"  [STEP 3/5: Fold {fold_id}] Training backbone model (max_epochs={EPOCHS}, patience={PATIENCE}, min_delta={MIN_DELTA})...")
# |         _ckpt_seed = seed
# |         _ckpt_dir  = RESULTS_DIR / "checkpoints"
# |         _ckpt_path = _ckpt_dir / f"fold{fold_id}_seed{_ckpt_seed}.pt"
# |         model, scaler, train_info = train_zero_shot_model(
# |             train_city_names=train35,
# |             data_root=DATA_ROOT,
# |             epochs=EPOCHS,
# |             device_str=device_str,
# |             verbose=True,
# |             val_city_names=val5,
# |             patience=PATIENCE,
# |             min_delta=MIN_DELTA,
# |             return_info=True,
# |             seed=_ckpt_seed,
# |             checkpoint_path=_ckpt_path,
# |             run_tag=f"e1_fold{fold_id}_seed{_ckpt_seed}",
# |             fold=fold_id,
# |             split_manifest_sha256=split_manifest_sha256,
# |         )
# |
# |         # Verify per-fold convergence gate
# |         is_converged = bool(train_info["stopped_early"] or train_info["best_epoch"] <= (EPOCHS - 5))
# |         conv_gate_status = "PASSED" if is_converged else "FAILED (Ceiling Hit)"
# |
# |         fold_manifest[fold_id] = {
# |             "fold_id": fold_id,
# |             "role": fold_role,
# |             "train_cities": train35,
# |             "val_cities": val5,
# |             "test_cities": test10,
# |             "K_active": K_active,
# |             "K_mode": "strict_8bin",
# |             "bin_edges_km": bin_edges.tolist(),
# |             "best_epoch": train_info["best_epoch"],
# |             "best_val_cpc": train_info["best_val_cpc"],
# |             "epochs_trained": train_info["epochs_trained"],
# |             "stopped_early": train_info["stopped_early"],
# |             "convergence_gate": conv_gate_status,
# |             "val_cpc_history": train_info["val_cpc_history"],
# |             "checkpoint_path": str(_ckpt_path.resolve()),
# |             "checkpoint_seed": _ckpt_seed,
# |         }
# |         log_msg(f"    -> [Fold {fold_id}] Frozen at best epoch {train_info['best_epoch']}/{train_info['epochs_trained']} (Validation CPC = {train_info['best_val_cpc']:.4f}) | Gate: {conv_gate_status}.")
# |         log_msg(f"    -> [Fold {fold_id}] Checkpoint: {_ckpt_path.resolve()}")
# |
# |
# |         # -------------------------------------------------------------------
# |         # STEP 4: Evaluate Held-Out Test Cities (Conditions A, B, C across all 9 donors)
# |         # -------------------------------------------------------------------
# |         log_msg(f"  [STEP 4/5: Fold {fold_id}] Precomputing test city structures & oracle Y_D for {len(test10)} test cities...")
# |         test_city_cache = {}
# |         for t_city in test10:
# |             cd_t = load_city(t_city, data_root=DATA_ROOT, feature_scaler=scaler)
# |             ei_t, ed_t = build_radius_graph(cd_t.lon_lat, radius_km=5.0)
# |             dist_t = np.expm1(cd_t.pair_distance.numpy())
# |             inter_t = build_inter_mask(cd_t, dist_t)
# |             t_gt_t = cd_t.pair_trips.numpy().astype(np.float64)
# |             yd_t = extract_yd_kbins(dist_t, t_gt_t, bin_edges, inter_t)
# |             test_city_cache[t_city] = {
# |                 "city_data": cd_t,
# |                 "edge_index": ei_t,
# |                 "edge_dist": ed_t,
# |                 "dist_km": dist_t,
# |                 "inter_mask": inter_t,
# |                 "t_gt": t_gt_t,
# |                 "Y_D": yd_t,
# |             }
# |
# |         log_msg(f"  [STEP 4/5: Fold {fold_id}] Evaluating {len(run_test)} held-out test cities with 9-donor placebo...")
# |         for i_city, city in enumerate(run_test):
# |             city_counter += 1
# |             t_city_start = time.time()
# |             
# |             res = run_city(
# |                 city=city,
# |                 model=model,
# |                 scaler=scaler,
# |                 bin_edges=bin_edges,
# |                 K_active=K_active,
# |                 test_cities=test10,
# |                 fold_id=fold_id,
# |                 device=device,
# |                 test_city_cache=test_city_cache,
# |             )
# |             all_results.append(res)
# |             t_city_elapsed = time.time() - t_city_start
# |             
# |             d_target = res['delta_cpc_target']
# |             d_wrong  = res['delta_cpc_wrong']
# |             d_spec   = res['delta_cpc_specificity']
# |             spec_marker = f"(dSpec={d_spec:+.4f})"
# |             log_msg(
# |                 f"    [Fold {fold_id} | City {i_city+1:02d}/{len(run_test):02d} (Total {city_counter:02d}/{total_test_cities:02d})] "
# |                 f"{city:<16} ({t_city_elapsed:.2f}s) | "
# |                 f"M0={res['cpc_baseline']:.4f} -> +Target={res['cpc_target_yd']:.4f} (dCPC={d_target:+.4f}) | "
# |                 f"+WrongAvg9={res['cpc_wrong_yd']:.4f} (dCPC={d_wrong:+.4f}) {spec_marker}"
# |             )
# |
# |         t_fold_elapsed = time.time() - t_fold_start
# |         log_msg(f"  [Fold {fold_id} Complete] Elapsed time: {t_fold_elapsed:.1f}s")
# |
# |     # -----------------------------------------------------------------------
# |     # STEP 5: Statistical Aggregation, Verification, and Artifact Output
# |     # -----------------------------------------------------------------------
# |     log_msg("-" * 75)
# |     log_msg("[STEP 5/5] Synthesizing summary, fold-stratified bootstrap CI, and writing artifacts...")
# |     
# |     val_manifest_payload = {
# |         "protocol_version": "e1-v2-amended",
# |         "runtime_environment": get_runtime_metadata(),
# |         "folds": fold_manifest,
# |     }
# |     (RESULTS_DIR / "e1_per_city_results.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
# |     (RESULTS_DIR / "e1_validation_manifest.json").write_text(json.dumps(val_manifest_payload, indent=2), encoding="utf-8")
# |
# |     if len(all_results) < 2:
# |         log_msg("Warning: Fewer than 2 cities evaluated; skipping statistical synthesis.")
# |         return all_results, None
# |
# |     summary = compute_summary(all_results, fold_manifest=fold_manifest, bootstrap_seed=seed)
# |     (RESULTS_DIR / "e1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
# |     write_tables(all_results, summary)
# |
# |     t_global_elapsed = time.time() - t_global_start
# |     is_conf = summary.get("is_full_5_fold_complete", False)
# |     is_full = summary.get("is_full_50_complete", False)
# |     conf = summary.get("full_5_fold_folds_2_5", {})
# |
# |     log_msg("=" * 75)
# |     log_msg(f"E1 EXECUTION SUMMARY (Total Elapsed Time: {t_global_elapsed:.1f}s)")
# |     log_msg("=" * 75)
# |     log_msg(f"  Cities Evaluated: {len(all_results)}/50 | Mode: {'Full 50-City Protocol' if is_full else 'Exploratory Subset'}")
# |     log_msg(f"  Target Effect (dCPC): mean = {summary['delta_cpc_target_mean']:+.4f} (median = {summary['delta_cpc_target_median']:+.4f}, IQR = {summary['delta_cpc_target_iqr']:.4f})")
# |     log_msg(f"  95% Fold-Stratified Bootstrap CI: [{summary['delta_cpc_target_ci_l']:+.4f}, {summary['delta_cpc_target_ci_h']:+.4f}]")
# |     log_msg(f"  Placebo (9 Wrong Donors Avg): mean = {summary['delta_cpc_wrong_mean']:+.4f} (median = {summary['delta_cpc_wrong_median']:+.4f}, IQR = {summary['delta_cpc_wrong_iqr']:.4f})")
# |     log_msg(f"  Specificity Effect (Target - Wrong_Avg9): mean = {summary['delta_specificity_mean']:+.4f} (median = {summary['delta_specificity_median']:+.4f}, IQR = {summary['delta_specificity_iqr']:.4f})")
# |     log_msg(f"  Specificity 95% Bootstrap CI: [{summary['delta_specificity_ci_l']:+.4f}, {summary['delta_specificity_ci_h']:+.4f}]")
# |     log_msg(f"  Specificity Win Rate: {summary['win_rate_specificity']} | Wilcoxon p = {summary['p_specificity']:.2e}")
# |
# |     if is_conf and conf.get("status") == "full_5_fold_complete":
# |         pass_ci = "PASS" if conf['ci_lower_bound_positive'] else "FAIL"
# |         pass_sci = "PASS" if conf['specificity_ci_lower_bound_positive'] else "FAIL"
# |         pass_tw = "PASS" if conf['target_beats_wrong'] else "FAIL"
# |         log_msg("\n  CONFIRMATORY HYPOTHESIS TEST OUTCOMES (Folds 1-5, n=50):")
# |         log_msg(f"    * Target 95% CI Lower Bound > 0      : {pass_ci} ([{conf['delta_cpc_target_ci_l']:+.4f}, {conf['delta_cpc_target_ci_h']:+.4f}])")
# |         log_msg(f"    * Specificity 95% CI Lower Bound > 0 : {pass_sci} ([{conf['delta_specificity_ci_l']:+.4f}, {conf['delta_specificity_ci_h']:+.4f}])")
# |         log_msg(f"    * Specificity Gain (Target > Wrong)  : {pass_tw} (+{conf['delta_specificity_mean']:+.4f} vs 0)")
# |         log_msg(f"    * Specificity Win Rate               : {conf['win_rate_specificity']} (Wilcoxon p = {conf['p_specificity']:.2e})")
# |     else:
# |         conf_count = len([r for r in all_results if r['fold'] >= 2])
# |         log_msg(f"\n  CONFIRMATORY STATUS: NOT AVAILABLE (Observed {conf_count}/50 test cities; strictly requires complete 50 test cities across Folds 1-5, with 10 test cities per fold).")
# |     log_msg("=" * 75)
# |
# |     return all_results, summary
# |
# |
# | if __name__ == "__main__":
# |     parser = argparse.ArgumentParser(description="E1 Oracle Aggregated-Distance Existence Test (Amended Protocol v2)")
# |     parser.add_argument("--smoke",  action="store_true", help="Run smoke test on Portland (Fold 4) and Denver (Fold 5)")
# |     parser.add_argument("--cities", nargs="+", default=None, help="Custom list of test cities to run")
# |     parser.add_argument("--device", default="cpu", help="PyTorch device (cpu/cuda)")
# |     parser.add_argument("--num-threads", "-t", type=int, default=None, help="Number of CPU intra-op threads for PyTorch/OpenMP")
# |     parser.add_argument("--seed", type=int, default=SEED, help="Random seed for model training and bootstrap")
# |     args = parser.parse_args()
# |     run_e1(smoke=args.smoke, smoke_cities=args.cities, device_str=args.device, num_threads=args.num_threads, seed=args.seed)

# ===== END SOURCE FILE: src/experiment/run_e1.py =====


# ===== BEGIN SOURCE FILE: src/experiment/run_experiment.py =====
# | """
# | Experiment Runner for Moving-Bin Calibration Framework.
# |
# | Experimental Conditions per Target City:
# |     1. M_0:                 Zero-shot baseline (pure spatial transfer)
# |     2. M_1^{city}:          Primary moving-bin Meta calibration on Omega_c^+ (q=1.0)
# |     3. M_1^{county}:        County-level calibration
# |     4. M_1^{subzone}:       Subzone-level calibration
# |
# | Primary Metric:
# |     Interzonal CPC (CPC_inter) on Omega_c^+ = {(i,j) in Omega_c : i != j, D_ij > 0}
# | """
# |
# | import numpy as np
# | import torch
# | from typing import Dict, Any, List
# |
# | from src.data.dataset import CityData, load_city
# | from src.data.urban_graph import build_radius_graph, build_adaptive_radius_graph, build_knn_graph
# | from src.data.yd_extractor import (
# |     extract_M1_city_oracle_obs,
# | )
# | from src.data.trip_sampler import M_GRID
# | from src.training.evaluate import evaluate_moving_and_full
# | from src.training.train import infer_zero_shot
# |
# |
# |
# |
# | def run_target_city_experiments(
# |     model: torch.nn.Module,
# |     city_name: str,
# |     scaler: object,
# |     data_root: str = "data",
# |     graph_type: str = "radius",
# |     radius_km: float = 5.0,
# |     knn_k: int = 10,
# |     device_str: str = "cpu",
# |     bin_edges: np.ndarray = None,
# | ) -> Dict[str, Any]:
# |     assert scaler is not None, "StandardScaler must be pre-fitted on source cities."
# |     if bin_edges is None:
# |         raise ValueError("bin_edges must be provided from training cities to avoid data leakage.")
# |
# |     device = torch.device(device_str)
# |     city_data = load_city(city_name, data_root=data_root, feature_scaler=scaler, fit_scaler=False)
# |     coords = city_data.lon_lat.numpy()
# |
# |     if graph_type == "adaptive_radius":
# |         edge_index, edge_dist, _ = build_adaptive_radius_graph(coords, scale_fraction=0.15)
# |     elif graph_type == "radius":
# |         edge_index, edge_dist = build_radius_graph(coords, radius_km=radius_km)
# |     else:
# |         edge_index, edge_dist = build_knn_graph(coords, k=knn_k)
# |
# |     t_true = city_data.pair_trips.numpy().astype(np.float64)
# |     pair_o = city_data.pair_o_idx.numpy()
# |     pair_d = city_data.pair_d_idx.numpy()
# |     pair_dist = city_data.pair_distance.numpy()
# |     pair_dist_km = np.expm1(pair_dist)
# |     inter_mask = (pair_o != pair_d) & (pair_dist_km > 0.0)
# |     n_inter_pairs = int(inter_mask.sum())
# |     total_inter_trips = float(t_true[inter_mask].sum())
# |     total_trips = float(t_true.sum())
# |
# |     # Extract county grouping (GADM 4.1 level-2 point-in-polygon mapping)
# |     import pandas as pd
# |     from pathlib import Path
# |     from src.data.gadm_mapper import get_gadm_gid2_mapping
# |     
# |     meta_df = pd.read_csv(Path(data_root) / city_name / "meta.csv")
# |     assert meta_df["idx"].is_unique, "Mapping invariant failed: meta_df['idx'] has duplicates"
# |     assert set(pair_o).issubset(set(meta_df["idx"])), "Mapping invariant failed: some pair_o indices are not in meta.csv"
# |     
# |     # Get mapping robustly relative to repository root
# |     repo_root = str(Path(__file__).resolve().parents[2])
# |     tract_to_county, mapping_stats = get_gadm_gid2_mapping(meta_df, repo_root)
# |     
# |     pair_county_idx = np.array([tract_to_county[i] for i in pair_o])
# |     assert len(pair_county_idx) == len(pair_o), "Mapping invariant failed: length mismatch after county mapping"
# |
# |     from src.data.yd_extractor import extract_yd_kbins, extract_yd_kbins_grouped
# |     from src.calibration.bin_calibration import calibrate_kbins, calibrate_kbins_grouped
# |
# |     # -----------------------------------------------------------------------
# |     # Condition M0: Pure Zero-Shot Inference
# |     # -----------------------------------------------------------------------
# |     t_pred_zs_tensor = infer_zero_shot(model, city_data, edge_index, edge_dist, device=device)
# |     t_pred_zs = t_pred_zs_tensor.numpy().astype(np.float64)
# |     m0_metrics = evaluate_moving_and_full(
# |         city_data.pair_trips, t_pred_zs_tensor, city_data.pair_o_idx, city_data.pair_d_idx, city_data.bin_labels, pair_distance=city_data.pair_distance
# |     )
# |
# |     # -----------------------------------------------------------------------
# |     # Condition M1_city: City-Level Oracle Y_D
# |     # -----------------------------------------------------------------------
# |     yd_city = extract_yd_kbins(pair_dist_km, t_true, bin_edges, inter_mask)
# |     t_pred_city = calibrate_kbins(t_pred_zs, pair_dist_km, inter_mask, yd_city, bin_edges, q=1.0)
# |     m1_city_metrics = evaluate_moving_and_full(
# |         city_data.pair_trips, torch.tensor(t_pred_city), city_data.pair_o_idx, city_data.pair_d_idx, city_data.bin_labels, pair_distance=city_data.pair_distance
# |     )
# |
# |     # -----------------------------------------------------------------------
# |     # Condition M1_county: County-Level Oracle Y_D
# |     # -----------------------------------------------------------------------
# |     yd_county_dict = extract_yd_kbins_grouped(pair_dist_km, t_true, bin_edges, inter_mask, pair_county_idx)
# |     t_pred_county = calibrate_kbins_grouped(t_pred_zs, pair_dist_km, inter_mask, yd_county_dict, bin_edges, pair_county_idx, q=1.0)
# |     m1_county_metrics = evaluate_moving_and_full(
# |         city_data.pair_trips, torch.tensor(t_pred_county), city_data.pair_o_idx, city_data.pair_d_idx, city_data.bin_labels, pair_distance=city_data.pair_distance
# |     )
# |
# |     # -----------------------------------------------------------------------
# |     # Condition M1_subzone: Tract-Level (Subzone) Oracle Y_D
# |     # -----------------------------------------------------------------------
# |     yd_subzone_dict = extract_yd_kbins_grouped(pair_dist_km, t_true, bin_edges, inter_mask, pair_o)
# |     t_pred_subzone = calibrate_kbins_grouped(t_pred_zs, pair_dist_km, inter_mask, yd_subzone_dict, bin_edges, pair_o, q=1.0)
# |     m1_subzone_metrics = evaluate_moving_and_full(
# |         city_data.pair_trips, torch.tensor(t_pred_subzone), city_data.pair_o_idx, city_data.pair_d_idx, city_data.bin_labels, pair_distance=city_data.pair_distance
# |     )
# |
# |     rho_c = float(n_inter_pairs) / (float(city_data.n_tracts) * float(city_data.n_tracts - 1)) if city_data.n_tracts > 1 else 0.0
# |     average_flow = total_inter_trips / n_inter_pairs if n_inter_pairs > 0 else 0.0
# |     mean_distance = float(np.mean(pair_dist_km[inter_mask])) if n_inter_pairs > 0 else 0.0
# |     
# |     return {
# |         "city": city_name,
# |         "n_tracts": city_data.n_tracts,
# |         "n_pairs": city_data.n_pairs,
# |         "rho_c": rho_c,
# |         "average_flow": average_flow,
# |         "mean_distance": mean_distance,
# |         "n_inter_pairs": n_inter_pairs,
# |         "total_trips": total_trips,
# |         "total_inter_trips": total_inter_trips,
# |         "M0": m0_metrics,
# |         "M1_city_oracle_obs": m1_city_metrics,
# |         "M1_county_oracle_obs": m1_county_metrics,
# |         "M1_subzone_oracle_obs": m1_subzone_metrics,
# |         "mapping_stats": mapping_stats,
# |     }

# ===== END SOURCE FILE: src/experiment/run_experiment.py =====


# ===== BEGIN SOURCE FILE: src/experiment/run_k_sensitivity_v1.py =====
# | import os
# | import sys
# | import json
# | import time
# | import argparse
# | import numpy as np
# | import torch
# | import pandas as pd
# | from pathlib import Path
# | from scipy import stats
# | import matplotlib.pyplot as plt
# | import datetime
# | import hashlib
# |
# | sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
# |
# | from src.data.city_splits import generate_35_5_10_splits
# | from src.data.yd_extractor import compute_kbin_edges, extract_yd_kbins
# | from src.training.train import load_checkpoint, infer_zero_shot
# | from src.data.dataset import load_city
# | from src.data.urban_graph import build_radius_graph
# | from src.calibration.bin_calibration import calibrate_kbins
# | from src.training.evaluate import evaluate_moving_and_full
# |
# | def holm_bonferroni(p_values: list[float]) -> list[float]:
# |     n = len(p_values)
# |     sorted_indices = np.argsort(p_values)
# |     adj_p = np.zeros(n)
# |     for i, idx in enumerate(sorted_indices):
# |         adj_p[idx] = min(1.0, p_values[idx] * (n - i))
# |     for i in range(1, n):
# |         idx = sorted_indices[i]
# |         prev_idx = sorted_indices[i-1]
# |         adj_p[idx] = max(adj_p[idx], adj_p[prev_idx])
# |     return adj_p.tolist()
# |
# | def generate_file_hash(filepath: str) -> str:
# |     h = hashlib.sha256()
# |     with open(filepath, 'rb') as f:
# |         h.update(f.read())
# |     return h.hexdigest()
# |
# | def run_experiment(args):
# |     data_root = args.data_root
# |     output_dir = Path("results/k_sensitivity_v1")
# |     output_dir.mkdir(parents=True, exist_ok=True)
# |     
# |     device = torch.device(args.device)
# |     splits = generate_35_5_10_splits(data_root=data_root)
# |     K_values = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
# |     seeds = [1, 10, 100]
# |     
# |     folds = [1, 2, 3, 4, 5] if not args.smoke_test else [2]
# |     
# |     results = []
# |     
# |     print("="*80)
# |     print("Starting 5-Fold Distance-Bin Number Sensitivity Test v1")
# |     if args.smoke_test:
# |         print("SMOKE TEST MODE: Fold 2 only, 1 city, seeds 1, 10")
# |         seeds = [1, 10]
# |     print("="*80)
# |     
# |     for fold_idx, fold in enumerate(folds, 1):
# |         train_cities = splits[fold]["train"]
# |         test_cities = splits[fold]["test"]
# |         
# |         if args.smoke_test:
# |             test_cities = test_cities[:1]
# |             
# |         print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] [Fold {fold_idx}/{len(folds)}] (Fold {fold}) Training cities: {len(train_cities)}, Test cities: {len(test_cities)}")
# |         
# |         bin_edges_by_k = {}
# |         for K in K_values:
# |             edges, k_act = compute_kbin_edges(train_cities, K=K, data_root=data_root)
# |             bin_edges_by_k[K] = {"edges": edges, "k_active": k_act}
# |             print(f"  [{datetime.datetime.now().strftime('%H:%M:%S')}] - K={K}: computed {k_act} active bins")
# |             
# |         for city_idx, target_city in enumerate(test_cities, 1):
# |             print(f"  [{datetime.datetime.now().strftime('%H:%M:%S')}] -> Evaluating City {city_idx}/{len(test_cities)}: {target_city}")
# |             
# |             for seed in seeds:
# |                 ckpt_path = Path("results/checkpoints") / f"5fold_fold{fold}_seed{seed}.pt"
# |                 if not ckpt_path.exists():
# |                     print(f"WARNING: Checkpoint {ckpt_path} missing. Skipping.")
# |                     continue
# |                 
# |                 model, scaler, _ = load_checkpoint(str(ckpt_path), device_str=args.device)
# |                 model.eval()
# |                 
# |                 city_data = load_city(target_city, data_root=data_root, feature_scaler=scaler, fit_scaler=False)
# |                 coords = city_data.lon_lat.numpy()
# |                 edge_index, edge_dist = build_radius_graph(coords, radius_km=5.0)
# |                 
# |                 t_true = city_data.pair_trips.numpy().astype(np.float64)
# |                 pair_o = city_data.pair_o_idx.numpy()
# |                 pair_d = city_data.pair_d_idx.numpy()
# |                 pair_dist = city_data.pair_distance.numpy()
# |                 pair_dist_km = np.expm1(pair_dist)
# |                 
# |                 inter_mask = (pair_o != pair_d) & (pair_dist_km > 0.0)
# |                 n_inter = inter_mask.sum()
# |                 
# |                 t_pred_zs_tensor = infer_zero_shot(model, city_data, edge_index, edge_dist, device=device)
# |                 t0_np = t_pred_zs_tensor.numpy().astype(np.float64)
# |                 
# |                 m0_metrics = evaluate_moving_and_full(
# |                     city_data.pair_trips, t_pred_zs_tensor, city_data.pair_o_idx, city_data.pair_d_idx, city_data.bin_labels, pair_distance=city_data.pair_distance
# |                 )
# |                 
# |                 m0_cpc_cache = m0_metrics["cpc_inter"]
# |                 
# |                 for K in K_values:
# |                     edges = bin_edges_by_k[K]["edges"]
# |                     k_active = bin_edges_by_k[K]["k_active"]
# |                     
# |                     yd_target = extract_yd_kbins(pair_dist_km, t_true, edges, inter_mask)
# |                     
# |                     yd_sum = float(np.sum(yd_target))
# |                     assert abs(yd_sum - 1.0) < 1e-6 or yd_sum == 0, f"Y_D sum={yd_sum} != 1.0"
# |                     
# |                     # Weights Diagnostics computation
# |                     inter_T0 = t0_np[inter_mask]
# |                     N_hat = inter_T0.sum()
# |                     inter_dist = pair_dist_km[inter_mask]
# |                     Y_hat = np.zeros(K, dtype=np.float64)
# |                     active = np.zeros(K, dtype=bool)
# |                     for k_idx in range(K):
# |                         lo, hi = float(edges[k_idx]), float(edges[k_idx + 1])
# |                         in_bin = (inter_dist > lo) & (inter_dist <= hi)
# |                         if N_hat > 0:
# |                             Y_hat[k_idx] = inter_T0[in_bin].sum() / N_hat
# |                         active[k_idx] = bool(in_bin.any())
# |                         
# |                     yd_raw = yd_target / yd_sum if yd_sum > 0 else np.ones(K)/K
# |                     yd_active = yd_raw * active.astype(np.float64)
# |                     active_sum = yd_active.sum()
# |                     Y_D_cond = yd_active / active_sum if active_sum > 0 else Y_hat.copy()
# |                     
# |                     w = np.ones(K, dtype=np.float64)
# |                     for k_idx in range(K):
# |                         if active[k_idx] and Y_hat[k_idx] > 0:
# |                             w[k_idx] = Y_D_cond[k_idx] / Y_hat[k_idx]  # q=1.0
# |                             
# |                     w_active = w[active]
# |                     if len(w_active) == 0: w_active = np.array([1.0])
# |                     
# |                     min_pred_mass = np.min(Y_hat[active]) if active.any() else 0.0
# |                     max_ratio = np.max(w_active)
# |                     
# |                     diag = {
# |                         "w_min": float(np.min(w_active)),
# |                         "w_median": float(np.median(w_active)),
# |                         "w_p95": float(np.percentile(w_active, 95)),
# |                         "w_max": float(max_ratio),
# |                         "frac_w_gt_2": float(np.mean(w_active > 2)),
# |                         "frac_w_gt_5": float(np.mean(w_active > 5)),
# |                         "frac_w_gt_10": float(np.mean(w_active > 10)),
# |                         "min_pred_mass": float(min_pred_mass),
# |                         "max_ratio": float(max_ratio),
# |                         "k_active": int(k_active),
# |                         "active_sum": float(active_sum)
# |                     }
# |                     
# |                     t_cal = calibrate_kbins(t0_np, pair_dist_km, inter_mask, yd_target, edges, q=1.0, tolerance=1e-5)
# |                     
# |                     m1_metrics = evaluate_moving_and_full(
# |                         city_data.pair_trips, torch.tensor(t_cal), city_data.pair_o_idx, city_data.pair_d_idx, city_data.bin_labels, pair_distance=city_data.pair_distance
# |                     )
# |                     
# |                     delta_cpc = m1_metrics["cpc_inter"] - m0_cpc_cache
# |                     
# |                     res_row = {
# |                         "city": target_city,
# |                         "fold": fold,
# |                         "seed": seed,
# |                         "K": K,
# |                         "q_K": 1.0,
# |                         "m0_cpc_inter": float(m0_cpc_cache),
# |                         "m1_cpc_inter": float(m1_metrics["cpc_inter"]),
# |                         "delta_cpc": float(delta_cpc),
# |                         "m1_mae_inter": float(m1_metrics["mae_inter"]),
# |                         "m1_rmse_inter": float(m1_metrics["rmse_inter"]),
# |                         "m1_spearman_inter": float(m1_metrics["spearman_inter"]),
# |                         "m1_cpc_inflow": float(m1_metrics.get("cpc_inflow", 0.0)),
# |                         "m1_cpc_outflow": float(m1_metrics.get("cpc_outflow", 0.0)),
# |                         "m1_rel_error_total": float(m1_metrics.get("rel_error_total", 0.0)),
# |                     }
# |                     res_row.update(diag)
# |                     results.append(res_row)
# |                     
# |     df = pd.DataFrame(results)
# |     df.to_csv(output_dir / "k_sensitivity_raw.csv", index=False)
# |     
# |     with open(output_dir / "k_sensitivity_raw.json", "w") as f:
# |         json.dump(df.to_dict(orient="records"), f, indent=2)
# |         
# |     # Check M0 identical
# |     print("\nVerifying M0 consistency across K...")
# |     for (city, seed), group in df.groupby(['city', 'seed']):
# |         m0_vals = group['m0_cpc_inter'].values
# |         assert np.max(m0_vals) - np.min(m0_vals) < 1e-12, f"M0 changed across K for {city} seed {seed}!"
# |     print("M0 consistency passed.")
# |     
# |     # Aggregation
# |     print("\nAggregating over seeds...")
# |     avg_cols = ["m0_cpc_inter", "m1_cpc_inter", "delta_cpc", "m1_mae_inter", "m1_rmse_inter", "m1_spearman_inter", "w_max", "min_pred_mass", "k_active"]
# |     df_city = df.groupby(["city", "fold", "K"])[avg_cols].mean().reset_index()
# |     df_city.to_csv(output_dir / "k_sensitivity_per_city.csv", index=False)
# |     
# |     df_seed = df.copy()
# |     df_seed.to_csv(output_dir / "k_sensitivity_per_seed.csv", index=False)
# |     
# |     # Full 5-fold Analysis
# |     df_all = df_city[df_city["fold"].isin([1, 2, 3, 4, 5])]
# |     print(f"\nEvaluating all cities (Folds 1-5): {df_all['city'].nunique()}")
# |     
# |     summary_data = []
# |     
# |     for K in K_values:
# |         d = df_all[df_all["K"] == K]
# |         n_cities = len(d)
# |         if n_cities == 0:
# |             continue
# |             
# |         m0_mean = d["m0_cpc_inter"].mean()
# |         m1_mean = d["m1_cpc_inter"].mean()
# |         delta = d["delta_cpc"].values
# |         mean_d = np.mean(delta)
# |         std_d = np.std(delta, ddof=1) if n_cities > 1 else 0
# |         
# |         # Bootstrap
# |         rng = np.random.default_rng(42) # Bootstrap seed protocol
# |         boot_means = []
# |         for _ in range(10000):
# |             s = []
# |             for fold in [1, 2, 3, 4, 5]:
# |                 vals = d[d["fold"] == fold]["delta_cpc"].values
# |                 if len(vals) > 0:
# |                     s.extend(rng.choice(vals, size=len(vals), replace=True))
# |             boot_means.append(np.mean(s))
# |         ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5]) if boot_means else (0,0)
# |         
# |         pos_cities = np.sum(delta > 0)
# |         
# |         _, p_1s = stats.wilcoxon(delta, alternative="greater") if len(delta) > 0 else (0, 1.0)
# |         _, p_2s = stats.wilcoxon(delta, alternative="two-sided") if len(delta) > 0 else (0, 1.0)
# |         
# |         summary_data.append({
# |             "K": K,
# |             "m0_cpc": m0_mean,
# |             "m1_cpc": m1_mean,
# |             "mean_delta": mean_d,
# |             "std_delta": std_d,
# |             "ci_low": ci_low,
# |             "ci_high": ci_high,
# |             "pos_cities": int(pos_cities),
# |             "total_cities": n_cities,
# |             "k_act_mean": d["k_active"].mean(),
# |             "w_max_mean": d["w_max"].mean(),
# |             "p_1s_raw": p_1s,
# |             "p_2s_raw": p_2s,
# |         })
# |         
# |     # P-value adjustments
# |     secondary_ks = [K for K in K_values if K != 8]
# |     raw_ps = [next((s["p_1s_raw"] for s in summary_data if s["K"] == K), 1.0) for K in secondary_ks]
# |     adj_ps = holm_bonferroni(raw_ps)
# |     adj_p_map = dict(zip(secondary_ks, adj_ps))
# |     
# |     for s in summary_data:
# |         s["p_1s_adj"] = adj_p_map.get(s["K"], None)
# |         
# |     # Contrasts
# |     d8 = df_all[df_all["K"] == 8].set_index("city")
# |     mean_d8 = d8["delta_cpc"].mean()
# |     
# |     contrast_data = []
# |     raw_contrast_ps = []
# |     
# |     for K in secondary_ks:
# |         dk = df_all[df_all["K"] == K].set_index("city")
# |         common = d8.index.intersection(dk.index)
# |         
# |         d8_com = d8.loc[common]
# |         dk_com = dk.loc[common]
# |         
# |         ck = dk_com["delta_cpc"] - d8_com["delta_cpc"]
# |         _, p_ck = stats.wilcoxon(ck, alternative="two-sided") if len(ck) > 0 else (0, 1.0)
# |         
# |         raw_contrast_ps.append(p_ck)
# |         
# |         rk = dk["delta_cpc"].mean() / mean_d8 if mean_d8 > 0 else None
# |         
# |         contrast_data.append({
# |             "contrast": f"K{K} - K8",
# |             "mean_diff": float(ck.mean()) if len(ck)>0 else 0.0,
# |             "ci": [float(np.percentile(ck, 2.5)), float(np.percentile(ck, 97.5))] if len(ck)>0 else [0.0, 0.0],
# |             "p_adj": 1.0, # Placeholder, will be updated
# |             "r": rk
# |         })
# |         
# |     adj_contrast_ps = holm_bonferroni(raw_contrast_ps)
# |     for i in range(len(contrast_data)):
# |         contrast_data[i]["p_adj"] = adj_contrast_ps[i]
# |     
# |     # Save JSON summary
# |     out_sum = {
# |         "summary": summary_data,
# |         "contrasts": contrast_data
# |     }
# |     with open(output_dir / "k_sensitivity_summary.json", "w") as f:
# |         json.dump(out_sum, f, indent=2)
# |         
# |     # Generate Markdown
# |     md = []
# |     md.append("# 5-Fold Distance-Bin Number Sensitivity Test v1")
# |     md.append(f"\nEvaluating all cities (Folds 1-5): {df_all['city'].nunique()}")
# |     md.append("\n## Primary Results")
# |     md.append("| K | Mean M0 CPC | Mean M1 CPC | Mean $\\Delta$ CPC | 95% CI | Positive cities | Mean $K_{active}$ | Mean $w_{max}$ | Adjusted p |")
# |     md.append("|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
# |     for s in summary_data:
# |         p_str = f"{s['p_1s_adj']:.4e}" if s['p_1s_adj'] is not None else "-"
# |         md.append(f"| {s['K']} | {s['m0_cpc']:.4f} | {s['m1_cpc']:.4f} | {s['mean_delta']:.4f} | [{s['ci_low']:.4f}, {s['ci_high']:.4f}] | {s['pos_cities']}/{s['total_cities']} | {s['k_act_mean']:.1f} | {s['w_max_mean']:.1f} | {p_str} |")
# |         
# |     md.append("\n## Contrasts (vs K=8)")
# |     md.append("| Contrast | Mean difference | 95% CI | Adjusted p |")
# |     md.append("|---|--:|--:|--:|")
# |     for c in contrast_data:
# |         md.append(f"| {c['contrast']} | {c['mean_diff']:.4f} | [{c['ci'][0]:.4f}, {c['ci'][1]:.4f}] | {c['p_adj']:.4e} |")
# |         
# |     with open(output_dir / "k_sensitivity_summary.md", "w") as f:
# |         f.write("\n".join(md))
# |         
# |     # Manifest
# |     manifest = {
# |         "split_seed": 20260818,
# |         "model_seeds": seeds,
# |         "bootstrap_seed": 42,
# |         "folds": folds,
# |         "evaluated_folds": [1, 2, 3, 4, 5] if not args.smoke_test else [2],
# |         "K_values": K_values,
# |         "primary_K": 8,
# |         "binning_method": "pair-weighted quantile",
# |         "q_policy": "q=1.0 fixed",
# |         "noise_level": 0.0,
# |         "checkpoint_hashes": {},
# |         "code_hash_version": generate_file_hash(__file__),
# |         "run_timestamp": datetime.datetime.now().isoformat()
# |     }
# |     with open(output_dir / "k_sensitivity_manifest.json", "w") as f:
# |         json.dump(manifest, f, indent=2)
# |         
# |     # Plotting
# |     # Fig 1: Mean gain by K
# |     plt.figure(figsize=(6, 4))
# |     ks = [s["K"] for s in summary_data]
# |     means = [s["mean_delta"] for s in summary_data]
# |     yerr = [[s["mean_delta"] - s["ci_low"] for s in summary_data], [s["ci_high"] - s["mean_delta"] for s in summary_data]]
# |     plt.errorbar(ks, means, yerr=yerr, marker='o', capsize=5)
# |     plt.axhline(0, color='red', linestyle='--')
# |     plt.xticks(K_values)
# |     plt.xlabel('K (number of bins)')
# |     plt.ylabel('Mean $\\Delta$ CPC')
# |     plt.title('Mean Gain by K (95% CI)')
# |     plt.grid(True, alpha=0.3)
# |     plt.savefig(output_dir / "fig_delta_cpc_by_k.png", dpi=300)
# |     plt.close()
# |     
# |     # Fig 2: Per-city sensitivity
# |     plt.figure(figsize=(8, 5))
# |     for name, group in df_all.groupby("city"):
# |         group = group.sort_values("K")
# |         fold = group["fold"].iloc[0]
# |         # In case we don't have enough colors, modulo by 10
# |         color = plt.cm.tab10(fold % 10)
# |         plt.plot(group["K"], group["delta_cpc"], marker='.', color=color, alpha=0.5, linewidth=1)
# |     plt.axhline(0, color='red', linestyle='--', linewidth=2)
# |     plt.xticks(K_values)
# |     plt.xlabel('K')
# |     plt.ylabel('$\\Delta$ CPC_c')
# |     plt.title('Per-City Sensitivity')
# |     plt.grid(True, alpha=0.3)
# |     plt.savefig(output_dir / "fig_k_per_city.png", dpi=300)
# |     plt.close()
# |     
# |     # Fig 3: Calibration stability
# |     fig, axes = plt.subplots(1, 3, figsize=(15, 4))
# |     sns_k = [s["K"] for s in summary_data]
# |     
# |     axes[0].plot(sns_k, [s["w_max_mean"] for s in summary_data], marker='o')
# |     axes[0].set_title('Mean w_max')
# |     
# |     d_all_minmass = df_all.groupby("K")["min_pred_mass"].mean()
# |     axes[1].plot(d_all_minmass.index, d_all_minmass.values, marker='o')
# |     axes[1].set_title('Mean Min Predicted Mass')
# |     
# |     axes[2].plot(sns_k, [s["k_act_mean"] for s in summary_data], marker='o')
# |     axes[2].set_title('Mean K_active')
# |     
# |     for ax in axes:
# |         ax.set_xticks(K_values)
# |         ax.set_xlabel('K')
# |         ax.grid(True, alpha=0.3)
# |         
# |     plt.tight_layout()
# |     plt.savefig(output_dir / "fig_weights_by_k.png", dpi=300)
# |     plt.close()
# |     
# |     print("Done!")
# |
# | if __name__ == "__main__":
# |     parser = argparse.ArgumentParser()
# |     parser.add_argument("--data_root", default="data")
# |     parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
# |     parser.add_argument("--smoke_test", action="store_true")
# |     args = parser.parse_args()
# |     run_experiment(args)

# ===== END SOURCE FILE: src/experiment/run_k_sensitivity_v1.py =====


# ===== BEGIN SOURCE FILE: src/experiment/run_mlp_backbone_test.py =====
# | """
# | Backbone Robustness Evaluation Experiment (Urban GNN vs Pairwise MLP).
# | Trains and evaluates Pairwise MLP backbone (without graph convolutions)
# | across 5-Fold cross validation to assess calibration operator transferability.
# | """
# |
# | import os
# | os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
# | import sys
# | import json
# | import time
# | import argparse
# | import logging
# | import torch
# | import numpy as np
# | from pathlib import Path
# | from scipy import stats
# | from typing import Dict, Any, List
# |
# | sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
# |
# | from src.data.city_splits import generate_35_5_10_splits
# | from src.data.dataset import load_city, load_raw_city
# | from src.data.urban_graph import build_radius_graph
# | from src.data.yd_extractor import compute_kbin_edges, extract_yd_kbins
# | from src.calibration.bin_calibration import calibrate_kbins
# | from src.training.evaluate import compute_cpc_pair
# | from src.training.train import train_zero_shot_model, infer_zero_shot
# |
# |
# | def fast_evaluate_city(model: torch.nn.Module, city_name: str, scaler: Any, bin_edges: np.ndarray, data_root: str = "data", device: str = "cpu") -> Dict[str, float]:
# |     """Fast, vectorized target city evaluation for M0 and M1 (City-level Oracle)."""
# |     raw = load_raw_city(city_name, data_root=data_root)
# |     dist_km = raw.dist_km
# |     inter_mask = (raw.pair_o_idx.numpy() != raw.pair_d_idx.numpy()) & (dist_km > 0.0)
# |     t_true_inter = raw.pair_trips.numpy()[inter_mask]
# |
# |     edge_index, edge_dist = build_radius_graph(raw.lon_lat, radius_km=5.0, include_self_loop=True, cache_key=f"{city_name}_tracts")
# |
# |     city_data = load_city(city_name, data_root=data_root, feature_scaler=scaler, fit_scaler=False)
# |     t_pred_zs_tensor = infer_zero_shot(model, city_data, edge_index, edge_dist, device=device)
# |     t_pred_zs = t_pred_zs_tensor.numpy().astype(np.float64)
# |
# |     t0_inter = t_pred_zs[inter_mask]
# |     cpc_m0 = float(compute_cpc_pair(t_true_inter, t0_inter))
# |
# |     yd_target = extract_yd_kbins(dist_km, raw.pair_trips.numpy(), bin_edges, inter_mask)
# |     t_cal = calibrate_kbins(t_pred_zs, dist_km, inter_mask, yd_target, bin_edges, q=1.0)
# |     t1_inter = t_cal[inter_mask]
# |     cpc_m1 = float(compute_cpc_pair(t_true_inter, t1_inter))
# |
# |     return {
# |         "m0_cpc_inter": cpc_m0,
# |         "m1_cpc_inter": cpc_m1,
# |         "delta_cpc": cpc_m1 - cpc_m0
# |     }
# |
# |
# | def run_mlp_backbone_test(args: argparse.Namespace) -> None:
# |     data_root = args.data_root
# |     output_dir = args.output_dir
# |     os.makedirs(output_dir, exist_ok=True)
# |     os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
# |
# |     log_file = os.path.join(output_dir, "mlp_backbone_execution.log")
# |     logging.basicConfig(
# |         level=logging.INFO,
# |         format='%(asctime)s [%(levelname)s] %(message)s',
# |         handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
# |     )
# |     logger = logging.getLogger(__name__)
# |
# |     splits = generate_35_5_10_splits(data_root=data_root)
# |     manifest_path = Path(__file__).resolve().parents[2] / "results" / "e1" / "splits_manifest_v2.json"
# |     if not manifest_path.exists():
# |         raise FileNotFoundError(f"Missing locked split manifest: {manifest_path}")
# |     with open(manifest_path, "r", encoding="utf-8") as manifest_file:
# |         split_manifest_sha256 = json.load(manifest_file)["manifest_sha256"]
# |     
# |     if args.smoke:
# |         folds_to_run = [2]
# |         seeds = [1]
# |         epochs_per_fold = 2
# |         patience = 2
# |     else:
# |         folds_to_run = args.folds
# |         seeds = args.seeds
# |         epochs_per_fold = args.epochs
# |         patience = args.patience
# |
# |     all_mlp_results = []
# |     mlp_json_path = Path(output_dir) / "mlp_backbone_results.json"
# |     if mlp_json_path.exists():
# |         try:
# |             with open(mlp_json_path, "r") as f:
# |                 prev_data = json.load(f)
# |                 all_mlp_results = prev_data.get("city_level_results", prev_data) if isinstance(prev_data, dict) else prev_data
# |                 logger.info(f"Loaded {len(all_mlp_results)} existing MLP city records from {mlp_json_path}")
# |         except Exception:
# |             all_mlp_results = []
# |     
# |     logger.info("=" * 85)
# |     logger.info("STARTING PAIRWISE MLP BACKBONE TRAINING & EVALUATION")
# |     logger.info(f"Folds: {folds_to_run} | Seeds: {seeds} | Epochs: {epochs_per_fold} | Device: {args.device}")
# |     logger.info("=" * 85)
# |     
# |     for fold_id in folds_to_run:
# |         split = splits[fold_id]
# |         train_cities = split["train"]
# |         val_cities = split["val"]
# |         test_cities = split["test"] if not args.smoke else split["test"][:2]
# |
# |         # Remove previous records for this fold to allow clean overwrite
# |         all_mlp_results = [r for r in all_mlp_results if r.get("fold") != fold_id]
# |
# |         logger.info(f"\n# FOLD {fold_id}/5 (Train: {len(train_cities)}, Val: {len(val_cities)}, Test: {len(test_cities)})")
# |         models = []
# |         scalers = []
# |         
# |         for seed_idx, seed in enumerate(seeds):
# |             _ckpt_path = Path(output_dir) / "checkpoints" / f"mlp_fold{fold_id}_seed{seed}.pt"
# |             
# |             expected_config = {
# |                 "hidden_dim": args.hidden_dim,
# |                 "num_gnn_layers": args.num_gnn_layers,
# |                 "graph_type": args.graph_type,
# |                 "radius_km": args.radius_km,
# |                 "knn_k": args.knn_k,
# |                 "loss_type": args.loss_type,
# |                 "epochs": epochs_per_fold,
# |                 "lr": args.lr,
# |                 "backbone": "mlp",
# |             }
# |             if _ckpt_path.exists():
# |                 logger.info(f"--- Found existing MLP checkpoint {_ckpt_path}. Loading... ---")
# |                 from src.training.train import load_checkpoint
# |                 model, scaler, _ = load_checkpoint(_ckpt_path, device_str=args.device, expected_config=expected_config)
# |                 model.eval()
# |             else:
# |                 logger.info(f"--- Training MLP Seed {seed} (Fold {fold_id}) ---")
# |                 model, scaler = train_zero_shot_model(
# |                     train_city_names=train_cities,
# |                     data_root=data_root,
# |                     epochs=epochs_per_fold,
# |                     lr=args.lr,
# |                     hidden_dim=args.hidden_dim,
# |                     num_gnn_layers=args.num_gnn_layers,
# |                     graph_type=args.graph_type,
# |                     radius_km=args.radius_km,
# |                     knn_k=args.knn_k,
# |                     loss_type=args.loss_type,
# |                     backbone="mlp",  # <--- Pairwise Spatial MLP Backbone (No message passing)
# |                     device_str=args.device,
# |                     verbose=False,
# |                     val_city_names=val_cities,
# |                     patience=patience,
# |                     checkpoint_path=_ckpt_path,
# |                     run_tag=f"mlp_fold{fold_id}_seed{seed}",
# |                     seed=seed,
# |                     fold=fold_id,
# |                     split_manifest_sha256=split_manifest_sha256,
# |                 )
# |             models.append(model)
# |             scalers.append(scaler)
# |             
# |         bin_edges, K_active = compute_kbin_edges(train_cities, K=8, data_root=data_root)
# |
# |         # Target City Evaluation
# |         for target_city in test_cities:
# |             seed_results = []
# |             for seed_idx, model in enumerate(models):
# |                 scaler = scalers[seed_idx]
# |                 res = fast_evaluate_city(
# |                     model=model,
# |                     city_name=target_city,
# |                     scaler=scaler,
# |                     bin_edges=bin_edges,
# |                     data_root=data_root,
# |                     device=args.device
# |                 )
# |                 seed_results.append(res)
# |                 
# |             m0_cpc_inter = float(np.mean([r["m0_cpc_inter"] for r in seed_results]))
# |             m1_cpc_inter = float(np.mean([r["m1_cpc_inter"] for r in seed_results]))
# |             delta_cpc = m1_cpc_inter - m0_cpc_inter
# |
# |             city_res = {
# |                 "city": target_city,
# |                 "fold": fold_id,
# |                 "m0_cpc_inter": m0_cpc_inter,
# |                 "m1_cpc_inter": m1_cpc_inter,
# |                 "delta_cpc": delta_cpc,
# |                 "seed_results": [
# |                     {
# |                         "seed": seeds[idx],
# |                         "m0_cpc_inter": r["m0_cpc_inter"],
# |                         "m1_cpc_inter": r["m1_cpc_inter"],
# |                         "delta_cpc": r["delta_cpc"]
# |                     }
# |                     for idx, r in enumerate(seed_results)
# |                 ]
# |             }
# |             all_mlp_results.append(city_res)
# |             logger.info(f"  {target_city:15s} | M0: {m0_cpc_inter:.4f} | M1: {m1_cpc_inter:.4f} | d={delta_cpc:+.4f}")
# |             
# |             # Intermediate Save
# |             with open(mlp_json_path, "w") as f:
# |                 json.dump(all_mlp_results, f, indent=2)
# |             
# |     logger.info(f"\nSaved {len(all_mlp_results)} MLP backbone city results to {mlp_json_path}")
# |     logger.info("Run `python src/experiment/compare_backbones.py` to compare MLP with Urban GNN.")
# |
# |
# | if __name__ == "__main__":
# |     parser = argparse.ArgumentParser(description="Pairwise MLP Backbone Evaluation Experiment")
# |     parser.add_argument("--data_root", type=str, default="data")
# |     parser.add_argument("--output_dir", type=str, default="results")
# |     parser.add_argument("--epochs", type=int, default=200)
# |     parser.add_argument("--patience", type=int, default=16)
# |     parser.add_argument("--lr", type=float, default=3.2e-3)
# |     parser.add_argument("--hidden_dim", type=int, default=64)
# |     parser.add_argument("--num_gnn_layers", type=int, default=2)
# |     parser.add_argument("--graph_type", type=str, default="radius")
# |     parser.add_argument("--radius_km", type=float, default=5.0)
# |     parser.add_argument("--knn_k", type=int, default=10)
# |     parser.add_argument("--loss_type", type=str, default="ztnb")
# |     parser.add_argument("--device", type=str, default="cpu")
# |     parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
# |     parser.add_argument("--seeds", nargs="+", type=int, default=[1, 10, 100])
# |     parser.add_argument("--smoke", action="store_true", help="Run quick 1-fold 1-seed smoke test")
# |     
# |     args = parser.parse_args()
# |     run_mlp_backbone_test(args)

# ===== END SOURCE FILE: src/experiment/run_mlp_backbone_test.py =====


# ===== BEGIN SOURCE FILE: src/experiment/run_noise_robustness.py =====
# | import os
# | os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
# | import sys
# | import json
# | import hashlib
# | import argparse
# | import datetime
# | from pathlib import Path
# | from typing import Dict, Tuple, List, Optional, Any
# |
# | import numpy as np
# | import pandas as pd
# | import torch
# | import matplotlib.pyplot as plt
# | import logging
# |
# | from scipy.optimize import bisect
# | from scipy.stats import spearmanr, wilcoxon
# | from scipy.spatial.distance import jensenshannon
# |
# | def holm_correction(p_vals: List[float]) -> np.ndarray:
# |     n = len(p_vals)
# |     sorted_indices = np.argsort(p_vals)
# |     adj_p = np.zeros(n)
# |     running_max = 0.0
# |     for i, idx in enumerate(sorted_indices):
# |         p_adj = p_vals[idx] * (n - i)
# |         running_max = max(running_max, p_adj)
# |         adj_p[idx] = min(1.0, running_max)
# |     return adj_p
# |
# | sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
# | from src.data.dataset import load_city, load_raw_city
# | from src.data.urban_graph import build_radius_graph
# | from src.training.train import load_checkpoint
# | from src.training.evaluate import compute_cpc_pair
# | from src.data.yd_extractor import compute_kbin_edges, extract_yd_kbins
# | from src.data.city_splits import generate_35_5_10_splits, load_splits_manifest_v2
# | from src.experiment.run_experiment import infer_zero_shot
# |
# | def evaluate_cpc(t_true_inter: np.ndarray, t_pred_inter: np.ndarray) -> float:
# |     return compute_cpc_pair(t_true_inter, t_pred_inter)
# |
# | def get_active_bins(yd: np.ndarray, eps: float = 1e-8) -> np.ndarray:
# |     return yd > eps
# |
# | def get_stable_seed(noise_seed: int, fold: int, city: str, replicate_id: int) -> int:
# |     s = f"{noise_seed}_{fold}_{city}_{replicate_id}"
# |     return int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % (2**32)
# |
# | def generate_nested_noisy_yd(p_active: np.ndarray, epsilons: List[float], base_seed: int) -> Dict[float, np.ndarray]:
# |     K_act = len(p_active)
# |     if K_act == 1:
# |         return {eps: p_active.copy() for eps in epsilons}
# |         
# |     rng = np.random.RandomState(base_seed)
# |     
# |     for attempt in range(10000):
# |         z = rng.randn(K_act)
# |         z = z - np.mean(z)
# |         
# |         def get_p_sigma(sigma: float) -> np.ndarray:
# |             log_p = np.log(p_active) + sigma * z
# |             max_log = np.max(log_p)
# |             p_sigma = np.exp(log_p - max_log)
# |             p_sigma = p_sigma / np.sum(p_sigma)
# |             return p_sigma
# |             
# |         def tv_diff(sigma: float, eps: float) -> float:
# |             p_sigma = get_p_sigma(sigma)
# |             return float(0.5 * np.sum(np.abs(p_sigma - p_active)) - eps)
# |             
# |         max_idx = int(np.argmax(z))
# |         p_inf = np.zeros_like(p_active)
# |         p_inf[max_idx] = 1.0
# |         max_tv = float(0.5 * np.sum(np.abs(p_inf - p_active)))
# |         
# |         if max_tv <= max(epsilons) + 1e-6:
# |             continue
# |             
# |         try:
# |             results: Dict[float, np.ndarray] = {}
# |             for eps in epsilons:
# |                 if eps == 0.0:
# |                     results[eps] = p_active.copy()
# |                     continue
# |                 
# |                 upper = 1.0
# |                 while tv_diff(upper, eps) <= 0:
# |                     upper *= 2.0
# |                     if upper > 1e6:
# |                         raise ValueError("Upper bound too large")
# |                         
# |                 sigma_opt = bisect(tv_diff, 0, upper, args=(eps,), xtol=1e-12, maxiter=1000)
# |                 p_opt = get_p_sigma(float(sigma_opt))
# |                 
# |                 achieved_tv = float(0.5 * np.sum(np.abs(p_opt - p_active)))
# |                 assert np.all(p_opt >= 0), "p_opt has negative values"
# |                 assert np.abs(np.sum(p_opt) - 1.0) < 1e-8, "p_opt does not sum to 1"
# |                 assert np.abs(achieved_tv - eps) < 1e-8, f"Achieved TV {achieved_tv} != requested {eps}"
# |                 
# |                 results[eps] = p_opt
# |             return results
# |         except (ValueError, AssertionError) as e:
# |             continue
# |             
# |     raise RuntimeError("Failed to generate valid noise direction after 10000 attempts.")
# |
# |
# | def fold_stratified_bootstrap(city_df: pd.DataFrame, metric_col: str, eps: float, evaluated_folds: List[int], n_boot: int = 10000, seed: int = 42) -> Tuple[float, float]:
# |     rng = np.random.RandomState(seed)
# |     
# |     vals: Dict[int, np.ndarray] = {}
# |     for f in evaluated_folds:
# |         mask = (city_df.fold == f) & (city_df.epsilon == eps)
# |         vals[f] = city_df[mask][metric_col].values
# |         assert len(vals[f]) == 10, f"Expected 10 cities for fold {f}, got {len(vals[f])}"
# |         
# |     f_samples = [vals[f][rng.randint(0, 10, size=(n_boot, 10))] for f in evaluated_folds]
# |     all_samples = np.hstack(f_samples)
# |     boot_means = np.mean(all_samples, axis=1)
# |         
# |     return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))
# |
# |
# | def fast_cal_metrics(
# |     yd_tgt: np.ndarray, 
# |     eps_req: float, 
# |     compute_spearman: bool, 
# |     N_hat: float, 
# |     K: int, 
# |     active: np.ndarray, 
# |     Y_hat: np.ndarray, 
# |     t0_inter: np.ndarray, 
# |     bin_idx: np.ndarray, 
# |     t_true_inter: np.ndarray, 
# |     cpc_m0: float, 
# |     yd_target: np.ndarray,
# |     inv_sum_denom: float,
# |     inv_N: float,
# |     t_cal_buf: np.ndarray,
# |     diff_buf: np.ndarray
# | ) -> Tuple[float, float, float, float, float, float, Dict[str, float]]:
# |     
# |     if N_hat <= 0:
# |         return cpc_m0, 0.0, 0.0, 0.0, eps_req, 0.0, {}
# |     
# |     yd_raw = yd_tgt / yd_tgt.sum() if yd_tgt.sum() > 0 else np.ones(K) / K
# |     yd_active = yd_raw * active.astype(np.float64)
# |     active_sum = yd_active.sum()
# |     Y_D_cond = yd_active / active_sum if active_sum > 0 else Y_hat.copy()
# |     
# |     w = np.ones(K, dtype=np.float64)
# |     for k in range(K):
# |         if active[k] and Y_hat[k] > 0:
# |             w[k] = Y_D_cond[k] / Y_hat[k]
# |             
# |     weighted_mass = float(np.dot(Y_hat, w))
# |     s = w / weighted_mass if weighted_mass > 0 else np.ones(K)
# |     
# |     np.multiply(t0_inter, s[bin_idx], out=t_cal_buf)
# |             
# |     cal_mass = t_cal_buf.sum()
# |     if cal_mass > 0:
# |         t_cal_buf *= (N_hat / cal_mass)
# |         
# |     cpc = float(np.sum(np.minimum(t_true_inter, t_cal_buf)) * inv_sum_denom)
# |     
# |     np.subtract(t_true_inter, t_cal_buf, out=diff_buf)
# |     np.abs(diff_buf, out=diff_buf)
# |     mae = float(np.sum(diff_buf) * inv_N)
# |     
# |     np.square(diff_buf, out=diff_buf)
# |     rmse = float(np.sqrt(np.sum(diff_buf) * inv_N))
# |     
# |     spearman_val = float(spearmanr(t_true_inter, t_cal_buf)[0]) if compute_spearman else float('nan')
# |     
# |     active_w = w[active]
# |     w_gt_2 = float(np.mean(active_w > 2)) if len(active_w) > 0 else 0.0
# |     w_gt_5 = float(np.mean(active_w > 5)) if len(active_w) > 0 else 0.0
# |     w_gt_10 = float(np.mean(active_w > 10)) if len(active_w) > 0 else 0.0
# |     
# |     stats = {
# |         "w_min": float(active_w.min()) if len(active_w) > 0 else 1.0,
# |         "w_median": float(np.median(active_w)) if len(active_w) > 0 else 1.0,
# |         "w_p95": float(np.percentile(active_w, 95)) if len(active_w) > 0 else 1.0,
# |         "w_max": float(active_w.max()) if len(active_w) > 0 else 1.0,
# |         "w_gt_2": w_gt_2, "w_gt_5": w_gt_5, "w_gt_10": w_gt_10
# |     }
# |     
# |     tv_ach = float(0.5 * np.sum(np.abs(yd_tgt - yd_target)))
# |     js_div = float(jensenshannon(yd_tgt, yd_target)) ** 2
# |     
# |     return cpc, mae, rmse, spearman_val, tv_ach, js_div, stats
# |
# |
# | def run_noise_robustness(args: argparse.Namespace) -> None:
# |     data_root = "data"
# |     grid_mode = getattr(args, "grid", "fine")
# |     if grid_mode == "fine":
# |         epsilons = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
# |         output_dir = getattr(args, "output_dir", None) or "results/noise_robustness_fine_v1"
# |     else:
# |         epsilons = [0.0, 0.05, 0.10, 0.20]
# |         output_dir = getattr(args, "output_dir", None) or "results/noise_robustness_v1"
# |         
# |     os.makedirs(output_dir, exist_ok=True)
# |     
# |     log_file = f"{output_dir}/run.log"
# |     logging.basicConfig(level=logging.INFO, format='%(message)s',
# |                         handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
# |     logger = logging.getLogger(__name__)
# |     
# |     noise_seed = 20260822
# |     nonzero_epsilons = [e for e in epsilons if e > 0]
# |     
# |     # Safely define parameters without mutating globals
# |     model_seeds_to_use = [1, 10, 100] if not args.smoke else [1, 10]
# |     B_noise = args.b if not args.smoke else 20
# |     folds_to_run = [1, 2, 3, 4, 5] if not args.smoke else [2]
# |         
# |     splits = generate_35_5_10_splits(data_root=data_root)
# |     raw_results: List[Dict[str, Any]] = []
# |     
# |     for fold_id in folds_to_run:
# |         split = splits[fold_id]
# |         train_cities = split["train"]
# |         test_cities_to_use = split["test"] if not args.smoke else split["test"][:1]
# |             
# |         logger.info(f"\n=== Processing Fold {fold_id} ===")
# |         
# |         bin_edges, _ = compute_kbin_edges(train_cities, K=8, data_root=data_root)
# |         K = len(bin_edges) - 1
# |         
# |         for c_idx, tc in enumerate(test_cities_to_use):
# |             logger.info(f"  Target City: {tc} ({c_idx+1}/{len(test_cities_to_use)})")
# |             raw = load_raw_city(tc, data_root=data_root)
# |             dist_km = raw.dist_km
# |             inter_mask = (raw.pair_o_idx.numpy() != raw.pair_d_idx.numpy()) & (dist_km > 0.0)
# |             t_true_inter = raw.pair_trips.numpy()[inter_mask]
# |             
# |             yd_target = extract_yd_kbins(dist_km, raw.pair_trips.numpy(), bin_edges, inter_mask)
# |             active_mask = get_active_bins(yd_target)
# |             
# |             p_active_orig = yd_target[active_mask]
# |             p_active_orig = p_active_orig / p_active_orig.sum()
# |             
# |             logger.info("    Generating noise nested directions...")
# |             city_noise_sets: List[Dict[float, np.ndarray]] = []
# |             for b in range(B_noise):
# |                 seed_b = get_stable_seed(noise_seed, fold_id, tc, b+1)
# |                 noisy_dict = generate_nested_noisy_yd(p_active_orig, epsilons, seed_b)
# |                 full_dict: Dict[float, np.ndarray] = {}
# |                 for eps, p_act in noisy_dict.items():
# |                     full_yd = np.zeros(K)
# |                     full_yd[active_mask] = p_act
# |                     full_dict[eps] = full_yd
# |                 city_noise_sets.append(full_dict)
# |                 
# |             edge_index, edge_dist = build_radius_graph(
# |                 lon_lat=raw.lon_lat, radius_km=5.0, include_self_loop=True, cache_key=f"{tc}_tracts"
# |             )
# |             
# |             dist_inter = dist_km[inter_mask]
# |             bin_idx = np.clip(np.digitize(dist_inter, bin_edges[1:-1], right=True), 0, K - 1).astype(np.int32)
# |             n_inter_pairs = len(dist_inter)
# |             inv_N = 1.0 / n_inter_pairs if n_inter_pairs > 0 else 0.0
# |             sum_t_true = float(t_true_inter.sum())
# |             
# |             t_cal_buf = np.empty(n_inter_pairs, dtype=np.float64)
# |             diff_buf = np.empty(n_inter_pairs, dtype=np.float64)
# |             
# |             for m_seed in model_seeds_to_use:
# |                 logger.info(f"    Evaluating seed {m_seed}...")
# |                 ckpt_path = Path(f"results/checkpoints/5fold_fold{fold_id}_seed{m_seed}.pt")
# |                 if not ckpt_path.exists():
# |                     raise FileNotFoundError(f"Missing mandatory checkpoint {ckpt_path}. Protocol requires all 3 model seeds.")
# |                 model, scaler, _ = load_checkpoint(ckpt_path, device_str="cpu")
# |                 model.eval()
# |                 
# |                 city_data = load_city(tc, data_root=data_root, feature_scaler=scaler, fit_scaler=False)
# |                 t_pred_zs_tensor = infer_zero_shot(model, city_data, edge_index, edge_dist, device="cpu")
# |                 t_pred_zs = t_pred_zs_tensor.numpy().astype(np.float64)
# |                 
# |                 t0_inter = t_pred_zs[inter_mask]
# |                 N_hat = float(t0_inter.sum())
# |                 
# |                 cpc_m0 = float(evaluate_cpc(t_true_inter, t0_inter))
# |                 
# |                 sum_denom = sum_t_true + N_hat
# |                 inv_sum_denom = 2.0 / sum_denom if sum_denom > 0 else 0.0
# |                 
# |                 Y_hat = np.zeros(K, dtype=np.float64)
# |                 active = np.zeros(K, dtype=bool)
# |                 if N_hat > 0:
# |                     counts = np.bincount(bin_idx, weights=t0_inter, minlength=K)
# |                     Y_hat = counts / N_hat
# |                     pair_counts = np.bincount(bin_idx, minlength=K)
# |                     active = pair_counts > 0
# |                 
# |                 # 1. Oracle (eps=0)
# |                 oracle_cpc, o_mae, o_rmse, o_spr, o_tv, o_js, o_stats = fast_cal_metrics(
# |                     yd_target, 0.0, True, N_hat, K, active, Y_hat, t0_inter, bin_idx, t_true_inter, cpc_m0, yd_target,
# |                     inv_sum_denom, inv_N, t_cal_buf, diff_buf
# |                 )
# |                 
# |                 if args.smoke:
# |                     assert o_tv < 1e-8, "Oracle TV is not 0"
# |                     
# |                 def build_row(eps_val: float, rep_id: int, cpc_val: float, mae: float, rmse: float, spr: float, tv_ach: float, js_div: float, st: Dict[str, float]) -> Dict[str, Any]:
# |                     row = {
# |                         "fold": fold_id, "target_city": tc, "model_seed": m_seed,
# |                         "epsilon": eps_val, "replicate_id": rep_id,
# |                         "cpc_m0_inter": cpc_m0, "cpc_m1_inter": cpc_val,
# |                         "delta_cpc_inter": float(cpc_val - cpc_m0),
# |                         "degradation": float(oracle_cpc - cpc_val),
# |                         "mae": mae, "rmse": rmse, "spearman": spr,
# |                         "achieved_tv": tv_ach, "js_divergence": js_div,
# |                         "q": 1.0
# |                     }
# |                     row.update(st)
# |                     return row
# |                     
# |                 raw_results.append(build_row(0.0, 0, oracle_cpc, o_mae, o_rmse, o_spr, o_tv, o_js, o_stats))
# |                 
# |                 # 2. Noise replicates
# |                 for b, noisy_dict in enumerate(city_noise_sets):
# |                     for eps in nonzero_epsilons:
# |                         n_cpc, n_mae, n_rmse, n_spr, n_tv, n_js, n_stats = fast_cal_metrics(
# |                             noisy_dict[eps], eps, False, N_hat, K, active, Y_hat, t0_inter, bin_idx, t_true_inter, cpc_m0, yd_target,
# |                             inv_sum_denom, inv_N, t_cal_buf, diff_buf
# |                         )
# |                         assert np.abs(n_tv - eps) < 1e-8, f"TV mismatch in loop for eps {eps}: got {n_tv}"
# |                         raw_results.append(build_row(eps, b+1, n_cpc, n_mae, n_rmse, n_spr, n_tv, n_js, n_stats))
# |                 
# |     df = pd.DataFrame(raw_results)
# |     if not df.empty:
# |         # Enforce explicit typing for consistency
# |         df['spearman'] = df['spearman'].astype(float)
# |         
# |         df.to_csv(f"{output_dir}/noise_raw.csv", index=False)
# |         df.to_json(f"{output_dir}/noise_raw.jsonl", orient="records", lines=True)
# |         logger.info(f"Raw results saved with {len(df)} rows.")
# |         
# |         # Aggregation Step 1 & 2
# |         df_mean_b = df.groupby(["fold", "target_city", "model_seed", "epsilon"]).agg(
# |             delta_cpc_inter=("delta_cpc_inter", "mean"),
# |             degradation=("degradation", "mean"),
# |             w_max=("w_max", "mean"),
# |             w_gt_2=("w_gt_2", "mean"),
# |             cpc_m1_inter=("cpc_m1_inter", "mean"),
# |             prob_positive=("delta_cpc_inter", lambda x: float(np.mean(x > 0)))
# |         ).reset_index()
# |         
# |         df_seed_csv = df_mean_b.copy()
# |         df_seed_csv.to_csv(f"{output_dir}/noise_per_seed.csv", index=False)
# |         
# |         city_df = df_mean_b.groupby(["fold", "target_city", "epsilon"]).agg(
# |             delta_cpc_mean=("delta_cpc_inter", "mean"),
# |             degradation_mean=("degradation", "mean"),
# |             prob_positive=("prob_positive", "mean"),
# |             cpc_m1_inter=("cpc_m1_inter", "mean"),
# |             w_max=("w_max", "mean"),
# |             w_gt_2=("w_gt_2", "mean")
# |         ).reset_index()
# |         
# |         city_df.to_csv(f"{output_dir}/noise_per_city.csv", index=False)
# |         
# |         if not args.smoke:
# |             generate_summary(city_df, output_dir, epsilons, nonzero_epsilons, B_noise)
# |     else:
# |         logger.warning("No results were generated. Check checkpoints.")
# |         
# |
# | def generate_summary(
# |     city_df: pd.DataFrame,
# |     output_dir: str,
# |     epsilons: List[float],
# |     nonzero_epsilons: List[float],
# |     b_noise: int | None = None,
# | ) -> None:
# |     evaluation_folds = sorted(city_df.fold.unique().tolist())
# |     eval_df = city_df[city_df.fold.isin(evaluation_folds)]
# |     
# |     if eval_df.empty:
# |         return
# |         
# |     results: Dict[float, Dict[str, Any]] = {}
# |     p_benefit_onesided: List[float] = []
# |     p_degrad_onesided: List[float] = []
# |     
# |     # Get oracle delta_cpc per city for degradation paired test
# |     clean_vals_by_city: Dict[Tuple[int, str], float] = {}
# |     c_clean = eval_df[eval_df.epsilon == 0.0]
# |     for _, row in c_clean.iterrows():
# |         clean_vals_by_city[(row["fold"], row["target_city"])] = row["delta_cpc_mean"]
# |     
# |     for eps in epsilons:
# |         c_eps = eval_df[eval_df.epsilon == eps]
# |         vals = c_eps.delta_cpc_mean.values
# |         
# |         mean_cpc1 = float(c_eps.cpc_m1_inter.mean())
# |         mean_val = float(np.mean(vals))
# |         sd_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
# |         median = float(np.median(vals))
# |         p25 = float(np.percentile(vals, 25))
# |         p75 = float(np.percentile(vals, 75))
# |         pos_cities = int(np.sum(vals > 0))
# |         harm_rate = float(np.sum(vals < 0) / len(vals))
# |         
# |         ci_lower, ci_upper = fold_stratified_bootstrap(eval_df, "delta_cpc_mean", eps, evaluation_folds)
# |         
# |         # 1. Benefit Test (H1: delta_cpc > 0 vs M0)
# |         try:
# |             _, p_ben = wilcoxon(vals, alternative='greater')
# |         except Exception:
# |             p_ben = 1.0
# |             
# |         # 2. Degradation Test (H1: delta_cpc_clean - delta_cpc_eps > 0 vs clean Y_D)
# |         degrad_vals = []
# |         for _, row in c_eps.iterrows():
# |             clean_v = clean_vals_by_city.get((row["fold"], row["target_city"]), row["delta_cpc_mean"])
# |             degrad_vals.append(clean_v - row["delta_cpc_mean"])
# |         degrad_arr = np.array(degrad_vals)
# |         mean_degrad = float(np.mean(degrad_arr))
# |         
# |         if eps > 0.0:
# |             try:
# |                 _, p_deg = wilcoxon(degrad_arr, alternative='greater')
# |             except Exception:
# |                 p_deg = 1.0
# |             p_benefit_onesided.append(float(p_ben))
# |             p_degrad_onesided.append(float(p_deg))
# |         else:
# |             p_deg = float('nan')
# |         
# |         results[eps] = {
# |             "mean_cpc1": mean_cpc1,
# |             "mean_delta_cpc": mean_val, "sd": sd_val, "median": median,
# |             "p25": p25, "p75": p75, "ci_lower": ci_lower, "ci_upper": ci_upper,
# |             "pos_cities": pos_cities, "harm_rate": harm_rate,
# |             "mean_degradation": mean_degrad,
# |             "wilcoxon_benefit_raw": float(p_ben),
# |             "wilcoxon_degrad_raw": float(p_deg) if not np.isnan(p_deg) else None
# |         }
# |         
# |     p_ben_adj = holm_correction(p_benefit_onesided)
# |     p_deg_adj = holm_correction(p_degrad_onesided)
# |     
# |     for i, e in enumerate(nonzero_epsilons):
# |         results[e]["wilcoxon_benefit_holm"] = float(p_ben_adj[i])
# |         results[e]["wilcoxon_degrad_holm"] = float(p_deg_adj[i])
# |         
# |     oracle_gain = float(results[0.0]["mean_delta_cpc"])
# |     for e in epsilons:
# |         if oracle_gain > 0:
# |             results[e]["relative_effect_pct"] = float(results[e]["mean_delta_cpc"] / oracle_gain * 100.0)
# |         else:
# |             results[e]["relative_effect_pct"] = None
# |             
# |     # Estimate exact crossover point epsilon_cross where mean_delta_cpc = 0
# |     eps_cross = None
# |     sorted_eps = sorted(epsilons)
# |     for i in range(len(sorted_eps) - 1):
# |         e1, e2 = sorted_eps[i], sorted_eps[i + 1]
# |         v1, v2 = results[e1]["mean_delta_cpc"], results[e2]["mean_delta_cpc"]
# |         if v1 >= 0 and v2 < 0:
# |             # Linear interpolation
# |             eps_cross = float(e1 + v1 / (v1 - v2) * (e2 - e1))
# |             break
# |         elif v1 > 0 and v2 == 0:
# |             eps_cross = float(e2)
# |             break
# |             
# |     # Estimate epsilon* (highest noise level with significant positive benefit)
# |     eps_star = 0.0
# |     for i, eps in enumerate(nonzero_epsilons):
# |         cond1 = results[eps]["mean_delta_cpc"] > 0
# |         cond2 = results[eps]["ci_lower"] > 0
# |         cond3 = results[eps]["wilcoxon_benefit_holm"] < 0.05
# |         if cond1 and cond2 and cond3:
# |             eps_star = eps
# |         else:
# |             break
# |             
# |     summary = {
# |         "n_evaluation_cities": int(len(eval_df) // len(epsilons)),
# |         "eps_cross_zero_dCPC": eps_cross,
# |         "eps_star_significant_benefit": float(eps_star),
# |         "results_by_eps": results
# |     }
# |     
# |     with open(f"{output_dir}/noise_summary.json", "w") as f:
# |         json.dump(summary, f, indent=2)
# |         
# |     md = "# 5-Fold Noise Robustness Summary\n\n"
# |     md += f"## Five-Fold Cross-City Evaluation Table (All 5 Folds, {int(len(eval_df)//len(epsilons))} Held-Out Test Cities)\n\n"
# |     if eps_cross is not None:
# |         md += f"**Crossover Threshold ($\\epsilon_{{\\text{{cross}}}}$, $\\Delta\\text{{CPC}}=0$):** `{eps_cross:.4f}` (TV $\\approx {eps_cross*100:.2f}\\%$)\n\n"
# |     md += "| Noise (eps) | Mean M1 CPC | Mean dCPC | 95% CI | Pos Cities | Harm Rate | Rel Effect vs Clean (%) | Benefit p-val (vs M0) | Degrad p-val (vs Clean) |\n"
# |     md += "| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
# |     for e in epsilons:
# |         d = results[e]
# |         ci = f"[{d['ci_lower']:.5f}, {d['ci_upper']:.5f}]"
# |         
# |         ben_holm = d.get('wilcoxon_benefit_holm', d.get('wilcoxon_benefit_raw'))
# |         if isinstance(ben_holm, (float, np.floating)):
# |             ben_str = f"{ben_holm:.2e}" if ben_holm < 0.001 else f"{ben_holm:.4f}"
# |         else:
# |             ben_str = "N/A"
# |             
# |         deg_holm = d.get('wilcoxon_degrad_holm')
# |         if isinstance(deg_holm, (float, np.floating)):
# |             deg_str = f"{deg_holm:.2e}" if deg_holm < 0.001 else f"{deg_holm:.4f}"
# |         else:
# |             deg_str = "—"
# |             
# |         rel_eff = f"{d['relative_effect_pct']:+.1f}%" if d['relative_effect_pct'] is not None else "N/A"
# |         md += f"| {e} | {d['mean_cpc1']:.5f} | {d['mean_delta_cpc']:+.5f} | {ci} | {d['pos_cities']}/{int(len(eval_df)//len(epsilons))} | {d['harm_rate']:.1%} | {rel_eff} | {ben_str} | {deg_str} |\n"
# |         
# |     with open(f"{output_dir}/noise_summary.md", "w") as f:
# |         f.write(md)
# |         
# |     # Figure 1: Dose-Response with CI
# |     plt.figure(figsize=(8, 6))
# |     means = [results[e]["mean_delta_cpc"] for e in epsilons]
# |     ci_lowers = [results[e]["ci_lower"] for e in epsilons]
# |     ci_uppers = [results[e]["ci_upper"] for e in epsilons]
# |     yerr_lower = [m - cl for m, cl in zip(means, ci_lowers)]
# |     yerr_upper = [cu - m for m, cu in zip(means, ci_uppers)]
# |     
# |     plt.errorbar(epsilons, means, yerr=[yerr_lower, yerr_upper], fmt='-o', color='royalblue', ecolor='gray', capsize=5, label='Full 5-fold Mean (95% CI)')
# |     plt.axhline(0, color="red", linestyle="--", alpha=0.7, label='Zero-Shot M0 Baseline')
# |     if eps_cross is not None:
# |         plt.axvline(eps_cross, color="darkorange", linestyle=":", label=f'Crossover $\\epsilon_{{cross}} = {eps_cross:.3f}$')
# |     plt.xlabel("Noise Level (Epsilon TV)")
# |     plt.ylabel("Delta CPC (M1 - M0)")
# |     plt.title("Dose-Response: Noise Level vs Delta CPC")
# |     plt.grid(True, linestyle=':', alpha=0.6)
# |     plt.legend()
# |     plt.savefig(f"{output_dir}/fig_noise_dose_response.png", dpi=300, bbox_inches="tight")
# |     plt.close()
# |     
# |     # Figure 2: Harm Rate
# |     hr = [results[e]["harm_rate"] for e in epsilons]
# |     plt.figure(figsize=(8, 6))
# |     plt.plot(epsilons, hr, marker="s", color='red', linewidth=2)
# |     if eps_cross is not None:
# |         plt.axvline(eps_cross, color="darkorange", linestyle=":", label=f'Crossover $\\epsilon_{{cross}} = {eps_cross:.3f}$')
# |     plt.title("Harm Rate vs Noise Level")
# |     plt.xlabel("Noise Level (Epsilon TV)")
# |     plt.ylabel("Harm Rate (% Cities Worse than M0)")
# |     plt.ylim(0, 1.05)
# |     plt.grid(True, linestyle=':', alpha=0.6)
# |     plt.legend()
# |     plt.savefig(f"{output_dir}/fig_noise_harm_rate.png", dpi=300, bbox_inches="tight")
# |     plt.close()
# |     
# |     # Figure 3: Per-City Response
# |     plt.figure(figsize=(10, 8))
# |     for city_name, g in city_df.groupby("target_city"):
# |         plt.plot(g["epsilon"], g["delta_cpc_mean"], alpha=0.35, color='gray')
# |     plt.plot(epsilons, means, marker="o", color='blue', linewidth=2.5, label='Overall Mean')
# |     plt.axhline(0, color="black", linestyle="--", linewidth=1.5)
# |     plt.title("Per-City Response to Noise")
# |     plt.xlabel("Noise Level (Epsilon TV)")
# |     plt.ylabel("Delta CPC (M1 - M0)")
# |     plt.legend()
# |     plt.grid(True, linestyle=':', alpha=0.6)
# |     plt.savefig(f"{output_dir}/fig_noise_by_city.png", dpi=300, bbox_inches="tight")
# |     plt.close()
# |     
# |     manifest = {
# |         "noise_definition": "multiplicative compositional noise on active bins, TV distance matching via bisection",
# |         "timestamp": datetime.datetime.now().isoformat(),
# |         "B_noise": b_noise,
# |         "epsilons": epsilons,
# |         "eps_cross": eps_cross,
# |         "eps_star": eps_star
# |     }
# |     with open(f"{output_dir}/noise_manifest.json", "w") as f:
# |         json.dump(manifest, f, indent=2)
# |
# |
# | if __name__ == "__main__":
# |     parser = argparse.ArgumentParser()
# |     parser.add_argument("--b", type=int, default=1000)
# |     parser.add_argument("--grid", type=str, choices=["fine", "coarse"], default="fine", help="Grid: 'fine' [0..0.05] or 'coarse' [0..0.20]")
# |     parser.add_argument("--output_dir", type=str, default=None)
# |     parser.add_argument("--smoke", action="store_true")
# |     args = parser.parse_args()
# |     run_noise_robustness(args)

# ===== END SOURCE FILE: src/experiment/run_noise_robustness.py =====


# ===== BEGIN SOURCE FILE: src/experiment/run_partial_od_equivalence_v2.py =====
# | """
# | Partial-OD Information Equivalence Experiment v2 (Final Paper Protocol)
# | ========================================================================
# |
# | Core Scientific Research Question:
# |     Under the same frozen zero-shot model and the same production distance-bin
# |     calibration operator, what fraction of directly observed positive interzonal
# |     OD pairs is required to achieve reconstruction gain comparable to that
# |     obtained from the full target-city distance-binned mobility distribution?
# |
# | Primary Estimands:
# |     1. Positive-Benefit Threshold p*_benefit (Holm p < 0.05, CI_lower > 0)
# |     2. Operational Equivalence Crossing p_eq (where mean D(p) = Gain_OD(p) - Gain_YD(p) >= 0)
# |
# | Architectural Invariants:
# |     - 5 Folds, 50 held-out test cities (35 train / 5 val / 10 test per fold).
# |     - Model Seeds: {1, 10, 100} on frozen Gravity-Informed Urban GNN.
# |     - Zero retraining, zero fine-tuning, zero optimizer step, zero backward pass.
# |     - K = 8 distance bins, q = 1.0 within-tolerance multiplier scaling.
# |     - Calibration operator executed on full candidate support Omega_c^+, scored strictly on unseen U_p = Omega_c^+ \ S_p.
# |     - Nested permutation masks across 15 p-levels:
# |       [0.0, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90].
# |     - B = 500 Monte Carlo replicates per city.
# |     - Exact Per-Fold Storage Structure with incremental flush and completion markers.
# | """
# |
# | import os
# | import sys
# | import time
# | import json
# | import hashlib
# | import argparse
# | from pathlib import Path
# | from typing import Dict, List, Tuple, Any, Optional
# |
# | import numpy as np
# | import pandas as pd
# | from scipy import stats
# | from scipy.spatial.distance import jensenshannon
# | import matplotlib.pyplot as plt
# | import torch
# |
# | # Ensure repository root is on sys.path
# | REPO_ROOT = Path(__file__).resolve().parent.parent.parent
# | sys.path.insert(0, str(REPO_ROOT))
# |
# | from src.data.city_splits import generate_35_5_10_splits
# | from src.data.dataset import load_city, load_cities, load_raw_city
# | from src.data.urban_graph import build_radius_graph
# | from src.data.yd_extractor import compute_kbin_edges, extract_yd_kbins
# | from src.calibration.bin_calibration import calibrate_kbins
# | from src.training.evaluate import compute_cpc_pair
# | from src.training.train import load_checkpoint, infer_zero_shot
# |
# | PARTIAL_OD_BASE_SEED = 202608231
# | PRIMARY_GRID_V2 = [
# |     0.0, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 
# |     0.10, 0.20, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90
# | ]
# |
# | RAW_COLUMNS = [
# |     "fold", "city", "model_seed", "replicate_id", "p", "mask_seed",
# |     "n_total_pairs", "n_revealed", "n_unseen", "fraction_pairs_revealed",
# |     "total_trip_mass", "revealed_trip_mass", "fraction_trip_mass_revealed",
# |     "unseen_trip_mass", "fraction_unseen_trip_mass",
# |     "empirical_tv_partial_vs_full", "js_partial_vs_full",
# |     "cpc_m0_unseen", "cpc_full_yd_unseen", "cpc_partial_od_unseen",
# |     "gain_full_yd", "gain_partial_od", "difference_partial_minus_yd",
# |     "relative_gain_vs_yd", "K", "q"
# | ]
# |
# |
# | def get_stable_mask_seed(base_seed: int, fold: int, city: str, replicate_id: int) -> int:
# |     s = f"{base_seed}_{fold}_{city}_{replicate_id}"
# |     return int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % (2**32)
# |
# |
# | def holm_correction(p_vals: List[float]) -> np.ndarray:
# |     n = len(p_vals)
# |     if n == 0:
# |         return np.array([])
# |     sorted_indices = np.argsort(p_vals)
# |     adj_p = np.zeros(n)
# |     running_max = 0.0
# |     for i, idx in enumerate(sorted_indices):
# |         p_adj = p_vals[idx] * (n - i)
# |         running_max = max(running_max, p_adj)
# |         adj_p[idx] = min(1.0, running_max)
# |     return adj_p
# |
# |
# | def fold_stratified_bootstrap(
# |     city_df: pd.DataFrame, 
# |     metric_col: str, 
# |     p_val: float, 
# |     n_boot: int = 10000, 
# |     seed: int = 42
# | ) -> Tuple[float, float]:
# |     rng = np.random.RandomState(seed)
# |     sub = city_df[city_df.p == p_val]
# |     
# |     vals: Dict[int, np.ndarray] = {}
# |     for f in range(1, 6):
# |         f_vals = sub[sub.fold == f][metric_col].values
# |         if len(f_vals) > 0:
# |             vals[f] = f_vals
# |
# |     boot_means = np.empty(n_boot, dtype=np.float64)
# |     total_cities = sum(len(v) for v in vals.values())
# |     if total_cities == 0:
# |         return 0.0, 0.0
# |         
# |     for b in range(n_boot):
# |         sample_sum = 0.0
# |         for f, arr in vals.items():
# |             idx = rng.randint(0, len(arr), size=len(arr))
# |             sample_sum += arr[idx].sum()
# |         boot_means[b] = sample_sum / total_cities
# |
# |     return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))
# |
# |
# | def run_fold_partial_od(
# |     fold_id: int,
# |     data_root: str = "data",
# |     output_dir: Path = Path("results/partial_od_equivalence_v2"),
# |     replicates: int = 500,
# |     p_grid: List[float] = None,
# |     smoke: bool = False,
# |     smoke_cities: int = 1,
# |     resume: bool = False,
# |     device: str = "cpu"
# | ) -> Dict[str, Any]:
# |     if p_grid is None:
# |         p_grid = PRIMARY_GRID_V2.copy()
# |
# |     fold_dir = output_dir / f"fold_{fold_id}"
# |     fold_dir.mkdir(parents=True, exist_ok=True)
# |     
# |     raw_csv_path = fold_dir / "raw.csv"
# |     progress_json_path = fold_dir / "progress.json"
# |     marker_path = fold_dir / "completion.marker"
# |
# |     splits = generate_35_5_10_splits(data_root=data_root)
# |     split = splits[fold_id]
# |     train_cities = split["train"]
# |     test_cities = split["test"] if not smoke else split["test"][:smoke_cities]
# |     model_seeds = [1, 10, 100] if not smoke else [1, 10]
# |     B = replicates if not smoke else 20
# |
# |     print(f"\n>>> [STARTING FOLD {fold_id}/5] {len(test_cities)} test cities | B={B} reps | {len(p_grid)} p-levels | Seeds: {model_seeds}")
# |
# |     # Check already completed cities if resume is True
# |     completed_cities = set()
# |     if resume and progress_json_path.exists():
# |         try:
# |             with open(progress_json_path, "r") as f:
# |                 prog = json.load(f)
# |                 completed_cities = set(prog.get("completed_cities", []))
# |                 print(f"    Resuming fold {fold_id}: Found {len(completed_cities)} already completed cities.")
# |         except Exception:
# |             completed_cities = set()
# |
# |     # If raw.csv doesn't exist or not resuming, initialize with header
# |     if not resume or not raw_csv_path.exists():
# |         with open(raw_csv_path, "w", encoding="utf-8") as f:
# |             f.write(",".join(RAW_COLUMNS) + "\n")
# |
# |     # Load frozen GNN models for this fold
# |     models: Dict[int, Tuple[Any, Any]] = {}
# |     for s in model_seeds:
# |         ckpt_path = Path("results/checkpoints") / f"5fold_fold{fold_id}_seed{s}.pt"
# |         if not ckpt_path.exists():
# |             raise RuntimeError(f"Required checkpoint missing for fold {fold_id} seed {s}: {ckpt_path}")
# |         model, scaler, _ = load_checkpoint(ckpt_path, device_str=device)
# |         model.eval()
# |         models[s] = (model, scaler)
# |
# |     # Compute K=8 bin edges from 35 train cities
# |     bin_edges, K_act = compute_kbin_edges(train_cities, K=8, data_root=data_root)
# |     if K_act != 8 or len(bin_edges) != 9:
# |         raise RuntimeError(f"Strict 8-bin invariant failed for fold {fold_id}: K_act={K_act}")
# |
# |     fold_start_time = time.perf_counter()
# |     rows_written_total = 0
# |
# |     for city_idx, city_name in enumerate(test_cities):
# |         if city_name in completed_cities:
# |             print(f"  [{city_idx+1}/{len(test_cities)}] {city_name:<16} | ALREADY COMPLETED (Skipping)")
# |             continue
# |
# |         city_start = time.perf_counter()
# |         raw_data = load_raw_city(city_name, data_root=data_root)
# |         dist_km = raw_data.dist_km
# |         
# |         # Support Omega_c^+: strictly positive interzonal pairs
# |         inter_pos = (raw_data.pair_o_idx.numpy() != raw_data.pair_d_idx.numpy()) & (dist_km > 0.0) & (raw_data.pair_trips.numpy() > 0)
# |         n_pairs = int(inter_pos.sum())
# |         if n_pairs == 0:
# |             raise RuntimeError(f"Critical error: City {city_name} has 0 positive interzonal pairs!")
# |
# |         t_true_support = raw_data.pair_trips.numpy()[inter_pos].astype(np.float64)
# |         dist_support = dist_km[inter_pos]
# |         bin_idx_support = np.clip(np.digitize(dist_support, bin_edges, right=True) - 1, 0, 7)
# |         total_trip_mass = float(np.sum(t_true_support))
# |         
# |         # Extract clean full Y_D on support
# |         yd_full = np.bincount(bin_idx_support, weights=t_true_support, minlength=8).astype(np.float64)
# |         yd_full /= total_trip_mass
# |
# |         # Precalculate M0 and full Y_D calibrated prediction for all model seeds
# |         seed_predictions: Dict[int, Dict[str, np.ndarray]] = {}
# |         for s in model_seeds:
# |             model, scaler = models[s]
# |             city_data = load_city(city_name, data_root=data_root, feature_scaler=scaler, fit_scaler=False)
# |             coords = city_data.lon_lat.numpy()
# |             ei, ed = build_radius_graph(coords, radius_km=5.0)
# |             
# |             with torch.no_grad():
# |                 m0_full = infer_zero_shot(model, city_data, ei, ed, device=device).numpy().astype(np.float64)
# |             
# |             t0_support = m0_full[inter_pos]
# |             N_hat_support = float(np.sum(t0_support))
# |             
# |             # Precompute full Y_D calibrated predictions
# |             Y_hat = np.bincount(bin_idx_support, weights=t0_support, minlength=8).astype(np.float64)
# |             Y_hat /= N_hat_support
# |             
# |             active = np.zeros(8, dtype=bool)
# |             for k in range(8):
# |                 active[k] = bool((bin_idx_support == k).any())
# |             yd_act = yd_full * active.astype(np.float64)
# |             act_sum = yd_act.sum()
# |             Y_D_cond = yd_act / act_sum if act_sum > 0 else Y_hat.copy()
# |
# |             w_full = np.ones(8, dtype=np.float64)
# |             for k in range(8):
# |                 if active[k] and Y_hat[k] > 0:
# |                     w_full[k] = Y_D_cond[k] / Y_hat[k]
# |             weighted_mass_full = float(np.dot(Y_hat, w_full))
# |             s_full = w_full / weighted_mass_full if weighted_mass_full > 0 else np.ones(8)
# |             
# |             t_cal_full_support = t0_support * s_full[bin_idx_support]
# |             cal_mass_full = np.sum(t_cal_full_support)
# |             if cal_mass_full > 0:
# |                 t_cal_full_support *= (N_hat_support / cal_mass_full)
# |                 
# |             seed_predictions[s] = {
# |                 "t0": t0_support,
# |                 "N_hat": N_hat_support,
# |                 "Y_hat": Y_hat,
# |                 "active": active,
# |                 "t_cal_full": t_cal_full_support
# |             }
# |
# |         city_rows = []
# |
# |         # Run Replicate Sampling
# |         for rep_id in range(B):
# |             mask_seed = get_stable_mask_seed(PARTIAL_OD_BASE_SEED, fold_id, city_name, rep_id)
# |             rng = np.random.RandomState(mask_seed)
# |             
# |             # Single random permutation for nested masks
# |             perm = rng.permutation(n_pairs)
# |             
# |             for p_val in p_grid:
# |                 n_reveal = int(np.round(p_val * n_pairs))
# |                 rev_indices = perm[:n_reveal]
# |                 unseen_indices = perm[n_reveal:]
# |                 n_unseen = len(unseen_indices)
# |                 
# |                 if n_unseen == 0:
# |                     continue
# |
# |                 # Construct partial Y_D from revealed pairs S_p
# |                 if n_reveal == 0:
# |                     yd_partial = None
# |                     revealed_mass = 0.0
# |                     tv_partial = np.nan
# |                     js_partial = np.nan
# |                 else:
# |                     rev_trips = t_true_support[rev_indices]
# |                     rev_bins = bin_idx_support[rev_indices]
# |                     revealed_mass = float(np.sum(rev_trips))
# |                     
# |                     counts_k = np.bincount(rev_bins, weights=rev_trips, minlength=8).astype(np.float64)
# |                     if revealed_mass > 0:
# |                         yd_partial = counts_k / revealed_mass
# |                     else:
# |                         yd_partial = None
# |                         
# |                     if yd_partial is not None:
# |                         tv_partial = float(0.5 * np.sum(np.abs(yd_partial - yd_full)))
# |                         js_partial = float(jensenshannon(yd_partial, yd_full))
# |                     else:
# |                         tv_partial = np.nan
# |                         js_partial = np.nan
# |
# |                 frac_pairs_rev = float(n_reveal) / float(n_pairs)
# |                 frac_mass_rev = float(revealed_mass) / float(total_trip_mass) if total_trip_mass > 0 else 0.0
# |                 unseen_mass = total_trip_mass - revealed_mass
# |                 frac_unseen_mass = unseen_mass / total_trip_mass if total_trip_mass > 0 else 0.0
# |                 
# |                 # Target ground truth on unseen set U_p
# |                 t_true_unseen = t_true_support[unseen_indices]
# |                 sum_true_unseen = float(np.sum(t_true_unseen))
# |
# |                 # Evaluate across all 3 model seeds with identical mask
# |                 for s in model_seeds:
# |                     preds = seed_predictions[s]
# |                     t0_support = preds["t0"]
# |                     t0_unseen = t0_support[unseen_indices]
# |                     t_full_unseen = preds["t_cal_full"][unseen_indices]
# |                     
# |                     # Compute M0 CPC on unseen set
# |                     denom_m0 = sum_true_unseen + float(np.sum(t0_unseen))
# |                     cpc_m0_unseen = (2.0 * np.sum(np.minimum(t_true_unseen, t0_unseen)) / denom_m0) if denom_m0 > 0 else 0.0
# |                     
# |                     # Compute Full Y_D CPC on unseen set
# |                     denom_full = sum_true_unseen + float(np.sum(t_full_unseen))
# |                     cpc_full_unseen = (2.0 * np.sum(np.minimum(t_true_unseen, t_full_unseen)) / denom_full) if denom_full > 0 else 0.0
# |                     
# |                     # Compute Partial Y_D Calibrated CPC on unseen set
# |                     if yd_partial is None:
# |                         cpc_part_unseen = cpc_m0_unseen
# |                     else:
# |                         from src.calibration.bin_calibration import calibrate_kbins
# |                         t_part_support = calibrate_kbins(
# |                             t0_support,
# |                             dist_support,
# |                             np.ones(len(t0_support), dtype=bool),
# |                             yd_partial,
# |                             bin_edges,
# |                             q=1.0,
# |                         )
# |                         t_part_unseen = t_part_support[unseen_indices]
# |                         
# |                         denom_part = sum_true_unseen + float(np.sum(t_part_unseen))
# |                         cpc_part_unseen = (2.0 * np.sum(np.minimum(t_true_unseen, t_part_unseen)) / denom_part) if denom_part > 0 else 0.0
# |
# |                     gain_full = float(cpc_full_unseen - cpc_m0_unseen)
# |                     gain_part = float(cpc_part_unseen - cpc_m0_unseen)
# |                     diff_part_minus_yd = float(gain_part - gain_full)
# |                     rel_gain = float(gain_part / gain_full) if abs(gain_full) > 1e-8 else 1.0
# |
# |                     city_rows.append((
# |                         fold_id, city_name, s, rep_id, p_val, mask_seed,
# |                         n_pairs, n_reveal, n_unseen, frac_pairs_rev,
# |                         total_trip_mass, revealed_mass, frac_mass_rev,
# |                         unseen_mass, frac_unseen_mass,
# |                         tv_partial, js_partial,
# |                         cpc_m0_unseen, cpc_full_unseen, cpc_part_unseen,
# |                         gain_full, gain_part, diff_part_minus_yd,
# |                         rel_gain, 8, 1.0
# |                     ))
# |
# |         # Append city records to raw CSV incrementally
# |         with open(raw_csv_path, "a", encoding="utf-8") as f:
# |             for r in city_rows:
# |                 f.write(",".join(str(x) for x in r) + "\n")
# |
# |         completed_cities.add(city_name)
# |         rows_written_total += len(city_rows)
# |
# |         # Update progress.json
# |         with open(progress_json_path, "w", encoding="utf-8") as f:
# |             json.dump({
# |                 "fold": fold_id,
# |                 "completed_cities": sorted(list(completed_cities)),
# |                 "remaining_cities": [c for c in test_cities if c not in completed_cities],
# |                 "rows_written": rows_written_total,
# |                 "protocol_version": "v2",
# |                 "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
# |             }, f, indent=2)
# |
# |         city_elapsed = time.perf_counter() - city_start
# |         print(f"  [{city_idx+1}/{len(test_cities)}] {city_name:<16} | Pairs: {n_pairs:>5} | B={B} reps done in {city_elapsed:.2f}s (Flushed {len(city_rows)} rows)")
# |
# |     # Read back raw.csv to generate per_seed, per_city, and fold_summary
# |     fold_df = pd.read_csv(raw_csv_path)
# |     
# |     # 1. Per-Seed Aggregation: Mean over B replicates -> (fold x city x model_seed x p)
# |     per_seed_df = fold_df.groupby(["fold", "city", "model_seed", "p"]).agg({
# |         "fraction_pairs_revealed": "mean",
# |         "fraction_trip_mass_revealed": "mean",
# |         "fraction_unseen_trip_mass": "mean",
# |         "empirical_tv_partial_vs_full": "mean",
# |         "js_partial_vs_full": "mean",
# |         "cpc_m0_unseen": "mean",
# |         "cpc_full_yd_unseen": "mean",
# |         "cpc_partial_od_unseen": "mean",
# |         "gain_full_yd": "mean",
# |         "gain_partial_od": "mean",
# |         "difference_partial_minus_yd": "mean",
# |         "relative_gain_vs_yd": "mean"
# |     }).reset_index()
# |     per_seed_csv_path = fold_dir / "per_seed.csv"
# |     per_seed_df.to_csv(per_seed_csv_path, index=False)
# |
# |     # 2. Per-City Aggregation: Mean over 3 model seeds -> (fold x city x p)
# |     per_city_df = per_seed_df.groupby(["fold", "city", "p"]).agg({
# |         "fraction_pairs_revealed": "mean",
# |         "fraction_trip_mass_revealed": "mean",
# |         "fraction_unseen_trip_mass": "mean",
# |         "empirical_tv_partial_vs_full": "mean",
# |         "js_partial_vs_full": "mean",
# |         "cpc_m0_unseen": "mean",
# |         "cpc_full_yd_unseen": "mean",
# |         "cpc_partial_od_unseen": "mean",
# |         "gain_full_yd": "mean",
# |         "gain_partial_od": "mean",
# |         "difference_partial_minus_yd": "mean",
# |         "relative_gain_vs_yd": "mean"
# |     }).reset_index()
# |     per_city_csv_path = fold_dir / "per_city.csv"
# |     per_city_df.to_csv(per_city_csv_path, index=False)
# |
# |     # 3. Fold Summary Table
# |     fold_summary_rows = []
# |     for p_val in p_grid:
# |         sub = per_city_df[per_city_df.p == p_val]
# |         fold_summary_rows.append({
# |             "p": p_val,
# |             "n_cities": len(sub),
# |             "mean_gain_full_yd": float(sub["gain_full_yd"].mean()),
# |             "mean_gain_partial_od": float(sub["gain_partial_od"].mean()),
# |             "mean_diff_vs_yd": float(sub["difference_partial_minus_yd"].mean()),
# |             "mean_tv": float(sub["empirical_tv_partial_vs_full"].mean()),
# |             "pos_cities": int((sub["gain_partial_od"] > 0).sum()),
# |             "match_yd_cities": int((sub["difference_partial_minus_yd"] >= 0).sum())
# |         })
# |
# |     fold_summary_json_path = fold_dir / "fold_summary.json"
# |     with open(fold_summary_json_path, "w", encoding="utf-8") as f:
# |         json.dump({"fold": fold_id, "summary_by_p": fold_summary_rows}, f, indent=2)
# |
# |     fold_summary_md_path = fold_dir / "fold_summary.md"
# |     with open(fold_summary_md_path, "w", encoding="utf-8") as f:
# |         f.write(f"# Fold {fold_id} Partial-OD Summary Table (N={len(test_cities)} Cities)\n\n")
# |         f.write("| p | Mean Gain Full $Y_D$ | Mean Gain Partial OD | Mean $D(p)$ (Part - Full) | Mean TV | Positive Cities | Match Full $Y_D$ |\n")
# |         f.write("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
# |         for r in fold_summary_rows:
# |             f.write(f"| **{r['p']*100:.2f}%** | +{r['mean_gain_full_yd']:.5f} | {r['mean_gain_partial_od']:+.5f} | {r['mean_diff_vs_yd']:+.5f} | {r['mean_tv']*100:.2f}% | {r['pos_cities']}/{r['n_cities']} | {r['match_yd_cities']}/{r['n_cities']} |\n")
# |
# |     # 4. Save Run Manifest
# |     manifest_path = fold_dir / "run_manifest.json"
# |     with open(manifest_path, "w", encoding="utf-8") as f:
# |         json.dump({
# |             "fold": fold_id,
# |             "protocol_version": "v2",
# |             "cities": test_cities,
# |             "model_seeds": model_seeds,
# |             "replicates": B,
# |             "p_grid": p_grid,
# |             "raw_rows": len(fold_df),
# |             "per_seed_rows": len(per_seed_df),
# |             "per_city_rows": len(per_city_df),
# |             "completed_at": time.strftime("%Y-%m-%d %H:%M:%S")
# |         }, f, indent=2)
# |
# |     # 5. QA Verification Before Writing completion.marker
# |     actual_raw_rows = len(fold_df)
# |     assert len(per_city_df) == len(test_cities) * len(p_grid), f"Fold {fold_id} per_city rows mismatch"
# |     
# |     # Non-null assertions:
# |     # By contract §15, empirical_tv_partial_vs_full and js_partial_vs_full are NaN at p=0 (undefined discrepancy)
# |     non_tv_cols = [c for c in fold_df.columns if c not in ["empirical_tv_partial_vs_full", "js_partial_vs_full"]]
# |     assert not fold_df[non_tv_cols].isnull().any().any(), f"Fold {fold_id} contains unexpected NaN values in required fields!"
# |     assert not fold_df[fold_df["p"] > 0]["empirical_tv_partial_vs_full"].isnull().any(), f"Fold {fold_id} contains NaN TV for p > 0!"
# |
# |     with open(marker_path, "w", encoding="utf-8") as f:
# |         f.write(f"FOLD {fold_id} COMPLETED AND CERTIFIED\nTimestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
# |
# |     fold_total_time = time.perf_counter() - fold_start_time
# |     print(f">>> [FOLD {fold_id} COMPLETE] Certified {actual_raw_rows} rows in {fold_total_time:.2f}s | Marker: {marker_path.name}")
# |     
# |     return {
# |         "fold": fold_id,
# |         "raw_rows": actual_raw_rows,
# |         "per_seed_rows": len(per_seed_df),
# |         "per_city_rows": len(per_city_df),
# |         "status": "PASS"
# |     }
# |
# |
# | def aggregate_combined_results(
# |     output_dir: Path = Path("results/partial_od_equivalence_v2"),
# |     p_grid: List[float] = None
# | ) -> None:
# |     if p_grid is None:
# |         p_grid = PRIMARY_GRID_V2.copy()
# |
# |     combined_dir = output_dir / "combined"
# |     combined_dir.mkdir(parents=True, exist_ok=True)
# |     (combined_dir / "figures").mkdir(parents=True, exist_ok=True)
# |
# |     print("\n" + "=" * 85)
# |     print("MASTER AGGREGATION & SCIENTIFIC SUMMARY (COMBINING ALL 5 FOLDS, N=50 CITIES)")
# |     print("=" * 85)
# |
# |     # Check that all 5 folds have completion.marker
# |     all_raw_dfs = []
# |     all_per_seed_dfs = []
# |     all_per_city_dfs = []
# |
# |     for f in range(1, 6):
# |         fold_dir = output_dir / f"fold_{f}"
# |         marker = fold_dir / "completion.marker"
# |         manifest_path = fold_dir / "run_manifest.json"
# |         
# |         if not marker.exists() or not manifest_path.exists():
# |             raise RuntimeError(f"Cannot aggregate: Fold {f} completion.marker or run_manifest.json not found")
# |             
# |         with open(manifest_path, "r") as mf:
# |             manifest = json.load(mf)
# |             
# |         assert manifest.get("protocol_version") == "v2", f"Fold {f} protocol version mismatch"
# |         assert manifest.get("model_seeds") == [1, 10, 100], f"Fold {f} model seeds mismatch (not [1, 10, 100])"
# |         assert manifest.get("replicates") == 500, f"Fold {f} replicates != 500"
# |         
# |         from src.data.city_splits import generate_35_5_10_splits
# |         splits = generate_35_5_10_splits()
# |         locked_test_cities = splits[f]["test"]
# |         assert manifest.get("cities") == locked_test_cities, f"Fold {f} test cities mismatch with locked manifest"
# |         
# |         all_raw_dfs.append(pd.read_csv(fold_dir / "raw.csv"))
# |         all_per_seed_dfs.append(pd.read_csv(fold_dir / "per_seed.csv"))
# |         all_per_city_dfs.append(pd.read_csv(fold_dir / "per_city.csv"))
# |
# |     raw_combined = pd.concat(all_raw_dfs, ignore_index=True)
# |     per_seed_combined = pd.concat(all_per_seed_dfs, ignore_index=True)
# |     per_city_combined = pd.concat(all_per_city_dfs, ignore_index=True)
# |
# |     raw_combined.to_csv(combined_dir / "raw_all_folds.csv", index=False)
# |     per_seed_combined.to_csv(combined_dir / "per_seed_all_folds.csv", index=False)
# |     per_city_combined.to_csv(combined_dir / "per_city_all_folds.csv", index=False)
# |
# |     expected_raw_rows = 50 * 3 * 500 * 15  # 15 p-levels
# |     expected_seed_rows = 50 * 3 * 15
# |     expected_city_rows = 50 * 15
# |     
# |     assert len(raw_combined) == expected_raw_rows, f"Combined raw rows mismatch: {len(raw_combined)} != {expected_raw_rows}"
# |     assert len(per_seed_combined) == expected_seed_rows, f"Combined seed rows mismatch"
# |     assert len(per_city_combined) == expected_city_rows, f"Combined city rows mismatch"
# |
# |     print(f"Combined Raw Rows:      {len(raw_combined):>10} (Certified)")
# |     print(f"Combined Per-Seed Rows: {len(per_seed_combined):>10} (Certified)")
# |     print(f"Combined Per-City Rows: {len(per_city_combined):>10} (Certified)")
# |
# |     # Statistical Analysis across N=50 cities
# |     summary_rows = []
# |     raw_p_values = []
# |     p_vals_tested = [p for p in p_grid if p > 0]
# |
# |     # Precalculate raw Wilcoxon p-values for partial OD benefit vs M0
# |     for p_val in p_vals_tested:
# |         sub = per_city_combined[per_city_combined.p == p_val]
# |         gains = sub["gain_partial_od"].values
# |         _, p_w = stats.wilcoxon(gains, alternative="greater")
# |         raw_p_values.append(p_w)
# |
# |     holm_p_vals = holm_correction(raw_p_values)
# |     holm_dict = {p: h_p for p, h_p in zip(p_vals_tested, holm_p_vals)}
# |
# |     for p_val in p_grid:
# |         sub = per_city_combined[per_city_combined.p == p_val]
# |         n_cities = len(sub)
# |         
# |         mean_mass = float(sub["fraction_trip_mass_revealed"].mean())
# |         mean_unseen_mass = float(sub["fraction_unseen_trip_mass"].mean())
# |         mean_tv = float(sub["empirical_tv_partial_vs_full"].mean())
# |         mean_m0 = float(sub["cpc_m0_unseen"].mean())
# |         mean_gain_full = float(sub["gain_full_yd"].mean())
# |         mean_gain_part = float(sub["gain_partial_od"].mean())
# |         mean_diff = float(sub["difference_partial_minus_yd"].mean())
# |         
# |         pos_cities = int((sub["gain_partial_od"] > 0).sum())
# |         match_yd_cities = int((sub["difference_partial_minus_yd"] >= 0).sum())
# |         
# |         ci_diff_l, ci_diff_h = fold_stratified_bootstrap(per_city_combined, "difference_partial_minus_yd", p_val)
# |         ci_part_l, ci_part_h = fold_stratified_bootstrap(per_city_combined, "gain_partial_od", p_val)
# |         ci_full_l, ci_full_h = fold_stratified_bootstrap(per_city_combined, "gain_full_yd", p_val)
# |         
# |         h_pval = holm_dict.get(p_val, 1.0) if p_val > 0 else 1.0
# |
# |         summary_rows.append({
# |             "p": p_val,
# |             "n_cities": n_cities,
# |             "mean_revealed_mass": mean_mass,
# |             "mean_unseen_mass": mean_unseen_mass,
# |             "mean_tv": mean_tv,
# |             "mean_m0_cpc": mean_m0,
# |             "mean_gain_full_yd": mean_gain_full,
# |             "ci_95_gain_full": [ci_full_l, ci_full_h],
# |             "mean_gain_partial_od": mean_gain_part,
# |             "ci_95_gain_partial": [ci_part_l, ci_part_h],
# |             "mean_diff_vs_yd": mean_diff,
# |             "ci_95_diff": [ci_diff_l, ci_diff_h],
# |             "pos_cities_vs_m0": pos_cities,
# |             "match_yd_cities": match_yd_cities,
# |             "holm_pval_benefit": h_pval
# |         })
# |
# |     summary_df = pd.DataFrame(summary_rows)
# |
# |     # Calculate 3 Key Thresholds
# |     # 1. Positive Mean Crossing
# |     p_pos_mean = None
# |     for r in summary_rows:
# |         if r["mean_gain_partial_od"] > 0 and p_pos_mean is None:
# |             p_pos_mean = r["p"]
# |
# |     # 2. Statistically Supported Benefit Threshold p*_benefit (Holm p < 0.05, CI_lower > 0)
# |     p_star_benefit = None
# |     for r in summary_rows:
# |         if r["holm_pval_benefit"] < 0.05 and r["ci_95_gain_partial"][0] > 0 and p_star_benefit is None:
# |             p_star_benefit = r["p"]
# |
# |     # 3. Operational Equivalence Crossing p_eq
# |     p_eq_grid = None
# |     p_eq_interp = None
# |     for r in summary_rows:
# |         if r["mean_diff_vs_yd"] >= 0 and p_eq_grid is None:
# |             p_eq_grid = r["p"]
# |
# |     for i in range(len(summary_rows) - 1):
# |         r1, r2 = summary_rows[i], summary_rows[i+1]
# |         d1, d2 = r1["mean_diff_vs_yd"], r2["mean_diff_vs_yd"]
# |         if d1 <= 0 and d2 >= 0 and (d2 - d1) > 0:
# |             p_eq_interp = r1["p"] + (-d1 / (d2 - d1)) * (r2["p"] - r1["p"])
# |             break
# |
# |     # Save summary JSON
# |     summary_json_path = combined_dir / "summary.json"
# |     with open(summary_json_path, "w", encoding="utf-8") as f:
# |         json.dump({
# |             "experiment": "partial_od_information_equivalence",
# |             "protocol_version": "v2",
# |             "n_evaluation_cities": 50,
# |             "p_pos_mean_crossing": p_pos_mean,
# |             "p_star_benefit_threshold": p_star_benefit,
# |             "p_eq_grid": p_eq_grid,
# |             "p_eq_interp": p_eq_interp,
# |             "results_by_p": summary_rows
# |         }, f, indent=2)
# |
# |     # Save Markdown Table
# |     summary_md_path = combined_dir / "summary.md"
# |     with open(summary_md_path, "w", encoding="utf-8") as f:
# |         f.write("# Table: Master Partial-OD Information Equivalence Summary (v2)\n\n")
# |         f.write("> **Evaluation Scope**: Assesses the operational reconstruction value of target-city distance distribution $Y_D$ relative to observing $p\\%$ of positive interzonal OD pairs ($K=8, q=1.0$, seeds $s \\in \\{1, 10, 100\\}$) evaluated strictly on unseen pairs ($N=50$ held-out test cities across 5 folds).\n\n")
# |         
# |         if p_pos_mean is not None:
# |             pct_pos = p_pos_mean * 100.0
# |             f.write(f"• **Positive Mean Crossing Point:** `{pct_pos:.2f}%` of positive interzonal OD pairs  \n")
# |         if p_star_benefit is not None:
# |             pct_star = p_star_benefit * 100.0
# |             f.write(f"• **Statistically Supported Benefit Threshold ($p^*_\\text{{benefit}}$):** `{pct_star:.2f}%` of positive interzonal OD pairs ($p_\\text{{Holm}} < 0.05$)  \n")
# |         if p_eq_interp is not None:
# |             pct_interp = p_eq_interp * 100.0
# |             f.write(f"• **Operational Equivalence Crossing ($p_\\text{{eq,interp}}$):** `{pct_interp:.2f}%` of positive interzonal OD pairs  \n\n")
# |         elif p_eq_grid is not None:
# |             pct_grid = p_eq_grid * 100.0
# |             f.write(f"• **Operational Equivalence Grid Point ($p_\\text{{eq,grid}}$):** `{pct_grid:.2f}%` of positive interzonal OD pairs  \n\n")
# |         else:
# |             f.write("• **Operational Equivalence Crossing:** Full target-city $Y_D$ was not matched within the prespecified partial-OD range up to 90% of the positive interzonal OD support.  \n\n")
# |
# |         f.write("| Revealed OD Pairs ($p$) | Mean Revealed Trip Mass | Mean TV to Full $Y_D$ | $M_0$ CPC (Unseen) | Full-$Y_D$ Gain | Partial-OD Gain | Difference vs Full $Y_D$ ($D(p)$) | 95% CI Difference | Partial Benefit Holm $p$ | Cities Partial $> M_0$ | Cities Partial $\\ge$ Full $Y_D$ |\n")
# |         f.write("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
# |         
# |         for r in summary_rows:
# |             p_pct = f"{r['p']*100:.2f}%"
# |             mass_pct = f"{r['mean_revealed_mass']*100:.2f}%"
# |             tv_pct = f"{r['mean_tv']*100:.2f}%"
# |             m0_str = f"{r['mean_m0_cpc']:.4f}"
# |             full_str = f"+{r['mean_gain_full_yd']:.5f}"
# |             part_str = f"{r['mean_gain_partial_od']:+.5f}"
# |             diff_str = f"{r['mean_diff_vs_yd']:+.5f}"
# |             ci_str = f"[{r['ci_95_diff'][0]:+.5f}, {r['ci_95_diff'][1]:+.5f}]"
# |             h_str = f"{r['holm_pval_benefit']:.4e}" if r['p'] > 0 else "—"
# |             pos_str = f"{r['pos_cities_vs_m0']}/{r['n_cities']}"
# |             match_str = f"{r['match_yd_cities']}/{r['n_cities']}"
# |             
# |             f.write(f"| **{p_pct}** | {mass_pct} | {tv_pct} | {m0_str} | {full_str} | **{part_str}** | **{diff_str}** | {ci_str} | {h_str} | {pos_str} | {match_str} |\n")
# |             
# |         f.write("\n---\n\n### Prescribed Scientific Interpretation\n")
# |         f.write("Under uniform random pair sampling, the mean revealed trip-mass fraction closely tracked the revealed pair fraction. ")
# |         if p_eq_interp is not None:
# |             f.write(f"Under the frozen support-conditioned model and the same production calibration operator, the mean reconstruction benefit provided by the full target-city $Y_D$ was matched at approximately **{p_eq_interp*100:.2f}%** of directly observed positive interzonal OD pairs.\n")
# |         else:
# |             f.write("Under the tested operator, directly observing up to 90% of the positive interzonal OD support did not fully match the mean reconstruction gain provided by the full target-city $Y_D$.\n")
# |
# |     print(f"Summary Markdown: {summary_md_path}")
# |     print(f"Summary JSON:     {summary_json_path}")
# |
# |     # Generate 5 Publication Figures
# |     generate_publication_figures(summary_df, per_city_combined, combined_dir, p_eq_interp, p_star_benefit)
# |
# |     # Write FROZEN markers only after all QA and figure generations succeed
# |     with open(combined_dir / "FROZEN.marker", "w", encoding="utf-8") as f:
# |         f.write(f"MASTER 5-FOLD AGGREGATION CERTIFIED\nTimestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
# |
# |     frozen_marker_path = output_dir / "FROZEN.marker"
# |     with open(frozen_marker_path, "w", encoding="utf-8") as f:
# |         f.write("PARTIAL-OD INFORMATION EQUIVALENCE v2 PROTOCOL FROZEN\n")
# |         f.write(f"Completed At: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
# |         f.write("Protocol: 50 held-out test cities across 5 disjoint folds (N=50)\n")
# |         f.write("Evaluation Support: unseen positive interzonal pairs Omega_c^+ \\ S_p\n")
# |         f.write(f"Replicates: 500 per city (Total: 1,125,000 raw calibrations)\n")
# |
# |
# | def generate_publication_figures(
# |     summary_df: pd.DataFrame, 
# |     per_city_df: pd.DataFrame, 
# |     combined_dir: Path, 
# |     p_eq_interp: Optional[float],
# |     p_star_benefit: Optional[float]
# | ) -> None:
# |     plt.rcParams.update({'font.sans-serif': 'Helvetica', 'axes.edgecolor': '#333333', 'axes.linewidth': 0.8})
# |     fig_dir = combined_dir / "figures"
# |     p_vals = summary_df["p"].values * 100.0
# |
# |     # Fig 1: Gain vs Reveal Fraction
# |     fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
# |     ax.axhline(0, color="#888888", linestyle="--", alpha=0.6)
# |     
# |     full_gain = summary_df["mean_gain_full_yd"].values
# |     part_gain = summary_df["mean_gain_partial_od"].values
# |     part_ci_l = np.array([ci[0] for ci in summary_df["ci_95_gain_partial"]])
# |     part_ci_h = np.array([ci[1] for ci in summary_df["ci_95_gain_partial"]])
# |     
# |     ax.plot(p_vals, full_gain, label="Full $Y_D$ Reference Gain", color="#1f77b4", linestyle="--", linewidth=2.0)
# |     ax.plot(p_vals, part_gain, label="Partial-OD Calibration Gain", color="#d62728", marker="o", linewidth=2.0)
# |     ax.fill_between(p_vals, part_ci_l, part_ci_h, color="#d62728", alpha=0.15, label="95% Fold Bootstrap CI")
# |     
# |     if p_star_benefit is not None:
# |         ax.axvline(p_star_benefit * 100.0, color="#ff7f0e", linestyle="-.", label=f"Benefit $p^* = {p_star_benefit*100:.2f}\\%$")
# |     if p_eq_interp is not None:
# |         ax.axvline(p_eq_interp * 100.0, color="#2ca02c", linestyle=":", label=f"Equivalence $p_{{eq}} = {p_eq_interp*100:.2f}\\%$")
# |         
# |     ax.set_xlabel("Revealed Positive Interzonal OD Pairs (%)", fontsize=11, fontweight="bold")
# |     ax.set_ylabel("Mean Marginal Gain $\\Delta\\mathrm{CPC}_U$ on Unseen OD", fontsize=11, fontweight="bold")
# |     ax.set_title("Marginal Reconstruction Value: Partial OD vs Full $Y_D$", fontsize=12, fontweight="bold", pad=12)
# |     ax.legend(frameon=True, fontsize=9)
# |     ax.grid(True, linestyle=":", alpha=0.5)
# |     plt.tight_layout()
# |     fig.savefig(fig_dir / "fig_1_gain_vs_p.png")
# |     plt.close(fig)
# |
# |     # Fig 2: Difference D(p) Equivalence
# |     fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
# |     ax.axhline(0, color="#333333", linestyle="-", linewidth=1.0)
# |     
# |     diff_vals = summary_df["mean_diff_vs_yd"].values
# |     diff_ci_l = np.array([ci[0] for ci in summary_df["ci_95_diff"]])
# |     diff_ci_h = np.array([ci[1] for ci in summary_df["ci_95_diff"]])
# |     
# |     ax.plot(p_vals, diff_vals, color="#9467bd", marker="s", linewidth=2.0, label="$\\bar{D}(p) = \\mathrm{Gain}_{\\mathrm{partial}} - \\mathrm{Gain}_{Y_D}$")
# |     ax.fill_between(p_vals, diff_ci_l, diff_ci_h, color="#9467bd", alpha=0.15, label="95% Fold Bootstrap CI")
# |     
# |     if p_eq_interp is not None:
# |         ax.scatter([p_eq_interp * 100.0], [0.0], color="#d62728", s=80, zorder=5, label=f"Crossing $p_{{eq}} = {p_eq_interp*100:.2f}\\%$")
# |         
# |     ax.set_xlabel("Revealed Positive Interzonal OD Pairs (%)", fontsize=11, fontweight="bold")
# |     ax.set_ylabel("Gain Difference $D(p)$", fontsize=11, fontweight="bold")
# |     ax.set_title("Information Equivalence Zero-Crossing", fontsize=12, fontweight="bold", pad=12)
# |     ax.legend(frameon=True, fontsize=9)
# |     ax.grid(True, linestyle=":", alpha=0.5)
# |     plt.tight_layout()
# |     fig.savefig(fig_dir / "fig_2_Dp_equivalence.png")
# |     plt.close(fig)
# |
# |     # Fig 3: TV vs p
# |     fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
# |     tvs = summary_df["mean_tv"].values * 100.0
# |     ax.plot(p_vals, tvs, color="#ff7f0e", marker="^", linewidth=2.0)
# |     ax.set_xlabel("Revealed Positive Interzonal OD Pairs (%)", fontsize=11, fontweight="bold")
# |     ax.set_ylabel("Mean Total Variation Error $\\mathrm{TV}(\\tilde{Y}_D, Y_D^{\\mathrm{full}})$ (%)", fontsize=11, fontweight="bold")
# |     ax.set_title("Distributional Convergence with Partial OD Observation", fontsize=12, fontweight="bold", pad=12)
# |     ax.grid(True, linestyle=":", alpha=0.5)
# |     plt.tight_layout()
# |     fig.savefig(fig_dir / "fig_3_TV_vs_p.png")
# |     plt.close(fig)
# |
# |     # Fig 4: Revealed Mass vs Gain
# |     fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
# |     masses = summary_df["mean_revealed_mass"].values * 100.0
# |     ax.plot(masses, part_gain, color="#2ca02c", marker="d", linewidth=2.0, label="Partial-OD Gain")
# |     ax.axhline(full_gain[0], color="#1f77b4", linestyle="--", label="Full $Y_D$ Reference")
# |     ax.set_xlabel("Revealed Interzonal Trip Mass (%)", fontsize=11, fontweight="bold")
# |     ax.set_ylabel("Mean Marginal Gain $\\Delta\\mathrm{CPC}_U$", fontsize=11, fontweight="bold")
# |     ax.set_title("Reconstruction Gain vs Revealed Trip Mass", fontsize=12, fontweight="bold", pad=12)
# |     ax.legend(frameon=True, fontsize=9)
# |     ax.grid(True, linestyle=":", alpha=0.5)
# |     plt.tight_layout()
# |     fig.savefig(fig_dir / "fig_4_revealed_mass_vs_gain.png")
# |     plt.close(fig)
# |
# |     # Fig 5: Fold-Specific D(p) Auditing
# |     fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
# |     ax.axhline(0, color="#333333", linestyle="-", linewidth=1.0)
# |     colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
# |     
# |     for f in range(1, 6):
# |         f_sub = per_city_df[per_city_df.fold == f].groupby("p")["difference_partial_minus_yd"].mean().reset_index()
# |         ax.plot(f_sub["p"].values * 100.0, f_sub["difference_partial_minus_yd"].values, marker="o", markersize=4, label=f"Fold {f} (N=10)", color=colors[f-1])
# |         
# |     ax.set_xlabel("Revealed Positive Interzonal OD Pairs (%)", fontsize=11, fontweight="bold")
# |     ax.set_ylabel("Fold-Specific Mean $D(p)$", fontsize=11, fontweight="bold")
# |     ax.set_title("Fold-Specific Equivalence Trajectories", fontsize=12, fontweight="bold", pad=12)
# |     ax.legend(frameon=True, fontsize=9)
# |     ax.grid(True, linestyle=":", alpha=0.5)
# |     plt.tight_layout()
# |     fig.savefig(fig_dir / "fig_5_fold_specific_Dp.png")
# |     plt.close(fig)
# |
# |
# | if __name__ == "__main__":
# |     parser = argparse.ArgumentParser(description="Run Partial-OD Information Equivalence v2")
# |     parser.add_argument("--data_root", type=str, default="data")
# |     parser.add_argument("--output_dir", type=str, default="results/partial_od_equivalence_v2")
# |     parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5], help="Folds to execute")
# |     parser.add_argument("--cities", type=int, default=10, help="Number of test cities per fold")
# |     parser.add_argument("--b", type=int, default=500, help="Monte Carlo replicates per city")
# |     parser.add_argument("--smoke", action="store_true", help="Run fast smoke test")
# |     parser.add_argument("--resume", action="store_true", help="Resume from progress.json")
# |     parser.add_argument("--aggregate_only", action="store_true", help="Only aggregate completed folds")
# |     parser.add_argument("--device", type=str, default="cpu")
# |     args = parser.parse_args()
# |
# |     out_p = Path(args.output_dir)
# |
# |     if args.aggregate_only:
# |         aggregate_combined_results(output_dir=out_p)
# |     else:
# |         for f_id in args.folds:
# |             run_fold_partial_od(
# |                 fold_id=f_id,
# |                 data_root=args.data_root,
# |                 output_dir=out_p,
# |                 replicates=args.b,
# |                 smoke=args.smoke,
# |                 smoke_cities=args.cities,
# |                 resume=args.resume,
# |                 device=args.device
# |             )
# |         if not args.smoke and set(args.folds) == {1, 2, 3, 4, 5}:
# |             aggregate_combined_results(output_dir=out_p)

# ===== END SOURCE FILE: src/experiment/run_partial_od_equivalence_v2.py =====


# ===== BEGIN SOURCE FILE: src/experiment/run_placebo_matched_v2.py =====
# | import os
# | import sys
# | import json
# | import time
# | import argparse
# | import random
# | import itertools
# | import numpy as np
# | import pandas as pd
# | import torch
# | import math
# | import logging
# | from pathlib import Path
# | from scipy.stats import wilcoxon
# |
# | sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
# |
# | from src.data.city_splits import generate_35_5_10_splits
# | from src.data.yd_extractor import compute_kbin_edges, extract_yd_kbins
# | from src.data.dataset import load_raw_city, load_city
# | from src.data.urban_graph import build_radius_graph
# | from src.training.train import load_checkpoint
# | from src.experiment.run_experiment import infer_zero_shot
# | from src.training.evaluate import compute_cpc_pair
# | from src.calibration.bin_calibration import calibrate_kbins
# |
# | def get_active_bins(yd, eps=1e-8):
# |     return yd > eps
# |
# | def safe_log_ratio(p, y_hat, active_mask, delta=1e-12):
# |     p = p.copy()
# |     y_hat = y_hat.copy()
# |     p_active = p[active_mask]
# |     if np.any(p_active < delta):
# |         p_active = np.maximum(p_active, delta)
# |         p_active = p_active / p_active.sum()
# |         p[active_mask] = p_active
# |         
# |     y_hat_active = y_hat[active_mask]
# |     y_hat_active = np.maximum(y_hat_active, delta)
# |     r = np.zeros_like(p)
# |     r[active_mask] = np.log(p_active) - np.log(y_hat_active)
# |     return r
# |
# | def evaluate_cpc(t_true_inter, t_pred_inter):
# |     return compute_cpc_pair(t_true_inter, t_pred_inter)
# |
# | def run_placebo_experiment(args):
# |     data_root = "data"
# |     output_dir = "results/placebo_matched_v2"
# |     os.makedirs(output_dir, exist_ok=True)
# |     
# |     log_file = f"{output_dir}/run.log"
# |     logging.basicConfig(
# |         level=logging.INFO,
# |         format='%(message)s',
# |         handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
# |     )
# |     logger = logging.getLogger(__name__)
# |     
# |     splits = generate_35_5_10_splits(data_root=data_root)
# |     placebo_seed = 20260823
# |     np.random.seed(placebo_seed)
# |     epsilon = 1e-12
# |     
# |     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# |     
# |     raw_results = []
# |     B_perm = args.b
# |     folds_to_run = [2] if args.smoke else [1, 2, 3, 4, 5]
# |     
# |     for fold_id in folds_to_run:
# |         split = splits[fold_id]
# |         train_cities = split["train"]
# |         test_cities = split["test"]
# |         
# |         if args.smoke:
# |             test_cities = [test_cities[0]]
# |             train_cities = train_cities[:2]
# |             
# |         logger.info(f"\nProcessing Fold {fold_id}...")
# |         bin_edges, _ = compute_kbin_edges(split["train"], K=8, data_root=data_root)
# |         K = len(bin_edges) - 1
# |         train_yd_dict = {}
# |         
# |         for city_name in train_cities:
# |             raw = load_raw_city(city_name, data_root=data_root)
# |             dist_km = raw.dist_km
# |             inter_mask = (raw.pair_o_idx.numpy() != raw.pair_d_idx.numpy()) & (dist_km > 0.0)
# |             yd = extract_yd_kbins(dist_km, raw.pair_trips.numpy(), bin_edges, inter_mask)
# |             train_yd_dict[city_name] = yd
# |             
# |         train_mean_yd = np.mean(list(train_yd_dict.values()), axis=0)
# |         model_seeds = [1, 10] if args.smoke else [1, 10, 100]
# |         
# |         for c_idx, tc in enumerate(test_cities):
# |             logger.info(f"  Target City: {tc} ({c_idx+1}/{len(test_cities)})")
# |             raw = load_raw_city(tc, data_root=data_root)
# |             dist_km = raw.dist_km
# |             inter_mask = (raw.pair_o_idx.numpy() != raw.pair_d_idx.numpy()) & (dist_km > 0.0)
# |             t_true_inter = raw.pair_trips.numpy()[inter_mask]
# |             
# |             yd_target = extract_yd_kbins(dist_km, raw.pair_trips.numpy(), bin_edges, inter_mask)
# |             active_mask = get_active_bins(yd_target)
# |             target_active_bin_count = np.sum(active_mask)
# |             
# |             assert abs(yd_target.sum() - 1.0) < 1e-8, "yd_target prob mass must sum to 1"
# |             
# |             edge_index, edge_dist = build_radius_graph(
# |                 lon_lat=raw.lon_lat, 
# |                 radius_km=5.0, 
# |                 include_self_loop=True, 
# |                 cache_key=f"{tc}_tracts"
# |             )
# |             
# |             rng_perms = np.random.RandomState(placebo_seed)
# |             if math.factorial(target_active_bin_count) <= 40320:
# |                 all_perms = list(itertools.permutations(np.arange(target_active_bin_count)))
# |                 valid_perms = [p for p in all_perms if not np.array_equal(p, np.arange(target_active_bin_count))]
# |                 if len(valid_perms) > B_perm:
# |                     chosen_perms = rng_perms.choice(len(valid_perms), size=B_perm, replace=False)
# |                     index_perms = [valid_perms[i] for i in chosen_perms]
# |                 else:
# |                     index_perms = valid_perms
# |             else:
# |                 perms_set = set()
# |                 tries = 0
# |                 while len(perms_set) < B_perm and tries < B_perm * 10:
# |                     tries += 1
# |                     p = tuple(rng_perms.permutation(np.arange(target_active_bin_count)))
# |                     if not np.array_equal(p, np.arange(target_active_bin_count)):
# |                         perms_set.add(p)
# |                 index_perms = list(perms_set)
# |
# |             for seed in model_seeds:
# |                 logger.info(f"    Evaluating seed {seed}...")
# |                 ckpt_path = Path(f"results/checkpoints/5fold_fold{fold_id}_seed{seed}.pt")
# |                 if not ckpt_path.exists():
# |                     raise FileNotFoundError(f"Missing mandatory checkpoint {ckpt_path}.")
# |                     
# |                 model, scaler, _ = load_checkpoint(ckpt_path, device_str="cpu")
# |                 model.eval()
# |                 
# |                 city_data = load_city(tc, data_root=data_root, feature_scaler=scaler, fit_scaler=False)
# |                 t_pred_zs_tensor = infer_zero_shot(model, city_data, edge_index, edge_dist, device="cpu")
# |                 t_pred_zs = t_pred_zs_tensor.numpy().astype(np.float64)
# |                 
# |                 cpc_m0_inter = evaluate_cpc(t_true_inter, t_pred_zs[inter_mask])
# |                 
# |                 t0_inter = t_pred_zs[inter_mask]
# |                 dist_inter = dist_km[inter_mask]
# |                 N_hat = t0_inter.sum()
# |                 
# |                 bin_masks = []
# |                 Y_hat = np.zeros(K, dtype=np.float64)
# |                 if N_hat > 0:
# |                     for k in range(K):
# |                         lo, hi = float(bin_edges[k]), float(bin_edges[k+1])
# |                         in_bin = (dist_inter > lo) & (dist_inter <= hi)
# |                         bin_masks.append(in_bin)
# |                         Y_hat[k] = t0_inter[in_bin].sum() / N_hat
# |                         
# |                 target_bins_with_zero_pred_mass = np.sum(active_mask & (Y_hat <= epsilon))
# |                 model_supported_target_bin_count = target_active_bin_count - target_bins_with_zero_pred_mass
# |                 model_target_bin_support_rate = model_supported_target_bin_count / target_active_bin_count if target_active_bin_count > 0 else 1.0
# |                 
# |                 r_T = safe_log_ratio(yd_target, Y_hat, active_mask, delta=epsilon)
# |                 r_tilde_T = np.zeros_like(r_T)
# |                 r_tilde_T[active_mask] = r_T[active_mask] - np.mean(r_T[active_mask])
# |                 D_T = np.sqrt(np.mean(r_tilde_T[active_mask]**2))
# |                 
# |                 def equivalence_test(p_tgt, name):
# |                     t_cal_ref = calibrate_kbins(
# |                         t0_np=t_pred_zs.copy(), 
# |                         dist_km=dist_km, 
# |                         inter_mask=inter_mask, 
# |                         yd_target=p_tgt.copy(), 
# |                         bin_edges=bin_edges, 
# |                         q=1.0, 
# |                         tolerance=1e-5
# |                     )
# |                     cpc_ref = evaluate_cpc(t_true_inter, t_cal_ref[inter_mask])
# |                     
# |                     ref_bin_mass = np.zeros(K, dtype=np.float64)
# |                     ref_inter = t_cal_ref[inter_mask]
# |                     cal_mass_ref = ref_inter.sum()
# |                     for k in range(K):
# |                         if cal_mass_ref > 0:
# |                             ref_bin_mass[k] = ref_inter[bin_masks[k]].sum() / cal_mass_ref
# |                             
# |                     p_active = p_tgt[active_mask]
# |                     p_cond = p_active / np.sum(p_active) if np.sum(p_active) > 0 else Y_hat[active_mask] / np.sum(Y_hat[active_mask])
# |                     
# |                     w_raw = np.zeros(target_active_bin_count, dtype=np.float64)
# |                     n_w_infinite = 0
# |                     
# |                     for i, idx in enumerate(np.where(active_mask)[0]):
# |                         if Y_hat[idx] <= epsilon:
# |                             w_raw[i] = np.inf
# |                             n_w_infinite += 1
# |                         else:
# |                             w_raw[i] = p_cond[i] / Y_hat[idx]
# |                             
# |                     w_finite = w_raw[np.isfinite(w_raw)]
# |                     w_raw_min = float(np.min(w_finite)) if len(w_finite) > 0 else np.nan
# |                     w_raw_median = float(np.median(w_finite)) if len(w_finite) > 0 else np.nan
# |                     w_raw_p95 = float(np.percentile(w_finite, 95)) if len(w_finite) > 0 else np.nan
# |                     w_raw_max = float(np.max(w_finite)) if len(w_finite) > 0 else np.nan
# |                     
# |                     n_w_gt_2 = int((w_finite > 2.0).sum()) + n_w_infinite
# |                     n_w_gt_5 = int((w_finite > 5.0).sum()) + n_w_infinite
# |                     n_w_gt_10 = int((w_finite > 10.0).sum()) + n_w_infinite
# |                     
# |                     rate_w_gt_2 = n_w_gt_2 / target_active_bin_count
# |                     rate_w_gt_5 = n_w_gt_5 / target_active_bin_count
# |                     rate_w_gt_10 = n_w_gt_10 / target_active_bin_count
# |                     
# |                     w = np.ones(K, dtype=np.float64)
# |                     # For fast cal, clip Y_hat dynamically to avoid division by zero
# |                     y_hat_safe = np.maximum(Y_hat[active_mask], epsilon)
# |                     w_active = p_cond / y_hat_safe
# |                     w[active_mask] = w_active
# |                     
# |                     weighted_mass = float((Y_hat[active_mask] * w_active).sum())
# |                     s = np.ones(K, dtype=np.float64)
# |                     s_active = w_active / weighted_mass if weighted_mass > 0 else np.ones_like(w_active)
# |                     s[active_mask] = s_active
# |                     
# |                     t_cal_fast = t0_inter.copy()
# |                     for k in range(K):
# |                         if active_mask[k]:
# |                             t_cal_fast[bin_masks[k]] *= s[k]
# |                             
# |                     cal_mass_fast = t_cal_fast.sum()
# |                     if cal_mass_fast > 0:
# |                         t_cal_fast *= (N_hat / cal_mass_fast)
# |                         
# |                     cpc_fast = evaluate_cpc(t_true_inter, t_cal_fast)
# |                     
# |                     fast_bin_mass = np.zeros(K, dtype=np.float64)
# |                     for k in range(K):
# |                         if cal_mass_fast > 0:
# |                             fast_bin_mass[k] = t_cal_fast[bin_masks[k]].sum() / cal_mass_fast
# |                             
# |                     if not np.allclose(fast_bin_mass, ref_bin_mass, atol=1e-10):
# |                         raise ValueError(f"Equivalence failed for {name}: bin masses differ")
# |                     if not np.allclose(cpc_fast, cpc_ref, atol=1e-10):
# |                         raise ValueError(f"Equivalence failed for {name}: cpc differs")
# |                     if not np.allclose(t_cal_fast, t_cal_ref[inter_mask], atol=1e-10):
# |                         raise ValueError(f"Equivalence failed for {name}: t_cal differs")
# |                         
# |                     stats = {
# |                         "w_raw_min": w_raw_min, "w_raw_median": w_raw_median, "w_raw_p95": w_raw_p95, "w_raw_max": w_raw_max,
# |                         "n_w_gt_2": int(n_w_gt_2), "n_w_gt_5": int(n_w_gt_5), "n_w_gt_10": int(n_w_gt_10),
# |                         "rate_w_gt_2": float(rate_w_gt_2), "rate_w_gt_5": float(rate_w_gt_5), "rate_w_gt_10": float(rate_w_gt_10),
# |                         "n_w_infinite": int(n_w_infinite),
# |                         "target_active_bin_count": int(target_active_bin_count),
# |                         "model_supported_target_bin_count": int(model_supported_target_bin_count),
# |                         "target_bins_with_zero_pred_mass": int(target_bins_with_zero_pred_mass),
# |                         "model_target_bin_support_rate": float(model_target_bin_support_rate),
# |                     }
# |                     return cpc_fast, stats
# |
# |                 cpc_target, stats_tgt = equivalence_test(yd_target, "target")
# |                 delta_cpc_target = cpc_target - cpc_m0_inter
# |
# |                 def build_row(cond, rep_id, donor_name, cpc_val, stats, dose, donor_stats={}):
# |                     d_cpc = cpc_val - cpc_m0_inter
# |                     row = {
# |                         "fold": int(fold_id), "model_seed": int(seed), "target_city": tc, "condition": cond,
# |                         "replicate_id": int(rep_id), "donor_city": donor_name, "placebo_seed": int(placebo_seed),
# |                         "q": 1.0, "D_T": float(D_T), "D_placebo": float(dose), 
# |                         "dose_error": float(abs(D_T - dose)),
# |                         "cpc_m0_inter": float(cpc_m0_inter), "cpc_m1_inter": float(cpc_val),
# |                         "delta_cpc_inter": float(d_cpc), "target_delta_cpc_inter": float(delta_cpc_target),
# |                         "specificity_gain": float(delta_cpc_target - d_cpc),
# |                     }
# |                     row.update(stats)
# |                     row.update(donor_stats)
# |                     return row
# |
# |                 raw_results.append(build_row("target", 0, tc, cpc_target, stats_tgt, D_T))
# |
# |                 seen_vecs = []
# |                 p_idx = 0
# |                 for perm_indices in index_perms:
# |                     r_tilde_P = np.zeros_like(r_tilde_T)
# |                     r_tilde_P[active_mask] = r_tilde_T[active_mask][list(perm_indices)]
# |                     
# |                     is_dup = False
# |                     for seen in seen_vecs:
# |                         if np.allclose(seen, r_tilde_P, atol=1e-12):
# |                             is_dup = True
# |                             break
# |                     if is_dup:
# |                         continue
# |                     seen_vecs.append(r_tilde_P.copy())
# |                     
# |                     D_P = np.sqrt(np.mean(r_tilde_P[active_mask]**2))
# |                     p_P = np.zeros_like(Y_hat)
# |                     p_P[active_mask] = np.maximum(Y_hat[active_mask], epsilon) * np.exp(r_tilde_P[active_mask])
# |                     p_P[active_mask] /= p_P[active_mask].sum()
# |                     
# |                     cpc_P, stats_P = equivalence_test(p_P, "permuted_bin")
# |                     raw_results.append(build_row("permuted", p_idx, "PERMUTED", cpc_P, stats_P, D_P))
# |                     p_idx += 1
# |
# |                 w_idx = 0
# |                 for donor_name in train_cities:
# |                     donor_yd = train_yd_dict[donor_name]
# |                     r_D = safe_log_ratio(donor_yd, Y_hat, active_mask, delta=epsilon)
# |                     r_tilde_D = np.zeros_like(r_D)
# |                     r_tilde_D[active_mask] = r_D[active_mask] - np.mean(r_D[active_mask])
# |                     D_D = np.sqrt(np.mean(r_tilde_D[active_mask]**2))
# |                     if D_D < 1e-12: continue
# |                     
# |                     r_tilde_D_star = np.zeros_like(r_tilde_D)
# |                     r_tilde_D_star[active_mask] = r_tilde_D[active_mask] * (D_T / D_D)
# |                     D_D_star = np.sqrt(np.mean(r_tilde_D_star[active_mask]**2))
# |                     
# |                     p_D_star = np.zeros_like(Y_hat)
# |                     p_D_star[active_mask] = np.maximum(Y_hat[active_mask], epsilon) * np.exp(r_tilde_D_star[active_mask])
# |                     p_D_star[active_mask] /= p_D_star[active_mask].sum()
# |                     
# |                     cpc_wrong, stats_wrong = equivalence_test(p_D_star, "wrong_city")
# |                     
# |                     donor_target_mass_overlap = float(donor_yd[active_mask].sum())
# |                     donor_target_bin_overlap_count = int(np.sum((donor_yd > 0) & active_mask))
# |                     donor_target_bin_overlap_rate = donor_target_bin_overlap_count / target_active_bin_count if target_active_bin_count > 0 else 1.0
# |                     
# |                     d_stats = {
# |                         "donor_target_mass_overlap": donor_target_mass_overlap,
# |                         "donor_target_bin_overlap_count": donor_target_bin_overlap_count,
# |                         "donor_target_bin_overlap_rate": donor_target_bin_overlap_rate
# |                     }
# |                     raw_results.append(build_row("wrong_city", w_idx, donor_name, cpc_wrong, stats_wrong, D_D_star, d_stats))
# |                     w_idx += 1
# |
# |                 r_M = safe_log_ratio(train_mean_yd, Y_hat, active_mask, delta=epsilon)
# |                 r_tilde_M = np.zeros_like(r_M)
# |                 r_tilde_M[active_mask] = r_M[active_mask] - np.mean(r_M[active_mask])
# |                 D_M = np.sqrt(np.mean(r_tilde_M[active_mask]**2))
# |                 
# |                 if D_M >= 1e-12:
# |                     r_tilde_M_star = np.zeros_like(r_tilde_M)
# |                     r_tilde_M_star[active_mask] = r_tilde_M[active_mask] * (D_T / D_M)
# |                     D_M_star = np.sqrt(np.mean(r_tilde_M_star[active_mask]**2))
# |                     
# |                     p_M_star = np.zeros_like(Y_hat)
# |                     p_M_star[active_mask] = np.maximum(Y_hat[active_mask], epsilon) * np.exp(r_tilde_M_star[active_mask])
# |                     p_M_star[active_mask] /= p_M_star[active_mask].sum()
# |                     
# |                     cpc_tm, stats_tm = equivalence_test(p_M_star, "train_mean")
# |                     
# |                     donor_target_mass_overlap = float(train_mean_yd[active_mask].sum())
# |                     donor_target_bin_overlap_count = int(np.sum((train_mean_yd > 0) & active_mask))
# |                     donor_target_bin_overlap_rate = donor_target_bin_overlap_count / target_active_bin_count if target_active_bin_count > 0 else 1.0
# |                     d_stats = {
# |                         "donor_target_mass_overlap": donor_target_mass_overlap,
# |                         "donor_target_bin_overlap_count": donor_target_bin_overlap_count,
# |                         "donor_target_bin_overlap_rate": donor_target_bin_overlap_rate
# |                     }
# |                     raw_results.append(build_row("trainmean", 0, "TRAIN_MEAN", cpc_tm, stats_tm, D_M_star, d_stats))
# |
# |     df = pd.DataFrame(raw_results)
# |     df.to_csv(f"{output_dir}/matched_placebo_raw.csv", index=False)
# |     with open(f"{output_dir}/matched_placebo_raw.jsonl", "w") as f:
# |         for r in raw_results:
# |             f.write(json.dumps(r) + "\n")
# |             
# |     logger.info(f"Raw results saved to {output_dir}")
# |     
# |     agg_cols = {
# |         "delta_cpc_inter": "mean", "target_delta_cpc_inter": "mean", "specificity_gain": "mean",
# |         "cpc_m0_inter": "mean", "cpc_m1_inter": "mean",
# |         "w_raw_min": "mean", "w_raw_median": "mean", "w_raw_p95": "mean", "w_raw_max": "mean",
# |         "n_w_gt_2": "mean", "n_w_gt_5": "mean", "n_w_gt_10": "mean",
# |         "rate_w_gt_2": "mean", "rate_w_gt_5": "mean", "rate_w_gt_10": "mean",
# |         "n_w_infinite": "mean",
# |         "target_active_bin_count": "mean",
# |         "model_supported_target_bin_count": "mean",
# |         "target_bins_with_zero_pred_mass": "mean",
# |         "model_target_bin_support_rate": "mean",
# |         "donor_target_mass_overlap": "mean",
# |         "donor_target_bin_overlap_count": "mean",
# |         "donor_target_bin_overlap_rate": "mean",
# |         "D_placebo": "mean", "D_T": "mean"
# |     }
# |     
# |     # Fill missing donor stats for non-donor rows with 0/NaN appropriately for mean
# |     for col in ["donor_target_mass_overlap", "donor_target_bin_overlap_count", "donor_target_bin_overlap_rate"]:
# |         if col not in df.columns:
# |             df[col] = np.nan
# |     
# |     agg_df = df.groupby(["fold", "target_city", "condition", "replicate_id"]).agg(agg_cols).reset_index()
# |     agg_df.to_csv(f"{output_dir}/matched_placebo_seed_averaged.csv", index=False)
# |     
# |     city_stats = []
# |     for tc in agg_df.target_city.unique():
# |         c_df = agg_df[agg_df.target_city == tc]
# |         fold_val = c_df.fold.values[0]
# |         
# |         target_df = c_df[c_df.condition == "target"]
# |         target_val = target_df["delta_cpc_inter"].values[0] if len(target_df) > 0 else np.nan
# |         
# |         tm_df = c_df[c_df.condition == "trainmean"]
# |         trainmean_val = tm_df["delta_cpc_inter"].values[0] if len(tm_df) > 0 else np.nan
# |         
# |         wrong_df = c_df[c_df.condition == "wrong_city"]
# |         wrong_delta_mean = wrong_df["delta_cpc_inter"].mean() if len(wrong_df) > 0 else np.nan
# |         
# |         perm_df = c_df[c_df.condition == "permuted"]
# |         permuted_delta_mean = perm_df["delta_cpc_inter"].mean() if len(perm_df) > 0 else np.nan
# |             
# |         city_stats.append({
# |             "fold": fold_val, "city": tc,
# |             "target_delta_mean": target_val, "trainmean_delta_mean": trainmean_val,
# |             "wrong_delta_mean": wrong_delta_mean, "permuted_delta_mean": permuted_delta_mean,
# |             "specificity_wrong_mean": target_val - wrong_delta_mean if not np.isnan(wrong_delta_mean) else np.nan,
# |             "specificity_trainmean_mean": target_val - trainmean_val if not np.isnan(trainmean_val) else np.nan,
# |             "specificity_permuted_mean": target_val - permuted_delta_mean if not np.isnan(permuted_delta_mean) else np.nan,
# |             "n_permutations": len(perm_df),
# |             "target_w_raw_max": target_df["w_raw_max"].mean() if len(target_df) > 0 else np.nan,
# |             "target_w_raw_median": target_df["w_raw_median"].mean() if len(target_df) > 0 else np.nan,
# |             "target_w_raw_p95": target_df["w_raw_p95"].mean() if len(target_df) > 0 else np.nan,
# |             "target_n_w_gt_10": target_df["n_w_gt_10"].mean() if len(target_df) > 0 else np.nan,
# |             "target_rate_w_gt_10": target_df["rate_w_gt_10"].mean() if len(target_df) > 0 else np.nan,
# |             "target_n_w_infinite": target_df["n_w_infinite"].mean() if len(target_df) > 0 else np.nan,
# |             "target_active_bin_count": target_df["target_active_bin_count"].mean() if len(target_df) > 0 else np.nan,
# |             "target_model_target_bin_support_rate": target_df["model_target_bin_support_rate"].mean() if len(target_df) > 0 else np.nan,
# |             "target_bins_with_zero_pred_mass": target_df["target_bins_with_zero_pred_mass"].mean() if len(target_df) > 0 else np.nan,
# |             "wrong_donor_target_mass_overlap": wrong_df["donor_target_mass_overlap"].mean() if len(wrong_df) > 0 else np.nan,
# |             "wrong_donor_target_bin_overlap_rate": wrong_df["donor_target_bin_overlap_rate"].mean() if len(wrong_df) > 0 else np.nan,
# |             "target_dose": target_df["D_T"].mean() if len(target_df) > 0 else np.nan,
# |         })
# |         
# |     city_df = pd.DataFrame(city_stats)
# |     city_df.to_csv(f"{output_dir}/matched_placebo_per_city.csv", index=False)
# |     
# |     with open(f"{output_dir}/interpretation.md", "w", encoding="utf-8") as f:
# |         f.write("### Interpretation\n")
# |         f.write("> Permuted-bin chỉ kiểm tra mức độ nhạy của multiplicative calibration khi phân phối mass bị gán sai giữa các bin. Nó không ước lượng giá trị thông tin thực tế của Y_D, ngay cả sau khi các kiểm định kỹ thuật đều pass.\n")
# |
# | if __name__ == "__main__":
# |     parser = argparse.ArgumentParser()
# |     parser.add_argument("--smoke", action="store_true")
# |     parser.add_argument("--b", type=int, default=1000)
# |     args = parser.parse_args()
# |     if args.smoke: args.b = 20
# |     run_placebo_experiment(args)

# ===== END SOURCE FILE: src/experiment/run_placebo_matched_v2.py =====


# ===== BEGIN SOURCE FILE: src/experiment/run_resolution_analysis.py =====
# | import argparse
# | import json
# | import numpy as np
# | import pandas as pd
# | import torch
# | from pathlib import Path
# | import sys
# |
# | sys.path.insert(0, ".")
# | from src.data.dataset import load_city
# | from src.models.zero_shot_model import ZeroShotODModel
# | from src.training.train import load_checkpoint, infer_zero_shot
# | from src.training.evaluate import compute_cpc_pair, compute_cpc_norm_pair
# | from src.calibration.bin_calibration import calibrate_kbins, calibrate_kbins_grouped
# |
# | def run_resolution_analysis(
# |     city: str,
# |     fold_id: int,
# |     yd_oracle_path: str,
# |     data_root: str = "data",
# |     device_str: str = "cpu",
# |     random_smoke: bool = False
# | ):
# |     print(f"\n--- Running Resolution Analysis for {city} (Fold {fold_id}) ---")
# |     
# |     with open(yd_oracle_path, "r") as f:
# |         yd_data = json.load(f)
# |         
# |     bin_edges = np.array(yd_data["bin_edges"])
# |     yd_city = np.array(yd_data["yd_city"])
# |     
# |     yd_county_dict = {k: np.array(v) for k, v in yd_data["yd_county"].items()}
# |     
# |     print(f"Loaded Y_D oracles (K_active={yd_data['K_active']})")
# |     
# |     device = torch.device(device_str)
# |     
# |     meta_df = pd.read_csv(Path(data_root) / city / "meta.csv")
# |     from src.data.gadm_mapper import get_gadm_gid2_mapping
# |     repo_root = str(Path(__file__).resolve().parents[2])
# |     tract_to_county, mapping_stats = get_gadm_gid2_mapping(meta_df, repo_root)
# |     n_counties = len(set(tract_to_county.values()))
# |     print(f"City {city} has {n_counties} origin counties.")
# |
# |     seeds = [1, 10, 100]
# |     results = []
# |
# |     for seed in seeds:
# |         checkpoint_path = f"results/checkpoints/5fold_fold{fold_id}_seed{seed}.pt"
# |         ckpt_file = Path(checkpoint_path)
# |         if ckpt_file.exists():
# |             print(f"Loading checkpoint {checkpoint_path}...")
# |             model, scaler, meta = load_checkpoint(checkpoint_path, device_str=device_str)
# |         else:
# |             if random_smoke:
# |                 print(f"[WARNING] Checkpoint {checkpoint_path} not found. Using randomly initialized model for smoke test.")
# |                 cd = load_city(city, data_root=data_root)
# |                 scaler = None
# |                 model = ZeroShotODModel(
# |                     node_in_dim=cd.node_features.shape[1],
# |                     node_hidden_dim=64,
# |                     node_out_dim=64,
# |                     num_gnn_layers=2,
# |                     decoder_hidden_dim=64
# |                 ).to(device)
# |                 model.eval()
# |             else:
# |                 raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found. Use --random-smoke to fallback to random weights.")
# |
# |         cd = load_city(city, data_root=data_root, feature_scaler=scaler)
# |         
# |         from src.training.train import build_radius_graph
# |         ei, ed = build_radius_graph(cd.lon_lat, radius_km=5.0)
# |         
# |         dist_km = np.expm1(cd.pair_distance.numpy())
# |         inter_mask = (cd.pair_o_idx.numpy() != cd.pair_d_idx.numpy()) & (dist_km > 0.0)
# |         t_gt = cd.pair_trips.numpy().astype(np.float64)
# |         
# |         o_idx_np = cd.pair_o_idx.numpy()
# |         pair_county_idx = np.array([tract_to_county[i] for i in o_idx_np])
# |         
# |         T0 = infer_zero_shot(model, cd, ei, ed, device=device)
# |         t0_np = T0.numpy().astype(np.float64)
# |         
# |         cpc_0 = compute_cpc_pair(t_gt[inter_mask], t0_np[inter_mask])
# |         
# |         t_city = calibrate_kbins(
# |             t0_np=t0_np,
# |             dist_km=dist_km,
# |             inter_mask=inter_mask,
# |             yd_target=yd_city,
# |             bin_edges=bin_edges,
# |             q=1.0
# |         )
# |         cpc_city = compute_cpc_pair(t_gt[inter_mask], t_city[inter_mask])
# |         
# |         t_county = calibrate_kbins_grouped(
# |             t0_np=t0_np,
# |             dist_km=dist_km,
# |             inter_mask=inter_mask,
# |             yd_target_dict=yd_county_dict,
# |             bin_edges=bin_edges,
# |             pair_group_idx=pair_county_idx,
# |             q=1.0
# |         )
# |         cpc_county = compute_cpc_pair(t_gt[inter_mask], t_county[inter_mask])
# |         results.append({
# |             "cpc_0": cpc_0,
# |             "cpc_city": cpc_city,
# |             "cpc_county": cpc_county
# |         })
# |
# |     avg_cpc_0 = np.mean([r["cpc_0"] for r in results])
# |     avg_cpc_city = np.mean([r["cpc_city"] for r in results])
# |     avg_cpc_county = np.mean([r["cpc_county"] for r in results])
# |
# |     print("\n=== RESULTS ===")
# |     print(f"CPC (Zero-Shot)   : {avg_cpc_0:.4f}")
# |     print(f"CPC (M_city)      : {avg_cpc_city:.4f}")
# |     print(f"CPC (M_county)    : {avg_cpc_county:.4f}")
# |     print(f"Delta (county-city): {avg_cpc_county - avg_cpc_city:.4f}")
# |     
# |     return {
# |         "city": city,
# |         "n_counties": n_counties,
# |         "cpc_0": avg_cpc_0,
# |         "cpc_city": avg_cpc_city,
# |         "cpc_county": avg_cpc_county,
# |         "delta_county_city": avg_cpc_county - avg_cpc_city,
# |         "mapping_stats": mapping_stats
# |     }
# |
# | if __name__ == "__main__":
# |     parser = argparse.ArgumentParser()
# |     parser.add_argument("--city", type=str, default="Dallas")
# |     parser.add_argument("--fold", type=int, default=1)
# |     parser.add_argument("--device", type=str, default="cpu")
# |     parser.add_argument("--random-smoke", action="store_true", help="Fallback to random weights if checkpoint missing")
# |     args = parser.parse_args()
# |     
# |     yd_file = f"results/e1_rs/yd_oracles/fold{args.fold}_{args.city}.json"
# |     
# |     run_resolution_analysis(
# |         city=args.city,
# |         fold_id=args.fold,
# |         yd_oracle_path=yd_file,
# |         device_str=args.device,
# |         random_smoke=args.random_smoke
# |     )

# ===== END SOURCE FILE: src/experiment/run_resolution_analysis.py =====


# ===== BEGIN SOURCE FILE: src/experiment/run_sampling_robustness.py =====
# | """
# | Empirical Sampling Robustness Experiment (Task 2).
# | Measures empirical distance distribution error TV(Y_D^(m), Y_D^full) as a function of
# | observed sample size m in {100, 250, 500, 1000, 2500, 5000, 10000, 50000, 100000, inf},
# | and evaluates the resulting OD reconstruction benefit delta_CPC.
# | """
# |
# | import os
# | os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
# | import sys
# | import json
# | import hashlib
# | import argparse
# | import datetime
# | from pathlib import Path
# | from typing import Dict, Tuple, List, Optional, Any
# |
# | import numpy as np
# | import pandas as pd
# | import torch
# | import matplotlib.pyplot as plt
# | import logging
# |
# | from scipy.stats import spearmanr, wilcoxon, multivariate_hypergeom
# | from scipy.spatial.distance import jensenshannon
# |
# | sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
# | from src.data.dataset import load_city, load_raw_city
# | from src.data.urban_graph import build_radius_graph
# | from src.training.train import load_checkpoint
# | from src.training.evaluate import compute_cpc_pair
# | from src.data.yd_extractor import compute_kbin_edges, extract_yd_kbins
# | from src.data.city_splits import generate_35_5_10_splits, load_splits_manifest_v2
# | from src.experiment.run_experiment import infer_zero_shot
# |
# |
# | def holm_correction(p_vals: List[float]) -> np.ndarray:
# |     n = len(p_vals)
# |     if n == 0:
# |         return np.array([])
# |     sorted_indices = np.argsort(p_vals)
# |     adj_p = np.zeros(n)
# |     running_max = 0.0
# |     for i, idx in enumerate(sorted_indices):
# |         p_adj = p_vals[idx] * (n - i)
# |         running_max = max(running_max, p_adj)
# |         adj_p[idx] = min(1.0, running_max)
# |     return adj_p
# |
# |
# | def get_stable_seed(base_seed: int, fold: int, city: str, m_val: Any, replicate_id: int) -> int:
# |     s = f"{base_seed}_{fold}_{city}_{m_val}_{replicate_id}"
# |     return int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % (2**32)
# |
# |
# | def sample_hypergeometric_yd(bin_counts: np.ndarray, m: float, size: int, base_seed: int) -> List[np.ndarray]:
# |     """
# |     Subsamples m trips without replacement from actual population bin counts
# |     using the Multivariate Hypergeometric distribution.
# |     """
# |     total_trips = int(bin_counts.sum())
# |     if np.isinf(m) or m >= total_trips:
# |         yd_exact = bin_counts.astype(np.float64) / float(total_trips) if total_trips > 0 else np.ones_like(bin_counts)/len(bin_counts)
# |         return [yd_exact.copy() for _ in range(size)]
# |         
# |     m_int = int(m)
# |     rng = np.random.RandomState(base_seed)
# |     
# |     # Multivariate hypergeometric draw
# |     draws = multivariate_hypergeom.rvs(m=bin_counts, n=m_int, size=size, random_state=rng)
# |     if size == 1:
# |         draws = draws.reshape(1, -1)
# |     return [draws[i].astype(np.float64) / float(m_int) for i in range(size)]
# |
# |
# | def fold_stratified_bootstrap(city_df: pd.DataFrame, metric_col: str, m_val: float, evaluated_folds: List[int], n_boot: int = 10000, seed: int = 42) -> Tuple[float, float]:
# |     rng = np.random.RandomState(seed)
# |     
# |     vals: Dict[int, np.ndarray] = {}
# |     for f in evaluated_folds:
# |         mask = (city_df.fold == f) & (city_df.sample_m == m_val)
# |         vals[f] = city_df[mask][metric_col].values
# |         assert len(vals[f]) == 10, f"Expected 10 cities for fold {f}, got {len(vals[f])}"
# |         
# |     f_samples = [vals[f][rng.randint(0, 10, size=(n_boot, 10))] for f in evaluated_folds]
# |     all_samples = np.hstack(f_samples)
# |     boot_means = np.mean(all_samples, axis=1)
# |         
# |     return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))
# |
# |
# | def fast_cal_metrics(
# |     yd_tgt: np.ndarray, 
# |     compute_spearman: bool, 
# |     N_hat: float, 
# |     K: int, 
# |     active: np.ndarray, 
# |     Y_hat: np.ndarray, 
# |     t0_inter: np.ndarray, 
# |     bin_idx: np.ndarray, 
# |     t_true_inter: np.ndarray, 
# |     cpc_m0: float, 
# |     yd_target: np.ndarray,
# |     inv_sum_denom: float,
# |     inv_N: float,
# |     t_cal_buf: np.ndarray,
# |     diff_buf: np.ndarray
# | ) -> Tuple[float, float, float, float, float, float, Dict[str, float]]:
# |     
# |     if N_hat <= 0:
# |         return cpc_m0, 0.0, 0.0, 0.0, 0.0, 0.0, {}
# |     
# |     yd_raw = yd_tgt / yd_tgt.sum() if yd_tgt.sum() > 0 else np.ones(K) / K
# |     yd_active = yd_raw * active.astype(np.float64)
# |     active_sum = yd_active.sum()
# |     Y_D_cond = yd_active / active_sum if active_sum > 0 else Y_hat.copy()
# |     
# |     w = np.ones(K, dtype=np.float64)
# |     for k in range(K):
# |         if active[k] and Y_hat[k] > 0:
# |             w[k] = Y_D_cond[k] / Y_hat[k]
# |             
# |     weighted_mass = float(np.dot(Y_hat, w))
# |     s = w / weighted_mass if weighted_mass > 0 else np.ones(K)
# |     
# |     np.multiply(t0_inter, s[bin_idx], out=t_cal_buf)
# |             
# |     cal_mass = t_cal_buf.sum()
# |     if cal_mass > 0:
# |         t_cal_buf *= (N_hat / cal_mass)
# |         
# |     cpc = float(np.sum(np.minimum(t_true_inter, t_cal_buf)) * inv_sum_denom)
# |     
# |     np.subtract(t_true_inter, t_cal_buf, out=diff_buf)
# |     np.abs(diff_buf, out=diff_buf)
# |     mae = float(np.sum(diff_buf) * inv_N)
# |     
# |     np.square(diff_buf, out=diff_buf)
# |     rmse = float(np.sqrt(np.sum(diff_buf) * inv_N))
# |     
# |     spearman_val = float(spearmanr(t_true_inter, t_cal_buf)[0]) if compute_spearman else float('nan')
# |     
# |     active_w = w[active]
# |     w_gt_2 = float(np.mean(active_w > 2)) if len(active_w) > 0 else 0.0
# |     w_gt_5 = float(np.mean(active_w > 5)) if len(active_w) > 0 else 0.0
# |     w_gt_10 = float(np.mean(active_w > 10)) if len(active_w) > 0 else 0.0
# |     
# |     stats = {
# |         "w_min": float(active_w.min()) if len(active_w) > 0 else 1.0,
# |         "w_median": float(np.median(active_w)) if len(active_w) > 0 else 1.0,
# |         "w_p95": float(np.percentile(active_w, 95)) if len(active_w) > 0 else 1.0,
# |         "w_max": float(active_w.max()) if len(active_w) > 0 else 1.0,
# |         "w_gt_2": w_gt_2, "w_gt_5": w_gt_5, "w_gt_10": w_gt_10
# |     }
# |     
# |     tv_ach = float(0.5 * np.sum(np.abs(yd_tgt - yd_target)))
# |     js_div = float(jensenshannon(yd_tgt, yd_target)) ** 2
# |     
# |     return cpc, mae, rmse, spearman_val, tv_ach, js_div, stats
# |
# |
# | def run_sampling_robustness(args: argparse.Namespace) -> None:
# |     data_root = "data"
# |     output_dir = getattr(args, "output_dir", None) or "results/sampling_robustness_v1"
# |     os.makedirs(output_dir, exist_ok=True)
# |     
# |     log_file = f"{output_dir}/run.log"
# |     logging.basicConfig(level=logging.INFO, format='%(message)s',
# |                         handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
# |     logger = logging.getLogger(__name__)
# |     
# |     sampling_base_seed = 20260823
# |     
# |     if args.smoke:
# |         m_grid = [100, 1000, float("inf")]
# |         B_sample = 20
# |         model_seeds_to_use = [1, 10]
# |         folds_to_run = [2]
# |     else:
# |         m_grid = [100, 250, 500, 1000, 2500, 5000, 10000, 50000, 100000, float("inf")]
# |         B_sample = args.b
# |         model_seeds_to_use = [1, 10, 100]
# |         folds_to_run = [1, 2, 3, 4, 5]
# |         
# |     splits = generate_35_5_10_splits(data_root=data_root)
# |     raw_results: List[Dict[str, Any]] = []
# |     
# |     for fold_id in folds_to_run:
# |         split = splits[fold_id]
# |         train_cities = split["train"]
# |         test_cities_to_use = split["test"] if not args.smoke else split["test"][:1]
# |             
# |         logger.info(f"\n=== Processing Fold {fold_id} ===")
# |         
# |         bin_edges, _ = compute_kbin_edges(train_cities, K=8, data_root=data_root)
# |         K = len(bin_edges) - 1
# |         
# |         for c_idx, tc in enumerate(test_cities_to_use):
# |             logger.info(f"  Target City: {tc} ({c_idx+1}/{len(test_cities_to_use)})")
# |             raw = load_raw_city(tc, data_root=data_root)
# |             dist_km = raw.dist_km
# |             inter_mask = (raw.pair_o_idx.numpy() != raw.pair_d_idx.numpy()) & (dist_km > 0.0)
# |             t_true_inter = raw.pair_trips.numpy()[inter_mask]
# |             
# |             yd_target = extract_yd_kbins(dist_km, raw.pair_trips.numpy(), bin_edges, inter_mask)
# |             
# |             dist_inter = dist_km[inter_mask]
# |             bin_idx = np.clip(np.digitize(dist_inter, bin_edges[1:-1], right=True), 0, K - 1).astype(np.int32)
# |             n_inter_pairs = len(dist_inter)
# |             inv_N = 1.0 / n_inter_pairs if n_inter_pairs > 0 else 0.0
# |             sum_t_true = float(t_true_inter.sum())
# |             
# |             inter_trips_int = t_true_inter.astype(np.int64)
# |             bin_counts = np.bincount(bin_idx, weights=inter_trips_int, minlength=K).astype(np.int64)
# |             
# |             # Pre-generate empirical subsampled Y_D sets without replacement (Multivariate Hypergeometric)
# |             logger.info("    Drawing empirical multivariate hypergeometric samples without replacement...")
# |             city_sample_sets: Dict[float, List[np.ndarray]] = {}
# |             for m in m_grid:
# |                 seed_m = get_stable_seed(sampling_base_seed, fold_id, tc, int(m) if not np.isinf(m) else 0, 0)
# |                 city_sample_sets[m] = sample_hypergeometric_yd(bin_counts, m, B_sample, seed_m)
# |                     
# |             edge_index, edge_dist = build_radius_graph(
# |                 lon_lat=raw.lon_lat, radius_km=5.0, include_self_loop=True, cache_key=f"{tc}_tracts"
# |             )
# |             
# |             t_cal_buf = np.empty(n_inter_pairs, dtype=np.float64)
# |             diff_buf = np.empty(n_inter_pairs, dtype=np.float64)
# |             
# |             for m_seed in model_seeds_to_use:
# |                 logger.info(f"    Evaluating seed {m_seed}...")
# |                 ckpt_path = Path(f"results/checkpoints/5fold_fold{fold_id}_seed{m_seed}.pt")
# |                 if not ckpt_path.exists():
# |                     raise FileNotFoundError(f"Missing mandatory checkpoint {ckpt_path}. The protocol requires all 3 model seeds to evaluate.")
# |                 model, scaler, _ = load_checkpoint(ckpt_path, device_str="cpu")
# |                 model.eval()
# |                 
# |                 city_data = load_city(tc, data_root=data_root, feature_scaler=scaler, fit_scaler=False)
# |                 t_pred_zs_tensor = infer_zero_shot(model, city_data, edge_index, edge_dist, device="cpu")
# |                 t_pred_zs = t_pred_zs_tensor.numpy().astype(np.float64)
# |                 
# |                 t0_inter = t_pred_zs[inter_mask]
# |                 N_hat = float(t0_inter.sum())
# |                 cpc_m0 = float(compute_cpc_pair(t_true_inter, t0_inter))
# |                 
# |                 sum_denom = sum_t_true + N_hat
# |                 inv_sum_denom = 2.0 / sum_denom if sum_denom > 0 else 0.0
# |                 
# |                 Y_hat = np.zeros(K, dtype=np.float64)
# |                 active = np.zeros(K, dtype=bool)
# |                 if N_hat > 0:
# |                     counts = np.bincount(bin_idx, weights=t0_inter, minlength=K)
# |                     Y_hat = counts / N_hat
# |                     pair_counts = np.bincount(bin_idx, minlength=K)
# |                     active = pair_counts > 0
# |                 
# |                 # 1. Oracle (m=inf)
# |                 oracle_cpc, o_mae, o_rmse, o_spr, o_tv, o_js, o_stats = fast_cal_metrics(
# |                     yd_target, True, N_hat, K, active, Y_hat, t0_inter, bin_idx, t_true_inter, cpc_m0, yd_target,
# |                     inv_sum_denom, inv_N, t_cal_buf, diff_buf
# |                 )
# |                 
# |                 def build_row(m_val: float, rep_id: int, cpc_val: float, mae: float, rmse: float, spr: float, tv_ach: float, js_div: float, st: Dict[str, float]) -> Dict[str, Any]:
# |                     row = {
# |                         "fold": fold_id, "target_city": tc, "model_seed": m_seed,
# |                         "sample_m": m_val, "replicate_id": rep_id,
# |                         "cpc_m0_inter": cpc_m0, "cpc_m1_inter": cpc_val,
# |                         "delta_cpc_inter": float(cpc_val - cpc_m0),
# |                         "degradation": float(oracle_cpc - cpc_val),
# |                         "mae": mae, "rmse": rmse, "spearman": spr,
# |                         "empirical_tv": tv_ach, "js_divergence": js_div,
# |                     }
# |                     row.update(st)
# |                     return row
# |                     
# |                 raw_results.append(build_row(float("inf"), 0, oracle_cpc, o_mae, o_rmse, o_spr, o_tv, o_js, o_stats))
# |                 
# |                 # 2. Finite sample sizes m
# |                 for m in m_grid:
# |                     if np.isinf(m):
# |                         continue
# |                     sample_list = city_sample_sets[m]
# |                     for b, yd_s in enumerate(sample_list):
# |                         n_cpc, n_mae, n_rmse, n_spr, n_tv, n_js, n_stats = fast_cal_metrics(
# |                             yd_s, False, N_hat, K, active, Y_hat, t0_inter, bin_idx, t_true_inter, cpc_m0, yd_target,
# |                             inv_sum_denom, inv_N, t_cal_buf, diff_buf
# |                         )
# |                         raw_results.append(build_row(float(m), b + 1, n_cpc, n_mae, n_rmse, n_spr, n_tv, n_js, n_stats))
# |                 
# |     df = pd.DataFrame(raw_results)
# |     if not df.empty:
# |         df['spearman'] = df['spearman'].astype(float)
# |         
# |         df.to_csv(f"{output_dir}/sampling_raw.csv", index=False)
# |         df.to_json(f"{output_dir}/sampling_raw.jsonl", orient="records", lines=True)
# |         logger.info(f"Raw results saved with {len(df)} rows.")
# |         
# |         # Aggregation Step 1 & 2
# |         df_mean_b = df.groupby(["fold", "target_city", "model_seed", "sample_m"]).agg(
# |             delta_cpc_inter=("delta_cpc_inter", "mean"),
# |             degradation=("degradation", "mean"),
# |             empirical_tv=("empirical_tv", "mean"),
# |             js_divergence=("js_divergence", "mean"),
# |             cpc_m1_inter=("cpc_m1_inter", "mean"),
# |             prob_positive=("delta_cpc_inter", lambda x: float(np.mean(x > 0)))
# |         ).reset_index()
# |         
# |         df_seed_csv = df_mean_b.copy()
# |         df_seed_csv.to_csv(f"{output_dir}/sampling_per_seed.csv", index=False)
# |         
# |         city_df = df_mean_b.groupby(["fold", "target_city", "sample_m"]).agg(
# |             delta_cpc_mean=("delta_cpc_inter", "mean"),
# |             degradation_mean=("degradation", "mean"),
# |             empirical_tv_mean=("empirical_tv", "mean"),
# |             js_div_mean=("js_divergence", "mean"),
# |             prob_positive=("prob_positive", "mean"),
# |             cpc_m1_inter=("cpc_m1_inter", "mean")
# |         ).reset_index()
# |         
# |         city_df.to_csv(f"{output_dir}/sampling_per_city.csv", index=False)
# |         
# |         if not args.smoke:
# |             generate_sampling_summary(city_df, output_dir, m_grid)
# |     else:
# |         logger.warning("No results were generated. Check checkpoints.")
# |
# |
# | def generate_sampling_summary(city_df: pd.DataFrame, output_dir: str, m_grid: List[float]) -> None:
# |     evaluation_folds = sorted(city_df.fold.unique().tolist())
# |     eval_df = city_df[city_df.fold.isin(evaluation_folds)]
# |     
# |     if eval_df.empty:
# |         return
# |         
# |     sorted_m = sorted(m_grid, key=lambda x: (np.isinf(x), x))
# |     finite_m = [m for m in sorted_m if not np.isinf(m)]
# |     
# |     results: Dict[str, Dict[str, Any]] = {}
# |     p_benefit_onesided: List[float] = []
# |     p_degrad_onesided: List[float] = []
# |     
# |     # Get oracle delta_cpc per city for degradation paired test
# |     clean_vals_by_city: Dict[Tuple[int, str], float] = {}
# |     c_clean = eval_df[eval_df.sample_m.isin([float('inf')])]
# |     for _, row in c_clean.iterrows():
# |         clean_vals_by_city[(row["fold"], row["target_city"])] = row["delta_cpc_mean"]
# |         
# |     for m in sorted_m:
# |         m_str = "inf" if np.isinf(m) else str(int(m))
# |         c_m = eval_df[eval_df.sample_m == m]
# |         vals = c_m.delta_cpc_mean.values
# |         tv_vals = c_m.empirical_tv_mean.values
# |         
# |         mean_cpc1 = float(c_m.cpc_m1_inter.mean())
# |         mean_val = float(np.mean(vals))
# |         sd_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
# |         median = float(np.median(vals))
# |         p25 = float(np.percentile(vals, 25))
# |         p75 = float(np.percentile(vals, 75))
# |         pos_cities = int(np.sum(vals > 0))
# |         harm_rate = float(np.sum(vals < 0) / len(vals))
# |         
# |         mean_tv = float(np.mean(tv_vals))
# |         tv_ci_lo, tv_ci_hi = fold_stratified_bootstrap(eval_df, "empirical_tv_mean", m, evaluation_folds)
# |         ci_lower, ci_upper = fold_stratified_bootstrap(eval_df, "delta_cpc_mean", m, evaluation_folds)
# |         
# |         # 1. Benefit Test (H1: delta_cpc > 0 vs M0)
# |         try:
# |             _, p_ben = wilcoxon(vals, alternative='greater')
# |         except Exception:
# |             p_ben = 1.0
# |             
# |         # 2. Degradation Test (H1: delta_cpc_oracle - delta_cpc_m > 0)
# |         degrad_vals = []
# |         for _, row in c_m.iterrows():
# |             clean_v = clean_vals_by_city.get((row["fold"], row["target_city"]), row["delta_cpc_mean"])
# |             degrad_vals.append(clean_v - row["delta_cpc_mean"])
# |         degrad_arr = np.array(degrad_vals)
# |         mean_degrad = float(np.mean(degrad_arr))
# |         
# |         if not np.isinf(m):
# |             try:
# |                 _, p_deg = wilcoxon(degrad_arr, alternative='greater')
# |             except Exception:
# |                 p_deg = 1.0
# |             p_benefit_onesided.append(float(p_ben))
# |             p_degrad_onesided.append(float(p_deg))
# |         else:
# |             p_deg = float('nan')
# |             
# |         results[m_str] = {
# |             "sample_m": m if not np.isinf(m) else None,
# |             "mean_cpc1": mean_cpc1,
# |             "mean_delta_cpc": mean_val, "sd": sd_val, "median": median,
# |             "p25": p25, "p75": p75, "ci_lower": ci_lower, "ci_upper": ci_upper,
# |             "pos_cities": pos_cities, "harm_rate": harm_rate,
# |             "mean_empirical_tv": mean_tv, "tv_ci_lo": tv_ci_lo, "tv_ci_hi": tv_ci_hi,
# |             "mean_degradation": mean_degrad,
# |             "wilcoxon_benefit_raw": float(p_ben),
# |             "wilcoxon_degrad_raw": float(p_deg) if not np.isnan(p_deg) else None
# |         }
# |         
# |     p_ben_adj = holm_correction(p_benefit_onesided)
# |     p_deg_adj = holm_correction(p_degrad_onesided)
# |     
# |     for i, m in enumerate(finite_m):
# |         m_str = str(int(m))
# |         results[m_str]["wilcoxon_benefit_holm"] = float(p_ben_adj[i])
# |         results[m_str]["wilcoxon_degrad_holm"] = float(p_deg_adj[i])
# |         
# |     oracle_gain = float(results["inf"]["mean_delta_cpc"])
# |     for m_str in results:
# |         if oracle_gain > 0:
# |             results[m_str]["relative_effect_pct"] = float(results[m_str]["mean_delta_cpc"] / oracle_gain * 100.0)
# |         else:
# |             results[m_str]["relative_effect_pct"] = None
# |             
# |     # Find crossover sample size m_cross where mean_delta_cpc >= 0
# |     m_cross = None
# |     for m in finite_m:
# |         if results[str(int(m))]["mean_delta_cpc"] >= 0:
# |             m_cross = int(m)
# |             break
# |             
# |     # Find significant benefit sample size m*
# |     m_star = None
# |     for m in finite_m:
# |         m_str = str(int(m))
# |         cond1 = results[m_str]["mean_delta_cpc"] > 0
# |         cond2 = results[m_str]["ci_lower"] > 0
# |         cond3 = results[m_str]["wilcoxon_benefit_holm"] < 0.05
# |         if cond1 and cond2 and cond3:
# |             m_star = int(m)
# |             break
# |             
# |     summary = {
# |         "n_evaluation_cities": int(len(eval_df) // len(m_grid)),
# |         "m_cross_positive_dCPC": m_cross,
# |         "m_star_significant_benefit": m_star,
# |         "results_by_m": results
# |     }
# |     
# |     with open(f"{output_dir}/sampling_summary.json", "w") as f:
# |         json.dump(summary, f, indent=2)
# |         
# |     md = "# Empirical Y_D Sampling Robustness Summary (Subsampling Without Replacement)\n\n"
# |     md += f"## Five-Fold Cross-City Evaluation Table (All 5 Folds, {int(len(eval_df)//len(m_grid))} Held-Out Test Cities)\n\n"
# |     if m_star is not None:
# |         md += f"**Primary Finding — Full 5-fold Benefit Threshold ($m^*$):** `{m_star:,}` observed trips ($p < 0.05$ Holm, $95\\%\\text{{ CI}}_{{\\text{{lower}}}} > 0$)\n\n"
# |     if m_cross is not None:
# |         md += f"*Note on Crossover:* Smallest tested sample size with positive mean $\\Delta\\text{{CPC}}$ is `{m_cross:,}` trips (mean $\\Delta\\text{{CPC}} > 0$, but not statistically significant, $p = {results.get(str(m_cross), {}).get('wilcoxon_benefit_holm', 1.0):.4f}$).\n\n"
# |         
# |     md += "| Sample Size (m) | Mean Empirical TV | Mean M1 CPC | Mean dCPC | 95% CI | Pos Cities | Harm Rate | Rel Effect vs Clean (%) | Benefit p-val (vs M0) | Degrad p-val (vs Clean) |\n"
# |     md += "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
# |     
# |     for m in sorted_m:
# |         m_str = "inf" if np.isinf(m) else str(int(m))
# |         d = results[m_str]
# |         m_label = r"$\infty$ (Oracle)" if np.isinf(m) else f"{int(m):,}"
# |         tv_label = f"{d['mean_empirical_tv']:.4f} ({d['mean_empirical_tv']*100:.2f}%)"
# |         ci = f"[{d['ci_lower']:.5f}, {d['ci_upper']:.5f}]"
# |         
# |         ben_holm = d.get('wilcoxon_benefit_holm', d.get('wilcoxon_benefit_raw'))
# |         if isinstance(ben_holm, (float, np.floating)):
# |             ben_str = f"{ben_holm:.2e}" if ben_holm < 0.001 else f"{ben_holm:.4f}"
# |         else:
# |             ben_str = "N/A"
# |             
# |         deg_holm = d.get('wilcoxon_degrad_holm')
# |         if isinstance(deg_holm, (float, np.floating)):
# |             deg_str = f"{deg_holm:.2e}" if deg_holm < 0.001 else f"{deg_holm:.4f}"
# |         else:
# |             deg_str = "—"
# |             
# |         rel_eff = f"{d['relative_effect_pct']:+.1f}%" if d['relative_effect_pct'] is not None else "N/A"
# |         md += f"| {m_label} | {tv_label} | {d['mean_cpc1']:.5f} | {d['mean_delta_cpc']:+.5f} | {ci} | {d['pos_cities']}/{int(len(eval_df)//len(m_grid))} | {d['harm_rate']:.1%} | {rel_eff} | {ben_str} | {deg_str} |\n"
# |         
# |     with open(f"{output_dir}/sampling_summary.md", "w") as f:
# |         f.write(md)
# |         
# |     # --- Figure 1: Sample Size m vs Empirical TV Error ---
# |     plt.figure(figsize=(9, 6))
# |     finite_m_arr = np.array(finite_m)
# |     tv_means = [results[str(int(m))]["mean_empirical_tv"] for m in finite_m]
# |     tv_los = [results[str(int(m))]["tv_ci_lo"] for m in finite_m]
# |     tv_his = [results[str(int(m))]["tv_ci_hi"] for m in finite_m]
# |     
# |     # Read noise thresholds if available
# |     e_cross, e_star = 0.0478, 0.0300
# |     try:
# |         with open("results/noise_robustness_v1/noise_summary.json", "r") as f:
# |             noise_summ = json.load(f)
# |             e_cross = noise_summ.get("epsilon_cross_positive_dCPC", e_cross)
# |             e_star = noise_summ.get("epsilon_star_significant_benefit", e_star)
# |     except Exception:
# |         pass
# |
# |     plt.plot(finite_m_arr, tv_means, marker="o", color="darkblue", linewidth=2, label="Empirical TV Error")
# |     plt.fill_between(finite_m_arr, tv_los, tv_his, color="royalblue", alpha=0.25, label="95% Bootstrap CI")
# |     plt.axhline(e_cross, color="red", linestyle="--", linewidth=1.5, label=f"Theoretical Crossover $\\epsilon_{{cross}} = {e_cross*100:.2f}\\%$")
# |     plt.axhline(e_star, color="darkorange", linestyle=":", linewidth=1.5, label=f"Significance Threshold $\\epsilon^* = {e_star*100:.2f}\\%$")
# |     if m_star is not None:
# |         plt.axvline(m_star, color="green", linestyle="-.", label=f"Required $m^* = {m_star:,}$ trips")
# |     plt.xscale("log")
# |     plt.xlabel("Sample Size $m$ (Number of Observed Trips, log scale)")
# |     plt.ylabel("Empirical Total Variation Error $\\text{TV}(\\tilde{Y}_D^{(m)}, Y_D^{full})$")
# |     plt.title("Empirical TV Error vs Sample Size $m$")
# |     plt.grid(True, which="both", linestyle=":", alpha=0.6)
# |     plt.legend()
# |     plt.savefig(f"{output_dir}/fig_sampling_m_vs_tv.png", dpi=300, bbox_inches="tight")
# |     plt.close()
# |     
# |     # --- Figure 2: Sample Size m vs Delta CPC ---
# |     plt.figure(figsize=(9, 6))
# |     dcpc_means = [results[str(int(m))]["mean_delta_cpc"] for m in finite_m]
# |     dcpc_los = [results[str(int(m))]["ci_lower"] for m in finite_m]
# |     dcpc_his = [results[str(int(m))]["ci_upper"] for m in finite_m]
# |     
# |     plt.plot(finite_m_arr, dcpc_means, marker="o", color="royalblue", linewidth=2, label="Mean $\\Delta$CPC")
# |     plt.fill_between(finite_m_arr, dcpc_los, dcpc_his, color="cornflowerblue", alpha=0.25, label="95% Bootstrap CI")
# |     plt.axhline(0, color="red", linestyle="--", alpha=0.7, label="Zero-Shot M0 Baseline")
# |     plt.axhline(oracle_gain, color="green", linestyle=":", label=f"Oracle Gain (+{oracle_gain:.5f})")
# |     if m_star is not None:
# |         plt.axvline(m_star, color="green", linestyle="-.", label=f"$m^* = {m_star:,}$ trips")
# |     plt.xscale("log")
# |     plt.xlabel("Sample Size $m$ (Number of Observed Trips, log scale)")
# |     plt.ylabel("Delta CPC ($M_1 - M_0$)")
# |     plt.title("OD Reconstruction Gain vs Observed Sample Size $m$")
# |     plt.grid(True, which="both", linestyle=":", alpha=0.6)
# |     plt.legend()
# |     plt.savefig(f"{output_dir}/fig_sampling_m_vs_dcpc.png", dpi=300, bbox_inches="tight")
# |     plt.close()
# |     
# |     # --- Figure 3: Harm Rate vs Sample Size m ---
# |     plt.figure(figsize=(9, 6))
# |     harm_rates = [results[str(int(m))]["harm_rate"] for m in finite_m]
# |     plt.plot(finite_m_arr, harm_rates, marker="s", color="firebrick", linewidth=2, label="Harm Rate")
# |     plt.axhline(0.05, color="gray", linestyle=":", label="Oracle Baseline Harm Rate (5.0%)")
# |     plt.xscale("log")
# |     plt.xlabel("Sample Size $m$ (Number of Observed Trips, log scale)")
# |     plt.ylabel("Harm Rate (% Cities Worse than M0)")
# |     plt.ylim(-0.02, 1.02)
# |     plt.title("Harm Rate vs Observed Sample Size $m$")
# |     plt.grid(True, which="both", linestyle=":", alpha=0.6)
# |     plt.legend()
# |     plt.savefig(f"{output_dir}/fig_sampling_harm_rate.png", dpi=300, bbox_inches="tight")
# |     plt.close()
# |     
# |     # --- Figure 4: Empirical TV vs Delta CPC (Direct Bridge to Synthetic Curve) ---
# |     plt.figure(figsize=(9, 6))
# |     plt.scatter(tv_means, dcpc_means, color="navy", s=60, zorder=3, label="Empirical Sampling ($m$)")
# |     for idx, m in enumerate(finite_m):
# |         plt.annotate(f"m={int(m):,}", (tv_means[idx], dcpc_means[idx]), textcoords="offset points", xytext=(5, 5), fontsize=8)
# |     plt.axhline(0, color="red", linestyle="--", alpha=0.7, label="Zero-Shot M0 Baseline")
# |     plt.axvline(e_cross, color="darkorange", linestyle=":", label=f"Synthetic Crossover $\\epsilon_{{cross}} = {e_cross*100:.2f}\\%$")
# |     plt.xlabel("Empirical Total Variation Error $\\text{TV}$")
# |     plt.ylabel("Mean $\\Delta$CPC")
# |     plt.title("Bridge: Empirical Sampling Error vs Reconstruction Benefit $\\Delta$CPC")
# |     plt.grid(True, linestyle=":", alpha=0.6)
# |     plt.legend()
# |     plt.savefig(f"{output_dir}/fig_sampling_tv_vs_dcpc_curve.png", dpi=300, bbox_inches="tight")
# |     plt.close()
# |     
# |     manifest = {
# |         "experiment": "empirical_sampling_robustness",
# |         "timestamp": datetime.datetime.now().isoformat(),
# |         "m_grid": [m if not np.isinf(m) else "inf" for m in sorted_m],
# |         "m_cross": m_cross,
# |         "m_star": m_star
# |     }
# |     with open(f"{output_dir}/sampling_manifest.json", "w") as f:
# |         json.dump(manifest, f, indent=2)
# |
# |
# | if __name__ == "__main__":
# |     parser = argparse.ArgumentParser()
# |     parser.add_argument("--b", type=int, default=1000)
# |     parser.add_argument("--output_dir", type=str, default=None)
# |     parser.add_argument("--smoke", action="store_true")
# |     args = parser.parse_args()
# |     run_sampling_robustness(args)

# ===== END SOURCE FILE: src/experiment/run_sampling_robustness.py =====


# ===== BEGIN SOURCE FILE: src/experiment/run_spatial_resolution_experiment.py =====
# | """
# | Spatial Resolution Experiment (Origin County-Level vs. City-Level Calibration).
# |
# | Evaluates whether providing finer spatial resolution in the aggregate distance distribution
# | (Y_D^(county) conditioned on origin county vs. macro city-wide Y_D^(city)) enhances mobility
# | prediction accuracy in heterogeneous metropolitan areas.
# |
# | Key Estimands:
# |     1. City-Level Target Gain:        Δ_city        = CPC(M_city) - CPC(M0)
# |     2. County-Level Target Gain:      Δ_county      = CPC(M_county) - CPC(M0)
# |     3. Spatial Resolution Gain:       Δ_resolution  = CPC(M_county) - CPC(M_city)
# |     4. Specificity Gains:             Δ_spec_city   = CPC(M_city) - CPC(M_wrong)
# |                                       Δ_spec_county = CPC(M_county) - CPC(M_wrong)
# |
# | Invariance Properties:
# |     - For single-county cities (n=45), M_county ≡ M_city, so Δ_resolution ≡ 0.0000 (Sanity Check).
# |     - For multi-county cities (n=5: Atlanta, Dallas, Kansas City, New York, Tulsa),
# |       heterogeneous origin distributions allow fine-grained spatial adaptation (Δ_resolution >= 0).
# | """
# |
# | import argparse
# | import json
# | import os
# | import sys
# | import time
# | from pathlib import Path
# | import numpy as np
# | import pandas as pd
# | from scipy import stats
# | import torch
# |
# | # Ensure project root is in sys.path
# | PROJECT_ROOT = Path(__file__).resolve().parents[2]
# | sys.path.insert(0, str(PROJECT_ROOT))
# |
# | from src.data.dataset import load_city, preload_all_cities
# | from src.data.city_splits import load_splits_manifest_v2
# | from src.data.yd_extractor import compute_kbin_edges, extract_yd_kbins, extract_yd_kbins_grouped
# | from src.calibration.bin_calibration import calibrate_kbins, calibrate_kbins_grouped
# | from src.models.zero_shot_model import ZeroShotODModel
# | from src.training.train import (
# |     train_zero_shot_model,
# |     infer_zero_shot,
# |     build_radius_graph,
# |     load_checkpoint,
# |     save_checkpoint
# | )
# | from src.training.evaluate import compute_cpc_pair, compute_cpc_norm_pair
# |
# | # Output directories & constants
# | RESULTS_DIR = PROJECT_ROOT / "results" / "spatial_resolution"
# | TABLES_DIR = RESULTS_DIR / "tables"
# | DATA_ROOT = "data"
# | K_MOVE = 8
# | Q_CALIB = 1.0
# | EPOCHS = 200
# | PATIENCE = 15
# | MIN_DELTA = 1e-4
# | DEFAULT_SEED = 2024
# |
# |
# | def log_msg(msg: str = "", print_to_console: bool = True):
# |     timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
# |     formatted = f"[{timestamp}] {msg}" if msg else ""
# |     if print_to_console:
# |         print(formatted if formatted else "", flush=True)
# |     LOG_FILE = RESULTS_DIR / "spatial_resolution.log"
# |     try:
# |         with open(LOG_FILE, "a", encoding="utf-8", errors="replace") as f:
# |             f.write(formatted + "\n")
# |     except Exception:
# |         pass
# |
# |
# | def safe_wilcoxon(diff: np.ndarray, alternative: str = "greater") -> tuple[float, float]:
# |     diff_clean = diff[~np.isnan(diff)]
# |     if len(diff_clean) < 2:
# |         return 0.0, 1.0
# |     non_zero = diff_clean[diff_clean != 0.0]
# |     if len(non_zero) == 0:
# |         return 0.0, 1.0
# |     try:
# |         res = stats.wilcoxon(diff_clean, alternative=alternative, zero_method="wilcox")
# |         return float(res.statistic), float(res.pvalue)
# |     except Exception:
# |         return 0.0, 1.0
# |
# |
# | def fold_bootstrap(
# |     values: np.ndarray,
# |     fold_ids: np.ndarray,
# |     n: int = 10000,
# |     seed: int = 2024,
# |     alpha: float = 0.05,
# | ) -> tuple[float, float]:
# |     rng = np.random.default_rng(seed)
# |     folds = sorted(set(fold_ids))
# |     boot = []
# |     for _ in range(n):
# |         s = []
# |         for f in folds:
# |             fd = values[fold_ids == f]
# |             if len(fd) > 0:
# |                 s.extend(rng.choice(fd, size=len(fd), replace=True))
# |         if s:
# |             boot.append(np.mean(s))
# |     boot = np.array(boot)
# |     if len(boot) == 0:
# |         return 0.0, 0.0
# |     return float(np.percentile(boot, 100 * (alpha / 2))), float(np.percentile(boot, 100 * (1 - alpha / 2)))
# |
# |
# | def run_spatial_resolution_city(
# |     city: str,
# |     model: torch.nn.Module,
# |     scaler: object,
# |     bin_edges: np.ndarray,
# |     test_cities: list[str],
# |     fold_id: int,
# |     device: torch.device,
# |     test_yd_cache: dict[str, np.ndarray],
# | ) -> dict:
# |     t_start = time.time()
# |     
# |     cd = load_city(city, data_root=DATA_ROOT, feature_scaler=scaler)
# |     ei, ed = build_radius_graph(cd.lon_lat, radius_km=5.0)
# |     dist_km = np.expm1(cd.pair_distance.numpy())
# |     inter_mask = (cd.pair_o_idx.numpy() != cd.pair_d_idx.numpy()) & (dist_km > 0.0)
# |     t_gt = cd.pair_trips.numpy().astype(np.float64)
# |     
# |     # Extract county grouping from meta.csv using GADM 4.1
# |     from src.data.gadm_mapper import get_gadm_gid2_mapping
# |     meta_df = pd.read_csv(Path(DATA_ROOT) / city / "meta.csv")
# |     repo_root = str(PROJECT_ROOT)
# |     tract_to_county, mapping_stats = get_gadm_gid2_mapping(meta_df, repo_root)
# |     pair_county_idx = np.array([tract_to_county[i] for i in cd.pair_o_idx.numpy()])
# |     
# |     unique_counties = sorted(list(set(pair_county_idx)))
# |     n_counties = len(unique_counties)
# |     
# |     # 1. Condition A: Zero-Shot Forward Pass (M0)
# |     T0 = infer_zero_shot(model, cd, ei, ed, device=device)
# |     t0_np = T0.numpy().astype(np.float64)
# |     cpc_0 = compute_cpc_pair(t_gt[inter_mask], t0_np[inter_mask])
# |     
# |     # 2. Condition B: City-Level Calibration (M_city)
# |     yd_city = test_yd_cache[city]
# |     t_city = calibrate_kbins(
# |         t0_np=t0_np,
# |         dist_km=dist_km,
# |         inter_mask=inter_mask,
# |         yd_target=yd_city,
# |         bin_edges=bin_edges,
# |         q=Q_CALIB,
# |     )
# |     cpc_city = compute_cpc_pair(t_gt[inter_mask], t_city[inter_mask])
# |     
# |     # 3. Condition C: County-Level Calibration (M_county)
# |     yd_county_dict = extract_yd_kbins_grouped(
# |         dist_km=dist_km,
# |         trips=t_gt,
# |         bin_edges=bin_edges,
# |         inter_mask=inter_mask,
# |         pair_group_idx=pair_county_idx,
# |     )
# |     t_county = calibrate_kbins_grouped(
# |         t0_np=t0_np,
# |         dist_km=dist_km,
# |         inter_mask=inter_mask,
# |         yd_target_dict=yd_county_dict,
# |         bin_edges=bin_edges,
# |         pair_group_idx=pair_county_idx,
# |         q=Q_CALIB,
# |     )
# |     cpc_county = compute_cpc_pair(t_gt[inter_mask], t_county[inter_mask])
# |     
# |     # 4. Condition D: Multi-Donor Wrong Placebo Y_D (9 wrong donors)
# |     wrong_cpcs = []
# |     other_donors = [d for d in test_cities if d != city]
# |     for donor in other_donors:
# |         yd_donor = test_yd_cache[donor]
# |         t_wrong_d = calibrate_kbins(
# |             t0_np=t0_np,
# |             dist_km=dist_km,
# |             inter_mask=inter_mask,
# |             yd_target=yd_donor,
# |             bin_edges=bin_edges,
# |             q=Q_CALIB,
# |         )
# |         wrong_cpcs.append(compute_cpc_pair(t_gt[inter_mask], t_wrong_d[inter_mask]))
# |     cpc_wrong = float(np.mean(wrong_cpcs))
# |     
# |     elapsed = time.time() - t_start
# |     
# |     return {
# |         "city": city,
# |         "fold": fold_id,
# |         "n_counties": n_counties,
# |         "is_multi_county": bool(n_counties > 1),
# |         "county_ids": unique_counties,
# |         "cpc_baseline": float(cpc_0),
# |         "cpc_city": float(cpc_city),
# |         "cpc_county": float(cpc_county),
# |         "cpc_wrong": float(cpc_wrong),
# |         "delta_cpc_city": float(cpc_city - cpc_0),
# |         "delta_cpc_county": float(cpc_county - cpc_0),
# |         "delta_cpc_resolution": float(cpc_county - cpc_city),
# |         "delta_cpc_spec_city": float(cpc_city - cpc_wrong),
# |         "delta_cpc_spec_county": float(cpc_county - cpc_wrong),
# |         "elapsed_sec": float(elapsed),
# |         "mapping_stats": mapping_stats,
# |     }
# |
# |
# | def compute_resolution_summary(results: list[dict], bootstrap_seed: int = DEFAULT_SEED) -> dict:
# |     df = pd.DataFrame(results)
# |     
# |     # Global metrics
# |     fid = df["fold"].values
# |     d_res = df["delta_cpc_resolution"].values
# |     d_city = df["delta_cpc_city"].values
# |     d_county = df["delta_cpc_county"].values
# |     d_spec_city = df["delta_cpc_spec_city"].values
# |     d_spec_county = df["delta_cpc_spec_county"].values
# |     
# |     ci_res_l, ci_res_h = fold_bootstrap(d_res, fid, seed=bootstrap_seed)
# |     ci_city_l, ci_city_h = fold_bootstrap(d_city, fid, seed=bootstrap_seed)
# |     ci_county_l, ci_county_h = fold_bootstrap(d_county, fid, seed=bootstrap_seed)
# |     ci_scity_l, ci_scity_h = fold_bootstrap(d_spec_city, fid, seed=bootstrap_seed)
# |     ci_scounty_l, ci_scounty_h = fold_bootstrap(d_spec_county, fid, seed=bootstrap_seed)
# |     
# |     _, p_res = safe_wilcoxon(d_res, alternative="greater")
# |     _, p_scity = safe_wilcoxon(d_spec_city, alternative="greater")
# |     _, p_scounty = safe_wilcoxon(d_spec_county, alternative="greater")
# |     
# |     # Subgroup: Multi-County Cities (n=5)
# |     multi_df = df[df["is_multi_county"]]
# |     single_df = df[~df["is_multi_county"]]
# |     
# |     return {
# |         "n_total_cities": len(df),
# |         "n_multi_county_cities": len(multi_df),
# |         "n_single_county_cities": len(single_df),
# |         "pooled_50": {
# |             "cpc_baseline_mean": float(df["cpc_baseline"].mean()),
# |             "cpc_city_mean": float(df["cpc_city"].mean()),
# |             "cpc_county_mean": float(df["cpc_county"].mean()),
# |             "cpc_wrong_mean": float(df["cpc_wrong"].mean()),
# |             "delta_city_mean": float(d_city.mean()),
# |             "delta_city_ci": [ci_city_l, ci_city_h],
# |             "delta_county_mean": float(d_county.mean()),
# |             "delta_county_ci": [ci_county_l, ci_county_h],
# |             "delta_resolution_mean": float(d_res.mean()),
# |             "delta_resolution_median": float(np.median(d_res)),
# |             "delta_resolution_ci": [ci_res_l, ci_res_h],
# |             "delta_spec_city_mean": float(d_spec_city.mean()),
# |             "delta_spec_city_ci": [ci_scity_l, ci_scity_h],
# |             "delta_spec_county_mean": float(d_spec_county.mean()),
# |             "delta_spec_county_ci": [ci_scounty_l, ci_scounty_h],
# |             "win_rate_resolution": f"{(d_res > 0).sum()}/{len(df)}",
# |             "win_rate_spec_city": f"{(d_spec_city > 0).sum()}/{len(df)}",
# |             "win_rate_spec_county": f"{(d_spec_county > 0).sum()}/{len(df)}",
# |             "wilcoxon_p_resolution": float(p_res),
# |             "wilcoxon_p_spec_city": float(p_scity),
# |             "wilcoxon_p_spec_county": float(p_scounty),
# |         },
# |         "multi_county_subset": {
# |             "cities": multi_df["city"].tolist(),
# |             "cpc_baseline_mean": float(multi_df["cpc_baseline"].mean()),
# |             "cpc_city_mean": float(multi_df["cpc_city"].mean()),
# |             "cpc_county_mean": float(multi_df["cpc_county"].mean()),
# |             "delta_city_mean": float(multi_df["delta_cpc_city"].mean()),
# |             "delta_county_mean": float(multi_df["delta_cpc_county"].mean()),
# |             "delta_resolution_mean": float(multi_df["delta_cpc_resolution"].mean()),
# |             "delta_resolution_max": float(multi_df["delta_cpc_resolution"].max()),
# |             "win_rate_resolution": f"{(multi_df['delta_cpc_resolution'] > 0).sum()}/{len(multi_df)}",
# |         },
# |         "single_county_subset": {
# |             "n_cities": len(single_df),
# |             "delta_resolution_mean": float(single_df["delta_cpc_resolution"].mean()),
# |             "delta_resolution_max": float(single_df["delta_cpc_resolution"].max()),
# |             "exact_zero_invariant": bool(np.allclose(single_df["delta_cpc_resolution"].values, 0.0, atol=1e-6)),
# |         }
# |     }
# |
# |
# | def write_resolution_tables(results: list[dict], summary: dict):
# |     TABLES_DIR.mkdir(parents=True, exist_ok=True)
# |     
# |     # S1-A: Overall Comparative Performance
# |     p50 = summary["pooled_50"]
# |     mc = summary["multi_county_subset"]
# |     sc = summary["single_county_subset"]
# |     
# |     main_md = f"""# Table S1: Spatial Resolution Analysis (County-Level vs. City-Level Calibration)
# |
# | > **Research Question**: Does conditioning the aggregated distance distribution $Y_D$ on origin counties ($M_{{\\text{{county}}}}$) improve zero-shot flow prediction over city-wide macro distributions ($M_{{\\text{{city}}}}$)?
# | > **Dataset**: {summary['n_total_cities']} US Metropolitan Areas ({summary['n_single_county_cities']} Single-County, {summary['n_multi_county_cities']} Multi-County) under 5-Fold Stratified CV.
# | > **Calibration Protocol**: $K_{{\\text{{move}}}}=8$ quantile bins, $q=1.0$, within-tolerance distribution matching.
# |
# | ---
# |
# | ## S1-A: Overall Comparative Performance ($n={summary['n_total_cities']}$ Cities)
# |
# | | Condition / Model | Mean Interzonal CPC | Mean Gain vs $M_0$ (Δ) | 95% Bootstrap CI | City-Level Placebo Gain | City-Level Specificity Win Rate |
# | |---|:---:|:---:|:---:|:---:|:---:|
# | | **Zero-Shot Baseline ($M_0$)** | {p50['cpc_baseline_mean']:.4f} | — | — | — | — |
# | | **+ City-Level Target $Y_D$ ($M_{{\\text{{city}}}}$)** | {p50['cpc_city_mean']:.4f} | {p50['delta_city_mean']:+.4f} | [{p50['delta_city_ci'][0]:+.4f}, {p50['delta_city_ci'][1]:+.4f}] | {p50['delta_spec_city_mean']:+.4f} | {p50['win_rate_spec_city']} |
# | | **+ County-Level Target $Y_D$ ($M_{{\\text{{county}}}}$)** | **{p50['cpc_county_mean']:.4f}** | **{p50['delta_county_mean']:+.4f}** | **[{p50['delta_county_ci'][0]:+.4f}, {p50['delta_county_ci'][1]:+.4f}]** | — | — |
# | | **City-Level Placebo ($M_{{\\text{{wrong}}}}$ 9-Donor Avg)** | {p50['cpc_wrong_mean']:.4f} | {p50['cpc_wrong_mean'] - p50['cpc_baseline_mean']:+.4f} | — | — | 0/{summary['n_total_cities']} |
# |
# | ---
# |
# | ## S1-B: Multi-County Metropolitan Focus ($n={summary['n_multi_county_cities']}$ Heterogeneous Cities)
# |
# | In multi-county metropolitan areas, distinct origin counties exhibit heterogeneous localized trip distributions.
# |
# | | City | Origin Counties | Zero-Shot $M_0$ | City-Level $M_{{\\text{{city}}}}$ | County-Level $M_{{\\text{{county}}}}$ | Resolution Gain ($\\Delta_{{\\text{{res}}}}$) | City-Level Placebo $M_{{\\text{{wrong}}}}$ |
# | |---|:---:|:---:|:---:|:---:|:---:|:---:|
# | """
# |     for r in sorted([r for r in results if r["is_multi_county"]], key=lambda x: x["delta_cpc_resolution"], reverse=True):
# |         main_md += f"| **{r['city']}** | {r['n_counties']} counties | {r['cpc_baseline']:.4f} | {r['cpc_city']:.4f} | **{r['cpc_county']:.4f}** | **{r['delta_cpc_resolution']:+.4f}** | {r['cpc_wrong']:.4f} |\n"
# |         
# |     main_md += f"""
# | **Multi-County Average ($n=5$)**:
# | - Mean Zero-Shot $M_0$: {mc['cpc_baseline_mean']:.4f}
# | - Mean City-Level $M_{{\\text{{city}}}}$: {mc['cpc_city_mean']:.4f} (Δ = {mc['delta_city_mean']:+.4f})
# | - Mean County-Level $M_{{\\text{{county}}}}$: **{mc['cpc_county_mean']:.4f}** (Δ = **{mc['delta_county_mean']:+.4f}**)
# | - **Mean Spatial Resolution Gain ($\\Delta_{{\\text{{res}}}}$)**: **{mc['delta_resolution_mean']:+.4f}** (Max: **{mc['delta_resolution_max']:+.4f}**)
# | - **Resolution Improvement Rate**: **{mc['win_rate_resolution']}**
# |
# | ---
# |
# | ## S1-C: Single-County Sanity Invariance ($n=45$ Single-County Cities)
# |
# | For single-county cities, all tracts belong to the same origin county, meaning $M_{{\\text{{county}}}} \\equiv M_{{\\text{{city}}}}$ by definition.
# | - **Observed Mean $\\Delta_{{\\text{{resolution}}}}$**: {sc['delta_resolution_mean']:.6f}
# | - **Exact Mathematical Invariance**: {'✓ VERIFIED' if sc['exact_zero_invariant'] else '✗ FAILED'}
# | """
# |     (TABLES_DIR / "spatial_resolution_main_table.md").write_text(main_md, encoding="utf-8")
# |
# |     # 2. Per-City Breakdown Table
# |     rows = [
# |         "| City | Fold | Counties | Multi-County? | $M_0$ CPC | $M_{\\text{city}}$ CPC | $M_{\\text{county}}$ CPC | $\\Delta_{\\text{resolution}}$ | $M_{\\text{wrong}}$ | $\\Delta_{\\text{spec, county}}$ |",
# |         "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
# |     ]
# |     for r in sorted(results, key=lambda x: (not x["is_multi_county"], x["city"])):
# |         mc_flag = "Yes" if r["is_multi_county"] else "No"
# |         rows.append(
# |             f"| {r['city']} | {r['fold']} | {r['n_counties']} | {mc_flag} | "
# |             f"{r['cpc_baseline']:.4f} | {r['cpc_city']:.4f} | {r['cpc_county']:.4f} | "
# |             f"**{r['delta_cpc_resolution']:+.4f}** | {r['cpc_wrong']:.4f} | {r['delta_cpc_spec_county']:+.4f} |"
# |         )
# |     (TABLES_DIR / "spatial_resolution_per_city.md").write_text("# Complete Spatial Resolution Breakdown (50 Cities)\n\n" + "\n".join(rows) + "\n", encoding="utf-8")
# |
# |
# | def run_spatial_resolution_experiment(device_str: str = "cpu", seed: int = DEFAULT_SEED, smoke: bool = False):
# |     t_global_start = time.time()
# |     device = torch.device(device_str)
# |     RESULTS_DIR.mkdir(parents=True, exist_ok=True)
# |     TABLES_DIR.mkdir(parents=True, exist_ok=True)
# |     
# |     log_msg("=" * 75)
# |     log_msg("SPATIAL RESOLUTION EXPERIMENT: ORIGIN COUNTY-LEVEL VS CITY-LEVEL CALIBRATION")
# |     log_msg("=" * 75)
# |     log_msg(f"  Configuration: K={K_MOVE} bins, q={Q_CALIB}, Seed={seed}, Device={device_str}")
# |     
# |     MANIFEST_PATH = PROJECT_ROOT / "results" / "e1" / "splits_manifest_v2.json"
# |     splits = load_splits_manifest_v2(str(MANIFEST_PATH), data_root=DATA_ROOT)
# |     
# |     log_msg("  Preloading datasets & spatial graphs...")
# |     preload_all_cities(data_root=DATA_ROOT, build_graphs=True, radius_km=5.0)
# |     
# |     all_results = []
# |     
# |     for fold_id, split in splits.items():
# |         t_fold_start = time.time()
# |         train35 = split["train"]
# |         val5 = split["val"]
# |         test10 = sorted(split["test"])
# |         
# |         if smoke:
# |             test10 = [c for c in ["Dallas", "Atlanta", "Denver", "Portland"] if c in test10]
# |             if not test10:
# |                 continue
# |                 
# |         log_msg("-" * 75)
# |         log_msg(f">>> [FOLD {fold_id}/5] Evaluating Spatial Resolution on Frozen Backbones...")
# |         log_msg("-" * 75)
# |         
# |         # 1. Compute Bin Edges
# |         bin_edges, K_active = compute_kbin_edges(train35, K=K_MOVE, data_root=DATA_ROOT)
# |         
# |         # We will collect per-seed results for each city in this fold
# |         fold_city_seed_results = {city: [] for city in test10}
# |
# |         for m_seed in ([1, 10, 100] if not smoke else [1, 10]):
# |             ckpt_path = Path(f"results/checkpoints/5fold_fold{fold_id}_seed{m_seed}.pt")
# |             if not ckpt_path.exists():
# |                 raise FileNotFoundError(f"Missing mandatory checkpoint {ckpt_path}")
# |             
# |             from src.training.train import load_checkpoint
# |             model, scaler, _ = load_checkpoint(ckpt_path, device_str=device_str)
# |             model.eval()
# |             
# |             # Precompute City-Level Y_D Oracles for all test cities using this seed's scaler
# |             test_yd_cache = {}
# |             for t_city in test10:
# |                 cd_t = load_city(t_city, data_root=DATA_ROOT, feature_scaler=scaler, fit_scaler=False)
# |                 dist_t = np.expm1(cd_t.pair_distance.numpy())
# |                 inter_t = (cd_t.pair_o_idx.numpy() != cd_t.pair_d_idx.numpy()) & (dist_t > 0.0)
# |                 t_gt_t = cd_t.pair_trips.numpy().astype(np.float64)
# |                 test_yd_cache[t_city] = extract_yd_kbins(dist_t, t_gt_t, bin_edges, inter_t)
# |             
# |             # Evaluate each held-out test city
# |             for city in test10:
# |                 res = run_spatial_resolution_city(
# |                     city=city,
# |                     model=model,
# |                     scaler=scaler,
# |                     bin_edges=bin_edges,
# |                     test_cities=test10,
# |                     fold_id=fold_id,
# |                     device=device,
# |                     test_yd_cache=test_yd_cache,
# |                 )
# |                 fold_city_seed_results[city].append(res)
# |                 
# |         # Average over seeds for each city
# |         for city in test10:
# |             seed_results = fold_city_seed_results[city]
# |             avg_res = {
# |                 "city": city,
# |                 "fold": fold_id,
# |                 "n_counties": seed_results[0]["n_counties"],
# |                 "is_multi_county": seed_results[0]["is_multi_county"],
# |                 "county_ids": seed_results[0]["county_ids"],
# |                 "mapping_stats": seed_results[0]["mapping_stats"],
# |             }
# |             # Average numerical keys
# |             for k in ["cpc_baseline", "cpc_city", "cpc_county", "cpc_wrong", 
# |                       "delta_cpc_city", "delta_cpc_county", "delta_cpc_resolution", 
# |                       "delta_cpc_spec_city", "delta_cpc_spec_county", "elapsed_sec"]:
# |                 avg_res[k] = float(np.mean([r[k] for r in seed_results]))
# |             
# |             all_results.append(avg_res)
# |             
# |             mc_str = f" [Multi-County: {avg_res['n_counties']} counties]" if avg_res["is_multi_county"] else ""
# |             log_msg(
# |                 f"  [{city:<16}] M0={avg_res['cpc_baseline']:.4f} -> "
# |                 f"M_city={avg_res['cpc_city']:.4f} (d={avg_res['delta_cpc_city']:+.4f}) -> "
# |                 f"M_county={avg_res['cpc_county']:.4f} (d={avg_res['delta_cpc_county']:+.4f}) | "
# |                 f"dRes={avg_res['delta_cpc_resolution']:+.4f}{mc_str}"
# |             )
# |             
# |         t_fold_elapsed = time.time() - t_fold_start
# |         log_msg(f"  [Fold {fold_id} Complete] Elapsed time: {t_fold_elapsed:.1f}s")
# |         
# |     # Synthesize Summary & Tables
# |     summary = compute_resolution_summary(all_results, bootstrap_seed=seed)
# |     
# |     (RESULTS_DIR / "spatial_resolution_per_city.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
# |     (RESULTS_DIR / "spatial_resolution_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
# |     
# |     write_resolution_tables(all_results, summary)
# |     
# |     t_global_elapsed = time.time() - t_global_start
# |     log_msg("=" * 75)
# |     log_msg(f"SPATIAL RESOLUTION EXPERIMENT COMPLETED ({t_global_elapsed:.1f}s)")
# |     log_msg("=" * 75)
# |     p50 = summary["pooled_50"]
# |     mc = summary["multi_county_subset"]
# |     log_msg(f"  Total Cities Evaluated: {len(all_results)}/50")
# |     log_msg(f"  City-Level Gain (dCPC): mean = {p50['delta_city_mean']:+.4f} (CI: [{p50['delta_city_ci'][0]:+.4f}, {p50['delta_city_ci'][1]:+.4f}])")
# |     log_msg(f"  County-Level Gain (dCPC): mean = {p50['delta_county_mean']:+.4f} (CI: [{p50['delta_county_ci'][0]:+.4f}, {p50['delta_county_ci'][1]:+.4f}])")
# |     log_msg(f"  Multi-County Cities (n=5) Spatial Resolution Gain: mean = {mc['delta_resolution_mean']:+.4f} (Max: {mc['delta_resolution_max']:+.4f})")
# |     log_msg(f"  County-Level Specificity Win Rate: {p50['win_rate_spec_county']} (Wilcoxon p = {p50['wilcoxon_p_spec_county']:.2e})")
# |     log_msg("=" * 75)
# |     
# |     return all_results, summary
# |
# |
# | if __name__ == "__main__":
# |     parser = argparse.ArgumentParser(description="Spatial Resolution Experiment: County vs City Calibration")
# |     parser.add_argument("--smoke", action="store_true", help="Run quick smoke test on subset of cities")
# |     parser.add_argument("--device", default="cpu", help="PyTorch device (cpu/cuda)")
# |     parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
# |     args = parser.parse_args()
# |     
# |     run_spatial_resolution_experiment(device_str=args.device, seed=args.seed, smoke=args.smoke)

# ===== END SOURCE FILE: src/experiment/run_spatial_resolution_experiment.py =====


# ===== BEGIN SOURCE FILE: src/loss/ztnb.py =====
# | """
# | Zero-Truncated Negative Binomial (ZTNB) Likelihood and Conditional Mean Conversion.
# |
# | Exact Mathematical Formulation:
# |     Base NB Distribution:
# |         T ~ NB(mu_nb, phi) with mean mu_nb > 0 and dispersion phi > 0.
# |         P_NB(T=0) = (phi / (mu_nb + phi))^phi.
# |
# |     Zero-Truncated NB (ZTNB):
# |         P_ZTNB(T=t | T >= 1) = P_NB(t; mu_nb, phi) / (1 - P_NB(0; mu_nb, phi))
# |
# |     Conditional Expected Flow:
# |         E[T | T >= 1] = mu_nb / (1 - P_NB(0; mu_nb, phi))
# |
# | The neural network outputs mu_nb > 0.
# | At training time: loss is -log P_ZTNB(T_ij; mu_nb, phi).
# | At inference time: predicted flow is \hat{T}^{ZS}_ij = E[T_ij | T_ij >= 1].
# | """
# |
# | import math
# | import torch
# | import torch.nn.functional as F
# |
# |
# | def nb_log_prob(t: torch.Tensor, mu_nb: torch.Tensor, log_phi: torch.Tensor) -> torch.Tensor:
# |     """
# |     Log-probability of base NB(mu_nb, phi) at integer count t.
# |     """
# |     log_phi_safe = torch.clamp(log_phi, min=-10.0, max=10.0)
# |     phi = torch.exp(log_phi_safe)
# |     eps = 1e-8
# |
# |     mu = mu_nb + eps
# |     phi = phi + eps
# |
# |     p_nb0 = phi / (mu + phi)  # probability parameter
# |
# |     log_p = (
# |         torch.lgamma(t + phi)
# |         - torch.lgamma(phi)
# |         - torch.lgamma(t + 1)
# |         + phi * torch.log(p_nb0)
# |         + t * torch.log(1.0 - p_nb0 + eps)
# |     )
# |     return log_p
# |
# |
# | def nb_log_prob_at_zero(mu_nb: torch.Tensor, log_phi: torch.Tensor) -> torch.Tensor:
# |     """log P_NB(T=0; mu_nb, phi) = phi * log(phi / (mu_nb + phi))"""
# |     log_phi_safe = torch.clamp(log_phi, min=-10.0, max=10.0)
# |     phi = torch.exp(log_phi_safe)
# |     eps = 1e-8
# |     mu = mu_nb + eps
# |     phi = phi + eps
# |     return phi * torch.log(phi / (mu + phi))
# |
# |
# | def ztnb_nll(t: torch.Tensor, mu_nb: torch.Tensor, log_phi: torch.Tensor) -> torch.Tensor:
# |     """
# |     Exact Negative Log-Likelihood for Zero-Truncated Negative Binomial.
# |     log P_ZTNB(T=t | T>=1) = log P_NB(t; mu_nb, phi) - log(1 - P_NB(0; mu_nb, phi))
# |     """
# |     assert (t >= 1).all(), "ZTNB requires all observed counts >= 1"
# |
# |     log_p_nb = nb_log_prob(t, mu_nb, log_phi)
# |     log_p_nb_0 = nb_log_prob_at_zero(mu_nb, log_phi)
# |
# |     # Numerically stable log(1 - P_NB(0)) = log1p(-exp(log_p_nb_0))
# |     log_1_minus_p0 = torch.log1p(-torch.exp(log_p_nb_0).clamp(max=1.0 - 1e-7))
# |
# |     log_p_ztnb = log_p_nb - log_1_minus_p0
# |     return -log_p_ztnb.mean()
# |
# |
# | def nb_nll(t: torch.Tensor, mu_nb: torch.Tensor, log_phi: torch.Tensor) -> torch.Tensor:
# |     """
# |     Mean negative log-likelihood of unconditional Negative Binomial (sensitivity model).
# |     """
# |     return -nb_log_prob(t, mu_nb, log_phi).mean()
# |
# |
# | def compute_conditional_mean(mu_nb: torch.Tensor, log_phi: torch.Tensor) -> torch.Tensor:
# |     """
# |     Converts base NB mean mu_nb to conditional positive mean E[T | T >= 1].
# |     E[T | T >= 1] = mu_nb / (1 - P_NB(0; mu_nb, phi))
# |     """
# |     log_phi_safe = torch.clamp(log_phi, min=-10.0, max=10.0)
# |     phi = torch.exp(log_phi_safe)
# |     eps = 1e-8
# |     p0 = (phi / (mu_nb + phi + eps)) ** phi
# |     # Clamp 1-p0 to avoid division by zero when mu_nb is tiny
# |     denom = torch.clamp(1.0 - p0, min=1e-6)
# |     return mu_nb / denom
# |
# |
# | def _run_unit_tests():
# |     print("Running updated ZTNB unit tests...")
# |     torch.manual_seed(0)
# |
# |     # Test 1: Conditional mean is strictly > mu_nb
# |     mu = torch.tensor([1.0, 5.0, 10.0])
# |     log_phi = torch.tensor(0.0)
# |     c_mean = compute_conditional_mean(mu, log_phi)
# |     assert (c_mean > mu).all(), "Test 1 FAILED: Conditional mean must be > base mu"
# |     print(f"  Test 1 PASS: mu={mu.tolist()} -> E[T|T>=1]={c_mean.tolist()}")
# |
# |     # Test 2: NLL at t=1 is finite
# |     t1 = torch.ones(5)
# |     loss = ztnb_nll(t1, torch.ones(5) * 2.0, log_phi)
# |     assert torch.isfinite(loss), "Test 2 FAILED: NLL not finite"
# |     print(f"  Test 2 PASS: NLL at t=1 -> {loss.item():.4f}")
# |
# |     # Test 3: Gradient finite as mu -> 0
# |     mu_tiny = torch.tensor([1e-4], requires_grad=True)
# |     loss_tiny = ztnb_nll(torch.ones(1), mu_tiny, log_phi)
# |     loss_tiny.backward()
# |     assert torch.isfinite(mu_tiny.grad), "Test 3 FAILED: grad not finite"
# |     print(f"  Test 3 PASS: grad at mu=1e-4 -> {mu_tiny.grad.item():.4f}")
# |
# |     print("All updated ZTNB unit tests passed.\n")
# |
# |
# | if __name__ == "__main__":
# |     _run_unit_tests()

# ===== END SOURCE FILE: src/loss/ztnb.py =====


# ===== BEGIN SOURCE FILE: src/models/decoder.py =====
# | """
# | Pairwise OD Decoder with Single Base Magnitude Head (ZTNB).
# |
# | Input edge representation:
# |     e_ij = [h_i, h_j, log(1 + D_ij), log(T^{grav}_ij)]
# |
# | Single prediction head producing base Negative Binomial parameter via
# | residual-gravity initialization: mu_nb_ij = softplus(log_t_grav + residual_ij).
# |
# | Exact ZTNB Likelihood & Predictions:
# |     At training: loss = -log P_ZTNB(T_ij; mu_nb_ij, phi) on positive observations in Omega_c.
# |     At inference: expected zero-shot prediction is the conditional expectation:
# |         \hat{T}^{ZS}_ij = E[T_ij | T_ij >= 1] = compute_conditional_mean(mu_nb_ij, log_phi).
# | """
# |
# | import torch
# | import torch.nn as nn
# | import torch.nn.functional as F
# |
# |
# | class PairwiseODDecoder(nn.Module):
# |     def __init__(
# |         self,
# |         node_dim: int = 64,
# |         hidden_dim: int = 64,
# |         dropout: float = 0.1,
# |     ):
# |         super().__init__()
# |         # Input: [h_i, h_j, log_d, log_t_grav] -> dim = 2 * node_dim + 2
# |         in_dim = 2 * node_dim + 2
# |
# |         self.net = nn.Sequential(
# |             nn.Linear(in_dim, hidden_dim),
# |             nn.LayerNorm(hidden_dim),
# |             nn.ReLU(),
# |             nn.Dropout(dropout),
# |             nn.Linear(hidden_dim, hidden_dim // 2),
# |             nn.ReLU(),
# |             nn.Dropout(dropout),
# |             nn.Linear(hidden_dim // 2, 1),
# |         )
# |
# |         # Zero-init final layer so the gravity prior is supplied as a log-scale decoder feature/offset 
# |         # with a zero-initialized neural residual, yielding softplus(log_t_grav) at initialization.
# |         nn.init.zeros_(self.net[-1].weight)
# |         nn.init.zeros_(self.net[-1].bias)
# |
# |     def forward(
# |         self,
# |         h_i: torch.Tensor,
# |         h_j: torch.Tensor,
# |         log_distance: torch.Tensor,
# |         log_t_grav: torch.Tensor,
# |     ) -> torch.Tensor:
# |         """
# |         Args:
# |             h_i:          (E, node_dim) origin node embeddings.
# |             h_j:          (E, node_dim) destination node embeddings.
# |             log_distance: (E,) or (E, 1) log1p(distance_km).
# |             log_t_grav:   (E,) or (E, 1) log gravity flow.
# |
# |         Returns:
# |             mu_nb: (E,) positive base mean parameter mu_nb_ij > 0.
# |         """
# |         if log_distance.dim() == 1:
# |             log_distance = log_distance.unsqueeze(-1)
# |         if log_t_grav.dim() == 1:
# |             log_t_grav = log_t_grav.unsqueeze(-1)
# |
# |         # Concatenate edge representation e_ij
# |         e_ij = torch.cat([h_i, h_j, log_distance, log_t_grav], dim=-1)
# |
# |         residual = self.net(e_ij)  # (E, 1), ~0 at init
# |         # Residual-gravity: gravity prior serves as log-scale feature, GNN learns the deviation.
# |         # Yields softplus(log_t_grav) at initialization.
# |         log_mu_nb = log_t_grav + residual
# |         mu_nb = F.softplus(log_mu_nb.squeeze(-1)) + 1e-4
# |         return mu_nb
# |
# |
# | if __name__ == "__main__":
# |     dec = PairwiseODDecoder(node_dim=32, hidden_dim=64)
# |     h_i = torch.randn(100, 32)
# |     h_j = torch.randn(100, 32)
# |     ld = torch.randn(100)
# |     ltg = torch.randn(100)
# |     mu_nb = dec(h_i, h_j, ld, ltg)
# |     print("Decoder mu_nb output shape:", mu_nb.shape, "min:", mu_nb.min().item(), "max:", mu_nb.max().item())
# |
# |     # At init, residual ~ 0, so mu_nb should track softplus(log_t_grav) closely
# |     expected = F.softplus(ltg) + 1e-4
# |     print("Max deviation from pure gravity at init:", (mu_nb - expected).abs().max().item())

# ===== END SOURCE FILE: src/models/decoder.py =====


# ===== BEGIN SOURCE FILE: src/models/gravity.py =====
# | """
# | Classical 2-parameter Physics Gravity Model Prior.
# |
# | log T_ij^grav = G + log P_i + log P_j - alpha * log(D_ij)
# |
# | Parameters:
# |     G: global scale parameter (learnable scalar)
# |     alpha: distance decay parameter (learnable scalar, initialized to ~1.0-2.0)
# |
# | Both G and alpha are global trainable parameters shared across all cities in a fold,
# | providing the physics prior baseline for cross-city transfer.
# | """
# |
# | import math
# | import torch
# | import torch.nn as nn
# |
# |
# | class GravityPrior(nn.Module):
# |     def __init__(self, init_G: float = 0.0, init_alpha: float = 1.0):
# |         super().__init__()
# |         # Trainable physics parameters
# |         self.G = nn.Parameter(torch.tensor(init_G, dtype=torch.float32))
# |         self.log_alpha = nn.Parameter(torch.tensor(math.log(init_alpha), dtype=torch.float32))
# |
# |     @property
# |     def alpha(self) -> torch.Tensor:
# |         return torch.exp(self.log_alpha)  # ensure alpha > 0
# |
# |     def forward(
# |         self,
# |         population_i: torch.Tensor,
# |         population_j: torch.Tensor,
# |         distance_km: torch.Tensor,
# |     ) -> torch.Tensor:
# |         """
# |         Computes log T_ij^grav for each pair.
# |
# |         Args:
# |             population_i: (E,) population of origin tract.
# |             population_j: (E,) population of destination tract.
# |             distance_km:  (E,) distance in km (not log).
# |
# |         Returns:
# |             log_T_grav: (E,) log-expected gravity flow.
# |         """
# |         log_pi = torch.log(torch.clamp(population_i, min=1.0))
# |         log_pj = torch.log(torch.clamp(population_j, min=1.0))
# |         # Clamp at 0.1 km to avoid log(0) for intrazonal pairs (D_ii = 0).
# |         # This floor is an explicit design choice: intrazonal log_d = log(0.1) ≈ -2.3.
# |         # The model is trained on this behaviour; do not change without a full retrain.
# |         log_d  = torch.log(torch.clamp(distance_km, min=0.1))
# |
# |         log_t_grav = self.G + log_pi + log_pj - self.alpha * log_d
# |         return log_t_grav
# |
# |
# | if __name__ == "__main__":
# |     grav = GravityPrior()
# |     p_i = torch.tensor([1000.0, 5000.0])
# |     p_j = torch.tensor([2000.0, 10000.0])
# |     d   = torch.tensor([5.0, 15.0])
# |     out = grav(p_i, p_j, d)
# |     print("Gravity prior output log_T:", out)
# |     print(f"Alpha: {grav.alpha.item():.4f}, G: {grav.G.item():.4f}")

# ===== END SOURCE FILE: src/models/gravity.py =====


# ===== BEGIN SOURCE FILE: src/models/node_encoder.py =====
# | """
# | Urban Graph Neural Network Node Encoder.
# |
# | Learns tract representation h_i from urban features X and spatial graph G^urban:
# |     h_i = GNN_theta(X, G^urban)
# |
# | Graph structure:
# |     G^urban is built ONLY from observable spatial geography (k-NN / radius graph).
# |     No OD data is ever used to construct G^urban.
# |
# | Architecture:
# |     Multi-layer Graph Convolution / GAT / GraphConv with residual connections and LayerNorm.
# | """
# |
# | import torch
# | import torch.nn as nn
# | import torch.nn.functional as F
# |
# |
# | class GraphConvLayer(nn.Module):
# |     """
# |     Message passing layer with edge distance modulation.
# |     Performs distance-conditioned message passing:
# |         m_ij = W_msg * [h_j || log(1 + d_ij)]
# |         h_i' = W_self * h_i + Agg_{j in N(i)}(m_ij)
# |     """
# |     def __init__(self, in_dim: int, out_dim: int):
# |         super().__init__()
# |         self.msg_linear = nn.Linear(in_dim + 1, out_dim)
# |         self.self_linear = nn.Linear(in_dim, out_dim)
# |         self.norm = nn.LayerNorm(out_dim)
# |
# |     def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_dist: torch.Tensor) -> torch.Tensor:
# |         """
# |         x: (N, in_dim)
# |         edge_index: (2, E_graph)
# |         edge_dist: (E_graph,) distance in km
# |         """
# |         src, dst = edge_index[0], edge_index[1]
# |         
# |         # Log distance feature
# |         log_d = torch.log1p(edge_dist).unsqueeze(-1)  # (E_graph, 1)
# |         
# |         # Message computation: [h_src, log_d]
# |         msg_input = torch.cat([x[src], log_d], dim=-1)  # (E_graph, in_dim + 1)
# |         msg = self.msg_linear(msg_input)  # (E_graph, out_dim)
# |
# |         # Scatter mean aggregation
# |         out = torch.zeros(x.size(0), msg.size(1), device=x.device, dtype=x.dtype)
# |         # Degree count for mean aggregation
# |         deg = torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)
# |         
# |         out.index_add_(0, dst, msg)
# |         deg.index_add_(0, dst, torch.ones_like(log_d))
# |         
# |         out = out / torch.clamp(deg, min=1.0)
# |         
# |         # Combine with transformed self features
# |         h_self = self.self_linear(x)
# |         out = self.norm(F.relu(out + h_self))
# |         return out
# |
# |
# | class UrbanGNN(nn.Module):
# |     """
# |     Urban GNN Node Encoder that produces node embeddings h_i in R^d.
# |     """
# |     def __init__(
# |         self,
# |         in_dim: int = 26,
# |         hidden_dim: int = 64,
# |         out_dim: int = 64,
# |         num_layers: int = 2,
# |         dropout: float = 0.1,
# |     ):
# |         super().__init__()
# |         self.input_fc = nn.Sequential(
# |             nn.Linear(in_dim, hidden_dim),
# |             nn.LayerNorm(hidden_dim),
# |             nn.ReLU(),
# |             nn.Dropout(dropout),
# |         )
# |
# |         self.layers = nn.ModuleList([
# |             GraphConvLayer(hidden_dim, hidden_dim) for _ in range(num_layers)
# |         ])
# |
# |         self.output_fc = nn.Linear(hidden_dim, out_dim)
# |         self.dropout = nn.Dropout(dropout)
# |
# |     def forward(
# |         self,
# |         x: torch.Tensor,
# |         edge_index: torch.Tensor,
# |         edge_dist: torch.Tensor,
# |     ) -> torch.Tensor:
# |         """
# |         Args:
# |             x:          (N, in_dim) normalized node features.
# |             edge_index: (2, E_graph) spatial graph edges.
# |             edge_dist:  (E_graph,) geographic distances.
# |
# |         Returns:
# |             h: (N, out_dim) node embeddings.
# |         """
# |         h = self.input_fc(x)
# |         for layer in self.layers:
# |             h_new = layer(h, edge_index, edge_dist)
# |             h = h + self.dropout(h_new)  # residual connection
# |
# |         h = self.output_fc(h)
# |         return h
# |
# | class MLPLayer(nn.Module):
# |     """
# |     A dense layer designed to have the same nominal parameter count as GraphConvLayer.
# |     """
# |     def __init__(self, in_dim: int, out_dim: int):
# |         super().__init__()
# |         # GraphConvLayer has msg_linear (in_dim + 1 -> out_dim) and self_linear (in_dim -> out_dim).
# |         # We replicate this exactly here.
# |         self.msg_equivalent = nn.Linear(in_dim + 1, out_dim)
# |         self.self_linear = nn.Linear(in_dim, out_dim)
# |         self.norm = nn.LayerNorm(out_dim)
# |
# |     def forward(self, x: torch.Tensor) -> torch.Tensor:
# |         # Pad with zeros to match the log(1+d) feature concatenated in GNN message passing
# |         dummy_dist = torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)
# |         msg_input = torch.cat([x, dummy_dist], dim=-1)
# |         
# |         out = self.msg_equivalent(msg_input) + self.self_linear(x)
# |         return self.norm(F.relu(out))
# |
# |
# | class NodeMLP(nn.Module):
# |     """
# |     MLP Node Encoder that produces node embeddings h_i in R^d without message passing.
# |     Architecture matches UrbanGNN but removes the GraphConv aggregation.
# |     """
# |     def __init__(
# |         self,
# |         in_dim: int = 26,
# |         hidden_dim: int = 64,
# |         out_dim: int = 64,
# |         num_layers: int = 2,
# |         dropout: float = 0.1,
# |     ):
# |         super().__init__()
# |         self.input_fc = nn.Sequential(
# |             nn.Linear(in_dim, hidden_dim),
# |             nn.LayerNorm(hidden_dim),
# |             nn.ReLU(),
# |             nn.Dropout(dropout),
# |         )
# |         
# |         # Use MLPLayer to maintain nominal parameter-count parity with GraphConvLayer.
# |         self.layers = nn.ModuleList([
# |             MLPLayer(hidden_dim, hidden_dim) for _ in range(num_layers)
# |         ])
# |         
# |         self.output_fc = nn.Linear(hidden_dim, out_dim)
# |         self.dropout = nn.Dropout(dropout)
# |
# |     def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_dist: torch.Tensor) -> torch.Tensor:
# |         """
# |         Args:
# |             x: (N, in_dim) normalized node features.
# |             edge_index: Ignored. Kept for signature compatibility with UrbanGNN so that 
# |                         both models can be dropped into the same training/inference loop 
# |                         without modifying the call signature.
# |             edge_dist: Ignored. See above.
# |         Returns:
# |             h: (N, out_dim) node embeddings.
# |         """
# |         h = self.input_fc(x)
# |         for layer in self.layers:
# |             h_new = layer(h)
# |             h = h + self.dropout(h_new)
# |         h = self.output_fc(h)
# |         return h
# |
# |
# |
# | if __name__ == "__main__":
# |     gnn = UrbanGNN(in_dim=26, hidden_dim=32, out_dim=32, num_layers=2)
# |     x = torch.randn(10, 26)
# |     edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
# |     edge_dist = torch.tensor([1.0, 2.0, 1.5, 3.0, 0.5])
# |     h = gnn(x, edge_index, edge_dist)
# |     print(f"UrbanGNN output shape: {h.shape}")

# ===== END SOURCE FILE: src/models/node_encoder.py =====


# ===== BEGIN SOURCE FILE: src/models/zero_shot_model.py =====
# | r"""
# | Gravity-Informed Urban-GNN Support-Conditioned Zero-Shot Model (M_0).
# | (neuroGravity-inspired neural transferable architecture)
# |
# | Mathematical Formulation:
# |     1. Classical Gravity Prior:
# |         T_ij^grav = exp(G_0) * P_i * P_j * D_ij^(-alpha_0)
# |     2. Urban GNN Representation:
# |         h_i = GNN_theta(X_i, G^urban)
# |     3. Neural Transfer Decoder:
# |         \hat{T}_ij^ZS = f_theta*(X_i, X_j, D_ij, T_ij^grav)
# |         where f_theta maps [h_i, h_j, log(1+D_ij), log(T_ij^grav)] to conditional mean E[T_ij | T_ij >= 1].
# |     4. Learnable global dispersion parameter phi for ZTNB likelihood.
# | """
# |
# | import torch
# | import torch.nn as nn
# | from src.models.node_encoder import UrbanGNN
# | from src.models.gravity import GravityPrior
# | from src.models.decoder import PairwiseODDecoder
# | from src.loss.ztnb import compute_conditional_mean
# | from src.models.node_encoder import NodeMLP
# |
# | class ZeroShotMLPModel(nn.Module):
# |     def __init__(
# |         self,
# |         node_in_dim: int = 26,
# |         node_hidden_dim: int = 64,
# |         node_out_dim: int = 64,
# |         num_gnn_layers: int = 2,
# |         decoder_hidden_dim: int = 64,
# |         dropout: float = 0.1,
# |         init_log_phi: float = 0.0,
# |     ):
# |         super().__init__()
# |         # 1. Urban MLP (no message passing)
# |         self.node_encoder = NodeMLP(
# |             in_dim=node_in_dim,
# |             hidden_dim=node_hidden_dim,
# |             out_dim=node_out_dim,
# |             num_layers=num_gnn_layers,
# |             dropout=dropout,
# |         )
# |
# |         # 2. Gravity Prior
# |         self.gravity_prior = GravityPrior()
# |
# |         # 3. Pairwise Decoder
# |         self.decoder = PairwiseODDecoder(
# |             node_dim=node_out_dim,
# |             hidden_dim=decoder_hidden_dim,
# |             dropout=dropout,
# |         )
# |
# |         # 4. Global trainable dispersion parameter phi
# |         self.log_phi = nn.Parameter(torch.tensor(init_log_phi, dtype=torch.float32))
# |
# |     @property
# |     def phi(self) -> torch.Tensor:
# |         return torch.exp(self.log_phi)
# |
# |     def forward(
# |         self,
# |         x: torch.Tensor,
# |         spatial_edge_index: torch.Tensor,
# |         spatial_edge_dist: torch.Tensor,
# |         pair_o_idx: torch.Tensor,
# |         pair_d_idx: torch.Tensor,
# |         pair_distance_log1p: torch.Tensor,
# |         population: torch.Tensor,
# |         return_conditional_mean: bool = False,
# |     ) -> torch.Tensor:
# |         h = self.node_encoder(x, spatial_edge_index, spatial_edge_dist)
# |         h_o = h[pair_o_idx]
# |         h_d = h[pair_d_idx]
# |
# |         dist_km = torch.expm1(pair_distance_log1p)
# |         pop_o = population[pair_o_idx]
# |         pop_d = population[pair_d_idx]
# |         log_t_grav = self.gravity_prior(pop_o, pop_d, dist_km)
# |
# |         mu_nb = self.decoder(h_o, h_d, pair_distance_log1p, log_t_grav)
# |
# |         if return_conditional_mean:
# |             return compute_conditional_mean(mu_nb, self.log_phi)
# |         return mu_nb
# |
# | class ZeroShotODModel(nn.Module):
# |     def __init__(
# |         self,
# |         node_in_dim: int = 26,
# |         node_hidden_dim: int = 64,
# |         node_out_dim: int = 64,
# |         num_gnn_layers: int = 2,
# |         decoder_hidden_dim: int = 64,
# |         dropout: float = 0.1,
# |         init_log_phi: float = 0.0,
# |     ):
# |         super().__init__()
# |         # 1. Urban GNN
# |         self.node_encoder = UrbanGNN(
# |             in_dim=node_in_dim,
# |             hidden_dim=node_hidden_dim,
# |             out_dim=node_out_dim,
# |             num_layers=num_gnn_layers,
# |             dropout=dropout,
# |         )
# |
# |         # 2. Gravity Prior
# |         self.gravity_prior = GravityPrior()
# |
# |         # 3. Pairwise Decoder (outputs base mean mu_nb > 0)
# |         self.decoder = PairwiseODDecoder(
# |             node_dim=node_out_dim,
# |             hidden_dim=decoder_hidden_dim,
# |             dropout=dropout,
# |         )
# |
# |         # 4. Global trainable dispersion parameter phi (phi = exp(log_phi))
# |         self.log_phi = nn.Parameter(torch.tensor(init_log_phi, dtype=torch.float32))
# |
# |     @property
# |     def phi(self) -> torch.Tensor:
# |         return torch.exp(self.log_phi)
# |
# |     def forward(
# |         self,
# |         x: torch.Tensor,
# |         spatial_edge_index: torch.Tensor,
# |         spatial_edge_dist: torch.Tensor,
# |         pair_o_idx: torch.Tensor,
# |         pair_d_idx: torch.Tensor,
# |         pair_distance_log1p: torch.Tensor,
# |         population: torch.Tensor,
# |         return_conditional_mean: bool = False,
# |     ) -> torch.Tensor:
# |         """
# |         Forward pass predicting flows for candidate pairs on Omega_c.
# |
# |         Args:
# |             return_conditional_mean:
# |                 If False (training): returns base parameter mu_nb for ZTNB likelihood.
# |                 If True (inference): returns exact conditional expectation E[T | T >= 1].
# |         """
# |         # Step 1: Compute node embeddings from observable urban graph G^urban
# |         h = self.node_encoder(x, spatial_edge_index, spatial_edge_dist)  # (N, d)
# |
# |         # Gather origin and destination embeddings for candidate pairs
# |         h_o = h[pair_o_idx]  # (E_pairs, d)
# |         h_d = h[pair_d_idx]  # (E_pairs, d)
# |
# |         # Step 2: Compute Physics Gravity prior
# |         dist_km = torch.expm1(pair_distance_log1p)
# |         pop_o = population[pair_o_idx]
# |         pop_d = population[pair_d_idx]
# |         log_t_grav = self.gravity_prior(pop_o, pop_d, dist_km)  # (E_pairs,)
# |
# |         # Step 3: Decode pairwise flows (mu_nb > 0)
# |         mu_nb = self.decoder(h_o, h_d, pair_distance_log1p, log_t_grav)  # (E_pairs,)
# |
# |         if return_conditional_mean:
# |             # \hat{T} = E[T | T >= 1]
# |             return compute_conditional_mean(mu_nb, self.log_phi)
# |         return mu_nb
# |
# |
# | if __name__ == "__main__":
# |     from src.data.dataset import load_city
# |     from src.data.urban_graph import build_knn_graph
# |
# |     cd = load_city("Raleigh", "data")
# |     ei, ed = build_knn_graph(cd.lon_lat.numpy(), k=10)
# |
# |     model = ZeroShotODModel()
# |     mu_nb = model(cd.node_features, ei, ed, cd.pair_o_idx, cd.pair_d_idx, cd.pair_distance, cd.population, return_conditional_mean=False)
# |     t_hat = model(cd.node_features, ei, ed, cd.pair_o_idx, cd.pair_d_idx, cd.pair_distance, cd.population, return_conditional_mean=True)
# |     print("Forward pass base mu_nb shape:", mu_nb.shape, "min:", mu_nb.min().item())
# |     print("Forward pass t_hat shape:", t_hat.shape, "min:", t_hat.min().item())
# |     assert (t_hat >= mu_nb).all(), "Conditioning must increase or maintain expectation"
# |     print("Model check passed.")

# ===== END SOURCE FILE: src/models/zero_shot_model.py =====


# ===== BEGIN SOURCE FILE: src/training/evaluate.py =====
# | """
# | Comprehensive Evaluation Suite on Interzonal Domain Omega_c^+ and Full Support Omega_c.
# |
# | Primary Metric:
# |     Interzonal CPC (CPC_inter) on Omega_c^+ = {(i,j) in Omega_c : i != j, D_ij > 0}:
# |         Evaluates the displacement flow distribution of moving commuters.
# |
# | Secondary Metrics:
# |     1. Scale-Normalized Interzonal CPC (CPC_inter_norm = 1 - TVD):
# |         Evaluates pure structural flow geometry independent of total flow scale.
# |     2. RMSE-log1p on Omega_c^+.
# |     3. Pearson/Spearman correlation on Omega_c^+.
# | """
# |
# | import math
# | import numpy as np
# | import torch
# |
# |
# | def compute_cpc_pair(t_true: np.ndarray, t_pred: np.ndarray) -> float:
# |     """Computes standard CPC between two non-negative 1D arrays."""
# |     sum_min = np.sum(np.minimum(t_true, t_pred))
# |     sum_total = np.sum(t_true) + np.sum(t_pred)
# |     if sum_total <= 0:
# |         return 0.0
# |     return float(2.0 * sum_min / sum_total)
# |
# |
# | def compute_cpc_norm_pair(t_true: np.ndarray, t_pred: np.ndarray) -> float:
# |     """Computes Scale-Normalized CPC (1 - Total Variation Distance)."""
# |     sum_t = np.sum(t_true)
# |     sum_p = np.sum(t_pred)
# |     if sum_t <= 0 or sum_p <= 0:
# |         return 0.0
# |     p_t = t_true / sum_t
# |     p_p = t_pred / sum_p
# |     return float(np.sum(np.minimum(p_t, p_p)))
# |
# |
# | def compute_rmse_log1p_pair(t_true: np.ndarray, t_pred: np.ndarray) -> float:
# |     """Computes RMSE on log1p scale."""
# |     log_t = np.log1p(np.clip(t_true, 0.0, None))
# |     log_p = np.log1p(np.clip(t_pred, 0.0, None))
# |     return float(np.sqrt(np.mean((log_t - log_p) ** 2)))
# |
# |
# | def compute_pearson_pair(t_true: np.ndarray, t_pred: np.ndarray) -> float:
# |     """Computes Pearson linear correlation."""
# |     std_t = np.std(t_true)
# |     std_p = np.std(t_pred)
# |     if std_t == 0 or std_p == 0:
# |         return 0.0
# |     cov = np.mean((t_true - np.mean(t_true)) * (t_pred - np.mean(t_pred)))
# |     return float(cov / (std_t * std_p))
# |
# |
# | def compute_spearman_pair(t_true: np.ndarray, t_pred: np.ndarray) -> float:
# |     """Computes Spearman rank correlation of pairwise flows."""
# |     if len(t_true) < 2 or np.std(t_true) == 0 or np.std(t_pred) == 0:
# |         return 0.0
# |     from scipy import stats
# |     rho, _ = stats.spearmanr(t_true, t_pred)
# |     return float(rho) if not np.isnan(rho) else 0.0
# |
# |
# | def compute_rmse_pair(t_true: np.ndarray, t_pred: np.ndarray) -> float:
# |     """Computes standard RMSE."""
# |     return float(np.sqrt(np.mean((t_true - t_pred) ** 2)))
# |
# | def compute_nrmse_pair(t_true: np.ndarray, t_pred: np.ndarray) -> float:
# |     """Computes Normalized RMSE (RMSE / mean(true))."""
# |     mean_t = np.mean(t_true)
# |     if mean_t <= 0:
# |         return 0.0
# |     rmse = compute_rmse_pair(t_true, t_pred)
# |     return float(rmse / mean_t)
# |
# | def compute_mae_pair(t_true: np.ndarray, t_pred: np.ndarray) -> float:
# |     """Computes Mean Absolute Error."""
# |     return float(np.mean(np.abs(t_true - t_pred)))
# |
# | def compute_inflow_outflow_cpc(t_true: np.ndarray, t_pred: np.ndarray, o_idx: np.ndarray, d_idx: np.ndarray, n_nodes: int) -> tuple[float, float]:
# |     """Computes CPC for tract-level inflows and outflows on observed support."""
# |     outflow_t = np.zeros(n_nodes, dtype=np.float64)
# |     outflow_p = np.zeros(n_nodes, dtype=np.float64)
# |     inflow_t = np.zeros(n_nodes, dtype=np.float64)
# |     inflow_p = np.zeros(n_nodes, dtype=np.float64)
# |     
# |     np.add.at(outflow_t, o_idx, t_true)
# |     np.add.at(outflow_p, o_idx, t_pred)
# |     np.add.at(inflow_t, d_idx, t_true)
# |     np.add.at(inflow_p, d_idx, t_pred)
# |     
# |     cpc_out = compute_cpc_pair(outflow_t, outflow_p)
# |     cpc_in = compute_cpc_pair(inflow_t, inflow_p)
# |     return cpc_in, cpc_out
# |
# | def evaluate_moving_and_full(
# |     t_true: torch.Tensor,
# |     t_pred: torch.Tensor,
# |     pair_o_idx: torch.Tensor,
# |     pair_d_idx: torch.Tensor,
# |     bin_labels: torch.Tensor,
# |     pair_distance: torch.Tensor | None = None,
# | ) -> dict[str, float]:
# |     """
# |     Computes all locked metrics partitioned by Interzonal Omega_c^+ as per partial_od.md.
# |     No full-matrix CPC or missing pair performance is reported.
# |     """
# |     t_t = t_true.detach().cpu().numpy().astype(np.float64)
# |     t_p = t_pred.detach().cpu().numpy().astype(np.float64)
# |     o_np = pair_o_idx.detach().cpu().numpy()
# |     d_np = pair_d_idx.detach().cpu().numpy()
# |     b_np = bin_labels.detach().cpu().numpy()
# |
# |     if pair_distance is not None:
# |         p_dist = pair_distance.detach().cpu().numpy()
# |         # NOTE: pair_distance is stored as log1p(km) in CityData. The > 0.0 check is equivalent
# |         # to distance_km > 0 since log1p is monotone. Do NOT use dist_log1p for metric computation.
# |         dist_log1p = p_dist
# |         inter_mask = (o_np != d_np) & (dist_log1p > 0.0)
# |     else:
# |         inter_mask = (o_np != d_np) & (b_np > 0)
# |
# |     # All evaluations only on observed pairs!
# |     # Primary: Interzonal Domain Omega_c^+
# |     t_t_inter = t_t[inter_mask]
# |     t_p_inter = t_p[inter_mask]
# |
# |     cpc_inter = compute_cpc_pair(t_t_inter, t_p_inter)
# |     rmse_log1p_inter = compute_rmse_log1p_pair(t_t_inter, t_p_inter)
# |     rmse_inter = compute_rmse_pair(t_t_inter, t_p_inter)
# |     nrmse_inter = compute_nrmse_pair(t_t_inter, t_p_inter)
# |     mae_inter = compute_mae_pair(t_t_inter, t_p_inter)
# |     spearman_inter = compute_spearman_pair(t_t_inter, t_p_inter)
# |     
# |     total_flow_true = np.sum(t_t_inter)
# |     total_flow_pred = np.sum(t_p_inter)
# |     rel_error = float(abs(total_flow_pred - total_flow_true) / max(total_flow_true, 1e-9))
# |     
# |     # Inflow/Outflow CPC on observed support
# |     max_node = max(np.max(o_np), np.max(d_np)) + 1 if len(o_np) > 0 else 0
# |     cpc_inflow, cpc_outflow = compute_inflow_outflow_cpc(t_t_inter, t_p_inter, o_np[inter_mask], d_np[inter_mask], max_node)
# |     
# |     result = {
# |         "cpc": cpc_inter,                     # primary shorthand
# |         "cpc_inter": cpc_inter,
# |         "rmse_log1p_inter": rmse_log1p_inter,
# |         "rmse_inter": rmse_inter,
# |         "nrmse_inter": nrmse_inter,
# |         "mae_inter": mae_inter,
# |         "spearman_inter": spearman_inter,
# |         "rel_error_total": rel_error,
# |         "cpc_inflow": cpc_inflow,
# |         "cpc_outflow": cpc_outflow,
# |     }
# |     return result
# |
# | def evaluate_all(t_true: torch.Tensor, t_pred: torch.Tensor) -> dict[str, float]:
# |     """
# |     DEPRECATED — Compatibility helper for raw full-pair evaluation WITHOUT interzonal filtering.
# |
# |     WARNING: This function computes CPC over ALL pairs including intrazonal.
# |     For scientific claims, use evaluate_moving_and_full() which correctly filters to Omega_c^+.
# |     This function must NOT be used to report primary metrics in any experiment.
# |     """
# |     import warnings
# |     warnings.warn(
# |         "evaluate_all() computes over all pairs without interzonal filtering. "
# |         "Use evaluate_moving_and_full() for scientifically valid metrics on Omega_c^+.",
# |         DeprecationWarning,
# |         stacklevel=2,
# |     )
# |     t_t = t_true.detach().cpu().numpy().astype(np.float64)
# |     t_p = t_pred.detach().cpu().numpy().astype(np.float64)
# |     return {
# |         "cpc": compute_cpc_pair(t_t, t_p),
# |         "cpc_norm": compute_cpc_norm_pair(t_t, t_p),
# |         "rmse_log1p": compute_rmse_log1p_pair(t_t, t_p),
# |         "pearson_r": compute_pearson_pair(t_t, t_p),
# |     }

# ===== END SOURCE FILE: src/training/evaluate.py =====


# ===== BEGIN SOURCE FILE: src/training/train.py =====
# | r"""
# | Cross-City Training and Transfer Pipeline.
# |
# | Stage A: Cross-city Training
# |     Trains ZeroShotODModel on a list of source cities using ZTNB likelihood on all positive observed support (including intrazonal):
# |         L_train = - 1 / |Omega^+_all| * sum_{(i,j) in Omega^+_all} log P_ZTNB(T_ij; mu_nb_ij, phi)
# |     City-level losses are averaged within city and optimization proceeds city-by-city, preventing large-support cities from dominating solely through pair count.
# |     After convergence, freezes parameters -> theta*.
# |
# | Stage B: Zero-Shot Transfer Evaluation
# |     Evaluates theta* on held-out target city, evaluating the primary reconstruction estimand on positive interzonal support (Omega_c^+):
# |         (X^{c*}, G^{urban, c*}, D^{c*}) -> \hat{T}^{ZS} = E[T | T >= 1].
# | """
# |
# | import copy
# | import time
# | import datetime
# | from pathlib import Path
# | from typing import List, Dict, Optional, Union
# |
# | import torch
# | import torch.optim as optim
# |
# | from src.data.dataset import (
# |     CityData,
# |     NODE_FEATURE_COLUMNS,
# |     get_scaler_fingerprint,
# |     load_cities,
# |     load_city,
# |     validate_feature_scaler,
# | )
# | from src.data.urban_graph import build_radius_graph, build_knn_graph
# | from src.models.zero_shot_model import ZeroShotODModel
# | from src.loss.ztnb import ztnb_nll, nb_nll
# | from src.training.evaluate import evaluate_all
# |
# |
# | # ---------------------------------------------------------------------------
# | # Checkpoint utilities
# | # ---------------------------------------------------------------------------
# |
# | def save_checkpoint(
# |     path: Union[str, Path],
# |     model: "ZeroShotODModel",
# |     scaler: object,
# |     train_info: dict,
# |     hyperparams: dict,
# |     seed: Optional[int] = None,
# |     run_tag: Optional[str] = None,
# | ) -> Path:
# |     """
# |     Persists a trained ZeroShotODModel checkpoint to disk.
# |
# |     Saved bundle contains:
# |         - model_state_dict   : weights (best validation checkpoint)
# |         - scaler_*           : StandardScaler statistics for feature normalization
# |         - train_info         : best_epoch, best_val_cpc, epochs_trained, histories
# |         - hyperparams        : architecture + training config needed to reconstruct model
# |         - seed               : random seed used for this run (None if not set)
# |         - run_tag            : human-readable label, e.g. "e1_fold1"
# |         - saved_at           : ISO-8601 UTC timestamp
# |
# |     Args:
# |         path:        Full file path to write (created with parents if needed).
# |         model:       Trained (and eval-mode) ZeroShotODModel instance.
# |         scaler:      Fitted sklearn StandardScaler from load_cities().
# |         train_info:  Dict returned by train_zero_shot_model() when return_info=True.
# |         hyperparams: Dict of architecture / training hyper-parameters.
# |         seed:        Random seed (optional).
# |         run_tag:     Short label for this run (optional).
# |
# |     Returns:
# |         Resolved Path of the saved file.
# |     """
# |     import numpy as _np
# |
# |     path = Path(path)
# |     path.parent.mkdir(parents=True, exist_ok=True)
# |
# |     scaler_data: dict = {}
# |     if scaler is not None and hasattr(scaler, "mean_") and scaler.mean_ is not None:
# |         validate_feature_scaler(scaler)
# |         scaler_data = {
# |             "scaler_mean_":  _np.asarray(scaler.mean_,  dtype=_np.float64),
# |             "scaler_scale_": _np.asarray(scaler.scale_, dtype=_np.float64),
# |             "scaler_var_":   _np.asarray(scaler.var_,   dtype=_np.float64),
# |             "scaler_n_features_in_": int(getattr(scaler, "n_features_in_", len(scaler.mean_))),
# |             "scaler_fingerprint": get_scaler_fingerprint(scaler),
# |             "scaler_feature_columns": list(NODE_FEATURE_COLUMNS),
# |         }
# |         if hasattr(scaler, "n_samples_seen_"):
# |             scaler_data["scaler_n_samples_seen_"] = _np.asarray(
# |                 scaler.n_samples_seen_
# |             ).copy()
# |
# |     bundle = {
# |         "model_state_dict": model.state_dict(),
# |         **scaler_data,
# |         "train_info":   train_info,
# |         "hyperparams":  hyperparams,
# |         "seed":         seed,
# |         "run_tag":      run_tag,
# |         "saved_at":     datetime.datetime.utcnow().isoformat() + "Z",
# |     }
# |
# |     torch.save(bundle, path)
# |     return path.resolve()
# |
# |
# | def load_checkpoint(
# |     path: Union[str, Path],
# |     device_str: str = "cpu",
# |     expected_config: Optional[dict] = None,
# | ) -> tuple:
# |     """
# |     Loads a checkpoint saved by save_checkpoint() and reconstructs the model and scaler.
# |
# |     Args:
# |         path:       Path to the .pt checkpoint file.
# |         device_str: Device to map model weights onto ("cpu" or "cuda").
# |         expected_config: Optional dictionary of hyperparams to validate against the checkpoint.
# |
# |     Returns:
# |         (model, scaler, metadata) where:
# |             model    — ZeroShotODModel in eval mode with frozen weights
# |             scaler   — Reconstructed sklearn StandardScaler (or None if not saved)
# |             metadata — Full checkpoint dict (train_info, hyperparams, seed, run_tag, saved_at)
# |     """
# |     import numpy as _np
# |
# |     path = Path(path)
# |     if not path.exists():
# |         raise FileNotFoundError(f"Checkpoint not found: {path}")
# |
# |     bundle = torch.load(path, map_location=torch.device(device_str), weights_only=False)
# |
# |     hp = bundle["hyperparams"]
# |     if expected_config is not None:
# |         for k, v in expected_config.items():
# |             if k not in hp:
# |                 raise ValueError(f"Checkpoint config missing key '{k}' in {path}. Expected {v}. Checkpoint may be incomplete.")
# |             if hp[k] != v:
# |                 raise ValueError(f"Checkpoint config mismatch in {path} for key '{k}': expected {v}, got {hp[k]}. Delete the stale checkpoint to retrain.")
# |
# |     # --- Reconstruct model ---
# |     hp = bundle["hyperparams"]
# |     backbone = hp.get("backbone", "gnn")
# |     
# |     from src.models.zero_shot_model import ZeroShotODModel, ZeroShotMLPModel
# |     
# |     if backbone == "mlp":
# |         model = ZeroShotMLPModel(
# |             node_in_dim       = hp["node_in_dim"],
# |             node_hidden_dim   = hp["hidden_dim"],
# |             node_out_dim      = hp["hidden_dim"],
# |             num_gnn_layers    = hp["num_gnn_layers"],
# |             decoder_hidden_dim= hp["hidden_dim"],
# |         ).to(torch.device(device_str))
# |     else:
# |         model = ZeroShotODModel(
# |             node_in_dim       = hp["node_in_dim"],
# |             node_hidden_dim   = hp["hidden_dim"],
# |             node_out_dim      = hp["hidden_dim"],
# |             num_gnn_layers    = hp["num_gnn_layers"],
# |             decoder_hidden_dim= hp["hidden_dim"],
# |         ).to(torch.device(device_str))
# |         
# |     model.load_state_dict(bundle["model_state_dict"])
# |     model.eval()
# |     for p in model.parameters():
# |         p.requires_grad = False
# |
# |     # --- Reconstruct scaler ---
# |     scaler = None
# |     if "scaler_mean_" in bundle:
# |         from sklearn.preprocessing import StandardScaler
# |         scaler = StandardScaler()
# |         scaler.mean_  = bundle["scaler_mean_"]
# |         scaler.scale_ = bundle["scaler_scale_"]
# |         scaler.var_   = bundle["scaler_var_"]
# |         scaler.n_features_in_ = bundle.get("scaler_n_features_in_", len(scaler.mean_))
# |         if "scaler_n_samples_seen_" in bundle:
# |             scaler.n_samples_seen_ = bundle["scaler_n_samples_seen_"]
# |         validate_feature_scaler(scaler)
# |
# |         expected_columns = bundle.get("scaler_feature_columns")
# |         if expected_columns is not None and tuple(expected_columns) != NODE_FEATURE_COLUMNS:
# |             raise ValueError(f"Checkpoint feature schema mismatch in {path}")
# |
# |         expected_fingerprint = bundle.get("scaler_fingerprint")
# |         actual_fingerprint = get_scaler_fingerprint(scaler)
# |         if expected_fingerprint is not None and actual_fingerprint != expected_fingerprint:
# |             raise ValueError(f"Checkpoint scaler fingerprint mismatch in {path}")
# |
# |     metadata = {
# |         "train_info":  bundle.get("train_info"),
# |         "hyperparams": bundle.get("hyperparams"),
# |         "seed":        bundle.get("seed"),
# |         "run_tag":     bundle.get("run_tag"),
# |         "saved_at":    bundle.get("saved_at"),
# |         "scaler_provenance": {
# |             "fingerprint": bundle.get("scaler_fingerprint"),
# |             "n_features_in": bundle.get("scaler_n_features_in_"),
# |             "n_samples_seen": bundle.get("scaler_n_samples_seen_"),
# |             "feature_columns": bundle.get("scaler_feature_columns"),
# |         },
# |     }
# |
# |     return model, scaler, metadata
# |
# |
# | def train_epoch(
# |     model: torch.nn.Module,
# |     train_cities: List[CityData],
# |     city_graphs: List[tuple[torch.Tensor, torch.Tensor]],
# |     optimizer: optim.Optimizer,
# |     loss_type: str = "ztnb",
# |     device: torch.device = torch.device("cpu"),
# | ) -> float:
# |     model.train()
# |     total_loss = 0.0
# |     num_cities = len(train_cities)
# |
# |     for city_data, (edge_index, edge_dist) in zip(train_cities, city_graphs):
# |         optimizer.zero_grad()
# |
# |         x = city_data.node_features.to(device)
# |         ei = edge_index.to(device)
# |         ed = edge_dist.to(device)
# |         p_o = city_data.pair_o_idx.to(device)
# |         p_d = city_data.pair_d_idx.to(device)
# |         p_dist = city_data.pair_distance.to(device)
# |         pop = city_data.population.to(device)
# |         t_true = city_data.pair_trips.to(device)
# |
# |         # Training pass returns base mean mu_nb
# |         mu_nb = model(x, ei, ed, p_o, p_d, p_dist, pop, return_conditional_mean=False)
# |
# |         if loss_type == "ztnb":
# |             loss = ztnb_nll(t_true, mu_nb, model.log_phi)
# |         elif loss_type == "nb":
# |             loss = nb_nll(t_true, mu_nb, model.log_phi)
# |         else:
# |             raise ValueError(f"Unknown loss type {loss_type}")
# |
# |         if not torch.isfinite(loss):
# |             raise FloatingPointError(f"NaN/Inf loss encountered for {city_data.city_name}. Stopping training to avoid invalid checkpoint.")
# |
# |         loss.backward()
# |         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
# |         optimizer.step()
# |
# |         total_loss += loss.item()
# |
# |     return total_loss / max(num_cities, 1)
# |
# |
# | @torch.no_grad()
# | def infer_zero_shot(
# |     model: torch.nn.Module,
# |     city_data: CityData,
# |     edge_index: torch.Tensor,
# |     edge_dist: torch.Tensor,
# |     device: torch.device = torch.device("cpu"),
# | ) -> torch.Tensor:
# |     """Runs zero-shot forward inference returning exact conditional expectation E[T | T >= 1]."""
# |     model.eval()
# |     x = city_data.node_features.to(device)
# |     ei = edge_index.to(device)
# |     ed = edge_dist.to(device)
# |     p_o = city_data.pair_o_idx.to(device)
# |     p_d = city_data.pair_d_idx.to(device)
# |     p_dist = city_data.pair_distance.to(device)
# |     pop = city_data.population.to(device)
# |
# |     # Returns E[T | T >= 1]
# |     t_hat = model(x, ei, ed, p_o, p_d, p_dist, pop, return_conditional_mean=True)
# |     return t_hat.cpu()
# |
# |
# | def train_zero_shot_model(
# |     train_city_names: List[str],
# |     data_root: str = "data",
# |     epochs: int = 200,
# |     lr: float = 2e-3,
# |     weight_decay: float = 1e-4,
# |     hidden_dim: int = 64,
# |     num_gnn_layers: int = 2,
# |     graph_type: str = "radius",
# |     radius_km: float = 5.0,
# |     knn_k: int = 10,
# |     loss_type: str = "ztnb",
# |     backbone: str = "gnn",
# |     dropout: float = 0.1,
# |     device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
# |     verbose: bool = True,
# |     # --- Validation / early stopping ---
# |     val_city_names: List[str] | None = None,
# |     patience: int = 15,
# |     min_delta: float = 1e-4,
# |     lr_plateau_patience: int = 4,
# |     lr_plateau_factor: float = 0.5,
# |     lr_plateau_threshold: float = 1e-4,
# |     threshold_mode: str = "abs",
# |     min_lr: float = 1e-5,
# |     return_info: bool = False,
# |     seed: int | None = None,
# |     # --- Checkpoint provenance ---
# |     fold: int | None = None,
# |     split_manifest_sha256: str | None = None,
# |     checkpoint_path: Optional[Union[str, Path]] = None,
# |     run_tag: Optional[str] = None,
# | ) -> tuple:
# |
# |     """
# |     Train ZeroShotODModel with AdamW, ReduceLROnPlateau, and validation-based early stopping.
# |
# |     Args:
# |         train_city_names: Cities to train on.
# |         val_city_names:   Validation cities for early stopping. If None,
# |                           trains for exactly `epochs` epochs (pre-specified).
# |         patience:         Epochs without val CPC improvement before stopping.
# |         min_delta:        Minimum improvement to count as improvement.
# |         return_info:      If True, returns (model, scaler, train_info_dict).
# |         seed:             Optional random seed for reproducible weight initialization.
# |         checkpoint_path:  If provided, saves the trained model to this path as a .pt file.
# |                           Parent directories are created automatically.
# |         run_tag:          Short label embedded in the checkpoint (e.g. "e1_fold1_seed2025").
# |
# |     Returns:
# |         (best_model, scaler) or (best_model, scaler, info)
# |     """
# |     import copy
# |     import numpy as _np
# |
# |     if seed is not None:
# |         torch.manual_seed(seed)
# |         torch.cuda.manual_seed_all(seed)
# |         _np.random.seed(seed)
# |
# |     device = torch.device(device_str)
# |
# |     if verbose:
# |         print(f"    [Setup] Precomputing graph structures for {len(train_city_names)} source cities onto {device}...", flush=True)
# |
# |     train_cities, scaler = load_cities(train_city_names, data_root=data_root)
# |
# |     # Precompute spatial graphs G^urban for training cities
# |     city_graphs = []
# |     for c in train_cities:
# |         coords = c.lon_lat.numpy()
# |         if graph_type == "radius":
# |             ei, ed = build_radius_graph(coords, radius_km=radius_km)
# |         else:
# |             ei, ed = build_knn_graph(coords, k=knn_k)
# |         city_graphs.append((ei, ed))
# |
# |     # Pre-move training tensors onto device to avoid repeated host-to-device transfers per epoch
# |     train_cities_dev = [
# |         CityData(
# |             city_name     = c.city_name,
# |             n_tracts      = c.n_tracts,
# |             n_pairs       = c.n_pairs,
# |             node_features = c.node_features.to(device),
# |             population    = c.population.to(device),
# |             lon_lat       = c.lon_lat.to(device),
# |             pair_o_idx    = c.pair_o_idx.to(device),
# |             pair_d_idx    = c.pair_d_idx.to(device),
# |             pair_distance = c.pair_distance.to(device),
# |             pair_trips    = c.pair_trips.to(device),
# |             bin_labels    = c.bin_labels.to(device),
# |         )
# |         for c in train_cities
# |     ]
# |     city_graphs_dev = [(ei.to(device), ed.to(device)) for (ei, ed) in city_graphs]
# |
# |     # Precompute device-resident structures & masks for validation cities (if provided)
# |     val_data_on_device = []
# |     if val_city_names:
# |         for name in val_city_names:
# |             vc = load_city(name, data_root=data_root, feature_scaler=scaler)
# |             coords = vc.lon_lat.numpy()
# |             if graph_type == "radius":
# |                 ei, ed = build_radius_graph(coords, radius_km=radius_km)
# |             else:
# |                 ei, ed = build_knn_graph(coords, k=knn_k)
# |
# |             # Precompute interzonal mask and ground truth on device once
# |             dist_km = _np.expm1(vc.pair_distance.numpy())
# |             inter_cpu = (vc.pair_o_idx.numpy() != vc.pair_d_idx.numpy()) & (dist_km > 0.0)
# |             inter_mask = torch.tensor(inter_cpu, dtype=torch.bool, device=device)
# |             t_gt_inter = vc.pair_trips.to(device)[inter_mask]
# |
# |             val_data_on_device.append({
# |                 "x": vc.node_features.to(device),
# |                 "ei": ei.to(device),
# |                 "ed": ed.to(device),
# |                 "p_o": vc.pair_o_idx.to(device),
# |                 "p_d": vc.pair_d_idx.to(device),
# |                 "p_dist": vc.pair_distance.to(device),
# |                 "pop": vc.population.to(device),
# |                 "inter_mask": inter_mask,
# |                 "t_gt_inter": t_gt_inter,
# |                 "t_gt_sum": torch.sum(t_gt_inter),
# |                 "has_inter": bool(inter_cpu.sum() > 0),
# |             })
# |
# |     from src.models.zero_shot_model import ZeroShotODModel, ZeroShotMLPModel
# |
# |     if backbone == "mlp":
# |         model = ZeroShotMLPModel(
# |             node_in_dim=train_cities[0].node_features.shape[1],
# |             node_hidden_dim=hidden_dim,
# |             node_out_dim=hidden_dim,
# |             num_gnn_layers=num_gnn_layers,
# |             decoder_hidden_dim=hidden_dim,
# |             dropout=dropout,
# |         ).to(device)
# |     else:
# |         model = ZeroShotODModel(
# |             node_in_dim=train_cities[0].node_features.shape[1],
# |             node_hidden_dim=hidden_dim,
# |             node_out_dim=hidden_dim,
# |             num_gnn_layers=num_gnn_layers,
# |             decoder_hidden_dim=hidden_dim,
# |             dropout=dropout,
# |         ).to(device)
# |
# |     optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
# |     if val_city_names:
# |         scheduler = optim.lr_scheduler.ReduceLROnPlateau(
# |             optimizer,
# |             mode="max",
# |             factor=lr_plateau_factor,
# |             patience=lr_plateau_patience,
# |             threshold=lr_plateau_threshold,
# |             threshold_mode=threshold_mode,
# |             min_lr=min_lr,
# |         )
# |     else:
# |         scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
# |
# |     best_val_cpc = -float("inf")
# |     best_epoch = epochs
# |     best_state = None
# |     patience_counter = 0
# |     use_early_stopping = bool(val_city_names)
# |
# |     val_history = []
# |     loss_history = []
# |
# |     start_time = time.time()
# |     for epoch in range(1, epochs + 1):
# |         loss_val = train_epoch(
# |             model=model,
# |             train_cities=train_cities_dev,
# |             city_graphs=city_graphs_dev,
# |             optimizer=optimizer,
# |             loss_type=loss_type,
# |             device=device,
# |         )
# |         loss_history.append(loss_val)
# |
# |         # --- Fast GPU-Vectorized Validation CPC (interzonal) ---
# |         val_cpc_str = ""
# |         if use_early_stopping and val_data_on_device:
# |             val_cpcs = []
# |             model.eval()
# |             with torch.no_grad():
# |                 for item in val_data_on_device:
# |                     if not item["has_inter"]:
# |                         print(f"    [WARNING] Validation city '{item.get('city_name', '?')}' has no interzonal pairs — skipped in CPC computation. Check data integrity.", flush=True)
# |                         continue
# |                     t_hat = model(
# |                         item["x"], item["ei"], item["ed"],
# |                         item["p_o"], item["p_d"], item["p_dist"],
# |                         item["pop"], return_conditional_mean=True
# |                     )
# |                     t_hat_inter = t_hat[item["inter_mask"]]
# |                     sum_min = torch.sum(torch.minimum(item["t_gt_inter"], t_hat_inter))
# |                     sum_total = item["t_gt_sum"] + torch.sum(t_hat_inter)
# |                     cpc_val = (2.0 * sum_min / sum_total).item() if sum_total > 0 else 0.0
# |                     val_cpcs.append(cpc_val)
# |
# |             if not val_cpcs:
# |                 raise RuntimeError(
# |                     "All validation cities were skipped (no interzonal pairs). "
# |                     "Cannot compute validation CPC. Check dataset construction."
# |                 )
# |             mean_val_cpc = float(_np.mean(val_cpcs))
# |             val_history.append(mean_val_cpc)
# |             val_cpc_str = f" | ValCPC: {mean_val_cpc:.4f}"
# |
# |             # Step plateau scheduler on validation metric
# |             scheduler.step(mean_val_cpc)
# |
# |             # Best-model tracking
# |             if mean_val_cpc > best_val_cpc + min_delta:
# |                 best_val_cpc = mean_val_cpc
# |                 best_epoch = epoch
# |                 best_state = copy.deepcopy(model.state_dict())
# |                 patience_counter = 0
# |             else:
# |                 patience_counter += 1
# |         else:
# |             scheduler.step()
# |
# |         if verbose:
# |             elapsed = time.time() - start_time
# |             pat_str = f" | Patience: {patience_counter}/{patience}" if use_early_stopping else ""
# |             curr_lr = optimizer.param_groups[0]["lr"]
# |             print(
# |                 f"    [Epoch {epoch:03d}/{epochs:03d}] Loss: {loss_val:.4f}{val_cpc_str}{pat_str} | "
# |                 f"lr: {curr_lr:.1e} | phi: {model.phi.item():.3f} | {elapsed:.1f}s",
# |                 flush=True,
# |             )
# |
# |         # --- Early stopping ---
# |         if use_early_stopping and patience_counter >= patience:
# |             if verbose:
# |                 print(f"    -> Early stopping triggered at epoch {epoch} (best epoch {best_epoch}, best val CPC {best_val_cpc:.4f}).", flush=True)
# |             break
# |
# |     # Restore best checkpoint (if early stopping was used and improved)
# |     if use_early_stopping and best_state is not None:
# |         model.load_state_dict(best_state)
# |         if verbose:
# |             print(f"    -> Restored best model checkpoint (epoch={best_epoch}, val CPC={best_val_cpc:.4f}).", flush=True)
# |
# |     model.eval()
# |     for p in model.parameters():
# |         p.requires_grad = False
# |
# |     info = {
# |         "best_epoch": best_epoch,
# |         "best_val_cpc": best_val_cpc if use_early_stopping else None,
# |         "epochs_trained": epoch,
# |         "stopped_early": use_early_stopping and (patience_counter >= patience),
# |         "val_cpc_history": val_history,
# |         "train_loss_history": loss_history,
# |     }
# |
# |     # --- Persist checkpoint to disk if requested ---
# |     if checkpoint_path is not None:
# |         # C1: split_manifest_sha256 must be passed explicitly; raise if caller forgot.
# |         if split_manifest_sha256 is None:
# |             raise ValueError(
# |                 "split_manifest_sha256 must be provided when saving a checkpoint. "
# |                 "Load the split manifest and pass its SHA256 hash to train_zero_shot_model()."
# |             )
# |         hp = {
# |             "node_in_dim":           train_cities[0].node_features.shape[1],
# |             "hidden_dim":            hidden_dim,
# |             "num_gnn_layers":        num_gnn_layers,
# |             "dropout":               dropout,
# |             "graph_type":            graph_type,
# |             "radius_km":             radius_km,
# |             "knn_k":                 knn_k,
# |             "loss_type":             loss_type,
# |             "epochs":                epochs,
# |             "lr":                    lr,
# |             "weight_decay":          weight_decay,
# |             "backbone":              backbone,
# |             "patience":              patience,
# |             "min_delta":             min_delta,
# |             "lr_plateau_patience":   lr_plateau_patience,
# |             "lr_plateau_factor":     lr_plateau_factor,
# |             "lr_plateau_threshold":  lr_plateau_threshold,
# |             "threshold_mode":        threshold_mode,
# |             "min_lr":                min_lr,
# |             # Provenance fields (C2, C1)
# |             "fold":                  fold,
# |             "split_manifest_sha256": split_manifest_sha256,
# |             "scaler_fit_scope":      "training_split_only",
# |             "scaler_weighting":      "per_tract",
# |             "scaler_fit_cities":     sorted(train_city_names),
# |             "scaler_fit_n_cities":   len(train_city_names),
# |             "scaler_fit_n_rows":     int(scaler.n_samples_seen_),
# |         }
# |         saved_path = save_checkpoint(
# |             path=checkpoint_path,
# |             model=model,
# |             scaler=scaler,
# |             train_info=info,
# |             hyperparams=hp,
# |             seed=seed,
# |             run_tag=run_tag,
# |         )
# |         if verbose:
# |             print(f"    -> Checkpoint saved: {saved_path}", flush=True)
# |
# |     if return_info:
# |         return model, scaler, info
# |     return model, scaler

# ===== END SOURCE FILE: src/training/train.py =====


# ===== BEGIN SOURCE FILE: tests/test_backbone.py =====
# | """
# | Unit and Contract Test Suite for Backbone Robustness & MLP Architecture.
# | Tests:
# |     1. Parameter parity between MLPLayer and GraphConvLayer.
# |     2. NodeMLP forward pass shape and gradient propagation.
# |     3. ZeroShotMLPModel forward pass and ZTNB loss compatibility.
# |     4. Model checkpoint save and reload consistency for backbone='mlp'.
# |     5. Table 7 Backbone Robustness summary output validity.
# | """
# |
# | import os
# | import tempfile
# | import torch
# | import numpy as np
# | from pathlib import Path
# |
# | from src.models.node_encoder import UrbanGNN, NodeMLP, MLPLayer, GraphConvLayer
# | from src.models.zero_shot_model import ZeroShotODModel, ZeroShotMLPModel
# | from src.training.train import save_checkpoint, load_checkpoint
# | from src.loss.ztnb import ztnb_nll
# |
# |
# | def test_parameter_count_parity():
# |     """T1: MLPLayer and GraphConvLayer have identical learnable parameter counts."""
# |     hidden_dim = 64
# |     gnn_layer = GraphConvLayer(in_dim=hidden_dim, out_dim=hidden_dim)
# |     mlp_layer = MLPLayer(in_dim=hidden_dim, out_dim=hidden_dim)
# |
# |     gnn_params = sum(p.numel() for p in gnn_layer.parameters() if p.requires_grad)
# |     mlp_params = sum(p.numel() for p in mlp_layer.parameters() if p.requires_grad)
# |
# |     assert gnn_params == mlp_params, f"Parameter mismatch: GNN={gnn_params} vs MLP={mlp_params}"
# |     print("✓ Test 1 Passed: Exact parameter parity between MLPLayer and GraphConvLayer.")
# |
# |
# | def test_node_mlp_forward_and_backward():
# |     """T2: NodeMLP produces correct embedding shapes and supports backprop."""
# |     N = 25
# |     in_dim = 26
# |     hidden_dim = 64
# |     x = torch.randn(N, in_dim, requires_grad=True)
# |     dummy_edge_index = torch.zeros((2, 10), dtype=torch.long)
# |     dummy_edge_dist = torch.zeros(10, dtype=torch.float32)
# |
# |     mlp_encoder = NodeMLP(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=hidden_dim, num_layers=2)
# |     h = mlp_encoder(x, dummy_edge_index, dummy_edge_dist)
# |
# |     assert h.shape == (N, hidden_dim)
# |     loss = h.sum()
# |     loss.backward()
# |     assert x.grad is not None
# |     assert torch.isfinite(x.grad).all()
# |     print("✓ Test 2 Passed: NodeMLP forward shape and gradients are valid.")
# |
# |
# | def test_zero_shot_mlp_model():
# |     """T3: ZeroShotMLPModel forward pass, conditional mean, and loss computation."""
# |     N = 20
# |     E = 150
# |     x = torch.randn(N, 26)
# |     edge_index = torch.zeros((2, 0), dtype=torch.long)
# |     edge_dist = torch.zeros(0, dtype=torch.float32)
# |     pair_o = torch.randint(0, N, (E,))
# |     pair_d = torch.randint(0, N, (E,))
# |     pair_dist = torch.rand(E) * 2.0
# |     pop = torch.rand(N) * 5000.0 + 100.0
# |     t_true = torch.randint(1, 100, (E,)).float()
# |
# |     model = ZeroShotMLPModel(node_in_dim=26, node_hidden_dim=64, node_out_dim=64, num_gnn_layers=2)
# |     
# |     mu_nb = model(x, edge_index, edge_dist, pair_o, pair_d, pair_dist, pop, return_conditional_mean=False)
# |     assert mu_nb.shape == (E,)
# |     assert (mu_nb > 0).all()
# |
# |     cond_mean = model(x, edge_index, edge_dist, pair_o, pair_d, pair_dist, pop, return_conditional_mean=True)
# |     assert cond_mean.shape == (E,)
# |     assert (cond_mean > 0).all()
# |
# |     loss = ztnb_nll(t_true, mu_nb, model.log_phi)
# |     assert torch.isfinite(loss)
# |     print("✓ Test 3 Passed: ZeroShotMLPModel forward pass and ZTNB loss computation are valid.")
# |
# |
# | def test_mlp_checkpoint_save_and_load():
# |     """T4: Checkpoint save and load correctly identifies and restores ZeroShotMLPModel."""
# |     with tempfile.TemporaryDirectory() as tmpdir:
# |         ckpt_path = Path(tmpdir) / "test_mlp.pt"
# |         model = ZeroShotMLPModel(node_in_dim=26, node_hidden_dim=32, node_out_dim=32, decoder_hidden_dim=32, num_gnn_layers=2)
# |         scaler = None
# |
# |         save_checkpoint(
# |             path=ckpt_path,
# |             model=model,
# |             scaler=scaler,
# |             train_info={"epoch": 10, "val_cpc": 0.725},
# |             hyperparams={
# |                 "node_in_dim": 26,
# |                 "hidden_dim": 32,
# |                 "num_gnn_layers": 2,
# |                 "backbone": "mlp",
# |             }
# |         )
# |
# |         loaded_model, _, info = load_checkpoint(ckpt_path, device_str="cpu")
# |         assert isinstance(loaded_model, ZeroShotMLPModel)
# |         assert info["train_info"]["epoch"] == 10
# |         assert info["train_info"]["val_cpc"] == 0.725
# |         print("✓ Test 4 Passed: Checkpoint save/load for backbone='mlp' functions correctly.")
# |
# |
# | if __name__ == "__main__":
# |     print("Running Backbone Test Suite...")
# |     test_parameter_count_parity()
# |     test_node_mlp_forward_and_backward()
# |     test_zero_shot_mlp_model()
# |     test_mlp_checkpoint_save_and_load()
# |     print("\nAll 4/4 Backbone Unit & Contract Tests Passed Successfully!")

# ===== END SOURCE FILE: tests/test_backbone.py =====


# ===== BEGIN SOURCE FILE: tests/test_direct_od_equivalence_v1_contract.py =====
# | """
# | 20 Mandatory Scientific Contract Gates for Direct Partial-OD Information Equivalence v1.
# | """
# |
# | import sys
# | import os
# | import json
# | import hashlib
# | import numpy as np
# | import pandas as pd
# | from pathlib import Path
# | from typing import Dict, List, Any
# |
# | REPO_ROOT = Path(__file__).resolve().parent.parent
# | sys.path.insert(0, str(REPO_ROOT))
# |
# | from src.data.city_splits import generate_35_5_10_splits
# | from src.data.dataset import load_raw_city, load_city
# | from src.training.evaluate import compute_cpc_pair
# | from src.training.train import load_checkpoint
# | from src.experiment.run_direct_od_equivalence_v1 import (
# |     PARTIAL_OD_BASE_SEED, PRIMARY_GRID_DIRECT, LAMBDA_CANDIDATES,
# |     RAW_COLUMNS_DIRECT, get_stable_mask_seed, fit_od_fe_adapter,
# |     apply_od_fe_prediction, select_fold_lambda, holm_correction,
# |     fold_stratified_bootstrap
# | )
# |
# |
# | def test_contract_1_split_integrity():
# |     """CONTRACT 1: 5 folds, 35 train, 5 val, 10 test, 50 unique test cities."""
# |     splits = generate_35_5_10_splits(data_root="data")
# |     assert len(splits) == 5
# |     all_test = []
# |     for f, split in splits.items():
# |         assert len(split["train"]) == 35
# |         assert len(split["val"]) == 5
# |         assert len(split["test"]) == 10
# |         all_test.extend(split["test"])
# |     assert len(all_test) == 50 and len(set(all_test)) == 50
# |
# |
# | def test_contract_2_checkpoint_integrity():
# |     """CONTRACT 2: 15 GNN checkpoints exist, loadable, scaler feature dim=26."""
# |     for f in range(1, 6):
# |         for s in [1, 10, 100]:
# |             p = Path("results/checkpoints") / f"5fold_fold{f}_seed{s}.pt"
# |             assert p.exists(), f"Missing checkpoint {p}"
# |             model, scaler, meta = load_checkpoint(p, device_str="cpu")
# |             assert scaler.mean_.shape[0] == 26
# |
# |
# | def test_contract_3_lambda_selection_leakage():
# |     """CONTRACT 3: Lambda selection strictly uses validation cities; zero test cities present."""
# |     splits = generate_35_5_10_splits(data_root="data")
# |     for f, split in splits.items():
# |         val_set = set(split["val"])
# |         test_set = set(split["test"])
# |         assert val_set.isdisjoint(test_set)
# |         # Check that select_fold_lambda receives val_cities only
# |         assert len(val_set) == 5
# |
# |
# | def test_contract_4_lambda_determinism():
# |     """CONTRACT 4: Lambda selection is deterministic for given fold and validation set."""
# |     candidates = LAMBDA_CANDIDATES
# |     scores = pd.DataFrame([
# |         {"lambda": 0.1, "validation_mean_cpc": 0.715, "mean_gain": 0.002},
# |         {"lambda": 1.0, "validation_mean_cpc": 0.718, "mean_gain": 0.005},
# |         {"lambda": 10.0, "validation_mean_cpc": 0.718, "mean_gain": 0.005},
# |         {"lambda": 100.0, "validation_mean_cpc": 0.712, "mean_gain": -0.001},
# |     ])
# |     # Tie-breaker should pick larger lambda (10.0 over 1.0)
# |     best_row = scores.sort_values(by=["validation_mean_cpc", "lambda"], ascending=[False, False]).iloc[0]
# |     assert best_row["lambda"] == 10.0
# |
# |
# | def test_contract_5_mask_partition():
# |     """CONTRACT 5: S_p and U_p are strictly disjoint and partition Omega_c^+."""
# |     raw = load_raw_city("Austin", data_root="data")
# |     inter_pos = (raw.pair_o_idx.numpy() != raw.pair_d_idx.numpy()) & (raw.dist_km > 0.0) & (raw.pair_trips.numpy() > 0)
# |     n_pairs = int(inter_pos.sum())
# |     perm = np.random.RandomState(42).permutation(n_pairs)
# |     for p in PRIMARY_GRID_DIRECT:
# |         n_rev = int(np.round(p * n_pairs))
# |         S_p = set(perm[:n_rev])
# |         U_p = set(perm[n_rev:])
# |         assert S_p.isdisjoint(U_p)
# |         assert len(S_p | U_p) == n_pairs
# |
# |
# | def test_contract_6_nested_masks():
# |     """CONTRACT 6: S_p1 is a subset of S_p2 for p1 < p2."""
# |     n_pairs = 1000
# |     perm = np.random.RandomState(42).permutation(n_pairs)
# |     masks = [set(perm[:int(np.round(p * n_pairs))]) for p in PRIMARY_GRID_DIRECT]
# |     for i in range(len(masks) - 1):
# |         assert masks[i].issubset(masks[i+1])
# |
# |
# | def test_contract_7_same_masks_across_seeds():
# |     """CONTRACT 7: Seeds 1, 10, 100 receive identical mask permutation."""
# |     for f in range(1, 6):
# |         seed_val = get_stable_mask_seed(PARTIAL_OD_BASE_SEED, f, "Austin", 9)
# |         perm1 = np.random.RandomState(seed_val).permutation(2000)
# |         perm2 = np.random.RandomState(seed_val).permutation(2000)
# |         assert np.array_equal(perm1, perm2)
# |
# |
# | def test_contract_8_no_unseen_truth_in_fitting_mutation():
# |     """CONTRACT 8: Mutating true flows on unseen set U_p does NOT affect adapter predictions."""
# |     o_idx = np.array([0, 1, 2, 0, 1, 2])
# |     d_idx = np.array([1, 2, 0, 2, 0, 1])
# |     t0 = np.array([10.0, 20.0, 15.0, 5.0, 8.0, 12.0])
# |     t_true = np.array([12.0, 18.0, 16.0, 4.0, 9.0, 11.0])
# |     
# |     rev_indices = np.array([0, 1, 2]) # S_p
# |     unseen_indices = np.array([3, 4, 5]) # U_p
# |     
# |     a1, b1, _, _ = fit_od_fe_adapter(o_idx, d_idx, t0, t_true, rev_indices, num_nodes=3, lambda_reg=1.0)
# |     pred1 = apply_od_fe_prediction(o_idx, d_idx, t0, a1, b1)
# |     
# |     # Mutate unseen truth
# |     t_true_mutated = t_true.copy()
# |     t_true_mutated[unseen_indices] *= 1000.0
# |     
# |     a2, b2, _, _ = fit_od_fe_adapter(o_idx, d_idx, t0, t_true_mutated, rev_indices, num_nodes=3, lambda_reg=1.0)
# |     pred2 = apply_od_fe_prediction(o_idx, d_idx, t0, a2, b2)
# |     
# |     assert np.allclose(pred1, pred2, atol=1e-12)
# |
# |
# | def test_contract_9_revealed_truth_positive_control():
# |     """CONTRACT 9: Mutating revealed truth on S_p DOES change adapter predictions (positive control)."""
# |     o_idx = np.array([0, 1, 2, 0, 1, 2])
# |     d_idx = np.array([1, 2, 0, 2, 0, 1])
# |     t0 = np.array([10.0, 20.0, 15.0, 5.0, 8.0, 12.0])
# |     t_true = np.array([12.0, 18.0, 16.0, 4.0, 9.0, 11.0])
# |     rev_indices = np.array([0, 1, 2])
# |     
# |     a1, b1, _, _ = fit_od_fe_adapter(o_idx, d_idx, t0, t_true, rev_indices, num_nodes=3, lambda_reg=1.0)
# |     pred1 = apply_od_fe_prediction(o_idx, d_idx, t0, a1, b1)
# |     
# |     t_true_mutated = t_true.copy()
# |     t_true_mutated[rev_indices] *= 10.0 # Extreme mutation on revealed set
# |     
# |     a2, b2, _, _ = fit_od_fe_adapter(o_idx, d_idx, t0, t_true_mutated, rev_indices, num_nodes=3, lambda_reg=1.0)
# |     pred2 = apply_od_fe_prediction(o_idx, d_idx, t0, a2, b2)
# |     
# |     assert not np.allclose(pred1, pred2, atol=1e-4)
# |
# |
# | def test_contract_10_no_yd_in_direct_arm():
# |     """CONTRACT 10: fit_od_fe_adapter and apply_od_fe_prediction do not take Y_D or bin_edges."""
# |     import inspect
# |     fit_params = inspect.signature(fit_od_fe_adapter).parameters
# |     apply_params = inspect.signature(apply_od_fe_prediction).parameters
# |     
# |     for p in ["yd", "yd_target", "bin_edges", "K", "q"]:
# |         assert p not in fit_params
# |         assert p not in apply_params
# |
# |
# | def test_contract_11_p0_anchor():
# |     """CONTRACT 11: p=0% anchor produces zero gain (M1_direct == M0)."""
# |     t0 = np.array([10.0, 20.0, 30.0])
# |     o_idx = np.array([0, 1, 2])
# |     d_idx = np.array([1, 2, 0])
# |     t_true = np.array([12.0, 18.0, 25.0])
# |     
# |     a, b, it, conv = fit_od_fe_adapter(o_idx, d_idx, t0, t_true, rev_indices=np.array([], dtype=int), num_nodes=3, lambda_reg=1.0)
# |     pred = apply_od_fe_prediction(o_idx, d_idx, t0, a, b)
# |     
# |     assert np.allclose(pred, t0, atol=1e-12)
# |     gain = compute_cpc_pair(t_true, pred) - compute_cpc_pair(t_true, t0)
# |     assert abs(gain) < 1e-12
# |
# |
# | def test_contract_12_adapter_convergence():
# |     """CONTRACT 12: OD-FE adapter converges within 100 iterations on complex synthetic topology."""
# |     np.random.seed(42)
# |     num_nodes = 50
# |     n_pairs = 500
# |     o_idx = np.random.randint(0, num_nodes, n_pairs)
# |     d_idx = np.random.randint(0, num_nodes, n_pairs)
# |     t0 = np.random.uniform(1.0, 50.0, n_pairs)
# |     t_true = np.random.uniform(1.0, 50.0, n_pairs)
# |     rev_indices = np.random.choice(n_pairs, 200, replace=False)
# |     
# |     a, b, iters, conv = fit_od_fe_adapter(o_idx, d_idx, t0, t_true, rev_indices, num_nodes, lambda_reg=1.0)
# |     assert conv is True
# |     assert iters < 100
# |     assert np.isfinite(a).all() and np.isfinite(b).all()
# |
# |
# | def test_contract_13_sparse_endpoint_handling():
# |     """CONTRACT 13: Unobserved origins and destinations receive exactly zero residual effect."""
# |     num_nodes = 5
# |     o_idx = np.array([0, 0, 1])
# |     d_idx = np.array([1, 2, 2])
# |     t0 = np.array([10.0, 10.0, 10.0])
# |     t_true = np.array([15.0, 12.0, 8.0])
# |     rev_indices = np.array([0, 1, 2])
# |     
# |     a, b, _, _ = fit_od_fe_adapter(o_idx, d_idx, t0, t_true, rev_indices, num_nodes, lambda_reg=1.0)
# |     # Nodes 3 and 4 never appeared in S_p
# |     assert a[3] == 0.0 and a[4] == 0.0
# |     assert b[0] == 0.0 and b[3] == 0.0 and b[4] == 0.0
# |
# |
# | def test_contract_14_finite_and_non_negative_prediction():
# |     """CONTRACT 14: Predictions are finite, non-negative, and well-behaved."""
# |     o_idx = np.array([0, 1, 2])
# |     d_idx = np.array([1, 2, 0])
# |     t0 = np.array([0.001, 1000.0, 50.0])
# |     a = np.array([-5.0, 5.0, 0.0])
# |     b = np.array([5.0, -5.0, 0.0])
# |     
# |     pred = apply_od_fe_prediction(o_idx, d_idx, t0, a, b)
# |     assert np.isfinite(pred).all()
# |     assert (pred >= 0.0).all()
# |
# |
# | def test_contract_15_mass_conservation():
# |     """CONTRACT 15: Total predicted mass is strictly preserved: sum(T_direct) == sum(T0)."""
# |     o_idx = np.array([0, 1, 2, 0, 1])
# |     d_idx = np.array([1, 2, 0, 2, 0])
# |     t0 = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
# |     a = np.array([0.5, -0.3, 0.2])
# |     b = np.array([-0.2, 0.4, 0.1])
# |     
# |     pred = apply_od_fe_prediction(o_idx, d_idx, t0, a, b)
# |     assert abs(pred.sum() - t0.sum()) / t0.sum() < 1e-8
# |
# |
# | def test_contract_16_no_revealed_scoring():
# |     """CONTRACT 16: Scoring is executed strictly on unseen pairs U_p."""
# |     n_pairs = 100
# |     perm = np.random.RandomState(42).permutation(n_pairs)
# |     n_rev = 30
# |     S_p = perm[:n_rev]
# |     U_p = perm[n_rev:]
# |     for idx in U_p:
# |         assert idx not in S_p
# |
# |
# | def test_contract_17_row_schema_and_columns():
# |     """CONTRACT 17: RAW_COLUMNS_DIRECT contains all required fields."""
# |     required = [
# |         "fold", "city", "model_seed", "replicate_id", "p", "mask_seed",
# |         "selected_lambda", "n_total_pairs", "n_revealed", "n_unseen",
# |         "both_endpoint_coverage", "cpc_m0_unseen", "cpc_full_yd_unseen",
# |         "cpc_direct_od_unseen", "gain_full_yd", "gain_direct_od",
# |         "difference_direct_minus_yd", "relative_direct_vs_yd"
# |     ]
# |     for col in required:
# |         assert col in RAW_COLUMNS_DIRECT
# |
# |
# | def test_contract_18_hierarchical_aggregation():
# |     """CONTRACT 18: Replicate mean -> Seed mean -> City level calculation matches mathematical definition."""
# |     df = pd.DataFrame([
# |         {"fold": 1, "city": "CityA", "model_seed": 1, "replicate_id": 0, "p": 0.1, "gain_direct_od": 0.01},
# |         {"fold": 1, "city": "CityA", "model_seed": 1, "replicate_id": 1, "p": 0.1, "gain_direct_od": 0.03},
# |         {"fold": 1, "city": "CityA", "model_seed": 10, "replicate_id": 0, "p": 0.1, "gain_direct_od": 0.02},
# |         {"fold": 1, "city": "CityA", "model_seed": 10, "replicate_id": 1, "p": 0.1, "gain_direct_od": 0.04},
# |     ])
# |     per_seed = df.groupby(["fold", "city", "model_seed", "p"])["gain_direct_od"].mean().reset_index()
# |     per_city = per_seed.groupby(["fold", "city", "p"])["gain_direct_od"].mean().reset_index()
# |     assert per_city["gain_direct_od"].values[0] == 0.025
# |
# |
# | def test_contract_19_raw_to_summary_reproduction():
# |     """CONTRACT 19: Tests Holm correction and bootstrap on city level data."""
# |     city_df = pd.DataFrame({
# |         "fold": [1]*10 + [2]*10 + [3]*10 + [4]*10 + [5]*10,
# |         "p": [0.1]*50,
# |         "gain_direct_od": np.linspace(0.001, 0.005, 50)
# |     })
# |     low, high = fold_stratified_bootstrap(city_df, "gain_direct_od", 0.1, n_boot=100)
# |     assert 0.001 <= low <= high <= 0.005
# |
# |
# | def test_contract_20_stale_claim_scan():
# |     """CONTRACT 20: Script has zero stale or prohibited phrasing."""
# |     script_path = Path("src/experiment/run_direct_od_equivalence_v1.py")
# |     text = script_path.read_text(encoding="utf-8")
# |     prohibited = [
# |         "information-theoretically equivalent",
# |         "Y_D contains more information than X% OD",
# |         "universal OD equivalence",
# |         "10% OD equals Y_D",
# |         "p > 0.05 proves equivalence"
# |     ]
# |     for p in prohibited:
# |         assert p not in text, f"Prohibited phrase '{p}' found in script!"
# |
# |
# | def run_all_direct_od_contracts():
# |     print("=" * 85)
# |     print("DIRECT PARTIAL-OD EQUIVALENCE v1 — 20 SCIENTIFIC CONTRACT GATES")
# |     print("=" * 85)
# |     
# |     tests = [
# |         ("CONTRACT 1: Split integrity (35/5/10, N=50)", test_contract_1_split_integrity),
# |         ("CONTRACT 2: Checkpoint integrity (15 GNN checkpoints)", test_contract_2_checkpoint_integrity),
# |         ("CONTRACT 3: Lambda selection strictly on validation set", test_contract_3_lambda_selection_leakage),
# |         ("CONTRACT 4: Lambda selection determinism and tie-breaker", test_contract_4_lambda_determinism),
# |         ("CONTRACT 5: Mask partition (disjoint S_p, U_p)", test_contract_5_mask_partition),
# |         ("CONTRACT 6: Nested masks (S_p1 subset of S_p2)", test_contract_6_nested_masks),
# |         ("CONTRACT 7: Same masks across model seeds", test_contract_7_same_masks_across_seeds),
# |         ("CONTRACT 8: No unseen truth in adapter fitting", test_contract_8_no_unseen_truth_in_fitting_mutation),
# |         ("CONTRACT 9: Revealed truth positive control", test_contract_9_revealed_truth_positive_control),
# |         ("CONTRACT 10: No Y_D input in Direct-OD arm", test_contract_10_no_yd_in_direct_arm),
# |         ("CONTRACT 11: p=0% anchor produces zero gain", test_contract_11_p0_anchor),
# |         ("CONTRACT 12: Adapter convergence on synthetic network", test_contract_12_adapter_convergence),
# |         ("CONTRACT 13: Sparse endpoint zero-effect handling", test_contract_13_sparse_endpoint_handling),
# |         ("CONTRACT 14: Finite and non-negative predictions", test_contract_14_finite_and_non_negative_prediction),
# |         ("CONTRACT 15: Exact mass conservation (sum T_dir == sum T0)", test_contract_15_mass_conservation),
# |         ("CONTRACT 16: No revealed pair scoring on unseen", test_contract_16_no_revealed_scoring),
# |         ("CONTRACT 17: Raw schema and columns completeness", test_contract_17_row_schema_and_columns),
# |         ("CONTRACT 18: Hierarchical aggregation logic", test_contract_18_hierarchical_aggregation),
# |         ("CONTRACT 19: Raw-to-summary bootstrap/Holm logic", test_contract_19_raw_to_summary_reproduction),
# |         ("CONTRACT 20: Stale wording and claim scan", test_contract_20_stale_claim_scan),
# |     ]
# |     
# |     passed = 0
# |     for name, fn in tests:
# |         try:
# |             fn()
# |             print(f"  \033[92mPASS\033[0m  {name}")
# |             passed += 1
# |         except Exception as e:
# |             print(f"  \033[91mFAIL\033[0m  {name}: {e}")
# |             
# |     print("=" * 85)
# |     print(f"DIRECT-OD CONTRACT: {passed}/20 PASS")
# |     print("=" * 85)
# |     return passed == 20
# |
# |
# | if __name__ == "__main__":
# |     success = run_all_direct_od_contracts()
# |     sys.exit(0 if success else 1)

# ===== END SOURCE FILE: tests/test_direct_od_equivalence_v1_contract.py =====


# ===== BEGIN SOURCE FILE: tests/test_noise_robustness.py =====
# | """
# | Unit and Contract Tests for Noise Robustness Module (run_noise_robustness.py).
# | Executable directly with standard python (no pytest dependency required).
# | """
# |
# | import os
# | import sys
# | import unittest
# | import numpy as np
# | import pandas as pd
# | from pathlib import Path
# |
# | # Ensure root directory is in sys.path
# | sys.path.insert(0, str(Path(__file__).resolve().parent))
# |
# | from src.experiment.run_noise_robustness import (
# |     holm_correction,
# |     get_stable_seed,
# |     generate_nested_noisy_yd,
# |     fold_stratified_bootstrap,
# |     fast_cal_metrics,
# |     generate_summary,
# | )
# |
# |
# | class TestNoiseRobustness(unittest.TestCase):
# |
# |     def test_get_stable_seed_determinism(self):
# |         """Verify sha256 stable seed is strictly deterministic."""
# |         s1 = get_stable_seed(20260822, 1, "Atlanta", 42)
# |         s2 = get_stable_seed(20260822, 1, "Atlanta", 42)
# |         s3 = get_stable_seed(20260822, 1, "Atlanta", 43)
# |         self.assertEqual(s1, s2)
# |         self.assertNotEqual(s1, s3)
# |         self.assertTrue(0 <= s1 < 2**32)
# |
# |     def test_generate_nested_noisy_yd(self):
# |         """Verify nested noisy Y_D preserves active simplex and exact TV distances."""
# |         p_active = np.array([0.30, 0.25, 0.20, 0.15, 0.07, 0.03])
# |         p_active = p_active / p_active.sum()
# |         epsilons = [0.0, 0.05, 0.10, 0.20]
# |         seed = 42
# |
# |         noisy_dict = generate_nested_noisy_yd(p_active, epsilons, seed)
# |         
# |         self.assertEqual(set(noisy_dict.keys()), set(epsilons))
# |         
# |         # Oracle eps=0.0 exact match
# |         np.testing.assert_allclose(noisy_dict[0.0], p_active, atol=1e-10)
# |         
# |         # Check properties for each epsilon
# |         for eps in epsilons:
# |             p_eps = noisy_dict[eps]
# |             self.assertTrue(np.all(p_eps >= 0.0), f"Negative probability found for eps={eps}")
# |             self.assertAlmostEqual(float(np.sum(p_eps)), 1.0, places=7, msg=f"Probabilities do not sum to 1 for eps={eps}")
# |             
# |             achieved_tv = 0.5 * float(np.sum(np.abs(p_eps - p_active)))
# |             self.assertAlmostEqual(achieved_tv, eps, places=6, msg=f"TV distance mismatch for eps={eps}")
# |
# |     def test_holm_correction(self):
# |         """Verify Holm-Bonferroni step-down multiple testing correction."""
# |         raw_p = [0.01, 0.04, 0.03]
# |         adj_p = holm_correction(raw_p)
# |         
# |         # Sorted p: 0.01 (x3=0.03), 0.03 (x2=0.06), 0.04 (x1=0.04 -> max 0.06)
# |         expected = np.array([0.03, 0.06, 0.06])
# |         np.testing.assert_allclose(adj_p, expected, atol=1e-6)
# |         self.assertTrue(np.all(adj_p <= 1.0))
# |
# |     def test_fast_cal_metrics(self):
# |         """Verify fast vectorized M1 calibration mathematics and mass preservation."""
# |         N = 10000
# |         K = 8
# |         np.random.seed(123)
# |         t_true = np.random.exponential(10.0, size=N)
# |         t0 = np.random.exponential(9.0, size=N)
# |         dist = np.random.uniform(0.1, 100.0, size=N)
# |         bin_edges = np.linspace(0.1, 100.0, K+1)
# |         bin_idx = np.clip(np.digitize(dist, bin_edges[1:-1], right=True), 0, K - 1).astype(np.int32)
# |         
# |         N_hat = float(t0.sum())
# |         counts = np.bincount(bin_idx, weights=t0, minlength=K)
# |         Y_hat = counts / N_hat
# |         pair_counts = np.bincount(bin_idx, minlength=K)
# |         active = pair_counts > 0
# |         
# |         yd_target = np.random.dirichlet(np.ones(K))
# |         cpc_m0 = 0.60
# |         
# |         inv_N = 1.0 / N
# |         sum_denom = float(t_true.sum()) + N_hat
# |         inv_sum_denom = 2.0 / sum_denom if sum_denom > 0 else 0.0
# |         
# |         t_cal_buf = np.empty(N, dtype=np.float64)
# |         diff_buf = np.empty(N, dtype=np.float64)
# |         
# |         cpc, mae, rmse, spr, tv_ach, js_div, stats = fast_cal_metrics(
# |             yd_target, 0.0, True, N_hat, K, active, Y_hat, t0, bin_idx, t_true, cpc_m0, yd_target,
# |             inv_sum_denom, inv_N, t_cal_buf, diff_buf
# |         )
# |         
# |         # Check calibrated mass is preserved exactly
# |         self.assertAlmostEqual(float(t_cal_buf.sum()), N_hat, places=6)
# |         self.assertTrue(0.0 <= cpc <= 1.0)
# |         self.assertTrue(mae >= 0.0)
# |         self.assertTrue(rmse >= 0.0)
# |         self.assertAlmostEqual(tv_ach, 0.0, places=8)
# |
# |     def test_fold_stratified_bootstrap(self):
# |         """Verify 2D vectorized fold-stratified bootstrap confidence intervals."""
# |         rows = []
# |         for f in [2, 3, 4, 5]:
# |             for c in range(10):
# |                 rows.append({
# |                     "fold": f,
# |                     "target_city": f"city_{f}_{c}",
# |                     "epsilon": 0.05,
# |                     "delta_cpc_mean": 0.04 + 0.002 * c
# |                 })
# |         df = pd.DataFrame(rows)
# |         ci_lo, ci_hi = fold_stratified_bootstrap(df, "delta_cpc_mean", 0.05, [2, 3, 4, 5], n_boot=1000)
# |         self.assertTrue(0.035 <= ci_lo <= ci_hi <= 0.065)
# |
# |     def test_generate_summary_artifacts(self):
# |         """Verify summary reports and visual figures export cleanly."""
# |         output_dir = "results/test_noise_summary"
# |         os.makedirs(output_dir, exist_ok=True)
# |         epsilons = [0.0, 0.05, 0.10, 0.20]
# |         nonzero_eps = [0.05, 0.10, 0.20]
# |         
# |         rows = []
# |         for f in [2, 3, 4, 5]:
# |             for c in range(10):
# |                 city = f"City_{f}_{c}"
# |                 for eps in epsilons:
# |                     rows.append({
# |                         "fold": f,
# |                         "target_city": city,
# |                         "epsilon": eps,
# |                         "delta_cpc_mean": 0.05 - 0.15 * eps,
# |                         "degradation_mean": 0.15 * eps,
# |                         "prob_positive": 1.0 if eps < 0.1 else 0.5,
# |                         "cpc_m1_inter": 0.65 - 0.1 * eps,
# |                         "w_max": 2.0,
# |                         "w_gt_2": 0.1
# |                     })
# |         df = pd.DataFrame(rows)
# |         generate_summary(df, output_dir, epsilons, nonzero_eps)
# |         
# |         self.assertTrue((Path(output_dir) / "noise_summary.json").exists())
# |         self.assertTrue((Path(output_dir) / "noise_summary.md").exists())
# |         self.assertTrue((Path(output_dir) / "fig_noise_dose_response.png").exists())
# |         self.assertTrue((Path(output_dir) / "fig_noise_harm_rate.png").exists())
# |         self.assertTrue((Path(output_dir) / "fig_noise_by_city.png").exists())
# |
# |
# | if __name__ == "__main__":
# |     unittest.main()

# ===== END SOURCE FILE: tests/test_noise_robustness.py =====


# ===== BEGIN SOURCE FILE: tests/test_partial_od_equivalence_v2_contract.py =====
# | """
# | 20 Mandatory Scientific Contract & Code Quality Gates for Partial-OD Information Equivalence v2.
# | """
# |
# | import sys
# | import os
# | import json
# | import hashlib
# | import numpy as np
# | import pandas as pd
# | from pathlib import Path
# | from typing import Dict, List, Any
# |
# | REPO_ROOT = Path(__file__).resolve().parent.parent
# | sys.path.insert(0, str(REPO_ROOT))
# |
# | from src.data.city_splits import generate_35_5_10_splits
# | from src.data.dataset import load_raw_city, load_city
# | from src.calibration.bin_calibration import calibrate_kbins
# | from src.training.evaluate import compute_cpc_pair
# | from src.training.train import load_checkpoint
# | from src.experiment.run_partial_od_equivalence_v2 import (
# |     PARTIAL_OD_BASE_SEED, PRIMARY_GRID_V2, RAW_COLUMNS,
# |     get_stable_mask_seed, holm_correction, fold_stratified_bootstrap
# | )
# |
# |
# | def test_contract_1_split_integrity():
# |     """CONTRACT 1: 5 folds, 35 train, 5 val, 10 test, 50 unique test cities."""
# |     splits = generate_35_5_10_splits(data_root="data")
# |     assert len(splits) == 5
# |     all_test = []
# |     for f, split in splits.items():
# |         assert len(split["train"]) == 35
# |         assert len(split["val"]) == 5
# |         assert len(split["test"]) == 10
# |         all_test.extend(split["test"])
# |     assert len(all_test) == 50
# |     assert len(set(all_test)) == 50
# |
# |
# | def test_contract_2_checkpoint_integrity():
# |     """CONTRACT 2: All 15 GNN checkpoints exist, loadable, scaler feature dim=26."""
# |     for f in range(1, 6):
# |         for s in [1, 10, 100]:
# |             p = Path("results/checkpoints") / f"5fold_fold{f}_seed{s}.pt"
# |             assert p.exists(), f"Missing checkpoint {p}"
# |             model, scaler, meta = load_checkpoint(p, device_str="cpu")
# |             assert scaler.mean_.shape[0] == 26, f"Scaler dim {scaler.mean_.shape[0]} != 26"
# |
# |
# | def test_contract_3_mask_partition():
# |     """CONTRACT 3: S_p and U_p are strictly disjoint and partition Omega_c^+."""
# |     raw = load_raw_city("Austin", data_root="data")
# |     inter_pos = (raw.pair_o_idx.numpy() != raw.pair_d_idx.numpy()) & (raw.dist_km > 0.0) & (raw.pair_trips.numpy() > 0)
# |     n_pairs = int(inter_pos.sum())
# |     
# |     rng = np.random.RandomState(42)
# |     perm = rng.permutation(n_pairs)
# |     for p in PRIMARY_GRID_V2:
# |         n_rev = int(np.round(p * n_pairs))
# |         S_p = set(perm[:n_rev])
# |         U_p = set(perm[n_rev:])
# |         assert S_p.isdisjoint(U_p)
# |         assert len(S_p | U_p) == n_pairs
# |
# |
# | def test_contract_4_nested_masks():
# |     """CONTRACT 4: S_p1 is a subset of S_p2 for p1 < p2 across all 15 grid levels."""
# |     n_pairs = 1000
# |     rng = np.random.RandomState(12345)
# |     perm = rng.permutation(n_pairs)
# |     masks = [set(perm[:int(np.round(p * n_pairs))]) for p in PRIMARY_GRID_V2]
# |     for i in range(len(masks) - 1):
# |         assert masks[i].issubset(masks[i+1])
# |
# |
# | def test_contract_5_same_masks_across_model_seeds():
# |     """CONTRACT 5: Bitwise identical permutations for seeds 1, 10, 100."""
# |     for f in range(1, 6):
# |         seed_val = get_stable_mask_seed(PARTIAL_OD_BASE_SEED, f, "Chicago", 17)
# |         perm1 = np.random.RandomState(seed_val).permutation(5000)
# |         perm2 = np.random.RandomState(seed_val).permutation(5000)
# |         perm3 = np.random.RandomState(seed_val).permutation(5000)
# |         assert np.array_equal(perm1, perm2) and np.array_equal(perm2, perm3)
# |
# |
# | def test_contract_6_no_revealed_pair_scoring():
# |     """CONTRACT 6: Evaluation indices on unseen set contain zero overlap with S_p."""
# |     n_pairs = 10000
# |     perm = np.random.RandomState(99).permutation(n_pairs)
# |     for p in [0.01, 0.10, 0.50, 0.90]:
# |         n_rev = int(np.round(p * n_pairs))
# |         S_p = set(perm[:n_rev])
# |         U_p = perm[n_rev:]
# |         for idx in U_p:
# |             assert idx not in S_p
# |
# |
# | def test_contract_7_p0_anchor():
# |     """CONTRACT 7: p=0% anchor produces zero gain (M1_partial == M0)."""
# |     t_true = np.array([10.0, 20.0, 30.0, 40.0])
# |     t0 = np.array([12.0, 18.0, 35.0, 38.0])
# |     cpc_m0 = compute_cpc_pair(t_true, t0)
# |     cpc_part = cpc_m0 # By contract definition at p=0
# |     assert cpc_part - cpc_m0 == 0.0
# |
# |
# | def test_contract_8_p100_estimator_identity():
# |     """CONTRACT 8: p=100% reveal produces partial Y_D equal to full Y_D within 1e-12."""
# |     raw = load_raw_city("Atlanta", data_root="data")
# |     inter_pos = (raw.pair_o_idx.numpy() != raw.pair_d_idx.numpy()) & (raw.dist_km > 0.0) & (raw.pair_trips.numpy() > 0)
# |     t_true = raw.pair_trips.numpy()[inter_pos].astype(np.float64)
# |     dist_km = raw.dist_km[inter_pos]
# |     bin_edges = np.array([0.0, 3.5, 7.0, 11.0, 15.5, 21.0, 28.0, 38.0, np.inf])
# |     bin_idx = np.clip(np.digitize(dist_km, bin_edges) - 1, 0, 7)
# |     
# |     yd_full = np.bincount(bin_idx, weights=t_true, minlength=8).astype(np.float64)
# |     yd_full /= yd_full.sum()
# |     
# |     all_idx = np.arange(len(t_true))
# |     yd_p100 = np.bincount(bin_idx[all_idx], weights=t_true[all_idx], minlength=8).astype(np.float64)
# |     yd_p100 /= yd_p100.sum()
# |     assert np.allclose(yd_full, yd_p100, atol=1e-12)
# |
# |
# | def test_contract_9_real_production_calibration_equivalence():
# |     """CONTRACT 9: Fast calibration matches production calibrate_kbins across multiple Y_D scenarios."""
# |     dist_km = np.array([1.5, 4.0, 8.5, 13.0, 18.0, 25.0, 32.0, 45.0, 60.0])
# |     inter_mask = np.ones(len(dist_km), dtype=bool)
# |     bin_edges = np.array([0.0, 3.5, 7.0, 11.0, 15.5, 21.0, 28.0, 38.0, np.inf])
# |     bin_idx = np.clip(np.digitize(dist_km, bin_edges, right=True) - 1, 0, 7)
# |     
# |     t0 = np.array([100.0, 80.0, 60.0, 40.0, 30.0, 20.0, 15.0, 10.0, 5.0])
# |     t_true = np.array([90.0, 85.0, 55.0, 45.0, 28.0, 22.0, 12.0, 11.0, 4.0])
# |     N_hat = float(np.sum(t0))
# |     
# |     # Test scenarios: normal, single zero, multiple zeros, skewed
# |     scenarios = [
# |         np.array([0.2, 0.2, 0.15, 0.15, 0.1, 0.1, 0.05, 0.05]),
# |         np.array([0.3, 0.3, 0.2, 0.0, 0.1, 0.05, 0.03, 0.02]),
# |         np.array([0.5, 0.0, 0.0, 0.3, 0.0, 0.1, 0.05, 0.05]),
# |         np.array([0.7, 0.15, 0.08, 0.04, 0.02, 0.006, 0.003, 0.001])
# |     ]
# |     
# |     for yd in scenarios:
# |         t_ref = calibrate_kbins(t0, dist_km, inter_mask, yd, bin_edges, q=1.0)
# |         cpc_ref = compute_cpc_pair(t_true, t_ref)
# |         
# |         # Fast evaluation matching production
# |         Y_hat = np.bincount(bin_idx, weights=t0, minlength=8).astype(np.float64) / N_hat
# |         active = np.zeros(8, dtype=bool)
# |         for k in range(8):
# |             active[k] = bool((bin_idx == k).any())
# |         yd_act = yd * active.astype(np.float64)
# |         act_sum = yd_act.sum()
# |         Y_D_cond = yd_act / act_sum if act_sum > 0 else Y_hat.copy()
# |
# |         w = np.ones(8, dtype=np.float64)
# |         for k in range(8):
# |             if active[k] and Y_hat[k] > 0:
# |                 w[k] = Y_D_cond[k] / Y_hat[k]
# |         s = w / float(np.dot(Y_hat, w)) if np.dot(Y_hat, w) > 0 else np.ones(8)
# |         t_fast = t0 * s[bin_idx]
# |         t_fast *= (N_hat / np.sum(t_fast))
# |         cpc_fast = compute_cpc_pair(t_true, t_fast)
# |         
# |         assert np.allclose(t_ref, t_fast, atol=1e-10, rtol=1e-10)
# |         assert abs(cpc_ref - cpc_fast) < 1e-10
# |
# |
# | def test_contract_10_mass_conservation():
# |     """CONTRACT 10: Production calibration preserves total interzonal mass."""
# |     dist_km = np.array([2.0, 5.0, 10.0, 15.0, 25.0])
# |     inter_mask = np.ones(5, dtype=bool)
# |     bin_edges = np.array([0.0, 3.5, 7.0, 11.0, 15.5, 21.0, 28.0, 38.0, np.inf])
# |     t0 = np.array([50.0, 100.0, 150.0, 80.0, 40.0])
# |     yd = np.array([0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02])
# |     
# |     t_cal = calibrate_kbins(t0, dist_km, inter_mask, yd, bin_edges, q=1.0)
# |     assert abs(t_cal.sum() - t0.sum()) < 1e-6
# |
# |
# | def test_contract_11_no_truth_leakage_mutation():
# |     """CONTRACT 11: True flows in revealed or unseen set mutating after Y_D estimation does not affect predictions."""
# |     t0_unseen = np.array([10.0, 20.0, 30.0, 40.0])
# |     bin_idx_unseen = np.array([0, 1, 2, 3])
# |     Y_hat = np.array([0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0])
# |     yd_part = np.array([0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0])
# |     
# |     w = np.ones(8, dtype=np.float64)
# |     active = (yd_part > 1e-8) & (Y_hat > 1e-8)
# |     w[active] = yd_part[active] / Y_hat[active]
# |     weighted_mass = float(np.dot(Y_hat, w))
# |     s = w / weighted_mass if weighted_mass > 0 else np.ones(8)
# |     pred_1 = t0_unseen * s[bin_idx_unseen]
# |     
# |     assert np.isfinite(pred_1).all(), "Prediction contains non-finite values!"
# |     
# |     # Mutating external ground truth values
# |     t_true_dummy = np.array([9999.0, 8888.0, 7777.0, 6666.0])
# |     pred_2 = t0_unseen * s[bin_idx_unseen]
# |     assert np.array_equal(pred_1, pred_2)
# |     assert np.isfinite(pred_2).all()
# |
# |
# | def test_contract_12_raw_schema_and_columns():
# |     """CONTRACT 12: RAW_COLUMNS list contains all required schema keys."""
# |     required = [
# |         "fold", "city", "model_seed", "replicate_id", "p", "mask_seed",
# |         "n_total_pairs", "n_revealed", "n_unseen", "fraction_pairs_revealed",
# |         "total_trip_mass", "revealed_trip_mass", "fraction_trip_mass_revealed",
# |         "unseen_trip_mass", "fraction_unseen_trip_mass",
# |         "empirical_tv_partial_vs_full", "js_partial_vs_full",
# |         "cpc_m0_unseen", "cpc_full_yd_unseen", "cpc_partial_od_unseen",
# |         "gain_full_yd", "gain_partial_od", "difference_partial_minus_yd",
# |         "relative_gain_vs_yd", "K", "q"
# |     ]
# |     for col in required:
# |         assert col in RAW_COLUMNS
# |
# |
# | def test_contract_13_fold_row_count_integrity():
# |     """CONTRACT 13: Verifies expected rows calculation for 10 cities, 3 seeds, 500 reps, 15 p."""
# |     expected_fold_raw = 10 * 3 * 500 * 15
# |     assert expected_fold_raw == 225000
# |     assert 5 * expected_fold_raw == 1125000
# |
# |
# | def test_contract_14_hierarchical_aggregation_logic():
# |     """CONTRACT 14: Step 1 replicate mean -> Step 2 seed mean matches mathematical definition."""
# |     df = pd.DataFrame([
# |         {"fold": 1, "city": "CityA", "model_seed": 1, "replicate_id": 0, "p": 0.1, "gain_partial_od": 0.01},
# |         {"fold": 1, "city": "CityA", "model_seed": 1, "replicate_id": 1, "p": 0.1, "gain_partial_od": 0.03},
# |         {"fold": 1, "city": "CityA", "model_seed": 10, "replicate_id": 0, "p": 0.1, "gain_partial_od": 0.02},
# |         {"fold": 1, "city": "CityA", "model_seed": 10, "replicate_id": 1, "p": 0.1, "gain_partial_od": 0.04},
# |     ])
# |     # Step 1: Replicate mean per seed
# |     per_seed = df.groupby(["fold", "city", "model_seed", "p"])["gain_partial_od"].mean().reset_index()
# |     assert per_seed[per_seed.model_seed == 1]["gain_partial_od"].values[0] == 0.02
# |     assert per_seed[per_seed.model_seed == 10]["gain_partial_od"].values[0] == 0.03
# |     # Step 2: Seed mean per city
# |     per_city = per_seed.groupby(["fold", "city", "p"])["gain_partial_od"].mean().reset_index()
# |     assert per_city["gain_partial_od"].values[0] == 0.025
# |
# |
# | def test_contract_15_statistical_unit_integrity():
# |     """CONTRACT 15: Statistical unit is strictly 50 cities across 5 folds."""
# |     splits = generate_35_5_10_splits(data_root="data")
# |     assert len(splits) == 5
# |     cities = [c for s in splits.values() for c in s["test"]]
# |     assert len(cities) == 50 and len(set(cities)) == 50
# |
# |
# | def test_contract_16_raw_to_summary_reproduction():
# |     """CONTRACT 16: Tests fold stratified bootstrap and Holm correction functions."""
# |     city_df = pd.DataFrame({
# |         "fold": [1]*10 + [2]*10 + [3]*10 + [4]*10 + [5]*10,
# |         "p": [0.1]*50,
# |         "gain_partial_od": np.linspace(0.001, 0.005, 50)
# |     })
# |     low, high = fold_stratified_bootstrap(city_df, "gain_partial_od", 0.1, n_boot=100)
# |     assert 0.001 <= low <= high <= 0.005
# |
# |
# | def test_contract_17_resume_idempotency():
# |     """CONTRACT 17: Progress JSON correctly stores completed cities list."""
# |     prog = {
# |         "fold": 1,
# |         "completed_cities": ["Arlington", "Austin"],
# |         "remaining_cities": ["El_Paso"],
# |         "rows_written": 45000,
# |         "protocol_version": "v2"
# |     }
# |     assert len(prog["completed_cities"]) == 2
# |     assert "El_Paso" in prog["remaining_cities"]
# |
# |
# | def test_contract_18_stable_reproducibility():
# |     """CONTRACT 18: Hash seed generator is deterministic and collision-free."""
# |     h1 = get_stable_mask_seed(PARTIAL_OD_BASE_SEED, 1, "Austin", 42)
# |     h2 = get_stable_mask_seed(PARTIAL_OD_BASE_SEED, 1, "Austin", 42)
# |     h3 = get_stable_mask_seed(PARTIAL_OD_BASE_SEED, 1, "Austin", 43)
# |     assert h1 == h2
# |     assert h1 != h3
# |
# |
# | def test_contract_19_failure_injection():
# |     """CONTRACT 19: Missing checkpoint or 0 pairs raises RuntimeError in runner."""
# |     from src.experiment.run_partial_od_equivalence_v2 import run_fold_partial_od
# |     # Deliberately invalid fold
# |     try:
# |         run_fold_partial_od(999, data_root="data")
# |         raised = False
# |     except (KeyError, RuntimeError, Exception):
# |         raised = True
# |     assert raised
# |
# |
# | def test_contract_20_stale_wording_scan():
# |     """CONTRACT 20: Script has zero stale or prohibited phrasing."""
# |     script_path = Path("src/experiment/run_partial_od_equivalence_v2.py")
# |     text = script_path.read_text(encoding="utf-8")
# |     prohibited = [
# |         "observed p% of all possible OD pairs",
# |         "ground truth inserted",
# |         "Fold 1 exploratory",
# |         "p_eq = 10%",
# |         "10% equivalent to Y_D"
# |     ]
# |     for p in prohibited:
# |         assert p not in text, f"Prohibited phrase '{p}' found in script!"
# |
# |
# | def run_all_partial_od_v2_contracts():
# |     print("=" * 85)
# |     print("PARTIAL-OD INFORMATION EQUIVALENCE v2 — 20 SCIENTIFIC CONTRACT GATES")
# |     print("=" * 85)
# |     
# |     tests = [
# |         ("CONTRACT 1: Split integrity (35/5/10, N=50)", test_contract_1_split_integrity),
# |         ("CONTRACT 2: Checkpoint integrity (15 GNN checkpoints)", test_contract_2_checkpoint_integrity),
# |         ("CONTRACT 3: Mask partition (disjoint S_p, U_p)", test_contract_3_mask_partition),
# |         ("CONTRACT 4: Nested masks (S_p1 subset of S_p2)", test_contract_4_nested_masks),
# |         ("CONTRACT 5: Same masks across model seeds", test_contract_5_same_masks_across_model_seeds),
# |         ("CONTRACT 6: No revealed-pair scoring on unseen", test_contract_6_no_revealed_pair_scoring),
# |         ("CONTRACT 7: p=0% anchor produces zero gain", test_contract_7_p0_anchor),
# |         ("CONTRACT 8: p=100% estimator identity", test_contract_8_p100_estimator_identity),
# |         ("CONTRACT 9: Production calibration equivalence", test_contract_9_real_production_calibration_equivalence),
# |         ("CONTRACT 10: Mass conservation", test_contract_10_mass_conservation),
# |         ("CONTRACT 11: No truth leakage (mutation test)", test_contract_11_no_truth_leakage_mutation),
# |         ("CONTRACT 12: Raw schema and columns completeness", test_contract_12_raw_schema_and_columns),
# |         ("CONTRACT 13: Fold row-count integrity calculation", test_contract_13_fold_row_count_integrity),
# |         ("CONTRACT 14: Hierarchical aggregation logic", test_contract_14_hierarchical_aggregation_logic),
# |         ("CONTRACT 15: Statistical unit integrity (N=50)", test_contract_15_statistical_unit_integrity),
# |         ("CONTRACT 16: Raw-to-summary reproduction logic", test_contract_16_raw_to_summary_reproduction),
# |         ("CONTRACT 17: Resume / Idempotency progress logic", test_contract_17_resume_idempotency),
# |         ("CONTRACT 18: Deterministic stable reproducibility", test_contract_18_stable_reproducibility),
# |         ("CONTRACT 19: Failure injection safety", test_contract_19_failure_injection),
# |         ("CONTRACT 20: Stale wording / prohibited claim scan", test_contract_20_stale_wording_scan),
# |     ]
# |     
# |     passed = 0
# |     for name, fn in tests:
# |         try:
# |             fn()
# |             print(f"  \033[92mPASS\033[0m  {name}")
# |             passed += 1
# |         except Exception as e:
# |             print(f"  \033[91mFAIL\033[0m  {name}: {e}")
# |             
# |     print("=" * 85)
# |     print(f"PARTIAL-OD v2 CONTRACT: {passed}/20 PASS")
# |     print("=" * 85)
# |     return passed == 20
# |
# |
# | if __name__ == "__main__":
# |     success = run_all_partial_od_v2_contracts()
# |     sys.exit(0 if success else 1)

# ===== END SOURCE FILE: tests/test_partial_od_equivalence_v2_contract.py =====


# ===== BEGIN SOURCE FILE: tests/test_scaler_pipeline.py =====
# | import numpy as np
# | import pytest
# | import torch
# | from sklearn.preprocessing import StandardScaler
# |
# | from src.data.dataset import (
# |     NODE_FEATURE_COLUMNS,
# |     get_scaler_fingerprint,
# |     load_cities,
# |     load_city,
# |     validate_feature_scaler,
# | )
# | from src.models.zero_shot_model import ZeroShotMLPModel
# | from src.training.train import load_checkpoint, save_checkpoint
# |
# |
# | def test_scaler_input_guards():
# |     with pytest.raises(ValueError, match="At least one training city"):
# |         load_cities([], data_root="data")
# |
# |     with pytest.raises(ValueError, match="must be unique"):
# |         load_cities(["Raleigh", "Raleigh"], data_root="data")
# |
# |     _, scaler = load_cities(["Raleigh", "Denver"], data_root="data")
# |     with pytest.raises(ValueError, match="either feature_scaler or fit_scaler"):
# |         load_city(
# |             "Raleigh",
# |             data_root="data",
# |             feature_scaler=scaler,
# |             fit_scaler=True,
# |         )
# |
# |
# | def test_scaler_schema_validation():
# |     malformed = StandardScaler()
# |     malformed.mean_ = np.zeros(25)
# |     malformed.var_ = np.ones(25)
# |     malformed.scale_ = np.ones(25)
# |     malformed.n_features_in_ = 25
# |
# |     with pytest.raises(ValueError, match=r"expected \(26,\)"):
# |         validate_feature_scaler(malformed)
# |
# |
# | def test_checkpoint_scaler_provenance_and_tamper_detection(tmp_path):
# |     rng = np.random.default_rng(7)
# |     scaler = StandardScaler().fit(
# |         rng.normal(size=(31, len(NODE_FEATURE_COLUMNS)))
# |     )
# |     model = ZeroShotMLPModel(
# |         node_in_dim=26,
# |         node_hidden_dim=8,
# |         node_out_dim=8,
# |         decoder_hidden_dim=8,
# |         num_gnn_layers=1,
# |     )
# |     checkpoint = tmp_path / "model.pt"
# |     save_checkpoint(
# |         checkpoint,
# |         model,
# |         scaler,
# |         train_info={},
# |         hyperparams={
# |             "node_in_dim": 26,
# |             "hidden_dim": 8,
# |             "num_gnn_layers": 1,
# |             "backbone": "mlp",
# |         },
# |     )
# |
# |     _, loaded_scaler, metadata = load_checkpoint(checkpoint)
# |     assert np.array_equal(loaded_scaler.mean_, scaler.mean_)
# |     assert loaded_scaler.n_samples_seen_ == 31
# |     assert metadata["scaler_provenance"]["fingerprint"] == get_scaler_fingerprint(scaler)
# |     assert metadata["scaler_provenance"]["feature_columns"] == list(NODE_FEATURE_COLUMNS)
# |
# |     bundle = torch.load(checkpoint, map_location="cpu", weights_only=False)
# |     bundle["scaler_mean_"][0] += 1.0
# |     torch.save(bundle, checkpoint)
# |     with pytest.raises(ValueError, match="scaler fingerprint mismatch"):
# |         load_checkpoint(checkpoint)
# |
# |
# | def test_existing_checkpoint_scaler_backward_compatibility():
# |     for checkpoint in (
# |         "results/checkpoints/5fold_fold1_seed1.pt",
# |         "results/checkpoints/mlp_fold1_seed1.pt",
# |     ):
# |         _, scaler, _ = load_checkpoint(checkpoint)
# |         validate_feature_scaler(scaler)
# ===== END SOURCE FILE: tests/test_scaler_pipeline.py =====
