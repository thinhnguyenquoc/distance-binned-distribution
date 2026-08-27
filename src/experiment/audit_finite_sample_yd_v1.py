"""Comprehensive Audit & Precision Certification for Finite-Sample Y_D Observation Robustness v1.

Validates frozen result artifacts without rerunning model inference or mutating results.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_SAMPLE_SIZES = [50, 100, 250, 500, 1000, 2500, 5000]
EXPECTED_FOLDS = [1, 2, 3, 4, 5]
EXPECTED_SEEDS = [1, 10, 100]
EXPECTED_CITIES = 50
EXPECTED_REPLICATES = 1000
SUMMARY_TOLERANCE = 1e-10
CI_TOLERANCE = 1e-10


class AuditFailure(Exception):
    """Raised when a scientific artifact fails an audit gate."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditFailure(message)


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise AuditFailure(f"{label} cannot be boolean")
    try:
        number = float(value)
    except (ValueError, TypeError) as exc:
        raise AuditFailure(f"{label} is not a valid float: {value}") from exc
    _require(math.isfinite(number), f"{label} is not finite")
    return number


def _load(path: Path) -> Any:
    _require(path.exists(), f"Missing required artifact: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AuditFailure(f"Invalid JSON: {path}: {exc}") from exc


def _load_raw_replicates(path: Path) -> list[dict[str, Any]]:
    _require(path.exists(), f"Missing required artifact: {path}")
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise AuditFailure(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def _fold_bootstrap(rows: list[dict[str, Any]], sample: str, field: str, n_boot: int = 10000) -> tuple[float, float]:
    """Reproduce the runner's fold-stratified city bootstrap from per-city rows."""
    import numpy as np

    rng = np.random.default_rng(42)
    fold_means = []
    for fold in EXPECTED_FOLDS:
        values = np.array([row[field] for row in rows if row["sample"] == sample and row["fold"] == fold], dtype=float)
        _require(len(values) == 10, f"{sample}: expected 10 values in fold {fold} for bootstrap")
        sampled = values[rng.integers(0, len(values), size=(n_boot, len(values)))]
        fold_means.append(sampled.mean(axis=1))
    means = np.column_stack(fold_means).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _close(actual: float, expected: float, tolerance: float = SUMMARY_TOLERANCE) -> bool:
    return abs(actual - expected) <= tolerance


def audit_results(results_dir: Path) -> dict[str, Any]:
    summary = _load(results_dir / "summary.json")
    per_city = _load(results_dir / "per_city.json")
    raw_replicates = _load_raw_replicates(results_dir / "raw_replicates.jsonl")
    _require(isinstance(per_city, list), "per_city.json must contain a list")

    protocol = summary.get("protocol", {})
    _require(protocol.get("name") == "Finite-Sample Y_D Observation Robustness v1", "Protocol name mismatch")
    _require(protocol.get("K") == 8, "K must be 8")
    _require(protocol.get("bins") == "quantile", "Primary audit requires quantile bins")
    _require(protocol.get("sample_sizes") == EXPECTED_SAMPLE_SIZES, "Sample-size grid mismatch")
    _require(protocol.get("replicates_per_city") == EXPECTED_REPLICATES, "Replicate count mismatch")
    _require(protocol.get("raw_replicate_artifact") is True, "Raw replicate artifact flag is not true")
    _require(protocol.get("nested_multinomial") is True, "Nested multinomial flag is not true")
    _require(protocol.get("no_retraining") is True, "No-retraining flag is not true")
    _require(protocol.get("model_seeds") == EXPECTED_SEEDS, "Model seed list mismatch")
    _require(protocol.get("statistical_unit") == "city", "Statistical unit must be city")

    expected_keys = [str(size) for size in EXPECTED_SAMPLE_SIZES] + ["inf"]
    results = summary.get("results", {})
    _require(set(results) == set(expected_keys), "Summary result keys mismatch")
    _require(len(per_city) == EXPECTED_CITIES * len(expected_keys), "Unexpected per-city row count")

    expected_raw_count = EXPECTED_CITIES * len(expected_keys) * EXPECTED_REPLICATES
    _require(len(raw_replicates) == expected_raw_count, f"Unexpected raw replicate row count: {len(raw_replicates)}")
    raw_by_key: dict[tuple[int, str, str], list[dict[str, Any]]] = {}
    for row in raw_replicates:
        required = {"fold", "city", "sample", "sample_trips", "replicate_id", "delta_cpc", "empirical_tv"}
        _require(required <= set(row), "Raw replicate row is missing required fields")
        _require(
            isinstance(row.get("fold"), int) and not isinstance(row.get("fold"), bool) and row["fold"] in EXPECTED_FOLDS,
            f"Unexpected raw fold: {row.get('fold')}",
        )
        _require(row["sample"] in expected_keys, f"Unexpected raw sample: {row.get('sample')}")
        _require(isinstance(row.get("city"), str) and len(row["city"]) > 0, f"Invalid raw city: {row.get('city')}")
        key = (row["fold"], row["city"], row["sample"])
        raw_by_key.setdefault(key, []).append(row)
        try:
            rep_id = int(row["replicate_id"]) if not isinstance(row["replicate_id"], bool) else -1
        except (ValueError, TypeError):
            raise AuditFailure(f"Invalid raw replicate_id: {row.get('replicate_id')}")
        _require(0 <= rep_id < EXPECTED_REPLICATES, "Replicate id outside expected range")
        if row["sample"] == "inf":
            _require(row["sample_trips"] is None, "Raw sample_trips mismatch for inf")
        else:
            try:
                expected_trips = int(row["sample"]) if not isinstance(row["sample"], bool) else -1
            except (ValueError, TypeError):
                raise AuditFailure(f"Raw sample is not a valid integer: {row.get('sample')}")
            _require(row["sample_trips"] == expected_trips, "Raw sample_trips mismatch")
        delta = _finite(row["delta_cpc"], "Raw delta_cpc")
        _require(-1.0 <= delta <= 1.0, "Raw delta_cpc outside [-1, 1]")
        tv = _finite(row["empirical_tv"], "Raw empirical_tv")
        _require(0.0 <= tv <= 1.0, "Raw empirical_tv outside [0, 1]")
    for key, values in raw_by_key.items():
        _require(len(values) == EXPECTED_REPLICATES, f"{key}: expected 1000 raw replicates")
        _require({row["replicate_id"] for row in values} == set(range(EXPECTED_REPLICATES)), f"{key}: replicate ids are incomplete or duplicated")

    rows_by_sample: dict[str, list[dict[str, Any]]] = {}
    for row in per_city:
        _require(set(row) >= {"fold", "city", "sample", "sample_trips", "delta_cpc", "empirical_tv", "win_rate", "harm_rate"}, "Per-city row is missing required fields")
        _require(
            isinstance(row.get("fold"), int) and not isinstance(row.get("fold"), bool) and row["fold"] in EXPECTED_FOLDS,
            f"Unexpected fold: {row.get('fold')}",
        )
        _require(row["sample"] in expected_keys, f"Unexpected per-city sample: {row['sample']}")
        rows_by_sample.setdefault(row["sample"], []).append(row)
        delta = _finite(row["delta_cpc"], f"{row['sample']} {row['city']} delta_cpc")
        tv = _finite(row["empirical_tv"], f"{row['sample']} {row['city']} empirical_tv")
        win = _finite(row["win_rate"], f"{row['sample']} {row['city']} win_rate")
        harm = _finite(row["harm_rate"], f"{row['sample']} {row['city']} harm_rate")
        _require(-1.0 <= delta <= 1.0, f"{row['sample']} {row['city']}: delta_cpc outside [-1, 1]")
        _require(0.0 <= tv <= 1.0, f"{row['sample']} {row['city']}: TV distance outside [0, 1]")
        _require(0.0 <= win <= 1.0, f"{row['sample']} {row['city']}: win rate outside [0, 1]")
        _require(0.0 <= harm <= 1.0, f"{row['sample']} {row['city']}: harm rate outside [0, 1]")
        _require(win + harm <= 1.0 + SUMMARY_TOLERANCE, f"{row['sample']} {row['city']}: win_rate + harm_rate exceeds 1")

    expected_raw_keys = {
        (row["fold"], row["city"], row["sample"])
        for row in per_city
    }
    _require(
        set(raw_by_key) == expected_raw_keys,
        "Raw replicate city/fold/sample keys mismatch per_city",
    )

    _require(set(rows_by_sample) == set(expected_keys), "Per-city sample keys mismatch")
    reference_pairs = {(row["fold"], row["city"]) for row in rows_by_sample["inf"]}
    for sample, rows in rows_by_sample.items():
        _require(len(rows) == EXPECTED_CITIES, f"{sample}: expected 50 city rows, got {len(rows)}")
        current_pairs = {(row["fold"], row["city"]) for row in rows}
        _require(current_pairs == reference_pairs, f"{sample}: city/fold identities differ from clean condition")
        _require(len({row["city"] for row in rows}) == EXPECTED_CITIES, f"{sample}: duplicate or missing cities")
        _require(Counter(row["fold"] for row in rows) == Counter({fold: 10 for fold in EXPECTED_FOLDS}), f"{sample}: fold coverage is not 10 cities per fold")
        for row in rows:
            expected_trips = None if sample == "inf" else int(sample)
            _require(row["sample_trips"] == expected_trips, f"{sample} {row['city']}: sample_trips mismatch")
            raw_values = raw_by_key[(row["fold"], row["city"], sample)]
            raw_delta = [float(value["delta_cpc"]) for value in raw_values]
            raw_tv = [float(value["empirical_tv"]) for value in raw_values]
            if sample == "inf":
                _require(
                    max(raw_delta) - min(raw_delta) <= SUMMARY_TOLERANCE,
                    f"{sample} {row['city']}: clean delta_cpc varies across replicates",
                )
            raw_win = sum(value > 0.0 for value in raw_delta) / EXPECTED_REPLICATES
            raw_harm = sum(value < 0.0 for value in raw_delta) / EXPECTED_REPLICATES
            _require(_close(float(row["delta_cpc"]), sum(raw_delta) / EXPECTED_REPLICATES), f"{sample} {row['city']}: delta_cpc inconsistent with raw replicates")
            _require(_close(float(row["empirical_tv"]), sum(raw_tv) / EXPECTED_REPLICATES), f"{sample} {row['city']}: empirical_tv inconsistent with raw replicates")
            _require(_close(float(row["win_rate"]), raw_win), f"{sample} {row['city']}: win_rate inconsistent with raw replicates")
            _require(_close(float(row["harm_rate"]), raw_harm), f"{sample} {row['city']}: harm_rate inconsistent with raw replicates")

    mean_tv = []
    recomputed = {}
    for size in EXPECTED_SAMPLE_SIZES:
        key = str(size)
        rows = rows_by_sample[key]
        recomputed_mean_tv = sum(float(row["empirical_tv"]) for row in rows) / len(rows)
        mean_tv.append(recomputed_mean_tv)
        recomputed[key] = {
            "mean_delta_cpc": sum(float(row["delta_cpc"]) for row in rows) / len(rows),
            "mean_empirical_tv": recomputed_mean_tv,
            "win_rate": sum(float(row["delta_cpc"]) > 0.0 for row in rows) / len(rows),
            "harm_rate": sum(float(row["delta_cpc"]) < 0.0 for row in rows) / len(rows),
        }
    clean_rows = rows_by_sample["inf"]
    for row in clean_rows:
        _require(abs(float(row["empirical_tv"])) <= SUMMARY_TOLERANCE, f"{row['city']}: clean empirical_tv must be zero")
    recomputed["inf"] = {
        "mean_delta_cpc": sum(float(row["delta_cpc"]) for row in clean_rows) / len(clean_rows),
        "mean_empirical_tv": sum(float(row["empirical_tv"]) for row in clean_rows) / len(clean_rows),
        "win_rate": sum(float(row["delta_cpc"]) > 0.0 for row in clean_rows) / len(clean_rows),
        "harm_rate": sum(float(row["delta_cpc"]) < 0.0 for row in clean_rows) / len(clean_rows),
    }
    tv_nonincreasing = all(left >= right - 1e-12 for left, right in zip(mean_tv, mean_tv[1:]))

    clean = results["inf"]
    clean_gain = recomputed["inf"]["mean_delta_cpc"]
    _require(
        clean_gain > SUMMARY_TOLERANCE,
        "Clean Y_D does not provide a positive gain; relative-to-clean thresholds are not interpretable",
    )
    _require(_close(_finite(clean["mean_delta_cpc"], "clean gain"), clean_gain), "inf: mean_delta_cpc inconsistent with per_city")
    _require(_close(_finite(clean["mean_empirical_tv"], "clean mean TV"), recomputed["inf"]["mean_empirical_tv"]), "inf: mean_empirical_tv inconsistent with per_city")
    _require(clean["mean_empirical_tv"] == 0.0, "Clean Y_D must have zero empirical TV")
    _require(abs(_finite(clean["relative_to_clean_pct"], "clean relative effect") - 100.0) < 1e-9, "Clean relative effect must be 100%")

    for key in expected_keys:
        result = results[key]
        for field in ["mean_delta_cpc", "mean_empirical_tv", "win_rate", "harm_rate"]:
            reported = _finite(result[field], f"{key}: {field}")
            _require(_close(reported, recomputed[key][field]), f"{key}: {field} inconsistent with per_city")
        _require(-1.0 <= recomputed[key]["mean_delta_cpc"] <= 1.0, f"{key}: delta_cpc outside [-1, 1]")
        _require(recomputed[key]["win_rate"] + recomputed[key]["harm_rate"] <= 1.0 + SUMMARY_TOLERANCE, f"{key}: win/harm rates exceed 1")
        ci = result.get("ci95_delta_cpc")
        _require(isinstance(ci, list) and len(ci) == 2, f"{key}: malformed 95% CI")
        lower, upper = (_finite(ci[0], f"{key} CI lower"), _finite(ci[1], f"{key} CI upper"))
        _require(lower <= upper, f"{key}: CI lower exceeds upper")
        expected_ci = _fold_bootstrap(per_city, key, "delta_cpc")
        _require(abs(lower - expected_ci[0]) <= CI_TOLERANCE and abs(upper - expected_ci[1]) <= CI_TOLERANCE, f"{key}: CI inconsistent with per_city bootstrap")
        _require(result["n_cities"] == EXPECTED_CITIES, f"{key}: n_cities mismatch")
        _require(abs(_finite(result["relative_to_clean_pct"], f"{key} relative effect") - 100.0 * float(result["mean_delta_cpc"]) / clean_gain) < 1e-8, f"{key}: relative effect mismatch")

    thresholds = summary.get("thresholds", {})
    useful = next((size for size in EXPECTED_SAMPLE_SIZES if results[str(size)]["ci95_delta_cpc"][0] > 0.0), None)
    _require(thresholds.get("minimum_useful_sample_trips") == useful, f"Unexpected minimum useful sample threshold: {thresholds.get('minimum_useful_sample_trips')}; derived {useful}")
    _require(abs(_finite(thresholds["clean_gain"], "threshold clean gain") - clean_gain) < 1e-12, "Threshold clean gain mismatch")
    for fraction in [0.5, 0.8, 0.9, 0.95]:
        key = f"minimum_sample_trips_for_{int(fraction * 100)}pct_clean"
        expected = next((size for size in EXPECTED_SAMPLE_SIZES if results[str(size)]["mean_delta_cpc"] / clean_gain >= fraction), None)
        _require(thresholds.get(key) == expected, f"Unexpected threshold {key}: {thresholds.get(key)}; derived {expected}")

    return {
        "status": "PASS",
        "gates": {
            "protocol": "PASS",
            "coverage": "PASS",
            "numeric_integrity": "PASS",
            "threshold_consistency": "PASS",
            "scientific_recomputation": "PASS",
        },
        "limitations": {
            "sampling_provenance": (
                "Nested-multinomial generation cannot be independently verified "
                "from metric-only raw replicate records."
            ),
        },
        "n_cities": EXPECTED_CITIES,
        "sample_sizes": EXPECTED_SAMPLE_SIZES,
        "clean_gain": clean_gain,
        "minimum_useful_sample_trips": useful,
        "tv_monotonicity_observed": tv_nonincreasing,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit finite-sample Y_D result artifacts")
    parser.add_argument("--results-dir", type=Path, default=Path("results/finite_sample_yd_robustness_v1"))
    args = parser.parse_args(argv)
    try:
        report = audit_results(args.results_dir)
    except AuditFailure as exc:
        print(f"AUDIT FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    print("FINITE-SAMPLE Y_D AUDIT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())