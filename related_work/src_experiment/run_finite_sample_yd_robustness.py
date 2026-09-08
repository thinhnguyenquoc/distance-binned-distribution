"""Finite-sample robustness of target-city distance-marginal observation.

The target Y_D is treated as the population distribution. For each city and
replicate, one nested multinomial trajectory is drawn at N_max trips; prefixes
of that trajectory provide every finite sample size without retraining models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr, wilcoxon

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.calibration.bin_calibration import calibrate_kbins
from src.data.city_splits import load_splits_manifest_v2
from src.data.dataset import load_city
from src.data.urban_graph import build_radius_graph
from src.data.yd_extractor import compute_kbin_edges, extract_yd_kbins
from src.training.evaluate import compute_cpc_pair
from src.training.train import infer_zero_shot, load_checkpoint


SAMPLE_SIZES = [50, 100, 250, 500, 1000, 2500, 5000]
MODEL_SEEDS = [1, 10, 100]
N_MAX = 10000
K = 8
DEFAULT_OUTPUT = Path("results/finite_sample_yd_robustness_v1")
BASE_SEED = 20260826


def _stable_seed(fold: int, city: str, replicate: int, seed: int) -> int:
    value = f"{BASE_SEED}:{fold}:{city}:{replicate}:{seed}"
    return int(hashlib.sha256(value.encode()).hexdigest()[:8], 16)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _calibrated_cpc(
    prediction: np.ndarray,
    truth_inter: np.ndarray,
    bin_idx: np.ndarray,
    yd_sample: np.ndarray,
    yd_target: np.ndarray,
    prediction_yd: np.ndarray,
) -> float:
    """Apply q=1 calibration in vectorized form and return interzonal CPC."""
    pred_mass = float(prediction.sum())
    if pred_mass <= 0.0:
        return float(compute_cpc_pair(truth_inter, prediction))

    target = np.asarray(yd_sample, dtype=np.float64)
    target_sum = float(target.sum())
    if target_sum <= 0.0:
        target = np.asarray(yd_target, dtype=np.float64)
    else:
        target = target / target_sum

    active = prediction_yd > 0.0
    target_active = target * active
    active_sum = float(target_active.sum())
    if active_sum <= 0.0:
        target_active = prediction_yd.copy()
        active_sum = float(target_active.sum())
    target_active /= active_sum

    weights = np.ones(K, dtype=np.float64)
    weights[active] = target_active[active] / prediction_yd[active]
    weighted_mass = float(np.dot(prediction_yd, weights))
    if weighted_mass <= 0.0:
        return float(compute_cpc_pair(truth_inter, prediction))

    scales = weights / weighted_mass
    calibrated = prediction * scales[bin_idx]
    denominator = float(truth_inter.sum() + calibrated.sum())
    if denominator <= 0.0:
        return 0.0
    return float(2.0 * np.minimum(truth_inter, calibrated).sum() / denominator)


def _fold_bootstrap(rows: list[dict[str, Any]], metric: str, sample_key: str, n_boot: int = 10000) -> tuple[float, float]:
    rng = np.random.default_rng(42)
    all_by_fold = {
        fold: np.array([row[metric] for row in rows if row["fold"] == fold and row["sample"] == sample_key])
        for fold in range(1, 6)
    }
    by_fold = {fold: values for fold, values in all_by_fold.items() if len(values) > 0}
    if not by_fold:
        return float("nan"), float("nan")
    fold_means = []
    for values in by_fold.values():
        sampled = values[rng.integers(0, len(values), size=(n_boot, len(values)))]
        fold_means.append(sampled.mean(axis=1))
    samples = np.column_stack(fold_means)
    means = samples.mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def run_experiment(
    data_root: str = "data",
    output_dir: Path = DEFAULT_OUTPUT,
    replicates: int = 1000,
    smoke: bool = False,
) -> dict[str, Any]:
    sample_sizes = [50, 100, 250] if smoke else SAMPLE_SIZES
    folds = [1] if smoke else [1, 2, 3, 4, 5]
    city_limit = 1 if smoke else None
    replicate_count = 10 if smoke else replicates
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = load_splits_manifest_v2("results/e1/splits_manifest_v2.json", data_root=data_root)
    city_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    model_cache: dict[tuple[int, int], tuple[Any, Any]] = {}

    for fold in folds:
        split = splits[fold]
        bin_edges, k_active = compute_kbin_edges(split["train"], K=K, data_root=data_root)
        if k_active != K:
            raise RuntimeError(f"Expected K={K}, got {k_active} in fold {fold}")
        for city in sorted(split["test"])[:city_limit]:
            raw = load_city(city, data_root=data_root, feature_scaler=None, fit_scaler=True)
            distances = np.expm1(raw.pair_distance.numpy())
            origins = raw.pair_o_idx.numpy()
            destinations = raw.pair_d_idx.numpy()
            inter = (origins != destinations) & (distances > 0.0)
            truth = raw.pair_trips.numpy().astype(np.float64)
            truth_inter = truth[inter]
            yd_target = extract_yd_kbins(distances, truth, bin_edges, inter)
            inter_distances = distances[inter]
            bin_idx = np.clip(np.digitize(inter_distances, bin_edges[1:-1], right=True), 0, K - 1)
            bin_counts = np.bincount(bin_idx, weights=truth_inter, minlength=K).astype(np.int64)

            endpoints = np.asarray(sample_sizes, dtype=np.int64)
            prefix_counts = np.zeros((replicate_count, len(sample_sizes), K), dtype=np.int64)
            for replicate in range(replicate_count):
                rng = np.random.default_rng(_stable_seed(fold, city, replicate, 0))
                trajectory = rng.choice(K, size=N_MAX, p=yd_target)
                for bin_id in range(K):
                    prefix_counts[replicate, :, bin_id] = np.cumsum(trajectory == bin_id)[endpoints - 1]

            edge_index, edge_dist = build_radius_graph(raw.lon_lat.numpy(), radius_km=5.0, include_self_loop=True, cache_key=f"finite_{city}")
            seed_results: list[np.ndarray] = []
            clean_results: list[float] = []
            for model_seed in MODEL_SEEDS:
                checkpoint = Path("results/checkpoints") / f"5fold_fold{fold}_seed{model_seed}.pt"
                model, scaler, metadata = load_checkpoint(checkpoint, device_str="cpu")
                if metadata.get("seed") != model_seed or metadata.get("hyperparams", {}).get("fold") != fold:
                    raise RuntimeError(f"Checkpoint provenance mismatch: {checkpoint}")
                city_data = load_city(city, data_root=data_root, feature_scaler=scaler, fit_scaler=False)
                prediction = infer_zero_shot(model, city_data, edge_index, edge_dist, device="cpu").numpy().astype(np.float64)[inter]
                prediction_yd = np.bincount(bin_idx, weights=prediction, minlength=K)
                prediction_yd /= prediction.sum()
                baseline = float(compute_cpc_pair(truth_inter, prediction))
                clean = _calibrated_cpc(prediction, truth_inter, bin_idx, yd_target, yd_target, prediction_yd)
                clean_results.append(clean - baseline)
                values = np.empty((len(sample_sizes), replicate_count), dtype=np.float64)
                for size_index, sample_size in enumerate(sample_sizes):
                    for replicate in range(replicate_count):
                        counts = prefix_counts[replicate, size_index]
                        yd_sample = counts / float(sample_size)
                        values[size_index, replicate] = _calibrated_cpc(prediction, truth_inter, bin_idx, yd_sample, yd_target, prediction_yd) - baseline
                seed_results.append(values)

            mean_by_replicate = np.mean(seed_results, axis=0)
            for size_index, sample_size in enumerate(sample_sizes):
                deltas = mean_by_replicate[size_index]
                for replicate, delta in enumerate(deltas):
                    raw_rows.append({
                        "fold": fold,
                        "city": city,
                        "sample": str(sample_size),
                        "sample_trips": sample_size,
                        "replicate_id": replicate,
                        "delta_cpc": float(delta),
                        "empirical_tv": float(0.5 * np.abs((prefix_counts[replicate, size_index] / sample_size) - yd_target).sum()),
                    })
                city_rows.append({
                    "fold": fold,
                    "city": city,
                    "sample": str(sample_size),
                    "sample_trips": sample_size,
                    "delta_cpc": float(deltas.mean()),
                    "empirical_tv": float(np.mean([0.5 * np.abs((prefix_counts[r, size_index] / sample_size) - yd_target).sum() for r in range(replicate_count)])),
                    "win_rate": float(np.mean(deltas > 0.0)),
                    "harm_rate": float(np.mean(deltas < 0.0)),
                })
            city_rows.append({
                "fold": fold,
                "city": city,
                "sample": "inf",
                "sample_trips": None,
                "delta_cpc": float(np.mean(clean_results)),
                "empirical_tv": 0.0,
                "win_rate": float(np.mean(clean_results) > 0.0),
                "harm_rate": float(np.mean(clean_results) < 0.0),
            })
            for replicate in range(replicate_count):
                raw_rows.append({
                    "fold": fold,
                    "city": city,
                    "sample": "inf",
                    "sample_trips": None,
                    "replicate_id": replicate,
                    "delta_cpc": float(np.mean(clean_results)),
                    "empirical_tv": 0.0,
                })

    summary: dict[str, Any] = {"protocol": {"name": "Finite-Sample Y_D Observation Robustness v1", "K": K, "bins": "quantile", "sample_sizes": sample_sizes, "replicates_per_city": replicate_count, "raw_replicate_artifact": True, "nested_multinomial": True, "model_seeds": MODEL_SEEDS, "no_retraining": True, "statistical_unit": "city"}, "results": {}}
    finite_keys = [str(size) for size in sample_sizes]
    keys = finite_keys + ["inf"]
    for key in keys:
        rows = [row for row in city_rows if row["sample"] == key]
        deltas = np.array([row["delta_cpc"] for row in rows])
        tv = np.array([row["empirical_tv"] for row in rows])
        try:
            p_value = float(wilcoxon(deltas, alternative="greater").pvalue)
        except ValueError:
            p_value = 1.0
        summary["results"][key] = {"sample_trips": None if key == "inf" else int(key), "mean_delta_cpc": float(deltas.mean()), "median_delta_cpc": float(np.median(deltas)), "ci95_delta_cpc": list(_fold_bootstrap(city_rows, "delta_cpc", key, n_boot=10000)), "mean_empirical_tv": float(tv.mean()), "win_rate": float(np.mean(deltas > 0.0)), "harm_rate": float(np.mean(deltas < 0.0)), "wilcoxon_p_raw": p_value, "n_cities": len(rows)}
    adjusted = _holm([summary["results"][key]["wilcoxon_p_raw"] for key in finite_keys])
    for key, p_value in zip(finite_keys, adjusted):
        summary["results"][key]["wilcoxon_p_holm"] = p_value
    clean_gain = summary["results"]["inf"]["mean_delta_cpc"]
    for key in keys:
        summary["results"][key]["relative_to_clean_pct"] = float(100.0 * summary["results"][key]["mean_delta_cpc"] / clean_gain) if clean_gain > 0 else None
    useful = next((int(key) for key in finite_keys if summary["results"][key]["ci95_delta_cpc"][0] > 0.0), None)
    thresholds = {"minimum_useful_sample_trips": useful, "clean_gain": clean_gain}
    for fraction in [0.5, 0.8, 0.9, 0.95]:
        thresholds[f"minimum_sample_trips_for_{int(fraction * 100)}pct_clean"] = next((int(key) for key in finite_keys if summary["results"][key]["mean_delta_cpc"] / clean_gain >= fraction), None) if clean_gain > 0 else None
    summary["thresholds"] = thresholds
    _atomic_json(output_dir / "summary.json", summary)
    (output_dir / "per_city.json").write_text(json.dumps(city_rows, indent=2), encoding="utf-8")
    with (output_dir / "raw_replicates.jsonl").open("w", encoding="utf-8") as raw_file:
        for row in raw_rows:
            raw_file.write(json.dumps(row) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run finite-sample Y_D observation robustness")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    result = run_experiment(args.data_root, args.output_dir, args.replicates, args.smoke)
    print(json.dumps(result["thresholds"], indent=2))