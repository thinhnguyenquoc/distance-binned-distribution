"""Focused executor for ``new_plan_v4_budgeted_validation``.

This runner is deliberately separate from the historical v1 runner.  It owns a
protocol fingerprint, source-local checkpoints, raw-prediction caches, and
atomic per-source result bundles so a restart can reuse only artifacts that
match the frozen validation-selected design.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.optim as optim


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from implement_new_plan.calibration.dose_matching import (  # noqa: E402
    choose_donor,
    find_dose_matching_lambda,
    generate_donor_mapping,
)
from implement_new_plan.calibration.source_bins import (  # noqa: E402
    assign_to_source_bins,
    build_source_bin_edges,
    build_target_dbd,
    calibrate_dbd,
    compute_pure_calibration_ratios,
    compute_source_distance_cap,
)
from implement_new_plan.calibration.tv_noise import (  # noqa: E402
    derive_noise_seed,
    generate_exact_tv_noise,
)
from implement_new_plan.data.dataset import RawCityData, load_raw_city  # noqa: E402
from implement_new_plan.data.source_scaler import SourceCityFeatureScaler  # noqa: E402
from implement_new_plan.data.urban_graph import build_radius_graph  # noqa: E402
from implement_new_plan.experiment.master_protocol_runner import set_seed  # noqa: E402
from implement_new_plan.loss.log1p_mse import log1p_mse_loss  # noqa: E402
from implement_new_plan.models.od_models import (  # noqa: E402
    PairwiseMLP,
    TwoParameterGravity,
    UrbanGNN,
)
from implement_new_plan.training.evaluate import (  # noqa: E402
    compute_cpc_norm_pair,
    compute_cpc_pair,
    compute_mae_pair,
    compute_mse_pair,
    compute_rmse_pair,
    evaluate_calibration_transfer,
)


PROTOCOL_ID = "new_plan_v4_budgeted_validation"
PROTOCOL_VERSION = 1
SPLIT_SEED = 42
NOISE_SEED = 42
FRACTIONS = (0.10, 0.20, 0.30, 0.40)
MAIN_FRACTION = 0.40
VALIDATION_FRACTION = 0.02
SOURCE_TEST_FRACTION = 0.60
NEURAL_SEEDS = (1, 10, 100)
GRAVITY_SEED = 1
MAIN_K = 8
RESOLUTION_K = (4, 8, 12)
NOISE_EPSILONS = (0.00, 0.05, 0.10)
NOISE_REALIZATIONS = 5

TRAINING_CONFIG = {
    "max_epochs": 400,
    "early_stopping_patience": 20,
    "selection_metric": "source_validation_log1p_mse",
    "validation_frequency_epochs": 1,
    "prediction_chunk_size": 65536,
    "optimizer": "AdamW",
    "learning_rate": 2e-3,
    "weight_decay": 1e-4,
    "gradient_clip_max_norm": 5.0,
    "batch_mode": "full_batch_positive_interzonal_support",
    "loss": "log1p_mse",
    "graph_type": "radius",
    "graph_radius_km": 5.0,
    "graph_self_loops": True,
    "device": "cpu",
    "amp": False,
}

CODE_HASH_PATHS = (
    "implement_new_plan/experiment/run_new_plan_v4_budgeted_validation.py",
    "implement_new_plan/experiment/master_protocol_runner.py",
    "implement_new_plan/data/dataset.py",
    "implement_new_plan/data/source_scaler.py",
    "implement_new_plan/data/urban_graph.py",
    "implement_new_plan/models/od_models.py",
    "implement_new_plan/models/node_encoder.py",
    "implement_new_plan/models/gravity.py",
    "implement_new_plan/models/decoder.py",
    "implement_new_plan/loss/log1p_mse.py",
    "implement_new_plan/calibration/source_bins.py",
    "implement_new_plan/calibration/tv_noise.py",
    "implement_new_plan/calibration/dose_matching.py",
    "implement_new_plan/training/evaluate.py",
)

TRANSFER_METRICS = (
    "CPC_before",
    "CPC_after",
    "delta_CPC",
    "CPC_norm_before",
    "CPC_norm_after",
    "delta_CPC_norm",
    "MAE_before",
    "MAE_after",
    "delta_MAE",
    "MSE_before",
    "MSE_after",
    "delta_MSE",
    "RMSE_before",
    "RMSE_after",
    "delta_RMSE",
    "R_vol_before",
    "R_vol_after",
)

PER_SOURCE_FILES = (
    "source_validation_metrics.csv",
    "source_test_metrics.csv",
    "main_transfer_metrics.csv",
    "scarcity_transfer_metrics.csv",
    "resolution_metrics.csv",
    "noise_metrics.csv",
    "specificity_metrics.csv",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(payload: Any) -> str:
    return _sha256_bytes(_canonical_json(payload).encode("utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dataset_content_hash(data_root: Path) -> str:
    """Hash every source CSV once so resume cannot mix a changed dataset."""
    digest = hashlib.sha256()
    files = sorted(path for path in data_root.rglob("*.csv") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No CSV data files found below {data_root}")
    for path in files:
        digest.update(path.relative_to(data_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as input_file:
            for block in iter(lambda: input_file.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _code_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for rel in CODE_HASH_PATHS:
        path = REPO_ROOT / rel
        if not path.is_file():
            raise FileNotFoundError(f"Required protocol source file is missing: {path}")
        hashes[rel] = _sha256_file(path)
    return hashes


def _scientific_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the part of provenance that changes numerical estimands/results.

    The driver itself is intentionally excluded so an audited implementation-only
    migration (for example, reducing peak RAM while preserving all arithmetic)
    can reuse already-validated checkpoints and raw predictions.  Model,
    calibration, scaler, evaluator, data, and plan hashes remain included.
    """
    core_hashes = dict(payload["code_sha256"])
    core_hashes.pop("implement_new_plan/experiment/run_new_plan_v4_budgeted_validation.py", None)
    return {
        "config": payload["config"],
        "new_plan_sha256": payload["new_plan_sha256"],
        "canonical_cities_sha256": payload["canonical_cities_sha256"],
        "data_sha256": payload["data_sha256"],
        "core_code_sha256": core_hashes,
    }


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, bundle: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(dict(bundle), temporary)
    os.replace(temporary, path)


def _atomic_save_prediction(path: Path, prediction: np.ndarray, metadata: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as temporary:
        temporary_path = Path(temporary.name)
        np.savez_compressed(
            temporary,
            prediction=np.asarray(prediction, dtype=np.float32),
            metadata=np.asarray(_canonical_json(dict(metadata))),
        )
    os.replace(temporary_path, path)


def _safe_float(value: Any) -> float:
    value_float = float(value)
    if not np.isfinite(value_float):
        raise FloatingPointError(f"Encountered non-finite scalar result: {value_float}")
    return value_float


def _fraction_tag(fraction: float) -> str:
    return f"f{int(round(fraction * 100)):02d}"


def _model_seed_plan() -> tuple[tuple[str, float, int, bool], ...]:
    jobs: list[tuple[str, float, int, bool]] = []
    for fraction in FRACTIONS:
        jobs.append(("gravity_2param", fraction, GRAVITY_SEED, False))
        for seed in NEURAL_SEEDS:
            jobs.append(("urban_gnn", fraction, seed, True))
    for seed in NEURAL_SEEDS:
        jobs.append(("pairwise_mlp", MAIN_FRACTION, seed, True))
    return tuple(jobs)


def expected_source_row_counts(n_targets: int) -> dict[str, int]:
    return {
        "source_validation_metrics.csv": len(_model_seed_plan()),
        "source_test_metrics.csv": len(_model_seed_plan()),
        "main_transfer_metrics.csv": n_targets * 7,
        "scarcity_transfer_metrics.csv": n_targets * 16,
        "resolution_metrics.csv": n_targets * len(NEURAL_SEEDS) * len(RESOLUTION_K),
        "noise_metrics.csv": n_targets * len(NEURAL_SEEDS) * (1 + 2 * NOISE_REALIZATIONS),
        "specificity_metrics.csv": n_targets * len(NEURAL_SEEDS),
    }


def build_v4_city_split(raw_city: RawCityData, split_seed: int = SPLIT_SEED) -> dict[str, Any]:
    """Nested observed-label budgets with fixed 2% validation and 60% test."""
    origin = np.asarray(raw_city.pair_o_idx, dtype=np.int64)
    destination = np.asarray(raw_city.pair_d_idx, dtype=np.int64)
    distance = np.asarray(raw_city.dist_km, dtype=np.float64)
    flow = np.asarray(raw_city.pair_trips, dtype=np.float64)
    support_mask = (origin != destination) & (distance > 0.0) & (flow >= 1.0)
    raw_indices = np.flatnonzero(support_mask)
    if raw_indices.size == 0:
        raise ValueError(f"{raw_city.city_name}: positive interzonal support is empty")

    pairs = pd.DataFrame(
        {
            "source_city": raw_city.city_name,
            "raw_pair_index": raw_indices,
            "origin": origin[raw_indices],
            "destination": destination[raw_indices],
            "distance_km": distance[raw_indices],
        }
    ).sort_values(["origin", "destination"], kind="mergesort").reset_index(drop=True)
    if pairs.duplicated(["origin", "destination"]).any():
        duplicate = pairs.loc[pairs.duplicated(["origin", "destination"], keep=False), ["origin", "destination"]].head(3)
        raise ValueError(f"{raw_city.city_name}: duplicate positive OD keys: {duplicate.to_dict('records')}")

    n_support = len(pairs)
    counts = {fraction: int(np.floor(fraction * n_support)) for fraction in FRACTIONS}
    validation_count = int(np.floor(VALIDATION_FRACTION * n_support))
    if validation_count < 1 or counts[0.10] <= validation_count or counts[0.40] >= n_support:
        raise ValueError(
            f"{raw_city.city_name}: invalid v4 split sizes for N={n_support}; "
            f"Val={validation_count}, Budget_10={counts[0.10]}, Budget_40={counts[0.40]}"
        )

    rng = np.random.default_rng(split_seed)
    permutation = rng.permutation(n_support)
    permutation_rank = np.empty(n_support, dtype=np.int64)
    permutation_rank[permutation] = np.arange(n_support, dtype=np.int64)
    pairs["permutation_rank"] = permutation_rank
    pairs["in_source_validation"] = permutation_rank < validation_count
    for fraction, count in counts.items():
        tag = int(round(fraction * 100))
        pairs[f"in_budget_{tag:02d}"] = permutation_rank < count
        pairs[f"in_train_{tag:02d}"] = (permutation_rank >= validation_count) & (permutation_rank < count)
    pairs["in_source_test"] = permutation_rank >= counts[0.40]
    pairs["split_seed"] = int(split_seed)

    nested_masks = [pairs[f"in_train_{int(round(fraction * 100)):02d}"].to_numpy(dtype=bool) for fraction in FRACTIONS]
    if not all(np.all(previous <= following) for previous, following in zip(nested_masks, nested_masks[1:])):
        raise AssertionError(f"{raw_city.city_name}: nested source split invariant failed")
    if any(np.any(mask & pairs["in_source_validation"].to_numpy(dtype=bool)) for mask in nested_masks):
        raise AssertionError(f"{raw_city.city_name}: source validation overlaps a training subset")
    if any(np.any(mask & pairs["in_source_test"].to_numpy(dtype=bool)) for mask in nested_masks):
        raise AssertionError(f"{raw_city.city_name}: source test overlaps a training subset")
    if int(pairs["in_source_validation"].sum()) != validation_count:
        raise AssertionError(f"{raw_city.city_name}: fixed source validation count is incorrect")
    if int(pairs["in_source_test"].sum()) != n_support - counts[0.40]:
        raise AssertionError(f"{raw_city.city_name}: fixed source test count is incorrect")

    sorted_indices = pairs["raw_pair_index"].to_numpy(dtype=np.int64)
    train_indices = {
        fraction: sorted_indices[permutation[validation_count:count]] for fraction, count in counts.items()
    }
    source_validation_indices = sorted_indices[permutation[:validation_count]]
    source_test_indices = sorted_indices[permutation[counts[0.40] :]]
    support_hash = _sha256_json(
        {
            "city": raw_city.city_name,
            "origin": pairs["origin"].astype(int).tolist(),
            "destination": pairs["destination"].astype(int).tolist(),
            "distance_km": pairs["distance_km"].astype(float).tolist(),
            "permutation_rank": pairs["permutation_rank"].astype(int).tolist(),
        }
    )
    return {
        "support_indices": sorted_indices,
        "train_indices": train_indices,
        "source_validation_indices": source_validation_indices,
        "source_test_indices": source_test_indices,
        "validation_count": validation_count,
        "counts": counts,
        "support_hash": support_hash,
        "manifest": pairs,
    }


@dataclass(frozen=True)
class PreparedCity:
    raw: RawCityData
    split: Mapping[str, Any]
    edge_index: torch.Tensor
    edge_distance: torch.Tensor


@dataclass(frozen=True)
class SourcePreprocessing:
    scaler: SourceCityFeatureScaler
    scaler_hash: str
    scaler_records: tuple[dict[str, Any], ...]
    bins: Mapping[int, np.ndarray]
    bins_hash: str
    bin_records: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ModelInputs:
    origin: np.ndarray
    destination: np.ndarray
    distance: np.ndarray
    true_flow: np.ndarray
    origin_tensor: torch.Tensor
    destination_tensor: torch.Tensor
    distance_tensor: torch.Tensor
    population_tensor: torch.Tensor
    node_features: torch.Tensor
    distance_std: torch.Tensor


def _prepare_cities(cities: Sequence[str]) -> dict[str, PreparedCity]:
    prepared: dict[str, PreparedCity] = {}
    for city in cities:
        raw = load_raw_city(city, data_root="data")
        split = build_v4_city_split(raw)
        # The tabular manifest is streamed later.  Retaining all 50 full
        # DataFrames here can consume several GB before model work begins.
        split.pop("manifest")
        edge_index, edge_distance = build_radius_graph(
            np.asarray(raw.lon_lat),
            radius_km=float(TRAINING_CONFIG["graph_radius_km"]),
            include_self_loop=True,
            # PreparedCity owns the sparse graph.  Do not retain N x N
            # Haversine matrices in the global graph cache for every city.
            use_cache=False,
        )
        prepared[city] = PreparedCity(raw=raw, split=split, edge_index=edge_index, edge_distance=edge_distance)
    return prepared


def _build_source_preprocessing(city: str, prepared: PreparedCity) -> SourcePreprocessing:
    scaler = SourceCityFeatureScaler(city).fit(
        prepared.raw,
        prepared.split["train_indices"][MAIN_FRACTION],
        distance_fit_scope="source_train_od_40pct_reference",
    )
    scaler_records = tuple(scaler.to_manifest_records())
    scaler_hash = _sha256_json(scaler_records)
    d_cap = compute_source_distance_cap(
        np.asarray(prepared.raw.dist_km)[prepared.split["train_indices"][MAIN_FRACTION]]
    )
    bins = {K: build_source_bin_edges(d_cap, K) for K in RESOLUTION_K}
    bin_records: list[dict[str, Any]] = []
    for K, edges in bins.items():
        for bin_id in range(K):
            upper = edges[bin_id + 1]
            bin_records.append(
                {
                    "source_city": city,
                    "reference_train_fraction": MAIN_FRACTION,
                    "split_seed": SPLIT_SEED,
                    "support_hash": prepared.split["support_hash"],
                    "K": K,
                    "D_cap_p99": float(d_cap),
                    "bin_id": bin_id + 1,
                    "lower_km": float(edges[bin_id]),
                    "upper_km": "inf" if np.isinf(upper) else float(upper),
                }
            )
    bins_hash = _sha256_json({str(K): edges.tolist() for K, edges in bins.items()})
    return SourcePreprocessing(
        scaler=scaler,
        scaler_hash=scaler_hash,
        scaler_records=scaler_records,
        bins=bins,
        bins_hash=bins_hash,
        bin_records=tuple(bin_records),
    )


def _make_inputs(prepared: PreparedCity, scaler: SourceCityFeatureScaler, indices: np.ndarray) -> ModelInputs:
    raw = prepared.raw
    pair_origin = np.asarray(raw.pair_o_idx, dtype=np.int64)[indices]
    pair_destination = np.asarray(raw.pair_d_idx, dtype=np.int64)[indices]
    pair_distance = np.asarray(raw.dist_km, dtype=np.float64)[indices]
    pair_flow = np.asarray(raw.pair_trips, dtype=np.float64)[indices]
    if not (
        np.all(pair_origin != pair_destination)
        and np.all(pair_distance > 0.0)
        and np.all(pair_flow >= 1.0)
    ):
        raise AssertionError(f"{raw.city_name}: model input leaves fixed positive interzonal support")
    population = scaler.impute_raw_population(np.asarray(raw.population))
    node_features = scaler.transform_node_features(np.asarray(raw.X_raw))
    distance_std = scaler.transform_distances(pair_distance)
    return ModelInputs(
        origin=pair_origin,
        destination=pair_destination,
        distance=pair_distance,
        true_flow=pair_flow,
        origin_tensor=torch.from_numpy(pair_origin).long(),
        destination_tensor=torch.from_numpy(pair_destination).long(),
        distance_tensor=torch.from_numpy(pair_distance.astype(np.float32, copy=False)),
        population_tensor=torch.from_numpy(population.astype(np.float32, copy=False)),
        node_features=torch.from_numpy(node_features.astype(np.float32, copy=False)),
        distance_std=torch.from_numpy(distance_std.astype(np.float32, copy=False)).unsqueeze(-1),
    )


def _new_model(model_name: str, node_dim: int) -> torch.nn.Module:
    if model_name == "gravity_2param":
        return TwoParameterGravity(init_G=0.0, init_alpha=1.0)
    if model_name == "pairwise_mlp":
        return PairwiseMLP(node_in_dim=node_dim, hidden_dim=64, dropout=0.1)
    if model_name == "urban_gnn":
        return UrbanGNN(
            node_in_dim=node_dim,
            node_hidden_dim=64,
            node_out_dim=64,
            decoder_hidden_dim=64,
            dropout=0.1,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def _checkpoint_path(output_root: Path, source_city: str, model: str, fraction: float, seed: int) -> Path:
    return output_root / "checkpoints" / "models" / source_city / f"{model}_{_fraction_tag(fraction)}_seed{seed}.pt"


def _prediction_cache_path(
    output_root: Path,
    source_city: str,
    target_city: str,
    model: str,
    fraction: float,
    seed: int,
) -> Path:
    name = f"{source_city}__{target_city}__{model}_{_fraction_tag(fraction)}_seed{seed}.npz"
    return output_root / "cache" / "raw_predictions" / source_city / name


def _checkpoint_metadata(
    protocol_hash: str,
    source_city: str,
    model: str,
    fraction: float,
    seed: int,
    prepared_source: PreparedCity,
    preprocessing: SourcePreprocessing,
) -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "protocol_hash": protocol_hash,
        "source_city": source_city,
        "model": model,
        "train_fraction": fraction,
        "train_fraction_semantics": "total_source_label_budget_including_validation",
        "n_gradient_train_pairs": int(len(prepared_source.split["train_indices"][fraction])),
        "n_source_validation_pairs": int(len(prepared_source.split["source_validation_indices"])),
        "seed": seed,
        "stochastic_training": model != "gravity_2param",
        "support_hash": prepared_source.split["support_hash"],
        "source_scaler_hash": preprocessing.scaler_hash,
        "source_bins_hash": preprocessing.bins_hash,
        "training_config": TRAINING_CONFIG,
        "architecture": {"node_input_dim": int(prepared_source.raw.X_raw.shape[1])},
    }


def _metadata_matches(observed: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return _canonical_json(observed) == _canonical_json(expected)


def _train_with_source_validation(
    model_name: str,
    prepared_source: PreparedCity,
    train_inputs: ModelInputs,
    validation_inputs: ModelInputs,
    seed: int,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Select the lowest validation-loss epoch; stop after 10 stale epochs."""
    set_seed(seed)
    model = _new_model(model_name, train_inputs.node_features.shape[-1])
    learning_rate = float(TRAINING_CONFIG["learning_rate"])
    if model_name == "gravity_2param":
        # Preserve the original Gravity AdamW defaults (including weight decay).
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.AdamW(
            model.parameters(), lr=learning_rate,
            weight_decay=float(TRAINING_CONFIG["weight_decay"]),
        )
    truth = torch.from_numpy(train_inputs.true_flow.astype(np.float32, copy=False))
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    trace: list[dict[str, float | int]] = []
    for epoch in range(1, int(TRAINING_CONFIG["max_epochs"]) + 1):
        model.train()
        optimizer.zero_grad()
        if model_name == "gravity_2param":
            prediction = model(
                train_inputs.population_tensor[train_inputs.origin_tensor],
                train_inputs.population_tensor[train_inputs.destination_tensor],
                train_inputs.distance_tensor,
            )
        elif model_name == "pairwise_mlp":
            prediction = model(
                train_inputs.node_features[train_inputs.origin_tensor],
                train_inputs.node_features[train_inputs.destination_tensor],
                train_inputs.distance_std,
            )
        elif model_name == "urban_gnn":
            prediction = model(
                x=train_inputs.node_features,
                spatial_edge_index=prepared_source.edge_index,
                spatial_edge_dist_raw=prepared_source.edge_distance,
                pair_o_idx=train_inputs.origin_tensor,
                pair_d_idx=train_inputs.destination_tensor,
                pair_distance_km_raw=train_inputs.distance_tensor,
                population_raw=train_inputs.population_tensor,
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        loss = log1p_mse_loss(prediction, truth)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{model_name}: non-finite training loss at epoch {epoch}")
        loss.backward()
        if model_name != "gravity_2param":
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=float(TRAINING_CONFIG["gradient_clip_max_norm"])
            )
        optimizer.step()

        model.eval()
        validation_prediction = _predict(model_name, model, validation_inputs, prepared_source)
        log_error = np.log1p(validation_prediction.astype(np.float64)) - np.log1p(validation_inputs.true_flow)
        validation_loss = float(np.mean(np.square(log_error)))
        if not np.isfinite(validation_loss):
            raise FloatingPointError(f"{model_name}: non-finite validation loss at epoch {epoch}")
        trace.append({"epoch": epoch, "train_loss": float(loss.item()), "validation_loss": validation_loss})
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= int(TRAINING_CONFIG["early_stopping_patience"]):
            break
    if best_state is None:
        raise AssertionError(f"{model_name}: no validation-selected checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    detail = {
        "best_epoch": best_epoch,
        "epochs_run": len(trace),
        "best_validation_loss": best_loss,
        "stopping_reason": (
            f"patience_{TRAINING_CONFIG['early_stopping_patience']}"
            if stale_epochs >= int(TRAINING_CONFIG["early_stopping_patience"])
            else f"max_epochs_{TRAINING_CONFIG['max_epochs']}"
        ),
        "validation_pairs": len(validation_inputs.true_flow),
        "training_trace": trace,
    }
    return model, detail


def _load_or_train_model(
    output_root: Path,
    protocol_hash: str,
    source_city: str,
    prepared_source: PreparedCity,
    preprocessing: SourcePreprocessing,
    model_name: str,
    fraction: float,
    seed: int,
) -> tuple[torch.nn.Module, Path]:
    path = _checkpoint_path(output_root, source_city, model_name, fraction, seed)
    expected = _checkpoint_metadata(
        protocol_hash,
        source_city,
        model_name,
        fraction,
        seed,
        prepared_source,
        preprocessing,
    )
    if path.is_file():
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        if not _metadata_matches(bundle.get("metadata", {}), expected):
            raise ValueError(
                f"Checkpoint metadata mismatch: {path}. Refusing to reuse stale or incompatible checkpoint."
            )
        model = _new_model(model_name, prepared_source.raw.X_raw.shape[1])
        model.load_state_dict(bundle["model_state_dict"])
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model, path

    train_indices = prepared_source.split["train_indices"][fraction]
    if len(train_indices) != prepared_source.split["counts"][fraction] - prepared_source.split["validation_count"]:
        raise AssertionError(f"{source_city}: training index count mismatch for fraction={fraction}")
    train_inputs = _make_inputs(prepared_source, preprocessing.scaler, train_indices)
    validation_inputs = _make_inputs(
        prepared_source, preprocessing.scaler, prepared_source.split["source_validation_indices"]
    )
    model, train_detail = _train_with_source_validation(
        model_name, prepared_source, train_inputs, validation_inputs, seed
    )

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    _atomic_torch_save(
        path,
        {
            "metadata": expected,
            "training_detail": dict(train_detail),
            "model_state_dict": model.state_dict(),
            "saved_at": _utc_now(),
        },
    )
    return model, path


def _predict(
    model_name: str,
    model: torch.nn.Module,
    inputs: ModelInputs,
    prepared_city: PreparedCity,
) -> np.ndarray:
    count = len(inputs.true_flow)
    chunk_size = int(TRAINING_CONFIG["prediction_chunk_size"])
    result = np.empty(count, dtype=np.float32)
    with torch.no_grad():
        embeddings = (
            model.node_encoder(inputs.node_features, prepared_city.edge_index, prepared_city.edge_distance)
            if model_name == "urban_gnn" else None
        )
        for start in range(0, count, chunk_size):
            stop = min(start + chunk_size, count)
            origin = inputs.origin_tensor[start:stop]
            destination = inputs.destination_tensor[start:stop]
            distance = inputs.distance_tensor[start:stop]
            if model_name == "gravity_2param":
                prediction = model(
                    inputs.population_tensor[origin], inputs.population_tensor[destination], distance
                )
            elif model_name == "pairwise_mlp":
                prediction = model(
                    inputs.node_features[origin], inputs.node_features[destination],
                    inputs.distance_std[start:stop],
                )
            elif model_name == "urban_gnn":
                gravity_log_flow = model.gravity(
                    inputs.population_tensor[origin], inputs.population_tensor[destination], distance
                )
                prediction = model.decoder(
                    embeddings[origin], embeddings[destination], torch.log1p(distance), gravity_log_flow
                )
            else:
                raise ValueError(f"Unsupported model: {model_name}")
            result[start:stop] = prediction.detach().cpu().numpy().astype(np.float32, copy=False)
    if not np.isfinite(result).all() or np.any(result < 0.0):
        raise FloatingPointError(f"{model_name} produced non-finite or negative prediction values")
    return result


def _load_or_predict(
    output_root: Path,
    protocol_hash: str,
    source_city: str,
    target_city: str,
    model_name: str,
    fraction: float,
    seed: int,
    checkpoint_path: Path,
    prepared_target: PreparedCity,
    target_inputs: ModelInputs,
    model: torch.nn.Module,
) -> np.ndarray:
    checkpoint_hash = _sha256_file(checkpoint_path)
    expected_metadata = {
        "protocol_id": PROTOCOL_ID,
        "protocol_hash": protocol_hash,
        "source_city": source_city,
        "target_city": target_city,
        "model": model_name,
        "train_fraction": fraction,
        "seed": seed,
        "checkpoint_sha256": checkpoint_hash,
        "target_support_hash": prepared_target.split["support_hash"],
        "prediction_dtype": "float32",
    }
    cache_path = _prediction_cache_path(output_root, source_city, target_city, model_name, fraction, seed)
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as cached:
            metadata = json.loads(str(cached["metadata"].item()))
            prediction = np.asarray(cached["prediction"], dtype=np.float32)
        if _metadata_matches(metadata, expected_metadata):
            if len(prediction) != len(target_inputs.true_flow):
                raise ValueError(f"Prediction cache length mismatch: {cache_path}")
            if not np.isfinite(prediction).all() or np.any(prediction < 0.0):
                raise FloatingPointError(f"Invalid cached prediction: {cache_path}")
            return prediction
        raise ValueError(f"Prediction cache metadata mismatch: {cache_path}. Refusing stale reuse.")

    prediction = _predict(model_name, model, target_inputs, prepared_target)
    _atomic_save_prediction(cache_path, prediction, expected_metadata)
    return prediction


def _bin_diagnostics(prediction: np.ndarray, distance: np.ndarray, edges: np.ndarray, target_p: np.ndarray) -> dict[str, Any]:
    total = float(np.sum(prediction, dtype=np.float64))
    if not np.isfinite(total) or total <= 0.0:
        raise FloatingPointError(f"Baseline predicted total is non-positive/non-finite: {total}")
    bin_ids = assign_to_source_bins(distance, edges)
    q = np.zeros(len(edges) - 1, dtype=np.float64)
    for bin_id in range(len(q)):
        mask = bin_ids == bin_id
        if np.any(mask):
            q[bin_id] = float(np.sum(prediction[mask], dtype=np.float64)) / total
    if abs(float(np.sum(q)) - 1.0) >= 1e-10:
        raise AssertionError(f"Predicted DBD did not normalize: {q.sum()}")
    ratios = compute_pure_calibration_ratios(target_p, q)
    covered = float(np.sum(target_p[q > 0.0]))
    return {
        "bin_ids": bin_ids,
        "q": q,
        "predicted_total": total,
        "covered_target_mass": covered,
        "uncovered_target_mass": float(1.0 - covered),
        "n_uncovered_bins": int(np.sum((q == 0.0) & (target_p > 0.0))),
        "calibration_ratios_json": _canonical_json(ratios.tolist()),
    }


def _transfer_row(
    protocol_hash: str,
    arm: str,
    source_city: str,
    target_city: str,
    model_name: str,
    seed: int,
    fraction: float,
    K: int,
    metrics: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    **extra: Any,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "protocol_hash": protocol_hash,
        "arm": arm,
        "source_city": source_city,
        "target_city": target_city,
        "model": model_name,
        "seed": int(seed),
        "stochastic_training": model_name != "gravity_2param",
        "train_fraction": float(fraction),
        "label_budget_fraction": float(fraction),
        "nominal_gradient_train_fraction": float(fraction - VALIDATION_FRACTION),
        "K": int(K),
        "n_target_pairs": int(extra.pop("n_target_pairs")),
        "covered_target_mass": _safe_float(diagnostics["covered_target_mass"]),
        "uncovered_target_mass": _safe_float(diagnostics["uncovered_target_mass"]),
        "n_uncovered_bins": int(diagnostics["n_uncovered_bins"]),
        "calibration_ratios": str(diagnostics["calibration_ratios_json"]),
    }
    for name in TRANSFER_METRICS:
        row[name] = _safe_float(metrics[name])
    row.update(extra)
    return row


def _source_metric_row(
    protocol_hash: str,
    source_city: str,
    model_name: str,
    seed: int,
    fraction: float,
    true_flow: np.ndarray,
    prediction: np.ndarray,
    role: str,
) -> dict[str, Any]:
    if role not in {"validation", "test"}:
        raise ValueError(f"Unknown source evaluation role: {role}")
    total_true = float(np.sum(true_flow, dtype=np.float64))
    total_pred = float(np.sum(prediction, dtype=np.float64))
    if total_true <= 0.0 or total_pred < 0.0:
        raise ValueError(f"{source_city}: invalid source-test totals")
    return {
        "protocol_id": PROTOCOL_ID,
        "protocol_hash": protocol_hash,
        "source_city": source_city,
        "model": model_name,
        "seed": int(seed),
        "stochastic_training": model_name != "gravity_2param",
        "train_fraction": float(fraction),
        "label_budget_fraction": float(fraction),
        "nominal_gradient_train_fraction": float(fraction - VALIDATION_FRACTION),
        "evaluation_role": role,
        "evaluation_fraction": VALIDATION_FRACTION if role == "validation" else SOURCE_TEST_FRACTION,
        "n_evaluation_pairs": int(len(true_flow)),
        "selection_affected": role == "validation",
        "CPC": _safe_float(compute_cpc_pair(true_flow, prediction)),
        "CPC_norm": _safe_float(compute_cpc_norm_pair(true_flow, prediction)),
        "MAE": _safe_float(compute_mae_pair(true_flow, prediction)),
        "MSE": _safe_float(compute_mse_pair(true_flow, prediction)),
        "RMSE": _safe_float(compute_rmse_pair(true_flow, prediction)),
        "R_vol": _safe_float(total_pred / total_true),
    }


def _protocol_config(cities: Sequence[str]) -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "protocol_version": PROTOCOL_VERSION,
        "cities": list(cities),
        "split": {
            "seed": SPLIT_SEED,
            "fractions": list(FRACTIONS),
            "fraction_semantics": "total_source_OD_label_budget_including_validation",
            "nominal_gradient_training_fractions": [f - VALIDATION_FRACTION for f in FRACTIONS],
            "reference_fraction": MAIN_FRACTION,
            "budget_definition": "train_plus_fixed_validation_labels",
            "source_validation": "first_2pct_of_permutation",
            "source_test": "suffix_after_40pct_budget",
            "validation_fraction": VALIDATION_FRACTION,
            "source_test_fraction": SOURCE_TEST_FRACTION,
        },
        "model_coverage": {
            "urban_gnn": {"fractions": list(FRACTIONS), "seeds": list(NEURAL_SEEDS)},
            "gravity_2param": {"fractions": list(FRACTIONS), "seeds": [GRAVITY_SEED], "stochastic": False},
            "pairwise_mlp": {"fractions": [MAIN_FRACTION], "seeds": list(NEURAL_SEEDS)},
        },
        "calibration": {
            "operator": "support_conditioned_pure_dbd_source_bins_v1",
            "main_K": MAIN_K,
            "resolution_K": list(RESOLUTION_K),
            "noise_epsilons": list(NOISE_EPSILONS),
            "noise_realizations_nonzero": NOISE_REALIZATIONS,
            "global_noise_seed": NOISE_SEED,
            "specificity": "dose_matched_cyclic_donor",
        },
        "training": TRAINING_CONFIG,
    }


def _external_reuse_audit(output_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate_name in (
        "new_plan_result", "new_plan_result_test", "new_plan_result_test_opt",
        "results/new_plan_v2_focused", "results/new_plan_v3_source_validation",
    ):
        path = REPO_ROOT / candidate_name
        if not path.exists():
            continue
        protocol_path = path / "protocol.json"
        if not protocol_path.exists():
            reason = "no_protocol_fingerprint; historical_v1_artifact_not_reusable"
        else:
            try:
                payload = json.loads(protocol_path.read_text(encoding="utf-8"))
                reason = f"protocol_id={payload.get('protocol_id', 'unknown')}"
            except json.JSONDecodeError:
                reason = "invalid_protocol_json"
        rows.append({"candidate_path": str(path), "reuse": False, "reason": reason})
    current_protocol = output_root / "protocol.json"
    rows.append(
        {
            "candidate_path": str(output_root),
            "reuse": current_protocol.exists(),
            "reason": "v4_output_root_requires_hash_verified_per_source_status",
        }
    )
    return pd.DataFrame(rows)


def _initialise_output(output_root: Path, cities: Sequence[str]) -> tuple[dict[str, Any], str]:
    plan_path = REPO_ROOT / "new_approve" / "new_plan_v4_budgeted_validation.md"
    base_plan_path = REPO_ROOT / "new_approve" / "new_plan.md"
    prior_plan_path = REPO_ROOT / "new_approve" / "new_plan_v3_validation.md"
    city_manifest = REPO_ROOT / "manifests" / "cities_canonical.txt"
    payload = {
        "config": _protocol_config(cities),
        "new_plan_sha256": _sha256_file(plan_path),
        "base_plan_sha256": _sha256_file(base_plan_path),
        "prior_plan_sha256": _sha256_file(prior_plan_path),
        "canonical_cities_sha256": _sha256_file(city_manifest),
        "data_sha256": _dataset_content_hash(REPO_ROOT / "data"),
        "code_sha256": _code_hashes(),
    }
    protocol_hash = _sha256_json(payload)
    protocol_path = output_root / "protocol.json"
    if output_root.exists() and not protocol_path.exists() and any(
        path.name != "logs" for path in output_root.iterdir()
    ):
        raise RuntimeError(
            f"Refusing to use non-empty output directory without protocol.json: {output_root}. "
            "Choose a new output root rather than mixing artifacts."
        )
    output_root.mkdir(parents=True, exist_ok=True)
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing.get("protocol_hash") != protocol_hash or existing.get("payload") != payload:
            raise RuntimeError(
                f"Protocol fingerprint mismatch in {protocol_path}; existing artifacts are incompatible with this run."
            )
    else:
        _atomic_write_json(
            protocol_path,
            {
                "protocol_id": PROTOCOL_ID,
                "protocol_hash": protocol_hash,
                "created_at": _utc_now(),
                "payload": payload,
            },
        )
        _atomic_write_csv(output_root / "reuse_audit.csv", _external_reuse_audit(output_root))
    return payload, protocol_hash


def _write_static_manifests(
    output_root: Path,
    cities: Sequence[str],
    prepared: Mapping[str, PreparedCity],
    preprocessing: Mapping[str, SourcePreprocessing],
    protocol_hash: str,
) -> None:
    manifests = output_root / "manifests"
    support_rows: list[dict[str, Any]] = []
    scaler_rows: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []
    manifests.mkdir(parents=True, exist_ok=True)
    for city in cities:
        support_rows.append(
            {
                "protocol_hash": protocol_hash,
                "city": city,
                "support_hash": prepared[city].split["support_hash"],
                "n_positive_interzonal_pairs": int(len(prepared[city].split["support_indices"])),
                "n_budget_10": int(prepared[city].split["counts"][0.10]),
                "n_budget_20": int(prepared[city].split["counts"][0.20]),
                "n_budget_30": int(prepared[city].split["counts"][0.30]),
                "n_budget_40": int(prepared[city].split["counts"][0.40]),
                "n_train_10": int(len(prepared[city].split["train_indices"][0.10])),
                "n_train_20": int(len(prepared[city].split["train_indices"][0.20])),
                "n_train_30": int(len(prepared[city].split["train_indices"][0.30])),
                "n_train_40": int(len(prepared[city].split["train_indices"][0.40])),
                "n_source_validation": int(len(prepared[city].split["source_validation_indices"])),
                "n_source_test": int(len(prepared[city].split["source_test_indices"])),
                "split_seed": SPLIT_SEED,
            }
        )
        for row in preprocessing[city].scaler_records:
            scaler_rows.append({**row, "protocol_hash": protocol_hash, "scaler_hash": preprocessing[city].scaler_hash})
        for row in preprocessing[city].bin_records:
            bin_rows.append({**row, "protocol_hash": protocol_hash, "bins_hash": preprocessing[city].bins_hash})
    donor_mapping = generate_donor_mapping(list(cities), output_csv_path=None)
    donor_mapping.insert(0, "protocol_hash", protocol_hash)

    static_signature = _sha256_json(
        {
            "protocol_id": PROTOCOL_ID,
            "cities": list(cities),
            "support": [{key: row[key] for key in row if key != "protocol_hash"} for row in support_rows],
            "scaler_hashes": {city: preprocessing[city].scaler_hash for city in cities},
            "bin_hashes": {city: preprocessing[city].bins_hash for city in cities},
            "donor_mapping": donor_mapping.drop(columns=["protocol_hash"]).to_dict("records"),
        }
    )
    static_status_path = manifests / "static_manifest_status.json"
    if static_status_path.is_file():
        try:
            static_status = json.loads(static_status_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            static_status = {}
        if static_status.get("static_signature") == static_signature:
            return

    required = {
        "split": manifests / "source_split_manifest.csv",
        "support": manifests / "target_support_audit.csv",
        "scalers": manifests / "source_feature_scalers.csv",
        "bins": manifests / "source_distance_bins.csv",
        "donors": manifests / "donor_mapping.csv",
    }
    can_adopt_existing = all(path.is_file() and path.stat().st_size > 0 for path in required.values())
    if can_adopt_existing:
        try:
            existing_support = pd.read_csv(required["support"])
            expected_support = pd.DataFrame(support_rows).drop(columns=["protocol_hash"])
            support_matches = existing_support.drop(columns=["protocol_hash"], errors="ignore").equals(expected_support)
            existing_scalers = pd.read_csv(required["scalers"])
            scaler_matches = {
                city: preprocessing[city].scaler_hash
                for city in cities
            } == existing_scalers.groupby("source_city")["scaler_hash"].first().to_dict()
            existing_bins = pd.read_csv(required["bins"])
            bins_matches = {
                city: preprocessing[city].bins_hash
                for city in cities
            } == existing_bins.groupby("source_city")["bins_hash"].first().to_dict()
            existing_donors = pd.read_csv(required["donors"])
            donor_matches = existing_donors.drop(columns=["protocol_hash"], errors="ignore").equals(
                donor_mapping.drop(columns=["protocol_hash"])
            )
            can_adopt_existing = support_matches and scaler_matches and bins_matches and donor_matches
        except (OSError, KeyError, pd.errors.ParserError):
            can_adopt_existing = False
    if not can_adopt_existing:
        # This artifact has millions of rows.  Stream one city at a time
        # instead of retaining fifty split DataFrames and then concatenating
        # them in RAM.
        split_path = required["split"]
        split_temporary = split_path.with_name(f".{split_path.name}.tmp")
        with split_temporary.open("w", encoding="utf-8", newline="") as output_file:
            write_header = True
            for city in cities:
                rebuilt_split = build_v4_city_split(prepared[city].raw)
                if rebuilt_split["support_hash"] != prepared[city].split["support_hash"]:
                    raise AssertionError(f"{city}: split reconstruction changed while writing manifest")
                split_frame = rebuilt_split["manifest"].copy()
                split_frame.insert(0, "protocol_hash", protocol_hash)
                split_frame.to_csv(output_file, index=False, header=write_header)
                write_header = False
        os.replace(split_temporary, split_path)
        _atomic_write_csv(required["support"], pd.DataFrame(support_rows))
        _atomic_write_csv(required["scalers"], pd.DataFrame(scaler_rows))
        _atomic_write_csv(required["bins"], pd.DataFrame(bin_rows))
        _atomic_write_csv(required["donors"], donor_mapping)
    _atomic_write_json(
        static_status_path,
        {
            "static_signature": static_signature,
            "protocol_hash": protocol_hash,
            "scientific_hash": _sha256_json(_scientific_payload(json.loads((output_root / "protocol.json").read_text(encoding="utf-8"))["payload"])),
            "adopted_existing_files": bool(can_adopt_existing),
            "n_split_rows": int(sum(len(prepared[city].split["support_indices"]) for city in cities)),
            "created_at": _utc_now(),
        },
    )


def _source_status_path(output_root: Path, source_city: str) -> Path:
    return output_root / "per_source" / source_city / "source_status.json"


def _source_is_complete(
    output_root: Path,
    source_city: str,
    protocol_hash: str,
    targets: Sequence[str],
) -> bool:
    path = _source_status_path(output_root, source_city)
    if not path.is_file():
        return False
    try:
        status = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if status.get("status") != "complete" or status.get("protocol_hash") != protocol_hash:
        return False
    target_hash = _sha256_json(list(targets))
    if status.get("target_hash") != target_hash:
        return False
    expected_counts = expected_source_row_counts(len(targets))
    source_dir = path.parent
    for name, expected_count in expected_counts.items():
        csv_path = source_dir / name
        if not csv_path.is_file():
            return False
        try:
            actual_count = len(pd.read_csv(csv_path))
        except (OSError, pd.errors.ParserError):
            return False
        if actual_count != expected_count:
            return False
    return True


def _source_targets(all_cities: Sequence[str], source_city: str, requested_targets: Sequence[str] | None) -> list[str]:
    if requested_targets is None:
        return [city for city in all_cities if city != source_city]
    invalid = sorted(set(requested_targets) - set(all_cities))
    if invalid:
        raise ValueError(f"Unknown requested targets: {invalid}")
    targets = [city for city in requested_targets if city != source_city]
    if not targets:
        raise ValueError(f"{source_city}: requested target scope is empty")
    return targets


def _evaluate_source_subset(
    protocol_hash: str,
    source_city: str,
    prepared_source: PreparedCity,
    preprocessing: SourcePreprocessing,
    models: Mapping[tuple[str, float, int], torch.nn.Module],
    role: str,
) -> list[dict[str, Any]]:
    indices_key = "source_validation_indices" if role == "validation" else "source_test_indices"
    evaluation_inputs = _make_inputs(prepared_source, preprocessing.scaler, prepared_source.split[indices_key])
    rows: list[dict[str, Any]] = []
    for model_name, fraction, seed, _ in _model_seed_plan():
        prediction = _predict(model_name, models[(model_name, fraction, seed)], evaluation_inputs, prepared_source)
        rows.append(
            _source_metric_row(
                protocol_hash,
                source_city,
                model_name,
                seed,
                fraction,
                evaluation_inputs.true_flow,
                prediction,
                role,
            )
        )
    return rows


def _process_source(
    output_root: Path,
    protocol_hash: str,
    all_cities: Sequence[str],
    source_city: str,
    targets: Sequence[str],
    prepared: Mapping[str, PreparedCity],
    preprocessing: Mapping[str, SourcePreprocessing],
) -> None:
    if _source_is_complete(output_root, source_city, protocol_hash, targets):
        print(f"[resume] {source_city}: verified complete; no work repeated.", flush=True)
        return

    started_at = time.time()
    prepared_source = prepared[source_city]
    source_preprocessing = preprocessing[source_city]
    models: dict[tuple[str, float, int], torch.nn.Module] = {}
    checkpoints: dict[tuple[str, float, int], Path] = {}
    print(f"[{source_city}] training/loading {len(_model_seed_plan())} frozen source models", flush=True)
    for model_name, fraction, seed, _ in _model_seed_plan():
        model, checkpoint = _load_or_train_model(
            output_root,
            protocol_hash,
            source_city,
            prepared_source,
            source_preprocessing,
            model_name,
            fraction,
            seed,
        )
        models[(model_name, fraction, seed)] = model
        checkpoints[(model_name, fraction, seed)] = checkpoint

    validation_rows = _evaluate_source_subset(
        protocol_hash,
        source_city,
        prepared_source,
        source_preprocessing,
        models,
        "validation",
    )
    test_rows = _evaluate_source_subset(
        protocol_hash,
        source_city,
        prepared_source,
        source_preprocessing,
        models,
        "test",
    )
    main_rows: list[dict[str, Any]] = []
    scarcity_rows: list[dict[str, Any]] = []
    resolution_rows: list[dict[str, Any]] = []
    noise_rows: list[dict[str, Any]] = []
    specificity_rows: list[dict[str, Any]] = []
    k8_edges = source_preprocessing.bins[MAIN_K]

    for target_index, target_city in enumerate(targets, start=1):
        print(f"[{source_city}] target {target_index}/{len(targets)}: {target_city}", flush=True)
        prepared_target = prepared[target_city]
        target_inputs = _make_inputs(
            prepared_target,
            source_preprocessing.scaler,
            prepared_target.split["support_indices"],
        )
        target_p_by_k = {
            K: build_target_dbd(target_inputs.true_flow, target_inputs.distance, source_preprocessing.bins[K])
            for K in RESOLUTION_K
        }
        target_p_k8 = target_p_by_k[MAIN_K]

        for model_name, fraction, seed, _ in _model_seed_plan():
            model = models[(model_name, fraction, seed)]
            pred_before = _load_or_predict(
                output_root,
                protocol_hash,
                source_city,
                target_city,
                model_name,
                fraction,
                seed,
                checkpoints[(model_name, fraction, seed)],
                prepared_target,
                target_inputs,
                model,
            )
            k8_diag = _bin_diagnostics(pred_before, target_inputs.distance, k8_edges, target_p_k8)
            pred_after_k8 = calibrate_dbd(pred_before, target_inputs.distance, k8_edges, target_p_k8)
            main_metrics = evaluate_calibration_transfer(target_inputs.true_flow, pred_before, pred_after_k8)
            common_extra = {
                "epsilon": 0.0,
                "actual_TV": 0.0,
                "n_target_pairs": len(target_inputs.true_flow),
                "predicted_total_before": k8_diag["predicted_total"],
            }

            if model_name in {"urban_gnn", "gravity_2param"}:
                scarcity_rows.append(
                    _transfer_row(
                        protocol_hash,
                        "B_scarcity",
                        source_city,
                        target_city,
                        model_name,
                        seed,
                        fraction,
                        MAIN_K,
                        main_metrics,
                        k8_diag,
                        **common_extra,
                    )
                )
            if fraction == MAIN_FRACTION:
                main_rows.append(
                    _transfer_row(
                        protocol_hash,
                        "A_main",
                        source_city,
                        target_city,
                        model_name,
                        seed,
                        fraction,
                        MAIN_K,
                        main_metrics,
                        k8_diag,
                        **common_extra,
                    )
                )

            if model_name != "urban_gnn" or fraction != MAIN_FRACTION:
                continue

            # C1: three clean resolutions, reusing the clean K=8 result above.
            for K in RESOLUTION_K:
                if K == MAIN_K:
                    resolution_metrics = main_metrics
                    resolution_diag = k8_diag
                else:
                    edges = source_preprocessing.bins[K]
                    p_target = target_p_by_k[K]
                    resolution_diag = _bin_diagnostics(pred_before, target_inputs.distance, edges, p_target)
                    pred_after = calibrate_dbd(pred_before, target_inputs.distance, edges, p_target)
                    resolution_metrics = evaluate_calibration_transfer(target_inputs.true_flow, pred_before, pred_after)
                resolution_rows.append(
                    _transfer_row(
                        protocol_hash,
                        "C1_resolution",
                        source_city,
                        target_city,
                        model_name,
                        seed,
                        fraction,
                        K,
                        resolution_metrics,
                        resolution_diag,
                        epsilon=0.0,
                        actual_TV=0.0,
                        realization_id=0,
                        noise_seed=None,
                        n_target_pairs=len(target_inputs.true_flow),
                        predicted_total_before=resolution_diag["predicted_total"],
                    )
                )

            # C2: K=8 clean condition once, then five exact-TV realizations per nonzero epsilon.
            noise_rows.append(
                _transfer_row(
                    protocol_hash,
                    "C2_noise",
                    source_city,
                    target_city,
                    model_name,
                    seed,
                    fraction,
                    MAIN_K,
                    main_metrics,
                    k8_diag,
                    epsilon=0.0,
                    actual_TV=0.0,
                    realization_id=0,
                    noise_seed=None,
                    n_target_pairs=len(target_inputs.true_flow),
                    predicted_total_before=k8_diag["predicted_total"],
                )
            )
            for epsilon in (0.05, 0.10):
                for realization_id in range(NOISE_REALIZATIONS):
                    noise_seed = derive_noise_seed(
                        global_noise_seed=NOISE_SEED,
                        source_city=source_city,
                        target_city=target_city,
                        K=MAIN_K,
                        epsilon=epsilon,
                        realization_id=realization_id,
                    )
                    p_noisy, actual_tv, attempts = generate_exact_tv_noise(
                        p=target_p_k8,
                        epsilon=epsilon,
                        rng_or_seed=noise_seed,
                        source_city=source_city,
                        target_city=target_city,
                        K=MAIN_K,
                        realization_id=realization_id,
                    )
                    if abs(actual_tv - epsilon) >= 1e-10:
                        raise AssertionError(
                            f"{source_city}->{target_city}: exact-TV check failed ({actual_tv} vs {epsilon})"
                        )
                    noisy_diag = _bin_diagnostics(pred_before, target_inputs.distance, k8_edges, p_noisy)
                    pred_noisy = calibrate_dbd(pred_before, target_inputs.distance, k8_edges, p_noisy)
                    noisy_metrics = evaluate_calibration_transfer(target_inputs.true_flow, pred_before, pred_noisy)
                    noise_rows.append(
                        _transfer_row(
                            protocol_hash,
                            "C2_noise",
                            source_city,
                            target_city,
                            model_name,
                            seed,
                            fraction,
                            MAIN_K,
                            noisy_metrics,
                            noisy_diag,
                            epsilon=epsilon,
                            actual_TV=actual_tv,
                            realization_id=realization_id,
                            noise_seed=int(noise_seed),
                            noise_attempts=int(attempts),
                            n_target_pairs=len(target_inputs.true_flow),
                            predicted_total_before=noisy_diag["predicted_total"],
                        )
                    )

            # D: target-aligned DBD against the deterministic dose-matched donor control.
            donor_city = choose_donor(source_city, target_city, list(all_cities))
            prepared_donor = prepared[donor_city]
            donor_support = prepared_donor.split["support_indices"]
            donor_p = build_target_dbd(
                np.asarray(prepared_donor.raw.pair_trips, dtype=np.float64)[donor_support],
                np.asarray(prepared_donor.raw.dist_km, dtype=np.float64)[donor_support],
                k8_edges,
            )
            dose = find_dose_matching_lambda(
                p_s_target=target_p_k8,
                p_s_donor=donor_p,
                q_s_target=k8_diag["q"],
                tolerance=1e-6,
                eps=1e-9,
                source_city=source_city,
                target_city=target_city,
                donor_city=donor_city,
            )
            pred_control = calibrate_dbd(pred_before, target_inputs.distance, k8_edges, dose["p_scaled_donor"])
            control_metrics = evaluate_calibration_transfer(target_inputs.true_flow, pred_before, pred_control)
            specificity_rows.append(
                {
                    **_transfer_row(
                        protocol_hash,
                        "D_specificity",
                        source_city,
                        target_city,
                        model_name,
                        seed,
                        fraction,
                        MAIN_K,
                        main_metrics,
                        k8_diag,
                        n_target_pairs=len(target_inputs.true_flow),
                        donor_city=donor_city,
                        target_dose=_safe_float(dose["target_dose"]),
                        donor_raw_dose=_safe_float(dose["donor_raw_dose"]),
                        donor_scaled_dose=_safe_float(dose["donor_scaled_dose"]),
                        dose_error=_safe_float(dose["dose_error"]),
                        lambda_star=_safe_float(dose["lambda_star"]),
                        lambda_low=_safe_float(dose["lambda_low"]),
                        lambda_high=_safe_float(dose["lambda_high"]),
                        root_iterations=int(dose["root_iterations"]),
                        root_converged=bool(dose["root_converged"]),
                        CPC_donor_control=_safe_float(control_metrics["CPC_after"]),
                        CPC_norm_donor_control=_safe_float(control_metrics["CPC_norm_after"]),
                        delta_CPC_donor_control=_safe_float(control_metrics["delta_CPC"]),
                        delta_CPC_norm_donor_control=_safe_float(control_metrics["delta_CPC_norm"]),
                        structural_advantage_delta=_safe_float(
                            main_metrics["delta_CPC"] - control_metrics["delta_CPC"]
                        ),
                    ),
                }
            )

    source_dir = output_root / "per_source" / source_city
    output_frames = {
        "source_validation_metrics.csv": pd.DataFrame(validation_rows),
        "source_test_metrics.csv": pd.DataFrame(test_rows),
        "main_transfer_metrics.csv": pd.DataFrame(main_rows),
        "scarcity_transfer_metrics.csv": pd.DataFrame(scarcity_rows),
        "resolution_metrics.csv": pd.DataFrame(resolution_rows),
        "noise_metrics.csv": pd.DataFrame(noise_rows),
        "specificity_metrics.csv": pd.DataFrame(specificity_rows),
    }
    expected_counts = expected_source_row_counts(len(targets))
    for name, frame in output_frames.items():
        if len(frame) != expected_counts[name]:
            raise AssertionError(
                f"{source_city}: {name} has {len(frame)} rows, expected {expected_counts[name]}"
            )
        _atomic_write_csv(source_dir / name, frame)
    _atomic_write_json(
        source_dir / "source_status.json",
        {
            "status": "complete",
            "protocol_id": PROTOCOL_ID,
            "protocol_hash": protocol_hash,
            "source_city": source_city,
            "targets": list(targets),
            "target_hash": _sha256_json(list(targets)),
            "row_counts": expected_counts,
            "completed_at": _utc_now(),
            "elapsed_seconds": time.time() - started_at,
        },
    )
    print(f"[{source_city}] complete in {time.time() - started_at:.1f}s", flush=True)


def _read_per_source(output_root: Path, cities: Iterable[str], file_name: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for city in cities:
        path = output_root / "per_source" / city / file_name
        if path.is_file():
            frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _seed_aggregate(frame: pd.DataFrame, group_columns: Sequence[str], metric_columns: Sequence[str]) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    records: list[dict[str, Any]] = []
    for key, group in frame.groupby(list(group_columns), sort=True, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        record = dict(zip(group_columns, key_values))
        model = str(group["model"].iloc[0])
        expected_seed_count = 1 if model == "gravity_2param" else len(NEURAL_SEEDS)
        observed_seeds = sorted(int(seed) for seed in group["seed"].unique())
        if len(observed_seeds) != expected_seed_count:
            raise ValueError(
                f"Seed aggregation expected {expected_seed_count} seeds for {model}, got {observed_seeds}; key={record}"
            )
        record["seed_count"] = len(observed_seeds)
        record["seeds"] = ",".join(str(seed) for seed in observed_seeds)
        for column in metric_columns:
            if column not in group:
                continue
            values = group[column].astype(float).to_numpy()
            record[column] = float(np.mean(values))
            record[f"{column}_seed_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        records.append(record)
    return pd.DataFrame(records)


def _target_and_source_summary(frame: pd.DataFrame, outcome: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    target = (
        frame.groupby(["target_city", "model"], sort=True)[outcome]
        .agg(mean_gain="mean", median_gain="median", n_sources="count", positive_sources=lambda series: int((series > 0).sum()))
        .reset_index()
    )
    source = (
        frame.groupby(["source_city", "model"], sort=True)[outcome]
        .agg(mean_gain="mean", median_gain="median", n_targets="count", positive_targets=lambda series: int((series > 0).sum()))
        .reset_index()
    )
    return target, source


def _bootstrap_wilcoxon(values: np.ndarray, seed: int = 42) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        raise ValueError("Cannot infer from no values")
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(10_000, len(values)), replace=True).mean(axis=1)
    if np.all(values == 0.0) or len(np.unique(values)) <= 1:
        statistic, p_value = 0.0, 1.0
    else:
        from scipy.stats import wilcoxon

        test = wilcoxon(values, alternative="two-sided", zero_method="wilcox", correction=False, method="auto")
        statistic, p_value = float(test.statistic), float(test.pvalue)
    return {
        "n_targets": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
        "bootstrap_ci_low": float(np.percentile(draws, 2.5)),
        "bootstrap_ci_high": float(np.percentile(draws, 97.5)),
        "wilcoxon_stat": statistic,
        "wilcoxon_p_raw": p_value,
        "positive_targets": int(np.sum(values > 0.0)),
        "positive_target_fraction": float(np.mean(values > 0.0)),
    }


def _fit_crossed_effect(frame: pd.DataFrame, outcome: str, label: str) -> dict[str, Any]:
    from statsmodels.regression.mixed_linear_model import MixedLM

    if len(frame) != 2450 or frame["source_city"].nunique() != 50 or frame["target_city"].nunique() != 50:
        raise ValueError(f"{label}: crossed model requires the complete 50x49 transfer matrix")
    if (frame["source_city"] == frame["target_city"]).any() or frame[outcome].isna().any():
        raise ValueError(f"{label}: invalid crossed-model input")
    data = frame[["source_city", "target_city", outcome]].copy()
    data["_all"] = 1
    model = MixedLM.from_formula(
        f"{outcome} ~ 1",
        groups=data["_all"],
        re_formula="0",
        vc_formula={"source": "0 + C(source_city)", "target": "0 + C(target_city)"},
        data=data,
    )
    fit = model.fit(method="lbfgs", maxiter=1000, reml=True)
    if not fit.converged:
        raise RuntimeError(f"{label}: crossed mixed-effects model did not converge")
    beta = float(fit.params["Intercept"])
    se = float(fit.bse["Intercept"])
    p_value = float(fit.pvalues["Intercept"])
    if not np.isfinite(beta) or not np.isfinite(se) or not np.isfinite(p_value):
        raise RuntimeError(f"{label}: non-finite mixed-effects estimate")
    variances = list(fit.vcomp)
    return {
        "comparison": label,
        "outcome": outcome,
        "n_pairs": int(len(data)),
        "beta0": beta,
        "se_beta0": se,
        "ci_low": beta - 1.95996 * se,
        "ci_high": beta + 1.95996 * se,
        "p_raw": p_value,
        "source_variance": float(variances[0]) if variances else 0.0,
        "target_variance": float(variances[1]) if len(variances) > 1 else 0.0,
        "residual_variance": float(fit.scale),
        "converged": bool(fit.converged),
    }


def _holm_adjust(p_values: Sequence[float]) -> list[float]:
    array = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(array)
    output = np.zeros(len(array), dtype=np.float64)
    running_max = 0.0
    for rank, index in enumerate(order):
        adjusted = min(1.0, array[index] * (len(array) - rank))
        running_max = max(running_max, adjusted)
        output[index] = running_max
    return output.tolist()


def _finalize(output_root: Path, cities: Sequence[str], protocol_hash: str) -> bool:
    complete = [
        city
        for city in cities
        if _source_is_complete(output_root, city, protocol_hash, [other for other in cities if other != city])
    ]
    execution_status = {
        "protocol_id": PROTOCOL_ID,
        "protocol_hash": protocol_hash,
        "total_sources": len(cities),
        "complete_sources": complete,
        "n_complete_sources": len(complete),
        "complete": len(complete) == len(cities),
        "updated_at": _utc_now(),
    }
    _atomic_write_json(output_root / "execution_status.json", execution_status)
    if not complete:
        return False

    raw_files = {
        "source_validation_metrics.csv": "source_validation_metrics.csv",
        "source_test_metrics.csv": "source_test_metrics.csv",
        "main_transfer_metrics.csv": "main_transfer_metrics.csv",
        "scarcity_transfer_metrics.csv": "scarcity_transfer_metrics.csv",
        "resolution_metrics.csv": "resolution_metrics.csv",
        "noise_metrics.csv": "noise_metrics.csv",
        "specificity_metrics.csv": "specificity_metrics.csv",
    }
    frames = {output_name: _read_per_source(output_root, complete, source_name) for output_name, source_name in raw_files.items()}
    for output_name, frame in frames.items():
        _atomic_write_csv(output_root / output_name, frame)

    source_mean = _seed_aggregate(
        frames["source_validation_metrics.csv"],
        ["source_city", "model", "train_fraction"],
        ["CPC", "CPC_norm", "MAE", "MSE", "RMSE", "R_vol"],
    )
    source_test_mean = _seed_aggregate(
        frames["source_test_metrics.csv"],
        ["source_city", "model", "train_fraction"],
        ["CPC", "CPC_norm", "MAE", "MSE", "RMSE", "R_vol"],
    )
    main_mean = _seed_aggregate(
        frames["main_transfer_metrics.csv"],
        ["source_city", "target_city", "model", "train_fraction", "K"],
        list(TRANSFER_METRICS) + ["covered_target_mass", "uncovered_target_mass", "predicted_total_before"],
    )
    scarcity_mean = _seed_aggregate(
        frames["scarcity_transfer_metrics.csv"],
        ["source_city", "target_city", "model", "train_fraction", "K"],
        list(TRANSFER_METRICS) + ["covered_target_mass", "uncovered_target_mass", "predicted_total_before"],
    )
    resolution_mean = _seed_aggregate(
        frames["resolution_metrics.csv"],
        ["source_city", "target_city", "model", "train_fraction", "K"],
        list(TRANSFER_METRICS) + ["covered_target_mass", "uncovered_target_mass", "predicted_total_before"],
    )
    noise_per_seed = (
        frames["noise_metrics.csv"]
        .groupby(["source_city", "target_city", "model", "seed", "train_fraction", "K", "epsilon"], sort=True)[
            list(TRANSFER_METRICS) + ["actual_TV", "covered_target_mass", "uncovered_target_mass", "predicted_total_before"]
        ]
        .mean()
        .reset_index()
    )
    noise_mean = _seed_aggregate(
        noise_per_seed,
        ["source_city", "target_city", "model", "train_fraction", "K", "epsilon"],
        list(TRANSFER_METRICS) + ["actual_TV", "covered_target_mass", "uncovered_target_mass", "predicted_total_before"],
    )
    specificity_mean = _seed_aggregate(
        frames["specificity_metrics.csv"],
        ["source_city", "target_city", "model", "train_fraction", "K"],
        list(TRANSFER_METRICS)
        + [
            "target_dose",
            "donor_raw_dose",
            "donor_scaled_dose",
            "dose_error",
            "lambda_star",
            "CPC_donor_control",
            "CPC_norm_donor_control",
            "delta_CPC_donor_control",
            "delta_CPC_norm_donor_control",
            "structural_advantage_delta",
        ],
    )
    _atomic_write_csv(output_root / "source_validation_metrics_mean.csv", source_mean)
    _atomic_write_csv(output_root / "source_test_metrics_mean.csv", source_test_mean)
    _atomic_write_csv(output_root / "main_transfer_metrics_mean.csv", main_mean)
    _atomic_write_csv(output_root / "scarcity_transfer_metrics_mean.csv", scarcity_mean)
    _atomic_write_csv(output_root / "resolution_metrics_mean.csv", resolution_mean)
    _atomic_write_csv(output_root / "noise_metrics_per_seed_mean.csv", noise_per_seed)
    _atomic_write_csv(output_root / "noise_metrics_mean.csv", noise_mean)
    _atomic_write_csv(output_root / "specificity_metrics_mean.csv", specificity_mean)

    # Summary tables are meaningful for partial runs; inferential tables are gated below.
    if not main_mean.empty:
        main_target, main_source = _target_and_source_summary(main_mean, "delta_CPC")
        _atomic_write_csv(output_root / "main_target_summary.csv", main_target)
        _atomic_write_csv(output_root / "main_source_summary.csv", main_source)
    if not scarcity_mean.empty:
        scarcity_target = (
            scarcity_mean.groupby(["target_city", "model", "train_fraction"], sort=True)["delta_CPC"]
            .agg(mean_gain="mean", median_gain="median", n_sources="count")
            .reset_index()
        )
        _atomic_write_csv(output_root / "scarcity_target_summary.csv", scarcity_target)
    if not resolution_mean.empty:
        resolution_summary = (
            resolution_mean.groupby(["model", "K"], sort=True)["delta_CPC"]
            .agg(mean="mean", median="median", std="std", n_pairs="count")
            .reset_index()
        )
        _atomic_write_csv(output_root / "resolution_summary.csv", resolution_summary)
    if not noise_mean.empty:
        noise_summary = (
            noise_mean.groupby(["model", "K", "epsilon"], sort=True)["delta_CPC"]
            .agg(mean="mean", median="median", std="std", n_pairs="count")
            .reset_index()
        )
        _atomic_write_csv(output_root / "noise_summary.csv", noise_summary)
    if not specificity_mean.empty:
        specificity_summary = (
            specificity_mean.groupby(["target_city", "model"], sort=True)["structural_advantage_delta"]
            .agg(mean="mean", median="median", n_sources="count")
            .reset_index()
        )
        _atomic_write_csv(output_root / "specificity_target_summary.csv", specificity_summary)

    if len(complete) != len(cities):
        print(f"Finalization is partial ({len(complete)}/{len(cities)} sources); inferential outputs are deferred.", flush=True)
        return False

    # Confirmatory outputs: three main added-value effects + one pre-specified scarcity contrast.
    main_target, _ = _target_and_source_summary(main_mean, "delta_CPC")
    target_inference_rows = []
    for model in ("urban_gnn", "gravity_2param", "pairwise_mlp"):
        values = main_target.loc[main_target["model"] == model, "mean_gain"].to_numpy(dtype=np.float64)
        if len(values) != 50:
            raise AssertionError(f"Main target summary is incomplete for {model}")
        target_inference_rows.append({"comparison": f"A_{model}", "model": model, **_bootstrap_wilcoxon(values)})
    _atomic_write_csv(output_root / "main_target_inference.csv", pd.DataFrame(target_inference_rows))

    gnn_scarcity = scarcity_mean[scarcity_mean["model"] == "urban_gnn"].copy()
    pivot = gnn_scarcity.pivot(index=["source_city", "target_city"], columns="train_fraction", values="delta_CPC").reset_index()
    if not {0.10, 0.40}.issubset(set(pivot.columns)):
        raise AssertionError("Missing GNN 10%/40% scarcity gains")
    contrast = pivot[["source_city", "target_city", 0.10, 0.40]].copy()
    contrast["gain_10_minus_gain_40"] = contrast[0.10] - contrast[0.40]
    contrast_target = (
        contrast.groupby("target_city", sort=True)["gain_10_minus_gain_40"]
        .mean()
        .reset_index(name="mean_contrast")
    )
    if len(contrast_target) != 50:
        raise AssertionError("Scarcity target contrast must contain 50 target cities")
    _atomic_write_csv(output_root / "scarcity_primary_contrast.csv", contrast)
    _atomic_write_csv(output_root / "scarcity_primary_contrast_target_summary.csv", contrast_target)
    _atomic_write_csv(
        output_root / "scarcity_primary_contrast_inference.csv",
        pd.DataFrame([{"comparison": "B_urban_gnn_gain_10_minus_gain_40", **_bootstrap_wilcoxon(contrast_target["mean_contrast"].to_numpy())}]),
    )

    mixed_rows = []
    for model in ("urban_gnn", "gravity_2param", "pairwise_mlp"):
        mixed_rows.append(
            _fit_crossed_effect(
                main_mean[main_mean["model"] == model],
                "delta_CPC",
                f"A_{model}",
            )
        )
    contrast_with_model = contrast.assign(model="urban_gnn")
    mixed_rows.append(
        _fit_crossed_effect(
            contrast_with_model,
            "gain_10_minus_gain_40",
            "B_urban_gnn_gain_10_minus_gain_40",
        )
    )
    for row, adjusted in zip(mixed_rows, _holm_adjust([row["p_raw"] for row in mixed_rows])):
        row["p_holm_confirmatory_family"] = adjusted
    _atomic_write_csv(output_root / "confirmatory_crossed_effects.csv", pd.DataFrame(mixed_rows))

    _atomic_write_text(output_root / "EXECUTION_COMPLETE.marker", "computed\n")
    return True


def _load_cities() -> list[str]:
    city_file = REPO_ROOT / "manifests" / "cities_canonical.txt"
    cities = [line.strip() for line in city_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(cities) != 50 or len(set(cities)) != 50:
        raise ValueError(f"Canonical city manifest must contain 50 unique cities, got {len(cities)}")
    return cities


def run(
    output_root: Path,
    requested_sources: Sequence[str] | None = None,
    requested_targets: Sequence[str] | None = None,
    preflight_only: bool = False,
    finalize_only: bool = False,
) -> int:
    all_cities = _load_cities()
    unknown_sources = sorted(set(requested_sources or ()) - set(all_cities))
    if unknown_sources:
        raise ValueError(f"Unknown requested sources: {unknown_sources}")
    sources = list(requested_sources) if requested_sources else list(all_cities)
    if len(set(sources)) != len(sources):
        raise ValueError("Requested sources contain duplicates")
    if finalize_only and (requested_sources or requested_targets):
        raise ValueError("--finalize-only cannot be combined with --sources or --targets")

    _, protocol_hash = _initialise_output(output_root, all_cities)
    print(f"Protocol {PROTOCOL_ID} [{protocol_hash[:12]}]", flush=True)
    print("Preparing v4 support splits, source preprocessing, and geographic graphs...", flush=True)
    prepared = _prepare_cities(all_cities)
    preprocessing = {city: _build_source_preprocessing(city, prepared[city]) for city in all_cities}
    _write_static_manifests(output_root, all_cities, prepared, preprocessing, protocol_hash)
    if preflight_only:
        print("Preflight complete; no models trained.", flush=True)
        return 0
    if finalize_only:
        return 0 if _finalize(output_root, all_cities, protocol_hash) else 2

    for index, source_city in enumerate(sources, start=1):
        targets = _source_targets(all_cities, source_city, requested_targets)
        print(f"Source {index}/{len(sources)}: {source_city} -> {len(targets)} targets", flush=True)
        _process_source(
            output_root,
            protocol_hash,
            all_cities,
            source_city,
            targets,
            prepared,
            preprocessing,
        )

    full_scope = requested_sources is None and requested_targets is None
    if full_scope:
        return 0 if _finalize(output_root, all_cities, protocol_hash) else 2
    return 0


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/new_plan_v4_budgeted_validation", help="Dedicated v4 output root")
    parser.add_argument("--sources", nargs="+", help="Optional canonical source-city subset")
    parser.add_argument("--targets", nargs="+", help="Optional canonical target-city subset for smoke execution")
    parser.add_argument("--preflight-only", action="store_true", help="Write/verify v4 manifests without training")
    parser.add_argument("--finalize-only", action="store_true", help="Aggregate a completed full-scope v4 run")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output_root = (REPO_ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    try:
        return run(
            output_root=output_root,
            requested_sources=args.sources,
            requested_targets=args.targets,
            preflight_only=bool(args.preflight_only),
            finalize_only=bool(args.finalize_only),
        )
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
