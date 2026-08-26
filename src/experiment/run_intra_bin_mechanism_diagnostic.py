"""Mechanism diagnostic for distance-bin calibration gains.

This is a post hoc diagnostic, not a replacement for the primary analysis.
It measures whether the frozen M0 prediction preserves within-bin allocation
quality and whether that quality is associated with the M1 city-oracle gain.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.calibration.bin_calibration import calibrate_kbins
from src.data.city_splits import load_splits_manifest_v2
from src.data.dataset import load_city, load_raw_city
from src.data.urban_graph import build_radius_graph
from src.data.yd_extractor import compute_kbin_edges, extract_yd_kbins
from src.experiment.run_backbone_robustness import fit_gravity_parameters
from src.training.evaluate import compute_cpc_pair
from src.training.train import infer_zero_shot, load_checkpoint


CANONICAL_SEEDS = [1, 10, 100]
K_MOVE = 8
DEFAULT_OUTPUT = Path("results/intra_bin_mechanism_diagnostic.json")


def _within_bin_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    distances_km: np.ndarray,
    inter_mask: np.ndarray,
    bin_edges: np.ndarray,
) -> tuple[float, float, list[dict[str, Any]]]:
    """Return true-mass-weighted allocation CPC and rank quality by bin."""
    active_rows: list[dict[str, Any]] = []
    total_true_mass = float(truth[inter_mask].sum())
    weighted_alloc = 0.0
    weighted_rank = 0.0
    weight_sum = 0.0

    for bin_id in range(len(bin_edges) - 1):
        lo, hi = float(bin_edges[bin_id]), float(bin_edges[bin_id + 1])
        bin_mask = inter_mask & (distances_km > lo) & (distances_km <= hi)
        if not bin_mask.any():
            continue

        true_bin = truth[bin_mask].astype(np.float64)
        pred_bin = np.maximum(prediction[bin_mask].astype(np.float64), 0.0)
        true_mass = float(true_bin.sum())
        pred_mass = float(pred_bin.sum())
        if true_mass <= 0.0 or pred_mass <= 0.0:
            continue

        true_share = true_bin / true_mass
        pred_share = pred_bin / pred_mass
        allocation_cpc = float(np.minimum(true_share, pred_share).sum())
        rank = float(spearmanr(true_bin, pred_bin).statistic) if len(true_bin) >= 2 else 0.0
        if not np.isfinite(rank):
            rank = 0.0

        weight = true_mass / total_true_mass if total_true_mass > 0.0 else 0.0
        weighted_alloc += weight * allocation_cpc
        weighted_rank += weight * rank
        weight_sum += weight
        active_rows.append({
            "bin": bin_id,
            "n_pairs": int(bin_mask.sum()),
            "true_mass": true_mass,
            "weight": weight,
            "within_bin_cpc": allocation_cpc,
            "within_bin_spearman": rank,
        })

    if weight_sum == 0.0:
        return 0.0, 0.0, active_rows
    return weighted_alloc / weight_sum, weighted_rank / weight_sum, active_rows


def _distance_marginal_tv(
    prediction: np.ndarray,
    target_yd: np.ndarray,
    distances_km: np.ndarray,
    inter_mask: np.ndarray,
    bin_edges: np.ndarray,
) -> float:
    """Total variation between predicted and target distance-bin marginals."""
    pred_inter = np.maximum(prediction[inter_mask].astype(np.float64), 0.0)
    pred_total = float(pred_inter.sum())
    if pred_total <= 0.0:
        return 1.0

    pred_yd = np.zeros(len(bin_edges) - 1, dtype=np.float64)
    inter_dist = distances_km[inter_mask]
    for bin_id in range(len(pred_yd)):
        lo, hi = float(bin_edges[bin_id]), float(bin_edges[bin_id + 1])
        pred_yd[bin_id] = pred_inter[(inter_dist > lo) & (inter_dist <= hi)].sum()
    pred_yd /= pred_total
    target = np.asarray(target_yd, dtype=np.float64)
    return float(0.5 * np.abs(pred_yd - target).sum())


def _city_seed_diagnostic(
    city: str,
    fold: int,
    model: torch.nn.Module,
    scaler: object,
    bin_edges: np.ndarray,
    data_root: str,
    device: torch.device,
) -> dict[str, Any]:
    city_data = load_city(city, data_root=data_root, feature_scaler=scaler, fit_scaler=False)
    edge_index, edge_dist = build_radius_graph(city_data.lon_lat.numpy(), radius_km=5.0)
    truth = city_data.pair_trips.numpy().astype(np.float64)
    distances_km = np.expm1(city_data.pair_distance.numpy())
    inter_mask = (city_data.pair_o_idx.numpy() != city_data.pair_d_idx.numpy()) & (distances_km > 0.0)

    prediction = infer_zero_shot(model, city_data, edge_index, edge_dist, device=device).numpy().astype(np.float64)
    yd_target = extract_yd_kbins(distances_km, truth, bin_edges, inter_mask)
    calibrated = calibrate_kbins(prediction, distances_km, inter_mask, yd_target, bin_edges, q=1.0, tolerance=1e-5)

    m0_cpc = float(compute_cpc_pair(truth[inter_mask], prediction[inter_mask]))
    m1_cpc = float(compute_cpc_pair(truth[inter_mask], calibrated[inter_mask]))
    q_alloc, q_rank, bins = _within_bin_metrics(truth, prediction, distances_km, inter_mask, bin_edges)
    q_alloc_m1, q_rank_m1, _ = _within_bin_metrics(truth, calibrated, distances_km, inter_mask, bin_edges)
    d_pre = _distance_marginal_tv(prediction, yd_target, distances_km, inter_mask, bin_edges)

    return {
        "city": city,
        "fold": fold,
        "m0_cpc_inter": m0_cpc,
        "m1_cpc_inter": m1_cpc,
        "delta_cpc": m1_cpc - m0_cpc,
        "d_pre_tv": d_pre,
        "q_alloc": q_alloc,
        "q_rank": q_rank,
        "q_alloc_m1": q_alloc_m1,
        "q_rank_m1": q_rank_m1,
        "rank_invariance_abs_diff": abs(q_rank - q_rank_m1),
        "bins": bins,
    }


def _city_gravity_diagnostic(
    city: str,
    fold: int,
    gravity_g: float,
    gravity_alpha: float,
    bin_edges: np.ndarray,
    data_root: str,
) -> dict[str, Any]:
    raw = load_raw_city(city, data_root=data_root)
    truth = raw.pair_trips.numpy().astype(np.float64)
    distances_km = raw.dist_km.astype(np.float64)
    origins = raw.pair_o_idx.numpy()
    destinations = raw.pair_d_idx.numpy()
    inter_mask = (origins != destinations) & (distances_km > 0.0)

    population = raw.population.numpy()
    p_i = np.clip(population[origins], 1.0, None)
    p_j = np.clip(population[destinations], 1.0, None)
    distance = np.clip(distances_km, 0.1, None)
    prediction = np.exp(gravity_g) * p_i * p_j * (distance ** (-gravity_alpha))

    yd_target = extract_yd_kbins(distances_km, truth, bin_edges, inter_mask)
    calibrated = calibrate_kbins(prediction, distances_km, inter_mask, yd_target, bin_edges, q=1.0, tolerance=1e-5)
    m0_cpc = float(compute_cpc_pair(truth[inter_mask], prediction[inter_mask]))
    m1_cpc = float(compute_cpc_pair(truth[inter_mask], calibrated[inter_mask]))
    q_alloc, q_rank, bins = _within_bin_metrics(truth, prediction, distances_km, inter_mask, bin_edges)
    q_alloc_m1, q_rank_m1, _ = _within_bin_metrics(truth, calibrated, distances_km, inter_mask, bin_edges)

    return {
        "city": city,
        "fold": fold,
        "gravity_g": gravity_g,
        "gravity_alpha": gravity_alpha,
        "m0_cpc_inter": m0_cpc,
        "m1_cpc_inter": m1_cpc,
        "delta_cpc": m1_cpc - m0_cpc,
        "d_pre_tv": _distance_marginal_tv(prediction, yd_target, distances_km, inter_mask, bin_edges),
        "q_alloc": q_alloc,
        "q_rank": q_rank,
        "q_alloc_m1": q_alloc_m1,
        "q_rank_m1": q_rank_m1,
        "rank_invariance_abs_diff": abs(q_rank - q_rank_m1),
        "bins": bins,
    }


def run_diagnostic(
    data_root: str = "data",
    output_path: Path = DEFAULT_OUTPUT,
    device_str: str = "cpu",
    backbone: str = "gnn",
) -> dict[str, Any]:
    if backbone not in {"gnn", "mlp", "gravity"}:
        raise ValueError(f"Unsupported checkpoint backbone: {backbone}")
    manifest_path = Path("results/e1/splits_manifest_v2.json")
    splits = load_splits_manifest_v2(str(manifest_path), data_root=data_root)
    per_seed: list[dict[str, Any]] = []
    per_city: list[dict[str, Any]] = []
    fold_parameters: list[dict[str, Any]] = []

    for fold in range(1, 6):
        split = splits[fold]
        bin_edges, k_active = compute_kbin_edges(split["train"], K=K_MOVE, data_root=data_root)
        if k_active != K_MOVE:
            raise RuntimeError(f"Expected K_active={K_MOVE}, got {k_active} in fold {fold}")
        if backbone == "gravity":
            gravity_g, gravity_alpha = fit_gravity_parameters(split["train"], data_root=data_root)
            fold_parameters.append({"fold": fold, "G": gravity_g, "alpha": gravity_alpha})
            for city in sorted(split["test"]):
                per_city.append(_city_gravity_diagnostic(
                    city, fold, gravity_g, gravity_alpha, bin_edges, data_root
                ))
            continue

        models = {}
        for seed in CANONICAL_SEEDS:
            checkpoint_name = f"5fold_fold{fold}_seed{seed}.pt" if backbone == "gnn" else f"mlp_fold{fold}_seed{seed}.pt"
            checkpoint = Path("results/checkpoints") / checkpoint_name
            model, scaler, metadata = load_checkpoint(checkpoint, device_str=device_str)
            if metadata.get("seed") != seed or metadata.get("hyperparams", {}).get("fold") != fold:
                raise RuntimeError(f"Checkpoint provenance mismatch: {checkpoint}")
            models[seed] = (model, scaler)

        for city in sorted(split["test"]):
            city_seed_rows = []
            for seed in CANONICAL_SEEDS:
                model, scaler = models[seed]
                row = _city_seed_diagnostic(city, fold, model, scaler, bin_edges, data_root, torch.device(device_str))
                row["model_seed"] = seed
                per_seed.append(row)
                city_seed_rows.append(row)

            averaged = {"city": city, "fold": fold, "model_seeds": CANONICAL_SEEDS}
            for key in ["m0_cpc_inter", "m1_cpc_inter", "delta_cpc", "d_pre_tv", "q_alloc", "q_rank", "q_alloc_m1", "q_rank_m1", "rank_invariance_abs_diff"]:
                averaged[key] = float(np.mean([row[key] for row in city_seed_rows]))
            per_city.append(averaged)

    q_alloc = np.array([row["q_alloc"] for row in per_city])
    q_rank = np.array([row["q_rank"] for row in per_city])
    d_pre = np.array([row["d_pre_tv"] for row in per_city])
    delta = np.array([row["delta_cpc"] for row in per_city])

    def correlation(x: np.ndarray) -> dict[str, float | int | None]:
        result = spearmanr(x, delta)
        return {"n_cities": len(delta), "rho": float(result.statistic), "p_value": float(result.pvalue)}

    payload = {
        "diagnostic": "intra-bin allocation quality vs M1 city-oracle gain",
        "interpretation": "mechanistic evidence; not a causal claim",
        "protocol": {"backbone": backbone, "folds": [1, 2, 3, 4, 5], "seeds": [] if backbone == "gravity" else CANONICAL_SEEDS, "K": K_MOVE, "statistical_unit": "city"},
        "fold_parameters": fold_parameters,
        "correlations": {
            "d_pre_tv_vs_delta_cpc": correlation(d_pre),
            "q_alloc_vs_delta_cpc": correlation(q_alloc),
            "q_rank_vs_delta_cpc": correlation(q_rank),
        },
        "rank_invariance": {"max_abs_q_rank_m0_minus_m1": float(max(row["rank_invariance_abs_diff"] for row in per_city))},
        "per_city": per_city,
        "per_seed": per_seed,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the post hoc intra-bin mechanism diagnostic")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backbone", choices=["gnn", "mlp", "gravity"], default="gnn")
    args = parser.parse_args()
    result = run_diagnostic(args.data_root, args.output, args.device, args.backbone)
    print(json.dumps(result["correlations"], indent=2))
    print(json.dumps(result["rank_invariance"], indent=2))
