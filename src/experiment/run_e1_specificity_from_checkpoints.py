"""
Canonical E1-v2 9-donor specificity runner using frozen GNN checkpoints.

This runner evaluates the E1-v2 target-vs-wrong-donor specificity estimand
without retraining. It loads the 15 canonical GNN checkpoints from
results/checkpoints/5fold_fold{fold}_seed{seed}.pt, averages seeds within city,
and then applies the existing E1 city-level statistical summary.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.experiment import run_e1 as e1
from src.training.train import load_checkpoint


CANONICAL_SEEDS = [1, 10, 100]
DEFAULT_OUTPUT_DIR = Path("results/e1_canonical_specificity_v2")


def _mean_numeric(seed_results: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([r[key] for r in seed_results]))


def _average_city_seed_results(seed_results: list[dict[str, Any]], seeds: list[int]) -> dict[str, Any]:
    first = seed_results[0]
    averaged = {
        "city": first["city"],
        "fold": first["fold"],
        "donor_city": "all_9_fold_donors",
        "n_wrong_donors": first["n_wrong_donors"],
        "n_inter_pairs": first["n_inter_pairs"],
        "K_active": first["K_active"],
        "model_seeds": seeds,
        "cpc_baseline": _mean_numeric(seed_results, "cpc_baseline"),
        "cpc_baseline_norm": _mean_numeric(seed_results, "cpc_baseline_norm"),
        "cpc_target_yd": _mean_numeric(seed_results, "cpc_target_yd"),
        "cpc_target_yd_norm": _mean_numeric(seed_results, "cpc_target_yd_norm"),
        "delta_cpc_target": _mean_numeric(seed_results, "delta_cpc_target"),
        "cpc_wrong_yd": _mean_numeric(seed_results, "cpc_wrong_yd"),
        "cpc_wrong_yd_norm": _mean_numeric(seed_results, "cpc_wrong_yd_norm"),
        "delta_cpc_wrong": _mean_numeric(seed_results, "delta_cpc_wrong"),
        "delta_cpc_specificity": _mean_numeric(seed_results, "delta_cpc_specificity"),
        "Y_D_target": first["Y_D_target"],
        "wrong_donor_breakdown_by_seed": {
            str(seed): result["wrong_donor_breakdown"]
            for seed, result in zip(seeds, seed_results)
        },
    }
    return averaged


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_existing_completed(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not path.exists():
        return [], []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("per_city_seed_averaged", []), payload.get("per_city_per_seed", [])


def run_e1_specificity_from_checkpoints(
    data_root: str = "data",
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    folds: list[int] | None = None,
    seeds: list[int] | None = None,
    device: str = "cpu",
    smoke: bool = False,
    smoke_cities: int = 1,
    resume: bool = False,
) -> dict[str, Any]:
    if folds is None:
        folds = [1, 2, 3, 4, 5]
    if seeds is None:
        seeds = CANONICAL_SEEDS.copy()

    if seeds != CANONICAL_SEEDS:
        raise ValueError(f"E1 canonical specificity requires seeds {CANONICAL_SEEDS}, got {seeds}")

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "e1_specificity_results.json"
    tables_dir = output_dir / "tables"

    e1.DATA_ROOT = data_root
    e1.MANIFEST_PATH = Path("results/e1/splits_manifest_v2.json")
    splits = e1.load_splits_manifest_v2(str(e1.MANIFEST_PATH), data_root=data_root)
    with open(e1.MANIFEST_PATH, "r", encoding="utf-8") as manifest_file:
        split_manifest_sha256 = json.load(manifest_file)["manifest_sha256"]

    all_averaged, raw_seed_results = _load_existing_completed(results_path) if resume else ([], [])
    completed = {(r["fold"], r["city"]) for r in all_averaged}

    start = time.time()
    for fold_id in folds:
        split = splits[fold_id]
        train_cities = split["train"]
        test_cities = sorted(split["test"])
        run_cities = test_cities[:smoke_cities] if smoke else test_cities

        print(f"\n>>> [E1 canonical specificity] fold {fold_id}/5 | cities={len(run_cities)}/{len(test_cities)} | seeds={seeds}")
        bin_edges, k_active = e1.compute_kbin_edges(train_cities, K=e1.K_MOVE, data_root=data_root)
        if k_active != e1.K_MOVE:
            raise RuntimeError(f"Expected K_active={e1.K_MOVE}, got {k_active} for fold {fold_id}")

        models = {}
        for seed in seeds:
            ckpt_path = Path("results/checkpoints") / f"5fold_fold{fold_id}_seed{seed}.pt"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Missing mandatory canonical GNN checkpoint: {ckpt_path}")
            model, scaler, metadata = load_checkpoint(ckpt_path, device_str=device)
            hp = metadata.get("hyperparams", {})
            if metadata.get("seed") != seed or hp.get("fold") != fold_id:
                raise RuntimeError(f"Checkpoint provenance mismatch: {ckpt_path}")
            if hp.get("split_manifest_sha256") != split_manifest_sha256:
                raise RuntimeError(f"Split manifest mismatch in checkpoint: {ckpt_path}")
            model.eval()
            models[seed] = (model, scaler)

        for city in run_cities:
            if (fold_id, city) in completed:
                print(f"  -> Reusing saved city result: {city}")
                continue

            city_seed_results = []
            for seed in seeds:
                model, scaler = models[seed]
                result = e1.run_city(
                    city=city,
                    model=model,
                    scaler=scaler,
                    bin_edges=bin_edges,
                    K_active=k_active,
                    test_cities=test_cities,
                    fold_id=fold_id,
                    device=device,
                )
                result["model_seed"] = seed
                raw_seed_results.append(result)
                city_seed_results.append(result)

            averaged = _average_city_seed_results(city_seed_results, seeds)
            all_averaged.append(averaged)
            completed.add((fold_id, city))
            print(
                f"  -> {city:<16} M0={averaged['cpc_baseline']:.4f} "
                f"target_d={averaged['delta_cpc_target']:+.4f} "
                f"wrong9_d={averaged['delta_cpc_wrong']:+.4f} "
                f"specificity={averaged['delta_cpc_specificity']:+.4f}"
            )

            summary = e1.compute_summary(all_averaged, bootstrap_seed=2024)
            _write_json(results_path, {
                "protocol": "e1-v2-canonical-9-donor-specificity-from-checkpoints",
                "checkpoint_source": "results/checkpoints/5fold_fold{fold}_seed{seed}.pt",
                "seeds": seeds,
                "folds": folds,
                "smoke": smoke,
                "elapsed_sec": time.time() - start,
                "summary": summary,
                "per_city_seed_averaged": all_averaged,
                "per_city_per_seed": raw_seed_results,
            })

    summary = e1.compute_summary(all_averaged, bootstrap_seed=2024)
    e1.write_tables(all_averaged, summary, table_dir=tables_dir)
    payload = {
        "protocol": "e1-v2-canonical-9-donor-specificity-from-checkpoints",
        "checkpoint_source": "results/checkpoints/5fold_fold{fold}_seed{seed}.pt",
        "seeds": seeds,
        "folds": folds,
        "smoke": smoke,
        "elapsed_sec": time.time() - start,
        "summary": summary,
        "per_city_seed_averaged": all_averaged,
        "per_city_per_seed": raw_seed_results,
    }
    _write_json(results_path, payload)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run E1-v2 9-donor specificity on canonical frozen GNN checkpoints")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-cities", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    run_e1_specificity_from_checkpoints(
        data_root=args.data_root,
        output_dir=args.output_dir,
        folds=args.folds,
        device=args.device,
        smoke=args.smoke,
        smoke_cities=args.smoke_cities,
        resume=args.resume,
    )