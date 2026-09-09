"""Aggregate independently completed frozen-checkpoint noise fold outputs."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from src.experiment.run_noise_robustness import generate_summary


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--noise-seed", type=int, required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_frames = [
        pd.read_csv(input_dir / f"fold_{fold}" / "noise_raw.csv")
        for fold in range(1, 6)
    ]
    raw = pd.concat(raw_frames, ignore_index=True)
    raw.to_csv(output_dir / "noise_raw.csv", index=False)

    seed = raw.groupby(["fold", "target_city", "model_seed", "epsilon"], as_index=False).agg(
        delta_cpc_inter=("delta_cpc_inter", "mean"),
        degradation=("degradation", "mean"),
        w_max=("w_max", "mean"),
        w_gt_2=("w_gt_2", "mean"),
        cpc_m1_inter=("cpc_m1_inter", "mean"),
        prob_positive=("delta_cpc_inter", lambda values: float((values > 0).mean())),
    )
    seed.to_csv(output_dir / "noise_per_seed.csv", index=False)

    city = seed.groupby(["fold", "target_city", "epsilon"], as_index=False).agg(
        delta_cpc_mean=("delta_cpc_inter", "mean"),
        degradation_mean=("degradation", "mean"),
        prob_positive=("prob_positive", "mean"),
        cpc_m1_inter=("cpc_m1_inter", "mean"),
        w_max=("w_max", "mean"),
        w_gt_2=("w_gt_2", "mean"),
    )
    city.to_csv(output_dir / "noise_per_city.csv", index=False)

    epsilons = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
    generate_summary(city, str(output_dir), epsilons, epsilons[1:], 1000)

    checkpoint_dir = Path(args.checkpoint_dir)
    split_manifest = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
    checkpoint_hashes = {
        path.name: sha256_file(path)
        for path in sorted(checkpoint_dir.glob("5fold_fold*_seed*.pt"))
    }
    manifest = {
        "experiment_version": "noise_robustness_frozen_v1",
        "git_commit": os.environ.get("GIT_COMMIT", "unknown"),
        "source_fold_outputs": str(input_dir),
        "checkpoint_sha256": checkpoint_hashes,
        "split_manifest": str(args.split_manifest),
        "split_manifest_sha256": sha256_file(Path(args.split_manifest)),
        "split_manifest_content_sha256": split_manifest["manifest_sha256"],
        "data_root": "data",
        "data_version": "repository data/ snapshot used by the frozen five-fold run",
        "K": 8,
        "folds": [1, 2, 3, 4, 5],
        "cities": 50,
        "model_seeds": [1, 10, 100],
        "epsilons": epsilons,
        "noise_seed": args.noise_seed,
        "replicates_at_epsilon_zero": 1,
        "replicates_at_positive_epsilon": 1000,
        "support_definition": "active target Y_D bins only; support unchanged",
        "noise_definition": "multiplicative compositional noise with exact TV matching by bisection",
        "aggregation_order": "replicate -> model seed -> city -> macro mean",
        "no_post_hoc_offset_or_reconciliation": True,
    }
    with open(output_dir / "noise_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


if __name__ == "__main__":
    main()