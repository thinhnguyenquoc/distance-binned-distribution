"""Run the complete GNN and MLP experiment for a configurable seed set.

Each model is saved as results/checkpoints/{backbone}_fold{fold}_seed{seed}.pt.
Existing checkpoints are reused only when their protocol provenance matches.
"""

import argparse

from src.experiment.run_5fold import run_5fold_experiment


CANONICAL_SEEDS = [1, 10, 100]
DEFAULT_SEEDS = CANONICAL_SEEDS


def run_full_experiment(
    seeds: list[int],
    folds: list[int],
    epochs: int,
    device: str | None,
    data_root: str,
    output_dir: str,
) -> None:
    common = {
        "data_root": data_root,
        "output_dir": output_dir,
        "epochs_per_fold": epochs,
        "folds_to_run": folds,
        "seeds": seeds,
        "device_str": device,
    }

    print(f"Running GNN for seeds={seeds}, folds={folds}")
    run_5fold_experiment(backbone="gnn", **common)

    print(f"Running MLP for seeds={seeds}, folds={folds}")
    run_5fold_experiment(backbone="mlp", **common)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run full GNN + MLP experiments with reusable seed checkpoints"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must not contain duplicates")
    if any(seed < 0 for seed in args.seeds):
        parser.error("--seeds must be non-negative integers")
    if args.seeds != CANONICAL_SEEDS and args.output_dir == "results":
        parser.error(
            "non-canonical seeds require a separate --output-dir; "
            "use e.g. results/seed_robustness_333_5555_77777"
        )
    if any(fold not in {1, 2, 3, 4, 5} for fold in args.folds):
        parser.error("--folds must contain values from 1 through 5")

    run_full_experiment(
        seeds=args.seeds,
        folds=args.folds,
        epochs=args.epochs,
        device=args.device,
        data_root=args.data_root,
        output_dir=args.output_dir,
    )
