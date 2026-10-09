"""Source-only New York pilot for choosing the epoch limit of a future run.

This pilot uses the v4 train/validation split and never reads its source test
labels or another city's labels.  Its results live outside the v4 protocol.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from implement_new_plan.experiment import run_new_plan_v4_budgeted_validation as v4


OUTPUT = Path("results/new_plan_v4_epoch_pilot_new_york")
FRACTIONS = (0.10, 0.40)
MAX_EPOCHS = 600
PATIENCE = 40


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    v4.TRAINING_CONFIG = dict(v4.TRAINING_CONFIG, max_epochs=MAX_EPOCHS, early_stopping_patience=PATIENCE)
    prepared = v4._prepare_cities(("New_York",))["New_York"]
    preprocessing = v4._build_source_preprocessing("New_York", prepared)
    validation = v4._make_inputs(prepared, preprocessing.scaler, prepared.split["source_validation_indices"])
    print(f"New York source validation: {len(validation.true_flow)} pairs", flush=True)
    for fraction in FRACTIONS:
        path = OUTPUT / f"urban_gnn_f{int(fraction * 100)}_seed1.json"
        if path.exists():
            print(f"Existing pilot result: {path}", flush=True)
            continue
        train = v4._make_inputs(prepared, preprocessing.scaler, prepared.split["train_indices"][fraction])
        print(f"Training GNN fraction={fraction:.0%}, train={len(train.true_flow)}, max={MAX_EPOCHS}", flush=True)
        _, detail = v4._train_with_source_validation("urban_gnn", prepared, train, validation, seed=1)
        result = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "city": "New_York",
            "model": "urban_gnn",
            "fraction": fraction,
            "seed": 1,
            "train_pairs": len(train.true_flow),
            "validation_pairs": len(validation.true_flow),
            "split_seed": v4.SPLIT_SEED,
            "support_hash": prepared.split["support_hash"],
            "scaler_hash": preprocessing.scaler_hash,
            "training_config": v4.TRAINING_CONFIG,
            "training_detail": detail,
        }
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(
            f"fraction={fraction:.0%} best_epoch={detail['best_epoch']} "
            f"epochs_run={detail['epochs_run']} val_loss={detail['best_validation_loss']:.6f} "
            f"stop={detail['stopping_reason']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
