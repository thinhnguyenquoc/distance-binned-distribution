"""
Convergence Pilot for Zero-Shot OD Backbone (Fold 1)
====================================================

Purpose:
    Examine the optimization trajectory and empirical convergence limit of the
    ZeroShotODModel across 100 epochs on Fold 1 (35 train / 5 val cities).

Logging:
    - epoch (1 to 100)
    - train_loss (ZTNB NLL)
    - val_cpc (Interzonal CPC on 5 validation cities)
    - learning_rate
    - best_epoch & best_val_cpc tracking
    - early_stopping_epoch (if triggered, patience=15)

Outputs:
    - results/e1/convergence_pilot_fold1.json
    - results/e1/tables/convergence_pilot.md
"""

import json
import time
import argparse
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.city_splits import generate_35_5_10_splits
from src.training.train import train_zero_shot_model

MAX_EPOCHS = 100
PATIENCE = 15
DATA_ROOT = "data"
RESULTS_DIR = Path("results/e1")


def run_convergence_pilot(fold_id: int = 1, device_str: str = "cpu"):
    t0 = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "tables").mkdir(exist_ok=True)

    splits = generate_35_5_10_splits(DATA_ROOT)
    split = splits[fold_id]
    train35 = split["train"]
    val5    = split["val"]

    print(f"\n{'='*70}")
    print(f"CONVERGENCE PILOT: FOLD {fold_id} (Max {MAX_EPOCHS} Epochs, Patience={PATIENCE})")
    print(f"{'='*70}")
    print(f"Train ({len(train35)} cities): {train35[:4]}... {train35[-2:]}")
    print(f"Val   ({len(val5)} cities): {val5}")
    print(f"{'-'*70}")

    _ckpt_path = RESULTS_DIR / "checkpoints" / f"convergence_pilot_fold{fold_id}.pt"
    model, scaler, info = train_zero_shot_model(
        train_city_names=train35,
        data_root=DATA_ROOT,
        epochs=MAX_EPOCHS,
        device_str=device_str,
        verbose=True,
        val_city_names=val5,
        patience=PATIENCE,
        min_delta=1e-4,
        return_info=True,
        checkpoint_path=_ckpt_path,
        run_tag=f"convergence_pilot_fold{fold_id}",
    )

    elapsed = time.time() - t0

    # Build per-epoch history table
    history = []
    val_cpcs = info["val_cpc_history"]
    train_losses = info["train_loss_history"]
    n_epochs = len(val_cpcs)

    for ep in range(1, n_epochs + 1):
        history.append({
            "epoch": ep,
            "train_loss": float(train_losses[ep - 1]),
            "val_cpc": float(val_cpcs[ep - 1]),
            "is_best": bool(ep == info["best_epoch"]),
        })

    pilot_results = {
        "fold_id": fold_id,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "epochs_trained": info["epochs_trained"],
        "best_epoch": info["best_epoch"],
        "best_val_cpc": info["best_val_cpc"],
        "stopped_early": info["stopped_early"],
        "elapsed_seconds": elapsed,
        "train_cities": train35,
        "val_cities": val5,
        "checkpoint_path": str(_ckpt_path.resolve()),
        "epoch_history": history,
    }

    # Save JSON artifact
    json_path = RESULTS_DIR / f"convergence_pilot_fold{fold_id}.json"
    json_path.write_text(json.dumps(pilot_results, indent=2), encoding="utf-8")
    print(f"\nSaved raw convergence trajectory -> {json_path}")

    # Generate Markdown summary table
    md_lines = [
        f"# Convergence Pilot Trajectory (Fold {fold_id}, Max {MAX_EPOCHS} Epochs)",
        "",
        f"**Best Epoch**: {info['best_epoch']} | **Best Val CPC**: {info['best_val_cpc']:.4f} | **Total Epochs Trained**: {info['epochs_trained']} | **Early Stopped**: {info['stopped_early']} | **Elapsed Time**: {elapsed:.1f}s",
        "",
        "| Epoch | Train Loss (ZTNB) | Validation CPC (Interzonal) | Status |",
        "|---|---|---|---|",
    ]

    for h in history:
        status_marker = "**BEST CHECKPOINT**" if h["is_best"] else ""
        md_lines.append(f"| {h['epoch']:03d} | {h['train_loss']:.4f} | {h['val_cpc']:.4f} | {status_marker} |")

    table_path = RESULTS_DIR / "tables" / f"convergence_pilot_fold{fold_id}.md"
    table_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Saved Markdown report -> {table_path}")

    print(f"\n{'='*70}")
    print(f"Convergence Pilot Complete in {elapsed:.1f}s")
    print(f"  Best Epoch: {info['best_epoch']} / {info['epochs_trained']}")
    print(f"  Best Validation CPC: {info['best_val_cpc']:.4f}")
    print(f"  Early stopping triggered: {info['stopped_early']}")
    print(f"  Checkpoint saved: {_ckpt_path.resolve()}")
    print(f"{'='*70}\n")


    return pilot_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convergence Pilot for Zero-Shot Model")
    parser.add_argument("--fold", type=int, default=1, help="Fold ID to test (default: 1)")
    parser.add_argument("--device", default="cpu", help="PyTorch device")
    args = parser.parse_args()
    run_convergence_pilot(fold_id=args.fold, device_str=args.device)
