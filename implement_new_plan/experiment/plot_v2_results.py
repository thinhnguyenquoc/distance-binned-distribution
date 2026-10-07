"""Plot descriptive results from the completed v2 source-city outputs.

The figures deliberately use seed-averaged source-target rows; each of the
2,450 transfer pairs receives one vote. Confirmatory mixed-effects inference
is separate and may remain unavailable when the optimizer has not converged.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2] / "results" / "new_plan_v2_focused"
OUT = ROOT / "figures"
COLORS = {"urban_gnn": "#1764a3", "pairwise_mlp": "#d47d24", "gravity_2param": "#818991"}
LABELS = {"urban_gnn": "Urban GNN", "pairwise_mlp": "Pairwise MLP", "gravity_2param": "Gravity"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    main_rows = pd.read_csv(ROOT / "main_transfer_metrics_mean.csv")
    scarcity = pd.read_csv(ROOT / "scarcity_transfer_metrics_mean.csv")
    resolution = pd.read_csv(ROOT / "resolution_metrics_mean.csv")
    noise = pd.read_csv(ROOT / "noise_metrics_mean.csv")
    specificity = pd.read_csv(ROOT / "specificity_metrics_mean.csv")

    if len(main_rows) != 7350 or len(scarcity) != 19600 or len(specificity) != 2450:
        raise ValueError("Expected complete seed-averaged 50 x 49 v2 result tables")
    for frame in (main_rows, scarcity, resolution, noise, specificity):
        if frame.select_dtypes(include="number").isna().any().any():
            raise ValueError("Missing numeric result in seed-averaged v2 table")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9), constrained_layout=True)

    # A: raw CPC before and after calibration; gravity is visible close to zero.
    ax = axes[0, 0]
    order = ["urban_gnn", "pairwise_mlp", "gravity_2param"]
    grouped = main_rows.groupby("model")[["CPC_before", "CPC_after"]].mean().loc[order]
    x = np.arange(len(order))
    width = 0.32
    ax.bar(x - width / 2, grouped["CPC_before"], width, label="Zero-shot", color="#a9bfcc")
    ax.bar(x + width / 2, grouped["CPC_after"], width, label="After DBD", color="#1764a3")
    for i, model in enumerate(order):
        ax.text(x[i] + width / 2, grouped.loc[model, "CPC_after"] + 0.003,
                f"+{grouped.loc[model, 'CPC_after'] - grouped.loc[model, 'CPC_before']:.4f}",
                ha="center", fontsize=9)
    ax.set_xticks(x, [LABELS[m] for m in order])
    ax.set_ylim(0, 0.165)
    ax.set_ylabel("Mean CPC across 2,450 transfers")
    ax.set_title("A. Zero-shot transfer at 40% source training, K=8")
    ax.legend(frameon=False)

    # B: source scarcity, emphasizing changes on the same fixed target support.
    ax = axes[0, 1]
    by_fraction = scarcity.groupby(["model", "train_fraction"])["delta_CPC"].mean().unstack(0)
    model = "urban_gnn"
    ax.plot(100 * by_fraction.index, by_fraction[model], marker="o", linewidth=2,
            color=COLORS[model], label=LABELS[model])
    for fraction, value in by_fraction[model].items():
        ax.annotate(f"{value:.5f}", (100 * fraction, value), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks([10, 20, 30, 40])
    ax.set_ylim(0.01244, 0.01256)
    ax.set_xlabel("Source training OD (%)")
    ax.set_ylabel("Mean CPC gain after DBD")
    ax.set_title("B. GNN gain changes slightly across train fractions")

    ax = axes[1, 0]
    k_gain = resolution.groupby("K")["delta_CPC"].mean()
    ax.plot(k_gain.index, k_gain.values, marker="o", linewidth=2, color=COLORS["urban_gnn"], label="Distance bins K")
    ax.set_xticks([4, 8, 12])
    ax.set_xlabel("Number of distance bins, K")
    ax.set_ylabel("Mean GNN CPC gain")
    ax.set_title("C1. DBD resolution at 40% source training")

    ax = axes[1, 1]
    e_gain = noise.groupby("epsilon")["delta_CPC"].mean()
    ax.plot(e_gain.index, e_gain.values, marker="o", linewidth=2, color="#8b4eb2")
    ax.set_xticks([0, 0.05, 0.10])
    ax.set_xlabel("Target DBD perturbation (total variation)")
    ax.set_ylabel("Mean GNN CPC gain")
    ax.set_title("C2. Calibration remains useful with noisy DBD")

    fig.suptitle("DBD v2: descriptive results across 50 source and 49 target cities", fontsize=15, fontweight="bold")
    fig.savefig(OUT / "dbd_v2_main_and_robustness.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.7), constrained_layout=True)
    ax = axes[0]
    gain = specificity[["delta_CPC", "delta_CPC_donor_control"]].mean()
    ax.bar(["Target DBD", "Dose-matched donor"], gain.values,
           color=[COLORS["urban_gnn"], "#a9bfcc"])
    for i, value in enumerate(gain.values):
        ax.text(i, value + 0.0002, f"{value:.4f}", ha="center")
    ax.set_ylim(0, 0.0145)
    ax.set_ylabel("Mean GNN CPC gain")
    ax.set_title("D. Target information vs donor control")

    ax = axes[1]
    main_gnn = main_rows.loc[main_rows.model == "urban_gnn"]
    target_gain = main_gnn.groupby("target_city")["delta_CPC"].mean().sort_values()
    ax.bar(np.arange(len(target_gain)), target_gain.values, color=COLORS["urban_gnn"], width=0.9)
    ax.axhline(target_gain.mean(), color="#d47d24", linewidth=1.5, linestyle="--",
               label=f"Mean {target_gain.mean():.4f}")
    ax.set_xlabel("50 target cities, sorted by mean gain")
    ax.set_ylabel("Mean GNN CPC gain")
    ax.set_title("A. All 50 target-city means are positive")
    ax.legend(frameon=False)
    fig.suptitle("DBD v2: specificity and variation between target cities", fontsize=14, fontweight="bold")
    fig.savefig(OUT / "dbd_v2_specificity_and_targets.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
