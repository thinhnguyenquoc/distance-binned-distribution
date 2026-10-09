"""Create descriptive figures for the completed source-validation protocol."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2] / "results" / "new_plan_v3_source_validation"
FIGURES = ROOT / "figures"
BLUE = "#1764a3"
ORANGE = "#d47d24"
GREY = "#a9bfcc"


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    main_rows = pd.read_csv(ROOT / "main_transfer_metrics_mean.csv")
    scarcity = pd.read_csv(ROOT / "scarcity_transfer_metrics_mean.csv")
    resolution = pd.read_csv(ROOT / "resolution_summary.csv")
    noise = pd.read_csv(ROOT / "noise_summary.csv")
    specificity = pd.read_csv(ROOT / "specificity_metrics_mean.csv")
    assert len(main_rows) == 7350 and len(scarcity) == 19600 and len(specificity) == 2450

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    order = ["urban_gnn", "pairwise_mlp", "gravity_2param"]
    names = ["GNN", "MLP", "Gravity"]
    mean = main_rows.groupby("model")[["CPC_before", "CPC_after", "R_vol_before"]].mean().loc[order]
    x = np.arange(3)

    a = ax[0, 0]
    a.bar(x - 0.17, mean.CPC_before, 0.34, color=GREY, label="Zero-shot")
    a.bar(x + 0.17, mean.CPC_after, 0.34, color=BLUE, label="After DBD")
    for i, model in enumerate(order):
        a.annotate(f"+{mean.loc[model, 'CPC_after'] - mean.loc[model, 'CPC_before']:.3f}",
                   (i + 0.17, mean.loc[model, "CPC_after"]), xytext=(0, 5),
                   textcoords="offset points", ha="center", fontsize=9)
    a.set_xticks(x, names)
    a.set_ylim(0, 0.5)
    a.set_ylabel("Mean CPC")
    a.set_title("A. Cross-city transfer, 40% train, K=8")
    a.legend(frameon=False)

    a = ax[0, 1]
    gnn = scarcity.loc[scarcity.model == "urban_gnn"].groupby("train_fraction")[
        ["CPC_before", "CPC_after", "delta_CPC"]
    ].mean()
    fractions = 100 * gnn.index.to_numpy()
    a.plot(fractions, gnn.CPC_before, "o-", color=GREY, label="Zero-shot")
    a.plot(fractions, gnn.CPC_after, "o-", color=BLUE, label="After DBD")
    a.set_xticks([10, 20, 30, 40])
    a.set_ylim(0.34, 0.44)
    a.set_xlabel("Source training OD (%)")
    a.set_ylabel("Mean GNN CPC")
    a.set_title("B. Performance across source supervision")
    a.legend(frameon=False)

    a = ax[0, 2]
    a.bar(["GNN", "MLP", "Gravity"], [mean.loc[m, "R_vol_before"] for m in order],
          color=[BLUE, ORANGE, GREY])
    a.axhline(1, color="#444444", linestyle="--", linewidth=1)
    a.set_yscale("log")
    a.set_ylabel("Mean predicted / true total flow (log scale)")
    a.set_title("C. Volume mismatch remains")
    for i, model in enumerate(order):
        a.annotate(f"{mean.loc[model, 'R_vol_before']:.2f}×", (i, mean.loc[model, "R_vol_before"]),
                   xytext=(0, 5), textcoords="offset points", ha="center")

    a = ax[1, 0]
    gains = specificity[["delta_CPC", "delta_CPC_donor_control"]].mean()
    a.bar(["Correct target DBD", "Matched donor DBD"], gains.values, color=[BLUE, GREY])
    for i, value in enumerate(gains.values):
        a.annotate(f"{value:.3f}", (i, value), xytext=(0, 5), textcoords="offset points", ha="center")
    a.set_ylim(0, 0.067)
    a.set_ylabel("Mean GNN CPC gain")
    a.set_title("D. Target specificity control")

    a = ax[1, 1]
    a.plot(resolution.K, resolution["mean"], "o-", color=BLUE)
    a.set_xticks([4, 8, 12])
    a.set_xlabel("Distance bins K")
    a.set_ylabel("Mean GNN CPC gain")
    a.set_title("E. Resolution sensitivity")

    a = ax[1, 2]
    a.plot(noise.epsilon, noise["mean"], "o-", color="#8b4eb2")
    a.set_xticks([0, 0.05, 0.10])
    a.set_xlabel("DBD noise, total variation")
    a.set_ylabel("Mean GNN CPC gain")
    a.set_title("F. Target DBD noise")

    fig.suptitle("DBD v3: 50 source cities × 49 targets (2,450 transfers)", fontsize=15, fontweight="bold")
    fig.savefig(FIGURES / "dbd_v3_results.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
