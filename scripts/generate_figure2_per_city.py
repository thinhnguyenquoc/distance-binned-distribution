"""
Generate Figure 2: Ordered per-city Delta CPC plot for 50 test cities.

Canonical values from frozen K=8 benchmark:
- Mean Delta CPC: +0.00354
- 95% CI: [+0.0026, +0.0045]
- Positive cities: 45 / 50 (90.0%)
- Negative cities: 5 / 50 (10.0%)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import shutil

def generate_figure2():
    # 1. Load canonical K=8 per-city data
    csv_path = Path("results/k_sensitivity_v1/k_sensitivity_per_city.csv")
    df = pd.read_csv(csv_path)
    df_k8 = df[df["K"] == 8].copy()

    assert len(df_k8) == 50, f"Expected 50 cities, found {len(df_k8)}"

    # Format city names: replace underscore with space
    df_k8["city_display"] = df_k8["city"].str.replace("_", " ")

    # Sort ascending by delta_cpc
    df_sorted = df_k8.sort_values(by="delta_cpc").reset_index(drop=True)

    cities = df_sorted["city_display"].values
    deltas = df_sorted["delta_cpc"].values
    n_cities = len(deltas)

    # 2. Compute canonical aggregates
    mean_delta = float(np.mean(deltas))
    median_delta = float(np.median(deltas))
    n_pos = int(np.sum(deltas > 0))
    n_neg = int(np.sum(deltas <= 0))

    # 3. Setup styling (Nature/Science publication style)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "figure.titlesize": 12,
        "figure.dpi": 300,
    })

    fig, ax = plt.subplots(figsize=(11, 4.8))

    # Colors: Professional blue for positive, muted red for negative
    pos_color = "#2b5c8f"  # deep steel blue
    neg_color = "#c44e52"  # muted coral red
    colors = [neg_color if d < 0 else pos_color for d in deltas]

    # Bar plot with subtle edge
    bars = ax.bar(
        range(n_cities),
        deltas,
        color=colors,
        width=0.72,
        edgecolor=[c if d < 0 else "#1d4168" for c, d in zip(colors, deltas)],
        linewidth=0.5,
        zorder=3
    )

    # Horizontal zero line
    ax.axhline(0, color="#333333", linewidth=0.9, linestyle="-", zorder=4)

    # Horizontal mean line
    ax.axhline(
        mean_delta,
        color="#1b7837",
        linewidth=1.2,
        linestyle="--",
        label=f"Mean $\\Delta\\mathrm{{CPC}} = +0.00354$",
        zorder=4
    )

    # Horizontal median line
    ax.axhline(
        median_delta,
        color="#e08214",
        linewidth=1.0,
        linestyle=":",
        label=f"Median $\\Delta\\mathrm{{CPC}} = +0.00195$",
        zorder=4
    )

    # X-axis configuration
    ax.set_xticks(range(n_cities))
    ax.set_xticklabels(cities, rotation=90, ha="center", va="top")
    ax.set_xlim(-0.8, n_cities - 0.2)

    # Y-axis configuration
    ax.set_ylabel(r"City-level $\Delta\mathrm{CPC}_c$ ($\mathrm{M}_1 - \mathrm{M}_0$)", fontsize=10)
    ax.set_ylim(-0.004, 0.0185)
    ax.yaxis.grid(True, linestyle=":", alpha=0.6, color="#cccccc", zorder=0)
    ax.xaxis.grid(False)

    # Annotation badge: 45/50 cities improved (subtle styling)
    ax.text(
        0.82, 0.90,
        "45 / 50 cities improved (90.0%)",
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=8.5, color="#2b5c8f",
        bbox=dict(boxstyle="round,pad=0.3", fc="#f7fafd", ec="#c5d8eb", lw=0.6, alpha=0.9)
    )

    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#e0e0e0", framealpha=0.95)

    ax.set_title(
        "City-level improvement in interzonal CPC from oracle target-distance calibration",
        loc="left", pad=10, fontsize=11, fontweight="bold"
    )

    fig.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.25)

    # Save to paper/figures
    out_dir = Path("paper/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    png_path = out_dir / "fig2_per_city_delta_cpc.png"
    pdf_path = out_dir / "fig2_per_city_delta_cpc.pdf"

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"[OK] Figure 2 saved to:\n  - {png_path}\n  - {pdf_path}")

    # Also copy to artifact directory for markdown embedding if needed
    art_dir = Path(r"C:\Users\Thinh Nguyen\.gemini\antigravity\brain\c22018c8-8567-469c-8d56-a163dbb39813")
    if art_dir.exists():
        art_png = art_dir / "fig2_per_city_delta_cpc.png"
        shutil.copy2(png_path, art_png)
        print(f"[OK] Copied to artifact dir: {art_png}")

if __name__ == "__main__":
    generate_figure2()
