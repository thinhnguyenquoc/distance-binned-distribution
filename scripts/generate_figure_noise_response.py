"""
Generate Figure for Section 4.3.3: Observation Quality and Noise Dose-Response.

Output files:
- paper/figures/fig3_noise_dose_response.png
- paper/figures/fig3_noise_dose_response.pdf
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import os
from pathlib import Path

# Ensure writable matplotlib temp config dir
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_config"

def generate_figure():
    # 1. Load canonical data
    summary_json_path = Path("results/noise_robustness_fine_v1/noise_summary.json")
    with open(summary_json_path, "r") as f:
        data = json.load(f)

    res = data["results_by_eps"]
    eps_cross = data.get("eps_cross_zero_dCPC", 0.0444)

    epsilons = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
    eps_pct = [e * 100 for e in epsilons]

    means = [res[str(e)]["mean_delta_cpc"] for e in epsilons]
    ci_lowers = [res[str(e)]["ci_lower"] for e in epsilons]
    ci_uppers = [res[str(e)]["ci_upper"] for e in epsilons]
    pos_rates = [(res[str(e)]["pos_cities"] / 50.0) * 100 for e in epsilons]

    # Setup publication styling (Nature/Science style)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "figure.dpi": 300,
    })

    fig, ax1 = plt.subplots(figsize=(7.5, 4.4))

    # Primary axis: Delta CPC
    color_cpc = "#2b5c8f"
    ax1.plot(eps_pct, means, marker="o", markersize=6, color=color_cpc, linewidth=1.8, label="Mean $\\Delta\\mathrm{CPC}$ (50 cities)", zorder=5)
    
    # Error band (95% CI)
    ax1.fill_between(eps_pct, ci_lowers, ci_uppers, color=color_cpc, alpha=0.18, label="95% Bootstrap CI", zorder=3)

    # Reference zero line (M0 baseline)
    ax1.axhline(0, color="#333333", linestyle="-", linewidth=0.9, zorder=2)

    # Crossover vertical line
    cross_pct = eps_cross * 100
    ax1.axvline(cross_pct, color="#d95f02", linestyle="--", linewidth=1.3, label=f"Crossover $\\epsilon_{{\\mathrm{{cross}}}} = {cross_pct:.2f}\\%$ TV", zorder=4)

    # Secondary axis: Win Rate (%)
    ax2 = ax1.twinx()
    color_rate = "#7570b3"
    ax2.plot(eps_pct, pos_rates, marker="s", markersize=5, color=color_rate, linestyle=":", linewidth=1.5, label="Positive Cities (%)", zorder=4)
    ax2.set_ylabel("Positive Cities (%)", color=color_rate, fontsize=10)
    ax2.tick_params(axis='y', labelcolor=color_rate)
    ax2.set_ylim(0, 105)

    # Annotate points on primary line
    for ep, m in zip(eps_pct, means):
        va = "bottom" if m >= 0 else "top"
        offset = 0.00035 if m >= 0 else -0.00035
        ax1.annotate(f"{m:+.4f}", (ep, m + offset), textcoords="data", ha="center", va=va, fontsize=7.5, color="#1d4168", fontweight="semibold")

    # Labels and styling
    ax1.set_xlabel("Observation Perturbation / Noise Level $\\epsilon$ (% Total Variation Error)", fontsize=10)
    ax1.set_ylabel("Interzonal Flow Gain ($\\Delta\\mathrm{CPC}$)", color=color_cpc, fontsize=10)
    ax1.tick_params(axis='y', labelcolor=color_cpc)
    ax1.set_xlim(-0.3, 5.3)
    ax1.set_ylim(-0.0022, 0.0055)
    ax1.grid(True, linestyle=":", alpha=0.6, zorder=1)

    # Legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", framealpha=0.92, facecolor="white", edgecolor="#cccccc")

    plt.title("Figure 3 | Effect of Observation Fidelity on Calibration Benefit", fontsize=11, fontweight="bold", pad=10)

    plt.tight_layout()

    out_dir = Path("paper/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    png_path = out_dir / "fig3_noise_dose_response.png"
    pdf_path = out_dir / "fig3_noise_dose_response.pdf"

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print(f"Successfully generated: {png_path} and {pdf_path}")

if __name__ == "__main__":
    generate_figure()
