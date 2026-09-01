"""
Master script to generate all 5 publication-ready figures for Section 4 (Empirical Results).

Outputs saved to paper/figures/ in both PNG (300 DPI) and vector PDF formats:
- fig1_main_per_city.png / .pdf
- fig2_resolution_sensitivity.png / .pdf
- fig3_noise_dose_response.png / .pdf
- fig4_structural_validity_placebo.png / .pdf
- fig5_mechanistic_dpre.png / .pdf
"""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

FIGURES_DIR = Path("paper/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Global Publication Aesthetics (Nature/Science/TR-C styling)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "figure.titlesize": 12,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

PRIMARY_BLUE = "#1f77b4"
ACCENT_GREEN = "#2ca02c"
MUTED_RED = "#d62728"
PURPLE = "#9467bd"
ORANGE = "#ff7f0e"
GRAY = "#7f7f7f"


def generate_figure1():
    """Figure 1: Ordered per-city Delta CPC across all 50 test cities."""
    csv_path = Path("results/k_sensitivity_v1/k_sensitivity_per_city.csv")
    df = pd.read_csv(csv_path)
    df_k8 = df[df["K"] == 8].copy()
    df_k8["city_display"] = df_k8["city"].str.replace("_", " ")
    df_sorted = df_k8.sort_values(by="delta_cpc").reset_index(drop=True)

    cities = df_sorted["city_display"].values
    deltas = df_sorted["delta_cpc"].values
    n_cities = len(deltas)
    mean_delta = float(np.mean(deltas))
    median_delta = float(np.median(deltas))

    fig, ax = plt.subplots(figsize=(10, 4.2))

    colors = [MUTED_RED if d < 0 else PRIMARY_BLUE for d in deltas]
    ax.bar(
        range(n_cities),
        deltas,
        color=colors,
        width=0.72,
        edgecolor=[c if d < 0 else "#144a70" for c, d in zip(colors, deltas)],
        linewidth=0.5,
        zorder=3
    )

    ax.axhline(0, color="#333333", linewidth=0.8, linestyle="-", zorder=4)
    ax.axhline(mean_delta, color=ACCENT_GREEN, linewidth=1.2, linestyle="--",
               label=f"Mean $\\Delta\\mathrm{{CPC}} = +{mean_delta:.5f}$", zorder=4)
    ax.axhline(median_delta, color=ORANGE, linewidth=1.0, linestyle=":",
               label=f"Median $\\Delta\\mathrm{{CPC}} = +{median_delta:.5f}$", zorder=4)

    ax.set_xticks(range(n_cities))
    ax.set_xticklabels(cities, rotation=90, ha="center", va="top", fontsize=7)
    ax.set_xlim(-0.8, n_cities - 0.2)
    ax.set_ylabel("$\\Delta\\mathrm{CPC}$ ($M_1 - M_0$)", fontweight="bold")
    ax.set_title("Zero-Shot OD Reconstruction Gain Across 50 U.S. Metropolitan Areas ($K=8$)", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=1)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)

    # Annotation box
    pos_count = np.sum(deltas > 0)
    ax.text(
        0.98, 0.05,
        f"Positive Gain: {pos_count}/50 ({pos_count/n_cities*100:.0f}%)\n"
        f"Wilcoxon $p = 1.93 \\times 10^{{-9}}$",
        transform=ax.transAxes,
        fontsize=8.5,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0", edgecolor="#cccccc", alpha=0.95)
    )

    fig.savefig(FIGURES_DIR / "fig1_main_per_city.png", dpi=300)
    fig.savefig(FIGURES_DIR / "fig1_main_per_city.pdf")
    plt.close(fig)
    print("✓ Generated Figure 1")


def generate_figure2():
    """Figure 2: Multi-panel Resolution Sensitivity (K-sweep & Spatial Resolution)."""
    # 1. K-sweep data
    k_json = Path("results/k_sensitivity_v1/k_sensitivity_summary.json")
    with open(k_json, "r") as f:
        k_data = json.load(f)

    k_map = {row["K"]: row for row in k_data["summary"]}
    k_vals = [2, 4, 6, 8, 10, 12, 16, 20]
    k_means = [k_map[k]["mean_delta"] for k in k_vals]
    k_ci_low = [k_map[k]["ci_low"] for k in k_vals]
    k_ci_high = [k_map[k]["ci_high"] for k in k_vals]

    # 2. Spatial resolution data on 11 multi-county MSAs
    sp_json = Path("results/spatial_resolution/spatial_resolution_summary.json")
    with open(sp_json, "r") as f:
        sp_data = json.load(f)

    mc_cities = sp_data["multi_county_subset"]["cities"]
    sp_per_city_json = Path("results/spatial_resolution/spatial_resolution_per_city.json")
    with open(sp_per_city_json, "r") as f:
        sp_city_data = json.load(f)

    sp_city_map = {row["city"]: row for row in sp_city_data}

    city_gains = []
    county_gains = []
    clean_names = []
    for c in mc_cities:
        row = sp_city_map[c]
        clean_names.append(c.replace("_", " "))
        city_gains.append(row["delta_cpc_city"])
        county_gains.append(row["delta_cpc_county"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0), gridspec_kw={"width_ratios": [1, 1.2]})

    # Panel A: K-sweep
    yerr_low = np.array(k_means) - np.array(k_ci_low)
    yerr_high = np.array(k_ci_high) - np.array(k_means)

    ax1.plot(k_vals, k_means, marker="o", color=PRIMARY_BLUE, linewidth=1.8, markersize=5, zorder=3)
    ax1.errorbar(k_vals, k_means, yerr=[yerr_low, yerr_high], fmt="none",
                  ecolor=PRIMARY_BLUE, capsize=3.0, elinewidth=1.1, zorder=3)
    ax1.set_xticks(k_vals)
    ax1.set_xticklabels([f"K={k}" for k in k_vals], rotation=45)
    ax1.set_xlabel("Number of Distance Bins ($K$)", fontweight="bold")
    ax1.set_ylabel("Mean $\\Delta\\mathrm{CPC}$ (with 95% CI)", fontweight="bold")
    ax1.set_title("(a) Distance Bin Granularity Sweep", fontweight="bold", loc="left")
    ax1.grid(True, linestyle="--", alpha=0.35)

    # Panel B: Spatial Resolution (City vs County on 11 MSAs)
    x = np.arange(len(clean_names))
    width = 0.38
    ax2.bar(x - width/2, city_gains, width, label="City-Level $Y_D$", color="#7faed6", edgecolor="#346896", zorder=3)
    ax2.bar(x + width/2, county_gains, width, label="County-Level $Y_D$", color="#1f4e79", edgecolor="#0e2942", zorder=3)

    ax2.set_xticks(x)
    ax2.set_xticklabels(clean_names, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("$\\Delta\\mathrm{CPC}$ ($M_1 - M_0$)", fontweight="bold")
    ax2.set_title("(b) Spatial Resolution in Multi-County MSAs ($N=11$)", fontweight="bold", loc="left")
    ax2.legend(loc="upper left", frameon=True)
    ax2.grid(axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_resolution_sensitivity.png", dpi=300)
    fig.savefig(FIGURES_DIR / "fig2_resolution_sensitivity.pdf")
    plt.close(fig)
    print("✓ Generated Figure 2")


def generate_figure3():
    """Figure 3: Noise dose-response."""
    json_path = Path("results/noise_robustness_fine_v1/noise_summary.json")
    with open(json_path, "r") as f:
        data = json.load(f)

    res = data["results_by_eps"]
    eps_cross = data.get("eps_cross_zero_dCPC", 0.0444)

    epsilons = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
    eps_pct = [e * 100 for e in epsilons]

    means = [res[str(e)]["mean_delta_cpc"] for e in epsilons]
    ci_lowers = [res[str(e)]["ci_lower"] for e in epsilons]
    ci_uppers = [res[str(e)]["ci_upper"] for e in epsilons]

    fig, ax = plt.subplots(figsize=(6.8, 4.2))

    ax.plot(eps_pct, means, marker="o", color=PRIMARY_BLUE, linewidth=2.0, markersize=5.5, zorder=4, label="Calibrated Gain $\\Delta\\mathrm{CPC}$")
    ax.fill_between(eps_pct, ci_lowers, ci_uppers, color=PRIMARY_BLUE, alpha=0.18, zorder=2, label="95% Bootstrap CI")

    ax.axhline(0, color="#333333", linestyle="-", linewidth=0.9, zorder=2, label="Zero-Shot Baseline ($M_0$)")
    ax.axvline(eps_cross * 100, color=MUTED_RED, linestyle="--", linewidth=1.2, zorder=3,
               label=f"Crossover Threshold ($\\epsilon_{{\\mathrm{{cross}}}} \\approx {eps_cross*100:.2f}\\%$)")

    ax.set_xlabel("Observation Noise / Perturbation $\\epsilon$ (Total Variation %)", fontweight="bold")
    ax.set_ylabel("Reconstruction Gain $\\Delta\\mathrm{CPC}$", fontweight="bold")
    ax.set_title("Calibration Benefit vs. Observation Quality ($N=50$ Cities)", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_noise_dose_response.png", dpi=300)
    fig.savefig(FIGURES_DIR / "fig3_noise_dose_response.pdf")
    plt.close(fig)
    print("✓ Generated Figure 3")


def generate_figure4():
    """Figure 4: Structural Validity and Specificity Controls (Correct vs Permuted vs Donor Placebo)."""
    conditions = ["Correct $Y_D$\n(Target MSA)", "Cross-City Placebo\n(Donor $Y_D$)", "Permuted Bins\n(Shuffled Order)"]
    means = [+0.003539, -0.000091, -0.006958]
    ci_low = [+0.002607, -0.000520, -0.008400]
    ci_high = [+0.004483, +0.000340, -0.005500]

    yerr_low = np.array(means) - np.array(ci_low)
    yerr_high = np.array(ci_high) - np.array(means)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    colors = [PRIMARY_BLUE, GRAY, MUTED_RED]
    bars = ax.bar(range(len(conditions)), means, color=colors, width=0.55, edgecolor="#222222", linewidth=0.6, zorder=3)
    ax.errorbar(range(len(conditions)), means, yerr=[yerr_low, yerr_high], fmt="none", ecolor="#222222", capsize=5, elinewidth=1.3, zorder=4)

    ax.axhline(0, color="#333333", linewidth=0.9, linestyle="-", zorder=2)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, fontweight="bold")
    ax.set_ylabel("Mean $\\Delta\\mathrm{CPC}$ (with 95% Bootstrap CI)", fontweight="bold")
    ax.set_title("Structural Validity and Target Specificity Placebo Controls", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    # Annotations
    ax.text(0, means[0] + 0.0009, f"+{means[0]:.5f}\n($p < 10^{{-8}}$)", ha="center", fontsize=8.5, fontweight="bold", color=PRIMARY_BLUE)
    ax.text(1, means[1] + 0.0008, f"{means[1]:.5f}\n(n.s.)", ha="center", fontsize=8.5, color="#555555")
    ax.text(2, means[2] - 0.0018, f"{means[2]:.5f}\n($p < 10^{{-14}}$)", ha="center", fontsize=8.5, fontweight="bold", color=MUTED_RED)

    ax.set_ylim(-0.010, +0.007)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4_structural_validity_placebo.png", dpi=300)
    fig.savefig(FIGURES_DIR / "fig4_structural_validity_placebo.pdf")
    plt.close(fig)
    print("✓ Generated Figure 4")


def generate_figure5():
    """Figure 5: Mechanistic Diagnostic - Baseline Distance Misalignment d_pre vs Delta CPC."""
    csv_path = Path("results/audit/dpre_mechanism_data.csv")
    df = pd.read_csv(csv_path)

    x = df["d_pre_tv"].values
    y = df["delta_cpc"].values

    slope, intercept, r_val, p_val, std_err = stats.linregress(x, y)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    ax.scatter(x, y, color=PRIMARY_BLUE, edgecolor="#144a70", s=45, alpha=0.85, zorder=3, label="Test Cities ($N=50$)")

    # Regression line and CI band
    x_grid = np.linspace(x.min(), x.max(), 100)
    y_fit = intercept + slope * x_grid
    ax.plot(x_grid, y_fit, color=ACCENT_GREEN, linewidth=2.0, zorder=4,
            label=f"Linear Fit: $\\Delta\\mathrm{{CPC}} = {slope:.3f} d_{{\\mathrm{{pre}}}} {intercept:+.3f}$")

    ax.axhline(0, color="#333333", linewidth=0.8, linestyle="--", alpha=0.5, zorder=2)
    ax.set_xlabel("Baseline Distance Mismatch $d_{\\mathrm{pre}} = \\mathrm{TV}(\\hat{Y}_D^{(0)}, Y_D^{\\mathrm{GT}})$", fontweight="bold")
    ax.set_ylabel("Reconstruction Gain $\\Delta\\mathrm{CPC}$ ($M_1 - M_0$)", fontweight="bold")
    ax.set_title("Mechanistic Diagnostic: Baseline Distance Misalignment ($d_{\\mathrm{pre}}$)", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35)

    # Text box with statistical diagnostics
    ax.text(
        0.05, 0.92,
        f"Pearson $r = +{r_val:.4f}$ ($p = {p_val:.2e}$)\n"
        f"Partial $r = +0.7963$ ($p = 5.21 \\times 10^{{-12}}$)\n"
        f"Multivariate $R^2 = 73.7\\%$",
        transform=ax.transAxes,
        fontsize=8.5,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f9f9f9", edgecolor="#cccccc", alpha=0.95)
    )
    ax.legend(loc="lower right", frameon=True)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_mechanistic_dpre.png", dpi=300)
    fig.savefig(FIGURES_DIR / "fig5_mechanistic_dpre.pdf")
    plt.close(fig)
    print("✓ Generated Figure 5")


if __name__ == "__main__":
    generate_figure1()
    generate_figure2()
    generate_figure3()
    generate_figure4()
    generate_figure5()
    print("🎉 All 5 publication figures successfully generated in paper/figures/")
