"""
Master script to generate publication-ready empirical figures for interzonal_only dataset.

Outputs saved to results/interzonal_only/artifacts/figures/ (and optionally paper/figures/):
- fig2_main_per_city.png / .pdf
- fig3_structural_validity_placebo.png / .pdf
- fig4_resolution_sensitivity.png / .pdf
- fig5_noise_dose_response.png / .pdf
- fig6_mechanistic_dpre.png / .pdf
- fig_s1_spatial_resolution.png / .pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

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


def generate_figure2(artifacts_dir: Path, figures_dir: Path):
    """Figure 2: Ordered per-city Delta CPC across all 50 test cities."""
    json_path = artifacts_dir / "5fold_results.json"
    with open(json_path, "r", encoding="utf-8") as f:
        d5 = json.load(f)

    df_canonical = pd.DataFrame([
        {
            "city": item["city"],
            "delta_cpc": item["delta_city"],
        }
        for item in d5["city_level_results"]
    ])
    df_canonical["city_display"] = df_canonical["city"].str.replace("_", " ")
    df_sorted = df_canonical.sort_values(by="delta_cpc").reset_index(drop=True)

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
    ax.set_xticklabels(cities, rotation=90, ha="center", va="top", fontsize=8)
    ax.set_xlim(-0.8, n_cities - 0.2)
    ax.set_ylabel(r"$\Delta\mathrm{CPC}$ ($M_1 - M_0$)", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=1)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)

    # Annotation box
    pos_count = np.sum(deltas > 0)
    ax.text(
        0.98, 0.05,
        f"Positive gain: {pos_count}/50 ({pos_count/n_cities*100:.0f}%)",
        transform=ax.transAxes,
        fontsize=8.5,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0", edgecolor="#cccccc", alpha=0.95)
    )

    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / "fig2_main_per_city.png", dpi=300)
    fig.savefig(figures_dir / "fig2_main_per_city.pdf")
    plt.close(fig)
    print("Generated Figure 2")


def generate_figure3(artifacts_dir: Path, figures_dir: Path):
    """Figure 3: Target Specificity and Controls."""
    summary_path = artifacts_dir / "unified_placebo" / "bootstrap_fold_stratified" / "unified_placebo_summary.json"
    if not summary_path.exists():
        summary_path = artifacts_dir / "unified_placebo" / "unified_placebo_summary.json"
    with open(summary_path, "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    conditions = ["Target $Y_D$", "Dose-matched donor", "Permuted $Y_D$"]
    keys = ["target", "matched_train_b", "permuted_b"]
    means = [float(summary_data[k]["mean_delta_cpc"]) for k in keys]
    ci_low = [float(summary_data[k]["ci_95"][0]) for k in keys]
    ci_high = [float(summary_data[k]["ci_95"][1]) for k in keys]

    yerr_low = np.array(means) - np.array(ci_low)
    yerr_high = np.array(ci_high) - np.array(means)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    colors = [PRIMARY_BLUE, GRAY, MUTED_RED]
    bars = ax.bar(range(len(conditions)), means, color=colors, width=0.55, edgecolor="#222222", linewidth=0.6, zorder=3)
    ax.errorbar(range(len(conditions)), means, yerr=[yerr_low, yerr_high], fmt="none", ecolor="#222222", capsize=5, elinewidth=1.3, zorder=4)

    ax.axhline(0, color="#333333", linewidth=0.9, linestyle="-", zorder=2)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, fontweight="bold")
    ax.set_ylabel("Mean $\\Delta\\mathrm{CPC}$", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    ax.text(0, ci_high[0] + 0.0006, f"{means[0]:+.5f}", ha="center", fontsize=9.5, fontweight="bold", color=PRIMARY_BLUE)
    ax.text(1, ci_high[1] + 0.0006, f"{means[1]:+.5f}", ha="center", fontsize=9.5, fontweight="bold", color="#555555")
    ax.text(2, ci_low[2] - 0.0011, f"{means[2]:+.5f}", ha="center", fontsize=9.5, fontweight="bold", color=MUTED_RED)

    min_y = min(-0.015, min(ci_low) * 1.25)
    max_y = max(+0.012, max(ci_high) * 1.25)
    ax.set_ylim(min_y, max_y)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig3_structural_validity_placebo.png", dpi=300)
    fig.savefig(figures_dir / "fig3_structural_validity_placebo.pdf")
    plt.close(fig)
    print("Generated Figure 3 (Placebo controls)")


def generate_figure4(artifacts_dir: Path, figures_dir: Path):
    """Figure 4: Calibration Gain vs. Distance-Bin Resolution (K sweep)."""
    k_json = artifacts_dir / "k_sensitivity" / "k_sensitivity_summary.json"
    with open(k_json, "r") as f:
        k_data = json.load(f)

    k_map = {row["K"]: row for row in k_data["summary"]}
    k_vals = sorted(list(k_map.keys()))
    k_means = [k_map[k]["mean_delta"] for k in k_vals]
    k_ci_low = [k_map[k]["ci_low"] for k in k_vals]
    k_ci_high = [k_map[k]["ci_high"] for k in k_vals]

    fig, ax = plt.subplots(figsize=(6.2, 4.2))

    yerr_low = np.array(k_means) - np.array(k_ci_low)
    yerr_high = np.array(k_ci_high) - np.array(k_means)

    ax.plot(k_vals, k_means, marker="o", color=PRIMARY_BLUE, linewidth=2.0, markersize=5.5, zorder=3)
    ax.errorbar(k_vals, k_means, yerr=[yerr_low, yerr_high], fmt="none",
                ecolor=PRIMARY_BLUE, capsize=3.5, elinewidth=1.2, zorder=3)

    if 8 in k_vals:
        k8_idx = k_vals.index(8)
        ax.plot(8, k_means[k8_idx], marker="o", markersize=8.5, color=PRIMARY_BLUE,
                markeredgecolor="#0f3b5c", markeredgewidth=1.8, zorder=5)
        ax.axvline(8, color="#888888", linestyle=":", linewidth=1.1, alpha=0.7, zorder=2)
        ax.annotate(
            "Main setting",
            xy=(8, k_means[k8_idx]),
            xytext=(8.6, k_means[k8_idx] - 0.0012),
            fontsize=9.0,
            fontweight="bold",
            color="#172b3a",
            arrowprops=dict(arrowstyle="->", color="#333333", lw=1.0, shrinkA=3, shrinkB=4),
            zorder=6
        )

    ax.set_xticks(k_vals)
    ax.set_xticklabels([f"{k}" for k in k_vals])
    ax.set_xlabel("Distance bins ($K$)", fontweight="bold")
    ax.set_ylabel("Mean $\\Delta\\mathrm{CPC}$", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35)

    ax.set_ylim(+0.0000, max(k_ci_high) * 1.15)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig4_resolution_sensitivity.png", dpi=300)
    fig.savefig(figures_dir / "fig4_resolution_sensitivity.pdf")
    plt.close(fig)
    print("Generated Figure 4 (K-sensitivity)")


def generate_figure5(artifacts_dir: Path, figures_dir: Path):
    """Figure 5: Noise dose-response & TV crossover."""
    json_path = artifacts_dir / "noise_robustness" / "noise_summary.json"
    if not json_path.exists():
        print("Warning: noise_summary.json not found, skipping Figure 5")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    res = data["results_by_eps"]
    eps_cross = data.get("eps_cross_zero_dCPC") or data.get("eps_cross")

    epsilons = sorted([float(e) for e in res.keys()])
    eps_pct = [e * 100 for e in epsilons]

    means = [res[str(e)]["mean_delta_cpc"] for e in epsilons]
    ci_lowers = [res[str(e)]["ci_lower"] for e in epsilons]
    ci_uppers = [res[str(e)]["ci_upper"] for e in epsilons]

    fig, ax = plt.subplots(figsize=(6.4, 4.0))

    ax.plot(eps_pct, means, marker="o", color=PRIMARY_BLUE, linewidth=2.0, markersize=5.5, zorder=4, label="Mean $\\Delta\\mathrm{CPC}$")
    ax.fill_between(eps_pct, ci_lowers, ci_uppers, color=PRIMARY_BLUE, alpha=0.18, zorder=2, label="95% bootstrap CI")

    ax.axhline(0, color="#333333", linestyle="-", linewidth=0.9, zorder=2)
    if eps_cross is not None:
        ax.axvline(eps_cross * 100, color=MUTED_RED, linestyle="--", linewidth=1.2, zorder=3)
        ax.text(eps_cross * 100 + 0.12, 0.0002, f"$\\epsilon_{{\\mathrm{{cross}}}} \\approx {eps_cross*100:.2f}\\%$",
                color=MUTED_RED, fontsize=9.5, fontweight="bold", verticalalignment="bottom")

    ax.set_xlabel("TV noise $\\epsilon$ (%)", fontweight="bold")
    ax.set_ylabel("Mean $\\Delta\\mathrm{CPC}$", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(figures_dir / "fig5_noise_dose_response.png", dpi=300)
    fig.savefig(figures_dir / "fig5_noise_dose_response.pdf")
    plt.close(fig)
    print("Generated Figure 5 (Noise dose-response)")


def generate_figure6(artifacts_dir: Path, figures_dir: Path):
    """Figure 6: Mechanistic Diagnostic - Baseline Distance Misalignment d_pre vs Delta CPC."""
    csv_path = artifacts_dir / "audit" / "dpre_mechanism_data.csv"
    df = pd.read_csv(csv_path)

    x = df["d_pre_tv"].values
    y = df["delta_cpc"].values

    slope, intercept, r_val, p_val, std_err = stats.linregress(x, y)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))

    ax.scatter(x, y, color=PRIMARY_BLUE, edgecolor="#144a70", s=45, alpha=0.85, zorder=3, label=f"Cities ($N={len(df)}$)")

    x_grid = np.linspace(x.min(), x.max(), 100)
    y_fit = intercept + slope * x_grid
    ax.plot(x_grid, y_fit, color=ACCENT_GREEN, linewidth=2.0, zorder=4, label=f"Linear fit ($r = {r_val:+.3f}$)")

    ax.axhline(0, color="#333333", linewidth=0.8, linestyle="--", alpha=0.5, zorder=2)
    ax.set_xlabel("Baseline distance mismatch $d_{\\mathrm{pre}}$", fontweight="bold")
    ax.set_ylabel("Calibration gain $\\Delta\\mathrm{CPC}$", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(figures_dir / "fig6_mechanistic_dpre.png", dpi=300)
    fig.savefig(figures_dir / "fig6_mechanistic_dpre.pdf")
    plt.close(fig)
    print("Generated Figure 6")


def generate_figure_s1(artifacts_dir: Path, figures_dir: Path):
    """Figure S1: Spatial resolution comparison on multi-county MSAs."""
    sp_json = artifacts_dir / "spatial_resolution" / "spatial_resolution_summary.json"
    if not sp_json.exists():
        print("Warning: spatial_resolution_summary.json not found, skipping Figure S1")
        return

    with open(sp_json, "r") as f:
        sp_data = json.load(f)

    mc_cities = sp_data["multi_county_subset"]["cities"]
    sp_per_city_json = artifacts_dir / "spatial_resolution" / "spatial_resolution_per_city.json"
    with open(sp_per_city_json, "r") as f:
        sp_city_data = json.load(f)

    sp_city_map = {row["city"]: row for row in sp_city_data}

    items = []
    for c in mc_cities:
        row = sp_city_map[c]
        d_res = row["delta_cpc_county"] - row["delta_cpc_city"]
        items.append({
            "name": c.replace("_", " "),
            "city_gain": row["delta_cpc_city"],
            "county_gain": row["delta_cpc_county"],
            "d_res": d_res
        })

    items.sort(key=lambda x: x["d_res"], reverse=True)

    clean_names = [it["name"] for it in items]
    city_gains = [it["city_gain"] for it in items]
    county_gains = [it["county_gain"] for it in items]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(clean_names))
    width = 0.38
    ax.bar(x - width/2, city_gains, width, label="City-level", color="#7faed6", edgecolor="#346896", zorder=3)
    ax.bar(x + width/2, county_gains, width, label="County-level", color="#1f4e79", edgecolor="#0e2942", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(clean_names, rotation=45, ha="right", fontsize=8.5)
    ax.set_ylabel("Calibration gain $\\Delta\\mathrm{CPC}$", fontweight="bold")
    ax.legend(loc="upper right", frameon=True, framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(figures_dir / "fig_s1_spatial_resolution.png", dpi=300)
    fig.savefig(figures_dir / "fig_s1_spatial_resolution.pdf")
    plt.close(fig)
    print("Generated Figure S1 (Spatial resolution standalone)")


def main():
    parser = argparse.ArgumentParser(description="Generate publication figures for interzonal_only")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("results/interzonal_only/artifacts"))
    parser.add_argument("--figures-dir", type=Path, default=Path("results/interzonal_only/artifacts/figures"))
    args = parser.parse_args()

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    generate_figure2(args.artifacts_dir, args.figures_dir)
    generate_figure3(args.artifacts_dir, args.figures_dir)
    generate_figure4(args.artifacts_dir, args.figures_dir)
    generate_figure5(args.artifacts_dir, args.figures_dir)
    generate_figure6(args.artifacts_dir, args.figures_dir)
    generate_figure_s1(args.artifacts_dir, args.figures_dir)


if __name__ == "__main__":
    main()
