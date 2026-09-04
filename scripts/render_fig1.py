"""
Generate publication-quality PNG and PDF for Figure 1.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_fig1():
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    ax.set_xlim(0, 1200)
    ax.set_ylim(700, 0)  # Invert Y to match SVG coordinate system
    ax.axis("off")

    # Background
    bg = patches.Rectangle((0, 0), 1200, 700, facecolor="#fafafa", edgecolor="none")
    ax.add_patch(bg)

    # Styles
    header_color = "#243b53"
    train_face = "#e8f1f8"
    train_edge = "#1f5f8b"
    oracle_face = "#fdf0e4"
    oracle_edge = "#b45309"
    output_face = "#e8f4ec"
    output_edge = "#2f855a"
    neutral_face = "#f7f7f7"
    neutral_edge = "#5f6b76"

    def draw_box(x, y, w, h, fc, ec, lw=1.5, rx=6):
        box = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0,rounding_size={rx}",
            facecolor=fc, edgecolor=ec, linewidth=lw
        )
        ax.add_patch(box)

    def draw_arrow(x1, y1, x2, y2, color="#1f5f8b", dashed=False):
        style = "--" if dashed else "-"
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=2,
                linestyle=style,
                mutation_scale=14,
            ),
        )

    # Panels background
    panels = [
        (24, 30, 355, 625, "A. Cross-city model training"),
        (423, 30, 355, 625, "B. Held-out target-city inference"),
        (822, 30, 355, 625, "C. Inference-time calibration"),
    ]
    for px, py, pw, ph, title in panels:
        draw_box(px, py, pw, ph, "#ffffff", "#4b5563", lw=1.5)
        draw_box(px, py, pw, 46, header_color, header_color, lw=0)
        ax.text(px + pw / 2, py + 29, title, color="white", fontsize=12, fontweight="bold", ha="center", va="center")

    # PANEL A
    draw_box(63, 102, 275, 67, train_face, train_edge)
    ax.text(201, 126, "35 training cities + 5 validation cities", fontsize=10.5, fontweight="bold", color="#172b3a", ha="center", va="center")
    ax.text(201, 149, "source-city positive OD intensities", fontsize=9.5, color="#172b3a", ha="center", va="center")
    draw_arrow(201, 169, 201, 212)

    draw_box(63, 212, 275, 84, train_face, train_edge)
    ax.text(201, 238, "Cross-city inputs", fontsize=10.5, fontweight="bold", color="#172b3a", ha="center", va="center")
    ax.text(201, 260, "tract features + spatial graph", fontsize=9.5, color="#172b3a", ha="center", va="center")
    ax.text(201, 279, "+ pairwise distances", fontsize=9.5, color="#172b3a", ha="center", va="center")
    draw_arrow(201, 296, 201, 339)

    draw_box(63, 339, 275, 72, train_face, train_edge)
    ax.text(201, 365, "Train and select model $M_0$", fontsize=10.5, fontweight="bold", color="#172b3a", ha="center", va="center")
    ax.text(201, 388, "cross-city learning only", fontsize=9.5, color="#172b3a", ha="center", va="center")
    draw_arrow(201, 411, 201, 454)

    draw_box(63, 454, 275, 62, neutral_face, neutral_edge, lw=1.25)
    ax.text(201, 480, "Freeze all model parameters", fontsize=10.5, fontweight="bold", color="#172b3a", ha="center", va="center")
    ax.text(201, 502, "before target-city inference", fontsize=9.5, color="#172b3a", ha="center", va="center")

    draw_box(48, 553, 306, 65, "#fff7ed", "#b45309", lw=1.25)
    ax.text(201, 578, "No target-city OD intensity", fontsize=9.5, fontweight="bold", color="#9a4d00", ha="center", va="center")
    ax.text(201, 600, "is used for model training", fontsize=9.5, fontweight="bold", color="#9a4d00", ha="center", va="center")

    # PANEL B
    draw_box(463, 102, 275, 91, train_face, train_edge)
    ax.text(601, 127, "Held-out target city", fontsize=10.5, fontweight="bold", color="#172b3a", ha="center", va="center")
    ax.text(601, 149, "features + distances + known", fontsize=9.5, color="#172b3a", ha="center", va="center")
    ax.text(601, 171, r"positive support $\Omega_c$", fontsize=9.5, color="#172b3a", ha="center", va="center")
    draw_arrow(601, 193, 601, 232)

    draw_box(463, 232, 275, 65, train_face, train_edge)
    ax.text(601, 256, "Frozen $M_0$", fontsize=10.5, fontweight="bold", color="#172b3a", ha="center", va="center")
    ax.text(601, 278, "no fine-tuning or parameter update", fontsize=9.5, color="#172b3a", ha="center", va="center")
    draw_arrow(601, 297, 601, 336)

    draw_box(463, 336, 275, 59, train_face, train_edge)
    ax.text(601, 368, r"Baseline prediction $T^{(0)}$", fontsize=10.5, fontweight="bold", color="#172b3a", ha="center", va="center")

    draw_box(448, 438, 305, 72, oracle_face, oracle_edge)
    ax.text(601, 464, "Target positive ground-truth OD flows", fontsize=10, fontweight="bold", color="#172b3a", ha="center", va="center")
    ax.text(601, 486, "same target reference flows used for evaluation", fontsize=9, color="#172b3a", ha="center", va="center")
    draw_arrow(601, 510, 601, 544, color="#b45309", dashed=True)

    draw_box(448, 544, 305, 68, oracle_face, oracle_edge)
    ax.text(601, 569, "Deterministic distance-bin aggregation", fontsize=10, fontweight="bold", color="#172b3a", ha="center", va="center")
    ax.text(601, 591, r"$Y_{D,c}$ oracle", fontsize=9.5, fontweight="bold", color="#b45309", ha="center", va="center")
    ax.text(601, 638, "Dashed path: oracle intervention, not external telemetry", fontsize=8.5, fontweight="bold", color="#9a4d00", ha="center", va="center")

    # PANEL C
    draw_box(862, 102, 275, 66, neutral_face, neutral_edge, lw=1.25)
    ax.text(999, 127, r"$T^{(0)} + Y_{D,c}$ oracle", fontsize=10.5, fontweight="bold", color="#172b3a", ha="center", va="center")
    ax.text(999, 149, r"on the same known positive support $\Omega_c$", fontsize=9.5, color="#172b3a", ha="center", va="center")
    draw_arrow(999, 168, 999, 206, color="#2f855a")

    draw_box(847, 206, 305, 101, output_face, output_edge)
    ax.text(999, 236, "Compare bin shares", fontsize=10.5, fontweight="bold", color="#172b3a", ha="center", va="center")
    ax.text(999, 261, "Aggregate predicted flow by distance bin", fontsize=9.5, color="#172b3a", ha="center", va="center")
    ax.text(999, 283, "Compare baseline and target bin shares", fontsize=9.5, color="#172b3a", ha="center", va="center")
    draw_arrow(999, 307, 999, 344, color="#2f855a")

    draw_box(847, 344, 305, 75, output_face, output_edge)
    ax.text(999, 371, "Rescale flows within each bin", fontsize=10.5, fontweight="bold", color="#172b3a", ha="center", va="center")
    ax.text(999, 396, "Multiply by ratio of target to baseline share", fontsize=9.5, color="#172b3a", ha="center", va="center")
    draw_arrow(999, 419, 999, 456, color="#2f855a")

    draw_box(847, 456, 305, 65, output_face, output_edge)
    ax.text(999, 481, r"Calibrated prediction $T^{(1)}$", fontsize=10.5, fontweight="bold", color="#172b3a", ha="center", va="center")
    ax.text(999, 503, r"evaluate on the same $\Omega_c$", fontsize=9.5, color="#172b3a", ha="center", va="center")

    draw_box(847, 548, 305, 79, "#edf7ef", "#2f855a", lw=1.25)
    ax.text(999, 569, "Frozen model: no parameter update", fontsize=9, fontweight="bold", color="#1f4f38", ha="center", va="center")
    ax.text(999, 588, "Same support: no new OD links", fontsize=9, fontweight="bold", color="#1f4f38", ha="center", va="center")
    ax.text(999, 607, "Preserve support, total flow, and within-bin ranking", fontsize=9, fontweight="bold", color="#1f4f38", ha="center", va="center")

    out_dir = Path("paper/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "fig1_oracle_calibration_framework.png"
    pdf_path = out_dir / "fig1_oracle_calibration_framework.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Generated {png_path} and {pdf_path}")

if __name__ == "__main__":
    draw_fig1()
