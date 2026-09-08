"""
Generate publication-quality PNG and PDF for Figure 1.
Simplified schematic: minimal text, max 1-2 lines per box, clear workflow.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_fig1():
    fig, ax = plt.subplots(figsize=(12, 6.2), dpi=300)
    ax.set_xlim(0, 1200)
    ax.set_ylim(620, 0)  # Invert Y to match coordinate system
    ax.axis("off")

    # Background
    bg = patches.Rectangle((0, 0), 1200, 620, facecolor="#fafafa", edgecolor="none")
    ax.add_patch(bg)

    # Styling Palette
    header_color = "#243b53"
    train_face = "#e8f1f8"
    train_edge = "#1f5f8b"
    oracle_face = "#fdf0e4"
    oracle_edge = "#b45309"
    output_face = "#e8f4ec"
    output_edge = "#2f855a"
    neutral_face = "#f4f6f8"
    neutral_edge = "#64748b"

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
                lw=2.0,
                linestyle=style,
                mutation_scale=14,
            ),
        )


    # 3 Main Panel Columns
    panel_y = 25
    panel_h = 515
    panels = [
        (24, panel_y, 355, panel_h, "A. Cross-city training"),
        (423, panel_y, 355, panel_h, "B. Held-out target city"),
        (822, panel_y, 355, panel_h, "C. Inference-time calibration"),
    ]
    for px, py, pw, ph, title in panels:
        draw_box(px, py, pw, ph, "#ffffff", "#94a3b8", lw=1.2)
        draw_box(px, py, pw, 44, header_color, header_color, lw=0)
        ax.text(px + pw / 2, py + 26, title, color="white", fontsize=12, fontweight="bold", ha="center", va="center")

    # ==========================================
    # COLUMN A: Cross-city training
    # ==========================================
    draw_box(56, 105, 290, 75, train_face, train_edge)
    ax.text(201, 131, "Training cities", fontsize=11, fontweight="bold", color="#172b3a", ha="center", va="center")
    ax.text(201, 155, "Urban features + distances", fontsize=9.5, color="#2c4d63", ha="center", va="center")
    draw_arrow(201, 180, 201, 235)

    draw_box(56, 235, 290, 68, train_face, train_edge)
    ax.text(201, 269, r"Train baseline $M_0$", fontsize=11, fontweight="bold", color="#172b3a", ha="center", va="center")
    draw_arrow(201, 303, 201, 358)

    draw_box(56, 358, 290, 68, neutral_face, neutral_edge)
    ax.text(201, 392, r"Freeze baseline $M_0$", fontsize=11, fontweight="bold", color="#334155", ha="center", va="center")

    # ==========================================
    # COLUMN B: Held-out target city
    # ==========================================
    draw_box(455, 102, 290, 80, train_face, train_edge)
    ax.text(600, 122, "Target city", fontsize=11, fontweight="bold", color="#172b3a", ha="center", va="center")
    ax.text(600, 145, "Features + distances", fontsize=9.5, color="#2c4d63", ha="center", va="center")
    ax.text(600, 165, r"+ known support $\Omega_c$", fontsize=9.5, color="#2c4d63", ha="center", va="center")
    draw_arrow(600, 182, 600, 235)

    # Box 2: Baseline prediction
    draw_box(455, 235, 290, 68, train_face, train_edge)
    ax.text(600, 269, r"Baseline prediction $\widehat{T}^{(0)}$", fontsize=11, fontweight="bold", color="#172b3a", ha="center", va="center")

    draw_box(455, 345, 290, 60, oracle_face, oracle_edge)
    ax.text(600, 375, "Target OD reference", fontsize=10.5, fontweight="bold", color="#9a3412", ha="center", va="center")
    draw_arrow(600, 405, 600, 440, color="#b45309", dashed=True)

    draw_box(455, 440, 290, 68, oracle_face, oracle_edge, lw=1.8)
    ax.text(600, 464, "Distance-bin oracle", fontsize=10, color="#9a3412", ha="center", va="center")
    ax.text(600, 487, r"$Y_D$ (target shares)", fontsize=11.5, fontweight="bold", color="#b45309", ha="center", va="center")

    draw_arrow(745, 269, 854, 132, color="#1f5f8b")
    draw_arrow(745, 474, 854, 152, color="#b45309", dashed=True)

    # ==========================================
    # COLUMN C: Inference-time calibration
    # ==========================================
    draw_box(854, 105, 290, 75, output_face, output_edge)
    ax.text(999, 131, "Compare bin shares", fontsize=11, fontweight="bold", color="#14532d", ha="center", va="center")
    ax.text(999, 155, r"$\widehat{Y}_D^{(0)}$ vs. $Y_D$", fontsize=10.5, fontweight="bold", color="#166534", ha="center", va="center")
    draw_arrow(999, 180, 999, 235, color="#166534")

    draw_box(854, 235, 290, 68, output_face, output_edge)
    ax.text(999, 269, "Rescale within bins", fontsize=11, fontweight="bold", color="#14532d", ha="center", va="center")
    draw_arrow(999, 303, 999, 358, color="#166534")

    draw_box(854, 358, 290, 75, output_face, output_edge, lw=2.0)
    ax.text(999, 384, "Calibrated prediction", fontsize=11, fontweight="bold", color="#14532d", ha="center", va="center")
    ax.text(999, 408, r"$\widehat{T}^{(1)}$ on $\Omega_c$", fontsize=10.5, color="#166534", ha="center", va="center")

    # ==========================================
    # FOOTER BAR
    # ==========================================
    footer_box = patches.FancyBboxPatch(
        (120, 560), 960, 42,
        boxstyle="round,pad=0,rounding_size=6",
        facecolor="#f1f5f9", edgecolor="#94a3b8", linewidth=1.2
    )
    ax.add_patch(footer_box)
    ax.text(
        600, 581,
        r"Frozen model  $\cdot$  Same support $\Omega_c$  $\cdot$  No new OD links",
        fontsize=10.5, fontweight="bold", color="#334155", ha="center", va="center"
    )

    out_dir = Path("paper/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "fig1_oracle_calibration_framework.png"
    pdf_path = out_dir / "fig1_oracle_calibration_framework.pdf"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Successfully generated simplified {png_path} and {pdf_path}")

if __name__ == "__main__":
    draw_fig1()
