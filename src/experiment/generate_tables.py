"""
Automated Table Generation from 5-Fold Experiment JSON Results.
Outputs both GitHub Markdown and LaTeX formatted tables.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

# Ensure repo root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.experiment.compute_delta_r import analyze_delta_r
from src.experiment.compute_qstar import analyze_qstar


def generate_tables(json_path: str, output_dir: str = "results/tables") -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    with open(json_path, "r") as f:
        data = json.load(f)

    city_results = data.get("city_level_results", [])
    if not city_results:
        print("No city_level_results found in JSON.")
        return {}

    # Re-compute stats using sample SD (ddof=1) to ensure latest definitions
    delta_r = analyze_delta_r(city_results)
    qstar = analyze_qstar(city_results)

    n_cities = len(city_results)
    tables = {}

    # =========================================================================
    # TABLE 1: Primary RQ1 Performance & Calibration Impact (Omega_c^+)
    # =========================================================================
    op = delta_r.get("oracle_plus", {})
    rp = delta_r.get("real_plus", {})
    rg = rp.get("realization_gap", {})

    t1_md = []
    t1_md.append("### Table 1: Primary RQ1 Estimands — Marginal Value of Coarse Mobility Information (Interzonal $\\Omega_c^+$)")
    t1_md.append("")
    t1_md.append("| Metric / Condition | Zero-Shot ($M_0$) | Real Moving-Bin ($M_1^{\\text{real},+}$) | Oracle Reference ($M_1^{\\text{oracle},+}$) | Realization Gap |")
    t1_md.append("|---|---|---|---|---|")
    
    m0_mean = op.get('m0_cpc_inter', {}).get('mean', 0.0)
    m0_std = op.get('m0_cpc_inter', {}).get('std', 0.0)
    m1r_mean = rp.get('m1_real_cpc_inter', {}).get('mean', 0.0)
    m1r_std = rp.get('m1_real_cpc_inter', {}).get('std', 0.0)
    m1o_mean = op.get('m1_oracle_cpc_inter', {}).get('mean', 0.0)
    m1o_std = op.get('m1_oracle_cpc_inter', {}).get('std', 0.0)
    rg_mean = rg.get('mean', 0.0)
    rg_std = rg.get('std', 0.0)
    
    t1_md.append(f"| **Interzonal CPC (Mean +- Sample SD)** | {m0_mean:.4f} +- {m0_std:.4f} | **{m1r_mean:.4f} +- {m1r_std:.4f}** | {m1o_mean:.4f} +- {m1o_std:.4f} | {rg_mean:+.4f} +- {rg_std:.4f} |")
    
    m0_med = op.get('m0_cpc_inter', {}).get('median', 0.0)
    m0_iqr = op.get('m0_cpc_inter', {}).get('iqr', 0.0)
    m1r_med = rp.get('m1_real_cpc_inter', {}).get('median', 0.0)
    m1r_iqr = rp.get('m1_real_cpc_inter', {}).get('iqr', 0.0)
    m1o_med = op.get('m1_oracle_cpc_inter', {}).get('median', 0.0)
    m1o_iqr = op.get('m1_oracle_cpc_inter', {}).get('iqr', 0.0)
    rg_med = rg.get('median', 0.0)
    
    t1_md.append(f"| **Interzonal CPC (Median, IQR)** | {m0_med:.4f} ({m0_iqr:.4f}) | **{m1r_med:.4f} ({m1r_iqr:.4f})** | {m1o_med:.4f} ({m1o_iqr:.4f}) | {rg_med:+.4f} |")
    
    dr_r_mean = rp.get('delta_r_inter', {}).get('mean', 0.0)
    dr_r_std = rp.get('delta_r_inter', {}).get('std', 0.0)
    dr_o_mean = op.get('delta_r_inter', {}).get('mean', 0.0)
    dr_o_std = op.get('delta_r_inter', {}).get('std', 0.0)
    
    t1_md.append(f"| **Marginal Gain $\\Delta R$ (Mean +- Sample SD)** | --- | **{dr_r_mean:+.4f} +- {dr_r_std:.4f}** | {dr_o_mean:+.4f} +- {dr_o_std:.4f} | --- |")
    
    p_imp_r = rp.get('p_improved', 0.0) * 100
    p_imp_o = op.get('p_improved', 0.0) * 100
    t1_md.append(f"| **Improvement Rate $P(\\Delta R > 0)$** | --- | **{p_imp_r:.1f}%** ({int(round(p_imp_r*n_cities/100))}/{n_cities}) | {p_imp_o:.1f}% ({int(round(p_imp_o*n_cities/100))}/{n_cities}) | --- |")
    
    w_p_one = rp.get('wilcoxon_one_sided_p')
    w_p_str = f"{w_p_one:.4e}" if w_p_one is not None else "N/A"
    t1_md.append(f"| **Wilcoxon Signed-Rank Test ($p_1$)** | --- | **p = {w_p_str}** | --- | --- |")
    
    ov_mean = rp.get('distributional_overlap', {}).get('mean', 0.0) * 100 if rp.get('distributional_overlap') else 0.0
    ov_std = rp.get('distributional_overlap', {}).get('std', 0.0) * 100 if rp.get('distributional_overlap') else 0.0
    t1_md.append(f"| **Distributional Overlap with Prior** | --- | {ov_mean:.1f}% +- {ov_std:.1f}% | 100.0% | --- |")
    
    tables["table1_primary_rq1.md"] = "\n".join(t1_md)

    # =========================================================================
    # TABLE 2: Ablation Analysis (Moving-Bin vs 4-Bin Legacy)
    # =========================================================================
    ab = delta_r.get("4bin_ablation", {})
    t2_md = []
    t2_md.append("### Table 2: Ablation Study — Moving-Bin vs Legacy 4-Bin Calibration Across Evaluation Domains")
    t2_md.append("")
    t2_md.append("| Framework / Condition | Calibration Target | Evaluated Domain | Interzonal CPC ($\Omega_c^+$) | Full CPC ($\Omega_c$) | Interzonal $\\Delta R$ | $P(\\Delta R_{\\text{inter}} > 0)$ |")
    t2_md.append("|---|---|---|---|---|---|---|")
    
    m0_full_mean = op.get('m0_cpc_full', {}).get('mean', 0.0)
    m0_full_std = op.get('m0_cpc_full', {}).get('std', 0.0)
    t2_md.append(f"| **$M_0$ (Zero-Shot)** | None | All | {m0_mean:.4f} +- {m0_std:.4f} | {m0_full_mean:.4f} +- {m0_full_std:.4f} | --- | --- |")
    
    m1r_full_mean = rp.get('m1_real_cpc_full', {}).get('mean', 0.0)
    m1r_full_std = rp.get('m1_real_cpc_full', {}).get('std', 0.0)
    t2_md.append(f"| **$M_1^{{\\text{{real}},+}}$ (Moving-Bin)** | Bins 1,2,3 ($D>0$) | $\\Omega_c^+$ | **{m1r_mean:.4f} +- {m1r_std:.4f}** | {m1r_full_mean:.4f} +- {m1r_full_std:.4f} | **{dr_r_mean:+.4f} +- {dr_r_std:.4f}** | **{p_imp_r:.1f}%** |")
    
    m1ab_inter_mean = ab.get('m1_4bin_cpc_inter', {}).get('mean', 0.0)
    m1ab_inter_std = ab.get('m1_4bin_cpc_inter', {}).get('std', 0.0)
    m1ab_full_mean = ab.get('m1_4bin_cpc_full', {}).get('mean', 0.0)
    m1ab_full_std = ab.get('m1_4bin_cpc_full', {}).get('std', 0.0)
    dr_ab_inter_mean = ab.get('delta_r_inter', {}).get('mean', 0.0)
    dr_ab_inter_std = ab.get('delta_r_inter', {}).get('std', 0.0)
    p_imp_ab_inter = ab.get('p_improved_inter', 0.0) * 100
    t2_md.append(f"| **$M_1^{{\\text{{real, 4bin}}}}$ (Legacy 4-Bin)** | Bins 0,1,2,3 ($D\\ge 0$) | $\\Omega_c$ | {m1ab_inter_mean:.4f} +- {m1ab_inter_std:.4f} | {m1ab_full_mean:.4f} +- {m1ab_full_std:.4f} | {dr_ab_inter_mean:+.4f} +- {dr_ab_inter_std:.4f} | {p_imp_ab_inter:.1f}% |")
    
    m1o_full_mean = op.get('m1_oracle_cpc_full', {}).get('mean', 0.0)
    m1o_full_std = op.get('m1_oracle_cpc_full', {}).get('std', 0.0)
    t2_md.append(f"| **$M_1^{{\\text{{oracle}},+}}$ (Oracle Reference)** | Oracle Bins 1,2,3 | $\\Omega_c^+$ | {m1o_mean:.4f} +- {m1o_std:.4f} | {m1o_full_mean:.4f} +- {m1o_full_std:.4f} | {dr_o_mean:+.4f} +- {dr_o_std:.4f} | {p_imp_o:.1f}% |")
    
    tables["table2_ablation.md"] = "\n".join(t2_md)

    # =========================================================================
    # TABLE 3: City-Level Breakdown Table
    # =========================================================================
    t3_md = []
    t3_md.append("### Table 3: City-Level Performance Breakdown Across Held-Out Target Cities")
    t3_md.append("")
    t3_md.append("| Target City | Tracts ($N$) | Inter Pairs ($|\\Omega^+|$) | Meta Overlap | $M_0$ CPC | $M_1^{\\text{real},+}$ CPC | $\\Delta R^{\\text{real},+}$ | $M_1^{\\text{oracle},+}$ CPC | $M_1^{\\text{4bin}}$ CPC | $q^*_{\\text{real}}$ |")
    t3_md.append("|---|---|---|---|---|---|---|---|---|---|")
    
    for r in city_results:
        city = r.get("city", r.get("city_name", "Unknown"))
        n_tr = r.get("n_tracts", 0)
        n_p = r.get("n_inter_pairs", 0)
        ov = r.get("distributional_overlap", 0.0)
        ov_s = f"{ov*100:.1f}%" if ov is not None else "N/A"
        m0_c = r.get("M0", {}).get("cpc_inter", 0.0)
        m1r_c = r.get("M1_real_plus", {}).get("cpc_inter") if r.get("M1_real_plus") else None
        m1r_s = f"{m1r_c:.4f}" if m1r_c is not None else "N/A"
        dr_c = r.get("delta_r_real_plus")
        dr_s = f"{dr_c:+.4f}" if dr_c is not None else "N/A"
        m1o_c = r.get("M1_oracle_plus", {}).get("cpc_inter", 0.0)
        m1ab_c = r.get("M1_4bin_ablation", {}).get("cpc_inter") if r.get("M1_4bin_ablation") else None
        m1ab_s = f"{m1ab_c:.4f}" if m1ab_c is not None else "N/A"
        qr = r.get("q_star_real")
        qr_s = f"{qr:.4f}" if qr is not None else "N/A"
        
        t3_md.append(f"| **{city}** | {n_tr} | {n_p:,} | {ov_s} | {m0_c:.4f} | {m1r_s} | **{dr_s}** | {m1o_c:.4f} | {m1ab_s} | {qr_s} |")
        
    tables["table3_city_breakdown.md"] = "\n".join(t3_md)

    # =========================================================================
    # TABLE 4: City-Level Ablation Breakdown (Moving-Bin vs 4-Bin Penalty)
    # =========================================================================
    t4_md = []
    t4_md.append("### Table 4: City-Level Ablation Breakdown — Moving-Bin vs Legacy 4-Bin (Penalty of Bin 0 Inclusion)")
    t4_md.append("")
    t4_md.append("| City | $M_0$ $\\text{CPC}_{\\text{inter}}$ | $M_1^{\\text{real},+}$ $\\text{CPC}_{\\text{inter}}$ | $M_1^{\\text{4bin}}$ $\\text{CPC}_{\\text{inter}}$ | $\\Delta R^{\\text{real},+}$ | $\\Delta R^{\\text{4bin}}$ | Ablation Penalty ($\\Delta R^{\\text{real},+} - \\Delta R^{\\text{4bin}}$) |")
    t4_md.append("|---|---|---|---|---|---|---|")

    t4_csv_lines = [
        "city,cpc_inter_M0,cpc_inter_M1_real_plus,cpc_inter_M1_4bin,delta_real_plus,delta_4bin,ablation_penalty"
    ]

    for r in city_results:
        city = r.get("city", r.get("city_name", "Unknown"))
        m0_c = r.get("M0", {}).get("cpc_inter", 0.0)
        m1r_c = r.get("M1_real_plus", {}).get("cpc_inter") if r.get("M1_real_plus") else None
        m1ab_c = r.get("M1_4bin_ablation", {}).get("cpc_inter") if r.get("M1_4bin_ablation") else None

        dr_real = (m1r_c - m0_c) if m1r_c is not None else None
        dr_4bin = (m1ab_c - m0_c) if m1ab_c is not None else None
        penalty = (dr_real - dr_4bin) if (dr_real is not None and dr_4bin is not None) else None

        m1r_str = f"{m1r_c:.4f}" if m1r_c is not None else "N/A"
        m1ab_str = f"{m1ab_c:.4f}" if m1ab_c is not None else "N/A"
        dr_real_str = f"{dr_real:+.4f}" if dr_real is not None else "N/A"
        dr_4bin_str = f"{dr_4bin:+.4f}" if dr_4bin is not None else "N/A"
        pen_str = f"{penalty:+.4f}" if penalty is not None else "N/A"

        t4_md.append(f"| **{city}** | {m0_c:.4f} | {m1r_str} | {m1ab_str} | {dr_real_str} | {dr_4bin_str} | **{pen_str}** |")

        c_m1r = f"{m1r_c:.6f}" if m1r_c is not None else ""
        c_m1ab = f"{m1ab_c:.6f}" if m1ab_c is not None else ""
        c_dr_real = f"{dr_real:.6f}" if dr_real is not None else ""
        c_dr_4bin = f"{dr_4bin:.6f}" if dr_4bin is not None else ""
        c_pen = f"{penalty:.6f}" if penalty is not None else ""
        t4_csv_lines.append(f"{city},{m0_c:.6f},{c_m1r},{c_m1ab},{c_dr_real},{c_dr_4bin},{c_pen}")

    tables["table4_ablation_city_breakdown.md"] = "\n".join(t4_md)

    # Write CSV
    with open(Path(output_dir) / "ablation_city_breakdown.csv", "w", encoding="utf-8") as f:
        f.write("\n".join(t4_csv_lines))

    # Save all markdown tables
    for filename, content in tables.items():
        with open(Path(output_dir) / filename, "w", encoding="utf-8") as f:
            f.write(content)

    print(f"Generated {len(tables)} tables and CSVs in {output_dir}:")
    for k in tables:
        print(f" - {k}")
    print(f" - ablation_city_breakdown.csv")

    return tables


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, default="results/5fold_results.json")
    parser.add_argument("--output-dir", type=str, default="results/tables")
    args = parser.parse_args()
    generate_tables(args.json, args.output_dir)
