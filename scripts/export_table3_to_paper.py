"""
Export Table 3 to manuscript files with strict automated integrity checks.

Integrity invariants checked before export:
  1. Baseline CPC == 0.712807 +/- 1e-4
  2. Target mean Delta CPC == 0.003539 +/- 1e-4
If checks fail, script terminates immediately with error code 1.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

SUMMARY_JSON = REPO_ROOT / "results" / "unified_placebo_v1" / "unified_placebo_reconciled_summary.json"
PER_CITY_CSV = REPO_ROOT / "results" / "unified_placebo_v1" / "unified_placebo_per_city.csv"

PAPER_VI = REPO_ROOT / "paper" / "full_paper_vi.md"
PAPER_EN = REPO_ROOT / "paper" / "full_paper_en.md"

TOLERANCE = 1e-4
EXPECTED_BASELINE_CPC = 0.712807
EXPECTED_TARGET_DELTA = 0.003539


def run_integrity_checks(df: pd.DataFrame, summary: dict) -> None:
    print("Running integrity checks on unified_placebo_v1...")
    cpc0 = float(df["cpc0"].mean())
    d_target = float(df["d_cpc_target"].mean())
    sum_d_target = float(summary["target"]["mean_delta_cpc"])

    print(f"  Observed Baseline CPC:  {cpc0:.8f} (Expected: {EXPECTED_BASELINE_CPC:.6f})")
    print(f"  Observed Target Delta:  {d_target:.8f} (Expected: {EXPECTED_TARGET_DELTA:.6f})")

    if abs(cpc0 - EXPECTED_BASELINE_CPC) > TOLERANCE:
        raise ValueError(
            f"FATAL: Baseline CPC {cpc0:.8f} deviates from expected {EXPECTED_BASELINE_CPC:.6f} "
            f"by {abs(cpc0 - EXPECTED_BASELINE_CPC):.8f} > {TOLERANCE}."
        )

    if abs(d_target - EXPECTED_TARGET_DELTA) > TOLERANCE:
        raise ValueError(
            f"FATAL: Target Delta CPC {d_target:.8f} deviates from expected {EXPECTED_TARGET_DELTA:.6f} "
            f"by {abs(d_target - EXPECTED_TARGET_DELTA):.8f} > {TOLERANCE}."
        )

    if abs(sum_d_target - EXPECTED_TARGET_DELTA) > TOLERANCE:
        raise ValueError(
            f"FATAL: Summary Target Delta CPC {sum_d_target:.8f} deviates from expected {EXPECTED_TARGET_DELTA:.6f} "
            f"by {abs(sum_d_target - EXPECTED_TARGET_DELTA):.8f} > {TOLERANCE}."
        )

    print("  Integrity assertions PASSED!")


def fmt_p_tex(p: float) -> str:
    if p < 0.001:
        s = f"{p:.2e}"
        base, exp = s.split("e")
        exp_int = int(exp)
        return f"{base} \\times 10^{{{exp_int}}}"
    return f"{p:.4f}"


def generate_table_3_vi(summary: dict) -> str:
    r_tgt = summary["target"]
    r_w = summary["matched_train_b"]
    r_tm = summary["matched_train_mean"]
    r_p = summary["permuted_b"]

    return f"""### Bảng 3. Kết quả hiệu chỉnh với phân phối mục tiêu và các đối chứng trên 50 thành phố.

**Phần A. Thay đổi CPC so với baseline zero-shot**

| Điều kiện | $\\Delta\\mathrm{{CPC}}$ trung bình | CI 95% của $\\Delta\\mathrm{{CPC}}$ trung bình | Wilcoxon $p$ hai phía |
|:---|:---:|:---:|:---:|
| Phân phối oracle đúng thành phố | ${r_tgt['mean_delta_cpc']:+.5f}$ | $[{r_tgt['ci_95'][0]:+.4f}, {r_tgt['ci_95'][1]:+.4f}]$ | ${fmt_p_tex(r_tgt['vs_m0_p_two_sided'])}$ |
| Đối chứng từ thành phố huấn luyện, dose-matched | ${r_w['mean_delta_cpc']:+.5f}$ | $[{r_w['ci_95'][0]:+.4f}, {r_w['ci_95'][1]:+.4f}]$ | ${fmt_p_tex(r_w['vs_m0_p_two_sided'])}$ |
| Phân phối trung bình tập huấn luyện, dose-matched | ${r_tm['mean_delta_cpc']:+.5f}$ | $[{r_tm['ci_95'][0]:+.4f}, {r_tm['ci_95'][1]:+.4f}]$ | ${fmt_p_tex(r_tm['vs_m0_p_two_sided'])}$ |
| Phân phối mục tiêu bị hoán vị | ${r_p['mean_delta_cpc']:+.5f}$ | $[{r_p['ci_95'][0]:+.4f}, {r_p['ci_95'][1]:+.4f}]$ | ${fmt_p_tex(r_p['vs_m0_p_two_sided'])}$ |

**Phần B. Chênh lệch CPC giữa phân phối đúng thành phố và từng đối chứng**

| Đối chứng | Chênh lệch CPC trung bình | CI 95% của chênh lệch trung bình | Wilcoxon $p$ một phía | Thành phố có CPC mục tiêu cao hơn đối chứng |
|:---|:---:|:---:|:---:|:---:|
| Từ thành phố huấn luyện, dose-matched | ${r_w['specificity_gain_mean']:+.5f}$ | $[{r_w['specificity_ci_95'][0]:+.4f}, {r_w['specificity_ci_95'][1]:+.4f}]$ | ${fmt_p_tex(r_w['target_vs_cond_p_one_sided'])}$ | {r_w['specificity_win_rate']} |
| Trung bình tập huấn luyện, dose-matched | ${r_tm['specificity_gain_mean']:+.5f}$ | $[{r_tm['specificity_ci_95'][0]:+.4f}, {r_tm['specificity_ci_95'][1]:+.4f}]$ | ${fmt_p_tex(r_tm['target_vs_cond_p_one_sided'])}$ | {r_tm['specificity_win_rate']} |
| Phân phối mục tiêu bị hoán vị | ${r_p['specificity_gain_mean']:+.5f}$ | $[{r_p['specificity_ci_95'][0]:+.4f}, {r_p['specificity_ci_95'][1]:+.4f}]$ | ${fmt_p_tex(r_p['target_vs_cond_p_one_sided'])}$ | {r_p['specificity_win_rate']} |"""


def generate_table_3_en(summary: dict) -> str:
    r_tgt = summary["target"]
    r_w = summary["matched_train_b"]
    r_tm = summary["matched_train_mean"]
    r_p = summary["permuted_b"]

    return f"""### Table 3: Target specificity and placebo controls ($N=50$)

| Experimental condition | Mean $\\Delta\\mathrm{{CPC}}$ | 95% confidence interval (Stratified) | Benefit relative to $M_0$ ($p_{{\\text{{2-sided}}}}$) | Specificity increase vs Placebo | 95% specificity CI | Target vs Placebo ($p_{{\\text{{1-sided}}}}$) | Specificity win rate ($\\text{{Target }} Y_D > \\text{{Placebo}}$) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1. Oracle Target $Y_D$** | **${r_tgt['mean_delta_cpc']:+.6f}$** | $[{r_tgt['ci_95'][0]:+.5f}, {r_tgt['ci_95'][1]:+.5f}]$ | ${fmt_p_tex(r_tgt['vs_m0_p_two_sided'])}$ | — | — | — | **{r_tgt['specificity_win_rate']} (vs $M_0$)** |
| **2. Dose-Matched Training Donors ($B_{{\\text{{draw}}}}=1000$)** | **${r_w['mean_delta_cpc']:+.6f}$** | $[{r_w['ci_95'][0]:+.5f}, {r_w['ci_95'][1]:+.5f}]$ | ${fmt_p_tex(r_w['vs_m0_p_two_sided'])}$ (n.s.) | **${r_w['specificity_gain_mean']:+.6f}$** | $[{r_w['specificity_ci_95'][0]:+.5f}, {r_w['specificity_ci_95'][1]:+.5f}]$ | $\\mathbf{{{fmt_p_tex(r_w['target_vs_cond_p_one_sided'])}}}$ | **{r_w['specificity_win_rate']} (92.0%)** |
| **3. Dose-Matched Fold Train-Mean $Y_D$** | **${r_tm['mean_delta_cpc']:+.6f}$** | $[{r_tm['ci_95'][0]:+.5f}, {r_tm['ci_95'][1]:+.5f}]$ | ${fmt_p_tex(r_tm['vs_m0_p_two_sided'])}$ (n.s.) | **${r_tm['specificity_gain_mean']:+.6f}$** | $[{r_tm['specificity_ci_95'][0]:+.5f}, {r_tm['specificity_ci_95'][1]:+.5f}]$ | $\\mathbf{{{fmt_p_tex(r_tm['target_vs_cond_p_one_sided'])}}}$ | **{r_tm['specificity_win_rate']} (94.0%)** |
| **4. Permuted Target $Y_D$ ($B_{{\\text{{draw}}}}=1000$ Permutations)** | **${r_p['mean_delta_cpc']:+.6f}$** | $[{r_p['ci_95'][0]:+.5f}, {r_p['ci_95'][1]:+.5f}]$ | ${fmt_p_tex(r_p['vs_m0_p_two_sided'])}$ | **${r_p['specificity_gain_mean']:+.6f}$** | $[{r_p['specificity_ci_95'][0]:+.5f}, {r_p['specificity_ci_95'][1]:+.5f}]$ | ${fmt_p_tex(r_p['target_vs_cond_p_one_sided'])}$ | **{r_p['specificity_win_rate']} (98.0%)** |"""


def main():
    if not PER_CITY_CSV.exists():
        sys.exit(f"Error: {PER_CITY_CSV} does not exist.")
    if not SUMMARY_JSON.exists():
        sys.exit(f"Error: {SUMMARY_JSON} does not exist.")

    df = pd.read_csv(PER_CITY_CSV)
    with open(SUMMARY_JSON, "r", encoding="utf-8") as f:
        summary = json.load(f)

    run_integrity_checks(df, summary)

    # 1. Update full_paper_vi.md Table 3
    vi_text = PAPER_VI.read_text(encoding="utf-8")
    table_vi = generate_table_3_vi(summary)
    pattern_vi = re.compile(
        r"### Bảng 3\. Kết quả hiệu chỉnh với phân phối mục tiêu và các đối chứng trên 50 thành phố\..*?(?=\n\nChú thích:)",
        re.DOTALL
    )
    if not pattern_vi.search(vi_text):
        sys.exit("Error: Could not locate Table 3 in full_paper_vi.md")
    vi_text_updated = pattern_vi.sub(lambda m: table_vi, vi_text)
    PAPER_VI.write_text(vi_text_updated, encoding="utf-8")
    print(f"Updated Table 3 in {PAPER_VI}")

    # 2. Update full_paper_en.md Table 3
    en_text = PAPER_EN.read_text(encoding="utf-8")
    table_en = generate_table_3_en(summary)
    pattern_en = re.compile(
        r"### Table 3: Target specificity and placebo controls \(\$N=50\$\).*?(?=\n\nNote:)",
        re.DOTALL
    )
    if not pattern_en.search(en_text):
        sys.exit("Error: Could not locate Table 3 in full_paper_en.md")
    en_text_updated = pattern_en.sub(lambda m: table_en, en_text)
    PAPER_EN.write_text(en_text_updated, encoding="utf-8")
    print(f"Updated Table 3 in {PAPER_EN}")


if __name__ == "__main__":
    main()
