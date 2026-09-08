# PROTOCOL LOCK - RQ1

This document freezes the methodological protocol and primary results for the Q3 Research Question:
*Does the destination city Y_D improve OD reconstruction conditional on the observed positive OD support?*

## 1. Experimental Contracts
- **Test Set:** 50 independent held-out cities.
- **Cross-Validation:** 5-fold cross-validation (35 train, 5 val, 10 test per fold).
- **Primary Configuration:** K=8 distance bins, evaluated on seeds {1, 10, 100} averaged per city.
- **Primary Treatment:** \M1_city\ (City-level aggregate observation).
- **Secondary Ceiling:** \M1_subzone\ (Fine-grained oracle ceiling, not used as evidence for macroscopic Y_D).
- **Metric:** Interzonal Common Part of Commuters (CPC) on strictly positive support $\Omega_c^+$.

## 2. Locked Primary Results
- Mean $\Delta$ CPC: +0.00354
- Win Rate: 45/50 (90.0%)
- Wilcoxon Two-Sided p-value: $1.9326 \times 10^{-9}$
- Matched-pairs rank-biserial correlation ($r_{\text{rb}}$): 0.8698

## 3. Data Integrity
- The results are sealed via `manifest_rq1_v1.json`.
- Automated tests (T40-T49) enforce these constraints against the pipeline outputs.
- *Audit Trail: Primary benchmark values reconciled with `results/manifest_rq1_v1.json` and `results/5fold_results.json` on 2026-09-01.*
