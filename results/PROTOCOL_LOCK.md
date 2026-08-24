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
- Mean $\Delta$ CPC: +0.00357
- Win Rate: 47/50 (94.0%)
- Wilcoxon Two-Sided p-value: .79 \times 10^{-10}$
- Matched-pairs rank-biserial correlation ({rb}$): 0.892

## 3. Data Integrity
- The results are sealed via \manifest_rq1_v1.json\.
- Automated tests (T40-T49) enforce these constraints against the pipeline outputs.
