# Table E1: Oracle Aggregated-Distance Existence Test (Full 50-City Protocol)

> **Methodological Framing & Amendment Context**:
> *"We report the pooled five-fold out-of-fold benchmark across 50 cities as the primary cross-validated performance summary. Because Fold 1 contributed to protocol development, we additionally report the originally designated Folds 2–5 analysis as a confirmatory sensitivity analysis. Both analyses use five separately trained fold-specific models, and each city is evaluated exactly once when held out."*

### Analysis Sets Hierarchy

| Analysis set | n | Role |
|---|---:|---|
| All Folds 1–5 | 50 | Pooled out-of-fold benchmark |
| Excluding Fold 1 | 40 | Confirmatory sensitivity |
| Fold 1 | 10 | Development/exploratory diagnostic |

**Execution Status**: 50/50 test cities evaluated | is_confirmatory_complete=True | is_full_50_complete=True
**Protocol**: 5-fold stratified city CV (35 train / 5 val / 10 test per fold, locked manifest v2)
**Parameters**: K_move=8 bins (pair-weighted quantile), q=1.0 (within-tolerance calibration, tolerance 10⁻⁵), max_epochs=200, patience=15, std_ddof=1

## E1-A: Primary Pooled Out-of-Fold Benchmark (All Folds 1–5, n=50)

| Condition / Estimand | Interzonal CPC (Mean ± SD) | Mean Δ | Median Δ | IQR | 95% Fold-Stratified Bootstrap CI | Win Rate | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| Zero-Shot Baseline (M₀) | 0.6974 ± 0.0461 | — | — | — | — | — | — |
| + Oracle Y_D (target) | 0.7041 ± 0.0454 | +0.0067 | +0.0039 | 0.0081 | [+0.0052, +0.0082] | 47/50 | 3.30e-13 |
| + Oracle Y_D (wrong donors avg 9) | 0.6641 ± 0.0457 | -0.0333 | -0.0286 | 0.0187 | [-0.0392, -0.0281] | 0/50 | 1.00e+00 |
| **Specificity Gain (Target − Wrong)** | — | **+0.0399** | **+0.0352** | 0.0173 | **[+0.0352, +0.0454]** | **50/50** | **8.88e-16** |

## E1-B: Confirmatory Sensitivity Analysis (Excluding Fold 1: Folds 2–5, n=40)

| Condition / Estimand | Interzonal CPC (Mean ± SD) | Mean Δ | Median Δ | IQR | 95% Fold-Stratified Bootstrap CI | Win Rate | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| Zero-Shot Baseline (M₀) | 0.7008 ± 0.0369 | — | — | — | — | — | — |
| + Oracle Y_D (target) | 0.7082 ± 0.0346 | +0.0075 | +0.0052 | 0.0104 | [+0.0057, +0.0093] | 39/40 | 3.00e-11 |
| + Oracle Y_D (wrong donors avg 9) | 0.6669 ± 0.0227 | -0.0338 | -0.0286 | 0.0181 | [-0.0408, -0.0279] | 0/40 | 1.00e+00 |
| **Specificity Gain (Target − Wrong)** | — | **+0.0413** | **+0.0363** | 0.0159 | **[+0.0360, +0.0477]** | **40/40** | **9.09e-13** |

## E1-C: Per-Fold Independent Training & Evaluation Breakdown

| Fold | Role | Test Cities | Best Epoch | Best Val CPC | Convergence Gate | M₀ CPC | +Target CPC | Mean ΔTarget | Mean ΔWrong (9 Avg) | Specificity Win Rate |
|---|---|---|---|---|---|---|---|---|---|---|
| Fold 1 | Exploratory | 10 | 79 | 0.7018 | PASSED | 0.6840 | 0.6874 | +0.0035 | -0.0310 | 10/10 |
| Fold 2 | Confirmatory | 10 | 69 | 0.6708 | PASSED | 0.6910 | 0.7043 | +0.0133 | -0.0224 | 10/10 |
| Fold 3 | Confirmatory | 10 | 120 | 0.7124 | PASSED | 0.7181 | 0.7200 | +0.0019 | -0.0456 | 10/10 |
| Fold 4 | Confirmatory | 10 | 56 | 0.6868 | PASSED | 0.6783 | 0.6878 | +0.0095 | -0.0356 | 10/10 |
| Fold 5 | Confirmatory | 10 | 132 | 0.6955 | PASSED | 0.7157 | 0.7209 | +0.0053 | -0.0318 | 10/10 |

## Acceptance Criteria Verification (Confirmatory Folds 2–5, n=40)

| Criterion | Required Condition | Observed Value | Verdict |
|---|---|---|---|
| Confirmatory CI Lower Bound | CI_lower > 0 | [+0.0057, +0.0093] | ✓ PASS |
| Specificity Superiority | Mean ΔCPC_target > Mean ΔCPC_wrong | +0.0075 vs -0.0338 (Diff: +0.0413) | ✓ PASS |
| Specificity CI Lower Bound | Specificity CI_lower > 0 | [+0.0360, +0.0477] | ✓ PASS |
| Specificity Significance | Paired Wilcoxon p < 0.05 | p = 9.09e-13 | ✓ PASS |
| City-level Specificity Consistency | Win rate > 70% (>28/40) | 40/40 | ✓ PASS |
