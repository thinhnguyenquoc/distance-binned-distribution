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
| Zero-Shot Baseline (M₀) | 0.6922 ± 0.0476 | — | — | — | — | — | — |
| + Oracle Y_D (target) | 0.6988 ± 0.0473 | +0.0067 | +0.0041 | 0.0077 | [+0.0051, +0.0082] | 49/50 | 2.66e-15 |
| + Oracle Y_D (wrong donors avg 9) | 0.6591 ± 0.0477 | -0.0331 | -0.0296 | 0.0224 | [-0.0388, -0.0281] | 1/50 | 1.00e+00 |
| **Specificity Gain (Target − Wrong)** | — | **+0.0398** | **+0.0352** | 0.0164 | **[+0.0352, +0.0449]** | **50/50** | **8.88e-16** |

## E1-B: Confirmatory Sensitivity Analysis (Excluding Fold 1: Folds 2–5, n=40)

| Condition / Estimand | Interzonal CPC (Mean ± SD) | Mean Δ | Median Δ | IQR | 95% Fold-Stratified Bootstrap CI | Win Rate | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| Zero-Shot Baseline (M₀) | 0.7047 ± 0.0361 | — | — | — | — | — | — |
| + Oracle Y_D (target) | 0.7116 ± 0.0352 | +0.0068 | +0.0041 | 0.0072 | [+0.0051, +0.0085] | 39/40 | 2.73e-12 |
| + Oracle Y_D (wrong donors avg 9) | 0.6699 ± 0.0228 | -0.0349 | -0.0308 | 0.0221 | [-0.0417, -0.0292] | 1/40 | 1.00e+00 |
| **Specificity Gain (Target − Wrong)** | — | **+0.0417** | **+0.0368** | 0.0158 | **[+0.0366, +0.0480]** | **40/40** | **9.09e-13** |

## E1-C: Per-Fold Independent Training & Evaluation Breakdown

| Fold | Role | Test Cities | Best Epoch | Best Val CPC | Convergence Gate | M₀ CPC | +Target CPC | Mean ΔTarget | Mean ΔWrong (9 Avg) | Specificity Win Rate |
|---|---|---|---|---|---|---|---|---|---|---|
| Fold 1 | Exploratory | 10 | 39 | 0.6734 | PASSED | 0.6420 | 0.6480 | +0.0060 | -0.0261 | 10/10 |
| Fold 2 | Confirmatory | 10 | 72 | 0.6721 | PASSED | 0.6895 | 0.7042 | +0.0147 | -0.0210 | 10/10 |
| Fold 3 | Confirmatory | 10 | 148 | 0.7061 | PASSED | 0.7109 | 0.7125 | +0.0016 | -0.0444 | 10/10 |
| Fold 4 | Confirmatory | 10 | 139 | 0.7048 | PASSED | 0.7048 | 0.7106 | +0.0058 | -0.0425 | 10/10 |
| Fold 5 | Confirmatory | 10 | 110 | 0.6880 | PASSED | 0.7137 | 0.7190 | +0.0052 | -0.0317 | 10/10 |

## Acceptance Criteria Verification (Confirmatory Folds 2–5, n=40)

| Criterion | Required Condition | Observed Value | Verdict |
|---|---|---|---|
| Confirmatory CI Lower Bound | CI_lower > 0 | [+0.0051, +0.0085] | ✓ PASS |
| Specificity Superiority | Mean ΔCPC_target > Mean ΔCPC_wrong | +0.0068 vs -0.0349 (Diff: +0.0417) | ✓ PASS |
| Specificity CI Lower Bound | Specificity CI_lower > 0 | [+0.0366, +0.0480] | ✓ PASS |
| Specificity Significance | Paired Wilcoxon p < 0.05 | p = 9.09e-13 | ✓ PASS |
| City-level Specificity Consistency | Win rate > 70% (>28/40) | 40/40 | ✓ PASS |
