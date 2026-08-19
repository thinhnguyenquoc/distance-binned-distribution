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
| Zero-Shot Baseline (M₀) | 0.7048 ± 0.0425 | — | — | — | — | — | — |
| + Oracle Y_D (target) | 0.7098 ± 0.0428 | +0.0050 | +0.0026 | 0.0068 | [+0.0038, +0.0061] | 45/50 | 8.84e-12 |
| + Oracle Y_D (wrong donors avg 9) | 0.6688 ± 0.0439 | -0.0360 | -0.0309 | 0.0166 | [-0.0423, -0.0307] | 0/50 | 1.00e+00 |
| **Specificity Gain (Target − Wrong)** | — | **+0.0410** | **+0.0360** | 0.0181 | **[+0.0359, +0.0470]** | **50/50** | **8.88e-16** |

## E1-B: Confirmatory Sensitivity Analysis (Excluding Fold 1: Folds 2–5, n=40)

| Condition / Estimand | Interzonal CPC (Mean ± SD) | Mean Δ | Median Δ | IQR | 95% Fold-Stratified Bootstrap CI | Win Rate | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| Zero-Shot Baseline (M₀) | 0.7101 ± 0.0357 | — | — | — | — | — | — |
| + Oracle Y_D (target) | 0.7154 ± 0.0352 | +0.0054 | +0.0026 | 0.0075 | [+0.0040, +0.0068] | 36/40 | 1.15e-09 |
| + Oracle Y_D (wrong donors avg 9) | 0.6730 ± 0.0232 | -0.0371 | -0.0315 | 0.0174 | [-0.0442, -0.0310] | 0/40 | 1.00e+00 |
| **Specificity Gain (Target − Wrong)** | — | **+0.0425** | **+0.0365** | 0.0134 | **[+0.0369, +0.0492]** | **40/40** | **9.09e-13** |

## E1-C: Per-Fold Independent Training & Evaluation Breakdown

| Fold | Role | Test Cities | Best Epoch | Best Val CPC | Convergence Gate | M₀ CPC | +Target CPC | Mean ΔTarget | Mean ΔWrong (9 Avg) | Specificity Win Rate |
|---|---|---|---|---|---|---|---|---|---|---|
| Fold 1 | Exploratory | 10 | 59 | 0.7044 | PASSED | 0.6837 | 0.6871 | +0.0033 | -0.0318 | 10/10 |
| Fold 2 | Confirmatory | 10 | 96 | 0.6796 | PASSED | 0.7017 | 0.7116 | +0.0098 | -0.0259 | 10/10 |
| Fold 3 | Confirmatory | 10 | 94 | 0.7087 | PASSED | 0.7164 | 0.7177 | +0.0013 | -0.0462 | 10/10 |
| Fold 4 | Confirmatory | 10 | 149 | 0.6995 | PASSED | 0.6959 | 0.7032 | +0.0073 | -0.0400 | 10/10 |
| Fold 5 | Confirmatory | 10 | 123 | 0.7052 | PASSED | 0.7263 | 0.7293 | +0.0030 | -0.0361 | 10/10 |

## Acceptance Criteria Verification (Confirmatory Folds 2–5, n=40)

| Criterion | Required Condition | Observed Value | Verdict |
|---|---|---|---|
| Confirmatory CI Lower Bound | CI_lower > 0 | [+0.0040, +0.0068] | ✓ PASS |
| Specificity Superiority | Mean ΔCPC_target > Mean ΔCPC_wrong | +0.0054 vs -0.0371 (Diff: +0.0425) | ✓ PASS |
| Specificity CI Lower Bound | Specificity CI_lower > 0 | [+0.0369, +0.0492] | ✓ PASS |
| Specificity Significance | Paired Wilcoxon p < 0.05 | p = 9.09e-13 | ✓ PASS |
| City-level Specificity Consistency | Win rate > 70% (>28/40) | 40/40 | ✓ PASS |
