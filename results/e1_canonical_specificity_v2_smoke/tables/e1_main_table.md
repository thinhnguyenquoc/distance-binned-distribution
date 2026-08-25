# Table E1: Oracle Aggregated-Distance Existence Test (Exploratory / Smoke Subset)

> **Methodological Framing & Amendment Context**:
> *"We report the pooled five-fold out-of-fold benchmark across 50 cities as the primary cross-validated performance summary. Because Fold 1 contributed to protocol development, we additionally report the originally designated Folds 1-5 analysis as a full_5_fold sensitivity analysis. Both analyses use five separately trained fold-specific models, and each city is evaluated exactly once when held out."*

### Analysis Sets Hierarchy

| Analysis set | n | Role |
|---|---:|---|
| All Folds 1–5 | 50 | Pooled out-of-fold benchmark |
| Excluding Fold 1 | 40 | Full 5-fold sensitivity |
| Fold 1 | 10 | Development/exploratory diagnostic |

**Execution Status**: 1/50 test cities evaluated | is_full_5_fold_complete=False | is_full_50_complete=False
**Protocol**: 5-fold stratified city CV (35 train / 5 val / 10 test per fold, locked manifest v2)
**Parameters**: K_move=8 bins (pair-weighted quantile), q=1.0 (within-tolerance calibration, tolerance 10⁻⁵), max_epochs=200, patience=15, std_ddof=0

## E1-A: Primary Benchmark (Observed 1 Cities)

| Condition / Estimand | Interzonal CPC (Mean ± SD) | Mean Δ | Median Δ | IQR | 95% Fold-Stratified Bootstrap CI | Win Rate | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| Zero-Shot Baseline (M₀) | 0.7822 ± 0.0000 | — | — | — | — | — | — |
| + Oracle Y_D (target) | 0.7855 ± 0.0000 | +0.0032 | +0.0032 | 0.0000 | [+0.0032, +0.0032] | 1/1 | 1.00e+00 |
| + Oracle Y_D (wrong donors avg 9) | 0.7347 ± 0.0000 | -0.0475 | -0.0475 | 0.0000 | [-0.0475, -0.0475] | 0/1 | 1.00e+00 |
| **Specificity Gain (Target − Wrong)** | — | **+0.0507** | **+0.0507** | 0.0000 | **[+0.0507, +0.0507]** | **1/1** | **1.00e+00** |

## E1-B: Full 5-fold Sensitivity Analysis (Excluding Fold 1: Folds 1-5, n=50)

> *Status: NOT AVAILABLE (Observed 0/50 test cities; Full 5-fold evaluation strictly requires complete 50 test cities across Folds 1-5, with 10 test cities per fold).* 

## E1-C: Per-Fold Independent Training & Evaluation Breakdown

| Fold | Role | Test Cities | Best Epoch | Best Val CPC | Convergence Gate | M₀ CPC | +Target CPC | Mean ΔTarget | Mean ΔWrong (9 Avg) | Specificity Win Rate |
|---|---|---|---|---|---|---|---|---|---|---|
| Fold 1 | Exploratory | 1 | None | — | — | 0.7822 | 0.7855 | +0.0032 | -0.0475 | 1/1 |

## Acceptance Criteria Verification

> *Status: PENDING FULL 50-CITY EXECUTION (Evaluated 1/50 cities. Full 5-fold criteria will be locked upon full completion).* 
