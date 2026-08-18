# Table E1: Oracle Aggregated-Distance Existence Test (Full 50-City Protocol)

> **Methodological Grounding**:
> 1. **Oracle Upper Bound**: Y_D^{GT,+} is an outcome-derived oracle aggregate from target ground truth.
> 2. **Evaluation Split Role**: Folds 2–5 (n=40) serve as the prospectively designated confirmatory test set; Fold 1 (n=10) serves as exploratory; all 50 cities provide full out-of-fold descriptive coverage.

**Execution Status**: 50/50 test cities evaluated | is_confirmatory_complete=True | is_full_50_complete=True
**Protocol**: 5-fold stratified city CV (35 train / 5 val / 10 test per fold)
**Parameters**: K_move=8 bins (pair-weighted quantile), q=1.0 (within-tolerance calibration, tolerance 10⁻⁵), max_epochs=25, patience=5, std_ddof=1

## E1-A: Confirmatory Test Set Outcomes (Prospectively Untouched, Folds 2–5, n=40)

| Condition | Interzonal CPC (Mean ± SD) | Mean ΔCPC | Median ΔCPC | 95% Fold-Stratified Bootstrap CI | Win Rate | Wilcoxon p |
|---|---|---|---|---|---|---|
| Zero-Shot (M₀) | 0.3777 ± 0.0506 | — | — | — | — | — |
| + Oracle Y_D (target) | 0.4414 ± 0.0746 | +0.0637 | +0.0497 | [+0.0509, +0.0783] | 40/40 | 9.09e-13 |
| + Oracle Y_D (wrong donor) | 0.4227 ± 0.0492 | +0.0450 | +0.0432 | [+0.0309, +0.0605] | 35/40 | 5.82e-08 |

## E1-B: Full Out-of-Fold Descriptive Coverage (50 Cities, Folds 1–5)

| Condition | Interzonal CPC (Mean ± SD) | Mean ΔCPC | Median ΔCPC | 95% Fold-Stratified Bootstrap CI | Win Rate | Wilcoxon p |
|---|---|---|---|---|---|---|
| Zero-Shot (M₀) | 0.3731 ± 0.0556 | — | — | — | — | — |
| + Oracle Y_D (target) | 0.4345 ± 0.0768 | +0.0614 | +0.0497 | [+0.0509, +0.0733] | 50/50 | 8.88e-16 |
| + Oracle Y_D (wrong donor) | 0.4155 ± 0.0741 | +0.0424 | +0.0397 | [+0.0302, +0.0552] | 43/50 | 2.22e-09 |

## E1-C: Per-Fold Independent Training & Evaluation Breakdown

| Fold | Role | Test Cities Evaluated | Best Val Epoch | Best Val CPC | M₀ CPC | +Target Y_D CPC | Mean ΔCPC (Target) | Win Rate (Target) | Mean ΔCPC (Wrong) |
|---|---|---|---|---|---|---|---|---|---|
| Fold 1 | Exploratory | 10 | 24 | 0.3725 | 0.3546 | 0.4071 | +0.0525 | 10/10 | +0.0320 |
| Fold 2 | Confirmatory | 10 | 24 | 0.3528 | 0.3666 | 0.4310 | +0.0644 | 10/10 | +0.0483 |
| Fold 3 | Confirmatory | 10 | 24 | 0.3401 | 0.4004 | 0.4574 | +0.0570 | 10/10 | +0.0383 |
| Fold 4 | Confirmatory | 10 | 24 | 0.3403 | 0.3671 | 0.4308 | +0.0637 | 10/10 | +0.0371 |
| Fold 5 | Confirmatory | 10 | 24 | 0.3482 | 0.3769 | 0.4464 | +0.0695 | 10/10 | +0.0563 |

## Acceptance Criteria Verification (Confirmatory Folds 2–5, n=40)

| Criterion | Required Condition | Observed Value | Verdict |
|---|---|---|---|
| Confirmatory CI Lower Bound | CI_lower > 0 | [+0.0509, +0.0783] | ✓ PASS |
| Specificity Superiority | Mean ΔCPC_target > Mean ΔCPC_wrong | +0.0637 vs +0.0450 | ✓ PASS |
| Specificity Significance | Paired Wilcoxon p < 0.05 | p = 1.82e-12 | ✓ PASS |
| City-level Consistency | Win rate > 70% (>28/40) | 40/40 | ✓ PASS |
