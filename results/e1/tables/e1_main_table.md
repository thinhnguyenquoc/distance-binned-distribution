# Table E1: Oracle Aggregated-Distance Existence Test (Exploratory / Smoke Subset)

> **Methodological Grounding**:
> 1. **Oracle Upper Bound**: Y_D^{GT,+} is an outcome-derived oracle aggregate from target ground truth.
> 2. **Evaluation Split Role**: Folds 2–5 (n=40) serve as the prospectively designated confirmatory test set; Fold 1 (n=10) serves as exploratory; all 50 cities provide full out-of-fold descriptive coverage.

**Execution Status**: 2/50 test cities evaluated | is_confirmatory_complete=False | is_full_50_complete=False
**Protocol**: 5-fold stratified city CV (35 train / 5 val / 10 test per fold)
**Parameters**: K_move=8 bins (pair-weighted quantile), q=1.0 (within-tolerance calibration, tolerance 10⁻⁵), max_epochs=25, patience=5, std_ddof=1

## E1-A: Confirmatory Test Set Outcomes (Folds 2–5, n=40)

> *Status: NOT AVAILABLE (2/50 cities run; Confirmatory evaluation strictly requires complete 40 test cities across Folds 2–5).* 

## E1-B: Observed Test Subset (2 Cities)

| Condition | Interzonal CPC (Mean ± SD) | Mean ΔCPC | Median ΔCPC | 95% Fold-Stratified Bootstrap CI | Win Rate | Wilcoxon p |
|---|---|---|---|---|---|---|
| Zero-Shot (M₀) | 0.3979 ± 0.0118 | — | — | — | — | — |
| + Oracle Y_D (target) | 0.4468 ± 0.0210 | +0.0489 | +0.0489 | [+0.0489, +0.0489] | 2/2 | 2.50e-01 |
| + Oracle Y_D (wrong donor) | 0.4363 ± 0.0256 | +0.0383 | +0.0383 | [+0.0383, +0.0383] | 2/2 | 2.50e-01 |

## E1-C: Per-Fold Independent Training & Evaluation Breakdown

| Fold | Role | Test Cities Evaluated | Best Val Epoch | Best Val CPC | M₀ CPC | +Target Y_D CPC | Mean ΔCPC (Target) | Win Rate (Target) | Mean ΔCPC (Wrong) |
|---|---|---|---|---|---|---|---|---|---|
| Fold 4 | Confirmatory | 1 | 24 | 0.3721 | 0.3895 | 0.4320 | +0.0424 | 1/1 | +0.0286 |
| Fold 5 | Confirmatory | 1 | 24 | 0.3666 | 0.4063 | 0.4617 | +0.0554 | 1/1 | +0.0481 |

## Acceptance Criteria Verification

> *Status: PENDING FULL 50-CITY EXECUTION (Evaluated 2/50 cities. Confirmatory criteria will be locked upon full completion).* 
