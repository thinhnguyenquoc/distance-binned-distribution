# Table E1: Oracle Aggregated-Distance Existence Test

> **Methodological Note**: Y_D^{GT,+} is an outcome-derived oracle aggregate from target ground truth.
> Results establish an oracle existence upper bound under the proposed protocol.

**Protocol**: n=2 held-out test cities (5-fold stratified CV, 35 train / 5 val / 10 test per fold)
**Parameters**: K_move=8 bins (pair-weighted quantile), q=1.0, max_epochs=25, patience=5, std_ddof=1

## E1-A: Primary Outcomes Across All Test Cities

| Condition | Interzonal CPC (Mean ± SD) | Mean ΔCPC | Median ΔCPC | 95% Fold-Stratified Bootstrap CI | Win Rate | Wilcoxon p |
|---|---|---|---|---|---|---|
| Zero-Shot (M₀) | 0.3469 ± 0.0449 | — | — | — | — | — |
| + Oracle Y_D (target) | 0.3837 ± 0.0599 | +0.0368 | +0.0368 | [+0.0368, +0.0368] | 2/2 | 2.50e-01 |
| + Oracle Y_D (wrong donor) | 0.3768 ± 0.0601 | +0.0298 | +0.0298 | [+0.0298, +0.0298] | 2/2 | 2.50e-01 |

## E1-B: Per-Fold Summary Breakdown

| Fold | Test Cities | Best Val Epoch | Best Val CPC | M₀ CPC | +Target Y_D CPC | Mean ΔCPC (Target) | Win Rate (Target) | Mean ΔCPC (Wrong) |
|---|---|---|---|---|---|---|---|---|
| Fold 4 | 1 | 24 | 0.2998 | 0.3152 | 0.3414 | +0.0262 | 1/1 | +0.0191 |
| Fold 5 | 1 | 24 | 0.3391 | 0.3787 | 0.4260 | +0.0473 | 1/1 | +0.0405 |

## Acceptance Criteria Verification

| Criterion | Required Condition | Observed Value | Verdict |
|---|---|---|---|
| Confirmatory CI Lower Bound | CI_lower > 0 | [+0.0368, +0.0368] | ✓ PASS |
| Specificity Superiority | Mean ΔCPC_target > Mean ΔCPC_wrong | +0.0368 vs +0.0298 | ✓ PASS |
| Specificity Significance | Paired Wilcoxon p < 0.05 | p = 2.50e-01 | ✗ FAIL |
| City-level Consistency | Win rate > 70% (>35/50) | 2/2 | ✗ FAIL |
