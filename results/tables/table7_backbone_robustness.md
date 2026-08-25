# Backbone Robustness — Marginal Value of Calibration Across Model Architectures

> **Evaluation Scope**: Assesses whether distance-binned aggregate information ($Y_D^{\text{target}}$) improves interzonal reconstruction across different zero-shot model families.

## Part A: Full 5-fold Evaluation Set (Folds 1-5, $n=50$ Cities)
| Backbone Architecture | Zero-Shot $M_0$ CPC | Calibrated $M_1$ CPC | Marginal Gain $\Delta R$ | 95% Fold-Stratified Bootstrap CI | $P(\Delta R > 0)$ | Wilcoxon $p$ | $\Delta \text{RMSE}$ |
|---|---|---|---|---|---|---|---|
| **Classical 2-Parameter Gravity** | 0.3887 +- 0.1531 | **0.3895 +- 0.1544** | **+0.0008 +- 0.0034** | [+0.0002, +0.0016] | 44.0% (22/50) | p = 3.5448e-01 | -0.9335 |
| **Gravity-Informed Urban GNN** | 0.7128 +- 0.0443 | **0.7163 +- 0.0445** | **+0.0035 +- 0.0042** | [+0.0026, +0.0045] | 90.0% (45/50) | p = 9.6632e-10 | -2.9826 |