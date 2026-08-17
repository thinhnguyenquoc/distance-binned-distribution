### Table 7: Backbone Robustness — Marginal Value of Moving-Bin Calibration Across Model Architectures

> **Evaluation Scope**: Assesses whether coarse mobility information ($Y_D^{\text{Meta},+}$) improves interzonal reconstruction across different zero-shot model families, distinguishing general operator value from model-specific error correction.

#### Part A: Confirmatory Evaluation Set (Folds 2–5, $n=40$ Untouched Cities)
| Backbone Architecture | Zero-Shot $M_0$ CPC | Calibrated $M_1^{\text{real},+}$ CPC | Marginal Gain $\Delta R$ | 95% Fold-Stratified Bootstrap CI | $P(\Delta R > 0)$ | Wilcoxon $p_1$ | $\Delta \text{RMSE}_{\log1p}$ |
|---|---|---|---|---|---|---|---|
| **Classical 2-Parameter Gravity** | 0.3120 +- 0.1305 | **0.3054 +- 0.1231** | **-0.0065 +- 0.0158** | [-0.0118, -0.0022] | 30.0% (12/40) | p = 9.9604e-01 | +0.0218 |
| **Gravity-Informed Urban GNN** | 0.3954 +- 0.0486 | **0.4227 +- 0.0549** | **+0.0272 +- 0.0316** | [+0.0179, +0.0368] | 82.5% (33/40) | p = 3.9401e-07 | -0.1335 |

#### Part B: Full Out-of-Fold Descriptive Set ($N=50$ Cities)
| Backbone Architecture | Zero-Shot $M_0$ CPC | Calibrated $M_1^{\text{real},+}$ CPC | Marginal Gain $\Delta R$ | 95% Fold-Stratified Bootstrap CI | $P(\Delta R > 0)$ | Wilcoxon $p_1$ |
|---|---|---|---|---|---|---|
| **Classical 2-Parameter Gravity** | 0.3468 +- 0.1430 | **0.3401 +- 0.1365** | **-0.0067 +- 0.0154** | [-0.0115, -0.0029] | 34.0% (17/50) | p = 9.9770e-01 |
| **Gravity-Informed Urban GNN** | 0.3904 +- 0.0543 | **0.4169 +- 0.0587** | **+0.0265 +- 0.0291** | [+0.0185, +0.0346] | 84.0% (42/50) | p = 4.4852e-09 |