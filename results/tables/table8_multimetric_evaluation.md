### Table 8: Multi-Metric Evaluation on Interzonal Domain ($\Omega_c^+$, Folds 2–5, $n=40$ Untouched Cities)

> **Sign Conventions**: For CPC and Spearman $\rho_s$, $\Delta > 0$ indicates improvement. For Log-RMSE, $\Delta < 0$ indicates error reduction (improvement).

#### Part A: Gravity-Informed Urban GNN Backbone
| Metric | Zero-Shot ($M_0$) | Calibrated ($M_1^{\text{real},+}$) | Paired $\Delta$ (Mean +- SD) | 95% Fold-Stratified Bootstrap CI | Win Rate (Improved) | Wilcoxon Test |
|---|---|---|---|---|---|---|
| **Interzonal CPC (Primary)** | 0.3954 +- 0.0486 | **0.4227 +- 0.0549** | **+0.0272 +- 0.0316** | [+0.0179, +0.0368] | 82.5% (33/40) | $p_1 = 3.9401e-07$ |
| **Scale-Normalized CPC (1-TVD)** | 0.4861 +- 0.0536 | **0.5231 +- 0.0450** | **+0.0371 +- 0.0336** | [+0.0272, +0.0469] | 82.5% (33/40) | $p_1 = 2.5319e-08$ |
| **Log-RMSE (RMSE_log1p)** | 1.5342 +- 0.3495 | **1.4007 +- 0.2339** | **-0.1335 +- 0.1965** | [-0.1949, -0.0766] | 72.5% (29/40) | $p_1 = 9.7504e-06$ |
| **Pearson Correlation (r)** | 0.2361 +- 0.1288 | **0.2114 +- 0.1669** | **-0.0248 +- 0.1982** | [-0.0865, +0.0337] | 55.0% (22/40) | $p_1 = 6.2770e-01$ |

#### Part B: Classical 2-Parameter Gravity Backbone (Two-Sided Diagnostics)
| Metric | Zero-Shot ($M_0$) | Calibrated ($M_1^{\text{real},+}$) | Paired $\Delta$ (Mean +- SD) | 95% Fold-Stratified Bootstrap CI | Win Rate (Improved) | Wilcoxon Two-Sided $p_2$ |
|---|---|---|---|---|---|---|
| **Interzonal CPC (Primary)** | 0.3120 +- 0.1305 | 0.3054 +- 0.1231 | -0.0065 +- 0.0158 | [-0.0118, -0.0022] | 30.0% (12/40) | $p_2 = 8.2638e-03$ |
| **Scale-Normalized CPC (1-TVD)** | 0.6691 +- 0.0355 | 0.6603 +- 0.0403 | -0.0089 +- 0.0164 | [-0.0141, -0.0043] | 32.5% (13/40) | $p_2 = 1.3675e-03$ |
| **Log-RMSE (RMSE_log1p)** | 1.8144 +- 0.3956 | 1.8362 +- 0.3949 | +0.0218 +- 0.0980 | [-0.0079, +0.0519] | 50.0% (20/40) | $p_2 = 3.2689e-01$ |
| **Pearson Correlation (r)** | 0.6716 +- 0.0655 | 0.6617 +- 0.0729 | -0.0099 +- 0.0229 | [-0.0177, -0.0042] | 25.0% (10/40) | $p_2 = 4.7302e-05$ |

#### Part C: Intra-Bin Pairwise Ranking Diagnostic (Classical Gravity)
| Distance Bin | Evaluated Distance Range | Mean Intra-Bin Spearman $\rho_k$ (Mean +- SD) | Interpretation |
|---|---|---|---|
| **Bin 1** | $0 < D \le 10\text{ km}$ | 0.645 +- 0.060 | Moderate local ranking |
| **Bin 2** | $10 < D \le 40\text{ km}$ | 0.416 +- 0.098 | Weak intra-bin ordering |
| **Bin 3** | $40 < D \le 100\text{ km}$ | 0.000 +- 0.000 | High variance in long-distance pairs |