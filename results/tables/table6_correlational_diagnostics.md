### Table 6: Exploratory Correlational Diagnostics Across 50 Out-of-Fold Benchmark Cities

> **Diagnostic Scope & Disclaimer**: These exploratory associations are evaluation-time diagnostics (requiring ground-truth reference flows) that characterize observation–target mismatch. They do not establish causality, nor do they constitute an OD-free deployment rule.
> 
> **Collinearity Note**: Because $\sum b_k = 1$, GT Short-Distance Share ($b_1 = y_1$) and Long-Mass Bias ($(p_2+p_3) - (y_2+y_3) = y_1 - p_1$) share algebraic structure and are not independent evidence.

#### Part A: Spearman Rank Correlations with Marginal Gain ($\Delta R^{\text{real},+}, N=50$)
| Characteristic / Metric | Spearman $\rho_s$ | 95% Fold-Stratified Bootstrap CI | Raw $p$-value | Holm-Adjusted $p$ |
|---|---|---|---|---|
| **GT Short-Distance Share (b_1)** | **-0.686** | [-0.836, -0.478] | 3.9462e-08 | **1.9731e-07** |
| **Long-Mass Bias ((p_2+p_3) - (y_2+y_3))** | **-0.647** | [-0.803, -0.415] | 3.8656e-07 | **1.5462e-06** |
| **Number of Zones (Spatial Discretization N)** | **+0.574** | [+0.324, +0.751] | 1.2919e-05 | **3.8757e-05** |
| **Distributional Overlap (Magnitude Overlap)** | **+0.361** | [+0.079, +0.597] | 9.9732e-03 | **1.9946e-02** |
| **Zero-Shot Baseline (M_0 CPC)** | **-0.111** | [-0.358, +0.154] | 4.4176e-01 | **4.4176e-01** |

#### Part B: Diagnostic Inspection of Negative Marginal Gain Cities ($\Delta R^{\text{real},+} < 0, n=8$)

> **Interpretation**: All eight negative cases have exceptionally short-distance-concentrated commuter distributions ($>94\%$ in Bin 1, $<10\text{ km}$), while the Meta mobility prior allocates more mass to medium- and long-distance bins ($15\%–25\%$ in Bin 2/3), reflecting potential differences in temporal support, population coverage, and mobility constructs. Positive oracle-reference gains in all eight negative-real cases support the interpretation that degradation is associated with target-distribution mismatch rather than an inherent inability of the calibration operator to exploit correctly specified bin totals.

| Target City | Zones ($N$) | Overlap | Real $\Delta R$ | Oracle $\Delta R$ | GT Bin Proportions $[b_1, b_2, b_3]$ | Meta Bin Proportions $[p_1, p_2, p_3]$ | Primary Diagnostic Factor |
|---|---|---|---|---|---|---|---|
| **Long_Beach** | 103 | 84.3% | **-0.0088** | +0.0169 | ['97.1%', '2.9%', '0.0%'] | ['81.4%', '17.0%', '1.7%'] | Short-distance commuter concentration ($>94\%$) + Meta medium-bin bias |
| **Boston** | 178 | 82.4% | **-0.0159** | +0.0100 | ['96.6%', '3.4%', '0.0%'] | ['79.0%', '18.8%', '2.2%'] | Short-distance commuter concentration ($>94\%$) + Meta medium-bin bias |
| **Baltimore** | 200 | 86.5% | **-0.0049** | +0.0083 | ['94.8%', '5.2%', '0.0%'] | ['81.2%', '17.2%', '1.6%'] | Short-distance commuter concentration ($>94\%$) + Meta medium-bin bias |
| **Louisville** | 75 | 82.0% | **-0.0088** | +0.0093 | ['94.1%', '5.9%', '0.0%'] | ['76.1%', '20.8%', '3.1%'] | Short-distance commuter concentration ($>94\%$) + Meta medium-bin bias |
| **Oakland** | 113 | 85.8% | **-0.0056** | +0.0100 | ['94.8%', '5.2%', '0.0%'] | ['80.5%', '17.9%', '1.6%'] | Short-distance commuter concentration ($>94\%$) + Meta medium-bin bias |
| **Minneapolis** | 114 | 82.1% | **-0.0175** | +0.0109 | ['97.2%', '2.8%', '0.0%'] | ['79.3%', '17.7%', '3.0%'] | Short-distance commuter concentration ($>94\%$) + Meta medium-bin bias |
| **San_Francisco** | 193 | 88.3% | **-0.0177** | +0.0023 | ['98.8%', '1.2%', '0.0%'] | ['87.1%', '10.5%', '2.4%'] | Short-distance commuter concentration ($>94\%$) + Meta medium-bin bias |
| **Miami** | 97 | 76.9% | **-0.0404** | +0.0049 | ['98.4%', '1.6%', '0.0%'] | ['75.4%', '21.1%', '3.5%'] | Short-distance commuter concentration ($>94\%$) + Meta medium-bin bias |