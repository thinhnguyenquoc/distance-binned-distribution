### Table 6: Diagnostic Correlational Analysis Across All 50 Cities

#### Part A: Spearman Rank Correlations with Marginal Gain ($\Delta R^{\text{real},+}, N=50$)
| Predictor / Characteristic | Spearman $\rho_s$ | $p$-value | 95% Bootstrap CI | Statistical Significance |
|---|---|---|---|---|
| **Long-Mass Bias ((p2+p3) - (y2+y3))** | **-0.647** | 3.8656e-07 | [-0.809, -0.418] | p < 0.001 |
| **GT Short-Distance Share (b_1)** | **-0.686** | 3.9462e-08 | [-0.826, -0.483] | p < 0.001 |
| **Number of Tracts (N)** | **+0.574** | 1.2919e-05 | [+0.327, +0.753] | p < 0.001 |
| **Distributional Overlap** | **+0.361** | 9.9732e-03 | [+0.081, +0.622] | p < 0.05 |
| **Zero-Shot Baseline (M0 CPC)** | **-0.111** | 4.4176e-01 | [-0.351, +0.159] | Not Significant |

#### Part B: Diagnostic Inspection of Negative Marginal Gain Cities ($\Delta R^{\text{real},+} < 0, n=8$)

> **Interpretation**: All eight negative cases have exceptionally short-distance-concentrated commuter distributions ($>94\%$ in Bin 1, $<10\text{ km}$), while the Meta mobility prior over-allocates mass to medium/long-distance bins ($15\%–25\%$ in Bin 2/3). Positive oracle-reference gains in all eight negative-real cases support the interpretation that degradation is associated with target-distribution mismatch rather than an inherent inability of the calibration operator to exploit correctly specified bin totals.

| Target City | Tracts ($N$) | Overlap | Real $\Delta R$ | Oracle $\Delta R$ | GT Bin Proportions $[b_1, b_2, b_3]$ | Meta Bin Proportions $[p_1, p_2, p_3]$ | Primary Factor |
|---|---|---|---|---|---|---|---|
| **Long_Beach** | 103 | 84.3% | **-0.0088** | +0.0169 | ['97.1%', '2.9%', '0.0%'] | ['81.4%', '17.0%', '1.7%'] | Short-distance commuter concentration + Meta medium-bin overestimation |
| **Boston** | 178 | 82.4% | **-0.0159** | +0.0100 | ['96.6%', '3.4%', '0.0%'] | ['79.0%', '18.8%', '2.2%'] | Short-distance commuter concentration + Meta medium-bin overestimation |
| **Baltimore** | 200 | 86.5% | **-0.0049** | +0.0083 | ['94.8%', '5.2%', '0.0%'] | ['81.2%', '17.2%', '1.6%'] | Short-distance commuter concentration + Meta medium-bin overestimation |
| **Louisville** | 75 | 82.0% | **-0.0088** | +0.0093 | ['94.1%', '5.9%', '0.0%'] | ['76.1%', '20.8%', '3.1%'] | Short-distance commuter concentration + Meta medium-bin overestimation |
| **Oakland** | 113 | 85.8% | **-0.0056** | +0.0100 | ['94.8%', '5.2%', '0.0%'] | ['80.5%', '17.9%', '1.6%'] | Short-distance commuter concentration + Meta medium-bin overestimation |
| **Minneapolis** | 114 | 82.1% | **-0.0175** | +0.0109 | ['97.2%', '2.8%', '0.0%'] | ['79.3%', '17.7%', '3.0%'] | Short-distance commuter concentration + Meta medium-bin overestimation |
| **San_Francisco** | 193 | 88.3% | **-0.0177** | +0.0023 | ['98.8%', '1.2%', '0.0%'] | ['87.1%', '10.5%', '2.4%'] | Short-distance commuter concentration + Meta medium-bin overestimation |
| **Miami** | 97 | 76.9% | **-0.0404** | +0.0049 | ['98.4%', '1.6%', '0.0%'] | ['75.4%', '21.1%', '3.5%'] | Short-distance commuter concentration + Meta medium-bin overestimation |