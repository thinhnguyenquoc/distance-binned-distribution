# Section 4: Empirical Results

In this section, we present the empirical evaluation designed to answer **RQ1** and **RQ2**. Across all experiments, our objective is not to propose a novel calibration algorithm, but to employ a closed-form, mass-preserving calibration operator as an **experimental instrument** to quantify the information value of target-city aggregate distance distributions ($Y_D$).

All evaluations are conducted under a strict 5-fold cross-validation protocol (10 held-out test cities per fold, totaling $N=50$ metropolitan areas across the United States) on the observed positive interzonal support $\Omega_c^+ = \{(i, j) \mid i \ne j, D_{ij} > 0, T_{ij} \ge 1\}$. The headline metric is the Common Part of Commuters (CPC) on interzonal flows, evaluated relative to the frozen zero-shot cross-city baseline $M_0$.

---

## 4.1 Target Distance Information Improves Zero-Shot OD Reconstruction

To address **RQ1**, we evaluate whether conditioning zero-shot predictions on an oracle 8-bin target distance distribution improves interzonal flow reconstruction over a frozen cross-city neural baseline. Across the 50 held-out test cities, the uncalibrated zero-shot baseline ($M_0$) achieves a mean interzonal CPC of $0.71281 \pm 0.04434$. Conditioning these zero-shot predictions on the target city's distance-binned distribution via the mass-preserving scaling operator ($M_1$) increases the mean CPC to $0.71635 \pm 0.04454$. This corresponds to a population-average improvement of $\Delta\text{CPC} = +0.00354$ (median $+0.00195$).

To assess the estimation uncertainty of this effect, we compute a 10,000-iteration fold-stratified bootstrap resampling over test cities. The resulting 95% confidence interval is $[+0.0026, +0.0045]$ ($\text{SE} = 0.00049$). Because this confidence interval is strictly positive and bounded away from zero, the observed improvement cannot be attributed to sampling variability in the choice of test metropolitan areas.

The performance gain is consistent across diverse urban topographies rather than driven by a few isolated outliers. As shown in **Figure 2**, 45 out of the 50 test cities (90.0%) exhibit positive improvements ($\Delta\text{CPC}_c > 0$). In the remaining 5 cities where calibration does not improve accuracy (El Paso, Oklahoma City, Jacksonville, Louisville, and Long Beach), the declines are minor ($\Delta\text{CPC}_c \ge -0.00284$).

A two-sided paired Wilcoxon signed-rank test firmly rejects the null hypothesis of no performance change between uncalibrated and calibrated predictions ($W = 83.0, n = 50, p = 1.93 \times 10^{-9}$, matched-pairs rank-biserial correlation $r_{\text{rb}} = 0.870$). In absolute terms, the effect magnitude is small (representing approximately a 0.5% relative improvement over an already competitive neural baseline), but the improvement is systematic across cities.

---

![Figure 2](figures/fig2_per_city_delta_cpc.png)
**Figure 2 | City-level improvement in interzonal CPC from target-distance calibration.** Bars show the per-city performance change $\Delta\text{CPC}_c = \text{CPC}(M_{1,c}) - \text{CPC}(M_{0,c})$ for $N=50$ held-out test cities, ordered from lowest to highest. The dashed green line indicates the mean improvement ($+0.00354$) and the dotted orange line indicates the median improvement ($+0.00195$). Overall, 45 of 50 cities (90.0%) exhibit positive gains, with the primary fold-stratified 95% confidence interval spanning $[+0.0026, +0.0045]$.

---

### Table 1: Primary Zero-Shot Flow Reconstruction Benchmark ($N=50$ Cities, $K=8$ Bins)

| Model Condition | Mean Interzonal CPC | Median CPC | Mean $\Delta\text{CPC}$ | 95% Fold-Stratified CI | City Win Rate | Wilcoxon $p$ (Two-Sided) |
|---|---|---|---|---|---|---|
| **Zero-Shot Baseline ($M_0$)** | $0.71281 \pm 0.04434$ | $0.71632$ | — | — | — | — |
| **Calibrated Model ($M_1$)** | $0.71635 \pm 0.04454$ | $0.71988$ | **$+0.00354$** | **$[+0.0026, +0.0045]$** | **45 / 50 (90.0%)** | $\mathbf{1.93 \times 10^{-9}}$ |

*Note: Evaluated on observed positive interzonal support $\Omega_c^+$. Confidence interval computed via $B=10,000$ fold-stratified bootstrap over cities. Seed-averaged across 3 independent model seeds.*
