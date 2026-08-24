# Table: Empirical Subsampling Robustness (Multivariate Hypergeometric Sampling Without Replacement)

> **Evaluation Scope**: Assesses calibration robustness when $ is estimated from empirical subsamples of $ observed interzonal trips without replacement using Multivariate Hypergeometric draws on population bin counts (=8$, =1.0$, seeds  \in \{1, 10, 100\}$) evaluated on five-fold cross-city held-out test splits (=50$ unique cities).

## Master Evaluation Table (=50$ Held-Out Test Cities)

| Observed Sample Size ($) | Mean Empirical TV Error | Calibrated $ CPC | Marginal Gain $\Delta	ext{CPC}$ | 95% Fold-Stratified Bootstrap CI | Cities Improved | Degradation Rate | Relative Gain vs Oracle (%) | Benefit $-value (vs $) | Degradation $-value (vs Oracle) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **100** | 7.96% | 0.70316 | -0.01047 | [-0.01148, -0.00939] | 1 / 50 (2.0%) | 98.0% | -292.9% | 1.0000 | .99 \times 10^{-15}$ |
| **250** | 5.04% | 0.71137 | -0.00225 | [-0.00312, -0.00134] | 10 / 50 (20.0%) | 80.0% | -63.0% | 1.0000 | .99 \times 10^{-15}$ |
| **500** | 3.55% | 0.71425 | +0.00062 | [-0.00025, +0.00154] | 23 / 50 (46.0%) | 54.0% | +17.5% | 1.0000 | .99 \times 10^{-15}$ |
| **1,000 (^*$)** | **2.52%** | **0.71569** | **+0.00206** | **[+0.00117, +0.00299]** | **32 / 50 (64.0%)** | **36.0%** | **+57.8%** | **.0040$** ($<0.05$) | .99 \times 10^{-15}$ |
| **2,500** | 1.60% | 0.71658 | +0.00296 | [+0.00206, +0.00390] | 40 / 50 (80.0%) | 20.0% | +82.9% | .05 \times 10^{-7}$ | .99 \times 10^{-15}$ |
| **5,000** | 1.12% | 0.71689 | +0.00327 | [+0.00236, +0.00421] | 43 / 50 (86.0%) | 14.0% | +91.4% | .11 \times 10^{-8}$ | .99 \times 10^{-15}$ |
| **10,000** | 0.80% | 0.71704 | +0.00342 | [+0.00251, +0.00436] | 44 / 50 (88.0%) | 12.0% | +95.7% | .76 \times 10^{-9}$ | .99 \times 10^{-15}$ |
| **50,000** | 0.36% | 0.71716 | +0.00354 | [+0.00264, +0.00448] | 47 / 50 (94.0%) | 6.0% | +99.1% | .16 \times 10^{-9}$ | .99 \times 10^{-15}$ |
| **100,000** | 0.25% | 0.71718 | +0.00356 | [+0.00265, +0.00450] | 47 / 50 (94.0%) | 6.0% | +99.6% | .16 \times 10^{-9}$ | .99 \times 10^{-15}$ |
| **$\infty$ (Oracle)** | **0.00%** | **0.71720** | **+0.00357** | **[+0.00267, +0.00452]** | **47 / 50 (94.0%)** | **6.0%** | **+100.0%** | **.40 \times 10^{-10}$** | - |

---

### Empirical Parameters
* **Primary Practical Threshold (^*$):** $\mathbf{1{,}000 \text{ observed interzonal trips}}$ ($\text{TV} = 2.52\%$, $\Delta\text{CPC} = \mathbf{+0.00206}$, \%\text{ CI } [+0.00117, +0.00299]$, {\text{Holm}} = \mathbf{0.0040} < 0.05$, retaining $+57.8\%$ of clean oracle gain).
* **Smallest Positive Mean Sample Size ({\text{cross}}$):** $\mathbf{500 \text{ trips}}$ ($\Delta\text{CPC} = +0.00062$, {\text{Holm}} = 1.0000$, non-significant).
