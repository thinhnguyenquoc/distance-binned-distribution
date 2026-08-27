# FROZEN EXPERIMENTAL RESULTS — CANONICAL BENCHMARK
**Project**: Distance-Binned Distribution (DBD) Mobility Calibration  
**Repository**: `d:\DBD\distance-binned-distribution`  
**Status**: **FROZEN & VERIFIED** (Provenance Audit: 259/259 checks PASS | Pytest: 58/58 PASS)  
**Date Locked**: 2026-08-27  

---

## 1. Executive Summary & Canonical Evidence Package

This document serves as the immutable **Single Source of Truth** for all numerical values, statistical tests, confidence intervals, and tables reported in the manuscript. All quantities have been verified directly from frozen model checkpoints (`results/checkpoints/5fold_fold{1-5}_seed{1,10,100}.pt`) and ground-truth mobility data.

```
+----------------------------------------------------------------------------------------------------+
|                                    CANONICAL RESULTS AT A GLANCE                                   |
+------------------------------------+------------------------------------+--------------------------+
| Metric                             | Canonical Value                    | Statistical Significance |
+------------------------------------+------------------------------------+--------------------------+
| Zero-shot Baseline CPC (M0)        | 0.71281 ± 0.04434                  | 50 cities × 3 seeds      |
| Calibrated Intensity CPC (M1)      | 0.71635 ± 0.04454                  | K = 8 distance bins      |
| Primary Effect Size (ΔCPC)         | +0.00354 (median +0.00195)         | W = 83.0, p = 1.93 × 10⁻⁹|
| Primary 95% Confidence Interval    | [+0.0026, +0.0045]                 | Fold-stratified B=10,000 |
| City Win Rate                      | 45 / 50 (90.0%)                    | Rank-biserial r = 0.870  |
| Specificity vs Dose-Matched Donors | +0.00363, CI [+0.0029, +0.0045]    | p = 2.19 × 10⁻¹¹         |
| Noise Crossover Threshold (TV)     | 4.45% TV [95% CI: 4.16%, 4.77%]    | B=1000 noisy realizations|
| Mechanism Explanatory Power (d_pre)| r = +0.7951 (partial r = +0.7963)  | p = 5.35 × 10⁻¹²         |
| Flow Conservation Invariant        | Error < 3.72 × 10⁻¹⁶               | Exact to machine epsilon |
| Intra-bin Rank Invariance          | Kendall τ = 1.00000000             | Exact mathematical proof |
+------------------------------------+------------------------------------+--------------------------+
```

---

## 2. RQ1: Main Effect (Primary Benchmark)

### 2.1 Overall Performance ($N=50$ Cities $\times$ 3 Model Seeds, $K=8$)
- **Domain**: Observed positive interzonal support $\Omega_c^+ = \{(i,j) \in \Omega_c : i \ne j, D_{ij} > 0, T_{ij}^{\text{GT}} \ge 1\}$.
- **Cross-City Validation**: 5-fold cross-validation (10 held-out test cities per fold; 35 train / 5 val / 10 test).
- **Baseline ($M_0$)**: Zero-shot Gravity-Informed Urban GNN: $\text{CPC} = \mathbf{0.71281 \pm 0.04434}$.
- **Calibrated ($M_1$)**: Target-conditioned inference: $\text{CPC} = \mathbf{0.71635 \pm 0.04454}$.
- **Mean Improvement ($\Delta\text{CPC}$)**: $\mathbf{+0.00353949}$ (Reported: $\mathbf{+0.00354}$).
- **Median Improvement**: $\mathbf{+0.00195311}$ (Reported: $\mathbf{+0.00195}$).
- **Primary 95% Confidence Interval**: $\mathbf{[+0.0026, +0.0045]}$
  - *Fold-stratified city-level bootstrap* ($B=10,000$): $[+0.002611, +0.004511]$
  - *Fold-stratified hierarchical city $\times$ seed bootstrap* ($B=10,000$): $[+0.002591, +0.004509]$
  - *Standard Error*: $\text{SE} = 0.000494$
- **Hypothesis Test**: Two-sided paired Wilcoxon signed-rank test:
  - $W = 83.0$, $W^+ = 1192.0$, $n = 50$, $\mathbf{p = 1.9326 \times 10^{-9}}$.
  - Matched-pairs rank-biserial correlation: $r_{\text{rb}} = \frac{1192 - 83}{1275} = \mathbf{0.8698}$.
- **City Win Rate**: $\mathbf{45 / 50 \ (90.0\%)}$.
  - Exactly 5 cities show no improvement: `El_Paso` ($-0.00284$), `Oklahoma_City` ($-0.00261$), `Jacksonville` ($-0.00155$), `Louisville` ($-0.00022$), `Long_Beach` ($-0.00010$).
  - Maximum city gain: `Los_Angeles` ($+0.01543$).

### 2.2 Seed Robustness (Seeds 1, 10, 100)
| Seed | Mean $M_0$ CPC | Mean $M_1$ CPC | Mean $\Delta\text{CPC}$ | Median $\Delta\text{CPC}$ | 95% Fold-Stratified CI | Win Rate |
|---|---|---|---|---|---|---|
| **Seed 1** | 0.708606 | 0.712950 | **+0.004344** | +0.002074 | [+0.00322, +0.00547] | 41/50 (82.0%) |
| **Seed 10** | 0.714774 | 0.717850 | **+0.003077** | +0.001824 | [+0.00216, +0.00404] | 44/50 (88.0%) |
| **Seed 100** | 0.715042 | 0.718240 | **+0.003198** | +0.002167 | [+0.00236, +0.00408] | 44/50 (88.0%) |
| **Seed-Averaged** | **0.712807** | **0.716347** | **+0.003539** | **+0.001953** | **[+0.0026, +0.0045]** | **45/50 (90.0%)** |

- Across-seed standard deviation of mean $\Delta\text{CPC}$: $\text{SD} = \mathbf{0.000699} \approx 0.0007$.
- Mean per-city seed standard deviation: $\text{SD}_{\text{city}} = \mathbf{0.001264}$.

### 2.3 Backbone Model Generality
| Architecture | Zero-Shot $M_0$ CPC | Calibrated $M_1$ CPC | Mean $\Delta\text{CPC}$ | 95% Bootstrap CI | Win Rate | Wilcoxon $p$ | $\Delta\text{RMSE}$ |
|---|---|---|---|---|---|---|---|
| **Urban GNN (Message-Passing)** | 0.71281 ± 0.04434 | 0.71635 ± 0.04454 | **+0.003539** | [+0.0026, +0.0045] | **45/50 (90%)** | $1.93 \times 10^{-9}$ | -2.9826 |
| **Node MLP (No Graph MP)** | 0.70913 ± 0.04754 | 0.71242 ± 0.04737 | **+0.003288** | [+0.0025, +0.0042] | **47/50 (94%)** | $4.38 \times 10^{-11}$ | -2.5714 |
| **Classical 2-Param Gravity** | 0.38868 ± 0.15312 | 0.38952 ± 0.15435 | **+0.000835** | [+0.0002, +0.0016] | 22/50 (44%) | $0.3545$ (n.s.) | -0.9335 |

---

## 3. RQ2: Information Resolution ($K$-Sensitivity)

### 3.1 Scaling Behavior Across Distance Resolutions ($K \in \{2, 4, \dots, 20\}$)
| Resolution ($K$) | Mean $\Delta\text{CPC}$ | Median $\Delta\text{CPC}$ | 95% Bootstrap CI | Win Rate | Marginal Gain / Bin | $K_{\text{active}} / K$ |
|---|---|---|---|---|---|---|
| **$K = 2$** | +0.000976 | +0.000335 | [+0.00052, +0.00151] | 39/50 (78.0%) | 0.000488 | 2.0 / 2 |
| **$K = 4$** | +0.001976 | +0.000877 | [+0.00125, +0.00279] | 39/50 (78.0%) | 0.000494 | 4.0 / 4 |
| **$K = 6$** | +0.002888 | +0.001516 | [+0.00201, +0.00384] | 44/50 (88.0%) | 0.000481 | 6.0 / 6 |
| **$K = 8$ (Anchor)** | **+0.003539** | **+0.001953** | **[+0.00262, +0.00447]** | **45/50 (90.0%)** | **0.000442** | **8.0 / 8** |
| **$K = 10$** | +0.004130 | +0.002347 | [+0.00311, +0.00514] | 45/50 (90.0%) | 0.000413 | 10.0 / 10 |
| **$K = 12$** | +0.004796 | +0.002882 | [+0.00372, +0.00590] | 46/50 (92.0%) | 0.000400 | 12.0 / 12 |
| **$K = 14$** | +0.005378 | +0.003726 | [+0.00424, +0.00654] | 46/50 (92.0%) | 0.000384 | 14.0 / 14 |
| **$K = 16$** | +0.005741 | +0.004326 | [+0.00455, +0.00694] | 46/50 (92.0%) | 0.000359 | 16.0 / 16 |
| **$K = 18$** | +0.006028 | +0.004576 | [+0.00480, +0.00726] | 47/50 (94.0%) | 0.000335 | 18.0 / 18 |
| **$K = 20$** | +0.006387 | +0.004944 | [+0.00508, +0.00769] | 47/50 (94.0%) | 0.000319 | 20.0 / 20 |

- **Monotonic Total Gain**: $\Delta\text{CPC}$ increases strictly monotonically from $+0.00098$ ($K=2$) to $+0.00639$ ($K=20$).
- **Marginal Return Profile**: Marginal gain per additional bin increases slightly from $K=2 \to 4$ ($0.000488 \to 0.000494$), then steadily declines beyond $K=4$ down to $0.000319$ at $K=20$.
- **Aggregation Ratio**: Even at $K=20$, each scalar bin aggregates an average of $\approx 1,757$ OD pairs ($K / |\Omega_c^+| = 0.000569 \ll 1$), maintaining strict macro-level privacy.

---

## 4. RQ2: Target Specificity & Placebo Controls

### 4.1 Controlled Experimental Comparison ($N=50$ Cities $\times$ 3 Seeds, $B=1000$ Resamples)
| Experimental Condition | Mean $\Delta\text{CPC}$ | 95% Fold-Stratified CI | Benefit vs $M_0$ ($p_{\text{2-sided}}$) | Specificity Gain vs Placebo | Specificity 95% CI | Target vs Placebo ($p_{\text{1-sided}}$) | Win Rate ($Target > Placebo$) |
|---|---|---|---|---|---|---|---|
| **1. Oracle Target $Y_D$** | **+0.003539** | **[+0.00260, +0.00450]** | $\mathbf{1.93 \times 10^{-9}}$ | — | — | — | **45/50 (vs M0)** |
| **2. Dose-Matched Wrong Donors ($B=1000$)** | **-0.000091** | [-0.00089, +0.00071] | $0.4097$ (n.s.) | **+0.003630** | [+0.00287, +0.00445] | $\mathbf{2.19 \times 10^{-11}}$ | **46/50 (92.0%)** |
| **3. Dose-Matched Train-Mean $Y_D$** | **+0.000914** | [+0.00001, +0.00186] | $0.4319$ (n.s.) | **+0.002626** | [+0.00197, +0.00336] | $\mathbf{4.03 \times 10^{-11}}$ | **47/50 (94.0%)** |
| **4. Raw Test Donors (E1-v2 9 Donors)** | **-0.037721** | [-0.04357, -0.03268] | $1.78 \times 10^{-15}$ | **+0.041261** | [+0.03641, +0.04688] | $8.88 \times 10^{-16}$ | **50/50 (100%)** |
| **5. Raw Test Donors ($B=1000$ Draws)** | **-0.037787** | [-0.04358, -0.03278] | $1.78 \times 10^{-15}$ | **+0.041326** | [+0.03646, +0.04688] | $8.88 \times 10^{-16}$ | **50/50 (100%)** |
| **6. Raw Training Donors ($B=1000$ Draws)** | **-0.035148** | [-0.04014, -0.03067] | $1.78 \times 10^{-15}$ | **+0.038687** | [+0.03431, +0.04349] | $8.88 \times 10^{-16}$ | **50/50 (100%)** |
| **7. Raw Train-Mean $Y_D$** | **-0.017735** | [-0.02365, -0.01243] | $4.91 \times 10^{-12}$ | **+0.021275** | [+0.01613, +0.02706] | $4.44 \times 10^{-15}$ | **48/50 (96.0%)** |
| **8. Permuted Target $Y_D$ ($B=1000$ Draws)** | **-0.006964** | [-0.00914, -0.00512] | $1.78 \times 10^{-15}$ | **+0.010504** | [+0.00843, +0.01279] | $1.78 \times 10^{-15}$ | **49/50 (98.0%)** |

### 4.2 Scientific Interpretation of Placebo Benchmarks
1. **Primary Specificity (Dose-Matched Controls)**: When normalized to the exact L2 perturbation dose of the target ($D_T$), incorrect cross-city signals yield zero systematic gain ($\Delta\text{CPC} = -0.000091, p = 0.4097$). The target signal significantly outperforms dose-matched wrong donors on $46/50$ cities ($+0.003630, p = 2.19 \times 10^{-11}$).
2. **Generic Prior vs Target-Specific Signal**: A dose-matched training-mean distribution captures generic gravity decay producing a mild mean gain ($+0.000914$), but lacks systematic pairwise consistency ($p = 0.4319$). Target-specific $Y_D$ provides a substantial incremental advantage ($+0.002626, p = 4.03 \times 10^{-11}$).
3. **Macro-Scale Distortion (Raw Donors)**: Unmatched raw donor distributions from other cities cause catastrophic spatial distortion ($\Delta\text{CPC} \approx -0.038, p < 10^{-14}$) due to incompatible spatial extents and city diameters.

---

## 5. RQ2: Perturbation Tolerance (Noise Robustness)

### 5.1 Fine Total Variation (TV) Error Sweep ($0.0\% \dots 5.0\%$)
| TV Noise Level ($\epsilon$) | Mean $M_1$ CPC | Mean $\Delta\text{CPC}$ | 95% Bootstrap CI | Positive Cities | Degradation vs Clean ($p$) |
|---|---|---|---|---|---|
| **$\epsilon = 0.00$ (Clean)** | 0.71635 | **+0.00354** | [+0.00261, +0.00451] | **45 / 50 (90%)** | — |
| **$\epsilon = 0.01$ (1% TV)** | 0.71617 | **+0.00336** | [+0.00243, +0.00432] | **44 / 50 (88%)** | $4.44 \times 10^{-15}$ |
| **$\epsilon = 0.02$ (2% TV)** | 0.71563 | **+0.00282** | [+0.00189, +0.00379] | **36 / 50 (72%)** | $4.44 \times 10^{-15}$ |
| **$\epsilon = 0.03$ (3% TV)** | 0.71474 | **+0.00193** | [+0.00100, +0.00290] | **28 / 50 (56%)** | $4.44 \times 10^{-15}$ |
| **$\epsilon = 0.04$ (4% TV)** | 0.71351 | **+0.00070** | [-0.00025, +0.00167] | 18 / 50 (36%) | $4.44 \times 10^{-15}$ |
| **$\epsilon = 0.05$ (5% TV)** | 0.71193 | **-0.00087** | [-0.00183, +0.00012] | 17 / 50 (34%) | $4.44 \times 10^{-15}$ |

### 5.2 Signal Breakdown Threshold ($\epsilon_{\text{cross}}$)
- **Mean Crossover Threshold across $B=1,000$ Noise Directions**: $\mathbf{\epsilon_{\text{cross}} = 4.45\% \ [95\%\text{ CI}: 4.16\%, 4.77\%]}$.
- **Mean Crossover Threshold across $B=10,000$ City Resamples**: $\mathbf{\epsilon_{\text{cross}} = 4.39\% \ [95\%\text{ CI}: 3.66\%, 4.94\%]}$.
- **Practical Implication**: Under the synthetic TV perturbation protocol, positive utility is preserved up to $\approx 4.4\%$ Total Variation estimation error in the target aggregate distribution.

---

## 6. RQ2: Explanatory Mechanism

### 6.1 Distance-Distribution Mismatch Diagnostic ($d_{\text{pre}}$)
- **Definition**: $d_{\text{pre}} = \text{TV}(\hat{Y}_D^{(0)}, Y_D^{\text{GT}}) = \frac{1}{2} \sum_{k=1}^K |\hat{Y}_k^{(0)} - Y_k^{\text{GT}}|$.
- **Correlation with $\Delta\text{CPC}$**:
  - Pearson correlation: $\mathbf{r = +0.7951 \ (p = 5.35 \times 10^{-12})}$.
  - Spearman rank correlation: $\mathbf{\rho = +0.7644 \ (p = 7.73 \times 10^{-11})}$.
- **Partial Correlation Controlling for Baseline Scale ($M_0$ CPC & Log Total Flow)**:
  - Partial Pearson $r$: $\mathbf{r_{\text{partial}} = +0.7963 \ (p = 5.21 \times 10^{-12})}$.
  - $R^2$ increment when adding $d_{\text{pre}}$ to baseline covariates: $\mathbf{\Delta R^2 = +0.6322}$ (from $R^2 = 0.0019 \to 0.6341, F = 79.52, p = 5.21 \times 10^{-12}$).
- **Intra-Bin Ranking Quality ($Q_c^{\text{intra}}$)**:
  - Correlation with $\Delta\text{CPC}$: $r = +0.046 \ (p = 0.75)$, showing that overall gain is driven by macro-level distance correction rather than intra-bin baseline fidelity.

---

## 7. Mathematical Invariants & Implementation Integrity

1. **Mass Preservation**: Relative error in total predicted interzonal flow $\frac{|\sum_{\Omega_c^+} T_1 - \sum_{\Omega_c^+} T_0|}{\sum_{\Omega_c^+} T_0} \le \mathbf{3.72 \times 10^{-16}}$ across all 50 cities and 3 seeds.
2. **Support Preservation**: Evaluated support is strictly identical: $\text{supp}(T_1) \equiv \text{supp}(T_0) \equiv \Omega_c^+$. No artificial zero-filling or spurious edge creation.
3. **Intra-Bin Rank Invariance**: Because moving-bin scaling applies a strictly positive scalar $s_k > 0$ per distance bin $k$, pair orderings within every bin are mathematically preserved: $\text{Kendall } \tau(T_1|_{\text{bin } k}, T_0|_{\text{bin } k}) = \mathbf{1.00000000}$.
4. **Calibration Weight Profile**: For all 50 cities, calibration weights satisfy $w_{\min} < 1.0$ (mean $0.7546$, range $[0.224, 0.976]$) and $w_{\max} > 1.0$ (mean $1.3102$, range $[1.017, 3.345]$).

---

## 8. Verification & Sign-off

- **Data Provenance**: Verified by independent re-inference script `src/experiment/audit_data_provenance.py` (259/259 checks PASS).
- **Test Suite**: Verified by `pytest tests/` (58/58 unit tests PASS).
- **Master Contract**: Frozen in `manuscript_blueprint.md`.

*This document constitutes the canonical, frozen benchmark for all paper drafting.*
