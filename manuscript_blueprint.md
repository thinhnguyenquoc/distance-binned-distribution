# Manuscript Blueprint: Information Value of Distance-Binned Observations (FROZEN)

This document establishes the locked scientific framing, research questions, hypotheses, figure storyboard, and Results structure for the manuscript. All experimental numbers, estimators, and protocol boundaries are frozen.

---

## 1. Core Contribution Statement

> **"We quantify the incremental information value of target-specific aggregate distance observations for cross-city OD intensity reconstruction."**

### Key Scientific Takeaways:
1. **Systematic Information Value**: Oracle aggregate distance distributions ($Y_D$) provide a modest but highly consistent performance improvement across diverse urban topologies ($\Delta\text{CPC} = +0.00354 \ [95\%\text{ CI}: +0.0026, +0.0045], 45/50\text{ cities positive}, p = 1.93 \times 10^{-9}$).
2. **Directional Specificity**: Performance gain strictly requires target-specific distance information. Applying identical correction doses ($D_T$) in non-target directions yields no benefit ($\Delta\text{CPC} \approx -0.000091$).
3. **Information Quality & Baseline Coupling**: The value of the aggregate signal depends monotonically on its accuracy and resolution, and is largest in cities where the zero-shot baseline's aggregate distance allocation is most misaligned ($r_{\text{partial}} = +0.7951$).

---

## 2. Research Questions (Locked)

### RQ1 — Main Effect
> **Does an oracle target-city distance-binned mobility distribution improve support-conditioned zero-shot OD intensity reconstruction beyond a frozen cross-city baseline?**
>
> *Estimands*:
> - Per-city gain: $\Delta\text{CPC}_c = \text{CPC}(M_{1,c}) - \text{CPC}(M_{0,c})$
> - Population average effect: $\mathbb{E}_c[\Delta\text{CPC}_c]$

### RQ2 — Conditions, Specificity & Mechanisms
> **Under what conditions is the improvement larger, and does the gain depend on target-specific distance information rather than generic or incorrectly directed aggregate distance signals?**
>
> Subsumes three coupled dimensions:
> 1. *Resolution*: How does spatial partition granularity ($K \in \{2 \dots 20\}$) affect gain?
> 2. *Quality*: At what noise corruption level ($\epsilon \in [0, 0.05]$) does information utility break down?
> 3. *Specificity*: Does the gain require the true target direction, or does any perturbation of equal dose suffice?

---

## 3. Formal Hypotheses (Locked)

* **H1 (Main Headline Effect)**: $\mathbb{E}[\Delta\text{CPC}] > 0$. Conditioning zero-shot flow predictions on the true target distance distribution produces a systematic paired improvement over the frozen baseline $M_0$.
  - *Evidence*: $\Delta\text{CPC} = +0.00354$, $95\%\text{ CI } [+0.0026, +0.0045]$, $45/50\text{ cities positive}$, Wilcoxon $p_{\text{two-sided}} = 1.93 \times 10^{-9}$.
* **H2 (Target Specificity)**: $\mathbb{E}[\Delta\text{CPC}_{\text{wrong}}] \approx 0$ and $\mathbb{E}[\Delta\text{CPC}_{\text{target}} - \Delta\text{CPC}_{\text{wrong}}] > 0$. The improvement is not an artifact of random perturbation; it strictly requires target-specific direction.
  - *Evidence*: Dose-matched wrong donors yield $\Delta\text{CPC} = -0.000091$; Specificity Difference $= +0.003630 \ [95\%\text{ CI } +0.00287, +0.00445]$, $p = 2.19 \times 10^{-11}$.
* **H3 (Observation Quality Decay)**: $\epsilon \uparrow \implies \mathbb{E}[\Delta\text{CPC}] \downarrow$. Information utility decays monotonically with observation noise, vanishing around an empirical crossover threshold $\epsilon_{\text{cross}} \approx 4.4\%$ TV error.
  - *Evidence*: Monotonic decline from $+0.00354$ ($\epsilon=0$) to $-0.00087$ ($\epsilon=0.05$); $\epsilon_{\text{cross}} = 4.44\%\ [4.16\%, 4.77\%]$ across realizations; $4.39\%\ [3.66\%, 4.94\%]$ across cities.
* **H4 (Baseline-Misalignment Mechanism)**: $d_{\text{pre}} \uparrow \implies \Delta\text{CPC} \uparrow$. Gain is largest in cities where the baseline zero-shot representation has the greatest aggregate distance distribution mismatch.
  - *Evidence*: Partial correlation $r_{\text{partial}}(d_{\text{pre}}, \Delta\text{CPC} \mid M_0, \log N_{\text{pairs}}, \log N_{\text{tracts}}, \text{MeanDist}) = +0.7951$ ($p = 5.35 \times 10^{-12}$).
* **Formally Dropped Hypothesis**:
  - The initial hypothesis that intra-bin ranking quality $Q_c^{\text{intra}}$ predicts gain is empirically refuted ($r \approx 0$). This sharpens the mechanism: the calibration operator specifically remedies *inter-bin macro distance allocation*, not intra-bin topology. (Documented in Supplementary).

---

## 4. 6-Figure Storyboard

### Figure 1: Support-Conditioned Oracle Calibration Framework (Conceptual / Methods)
- **Panel A (Cross-City Training & Frozen Baseline)**:
  $5\text{ cross-validation folds} \times 35\text{ training cities} \to \text{Frozen } M_0$.
- **Panel B (Inference on Target Support $\Omega_c^+$)**:
  Target city features $X_c$ and known positive support $\Omega_c^+$ passed to frozen $M_0 \to \hat{T}^{(0)}$.
- **Panel C (Aggregate Target Observation & Intensity Calibration)**:
  Oracle target distance distribution $Y_D$ ($K=8$ bins) $\to$ closed-form mass-preserving scaling operator $s_k \to \hat{T}^{(1)} = s_{b(ij)} \hat{T}^{(0)}$.
- **Key Visual Message**: Zero parameters trained at inference; support is fixed; calibration strictly rescales intensity across distance bins. The oracle $Y_D$ is derived from target reference flows used for evaluation, not independently collected external telemetry.

### Figure 2: Primary Headline Finding (The Empirical Main Result)
- **Main Plot**: Ordered per-city bar/lollipop plot of $\Delta\text{CPC}_c$ across all 50 held-out test cities.
- **Reference Lines & Inset**:
  - Horizontal dashed line at $\Delta\text{CPC} = 0$.
  - Inset box showing population summary: $M_0 = 0.71281 \to M_1 = 0.71635$ ($\Delta\text{CPC} = +0.00354 \ [95\%\text{ CI } +0.0026, +0.0045]$).
  - Annotations: $45/50\text{ cities positive (90.0\%)}$, Wilcoxon $p = 1.93 \times 10^{-9}$.
- **Key Visual Message**: While the absolute magnitude is modest, the improvement is exceptionally systematic across heterogeneous urban regions.

### Figure 3: Distance and Spatial Observation Resolution
- **Panel A**: Mean calibration gain across $K \in \{2,4,6,8,10,12,14,16,18,20\}$, with fold-stratified 95% CIs.
- **Panel B**: City-level vs. county-level calibration across the 11 multi-county metropolitan areas.
- **Key Visual Message**: Distance resolution increases gain across the tested values; county-level information supplies small localized gains where it adds genuine spatial granularity.

### Figure 4: Synthetic Noise Dose-Response
- **Main Plot**: Mean $\Delta\mathrm{CPC}$ across 50 cities as a function of Total Variation perturbation magnitude $\epsilon$.
- **Key Visual Message**: Utility decreases under the synthetic noise design and crosses zero near $4.44\%$ TV error; this is a benchmark-specific crossover.

### Figure 5: Target Specificity via Dose-Matched Placebos
- **Main Plot**: Direct comparison of mean $\Delta\text{CPC}$ and 95% fold-stratified bootstrap CIs across 3 key conditions:
  1. Oracle Target $Y_D$ ($+0.003539, \text{CI } [+0.00260, +0.00450]$).
  2. Dose-Matched Wrong Donor ($-0.000091, \text{CI } [-0.00089, +0.00071]$).
  3. Dose-Matched Fold Train-Mean ($+0.000914, \text{CI } [+0.00001, +0.00186]$).
- **Callout Annotations**:
  - Specificity difference: $+0.003630$ ($p = 2.19 \times 10^{-11}$).
  - Generic train-mean decay vs Target: Target wins in $47/50\text{ cities}$ ($p = 4.03 \times 10^{-11}$).
- **Key Visual Message**: Gain strictly requires the target-specific direction; generic decay or wrong directions at equal dose fail to replicate the benefit.

### Figure 6: Mechanistic Explanation via Baseline Distance Misalignment
- **Main Plot**: Scatter plot of baseline distance mismatch $d_{\text{pre}} = \text{TV}(\hat{Y}_D^{M0}, Y_D)$ vs $\Delta\text{CPC}_c$ across all 50 cities.
- **Regression Line & Annotation**:
  - Raw correlation $r = +0.7995$ ($p = 3.36 \times 10^{-12}$).
  - Partial correlation controlling for baseline accuracy, network size, and urban extent: $r_{\text{partial}} = +0.7951$ ($p = 5.35 \times 10^{-12}$).
  - Highlighted exemplar cities: Los Angeles, Phoenix, Houston, El Paso, Oklahoma City.
- **Key Visual Message**: The calibration gain is highest precisely where the zero-shot baseline's distance allocation is most severely distorted.

Figures 2--6 are empirical or diagnostic results; Figure 1 is the methods schematic.

---

## 5. Structure of the Results Section (Section 4)

### 4.1 Target Distance Information Improves Zero-Shot OD Reconstruction
- State headline result: $M_0 = 0.71281 \to M_1 = 0.71635$, $\Delta\text{CPC} = +0.00354$.
- Provide inferential support: Primary fold-stratified hierarchical 95% CI $[+0.0026, +0.0045]$.
- Report consistency across cities: $45/50\text{ cities positive (90.0\%)}$, two-sided Wilcoxon $p = 1.93 \times 10^{-9}$.
- Calibrate magnitude interpretation: The effect magnitude is modest in absolute terms, but remarkably uniform and statistically overwhelming.

### 4.2 The Improvement is Stable Across Model Initialization and Neural Backbones
- Seed stability: Standard error across 10,000 hierarchical resamples only increases by $<0.5\%$ ($0.000494$ vs $0.000492$).
- Backbone comparison:
  - Urban GNN: $\Delta\text{CPC} = +0.00354, p = 1.93 \times 10^{-9}$ (win rate 90.0%).
  - Node MLP: $\Delta\text{CPC} = +0.00329, p = 4.38 \times 10^{-11}$ (win rate 94.0%).
  - Classical 2-parameter Gravity: $\Delta\text{CPC} = +0.00084, p = 0.3545$ (n.s., baseline $M_0 = 0.3887$).
- Claim boundary: The effect reproduces across modern neural mobility architectures that learn spatial representations.

### 4.3 The Value of $Y_D$ Depends on Observation Resolution and Quality
- Pre-specified canonical resolution: $K=8$ dynamic quantile bins.
- Resolution scaling: Beyond $K=4$, marginal gain per bin steadily decreases from $4.94 \times 10^{-4}$ to $3.19 \times 10^{-4}$ at $K=20$.
- Noise tolerance: Monotonic degradation as TV noise increases; benefit crosses zero at $\epsilon_{\text{cross}} \approx 4.44\%\ [4.16\%, 4.77\%]$ across realizations ($4.39\%\ [3.66\%, 4.94\%]$ across cities).

### 4.4 Improvement Depends on Target-Specific Distance Information
- Primary specificity evidence: Dose-matched wrong donors ($\Delta\text{CPC} = -0.000091$, Specificity gain $+0.003630, p = 2.19 \times 10^{-11}$).
- Baseline decay control: Dose-matched fold train-mean yields a weak mean bump ($+0.000914$) but no systematic paired improvement ($p = 0.4319$); Target outperforms it in $47/50\text{ cities}$ ($p = 4.03 \times 10^{-11}$).
- Secondary stress tests: Raw wrong donors ($-0.035$ to $-0.038$) and permutation ($-0.00696$) confirm that macro spatial mismatch destroys prediction ($p < 10^{-15}$).

### 4.5 Baseline Distance Misalignment Explains Where Calibration Helps Most
- Define $d_{\text{pre}} = \text{TV}(\hat{Y}_D^{M0}, Y_D)$.
- Multiple regression: $R^2 = 73.7\%$, $d_{\text{pre}}$ dominant predictor ($t = +8.70, p < 10^{-10}$).
- Combined mechanism story: Benefit is associated with both the magnitude of baseline distance-distribution mismatch ($d_{\text{pre}}$) and correcting it in the target-specific direction.

### 4.6 Secondary Error Metrics (Brief / Appendix)
- $\Delta\text{MAE} = -2.539$ (45/50 positive), $\Delta\text{RMSE} = -2.983$ (32/50 positive).
- Methodological caveat: MAE and CPC share an exact mathematical dependency under fixed predicted mass; MAE is reported for completeness, not as independent confirmation.

---

## 6. Division of Main Paper vs Supplementary Materials

| Main Paper (Direct Evidence for RQs) | Supplementary Materials (Audit & Completeness) |
|---|---|
| Headline 50-city result ($M_0 \to M_1$, $\Delta\text{CPC}$, CI) | Complete 50-city $\times$ 3 seeds numeric data table |
| Backbone summary (Urban GNN vs Node MLP vs Gravity) | Full training hyperparameters and loss convergence curves |
| Canonical $K=8$ result & $K$-resolution summary curve | Full numeric tables for $K \in \{2, 4, 8, 12, 16, 20\}$ |
| Noise summary curve & $\epsilon_{\text{cross}}$ confidence bounds | Detailed per-city noise trajectory breakdowns ($750,150$ evals) |
| Dose-matched placebo comparison & Specificity Gain | Raw donor and permutation placebo full tables |
| $d_{\text{pre}}$ partial correlation and scatter plot | OLS regression diagnostic tables, VIF checks, residual plots |
| Conceptual Method Diagram (Figure 1) | Intra-bin ranking diagnostic ($Q_c^{\text{intra}}$ null finding) |
| Core Discussion & Boundary Definitions | Calibration weight audit ($w_k$ distributions, flow preservation) |
| County-level spatial-resolution summary | Supplementary Table S1: Descriptive multi-county spatial-resolution results, sourced from `results/spatial_resolution/spatial_resolution_per_city.json` |

---

## 7. Scope & Boundary Definitions (Non-Negotiable)

1. **Oracle Existence Result**: $Y_D$ is an oracle target observation demonstrating the *information value* of aggregate distance constraints, not an empirical test of a specific real-world sensor.
2. **Observed Positive Support ($\Omega_c^+$)**: Evaluation is conditioned strictly on observed positive pairs ($i \ne j, D_{ij} > 0, T_{ij} \ge 1$), avoiding sparsity/zero-filling artifacts.
3. **Primary Reported CI**: Duly rounded to 4 decimals everywhere: **$\mathbf{\Delta\text{CPC} = +0.00354 \ [95\%\text{ CI}: +0.0026, +0.0045]}$**.
