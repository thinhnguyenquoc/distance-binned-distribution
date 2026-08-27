# Section 4: Empirical Results

In this section, we present the empirical evaluation designed to answer **RQ1** and **RQ2**. Across all experiments, our objective is not to introduce or optimize a bespoke calibration algorithm, but rather to use a closed-form, mass-preserving calibration operator as an **experimental instrument** to quantify the incremental information value of target-city aggregate distance distributions ($Y_D$). 

All evaluations are conducted under a strict cross-validation protocol (5 folds $\times$ 10 test cities = 50 held-out metropolitan areas across the United States) on the observed positive interzonal support $\Omega_c^+ = \{(i, j) \mid i \ne j, D_{ij} > 0, T_{ij} \ge 1\}$. The headline performance metric is the Common Part of Commuters (CPC) on interzonal flows, evaluated relative to the frozen zero-shot cross-city baseline $M_0$.

---

## 4.1 Target Distance Information Improves Zero-Shot OD Reconstruction

We begin by addressing **RQ1**: *Does an oracle target-city distance-binned mobility distribution improve support-conditioned zero-shot OD intensity reconstruction beyond a frozen cross-city baseline?* Under Hypothesis **H1**, we posited that incorporating exact target-city distance distribution constraints would produce a systematic paired improvement in reconstruction accuracy over the frozen baseline ($\mathbb{E}[\Delta\text{CPC}] > 0$).

Across the 50 held-out test cities, the frozen cross-city neural baseline ($M_0$) achieves a mean interzonal CPC of $0.71281 \pm 0.04434$. Conditioning these zero-shot predictions on the target city's 8-bin oracle distance distribution via the closed-form scaling operator ($M_1$) increases the mean CPC to $0.71635 \pm 0.04454$. This corresponds to a population-average improvement of:

$$\mathbf{\Delta\text{CPC} = +0.00354} \quad (\text{median } +0.00195).$$

To rigorously quantify the uncertainty of this estimand, we compute a 10,000-iteration fold-stratified hierarchical bootstrap that jointly resamples cities within cross-validation folds and neural model seeds within cities. The resulting primary 95% confidence interval is:

$$\mathbf{95\%\text{ CI: } [+0.0026,\ +0.0045]},$$

which excludes zero by a substantial margin ($\text{SE} = 0.00049$). 

Crucially, the observed benefit is not driven by large outliers in a handful of regions, but reflects a remarkably consistent paired improvement across heterogeneous urban topographies. As illustrated in the per-city distribution of performance changes (**Figure 2**):
- **45 out of 50 test cities (90.0%)** exhibit positive gains ($\Delta\text{CPC}_c > 0$).
- A two-sided paired Wilcoxon signed-rank test firmly rejects the null hypothesis of no location shift between calibrated and uncalibrated predictions:
  $$\mathbf{p = 1.93 \times 10^{-9}} \quad (\text{paired Wilcoxon } W = 1205.0; \text{ one-sided directional } p = 9.66 \times 10^{-10}).$$

In absolute terms, the magnitude of the improvement is modest—representing approximately a $0.5\%$ relative increase over an already competitive neural baseline. However, because the underlying mobility models remain entirely frozen and no individual OD flow data or trainable parameters are introduced at inference time, this uniform positive shift provides direct empirical confirmation of **H1**: coarse, macro-level distance distributions carry non-trivial, usable information that systematically refines zero-shot spatial flow allocation on known positive support.

Having established the existence and consistency of this main effect, we next investigate whether this gain is robust to model initializations and architectural backbones (Section 4.2), how it scales with observational resolution and noise (Section 4.3), whether it strictly demands target-specific directional alignment (Section 4.4), and what structural mechanisms explain where the information value is greatest (Section 4.5).
