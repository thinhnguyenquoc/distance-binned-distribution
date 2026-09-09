# Improving Zero-Shot OD Flow Intensity Reconstruction via the Target City's Distance-Binned Mobility Distribution

## Abstract

Origin–destination (OD) matrices are important inputs for transportation analysis and urban planning, but detailed data on target-city OD flow intensity are often difficult to collect. Studies using urban context and geographic distance have developed cross-city zero-shot baselines capable of predicting mobility flows without using observed target-city OD intensity data. This study examines whether the target city's distance-binned mobility distribution can improve interzonal OD flow-intensity reconstruction on the known positive interzonal support by calibrating the output of a frozen-parameter zero-shot baseline. Crucially, the evaluation adopts an oracle setting: the distance distribution is aggregated directly from the target city's ground-truth OD flows on the exact positive interzonal support used for evaluation, isolating the theoretical information value of the signal.

In the main experiment, the method is evaluated using a 5-fold cross-validation protocol across 50 U.S. metropolitan areas. City-level calibration produces a consistent improvement: mean CPC increases from a baseline of $0.71281$ to $0.71635$ ($\Delta\mathrm{CPC} = +0.00354$, $\sim 0.5\%$ relative gain; 95% CI: $[+0.0026, +0.0045]$, $p = 1.93 \times 10^{-9}$), with 45/50 cities improved. Dose-matched placebo controls establish structural target-specificity: donor distributions from training cities and randomly permuted distributions eliminate the benefit ($\Delta\mathrm{CPC} \le 0$), proving that improvements depend on the authentic distribution of the target city. The benefit diminishes as the observation resolution coarsens or quality degrades under noise. However, the findings are strictly bounded by this oracle setup on known positive support, leaving link discovery and evaluation with independently sourced observations to future work.

**Keywords:** origin–destination matrix; OD intensity reconstruction; distance-binned mobility distribution; zero-shot; cross-city transfer learning; aggregate observations; spatial mobility.

# 1. Introduction

Origin–destination (OD) matrices describe the intensity of movement between spatial units and are important inputs for transportation analysis and urban planning [@barbosa2018humanmobility]. However, detailed OD data are often difficult to collect completely in a target city and may be subject to limitations in coverage and representativeness [@gallotti2024distorted; @pappalardo2023future]. Mobility flows also depend on urban context and local characteristics, so patterns learned from a source city do not necessarily transfer fully to a target city. Consequently, cross-city transfer models can still exhibit systematic bias in the target city when local calibration information is unavailable [@yang2014limits].

Recent mobility models have combined urban context and distance to predict flows that can transfer across cities [@simini2021deepgravity; @guo2025ugnn; @enaya2026transgm]. However, a frozen-parameter cross-city zero-shot baseline infers the target city only from available input features. Although it knows the distance of each OD pair, the model does not directly observe how the total mobility volume of the target city is distributed across distance intervals. Previous studies show that distance-decay structure can vary substantially across urban contexts [@lenormand2016comparison; @verma2025distance].

This study tests whether the target city's distribution of mobility across distance intervals provides additional information for a trained cross-city baseline. This distribution describes only the share of total flow by distance interval and is used at inference time to calibrate predictions, while all model parameters remain fixed. The calibration is used as an experimental tool to quantify the information value of the additional mobility distribution.

The study focuses on two questions. First, does the target city's distance-binned mobility distribution improve OD intensity reconstruction relative to a frozen-parameter cross-city zero-shot baseline? Second, if it does, how does the improvement depend on the resolution, quality, distance-bin ordering, and specificity of the target observation?

In this study, the distribution is extracted from the reference flows of the target city itself and is therefore treated as an oracle observation. This setting is used to test the information value of the signal before considering whether it can be collected or estimated from an independent source.

The study is evaluated using 5-fold cross-city validation on 50 U.S. metropolitan areas. Each city is evaluated when it is not included in training, and all model parameters remain fixed before the calibration step.

The study contributes by quantifying the additional information value of the target city's distance-binned mobility distribution, identifying observational conditions that govern the improvement, and examining the mechanism and robustness of the effect across multiple initializations and baseline architectures.

# 2. Related Work

## 2.1. Spatial interaction models and distance-based calibration

Spatial interaction models have long represented OD flows through production potential, attraction, and spatial impedance, with distance or travel cost as a core component of flow structure [@ortuzar2011modelling; @wilson1971family].

Classical calibration methods show that aggregate trip-distance statistics can be used to identify impedance parameters. Hyman [@hyman1969calibration] proposed calibrating a trip-distribution model using mean trip length, while Merlin [@merlin2020medians] used median travel time to calibrate a one-parameter spatial interaction model.

Comparative studies also show that distance-decay laws are not fixed across datasets and urban contexts. Empirical decay patterns may vary by travel mode, trip purpose, degree of urbanization, and socioeconomic conditions [@verma2025distance].

These findings indicate that distance structure is context-specific. Methodologically, the binned multiplicative adjustment operator is closely rooted in Iterative Proportional Fitting (IPF) and Furness algorithms in classical transportation planning [@ortuzar2011modelling], originating from Deming and Stephan's contingency table adjustments and Wilson's maximum entropy formulation [@wilson1971family]. When constraints are imposed exclusively on one-dimensional distance bins without simultaneous origin–destination margin matching, the formulation reduces to a single-step, closed-form proportional scaling. Rather than re-solving a classical doubly-constrained matrix balancing problem from scratch, the present study adapts this principle as a post-hoc inference-time calibration operator applied directly to the complex representations of frozen zero-shot neural networks.

## 2.2. Cross-city machine-learning models and aggregate observations

Cross-city generalization remains challenging because the relationship between urban context and mobility flows can vary across cities. Yang et al. [@yang2014limits] show that the predictability of commuting flows is substantially limited when local calibration data are unavailable. This result indicates that using distance and urban features does not necessarily eliminate the need for target-domain-specific information.

In this context, aggregate observations from the target domain provide an intermediate level of information between two extremes: no information from the target city and direct observation of the complete OD matrix. Classical constraints such as total outflow, inflow, or cost moments have been used to impose macro-level consistency in spatial interaction models [@ortuzar2011modelling; @wilson1971family].

Unlike approaches that mainly calibrate one or a small number of parameters, this study directly uses a vector of flow shares across distance intervals, allowing the value of the signal to be evaluated at multiple resolutions through the number of intervals $K$. $Y_D$ also differs from origin/destination margins or directly observed OD pairs: it constrains only how total flow volume is distributed across distance bands, without determining how that volume is distributed among origin–destination pairs within the same band.

Previous studies have clarified the role of distance and constraints in spatial interaction models [@ortuzar2011modelling; @wilson1971family], while also demonstrating the generalization ability of flow-prediction models and their limitations when local calibration information is absent [@guo2025ugnn; @simini2021deepgravity; @yang2014limits]. However, it remains unclear how much additional value is provided by the target city's own distance-binned mobility distribution after a cross-city model has learned from urban context and pairwise distance, and under what observation conditions that value persists.

The present study differs from these directions in that the aggregate observation is not used to train or re-estimate the model, but to directly measure the additional information value of a target-city-specific signal after the cross-city baseline has been trained.

# 3. Data Sources, Spatial Units, and Methodology

## 3.1. Notation and input data

Let $c$ be a city and let $\mathcal{V}_c$ be the set of spatial units partitioning that city. Each ordered pair $(i,j)$ with $i,j \in \mathcal{V}_c$ represents an origin–destination (OD) pair.

### Table 1: Core notation, data sources, and information availability status

| Symbol | Mathematical description | Source / Role |
| :--- | :--- | :--- |
| $c$ | City index ($c \in \{1, \dots, C\}$) | City identifier ($C = 50$) |
| $i, j$ | Origin and destination spatial-unit indices | Basic spatial units |
| $t_{c,ij}$ | Observed mobility-flow intensity ($t_{c,ij} \ge 1$) | Reference data (ground truth) |
| $d_{c,ij}$ | Distance between the centroids of units $i$ and $j$ (km) | Computed from centroid coordinates (Haversine) |
| $\mathcal{P}_c$ | Space of all valid interzonal OD pairs ($\mathcal{P}_c = \{(i,j) \in \mathcal{V}_c \times \mathcal{V}_c : i \neq j, d_{c,ij} > 0\}$) | Candidate interzonal pair space |
| $\Omega_c$ | Known positive interzonal support ($\Omega_c = \{(i,j) \in \mathcal{P}_c : t_{c,ij} \ge 1\}$) | Known-support assumption |
| $I_b$ | Distance interval $b$ ($b = 1, \dots, K$) | Distance quantile |
| $K$ | Number of distance intervals ($K = 8$ in the main setting) | Fixed experimental configuration |
| $Y_{c,b}$ | Target mobility-flow share in interval $b$ ($\sum_{b=1}^K Y_{c,b} = 1$) | Oracle calibration-input data |
| $Y_{D,c}$ | Distance-binned mobility distribution vector of city $c$, with $Y_{D,c} = (Y_{c,1}, \dots, Y_{c,K})$ | Oracle aggregate observation used at calibration time |
| $\widehat{t}_{c,ij}^{(0)}$ | Flow-intensity prediction of the cross-city zero-shot baseline (condition $M_0$) | Frozen-parameter baseline output |
| $\widehat{t}_{c,ij}^{(1)}$ | Flow-intensity prediction after inference-time calibration (condition $M_1$) | Post-calibration output |
| $M_0, M_1$ | Names of the two experimental conditions (frozen-parameter zero-shot baseline and post-calibration prediction) | Comparative experimental conditions |

## 3.2. Support scope and spatial representation

The experimental data include 50 metropolitan areas in the United States, with tracts as the basic spatial units. Each tract is represented by centroid coordinates and urban features; the data also include distances between tract pairs and observed OD flow intensities. The source and construction process of the benchmark will be fully described according to the original data documentation before submission.

The space of all valid interzonal OD candidate pairs is defined as:

$$
\mathcal{P}_c = \left\{(i,j) \in \mathcal{V}_c \times \mathcal{V}_c : i \neq j,\ d_{c,ij} > 0\right\}.
$$

The evaluation scope is strictly restricted to the known positive interzonal support:

$$
\boxed{
\Omega_c = \left\{(i,j) \in \mathcal{P}_c : t_{c,ij} \ge 1\right\} = \left\{(i,j) \in \mathcal{V}_c \times \mathcal{V}_c : t_{c,ij} \ge 1,\ i \neq j,\ d_{c,ij} > 0\right\}.
}
$$

The model predicts flow intensities on the positive support $\Omega_c$, and does not address link discovery or zero-flow classification within $\mathcal{P}_c$. Throughout the paper, pairs outside $\Omega_c$ are treated as unknown and are outside the evaluation scope.

## 3.3. Distance-binned mobility distribution and city-level observation configuration

The main experiments use a single city-level distance-binned mobility distribution. In each fold, $K-1$ interior bin edges are determined from the $b/K$ quantiles ($b=1,\ldots,K-1$) of interzonal OD pair distances across the 35 training cities. Each pair contributes a single distance value with equal weight; thus, edges are defined on a pair-weighted basis. Validation and test cities are strictly excluded from edge construction. The two outer boundaries are fixed at $a_0 = 0$ and $a_K = +\infty$, forming intervals $I_b = (a_{b-1}, a_b]$ that completely cover all pairs with $d_{c,ij} > 0$. The share of target mobility flow falling in distance interval $b$ is defined as:

$$
Y_{c,b} = \frac{\sum_{(i,j) \in \Omega_c} t_{c,ij} \mathbf{1}(d_{c,ij} \in I_b)}{\sum_{(i,j) \in \Omega_c} t_{c,ij}}.
$$

The shares are normalized so that: $\sum_{b=1}^K Y_{c,b} = 1$.
The full distance-distribution vector of city $c$ is denoted by $Y_{D,c} = (Y_{c,1}, \dots, Y_{c,K})$. In the interpretation, $Y_D$ is used as shorthand for this type of observation.

Because bin edges are determined jointly from the training fold, certain long-distance intervals may contain zero OD pairs in cities with smaller geographic diameters. Let $\mathcal A_c$ denote the set of intervals containing at least one pair in $\Omega_c$, with $K_{\mathrm{act},c} = |\mathcal A_c|$ representing the number of active intervals for city $c$. Empty bins carry zero mass and are excluded from calculations; all operator quantities are evaluated over the active set (detailed in Supplementary Section S2).

$Y_{D,c}$ is aggregated from the ground-truth flow of the target city and used as an oracle observation at calibration time. An exploratory variant using an origin-county distribution is evaluated on multi-county metropolitan areas; the setup and limitations of this analysis are presented in Supplementary Section S7.

## 3.4. Model structure and inference-time calibration

### 3.4.1. Baselines and common prediction interface

Three baselines are evaluated under the same inference-time calibration protocol. Urban GNN is the primary baseline, Pairwise Node MLP is an additional neural baseline, and two-parameter Gravity is an additional classical baseline for assessing the extent to which calibration effectiveness depends on model architecture.

The Urban GNN uses two distance-conditioned message-passing layers with mean neighborhood aggregation, LayerNorm, residual connections, and 0.1 dropout. Each tract is represented by 26 urban features projected to a 64-dimensional embedding. Pairwise OD intensity is decoded from the origin and destination embeddings, log geographic distance, and an internal two-parameter gravity prior using a $130\!-\!64\!-\!32\!-\!1$ MLP. Crucially, the internal gravity parameters within the neural architecture are trainable weights optimized end-to-end alongside the network and stored in the model checkpoint.

The model is trained on the source cities of each fold, and all parameters remain fixed when inferring on the target city.

We additionally considered a standalone two-parameter classical gravity baseline, in which OD intensity is proportional to the product of origin and destination population and decays with geographic distance. The global scale and distance-decay coefficients for this standalone baseline are estimated on the training cities by ordinary least squares (OLS) and held fixed during zero-shot inference, rather than being shared with the neural checkpoints. The same closed-form $Y_D$ calibration operator was then applied without refitting the gravity model.

$$
\widehat{t}^{(0,\mathrm{grav})}_{c,ij} = \exp(G) \frac{P_{c,i} P_{c,j}}{d_{c,ij}^{\alpha}}, \qquad (i,j) \in \Omega_c.
$$

Here, $G$ is a global scale coefficient and $\alpha > 0$ is the distance-decay exponent. To ensure numerical stability, tract population is lower-bounded at

$$
P_{c,i} = \max(\operatorname{pop}_{c,i}, 1.0), \qquad P_{c,j} = \max(\operatorname{pop}_{c,j}, 1.0),
$$

and distance is lower-bounded at

$$
d_{c,ij} = \max(\mathrm{dist}_{c,ij}, 0.1\,\text{km}).
$$

The two parameters $(G, \alpha)$ of this standalone classical baseline are estimated using pooled log-linear ordinary least squares only on the training cities of each fold and remain fixed when inferring on the test cities. Gravity predictions are then passed through the same $Y_D$ calibration operator as the other baselines.

The MLP control replaces the two graph message-passing layers with two node-wise residual MLP blocks while retaining the same node-feature input, 64-dimensional embeddings, pairwise decoder, geographic distance, trainable gravity prior, optimization settings, and total parameter count. It therefore isolates the contribution of graph-based neighborhood aggregation.

### 3.4.2. Objective and training configuration

Because the modeling scope in this study considers only positive flows on the known support, the two neural baselines are trained with a Zero-Truncated Negative Binomial (ZTNB) likelihood, a conditional negative-binomial count model that excludes value 0 [@grogger1991truncated].

$$
p_+(t \mid \mu, \phi) = \frac{p_{\mathrm{NB}}(t \mid \mu, \phi)}{1 - p_{\mathrm{NB}}(0 \mid \mu, \phi)}, \qquad \mathcal{L}_c = -\frac{1}{\lvert\Omega_c\rvert} \sum_{(i,j) \in \Omega_c} \log p_+(t_{c,ij} \mid \mu_{c,ij}, \phi).
$$

The loss is averaged over pairs $(i,j) \in \Omega_c$ for each city to prevent cities with many OD pairs from dominating optimization.

Both neural baselines use the same training protocol with the AdamW optimization algorithm [@loshchilov2019adamw], select checkpoints by validation CPC, and are repeated over three model seeds. After checkpoint selection, all parameters remain fixed on target cities. Training-hyperparameter details are provided in the Appendix.

### 3.4.3. Inference-time distance calibration operator

For any frozen-parameter baseline model that produces initial predictions $\widehat{t}_{c,ij}^{(0)}$ on $\Omega_c$, the distance distribution implicitly produced by the baseline in interval $b$ is:

$$
\widehat{Y}_{c,b}^{(0)} = \frac{\sum_{(i,j) \in \Omega_c} \widehat{t}_{c,ij}^{(0)} \mathbf{1}(d_{c,ij} \in I_b)}{\sum_{(i,j) \in \Omega_c} \widehat{t}_{c,ij}^{(0)}}.
$$

The analytic calibration operator reallocates flow mass according to the closed-form solution:

$$
\widehat{t}_{c,ij}^{(1)} = \widehat{t}_{c,ij}^{(0)} \frac{Y_{c,b(i,j)}}{\widehat{Y}_{c,b(i,j)}^{(0)}}.
$$

Here, $b(i,j)$ is the interval containing $d_{c,ij}$. Every OD pair in the same interval is multiplied by the same coefficient. Calibration does not update model parameters. The main setting fixes $q = 1$; the general form $q \in [0, 1]$ is presented in Supplementary Section S2. Because the calibration coefficients are positive and constant within each interval, the operator preserves the support, within-interval ranking, and total predicted mass; proofs are presented in Supplementary Section S3.

![Figure 1](figures/fig1_oracle_calibration_framework.png)
**Figure 1. Inference-time oracle calibration framework.** Baseline $M_0$ is trained cross-city and kept frozen on the target city. The oracle distance distribution $Y_D$, extracted from the target city's reference flow, reallocates mass between intervals and creates $\widehat{\mathbf{T}}_c^{(1)}$ on the same support $\Omega_c$.

## 3.5. Cross-city evaluation protocol and statistical inference

### 3.5.1. 5-fold cross-city validation protocol

The study applies a 5-fold cross-city validation protocol on 50 U.S. metropolitan areas (each fold consists of 35 training cities, 5 validation cities, and 10 test cities). The unit assigned to a fold is the entire city; no OD pair or tract from the same city is distributed between training and test sets.

### 3.5.2. Evaluation metrics and model comparison

The primary quantitative metric for evaluating zero-shot mobility-flow reconstruction is Common Part of Commuters (CPC) [@lenormand2016comparison], computed on the positive interzonal support $\Omega_c$:

$$
\operatorname{CPC}_c(\widehat{t}) = \frac{2 \sum_{(i,j) \in \Omega_c} \min(t_{c,ij}, \widehat{t}_{c,ij})}{\sum_{(i,j) \in \Omega_c} t_{c,ij} + \sum_{(i,j) \in \Omega_c} \widehat{t}_{c,ij}}.
$$

CPC lies in $[0, 1]$, with larger values indicating greater overlap between predicted and observed intensity.

Additional error and ranking metrics are reported as robustness checks; their full definitions are provided in Supplementary Section S4.

In addition, the aggregated post-calibration distance distribution is compared with $Y_D$ as an internal mechanism diagnostic to confirm that the algorithm has reallocated mass as designed. All three model families (GNN, MLP, Gravity) are evaluated on the same support using the same CPC metric.

### 3.5.3. Statistical analysis and uncertainty quantification

For each city, the improvement is computed as the CPC difference between the post-calibration prediction and the baseline, then averaged across model seeds and macro-averaged across all 50 cities.

The 95% confidence interval is estimated using a city-level paired nonparametric bootstrap, stratified by fold [@efron1993bootstrap]. Paired differences are evaluated using a two-sided Wilcoxon signed-rank test [@wilcoxon1945ranking]. The proportion of cities with $\Delta\mathrm{CPC} > 0$ is reported as an additional descriptive statistic. Corresponding sensitivity and robustness analyses are presented in Section 4.

In addition to the primary tests, an exploratory mechanism analysis evaluates the relationship between the baseline's distance-distribution bias and improvement after calibration. The initial bias $d_{\mathrm{pre}}$ is computed as the Total Variation between the baseline-predicted distance distribution and the reference distribution. Pearson and partial correlations are reported; the partial correlation controls for baseline accuracy ($M_0$ CPC) and city spatial-size characteristics, including $\log N_{\mathrm{tracts}}$, $\log N_{\mathrm{pairs}}$, and mean geographic distance.

# 4. Experimental Results

## 4.1. Does $Y_D$ improve OD intensity reconstruction on the known positive interzonal support relative to the zero-shot baseline?

In the main experiment with Urban GNN, calibration with $Y_D$ increases cross-city CPC by an average of $+0.00354$. The improvement appears in most cities, but its absolute magnitude is small and varies considerably across cases.

![Figure 2](figures/fig2_main_per_city.png)
**Figure 2: Cross-city CPC improvement by city from target-distance calibration.**

The bar chart shows $\Delta\mathrm{CPC}_c = \operatorname{CPC}_c(M_1) - \operatorname{CPC}_c(M_0)$ across 50 cities, ordered from low to high. The dashed line represents the mean improvement and the dotted line represents the median.

### Table 2: Main benchmark with Urban GNN ($N=50$, $K=8$)

| Experimental condition | Mean interzonal CPC | Median CPC | Mean $\Delta\mathrm{CPC}$ | 95% confidence interval (Stratified) | Winning-city rate | Wilcoxon $p$ (Two-sided) |
|---|---|---|---|---|---|---|
| **Frozen-parameter zero-shot baseline ($M_0$)** | $0.71281 \pm 0.04434$ | $0.71632$ | — | — | — | — |
| **Post-calibration prediction ($M_1$)** | $0.71635 \pm 0.04454$ | $0.71988$ | **$+0.00354$** | **$[+0.0026, +0.0045]$** | **45 / 50 (90.0%)** | $\mathbf{1.93 \times 10^{-9}}$ |

## 4.2. Is the improvement target-city-specific and structurally meaningful?

Placebo controls show that the benefit of calibration depends on target-city-specific information: the $Y_D$ of the correct target city produces a larger improvement than dose-matched donor distributions. When evaluating the dose-matched fold training-mean control, the mean gain is $+0.00091$ with a 95% stratified bootstrap CI of $[+0.00001, +0.00186]$, but the city-level shift is inconsistent (median $+0.00007$, 27/50 positive cities, two-sided Wilcoxon $p=0.4319$). When the intervention log-ratio vector is randomly permuted across distance intervals (preserving intervention dose while breaking spatial ordering; Supplementary S6), the benefit no longer persists and performance decreases ($\Delta\mathrm{CPC} = -0.00696$, with true target outperforming permuted controls in 49/50 cities by $+0.01050$). This indicates that calibration benefit depends on preserving the correct association between flow shares and distance intervals, rather than merely applying an equally strong perturbation.

![Figure 3](figures/fig3_structural_validity_placebo.png)
**Figure 3. Controls for target specificity and distance structure.** The figure compares target $Y_D$, dose-matched training-donor placebo, and permuted intervention log-ratio $Y_D$ across 50 cities. Error bars represent stratified 95% bootstrap CIs by fold.

### Table 3: Target specificity and placebo controls ($N=50$)

| Experimental condition | Mean $\Delta\mathrm{CPC}$ | 95% confidence interval (Stratified) | Benefit relative to $M_0$ ($p_{\text{2-sided}}$) | Specificity increase vs Placebo | 95% specificity CI | Target vs Placebo ($p_{\text{1-sided}}$) | Specificity win rate ($\text{Target } Y_D > \text{Placebo}$) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1. Oracle Target $Y_D$** | **$+0.003539$** | $[+0.00260, +0.00450]$ | $1.93 \times 10^{-9}$ | — | — | — | **45/50 (vs $M_0$)** |
| **2. Dose-Matched Training Donors ($B_{\text{draw}}=1000$)** | **$-0.000091$** | $[-0.00089, +0.00071]$ | $0.4097$ (n.s.) | **$+0.003630$** | $[+0.00287, +0.00445]$ | $\mathbf{2.19 \times 10^{-11}}$ | **46/50 (92.0%)** |
| **3. Dose-Matched Fold Train-Mean $Y_D$** | **$+0.000914$** | $[+0.00001, +0.00186]$ | $0.4319$ (n.s.) | **$+0.002626$** | $[+0.00197, +0.00336]$ | $\mathbf{4.03 \times 10^{-11}}$ | **47/50 (94.0%)** |
| **4. Permuted Target $Y_D$ ($B_{\text{draw}}=1000$ Permutations)** | **$-0.006964$** | $[-0.00914, -0.00512]$ | $1.78 \times 10^{-15}$ | **$+0.010504$** | $[+0.00843, +0.01279]$ | $1.78 \times 10^{-15}$ | **49/50 (98.0%)** |

Note: Bootstrap confidence intervals are computed for mean $\Delta\mathrm{CPC}$, while $p$-values are obtained from the Wilcoxon signed-rank test on city-level paired differences; therefore, the two statistics do not test the same quantity and need not lead to the same conclusion. Donor placebos are averaged over 1,000 random donor draws; permutation placebos are averaged over 1,000 random permutations of the centered intervention log-ratio vector (with exhaustive permutations for cities with small active bin counts). Results across three model seeds are averaged prior to 50-city aggregation.

## 4.3. How does the additional value of $Y_D$ depend on observation resolution and quality?

The resolution and quality of the $Y_D$ observation are evaluated along three complementary dimensions: the number of distance intervals $K$, the spatial resolution of the aggregate signal, and observation-quality degradation due to noise. These three analyses evaluate which conditions govern the additional information supplied by $Y_D$.

First, as the number of distance intervals increases from $K=2$ to $K=20$, mean $\Delta\mathrm{CPC}$ increases from $+0.00098$ to $+0.00639$ (Table 4). At the main configuration $K=8$, the increase reaches $+0.00354$, with 45/50 cities improved over the baseline. Winning-city rates remain consistently high, ranging between 78% ($K=2$) and 94% ($K=18$) across all evaluated resolutions. The largest step increase occurs when moving from $K=2$ to $K=4$ (an additional gain of roughly 0.00100). Thereafter, marginal gains between successive resolution steps fluctuate between 0.0003 and 0.0009; for example, the step from $K=10$ to $K=12$ (+0.00067) is larger than that from $K=8$ to $K=10$ (+0.00059).

### Table 4: Scaling information resolution through distance intervals

| Resolution ($K$) | Mean interzonal CPC | Median CPC | Mean $\Delta\mathrm{CPC}$ | Median $\Delta\mathrm{CPC}$ | 95% confidence interval (Stratified) | Winning-city rate |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline ($M_0$)** | $0.71281 \pm 0.04434$ | $0.71632$ | — | — | — | — |
| **$K = 2$** | $0.71379 \pm 0.04441$ | $0.71665$ | **$+0.00098$** | $+0.00034$ | $[+0.00052, +0.00151]$ | **39 / 50 (78.0%)** |
| **$K = 4$** | $0.71479 \pm 0.04439$ | $0.71720$ | **$+0.00198$** | $+0.00088$ | $[+0.00125, +0.00279]$ | **39 / 50 (78.0%)** |
| **$K = 6$** | $0.71570 \pm 0.04445$ | $0.71784$ | **$+0.00289$** | $+0.00152$ | $[+0.00201, +0.00384]$ | **44 / 50 (88.0%)** |
| **$K = 8$ (Anchor)** | $0.71635 \pm 0.04454$ | $0.71988$ | **$+0.00354$** | $+0.00195$ | $[+0.00262, +0.00447]$ | **45 / 50 (90.0%)** |
| **$K = 10$** | $0.71694 \pm 0.04450$ | $0.72007$ | **$+0.00413$** | $+0.00235$ | $[+0.00311, +0.00514]$ | **45 / 50 (90.0%)** |
| **$K = 12$** | $0.71761 \pm 0.04453$ | $0.72060$ | **$+0.00480$** | $+0.00288$ | $[+0.00372, +0.00590]$ | **46 / 50 (92.0%)** |
| **$K = 14$** | $0.71819 \pm 0.04456$ | $0.72145$ | **$+0.00538$** | $+0.00373$ | $[+0.00424, +0.00654]$ | **45 / 50 (90.0%)** |
| **$K = 16$** | $0.71855 \pm 0.04458$ | $0.72205$ | **$+0.00574$** | $+0.00433$ | $[+0.00455, +0.00694]$ | **46 / 50 (92.0%)** |
| **$K = 18$** | $0.71884 \pm 0.04460$ | $0.72230$ | **$+0.00603$** | $+0.00458$ | $[+0.00480, +0.00726]$ | **47 / 50 (94.0%)** |
| **$K = 20$** | $0.71920 \pm 0.04462$ | $0.72266$ | **$+0.00639$** | $+0.00494$ | $[+0.00508, +0.00769]$ | **46 / 50 (92.0%)** |

Note: $K$ is the nominal number of distance intervals determined from the training fold. The number of active intervals $K_{\mathrm{act},c}$ may be smaller than $K$ for cities lacking OD pairs in long-distance intervals.

![Figure 4](figures/fig4_resolution_sensitivity.png)
**Figure 4. Sensitivity of improvement to the number of distance intervals $K$.** Points show mean $\Delta\mathrm{CPC}$ across 50 cities and error bars show stratified 95% bootstrap CIs by fold. $K=8$ is the main configuration of the study.

Mean improvement increases across the entire evaluated range of $K$. This indicates that a higher nominal resolution can provide additional calibration information, although the number of truly active intervals depends on each city's geographic distance extent.

In addition to distance resolution, we conducted an exploratory analysis of the spatial resolution of the observation. Across 11 metropolitan areas spanning multiple counties, calibration using an origin-county-level $Y_D$ distribution improved over city-level calibration in 9/11 cases. However, the pooled additional increase across all 50 metropolitan areas was only

$$ \Delta\mathrm{CPC}_{\mathrm{res}} = +0.00014, $$

because 39 single-county areas produce mathematically equivalent partitions and therefore have $\Delta\mathrm{CPC}_{\mathrm{res}}=0$ by construction. For the group of 11 multi-county areas alone, the mean additional increase was about $+0.00063$. Thus, this result is treated only as exploratory evidence that finer spatial resolution may provide additional information in some urban structures, rather than as general evidence that increasing spatial resolution always improves performance.

Separately from resolution, we further assessed sensitivity to the quality of the $Y_D$ observation itself. Noise was added to the target-city distribution at specified Total Variation error levels while keeping the baseline, evaluation cities, and calibration operator unchanged. As noise increased, mean $\Delta\mathrm{CPC}$ decreased monotonically and crossed baseline-equivalent performance at approximately

$$ \epsilon_{\mathrm{cross}} \approx 4.44\% \text{ TV}, $$

with 95% CI $[4.16\%,\,4.77\%]$. This is an empirical threshold specific to the benchmark and noise mechanism used and should not be interpreted as a universal tolerance level.

![Figure 5](figures/fig5_noise_dose_response.png)
**Figure 5. Sensitivity of improvement to Total Variation noise.** Points show mean $\Delta\mathrm{CPC}$ across 50 cities; the shaded band is the stratified 95% bootstrap CI by fold. The horizontal line at $\Delta\mathrm{CPC}=0$ indicates baseline-equivalent performance, and the dashed line marks the empirical crossing point $\epsilon_{\mathrm{cross}}\approx4.44\%$.

Overall, the results show that the value of $Y_D$ depends on both the granularity and the accuracy of the observation: increasing the number of distance intervals improves results, but the signal must remain sufficiently accurate to yield practical benefits. The county-level analysis further suggests that spatial detail may provide additional information in some cities, but this evidence is currently exploratory.

## 4.4. Robustness across initialization and baseline architecture

The improvement remains positive across all three model seeds, with mean $\Delta\mathrm{CPC}$ ranging from approximately $+0.0031$ to $+0.0043$, indicating that the main result is not driven by a single model initialization.

### Table 5: Robustness by baseline architecture ($N=50$ cities, $K=8$ intervals)

| Model architecture | Mean $\Delta\mathrm{CPC}$ | 95% Bootstrap confidence interval | Winning-city rate |
|:---|:---:|:---:|:---:|
| **Urban GNN (Message passing)**  | **$+0.00354$** | $[+0.0026, +0.0045]$ | **45 / 50 (90.0%)** |
| **Pairwise Node MLP (No graph message passing)**  | **$+0.00329$** | $[+0.0025, +0.0042]$ | **47 / 50 (94.0%)** |
| **Two-parameter Gravity** | $+0.00084$ | $[+0.0002, +0.0016]$ | 22 / 50 (44.0%) |

Note: The two neural baselines are aggregated across three model seeds. Gravity is estimated only on the training cities of each fold and does not use test-city flows.

The increase is reproduced for both Urban GNN and Pairwise Node MLP, while Gravity produces a smaller effect; therefore, current evidence for architectural robustness supports only the two evaluated neural baselines and should not be generalized to every model family.

## 4.5. Relationship between baseline distance-distribution bias and calibration improvement

The baseline's initial distance-distribution bias is strongly associated with improvement after calibration. After controlling for baseline accuracy, city size, and mean geographic distance, the partial correlation reaches $r_{\mathrm{partial}} = +0.7951$ ($p = 5.35 \times 10^{-12}$). This pattern is consistent with the method's mechanism but is interpreted only as an observational association, not a causal relationship.

![Figure 6](figures/fig6_mechanistic_dpre.png)
**Figure 6. Relationship between initial distance-distribution bias and improvement after calibration.** $d_{\mathrm{pre}}$ is the Total Variation distance between the baseline distance distribution and the ground truth. Each point represents a city; the line is a linear fit over 50 cities. The partial correlation after controlling for size–spatial variables is reported in Section 4.5.

# 5. Discussion

## 5.1. Information value, calibration mechanism, and methodological meaning

The fact that $Y_D$ continues to improve predictions after the baseline has used urban context and distances between spatial pairs indicates that these inputs do not fully infer how each target city's total mobility volume is distributed by distance. Because baseline parameters are not updated during calibration, this improvement is interpreted as the additional information value of $Y_D$, rather than a benefit from fine-tuning or retraining.

The calibration structure also clearly limits the type of bias that $Y_D$ can address. The signal provides information about how total flow volume should be distributed across distance bands, but it provides no additional information for distinguishing OD pairs within the same distance interval.

Therefore, $Y_D$ is primarily able to correct between-bin biases, where the baseline has allocated the wrong amount of mass across distance bands. Conversely, if the error is mainly within-bin, meaning in the allocation of flow among OD pairs with similar distances, $Y_D$ does not directly contain information to correct that error. The observed relationship between initial distance-allocation bias and city-level improvement is consistent with this mechanism, but is not interpreted as causal evidence.

This mechanism also clarifies the methodological meaning of the result. Models such as Deep Gravity and UGNN show that neural networks can learn transferable mobility patterns from source data [@simini2021deepgravity; @guo2025ugnn]. The result of this study adds that an aggregate target-domain observation can provide a calibration signal for a trained cross-city model without updating its parameters. However, this does not demonstrate deployment feasibility, because $Y_D$ here is an oracle and the calibration operates only on the known positive interzonal support $\Omega_c$.

## 5.2. Conditions governing the value of $Y_D$

The results show that the value of $Y_D$ depends on two distinct properties: the amount of structure retained by the observation and the accuracy of that structure. Increasing resolution is useful only when the additional information remains reliable; conversely, a high-resolution but biased distribution can eliminate the calibration benefit. Placebo and permutation analyses further show that the useful signal lies not merely in the general shape of the vector, but in correctly matching flow shares to distances and the target city.

## 5.3. Limitations and future research

Mobility datasets may contain biases in coverage, representativeness, and preprocessing [@gallotti2024distorted; @pappalardo2023future]. In addition, reducing resolution or aggregating data does not automatically create privacy guarantees. Mobility traces may still contain substantial identifying information after coarsening [@demontjoye2013unique], and providing user-level differential-privacy guarantees for aggregate location data remains practically difficult [@houssiau2022differential]. The present study does not perform a privacy analysis of $Y_D$; therefore, $Y_D$ should be called only a low-dimensional aggregate observation, not a demonstrated privacy-preserving mechanism.

The county-level analysis is exploratory. Only 11 metropolitan areas in the benchmark produce a genuinely multi-county partition, while the remaining 39 cases are equivalent to city-level calibration. Moreover, counties are administrative boundaries and may not accurately represent functional mobility areas. Therefore, this result does not support a general conclusion that finer spatial resolution improves performance.

Two direct limitations of the design are that $Y_D$ is extracted from the target city's ground-truth OD rather than an independent observation source, and that evaluation takes place only on the known positive interzonal support, so zero flows and link discovery are not addressed.

These limitations also define several natural directions for future research. One natural extension is to combine $Y_D$ with other aggregate constraints, such as total outflow by origin or total inflow by destination. Classical spatial interaction models provide a foundation for jointly applying production, attraction, and impedance constraints [@wilson1971family; @ortuzar2011modelling]. Recent discussions of the future of mobility science also emphasize the need for models that are both generalizable and more interpretable and connected to mobility mechanisms [@pappalardo2023future]. Future research could evaluate independent aggregate-observation sources, different geographic units, and real collection conditions; the present study does not use an external observation source.

The effectiveness of calibration varies across baseline architectures. Consistent CPC gains appear in most cities for the two neural baselines, but in only 22/50 cities for the Two-Parameter Gravity model. This indicates that the practical benefit of $Y_D$ depends on the initial prediction structure of the baseline.

# 6. Conclusion

This study examines whether the target city's distance-binned mobility distribution provides additional information for a frozen-parameter cross-city zero-shot baseline. The calibration uses $Y_D$ only at inference time and does not update model parameters.

Across 50 U.S. cities, calibration with $Y_D$ increases CPC by an average of $+0.00354$, with 45/50 cities improving over the baseline. This result shows that the target city's distance distribution contains a small but relatively consistent amount of additional information that the zero-shot baseline does not fully capture. Sensitivity analyses show that this value depends on the resolution and quality of the observation, and that the benefit depends on preserving the correct association between the target city's flow shares and distance intervals.

The improvement is small in absolute magnitude and should be understood as an additional calibration rather than a replacement for detailed OD data. The conclusions are limited to intensity reconstruction on known positive interzonal support with oracle $Y_D$; the study does not evaluate link discovery, full-matrix reconstruction, or the use of independently collected $Y_D$ in real-world deployment.

# 7. Data and Code Accessibility Statement

To be added later

# 8. Scientific Statements and Commitments

To be added later

# 9. References

1. **Barbosa, H., Barthelemy, M., Ghoshal, G., James, C. R., Lenormand, M., Louail, T., Menezes, R., Ramasco, J. J., Simini, F., & Tomasini, M.** (2018). Human mobility: Models and applications. *Physics Reports*, 734, 1–74. [https://doi.org/10.1016/j.physrep.2018.01.001](https://doi.org/10.1016/j.physrep.2018.01.001)

2. **de Montjoye, Y.-A., Hidalgo, C. A., Verleysen, M., & Blondel, V. D.** (2013). Unique in the crowd: The privacy bounds of human mobility. *Scientific Reports*, 3, 1376. [https://doi.org/10.1038/srep01376](https://doi.org/10.1038/srep01376)

3. **Efron, B., & Tibshirani, R. J.** (1993). *An introduction to the bootstrap*. Chapman & Hall.

4. **Enaya, A., Zhong, C., Batty, M., Morphet, R., & Lopane, F. D.** (2026). TransGM: Transferable gravity models for cross-city policy transfer. *Computers, Environment and Urban Systems*, 128, 102455. [https://doi.org/10.1016/j.compenvurbsys.2026.102455](https://doi.org/10.1016/j.compenvurbsys.2026.102455)

5. **GADM.** (n.d.). *GADM database of global administrative areas (Version 4.1)* [Data set]. Retrieved September 2, 2026, from [https://gadm.org/data.html](https://gadm.org/data.html)

6. **Gallotti, R., Maniscalco, D., Barthelemy, M., & De Domenico, M.** (2024). Distorted insights from human mobility data. *Communications Physics*, 7, 421. [https://doi.org/10.1038/s42005-024-01909-x](https://doi.org/10.1038/s42005-024-01909-x)

7. **Grogger, J. T., & Carson, R. T.** (1991). Models for truncated counts. *Journal of Applied Econometrics*, 6(3), 225–238. [https://doi.org/10.1002/jae.3950060302](https://doi.org/10.1002/jae.3950060302)

8. **Guo, J., Bai, S., Li, X., Xian, K., Liu, E., Ding, W., & Ma, X.** (2025). A universal geography neural network for mobility flow prediction in planning scenarios. *Computer-Aided Civil and Infrastructure Engineering*, 40, 5769–5789. [https://doi.org/10.1111/mice.13398](https://doi.org/10.1111/mice.13398)

9. **Holm, S.** (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70. [https://www.jstor.org/stable/4615733](https://www.jstor.org/stable/4615733)

10. **Houssiau, F., Rocher, L., & de Montjoye, Y.-A.** (2022). On the difficulty of achieving differential privacy in practice: User-level guarantees in aggregate location data. *Nature Communications*, 13, 29. [https://doi.org/10.1038/s41467-021-27566-0](https://doi.org/10.1038/s41467-021-27566-0)

11. **Hyman, G. M.** (1969). The calibration of trip distribution models. *Environment and Planning A*, 1(1), 105–112. [https://doi.org/10.1068/a010105](https://doi.org/10.1068/a010105)

12. **Lenormand, M., Bassolas, A., & Ramasco, J. J.** (2016). Systematic comparison of trip distribution laws and models. *Journal of Transport Geography*, 51, 158–169. [https://doi.org/10.1016/j.jtrangeo.2015.12.008](https://doi.org/10.1016/j.jtrangeo.2015.12.008)

13. **Loshchilov, I., & Hutter, F.** (2019). Decoupled weight decay regularization. In *International Conference on Learning Representations (ICLR)*. [https://openreview.net/forum?id=Bkg6RiCqY7](https://openreview.net/forum?id=Bkg6RiCqY7)

14. **Merlin, L. A.** (2020). A new method using medians to calibrate single-parameter spatial interaction models. *Journal of Transport and Land Use*, 13(1), 49–70. [https://doi.org/10.5198/jtlu.2020.1614](https://doi.org/10.5198/jtlu.2020.1614)

15. **Ortúzar, J. de D., & Willumsen, L. G.** (2011). *Modelling transport* (4th ed.). John Wiley & Sons. [https://doi.org/10.1002/9781119993308](https://doi.org/10.1002/9781119993308)

16. **Pappalardo, L., Manley, E., Sekara, V., & Alessandretti, L.** (2023). Future directions in human mobility science. *Nature Computational Science*, 3, 588–600. [https://doi.org/10.1038/s43588-023-00469-4](https://doi.org/10.1038/s43588-023-00469-4)

17. **Simini, F., Barlacchi, G., Luca, M., & Pappalardo, L.** (2021). A Deep Gravity model for mobility flows generation. *Nature Communications*, 12, 6576. [https://doi.org/10.1038/s41467-021-26752-4](https://doi.org/10.1038/s41467-021-26752-4)

18. **Verma, R., & Ukkusuri, S. V.** (2025). What determines travel time and distance decay in spatial interaction and accessibility? *Journal of Transport Geography*, 122, 104061. [https://doi.org/10.1016/j.jtrangeo.2024.104061](https://doi.org/10.1016/j.jtrangeo.2024.104061)

19. **Wilcoxon, F.** (1945). Individual comparisons by ranking methods. *Biometrics Bulletin*, 1(6), 80–83. [https://doi.org/10.2307/3001968](https://doi.org/10.2307/3001968)

20. **Wilson, A. G.** (1971). A family of spatial interaction models, and associated developments. *Environment and Planning A*, 3(1), 1–32. [https://doi.org/10.1068/a030001](https://doi.org/10.1068/a030001)

21. **Yang, Y., Herrera, C., Eagle, N., & González, M. C.** (2014). Limits of predictability in commuting flows in the absence of data for calibration. *Scientific Reports*, 4, 5662. [https://doi.org/10.1038/srep05662](https://doi.org/10.1038/srep05662)


# Supplementary Methods

## S1. Detailed GNN neural-network architecture and numerical stability

### S1.1. Urban GNN Encoder tensor layers

The Urban GNN maps the 26-dimensional urban-feature vector $\mathbf{x}_{c,i} \in \mathbb{R}^{26}$ and the spatial-radius graph structure $\mathcal{G}_c = (\mathcal{V}_c, \mathcal{E}_c)$ into a 64-dimensional hidden representation $\mathbf{h}_{c,i} \in \mathbb{R}^{64}$:

1. **Initial node projection**:
$$
\mathbf{h}_{c,i}^{(0)} = \operatorname{Dropout}\bigl(\operatorname{ReLU}\bigl(\operatorname{LayerNorm}(\mathbf{W}_{\mathrm{in}} \mathbf{x}_{c,i} + \mathbf{b}_{\mathrm{in}})\bigr)\bigr).
$$

2. **Distance-conditioned message**:
$$
\mathbf{m}_{ji}^{(\ell)} = \mathbf{W}_{\mathrm{msg}}^{(\ell)} \left[ \mathbf{h}_{c,j}^{(\ell-1)} \,\Vert\, \log(1 + d_{c,ji}) \right] + \mathbf{b}_{\mathrm{msg}}^{(\ell)}.
$$

3. **Message aggregation**:
$$
\mathbf{a}_{c,i}^{(\ell)} = \frac{1}{\max(\operatorname{deg}(i), 1)} \sum_{j \in \mathcal{N}(i)} \mathbf{m}_{ji}^{(\ell)}.
$$

4. **Node-state transformation**:
$$
\widetilde{\mathbf{h}}_{c,i}^{(\ell)} = \operatorname{LayerNorm}\bigl(\operatorname{ReLU}\bigl(\mathbf{a}_{c,i}^{(\ell)} + \mathbf{W}_{\mathrm{self}}^{(\ell)} \mathbf{h}_{c,i}^{(\ell-1)} + \mathbf{b}_{\mathrm{self}}^{(\ell)}\bigr)\bigr).
$$

5. **Residual update**:
$$
\mathbf{h}_{c,i}^{(\ell)} = \mathbf{h}_{c,i}^{(\ell-1)} + \operatorname{Dropout}\bigl(\widetilde{\mathbf{h}}_{c,i}^{(\ell)}\bigr).
$$

6. **Output projection**:
$$
\mathbf{h}_{c,i} = \mathbf{W}_{\mathrm{out}} \mathbf{h}_{c,i}^{(2)} + \mathbf{b}_{\mathrm{out}} \in \mathbb{R}^{64}.
$$

### S1.2. Numerical stability and gradient clipping

During training, the ZTNB log-likelihood is computed using `torch.lgamma`. To prevent numerical overflow or vanishing gradients:

* The base mean parameter is lower-bounded: $\mu_{c,ij} = \operatorname{softplus}(\log T_{c,ij}^{\mathrm{grav}} + \operatorname{residual}_{c,ij}) + 10^{-4}$.
* The dispersion parameter is bounded in log space: $\log \phi_{\mathrm{safe}} = \operatorname{clamp}(\log \phi, \text{min}=-10.0, \text{max}=10.0)$, followed by $\phi = \exp(\log \phi_{\mathrm{safe}})$.
* A stabilizing constant $\epsilon = 10^{-8}$ is added to $\mu$ and $\phi$ in logarithmic terms; the probability at 0 is normalized numerically through $\log(1 - p_{\mathrm{NB}}(0)) = \operatorname{log1p}(-\exp(\log p_{\mathrm{NB}}(0)))$ with an upper bound of $1.0 - 10^{-7}$. When inferring the conditional expectation, the denominator $1 - p_{\mathrm{NB}}(0)$ is lower-bounded by $10^{-6}$.
* The gradient of all model parameters is clipped to a maximum Euclidean norm of $\|\mathbf{g}\|_2 \le 5.0$ using `torch.nn.utils.clip_grad_norm_`.

### S1.3. Architecture hyperparameters and baseline separation

The exact hyperparameter configuration extracted directly from the trained model checkpoints (`results/checkpoints/5fold_*.pt` and `mlp_*.pt`) is presented in Table S1.

#### Table S1: Architectural and training hyperparameters of zero-shot baselines
| Component | Hyperparameter | Value | Description |
|:---|:---|:---:|:---|
| **Urban GNN Encoder** | Input feature dimension ($d_{\mathrm{in}}$) | 26 | Demographics, socio-economic, and urban features |
| | Message passing layers | 2 | Distance-conditioned `GraphConvLayer` |
| | Attention mechanism / heads | N/A (0) | Standard mean aggregation; no attention layers |
| | Hidden / Output dimension | 64 | LayerNorm(64) + ReLU + Dropout |
| | Dropout rate | 0.1 | Input projection, residual updates, output |
| | Spatial graph type | Radius graph | Geographic radius $r = 5.0$ km with self-loops |
| **Pairwise Decoder** | Input dimension | 130 | Concatenation $[\mathbf{h}_i \,(64) \parallel \mathbf{h}_j \,(64) \parallel \log(1+d) \,(1) \parallel \log T^{\mathrm{grav}} \,(1)]$ |
| | Hidden layers | [64, 32] | Layer 1: 64 (LayerNorm+ReLU+Dropout); Layer 2: 32 (ReLU+Dropout) |
| | Output layer | 1 | Zero-initialized neural residual head |
| | Internal gravity prior | Trainable $(G, \alpha)$ | Initialized at $G_0=0.0, \alpha_0=1.0$; trained end-to-end via AdamW |
| **Optimization** | Loss objective | ZTNB NLL | Zero-Truncated Negative Binomial likelihood |
| | Optimizer | AdamW | Macro-averaged city-by-city steps |
| | Initial learning rate | 0.0032 | $3.2 \times 10^{-3}$ |
| | Weight decay | 0.0001 | $10^{-4}$ |
| | Learning rate scheduler | ReduceLROnPlateau | Factor 0.5, patience 4 epochs, min LR $10^{-5}$ |
| | Early stopping patience | 16 epochs | Monitored on validation interzonal CPC ($\min \Delta = 10^{-4}$) |
| | Parameter count | 33,668 | Identical parameter count for Urban GNN and Node MLP |

**Parameter separation note:** The two parameters $(G_{\mathrm{NN}}, \alpha_{\mathrm{NN}})$ of the neural gravity prior are internal, trainable variables optimized end-to-end with the network via AdamW and saved inside the checkpoint bundle. Conversely, the standalone Two-Parameter Gravity baseline is fitted independently via pooled log-linear OLS on the training cities ($G_{\mathrm{OLS}} \approx -8.54, \alpha_{\mathrm{OLS}} \approx 1.66$ on Fold 1). The two models do not share coefficients.

## S2. General form of the analytic calibration operator ($q \in [0, 1]$)

The calibration-intensity parameter $q \in [0, 1]$ controls the degree of intervention from target-distance information:

- $q = 0$: retain the baseline's initial prediction ($\widehat{t}^{(1)} \equiv \widehat{t}^{(0)}$).
- $q = 1$: fully match flow shares across active distance intervals.
- The main study fixes $q = 1$.

At the main configuration $K=8$, 40/50 cities have all eight intervals active; in the remaining 10 cities, one or more long-distance intervals contain zero OD pairs, leading to $K_{\mathrm{act},c}\in[5,7]$. The algorithm operates strictly over the active set $\mathcal A_c$.

The general calibration procedure is performed as follows:

### S2.1. Set of active intervals

The set of active intervals $\mathcal A_c$ is determined directly by the presence of OD pairs in the support:
$$
\mathcal A_c = \left\{ b \in \{1, \dots, K\} : \exists(i,j) \in \Omega_c,\ d_{c,ij} \in I_b \right\},
$$
with $K_{\mathrm{act},c} = |\mathcal A_c|$. Because zero-shot predictions $\widehat{t}_{c,ij}^{(0)}$ are strictly positive on $\Omega_c$, this is mathematically equivalent to:
$$
\mathcal A_c = \{ b \in \{1, \dots, K\} : \widehat{Y}_{c,b}^{(0)} > 0 \}.
$$

### S2.2. Conditional target distribution over active intervals

The target shares are conditioned on the active intervals as follows:
$$
p_{c,b}^{\mathrm{cond}} = \frac{Y_{c,b} \mathbf{1}(b \in A_c)}{\sum_{r \in A_c} Y_{c,r}}.
$$
Conditioning ensures that the shares over active intervals sum to 1.

### S2.3. Soft calibration weights

For each active interval $b \in A_c$, the soft scaling ratio is computed as:
$$
w_{c,b}(q) = \biggl( \frac{p_{c,b}^{\mathrm{cond}}}{\widehat{Y}_{c,b}^{(0)}} \biggr)^q, \qquad b \in A_c.
$$

### S2.4. Normalization and scaling coefficients

The normalization coefficient that preserves total mass and the corresponding scaling coefficient are:
$$
Z_c(q) = \sum_{r \in A_c} \widehat{Y}_{c,r}^{(0)} w_{c,r}(q), \qquad s_{c,b}(q) = \frac{w_{c,b}(q)}{Z_c(q)}.
$$

### S2.5. Post-calibration prediction

The calibrated predicted flow intensity for pair $(i,j)$ is defined by:
$$
\widehat{t}_{c,ij}^{(1)} = s_{c,b(i,j)}(q) \widehat{t}_{c,ij}^{(0)},
$$
where $b(i,j)$ is the distance interval containing pair $(i,j)$.

### S2.6. Main case $q = 1$

When all distance intervals are active:
$$
A_c = \{1, \dots, K\},
$$
we have:
$$
p_{c,b}^{\mathrm{cond}} = Y_{c,b}, \qquad Z_c(1) = 1, \qquad s_{c,b}(1) = \frac{Y_{c,b}}{\widehat{Y}_{c,b}^{(0)}}.
$$
The general form then reduces exactly to the simplified calibration operator used in the main text.

## S3. Analytic proofs of invariant properties

### S3.1. Support preservation

Because $s_{c,b}(q) > 0$ on every active interval, a positive prediction before calibration remains positive afterward. The operator acts only on $\Omega_c$, so it does not create links outside the known support:
$$
\widehat{t}_{c,ij}^{(1)} > 0 \quad \Longleftrightarrow \quad \widehat{t}_{c,ij}^{(0)} > 0, \qquad (i,j) \in \Omega_c.
$$

### S3.2. Within-interval rank preservation

For two pairs $(i,j)$ and $(u,v)$ in the same interval $b$, we have:
$$
\frac{\widehat{t}_{c,ij}^{(1)}}{\widehat{t}_{c,uv}^{(1)}} = \frac{s_{c,b}(q) \widehat{t}_{c,ij}^{(0)}}{s_{c,b}(q) \widehat{t}_{c,uv}^{(0)}} = \frac{\widehat{t}_{c,ij}^{(0)}}{\widehat{t}_{c,uv}^{(0)}}.
$$
Therefore, the relative ordering of pairs within the same interval does not change ($\tau = 1$).

### S3.3. Preservation of total predicted mass

Let $S_c^{(0)}$ be the total predicted mass of the baseline:
$$
S_c^{(0)} = \sum_{(i,j) \in \Omega_c} \widehat{t}_{c,ij}^{(0)}.
$$
The total flow mass after calibration satisfies:
$$
\begin{aligned}
\sum_{(i,j) \in \Omega_c} \widehat{t}_{c,ij}^{(1)} &= S_c^{(0)} \sum_{b \in A_c} \widehat{Y}_{c,b}^{(0)} s_{c,b}(q) \\
&= \frac{S_c^{(0)}}{Z_c(q)} \sum_{b \in A_c} \widehat{Y}_{c,b}^{(0)} w_{c,b}(q) \\
&= S_c^{(0)}.
\end{aligned}
$$
Because $S_c^{(0)}$ is exactly the total predicted mass before calibration, the operator preserves the baseline's total predicted mass.

## S4. Mathematical definitions of supplementary error metrics

All supplementary error metrics are computed on the same known positive interzonal support $\Omega_c$. CPC remains the primary metric; the metrics below serve only as robustness checks.

1. **Mean absolute error (MAE)**:
$$
\operatorname{MAE}_c = \frac{1}{\lvert\Omega_c\rvert} \sum_{(i,j)\in\Omega_c} \lvert t_{c,ij} - \widehat{t}_{c,ij} \rvert.
$$

2. **Root mean squared error (RMSE)**:
$$
\operatorname{RMSE}_c = \sqrt{ \frac{1}{\lvert\Omega_c\rvert} \sum_{(i,j)\in\Omega_c} \bigl( t_{c,ij} - \widehat{t}_{c,ij} \bigr)^2 }.
$$

3. **Normalized RMSE (NRMSE)**:
$$
\overline{t}_c = \frac{1}{\lvert\Omega_c\rvert} \sum_{(i,j)\in\Omega_c} t_{c,ij}, \qquad \operatorname{NRMSE}_c = \frac{\operatorname{RMSE}_c}{\overline{t}_c}.
$$

4. **Log-scale RMSE ($\operatorname{RMSE}_{\mathrm{log1p}}$)**:
$$
\operatorname{RMSE}_{\mathrm{log1p},c} = \sqrt{ \frac{1}{\lvert\Omega_c\rvert} \sum_{(i,j)\in\Omega_c} \bigl[ \log(1+t_{c,ij}) - \log(1+\widehat{t}_{c,ij}) \bigr]^2 }.
$$

5. **Spearman rank-correlation coefficient ($\rho_{\mathrm{Spearman}}$)**: measures the monotonic association between observed and predicted intensities on $\Omega_c$. Larger values indicate better rank agreement.

6. **Total-flow relative error ($\operatorname{RelError}$)**:
$$
\operatorname{RelError}_c = \frac{ \left\lvert \sum_{(i,j)\in\Omega_c} \widehat{t}_{c,ij} - \sum_{(i,j)\in\Omega_c} t_{c,ij} \right\rvert }{ \sum_{(i,j)\in\Omega_c} t_{c,ij} }.
$$


## S5. Fold-stratified bootstrap protocol and statistical testing

1. **Stratified paired nonparametric bootstrap by fold**:
   - The resampling unit is the city.
   - Resampling with replacement is conducted separately within each fold.
   - Each fold resamples 10 cities from its 10 original test cities.
   - The two conditions $M_0$ and $M_1$ are strictly paired.
   - OD pairs are not resampled independently.

   Let $\mathcal{C}^{*(r)}$ be the multiset of 50 resampled cities in bootstrap replicate $r$ ($r = 1, \dots, B$ with $B = 10{,}000$ and $C = 50$):
$$
\overline{\Delta}^{*(r)} = \frac{1}{C} \sum_{c\in\mathcal{C}^{*(r)}} \Delta_c, \qquad r = 1, \dots, B.
$$

2. **95% bootstrap confidence interval**:
$$
\mathrm{CI}_{95\%} = \bigl[ Q_{0.025}\bigl(\overline{\Delta}^*\bigr), Q_{0.975}\bigl(\overline{\Delta}^*\bigr) \bigr].
$$

3. **Two-sided Wilcoxon signed-rank test**: tests whether paired differences are symmetrically distributed around 0.

4. **Holm–Bonferroni correction** [@holm1979sequential]:
   For a family of $M$ hypotheses:
$$
p_{(k)} \leq \frac{\alpha}{M - k + 1}, \qquad k = 1, \dots, M.
$$
   The $p$-values are sorted in ascending order. The step-down procedure stops at the first hypothesis that does not satisfy the rejection condition.


## S6. Technical details of robustness stress tests

1. **Total Variation noise synthesis (TV noise bisection)**:
   * Applied to active bins with $p_b > 0$ ($b = 1, \dots, K_{\mathrm{act}}$). The normalized noise vector is centered exactly as in the source-code implementation:
$$
z_b^{\mathrm{ctr}} = z_b - \frac{1}{K_{\mathrm{act}}} \sum_{r=1}^{K_{\mathrm{act}}} z_r, \qquad z_b \sim \mathcal{N}(0, 1).
$$
   * The noisy shares are obtained by exponential tilting:
$$
p_b(\sigma) = \frac{\exp\bigl(\log p_b + \sigma z_b^{\mathrm{ctr}}\bigr)}{\sum_{r=1}^{K_{\mathrm{act}}} \exp\bigl(\log p_r + \sigma z_r^{\mathrm{ctr}}\bigr)}.
$$
   * The scale $\sigma$ is solved by bisection so that the Total Variation distance reaches the specified level $\epsilon$:
$$
\operatorname{TV}\bigl(p(\sigma), p\bigr) = \frac{1}{2} \sum_{b=1}^{K_{\mathrm{act}}} \lvert p_b(\sigma) - p_b \rvert = \epsilon.
$$

2. **Placebo controls and intervention dose matching (Dose-Matched Controls)**:

   To isolate the specific informative value of the target-city distance distribution from the pure effect of intervention magnitude, control distributions are normalized to match the log-ratio norm of the target distribution $Y_D^{\mathrm{target}}$. For each evaluated city, let $\widehat{Y}^{(0)}$ denote the distance distribution predicted by the zero-shot baseline $M_0$ over active intervals ($b \in \mathcal A_c$). The log-ratio vector of the target distribution and its centered root-mean-square intervention magnitude $D_T$ ($D_T = \|\tilde{\mathbf{r}}_T\|_2 / \sqrt{K_{\mathrm{act}}}$) are given by:

$$
r_{T,b} = \log\left(\frac{Y_{D,b}^{\mathrm{target}}}{\widehat{Y}_b^{(0)}}\right), \qquad \tilde{r}_{T,b} = r_{T,b} - \frac{1}{K_{\mathrm{act}}} \sum_{m\in\mathcal A_c} r_{T,m}, \qquad D_T = \sqrt{\frac{1}{K_{\mathrm{act}}} \sum_{b\in\mathcal A_c} \tilde{r}_{T,b}^2}.
$$

   - **Training-city donor control (Wrong-City Donors, Dose-Matched)**: For each random donor draw from training cities within the same fold ($B_{\mathrm{draw}} = 1,000$), let $Y_D^{\mathrm{donor}}$ be the donor distribution. The raw log-ratio and donor intervention magnitude $D_D$ are computed as:

$$
r_{D,b} = \log\left(\frac{Y_{D,b}^{\mathrm{donor}}}{\widehat{Y}_b^{(0)}}\right), \qquad \tilde{r}_{D,b} = r_{D,b} - \frac{1}{K_{\mathrm{act}}} \sum_{m\in\mathcal A_c} r_{D,m}, \qquad D_D = \sqrt{\frac{1}{K_{\mathrm{act}}} \sum_{b\in\mathcal A_c} \tilde{r}_{D,b}^2}.
$$

   If $D_D > 0$, the log-ratio vector of the donor is scaled to match the target intervention dose:

$$
\tilde{r}_{D,b}^* = \tilde{r}_{D,b} \frac{D_T}{D_D}.
$$

   The dose-matched donor control distribution is then reconstructed via:

$$
p_{D,b}^* = \frac{\widehat{Y}_b^{(0)} \exp(\tilde{r}_{D,b}^*)}{\displaystyle\sum_{m\in\mathcal A_c} \widehat{Y}_m^{(0)} \exp(\tilde{r}_{D,m}^*)}, \qquad b \in \mathcal A_c.
$$

   In the degenerate case $D_D < 10^{-12}$ (where the donor distribution happens to perfectly match the baseline prediction), no perturbation direction can be scaled; the implementation directly assigns the target benchmark gain ($\Delta\mathrm{CPC} = \Delta\mathrm{CPC}_{\mathrm{target}}$).

   - **Fold training-mean donor control (Training-Mean Donor, Dose-Matched)**: The pooled mean distribution $\overline{Y}_{D,\mathrm{train}}$ is computed across all training cities in the corresponding fold. Its log-ratio and initial intervention magnitude $D_M$ are:

$$
r_{M,b} = \log\left(\frac{\overline{Y}_{D,\mathrm{train},b}}{\widehat{Y}_b^{(0)}}\right), \qquad \tilde{r}_{M,b} = r_{M,b} - \frac{1}{K_{\mathrm{act}}} \sum_{m\in\mathcal A_c} r_{M,m}, \qquad D_M = \sqrt{\frac{1}{K_{\mathrm{act}}} \sum_{b\in\mathcal A_c} \tilde{r}_{M,b}^2}.
$$

   If $D_M > 0$, the vector is scaled to match $D_T$:

$$
\tilde{r}_{M,b}^* = \tilde{r}_{M,b} \frac{D_T}{D_M}.
$$

   The dose-matched training-mean distribution is then reconstructed via:

$$
p_{M,b}^* = \frac{\widehat{Y}_b^{(0)} \exp(\tilde{r}_{M,b}^*)}{\displaystyle\sum_{m\in\mathcal A_c} \widehat{Y}_m^{(0)} \exp(\tilde{r}_{M,m}^*)}, \qquad b \in \mathcal A_c.
$$

   If $D_M < 10^{-12}$, the code directly assigns $\Delta\mathrm{CPC} = \Delta\mathrm{CPC}_{\mathrm{target}}$.

   - **Permuted distance-interval control (Permuted Target $Y_D$)**: To verify whether the physical ordering between flow shares and distance bins matters, the centered log-ratio vector $\tilde{\mathbf{r}}_T$ is randomly permuted across active bins ($B_{\mathrm{perm}} = 1,000$ independent random permutations; for cities with small active bin counts, exhaustive unique permutations are used): $\tilde{r}_{P,b} = \tilde{r}_{T,\pi(b)}$, where $\pi$ is a uniform permutation over $\mathcal A_c$. Because permutation strictly preserves the centered $\ell_2$ and RMS norms ($\|\tilde{\mathbf{r}}_P\|_2 = \|\tilde{\mathbf{r}}_T\|_2 = \sqrt{K_{\mathrm{act}}} D_T$), this control strictly maintains the intervention dose $D_T$ of the target distribution while completely severing the semantic association between distance and flow volume. The permuted distribution is reconstructed via:

$$
p_{P,b} = \frac{\widehat{Y}_b^{(0)} \exp(\tilde{r}_{P,b})}{\displaystyle\sum_{m\in\mathcal A_c} \widehat{Y}_m^{(0)} \exp(\tilde{r}_{P,m}^*)}, \qquad b \in \mathcal A_c.
$$


## S7. Exploratory analysis of county-level spatial resolution

### S7.1. Setup

This exploratory analysis tests whether providing an aggregate distance observation at a finer spatial resolution than the city level, specifically grouped by county, provides additional information.

County boundaries are obtained from the Database of Global Administrative Areas, version 4.1 (GADM 4.1) [@gadm41]. Each tract is assigned to its corresponding containing county through a point-in-polygon join between the tract centroid and the county polygon. If a tract centroid lies on a polygon boundary or near the coast, the procedure assigns the nearest polygon in EPSG:5070, subject to a maximum distance of 5 km. Each tract is assigned to exactly one county. GADM is used strictly for this spatial grouping step, not as the source of centroid coordinates, urban features, or OD flows.

Let $g(i)$ be the county assigned to tract $i$. OD pairs are grouped by the **county of the origin tract**:
$$
\Omega_{c,\ell} = \left\{(i,j) \in \Omega_c : g(i) = \ell\right\}.
$$
The destination tract $j$ may lie in the same or a different county within the metropolitan area. The distance distribution for origin-county group $\ell$ is defined as:
$$
Y_{c,\ell,b} = \frac{\sum_{(i,j) \in \Omega_{c,\ell}} t_{c,ij} \mathbf{1}(d_{c,ij} \in I_b)}{\sum_{(i,j) \in \Omega_{c,\ell}} t_{c,ij}}, \qquad \sum_{b=1}^K Y_{c,\ell,b} = 1.
$$
Because the inputs are restricted to the tract set of the laboratory-provided metropolitan area, $Y_{D,c,\ell}$ describes the distance distribution originating from tracts in county $\ell$ within that metropolitan area; it does not represent all movement across the county outside the study scope.

Each distribution $Y_{D,c,\ell}$ is used to calibrate OD pairs whose origin tract belongs to county $\ell$. The calibrated predictions from all county groups are then assembled into the complete metropolitan-area prediction:
$$
\widehat{\mathbf{T}}_c^{\mathrm{county}} = \bigcup_{\ell \in \mathcal{G}_c} \left\{ \widehat{t}_{c,ij}^{\mathrm{county}} : (i,j) \in \Omega_{c,\ell} \right\},
$$
where $\mathcal{G}_c$ is the set of counties appearing in the data for metropolitan area $c$.

Importantly, changing the observation resolution from city level to county level does not change the evaluation scope: the model still reconstructs and is evaluated on the target metropolitan area's support $\Omega_c$; only the aggregate supervision signal in the calibration step becomes spatially more detailed.

Among the 50 benchmark metropolitan areas, exactly 39 are single-county areas, where all tracts belong to one county and $\lvert\mathcal{G}_c\rvert = 1$. For these 39 areas, the county partition exactly matches the city-level partition, yielding $M_{1,\mathrm{county}} \equiv M_{1,\mathrm{city}}$ and $\Delta\mathrm{CPC}_{\mathrm{res},c} = 0$ mathematically. Only 11 metropolitan areas spanning 2 to 7 counties produce a genuinely new partition.

### S7.2. Results

Across all 50 metropolitan areas, the pooled additional increase from county-level calibration over city-level calibration is very small:
$$
\Delta\mathrm{CPC}_{\mathrm{res}} = +0.00014, \quad \text{95% CI } [+0.00002,\,+0.00028], \quad \text{Wilcoxon } p = 0.0064.
$$

This modest pooled increase is driven by the 39 single-county areas, whose increase is exactly zero by construction.

For the 11 multi-county metropolitan areas, which comprise 22% of the benchmark, county-level calibration improves performance in 9/11 areas, with a mean additional increase of $+0.00063$ (Table S2 and Figure S1).

![Figure S1](figures/fig_s1_spatial_resolution.png)
**Figure S1. Comparison of CPC gains from city-level and county-level calibration across 11 multi-county metropolitan areas. The analysis is exploratory; the 39 single-county areas are omitted because the two groupings are mathematically equivalent.**

### Table S2: Descriptive city-level results for the multi-county spatial-resolution analysis

*The table compares the zero-shot baseline ($M_0$), city-level oracle calibration ($M_{1,\mathrm{city}}$), and origin-county-conditioned oracle calibration ($M_{1,\mathrm{county}}$) for 11 metropolitan datasets whose tracts are assigned to more than one county. The resolution gain is defined as $\Delta\mathrm{CPC}_{\mathrm{res},c} = \operatorname{CPC}(M_{1,\mathrm{county}}) - \operatorname{CPC}(M_{1,\mathrm{city}})$. Values are descriptive estimates at the city level. Confidence intervals and hypothesis tests are not reported for the subgroup because no separately verified uncertainty artifact is available.*

| City | Number of origin counties | $M_0$ CPC | $M_{1,\mathrm{city}}$ CPC | $M_{1,\mathrm{county}}$ CPC | $\Delta\mathrm{CPC}_{\mathrm{city}}$ | $\Delta\mathrm{CPC}_{\mathrm{county}}$ | $\Delta\mathrm{CPC}_{\mathrm{res},c}$ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Kansas City | 3 | 0.721071 | 0.726877 | 0.729612 | +0.005807 | +0.008542 | +0.002735 |
| New York | 7 | 0.524464 | 0.525775 | 0.527870 | +0.001311 | +0.003407 | +0.002096 |
| Dallas | 3 | 0.685251 | 0.695768 | 0.696916 | +0.010517 | +0.011665 | +0.001148 |
| Denver | 3 | 0.715551 | 0.715713 | 0.716053 | +0.000162 | +0.000501 | +0.000339 |
| Omaha | 2 | 0.747005 | 0.752621 | 0.752828 | +0.005616 | +0.005822 | +0.000207 |
| Tulsa | 2 | 0.779746 | 0.781563 | 0.781750 | +0.001817 | +0.002005 | +0.000187 |
| Detroit | 2 | 0.684499 | 0.685059 | 0.685239 | +0.000560 | +0.000740 | +0.000180 |
| Chicago | 2 | 0.672433 | 0.674337 | 0.674358 | +0.001905 | +0.001925 | +0.000021 |
| Boston | 3 | 0.687180 | 0.687561 | 0.687578 | +0.000381 | +0.000398 | +0.000017 |
| Milwaukee | 2 | 0.741276 | 0.742868 | 0.742854 | +0.001591 | +0.001578 | -0.000014 |
| Atlanta | 2 | 0.710814 | 0.719676 | 0.719645 | +0.008862 | +0.008831 | -0.000031 |
| **Multi-county mean** | — | — | — | — | — | — | **+0.000626** |
| **Number of cities with positive increase** | — | — | — | — | — | — | **9 / 11** |

### S7.3. Interpretive limitations

The county-level analysis should be interpreted under the following strict limitations:

1. **Small sample and descriptive evidence**: The analysis is based on only 11 multi-county metropolitan areas. Because no separately stratified uncertainty estimate is available for this subset, the 9/11 improvement result is an empirical descriptive finding and is insufficient to establish a general statistical regularity.
2. **Administrative versus functional boundaries**: Counties are historical administrative boundaries, not boundaries designed around commuting sheds, transportation corridors, or functional urban zones. County grouping therefore need not accurately represent heterogeneity in mobility behavior.
3. **Incomplete spatial scope**: County groups include only tracts within the laboratory-provided metropolitan-area boundaries and do not represent all movement across the full territory of those counties.
4. **No causal or practical guarantee**: Assigning tract centroids geometrically and using an oracle distribution does not reflect real linkage errors. The experiment does not show that increasing spatial resolution generally or always improves OD-matrix reconstruction in practical applications.














