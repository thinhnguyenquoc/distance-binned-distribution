# Implementation Plan (Final — Locked)
## Quantifying the Marginal Value of Distance-Binned Mobility Information

---

## Core Framing

> **This study does not propose another zero-shot OD generator. It quantifies how much additional OD-relevant information a coarse target-city mobility observation contributes beyond zero-shot inference, evaluated over the candidate OD pair support available for each held-out city.**

NeuroGravity (Yang et al., 2026) demonstrated that a physics-informed GNN can reconstruct and transfer OD flows across unseen cities using only urban features and pairwise distance. **This study begins exactly there**: given such a frozen model, what does a target-city distance-binned mobility distribution ($Y_D$) add?

---

## Architecture: Physics-Informed GNN with Pairwise Decoder

$$\boxed{\text{Urban GNN} \rightarrow h_i \rightarrow \text{Pairwise OD Decoder} \rightarrow \pi_{ij},\ \mu_{ij}}$$

### Sub-module 1: Node Encoder (Urban GNN)
$$h_i = \text{GNN}_\theta(X,\ G^{\text{urban}})$$

- **Node features** $X_i$: census (population, income, employment), POI (office, commercial, education density), road (density, motorway length)
- **Graph** $G^{\text{urban}} = g(\text{observable geography only})$ — constructed from spatial proximity, NOT from OD data
- **Graph construction**: pilot both $k$-NN ($k \in \{5, 10, 20\}$) and radius-based graph (radius normalized to city spatial scale). Lock one rule after pilot; must be identical for all cities including held-out targets.
- **Architecture**: GraphBERT-style edge-conditioned attention (as in NeuroGravity)

### Sub-module 2: Gravity Prior
$$\log T^{\text{grav}}_{ij} = G + \log P_i + \log P_j - \alpha \log D_{ij}$$
Classical 2-parameter gravity model (stable for cross-city transfer per NeuroGravity findings). $G$ and $\alpha$ are global trainable parameters.

### Sub-module 3: Pairwise OD Decoder
Edge representation:
$$e_{ij} = [h_i,\ h_j,\ \log D_{ij},\ \log T^{\text{grav}}_{ij}]$$

Two prediction heads:
- **Observed-nonzero probability**: $\pi_{ij} = \sigma(f_\pi(e_{ij}))$
- **Positive flow magnitude**: $\mu_{ij} = \text{softplus}(f_\mu(e_{ij})) = E[T_{ij} \mid T_{ij} > 0]$

> **Note**: $\mu_{ij}$ is parameterized as the conditional mean of the zero-truncated NB, not the unconditional NB mean. This avoids the $\hat{T} \neq \pi\mu$ inconsistency.

Zero-shot expected OD:
$$\boxed{\hat{T}^{ZS}_{ij} = \pi_{ij} \cdot \mu_{ij}}$$

**Training**: negative-sample OD pairs for BCE efficiency, with class-prior correction (weighted BCE so $\pi_{ij}$ remains calibrated).
**Inference**: evaluate decoder on **all $N^2$ OD pairs** — no sampling.

---

## Loss Function (Locked)

$$\boxed{\mathcal{L}_{\text{train}} = -\frac{1}{|\Omega^+|}\sum_{(i,j)\in\Omega^+} \log P_{\text{ZTNB}}(T_{ij};\, \mu_{ij},\, \phi)}$$

where $\Omega^+ = \Omega_c$ = candidate pairs present in the pair files.

**Likelihood choice**: ZTNB is adopted as the **primary conservative assumption** — not proven correct by provenance audit. The audit established that $\Omega_c$ contains only positive-flow pairs and that the missingness mechanism is not a simple distance cutoff, making positive-only selection plausible. NB (unconditional) is used as a **sensitivity model** in pilot runs.

> **Methods wording**: *"Training OD records contain only pairs with observed positive flows. Pairs absent from both distance and OD files are excluded from training and evaluation; their missingness mechanism cannot be determined from available data and they are not treated as confirmed zero flows."*

| Component | Definition |
|---|---|
| $z_{ij}$ | $\mathbf{1}(T_{ij} > 0)$ — observed non-zero flow indicator |
| $\mathcal{L}_{\text{BCE}}$ | Weighted BCE on $z_{ij}$ vs $\pi_{ij}$; negative sampling with class-prior correction |
| $\mathcal{L}_{\text{ZTNB}}$ | Zero-truncated NB NLL on positive pairs only: $\log P_{NB}(t) - \log(1 - P_{NB}(0))$ |
| $\phi$ | Global trainable dispersion, per fold |
| $\lambda_1 = \lambda_2$ | $= 1$, mean losses; monitor loss scale and gradient norms in pilot |
| Zero labels | Acknowledged: $T_{ij}=0$ may be structural zero OR unobserved flow (noted in Limitations) |
| BCE head label | "Observed non-zero flow prediction" — NOT "structural link existence" |

---

## Data Split: 5-Fold City Cross-Validation

Cities stratified by tract count before folding (to balance size distribution):

| Fold | Train (40 cities) | Test (10 cities) |
|:---:|---|---|
| 1–5 | Rotating | Each city tested exactly once |

Total: all 50 cities evaluated as held-out target.

---

## Experimental Pipeline

### Stage A — Cross-city Training
Train $f_\theta$ on source cities. After convergence: **freeze all parameters → $\theta^*$**.

### Stage B — Zero-shot Transfer ($M_0$)
$$(X^{c^*},\ D^{c^*}) \xrightarrow{f_{\theta^*}} \hat{T}^{ZS}$$
No target-city OD used. Pure zero-shot baseline.

---

## Four Experimental Conditions

All conditions share:
1. The **same frozen model** $\theta^*$
2. The **same calibration operator**: bin-wise multiplicative calibration

$$\hat{Y}_D[k] = \frac{\sum_{(i,j) \in \Omega_c \cap B_k} \hat{T}^{ZS}_{ij}}{\sum_{(i,j)\in\Omega_c} \hat{T}^{ZS}_{ij}}, \qquad s_k = \frac{Y_D[k] + \epsilon}{\hat{Y}_D[k] + \epsilon}, \qquad \hat{T}^{\text{cal}}_{ij} = s_{k(i,j)} \cdot \hat{T}^{ZS}_{ij}$$

> **Why not IPF**: distance bins are non-overlapping → single-pass multiplicative rescaling suffices; no iteration needed.

| Condition | $Y_D$ source | Purpose |
|---|---|---|
| **$M_0$** | None | Baseline $R^{ZS}_c$ |
| **$M_1^{\text{oracle}}$** | $Y_D^{\text{GT}}$ computed from $T^{GT}$ | Ceiling: max gain from perfect 4-bin knowledge |
| **$M_1^{\text{real}}$** | $Y_D^{\text{Meta}}$ from Meta mobility data (county-level, averaged over temporal snapshots) | Real-world gain |
| **$M_q$** | $\tilde{Y}_D^{(q)}$ estimated from $q$-fraction Multinomial trip sample | Reference curve for $q^*$ |

---

## Y_D Sources

**Oracle**: $Y_D^{\text{oracle}}[k] = \sum_{(i,j) \in \Omega_c \cap B_k} T^{GT}_{ij}\ /\ \sum_{(i,j)\in\Omega_c} T^{GT}_{ij}$

> **Note**: $Y_D^{\text{oracle}}$ is computed over $\Omega_c$ (candidate pairs only), NOT over all $N^2$ pairs.

**Real**: extracted from Meta mobility distribution maps.
- City → county mapping via FIPS code.
- Average all available temporal snapshots (April–August 2026).
- 4 bins map to Meta categories: `0`, `(0,10)`, `[10,100)`, `100+` km.

> **Y_D support compatibility check** (required before running $M_1^{\text{real}}$): $Y_D^{\text{oracle}}$ is defined over $\Omega_c$; $Y_D^{\text{real}}$ reflects Meta's full observed mobility population. These two are over **different spatial and mobility supports**. Their difference ($\Delta R^{\text{realization}}$) therefore conflates measurement noise, temporal mismatch, and support mismatch — none of which can be separated without additional data. This must be acknowledged in Methods and is the primary reason $\Delta R^{\text{realization}}$ is reported as a descriptive secondary diagnostic, not interpreted causally.

**Multinomial sample** (for $M_q$):
$$n_{ij} \sim \text{Multinomial}\!\left(m,\ \frac{T^{GT}_{ij}}{\sum_{ij} T^{GT}_{ij}}\right), \qquad q = \frac{m}{\sum_{ij} T^{GT}_{ij}}$$
$$\tilde{Y}_D^{(q)}[k] = \frac{\sum_{(i,j) \in B_k} n_{ij}}{m}$$

Repeat with $S=20$ seeds per $m$ level; report mean $R_c(m)$ and uncertainty band.

**Grid defined on absolute trip count $m$** (not on fraction $q$ directly):
$$m \in \{100,\ 500,\ 1\text{k},\ 5\text{k},\ 10\text{k},\ 50\text{k},\ 100\text{k},\ \infty\}$$

Convert to per-city fraction: $q_c = m\ /\ \sum_{ij} T^{GT}_{ij}$.

> **Why $m$-based, not $q$-based grid**: the statistical precision of a 4-category Multinomial estimate depends on sample size $m$, not on city size. A fixed $q$ (e.g., $q=0.01$) corresponds to vastly different $m$ across cities (tens of trips in small cities vs. hundreds of thousands in New York), making the reference curve incomparable across cities. A fixed $m$ grid ensures equal estimation precision at each grid point, making cross-city comparison of $q^*$ meaningful.

> **Why Multinomial, not uniform cell sampling**: mobility data is collected per trip in practice (GPS, CDR, fare gates). Multinomial sampling reflects this; uniform cell sampling would sample mostly zeros, making $\tilde{Y}_D$ estimation noisy and unrealistic.

> **Statistical precision check**: to estimate a 4-bin distribution with per-bin standard error $\leq 0.01$, at least $m \geq 2500$ trips are needed — independent of city size. The grid above spans this threshold.

---

## Metrics

All metrics on **all $N^2$ OD pairs** of the held-out city:

| Metric | Formula | Role |
|---|---|---|
| **CPC** | $\frac{2\sum \min(T_{ij}, \hat{T}_{ij})}{\sum T_{ij} + \sum \hat{T}_{ij}}$ | **Primary** |
| **RMSE-log1p** | $\sqrt{\frac{1}{N_{\text{pairs}}}\sum_{ij}[\log(1+T_{ij}) - \log(1+\hat{T}_{ij})]^2}$ | Secondary (evaluation only); `log1p` handles $T_{ij}=0$ |
| **Pearson $r$** | — | Secondary |

---

## Answering RQ1: $\Delta R$ (Stage B)

$$\boxed{\Delta R_c = R_c^{YD,\text{real}} - R_c^{ZS}}$$

Report across 50 cities:
- Median $\Delta R_c$, mean ± std
- $\Pr(\Delta R_c > 0)$
- Distribution plot + paired Wilcoxon signed-rank test

**Answers**: *Does $Y_D$ improve OD reconstruction beyond zero-shot?*

---

## Answering RQ2: $q^*$ (Stage C)

Build reference curve $R_c(q)$ for each city. Then:

$$\boxed{q_c^* = R_c^{-1}\!\left(R_c^{YD,\text{real}}\right)}$$

Found via monotonic interpolation over the $m$-grid, then converted:
$$q^*_c = m^*_c\ /\ \sum_{ij} T^{GT}_{ij,c}$$

**Interpretation (locked)**:
> The fraction of ideal randomly sampled trips required to derive a distance-bin constraint with the same OD-reconstruction value as the real target-city $Y_D$.

> **Do NOT state**: "$Y_D \equiv q^*\%$ raw OD data." The equivalence holds only under the bin-calibration operator and the ideal Multinomial sampling experiment.

Report:
- Primary: distribution of $q^*_c$ across 50 cities (median, mean ± std) — relative, comparable across city sizes
- Appendix: distribution of $m^*_c$ — absolute trip count, reflects practical data collection cost

**Answers**: *How much ideal trip observation yields an equally useful distance-bin signal?*

---

## Secondary Diagnostic: Realization Gap (Stage D)

$$\boxed{\Delta R_c^{\text{realization}} = R_c^{YD,\text{oracle}} - R_c^{YD,\text{real}}}$$

**Interpretation**: how much of the reconstruction gain available under ideal 4-bin knowledge is realized by the empirical Meta observation. Does NOT attribute the gap to any specific cause (spatial aggregation, temporal mismatch, sampling bias, mobility-purpose mismatch).

Reported in Discussion; does NOT open a new research gap.

---

## Cross-city Analysis

Regress $\Delta R_c$ on city characteristics to answer: **when is $Y_D$ most valuable?**

| Predictor | Proxy for |
|---|---|
| Tract count | City size |
| Mean pairwise $D_{ij}$ | Urban sprawl |
| $H(Y_D^{\text{oracle}})$ | Distance distribution entropy |
| Median income | Socioeconomic profile |
| Population density | Urban compactness |

---

## Codebase Structure

```
src/
├── data/
│   ├── dataset.py          # 50-city loader (census/POI/road + OD + distance)
│   ├── urban_graph.py      # G^urban: k-NN and radius-based construction
│   ├── yd_extractor.py     # Y_D oracle (from GT) and real (from Meta CSVs)
│   ├── trip_sampler.py     # Multinomial trip sampling for M_q
│   └── city_splits.py      # 5-fold stratified city split
├── models/
│   ├── node_encoder.py     # GraphBERT-style urban GNN
│   ├── gravity.py          # Classical 2-param gravity prior
│   ├── decoder.py          # Pairwise OD decoder (π and μ heads)
│   └── zero_shot_model.py  # Full M0 pipeline
├── loss/
│   ├── bce_weighted.py     # Weighted BCE with class-prior correction
│   └── ztnb.py             # Zero-truncated NB NLL
├── calibration/
│   └── bin_calibration.py  # Bin-wise multiplicative calibration (M0 → M1, Mq)
├── training/
│   ├── train.py            # Cross-city training loop
│   └── evaluate.py         # CPC, RMSE-log1p, Pearson on full N² matrix
└── experiment/
    ├── run_experiment.py   # Orchestrates all 4 conditions per fold
    ├── compute_delta_r.py  # ΔR and Pr(ΔR>0) across 50 cities
    └── compute_qstar.py    # q* interpolation per city
```

---

## Decision Table (Complete)

| Component | Decision |
|---|---|
| Urban GNN graph | Spatial proximity graph; pilot k-NN vs radius; invariant: no OD used |
| OD decoder | All-pairs pairwise MLP; N² at inference |
| Gravity prior | Classical 2-param (stable for cross-city) |
| Link head label | "Observed non-zero flow prediction" (not "structural topology") |
| Loss | BCE (weighted) + ZTNB NLL |
| $\phi$ | Global trainable scalar per fold |
| $\lambda_1, \lambda_2$ | = 1, mean losses; monitor gradient norms |
| $Y_D$ calibration | Bin-wise multiplicative (NOT IPF) |
| $M_q$ sampling | Multinomial trip sampling |
| $q^*$ definition | $R_c(q^*_c) = R_c^{YD,\text{real}}$, via monotonic interpolation |
| $q^*$ interpretation | Reconstruction-value equivalent under ideal trip-sampling |
| Realization gap | $\Delta R^{\text{realization}}$; secondary diagnostic, no causal claim |
| $\theta^*$ | Frozen throughout all calibration stages |
| Target-city model update | None |
| Evaluation | Full $N^2$ matrix, no sampling |
