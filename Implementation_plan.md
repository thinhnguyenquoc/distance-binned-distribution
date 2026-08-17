# Implementation Plan (Final — Locked & Verified)
## Quantifying the Marginal Value of Distance-Binned Mobility Information

---

## Core Framing

> **This study does not propose another zero-shot OD generator. It quantifies how much additional OD-relevant information a coarse target-city mobility observation contributes beyond zero-shot inference, evaluated over the candidate OD pair support $\Omega_c$ available for each held-out city.**

NeuroGravity (Yang et al., 2026) demonstrated that a physics-informed GNN can reconstruct and transfer OD flows across unseen cities using only urban features and pairwise distance. **This study begins exactly there**: given such a frozen model, what does a target-city distance-binned mobility distribution ($Y_D$) add?

---

## Architecture: Physics-Informed GNN with Pairwise Decoder

$$\boxed{\text{Urban GNN} \rightarrow h_i \rightarrow \text{Pairwise OD Decoder} \rightarrow \mu_{NB, ij} \rightarrow \hat{T}^{ZS}_{ij} = E[T_{ij} \mid T_{ij} \ge 1]}$$

### Sub-module 1: Node Encoder (Urban GNN)
$$h_i = \text{GNN}_\theta(X,\ G^{\text{urban}})$$

- **Node features** $X_i$: 26 features spanning census (population, income, employment), POI (office, commercial, education density), road (density, motorway length).
- **Graph** $G^{\text{urban}} = g(\text{observable geography only})$ — constructed strictly from spatial coordinates, NEVER from OD data.
- **Graph construction (Locked after Pilot)**: **Radius graph with $r = 5.0\text{ km}$** (with fallback to 1-NN for isolated tracts). Evaluated in topology pilot against $k$-NN ($k \in \{5, 10, 20\}$) and demonstrated superior held-out transfer performance (CPC 0.0454 vs 0.0327–0.0357).
- **StandardScaler**: fitted strictly on source training cities' node features ($X_{\text{train}}$); target city node features are strictly normalized using the pre-fitted scaler.

### Sub-module 2: Gravity Prior
$$\log T^{\text{grav}}_{ij} = G + \log P_i + \log P_j - \alpha \log D_{ij}$$
Classical 2-parameter gravity model. $G$ and $\alpha$ are global trainable parameters shared across all cities in a fold.

### Sub-module 3: Pairwise OD Decoder (Single Base Magnitude Head)
Edge representation:
$$e_{ij} = [h_i,\ h_j,\ \log(1 + D_{ij}),\ \log T^{\text{grav}}_{ij}]$$

Single prediction head producing base Negative Binomial mean:
$$\mu_{NB, ij} = \text{softplus}(f_\mu(e_{ij})) > 0$$

- **Training**: minimizes exact Zero-Truncated Negative Binomial negative log-likelihood $\mathcal{L}_{\text{train}}$ on positive candidate observations $\Omega_c$.
- **Inference**: evaluates exact conditional expectation on candidate support $\Omega_c$:
$$\boxed{\hat{T}^{ZS}_{ij} = E[T_{ij} \mid T_{ij} \ge 1] = \frac{\mu_{NB, ij}}{1 - P_{NB}(0; \mu_{NB, ij}, \phi)} = \frac{\mu_{NB, ij}}{1 - \left(\frac{\phi}{\mu_{NB, ij} + \phi}\right)^\phi}}$$

---

## Loss Function (Locked)

$$\boxed{\mathcal{L}_{\text{train}} = -\frac{1}{|\Omega^+|}\sum_{(i,j)\in\Omega^+} \log P_{\text{ZTNB}}(T_{ij};\, \mu_{NB, ij},\, \phi)}$$

where:
$$\log P_{\text{ZTNB}}(T=t \mid T \ge 1) = \log P_{NB}(t; \mu_{NB}, \phi) - \log(1 - P_{NB}(0; \mu_{NB}, \phi))$$
and $\Omega^+ = \Omega_c$ is the set of candidate pairs present in the pair files ($T_{ij} \ge 1$).

- $\phi = \exp(\log \phi) > 0$ is a global trainable dispersion parameter per fold.
- Missing pairs outside $\Omega_c$ are treated as **unknown/excluded** (not zero-filled).

---

## 5-Fold Stratified City Cross-Validation

Cities stratified by tract count (10 strata $\times$ 5 cities) to balance size distributions across folds:

| Fold | Train (40 cities) | Test (10 held-out cities) |
|:---:|---|---|
| 1–5 | Rotating 40 source cities | Each city evaluated as held-out target exactly once |

Total: all 50 US cities evaluated out-of-sample.

---

## Four Experimental Conditions & Calibration

All conditions share:
1. The **same frozen model** $\theta^*$
2. The **same calibration operator**: Support-Conditioned KL Projection onto the distance-bin constraint set

### Calibration Formulation (Support-Conditioned KL Projection)
$$\hat{T}^{\text{cal}} = \arg\min_{T}\, D_{KL}(T \,\|\, \hat{T}^{ZS}) \quad \text{s.t.} \quad B(T)[k] = Y_D^{\Omega_c}[k]\ \forall k \in \text{active bins}$$

Where the target distribution is conditioned on the active spatial support of $\Omega_c$ (for cities with diameter $< 100$ km):
$$Y_D^{\Omega_c}[k] = \frac{Y_D[k] \cdot \mathbf{1}(k \in \text{active})}{\sum_{l} Y_D[l] \cdot \mathbf{1}(l \in \text{active})}$$

Closed-form mass-preserving multiplicative solution:
$$\hat{N} = \sum_{(i,j)\in\Omega_c} \hat{T}^{ZS}_{ij}, \qquad B_k^{\text{target}} = Y_D^{\Omega_c}[k] \cdot \hat{N}, \qquad \hat{B}_k = \sum_{(i,j)\in\Omega_c, b(i,j)=k} \hat{T}^{ZS}_{ij}$$
$$s_k = \frac{B_k^{\text{target}} + \epsilon}{\hat{B}_k + \epsilon}, \qquad \boxed{\hat{T}^{\text{cal}}_{ij} = s_{b(i,j)} \cdot \hat{T}^{ZS}_{ij}}$$

**Strict Invariants**:
1. Exact total flow mass preservation: $\sum \hat{T}^{\text{cal}} \equiv \sum \hat{T}^{ZS}$.
2. Exact bin proportion matching: $\frac{\sum_{b(i,j)=k} \hat{T}^{\text{cal}}_{ij}}{\sum \hat{T}^{\text{cal}}_{ij}} \equiv Y_D^{\Omega_c}[k]$.

| Condition | $Y_D$ source | Purpose |
|---|---|---|
| **$M_0$** | None | Baseline $R^{ZS}_c$ |
| **$M_1^{\text{oracle}}$** | $Y_D^{\text{oracle}}$ computed from GT over $\Omega_c$ | Theoretical ceiling of 4-bin constraint |
| **$M_1^{\text{real}}$** | $Y_D^{\text{real}}$ extracted from Meta mobility data (FIPS-mapped, snapshot-averaged) | Real-world information gain |
| **$M_q$** | $\tilde{Y}_D^{(m)}$ from Multinomial trip sampling ($S=20$ seeds per $m$ level) | Reference curve for $m^*$ and $q^*$ |

---

## Evaluation Metrics (Evaluated on $\Omega_c$)

| Metric | Formula | Role |
|---|---|---|
| **CPC** | $\frac{2\sum \min(T_{ij}, \hat{T}_{ij})}{\sum T_{ij} + \sum \hat{T}_{ij}}$ | **Primary** |
| **RMSE-log1p** | $\sqrt{\frac{1}{|\Omega_c|}\sum_{(i,j)\in\Omega_c}[\log(1+T_{ij}) - \log(1+\hat{T}_{ij})]^2}$ | Secondary |
| **Pearson $r$** | $\text{Corr}(T_{ij}, \hat{T}_{ij})$ over $\Omega_c$ | Secondary |

---

## Research Questions & Estimands

### RQ1: Marginal Information Value ($\Delta R$)
$$\boxed{\Delta R_c^{\text{real}} = R_c(M_1^{\text{real}}) - R_c(M_0)}$$
$$\Delta R_c^{\text{oracle}} = R_c(M_1^{\text{oracle}}) - R_c(M_0)$$
$$\Delta R_c^{\text{realization}} = R_c(M_1^{\text{oracle}}) - R_c(M_1^{\text{real}})$$

Statistical tests across 50 cities: Mean $\pm$ std, median, IQR, $\Pr(\Delta R > 0)$, and paired Wilcoxon signed-rank test.

### RQ2: Observation-Equivalence ($m^*$ and $q^*$)
$$\boxed{m_c^{*,\text{real}} = \text{Isotonic\_Invert}(R_c(m), R_c(M_1^{\text{real}}))}, \qquad \boxed{q_c^{*,\text{real}} = \frac{m_c^{*,\text{real}}}{T_c^{\text{total}}}}$$
- $S=20$ random Multinomial sampling seeds per $m \in \{100, 500, 1\text{k}, 5\text{k}, 10\text{k}, 50\text{k}, 100\text{k}, \infty\}$.
- Isotonic regression ensures strictly non-decreasing empirical curve $R_c(m)$.
- Reports both $m^*$ (absolute trips required) and $q^*$ (fraction of total trips).

---

## Decision Table (Final & Verified)

| Component | Final Locked Decision |
|---|---|
| **Zero-shot role** | Baseline instrument to measure marginal value $\Delta R$; not proposed as novel OD model |
| **Urban GNN graph** | **Radius graph ($r=5.0\text{ km}$)** with 1-NN fallback; selected via empirical topology pilot |
| **OD decoder** | Single head producing $\mu_{NB, ij}$; conditional expectation $\hat{T} = E[T \mid T \ge 1]$ at inference |
| **Likelihood** | Zero-Truncated Negative Binomial (ZTNB) on positive pairs in $\Omega_c$ |
| **Evaluation support** | Candidate pairs $\Omega_c$ (pairs outside $\Omega_c$ are unknown, not zero-filled) |
| **Scaler isolation** | Fitted strictly on $X_{\text{train}}$ of source cities; target city strictly transformed |
| **Parameter freeze** | $\theta^*$ completely frozen before target city inference |
| **Calibration** | Support-conditioned mass-preserving KL projection; closed-form bin multiplicative scaling |
| **Meta Y_D extraction** | Official FIPS $\rightarrow$ GADM county mapping, snapshot-level 4-bin aggregation across all temporal files |
| **$M_q$ sampling** | Multinomial trip sampling with $S=20$ seeds per grid point |
| **$m^*$ & $q^*$ inversion** | Isotonic regression over empirical curve $R_c(m)$ |
