# Implementation Plan (Final — Moving-Bin Framework Locked)
## Quantifying the Marginal Value of Distance-Binned Mobility Information

---

## Core Framing & Scope

> **This study quantifies how much additional OD-relevant information a coarse target-city mobility observation contributes beyond zero-shot inference, evaluated over the candidate interzonal OD support $\Omega_c^+$ available for each held-out city.**

- **Candidate Interzonal Support**: $\Omega_c^+ = \{(i,j) \in \Omega_c : i \neq j, D_{ij} > 0\}$.
- **Intrazonal Handling**: Intrazonal predictions $\hat{T}_{ii}^{ZS}$ are preserved intact ($\hat{T}_{ii}^{\text{cal}} \equiv \hat{T}_{ii}^{ZS}$).
- **Semantic Consistency**: Meta mobility data Bin 0 ("staying put at home / zero displacement") is excluded from calibration because it represents immobility rather than intrazonal travel. Calibration is strictly applied across actual displacement categories $\{1, 2, 3\}$:
  - Bin 1: $(0, 10)\text{ km}$
  - Bin 2: $[10, 100)\text{ km}$
  - Bin 3: $100+\text{ km}$
- **Legacy 4-Bin Ablation**: Evaluated to demonstrate the empirical penalty of conflating stay-at-home immobility with intrazonal travel.

---

## Architecture: Physics-Informed GNN with Pairwise Decoder

$$\boxed{\text{Urban GNN} \rightarrow h_i \rightarrow \text{Pairwise OD Decoder} \rightarrow \mu_{NB, ij} \rightarrow \hat{T}^{ZS}_{ij} = E[T_{ij} \mid T_{ij} \ge 1]}$$

### Information Regime
Both zero-shot baseline $\hat{T}^{(0)}$ and calibrated model $\hat{T}^{(YD)}$ share the exact same underlying representation and frozen parameters:
$$\hat{T}^{(0)} = f_{\theta^*}\left(X_{\text{urban}}^{26}, G_{\text{spatial}}, D_{ij}, P_i, P_j\right), \qquad \hat{T}^{(YD)} = \operatorname{Adjust}\left(\hat{T}^{(0)}, Y_D\right)$$

### Three Distinct Distance Channels
Geographic distance $D_{ij}$ enters the system through three dedicated functional pathways:
$$D_{ij} \longrightarrow \begin{cases} G^{\text{urban}},\ \text{edge\_dist} & \text{Urban GNN (Local spatial graph \& message passing)} \\ \log(1 + D_{ij}) & \text{Pairwise Decoder (Direct OD pair interaction)} \\ -\alpha \log D_{ij} & \text{Gravity Prior (Global physics decay prior)} \end{cases}$$
*Note on coordinates*: Coordinates (`lon_lat`) are not direct node features, but they determine the spatial graph ($G^{\text{urban}}$) and edge-distance attributes used by the Urban GNN.

### Sub-module 1: Node Encoder (Urban GNN)
$$h_i = \text{GNN}_\theta(X,\ G^{\text{urban}})$$
- **Node features** $X_i$: 26 features spanning census (demographics, commute/vehicle proxies), POI, road densities.
- **Graph** $G^{\text{urban}}$: Radius graph ($r = 5.0\text{ km}$ with 1-NN fallback) built strictly from spatial coordinates.
- **Architecture**: 2-layer Message Passing GNN with edge distance modulation.
- **StandardScaler**: fitted strictly on source training cities' node features ($X_{\text{train}}$); target city strictly transformed.

> **Graph Radius ($r=5.0\text{ km}$) & GNN Depth ($L=2$) Methodological Framing**:
> - **Radius ($r=5.0\text{ km}$)**: *"The 5-km radius was prespecified as an engineering choice and applied consistently across all folds. Its sensitivity is assessed separately; it is not claimed to be optimal."*
> - **Depth ($L=2$ Layers / 2-Hop Receptive Field)**: *"We use two message-passing layers as a prespecified engineering choice to capture second-order spatial context while limiting model depth and over-smoothing. Depth sensitivity is evaluated separately."*
> - **Methodological Policy**: Any depth/radius sensitivity tests on validation serve strictly as secondary robustness checks and do not alter the locked primary confirmatory E1 protocol. No optimality claims are made.

### Sub-module 2: Gravity Prior
$$\log T^{\text{grav}}_{ij} = G + \log P_i + \log P_j - \alpha \log D_{ij}$$
Classical 2-parameter global gravity prior.

> **Population Routing & Robustness Ablation**:
> `total_population` is present in the 26 node features and directly in the gravity prior ($\log P_i + \log P_j$). This intentional feature reuse/multi-channel prior is subject to a secondary robustness ablation (A: GNN only; B: Gravity prior only; C: Both [default]), without altering the locked primary E1 protocol.

### Sub-module 3: Pairwise OD Decoder (Single Base Magnitude Head)
Edge representation: $e_{ij} = [h_i,\ h_j,\ \log(1 + D_{ij}),\ \log T^{\text{grav}}_{ij}]$.
Single prediction head producing base Negative Binomial mean:
$$\mu_{NB, ij} = \text{softplus}(f_\mu(e_{ij})) > 0$$

- **Training**: minimizes exact Zero-Truncated Negative Binomial negative log-likelihood $\mathcal{L}_{\text{train}}$ on positive candidate observations $\Omega_c$.
- **Inference**: evaluates exact conditional expectation on $\Omega_c$:
$$\boxed{\hat{T}^{ZS}_{ij} = E[T_{ij} \mid T_{ij} \ge 1] = \frac{\mu_{NB, ij}}{1 - P_{NB}(0; \mu_{NB, ij}, \phi)} = \frac{\mu_{NB, ij}}{1 - \left(\frac{\phi}{\mu_{NB, ij} + \phi}\right)^\phi}}$$

> **Zero-Shot Definition**: Defined strictly as *"Zero-shot without target-city OD flows or target-city distance-binned distribution"* (recognizing that census features inherently contain mobility proxies such as commute and vehicle ownership).

---

## Moving-Bin Soft Calibration Formulation (KL Projection on $\Omega_c^+$)

### Moving-Bin Target Distributions:
$$Y_{c, k}^{\text{Meta}, +} = \frac{Y_{c, k}^{\text{Meta}}}{\sum_{\ell=1}^3 Y_{c, \ell}^{\text{Meta}}}, \qquad Y_{c, k}^{\text{oracle}, +} = \frac{\sum_{(i,j)\in\Omega_{c,k}^+} T_{ij}^{GT}}{\sum_{(i,j)\in\Omega_c^+} T_{ij}^{GT}} \quad (k \in \{1, 2, 3\})$$

### Support-Conditioning & Soft Multipliers ($0 \le q \le 1$):
$$\hat{N}^+ = \sum_{(i,j)\in\Omega_c^+} \hat{T}_{ij}^{ZS}, \qquad \hat{B}_k^+ = \sum_{(i,j)\in\Omega_{c,k}^+} \hat{T}_{ij}^{ZS}, \qquad \hat{Y}_k^{ZS, +} = \frac{\hat{B}_k^+}{\hat{N}^+}$$
$$w_k(q) = \left( \frac{p_k^{\text{cond}, +}}{\hat{Y}_k^{ZS, +}} \right)^q, \qquad s_k = \frac{w_k(q)}{\sum_{\ell \text{ active}} \hat{Y}_\ell^{ZS, +} w_\ell(q)}$$
$$\boxed{\hat{T}_{ij}^{\text{cal}} = s_{b(i,j)} \cdot \hat{T}_{ij}^{ZS} \quad \text{for } (i,j) \in \Omega_c^+}, \qquad \boxed{\hat{T}_{ii}^{\text{cal}} = \hat{T}_{ii}^{ZS} \quad \text{for intrazonal}}$$

**Strict Invariants**:
1. Exact interzonal mass preservation: $\sum_{\Omega^+} \hat{T}^{\text{cal}} \equiv \sum_{\Omega^+} \hat{T}^{ZS}$.
2. Intrazonal identity: $\hat{T}_{ii}^{\text{cal}} \equiv \hat{T}_{ii}^{ZS}$.
3. At $q=1.0$: implied moving-bin proportions match $p_k^{\text{cond}, +}$ within $10^{-5}$.
4. At $q=0.0$: $\hat{T}^{\text{cal}} \equiv \hat{T}^{ZS}$ (pure zero-shot identity).

---

## Experimental Conditions & Evaluation

| Condition | Domain | $Y_D$ Input | Role |
|---|---|---|---|
| **$M_0$** | $\Omega_c^+$ & $\Omega_c$ | None | Zero-shot baseline |
| **$M_1^{\text{real}, +}$** | $\Omega_c^+$ | $Y_D^{\text{Meta}, +}$ (Bins 1,2,3, $q=1.0$) | **Primary Real Intervention** |
| **$M_1^{\text{oracle}, +}$** | $\Omega_c^+$ | $Y_D^{\text{oracle}, +}$ (Bins 1,2,3, $q=1.0$) | **Oracle-Bin Reference** |
| **$M_1^{\text{real, 4bin}}$** | $\Omega_c$ | Raw 4-bin Meta (with Bin 0) | **Ablation (Semantic Mismatch Penalty)** |
| **$M_q^{\text{real}, +}$** | $\Omega_c^+$ | $Y_D^{\text{Meta}, +}$ ($q \in [0, 1]$) | Soft Calibration Response Curve |
| **$M_m^+$** | $\Omega_c^+$ | $\tilde{Y}_D^{(m), +}$ ($S=20$ seeds per $m$) | Multinomial Trip Sampling Reference Curve |

### Primary Evaluation Metric:
- **Interzonal CPC ($\text{CPC}_{\text{inter}}$)** evaluated on $\Omega_c^+$.
- **Distributional Overlap**: $\text{Overlap}(p, q) = \sum_k \min(p_k, q_k) = 1 - \frac{1}{2}\|p - q\|_1$.
- **Secondary Metrics**: $\text{CPC}_{\text{inter, norm}}$ ($1 - \text{TVD}$), $\text{CPC}_{\text{full}}$, $\text{RMSE}_{\text{inter}}$, $\text{Pearson}_{\text{inter}}$.
