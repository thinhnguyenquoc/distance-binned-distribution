# Table: Master Direct-OD Information Equivalence Summary (v1)

> **Evaluation Scope**: Evaluates the operational reconstruction value of directly observed positive interzonal OD pairs via low-capacity Origin-Destination Fixed-Effect residual adaptation (OD-FE), relative to the full target-city distance distribution $Y_D$ ($K=8, q=1.0$, seeds $s \in \{1, 10, 100\}$), evaluated strictly on unseen pairs ($N=50$ held-out test cities across 5 folds).

• **Validation-Selected Lambdas:** Fold 1: `10.0`, Fold 2: `10.0`, Fold 3: `10.0`, Fold 4: `10.0`, Fold 5: `10.0`  
• **Positive Mean Crossing Point ($p_\text{mean+}$):** `0.10%` of positive interzonal OD pairs  
• **Statistically Supported Benefit Threshold ($p^*_\text{DirectBenefit}$):** `0.10%` of positive interzonal OD pairs ($p_\text{Holm} < 0.05$)  
• **Operational Equivalence Crossing ($p_\text{eq,interp}$):** `0.20%` of positive interzonal OD pairs  

| Revealed OD Pairs ($p$) | Both Coverage | $M_0$ CPC (Unseen) | Full-$Y_D$ Gain | Direct-OD Gain | Difference vs Full $Y_D$ ($D(p)$) | 95% CI Difference | Direct Benefit Holm $p$ | Cities Direct $> M_0$ | Cities Direct $\ge$ Full $Y_D$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.00%** | 0.00% | 0.7128 | +0.00354 | **+0.00000** | **-0.00354** | [-0.00450, -0.00260] | — | 0/50 | 5/50 |
| **0.10%** | 4.75% | 0.7128 | +0.00354 | **+0.00180** | **-0.00174** | [-0.00279, -0.00068] | 1.2434e-14 | 50/50 | 22/50 |
| **0.25%** | 15.84% | 0.7128 | +0.00354 | **+0.00448** | **+0.00094** | [-0.00051, +0.00259] | 1.2434e-14 | 50/50 | 29/50 |
| **0.50%** | 34.02% | 0.7128 | +0.00354 | **+0.00859** | **+0.00505** | [+0.00289, +0.00765] | 1.2434e-14 | 50/50 | 36/50 |
| **1.00%** | 61.26% | 0.7128 | +0.00354 | **+0.01549** | **+0.01195** | [+0.00883, +0.01560] | 1.2434e-14 | 50/50 | 46/50 |
| **2.00%** | 86.85% | 0.7128 | +0.00354 | **+0.02584** | **+0.02230** | [+0.01827, +0.02673] | 1.2434e-14 | 50/50 | 49/50 |
| **5.00%** | 99.32% | 0.7128 | +0.00354 | **+0.04363** | **+0.04009** | [+0.03507, +0.04542] | 1.2434e-14 | 50/50 | 50/50 |
| **10.00%** | 99.99% | 0.7128 | +0.00354 | **+0.05581** | **+0.05227** | [+0.04684, +0.05812] | 1.2434e-14 | 50/50 | 50/50 |
| **20.00%** | 100.00% | 0.7128 | +0.00354 | **+0.06297** | **+0.05943** | [+0.05379, +0.06551] | 1.2434e-14 | 50/50 | 50/50 |
| **40.00%** | 100.00% | 0.7128 | +0.00353 | **+0.06527** | **+0.06174** | [+0.05602, +0.06792] | 1.2434e-14 | 50/50 | 50/50 |
| **50.00%** | 100.00% | 0.7128 | +0.00353 | **+0.06539** | **+0.06186** | [+0.05614, +0.06804] | 1.2434e-14 | 50/50 | 50/50 |
| **60.00%** | 100.00% | 0.7128 | +0.00353 | **+0.06538** | **+0.06185** | [+0.05614, +0.06801] | 1.2434e-14 | 50/50 | 50/50 |
| **70.00%** | 100.00% | 0.7128 | +0.00354 | **+0.06533** | **+0.06179** | [+0.05608, +0.06795] | 1.2434e-14 | 50/50 | 50/50 |
| **80.00%** | 100.00% | 0.7128 | +0.00354 | **+0.06522** | **+0.06169** | [+0.05598, +0.06783] | 1.2434e-14 | 50/50 | 50/50 |
| **90.00%** | 100.00% | 0.7129 | +0.00354 | **+0.06515** | **+0.06161** | [+0.05595, +0.06772] | 1.2434e-14 | 50/50 | 50/50 |

---

### Prescribed Scientific Interpretation
Under the prespecified OD fixed-effect residual adapter, directly observing approximately **0.20%** of the positive interzonal OD support produced a mean reconstruction gain on the remaining unseen pairs comparable to that obtained from the full target-city distance-binned distribution.
