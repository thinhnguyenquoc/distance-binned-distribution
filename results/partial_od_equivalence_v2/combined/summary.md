# Table: Master Partial-OD Information Equivalence Summary (v2)

> **Evaluation Scope**: Assesses the operational reconstruction value of target-city distance distribution $Y_D$ relative to observing $p\%$ of positive interzonal OD pairs ($K=8, q=1.0$, seeds $s \in \{1, 10, 100\}$) evaluated strictly on unseen pairs ($N=50$ held-out test cities across 5 folds).

• **Positive Mean Crossing Point:** `5.00%` of positive interzonal OD pairs  
• **Statistically Supported Benefit Threshold ($p^*_\text{benefit}$):** `20.00%` of positive interzonal OD pairs ($p_\text{Holm} < 0.05$)  
• **Operational Equivalence Crossing:** Full target-city $Y_D$ was not matched within the prespecified partial-OD range up to 90% of the positive interzonal OD support.  

| Revealed OD Pairs ($p$) | Mean Revealed Trip Mass | Mean TV to Full $Y_D$ | $M_0$ CPC (Unseen) | Full-$Y_D$ Gain | Partial-OD Gain | Difference vs Full $Y_D$ ($D(p)$) | 95% CI Difference | Partial Benefit Holm $p$ | Cities Partial $> M_0$ | Cities Partial $\ge$ Full $Y_D$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.00%** | 0.00% | nan% | 0.7128 | +0.00354 | **+0.00000** | **-0.00354** | [-0.00450, -0.00260] | — | 0/50 | 5/50 |
| **0.10%** | 0.10% | 25.80% | 0.7128 | +0.00354 | **-0.10675** | **-0.11029** | [-0.12845, -0.09268] | 1.0000e+00 | 2/50 | 0/50 |
| **0.25%** | 0.25% | 16.72% | 0.7128 | +0.00354 | **-0.05012** | **-0.05366** | [-0.06359, -0.04423] | 1.0000e+00 | 2/50 | 0/50 |
| **0.50%** | 0.50% | 11.98% | 0.7128 | +0.00354 | **-0.02616** | **-0.02970** | [-0.03549, -0.02429] | 1.0000e+00 | 4/50 | 0/50 |
| **1.00%** | 1.00% | 8.52% | 0.7128 | +0.00354 | **-0.01226** | **-0.01580** | [-0.01894, -0.01286] | 1.0000e+00 | 7/50 | 0/50 |
| **2.00%** | 2.00% | 6.00% | 0.7128 | +0.00354 | **-0.00461** | **-0.00815** | [-0.00979, -0.00662] | 1.0000e+00 | 8/50 | 0/50 |
| **5.00%** | 5.00% | 3.77% | 0.7128 | +0.00354 | **+0.00017** | **-0.00337** | [-0.00407, -0.00271] | 1.0000e+00 | 23/50 | 0/50 |
| **10.00%** | 10.00% | 2.60% | 0.7128 | +0.00354 | **+0.00187** | **-0.00167** | [-0.00202, -0.00134] | 6.4954e-02 | 29/50 | 0/50 |
| **20.00%** | 20.01% | 1.74% | 0.7128 | +0.00354 | **+0.00272** | **-0.00082** | [-0.00099, -0.00065] | 1.5205e-05 | 38/50 | 0/50 |
| **40.00%** | 40.00% | 1.07% | 0.7128 | +0.00354 | **+0.00315** | **-0.00039** | [-0.00047, -0.00031] | 2.7993e-07 | 44/50 | 0/50 |
| **50.00%** | 50.00% | 0.87% | 0.7128 | +0.00353 | **+0.00324** | **-0.00030** | [-0.00036, -0.00024] | 1.3189e-07 | 44/50 | 0/50 |
| **60.00%** | 60.00% | 0.71% | 0.7128 | +0.00354 | **+0.00330** | **-0.00024** | [-0.00029, -0.00019] | 8.1924e-08 | 44/50 | 0/50 |
| **70.00%** | 70.00% | 0.57% | 0.7128 | +0.00354 | **+0.00335** | **-0.00019** | [-0.00023, -0.00016] | 5.8637e-08 | 44/50 | 0/50 |
| **80.00%** | 80.00% | 0.43% | 0.7128 | +0.00354 | **+0.00337** | **-0.00016** | [-0.00019, -0.00013] | 5.8308e-08 | 44/50 | 0/50 |
| **90.00%** | 90.00% | 0.29% | 0.7129 | +0.00353 | **+0.00340** | **-0.00014** | [-0.00016, -0.00011] | 4.8420e-08 | 45/50 | 0/50 |

---

### Prescribed Scientific Interpretation
Under uniform random pair sampling, the mean revealed trip-mass fraction closely tracked the revealed pair fraction. Under the tested operator, directly observing up to 90% of the positive interzonal OD support did not fully match the mean reconstruction gain provided by the full target-city $Y_D$.
