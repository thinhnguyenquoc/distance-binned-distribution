### Table 0: Primary Confirmatory Hypothesis Test (Held-Out Fold 2–5, $n=40$ Cities)

> **Confirmatory Protocol**: Fold 1 ($n=10$) served as the development fold for calibration specification. Folds 2–5 ($n=40$) constitute the untouched confirmatory evaluation set.

| Estimand / Metric | Zero-Shot ($M_0$) | Real Moving-Bin ($M_1^{\text{real},+}$) | Oracle Reference ($M_1^{\text{oracle},+}$) | Marginal Gain ($\Delta R$) / Realization Gap |
|---|---|---|---|---|
| **Interzonal CPC (Mean +- Sample SD)** | 0.3954 +- 0.0486 | **0.4227 +- 0.0549** | 0.4286 +- 0.0545 | **+0.0272 +- 0.0316** (Gap: +0.0059) |
| **Interzonal CPC (Median, IQR)** | 0.4010 (0.0793) | **0.4275 (0.0820)** | 0.4336 (0.0792) | **+0.0260 (0.0304)** |
| **95% Bootstrap Confidence Interval** | --- | --- | --- | **[+0.0178, +0.0372]** |
| **Improvement Rate $P(\Delta R > 0)$** | --- | **82.5%** (33/40) | 100.0% (40/40) | --- |
| **Wilcoxon Signed-Rank Test** | --- | **$p_1 = 3.9401e-07$** (Two-sided: $p_2 = 7.8802e-07$) | --- | --- |

#### Per-Fold Stability Breakdown:
| Fold | Role | Cities ($n$) | Mean $M_0$ CPC | Mean $M_1^{\text{real},+}$ CPC | Mean $\Delta R^{\text{real},+}$ | Median $\Delta R$ | $P(\Delta R > 0)$ |
|---|---|---|---|---|---|---|---|
| **Fold 1** | Development | 10 | 0.3700 | 0.3936 | **+0.0236 +- 0.0169** | +0.0221 | 9/10 (90%) |
| **Fold 2** | Confirmatory | 10 | 0.3910 | 0.4207 | **+0.0296 +- 0.0258** | +0.0263 | 9/10 (90%) |
| **Fold 3** | Confirmatory | 10 | 0.4094 | 0.4272 | **+0.0178 +- 0.0204** | +0.0185 | 7/10 (70%) |
| **Fold 4** | Confirmatory | 10 | 0.4013 | 0.4311 | **+0.0298 +- 0.0379** | +0.0270 | 8/10 (80%) |
| **Fold 5** | Confirmatory | 10 | 0.3800 | 0.4117 | **+0.0317 +- 0.0408** | +0.0289 | 9/10 (90%) |