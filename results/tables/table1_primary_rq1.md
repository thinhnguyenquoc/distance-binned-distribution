### Table 1: Out-of-Fold Descriptive Statistics Across All 50 Cities

> **Summary**: Across 50 out-of-fold cities, Moving-Bin calibration increased mean interzonal CPC by 0.0265 (SD 0.0291), with positive improvements in 42 cities. This provides strong empirical support for a positive marginal contribution beyond zero-shot inference.

| Metric / Condition | Zero-Shot ($M_0$) | Real Moving-Bin ($M_1^{\text{real},+}$) | Oracle Reference ($M_1^{\text{oracle},+}$) | Realization Gap |
|---|---|---|---|---|
| **Interzonal CPC (Mean +- Sample SD)** | 0.3904 +- 0.0543 | **0.4169 +- 0.0587** | 0.4226 +- 0.0589 | +0.0057 +- 0.0094 |
| **Interzonal CPC (Median, IQR)** | 0.3966 (0.0812) | **0.4215 (0.1006)** | 0.4330 (0.0995) | +0.0015 |
| **Marginal Gain $\Delta R$ (Mean +- Sample SD)** | --- | **+0.0265 +- 0.0291** | +0.0322 +- 0.0251 | --- |
| **Improvement Rate $P(\Delta R > 0)$** | --- | **84.0%** (42/50) | 100.0% (50/50) | --- |
| **Wilcoxon Signed-Rank Test ($p_1$)** | --- | **p = 4.4852e-09** | --- | --- |
| **Distributional Overlap with Prior** | --- | 91.5% +- 5.1% | 100.0% | --- |