# Table S1: Spatial Resolution Analysis (County-Level vs. City-Level Calibration)

> **Research Question**: Does conditioning the aggregated distance distribution $Y_D$ on origin counties ($M_{\text{county}}$) improve zero-shot flow prediction over city-wide macro distributions ($M_{\text{city}}$)?
> **Dataset**: 50 US Metropolitan Areas (39 Single-County, 11 Multi-County) under 5-Fold Stratified CV.
> **Calibration Protocol**: $K_{\text{move}}=8$ quantile bins, $q=1.0$, within-tolerance distribution matching.

---

## S1-A: Overall Comparative Performance ($n=50$ Cities)

| Condition / Model | Mean Interzonal CPC | Mean Gain vs $M_0$ (Δ) | 95% Bootstrap CI | Specificity Gap vs Wrong-Donor | City-Level Specificity Win Rate |
|---|:---:|:---:|:---:|:---:|:---:|
| **Zero-Shot Baseline ($M_0$)** | 0.7128 | — | — | — | — |
| **+ City-Level Target $Y_D$ ($M_{\text{city}}$)** | 0.7163 | +0.0035 | [+0.0026, +0.0045] | +0.0413 | 50/50 |
| **+ County-Level Target $Y_D$ ($M_{\text{county}}$)** | **0.7165** | **+0.0037** | **[+0.0027, +0.0047]** | +0.0414 | 50/50 |
| **City-Level Placebo ($M_{\text{wrong}}$ 9-Donor Avg)** | 0.6751 | -0.0377 | [-0.0436, -0.0327] | — | 0/50 |

---

## S1-B: Multi-County Metropolitan Focus ($n=11$ Heterogeneous Cities)

In multi-county metropolitan areas, distinct origin counties exhibit heterogeneous localized trip distributions.

| City | Origin Counties | Zero-Shot $M_0$ | City-Level $M_{\text{city}}$ | County-Level $M_{\text{county}}$ | Resolution Gain ($\Delta_{\text{res}}$) | City-Level Placebo $M_{\text{wrong}}$ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Kansas_City** | 3 counties | 0.7211 | 0.7269 | **0.7296** | **+0.0027** | 0.7029 |
| **New_York** | 7 counties | 0.5245 | 0.5258 | **0.5279** | **+0.0021** | 0.5086 |
| **Dallas** | 3 counties | 0.6853 | 0.6958 | **0.6969** | **+0.0011** | 0.6670 |
| **Denver** | 3 counties | 0.7156 | 0.7157 | **0.7161** | **+0.0003** | 0.6941 |
| **Omaha** | 2 counties | 0.7470 | 0.7526 | **0.7528** | **+0.0002** | 0.7136 |
| **Tulsa** | 2 counties | 0.7797 | 0.7816 | **0.7818** | **+0.0002** | 0.7223 |
| **Detroit** | 2 counties | 0.6845 | 0.6851 | **0.6852** | **+0.0002** | 0.6505 |
| **Chicago** | 2 counties | 0.6724 | 0.6743 | **0.6744** | **+0.0000** | 0.6435 |
| **Boston** | 3 counties | 0.6872 | 0.6876 | **0.6876** | **+0.0000** | 0.6036 |
| **Milwaukee** | 2 counties | 0.7413 | 0.7429 | **0.7429** | **-0.0000** | 0.7042 |
| **Atlanta** | 2 counties | 0.7108 | 0.7197 | **0.7196** | **-0.0000** | 0.6800 |

**Multi-County Average ($n=11$)**:
- Mean Zero-Shot $M_0$: 0.6972
- Mean City-Level $M_{\text{city}}$: 0.7007 (Δ = +0.0035)
- Mean County-Level $M_{\text{county}}$: **0.7013** (Δ = **+0.0041**)
- **Mean Spatial Resolution Gain ($\Delta_{\text{res}}$)**: **+0.00063** (Max: **+0.0027**)
- **Resolution Improvement Rate**: **9/11** ($81.8\%$)

---

## S1-C: Single-County Sanity Invariance ($n=39$ Single-County Cities)

For single-county cities, all tracts belong to the same origin county, meaning $M_{\text{county}} \equiv M_{\text{city}}$ by definition.
- **Observed Mean $\Delta_{\text{resolution}}$**: 0.000000
- **Exact Mathematical Invariance**: ✓ VERIFIED
