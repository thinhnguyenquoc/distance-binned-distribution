### Table 2: Ablation Trade-off — Moving-Bin vs Four-Bin Legacy Across Evaluation Domains

> **Methodological Context**: Moving-Bin calibration focuses probability updates on interzonal travel categories ($D>0$), preserving intrazonal diagonal flows. Four-bin calibration updates full-matrix flows including intrazonal / zero-distance mass ($D=0$). The results highlight an estimand trade-off rather than isolated causality.

| Framework / Condition | Calibration Target | Evaluated Domain | Interzonal CPC ($\Omega_c^+$) | Full-Matrix CPC ($\Omega_c$) | Interzonal $\Delta R$ | $P(\Delta R_{\text{inter}} > 0)$ |
|---|---|---|---|---|---|---|
| **$M_0$ (Zero-Shot)** | None | All | 0.3904 +- 0.0543 | 0.3315 +- 0.0508 | --- | --- |
| **$M_1^{\text{real},+}$ (Moving-Bin)** | Bins 1,2,3 ($D>0$) | $\Omega_c^+$ | **0.4169 +- 0.0587** | 0.3541 +- 0.0563 | **+0.0265 +- 0.0291** | **84.0%** |
| **$M_1^{\text{real, 4bin}}$ (Legacy 4-Bin)** | Bins 0,1,2,3 ($D\ge 0$) | $\Omega_c$ | 0.3444 +- 0.0744 | **0.4025 +- 0.0774** | -0.0460 +- 0.0582 | 16.0% |
| **$M_1^{\text{oracle},+}$ (Oracle Reference)** | Oracle Bins 1,2,3 | $\Omega_c^+$ | 0.4226 +- 0.0589 | 0.3589 +- 0.0565 | +0.0322 +- 0.0251 | 100.0% |