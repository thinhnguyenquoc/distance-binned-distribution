### Table 2: Ablation Study — Moving-Bin vs Legacy 4-Bin Calibration Across Evaluation Domains

| Framework / Condition | Calibration Target | Evaluated Domain | Interzonal CPC ($\Omega_c^+$) | Full CPC ($\Omega_c$) | Interzonal $\Delta R$ | $P(\Delta R_{\text{inter}} > 0)$ |
|---|---|---|---|---|---|---|
| **$M_0$ (Zero-Shot)** | None | All | 0.3613 +- 0.0725 | 0.3088 +- 0.0661 | --- | --- |
| **$M_1^{\text{real},+}$ (Moving-Bin)** | Bins 1,2,3 ($D>0$) | $\Omega_c^+$ | **0.3839 +- 0.0708** | 0.3283 +- 0.0665 | **+0.0225 +- 0.0165** | **90.0%** |
| **$M_1^{\text{real, 4bin}}$ (Legacy 4-Bin)** | Bins 0,1,2,3 ($D\ge 0$) | $\Omega_c$ | 0.3141 +- 0.0826 | 0.3654 +- 0.0898 | -0.0472 +- 0.0548 | 10.0% |
| **$M_1^{\text{oracle},+}$ (Oracle Reference)** | Oracle Bins 1,2,3 | $\Omega_c^+$ | 0.3885 +- 0.0725 | 0.3321 +- 0.0674 | +0.0272 +- 0.0135 | 100.0% |