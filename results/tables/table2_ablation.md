### Table 2: Ablation Study — Moving-Bin vs Legacy 4-Bin Calibration Across Evaluation Domains

| Framework / Condition | Calibration Target | Evaluated Domain | Interzonal CPC ($\Omega_c^+$) | Full CPC ($\Omega_c$) | Interzonal $\Delta R$ | $P(\Delta R_{\text{inter}} > 0)$ |
|---|---|---|---|---|---|---|
| **$M_0$ (Zero-Shot)** | None | All | 0.3675 +- 0.0725 | 0.3146 +- 0.0656 | --- | --- |
| **$M_1^{\text{real},+}$ (Moving-Bin)** | Bins 1,2,3 ($D>0$) | $\Omega_c^+$ | **0.3908 +- 0.0704** | 0.3347 +- 0.0655 | **+0.0233 +- 0.0167** | **90.0%** |
| **$M_1^{\text{real, 4bin}}$ (Legacy 4-Bin)** | Bins 0,1,2,3 ($D\ge 0$) | $\Omega_c$ | 0.3217 +- 0.0810 | 0.3733 +- 0.0889 | -0.0458 +- 0.0555 | 10.0% |
| **$M_1^{\text{oracle},+}$ (Oracle Reference)** | Oracle Bins 1,2,3 | $\Omega_c^+$ | 0.3957 +- 0.0723 | 0.3387 +- 0.0666 | +0.0282 +- 0.0136 | 100.0% |