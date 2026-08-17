### Table 5: RQ2 Observation Equivalence ($m^*, q^*$) Inversion & Censoring Analysis

> **Interval-Censoring Finding**: The observation-equivalence analysis is strongly interval-censored: 66% of cities require no more than the minimum grid size, whereas 22% are not resolved before the oracle-reference endpoint.
> 
> **Observation Equivalence Ratio**: $q^* = m^* / T_{\text{inter}}$, where $T_{\text{inter}} = \sum_{\Omega_c^+} T_{ij}^{GT}$ is candidate interzonal trip volume.

| Inversion Regime / Status | Count ($n/50$) | Percentage | Mean $m^*$ (trips) | Median $m^*$ (trips) | Median $q^*$ ($m^* / T_{\text{inter}}$) | Interpretation |
|---|---|---|---|---|---|---|
| **Interior Crossing (`interpolated`)** | 6/50 | 12.0% | 587.4 | 415.0 | 0.000048 (0.0048%) | Exact crossing within sampled grid [100, 100k] |
| **Left-Censored (`below_min_grid`)** | 33/50 | 66.0% | <= 100.0 | <= 100.0 | <= 100 / T_inter | Real Meta matched by <= 100 random trips |
| **Right-Censored (`at_oracle_reference`)** | 11/50 | 22.0% | >= 100k | T_inter | Unresolved / >= 1.0 | Real Meta unresolved before finite oracle asymptote |

#### Interior Solution Details ($n=6$):
| City | Interzonal Trips ($T_{\text{inter}}$) | $\Delta R^{\text{real},+}$ | Equivalent Trips ($m^*$) | Equivalent Ratio ($q^* = m^* / T_{\text{inter}}$) |
|---|---|---|---|---|
| **Chicago** | 38,990,968.0 | +0.0379 | 602.5 | **0.000015** (0.0015%) |
| **Tulsa** | 5,914,210.0 | +0.0246 | 466.9 | **0.000079** (0.0079%) |
| **Dallas** | 26,048,712.0 | +0.1229 | 134.7 | **0.000005** (0.0005%) |
| **Philadelphia** | 22,981,410.0 | +0.0409 | 1,720.3 | **0.000075** (0.0075%) |
| **Tucson** | 7,042,119.0 | +0.0301 | 363.1 | **0.000052** (0.0052%) |
| **Wichita** | 5,381,867.0 | +0.0050 | 236.7 | **0.000044** (0.0044%) |