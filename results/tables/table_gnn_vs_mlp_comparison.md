# Neural Backbone Comparison: Gravity-Informed Urban GNN vs Pairwise Spatial MLP

> **Evaluation Goal**: Assesses whether distance-binned aggregate distribution calibration ($Y_D^{\text{target}}$) provides consistent reconstruction gain across distinct neural architectures (Spatial Graph Convolution vs Local Feature MLP).

## Part A: Confirmatory Evaluation Set (Folds 2–5, n=40 Cities)

| Backbone Architecture | Zero-Shot $M_0$ CPC | Calibrated $M_1$ CPC | Marginal Gain $\Delta\text{CPC}$ | 95% Fold-Stratified Bootstrap CI | Improved Cities | Wilcoxon $p$ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Gravity-Informed Urban GNN** | 0.7167 | **0.7208** | **+0.0041 +- 0.0043** | [+0.0030, +0.0053] | 38/40 (95.0%) | p = 4.53e-09 |
| **Pairwise Spatial MLP** | 0.7097 | **0.7138** | **+0.0041 +- 0.0044** | [+0.0030, +0.0054] | 38/40 (95.0%) | p = 2.79e-10 |
| **Architecture Advantage ($\Gamma = \Delta_\text{GNN} - \Delta_\text{MLP}$)** | — | — | **-0.0000 +- 0.0029** | [-0.0007, +0.0006] | — | p = 8.68e-01 |

## Part B: Five-Fold Cross-City Evaluation Set (All 5 Folds, N=50 Cities)

| Backbone Architecture | Zero-Shot $M_0$ CPC | Calibrated $M_1$ CPC | Marginal Gain $\Delta\text{CPC}$ | 95% Fold-Stratified Bootstrap CI | Improved Cities | Wilcoxon $p$ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Gravity-Informed Urban GNN** | 0.7167 | **0.7208** | **+0.0041 +- 0.0043** | [+0.0030, +0.0053] | 38/40 (95.0%) | p = 4.53e-09 |
| **Pairwise Spatial MLP** | 0.7097 | **0.7138** | **+0.0041 +- 0.0044** | [+0.0030, +0.0054] | 38/40 (95.0%) | p = 2.79e-10 |
| **Architecture Advantage ($\Gamma = \Delta_\text{GNN} - \Delta_\text{MLP}$)** | — | — | **-0.0000 +- 0.0029** | [-0.0007, +0.0006] | — | p = 8.68e-01 |

