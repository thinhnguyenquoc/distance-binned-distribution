# Neural Backbone Comparison: Gravity-Informed Urban GNN vs Pairwise Spatial MLP

> **Evaluation Goal**: Assesses whether distance-binned aggregate distribution calibration ($Y_D^{\text{target}}$) provides consistent reconstruction gain across distinct neural architectures (Spatial Graph Convolution vs Local Feature MLP).

## Five-Fold Cross-City Evaluation Set (All 5 Folds, N=50 Cities)

| Backbone Architecture | Zero-Shot $M_0$ CPC | Calibrated $M_1$ CPC | Marginal Gain $\Delta\text{CPC}$ | 95% Fold-Stratified Bootstrap CI | Improved Cities | Wilcoxon $p$ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Gravity-Informed Urban GNN** | 0.7136 | **0.7172** | **+0.0036 +- 0.0041** | [+0.0027, +0.0045] | 47/50 (94.0%) | p = 2.40e-10 |
| **Pairwise Spatial MLP** | 0.7078 | **0.7113** | **+0.0035 +- 0.0042** | [+0.0026, +0.0045] | 47/50 (94.0%) | p = 6.44e-11 |
| **Architecture Advantage ($\Gamma = \Delta_\text{GNN} - \Delta_\text{MLP}$)** | — | — | **+0.0001 +- 0.0027** | [-0.0005, +0.0006] | — | p = 4.84e-01 |

