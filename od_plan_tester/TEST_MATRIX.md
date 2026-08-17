# Test Matrix: Moving-Bin Framework Acceptance Test Mapping

This matrix maps every locked architectural and scientific decision from the Interzonal Moving-Bin Framework to the corresponding automated tests in `od_plan_tester/tests/`.

| ID | Category | Moving-Bin Implementation Decision | Test Name | File | Pytest Markers |
|---|---|---|---|---|---|
| **T01** | Loss / Oracle | Base NB $P(T=0) = (\phi / (\mu + \phi))^\phi$ | `test_ztnb_log_prob_zero_formula` | `test_ztnb_oracle.py` | `reference` |
| **T02** | Loss / Oracle | ZTNB NLL strictly on positive counts ($T_{ij} \ge 1$) | `test_ztnb_nll_strictly_positive_support` | `test_ztnb_oracle.py` | `contract, scientific` |
| **T03** | Loss / Oracle | Conditional expectation $E[T \mid T \ge 1] = \mu / (1 - P(0)) > \mu$ | `test_ztnb_conditional_mean_strictly_greater` | `test_ztnb_oracle.py` | `reference` |
| **T04** | Loss / Oracle | Conditional expectation asymptotic convergence to $\mu$ as $\mu \to \infty$ | `test_ztnb_conditional_mean_asymptotics` | `test_ztnb_oracle.py` | `reference` |
| **T05** | Physics Prior | Gravity prior $\log T = G + \log P_i + \log P_j - \alpha \log D$ | `test_gravity_prior_formula_and_decay` | `test_ztnb_oracle.py` | `reference` |
| **T06** | Physics Prior | Trainable shared physics parameters $(G, \alpha)$ produce valid gradients | `test_gravity_prior_learnable_gradients` | `test_ztnb_oracle.py` | `contract` |
| **T07** | Graph Topology | Radius graph $r = 5.0$ km threshold | `test_radius_graph_5km_threshold` | `test_graph_topology.py` | `reference` |
| **T08** | Graph Topology | Isolated node fallback to 1-NN (no degree 0 nodes) | `test_radius_graph_isolated_fallback_1nn` | `test_graph_topology.py` | `reference` |
| **T09** | Graph Topology | Zero OD leakage: graph constructed strictly from spatial coordinates | `test_urban_graph_zero_od_leakage` | `test_graph_topology.py` | `scientific` |
| **T10** | Graph Topology | Graph symmetry and self-loop inclusion | `test_graph_symmetric_and_self_loops` | `test_graph_topology.py` | `contract` |
| **T11** | Cross-Validation| 5-Fold 40/10 stratified split by tract count | `test_stratified_5fold_split_structure` | `test_leakage_and_splits.py` | `reference` |
| **T12** | Cross-Validation| All 50 cities evaluated as test target exactly once | `test_stratified_5fold_exact_single_coverage` | `test_leakage_and_splits.py` | `scientific` |
| **T13** | Leakage / Scaler| StandardScaler fitted strictly on source training cities ($X_{\text{train}}$) | `test_scaler_train_isolation` | `test_leakage_and_splits.py` | `scientific` |
| **T14** | Provenance | Target city candidate pairs $\Omega_c$ have $T_{ij} \ge 1$ | `test_omega_c_provenance_strictly_positive` | `test_leakage_and_splits.py` | `scientific` |
| **T15** | Provenance | Missing pairs outside $\Omega_c$ are excluded, not zero-filled | `test_unobserved_pairs_excluded_not_zero_filled` | `test_leakage_and_splits.py` | `scientific` |
| **T16** | Calibration | Interzonal mass preservation within numerical tolerance on $\Omega_c^+$ | `test_moving_mass_preservation` | `test_kl_calibration.py` | `scientific` |
| **T17** | Calibration | Intrazonal identity: diagonal pairs ($i=j$, $D=0$, bin 0) preserved identically $\hat T_{ii}^{\text{cal}} \equiv \hat T_{ii}^{\text{ZS}}$ | `test_intrazonal_identity` | `test_kl_calibration.py` | `scientific` |
| **T18** | Calibration | Zero-shot identity at $q=0$: $\hat T^{\text{cal}}(q=0) \equiv \hat T^{\text{ZS}}$ | `test_q_zero_is_zero_shot_identity` | `test_kl_calibration.py` | `contract` |
| **T19** | Calibration | Exact moving-bin distribution matching at $q=1$ within tolerance $< 10^{-5}$ | `test_q_one_matches_target_distribution` | `test_kl_calibration.py` | `scientific` |
| **T20** | Calibration | Soft response: $w_k(q) = (p_k / \hat Y_k)^q$ monotonically bridges $q=0$ to $q=1$ | `test_q_monotonic_soft_response` | `test_kl_calibration.py` | `reference` |
| **T21** | Calibration | Inactive bin conditioning on moving bins $\{1, 2, 3\}$ when diameter $< 100$ km | `test_inactive_bin_conditioning` | `test_kl_calibration.py` | `reference` |
| **T22** | $Y_D$ Extraction| $Y_D^{\text{oracle},+}$ interzonal 3-bin distribution on $\Omega_c^+$ (bins $\{1, 2, 3\}$, excluding intrazonal) | `test_yd_moving_oracle_assignment` | `test_yd_and_sampling.py` | `reference` |
| **T23** | $Y_D$ Extraction| $Y_D^{\text{real},+}$ Meta moving-bin distribution (3 bins, sums to 1.0) | `test_yd_moving_real_meta_sum` | `test_yd_and_sampling.py` | `contract` |
| **T24** | Distribution | Distributional overlap (Overlap / $CPC_{\text{dist}} = 1 - 0.5 \lVert p - q \rVert_1$) bounded in $[0, 1]$ | `test_distributional_overlap_bounds` | `test_yd_and_sampling.py` | `reference` |
| **T25** | Sampling | Multinomial sampling produces valid probability distribution | `test_multinomial_sampling_stochastic_validity` | `test_yd_and_sampling.py` | `reference` |
| **T26** | Sampling | Asymptotic convergence of multinomial sampling to empirical distribution as $m \to \infty$ | `test_multinomial_sampling_asymptotic_convergence` | `test_yd_and_sampling.py` | `reference` |
| **T27** | Metrics | CPC bounds in $[0, 1]$ and symmetry $CPC(A, B) = CPC(B, A)$ | `test_cpc_bounds_and_symmetry` | `test_metrics.py` | `reference` |
| **T28** | Metrics | Normalized CPC identity: $CPC_{\text{norm}} = 1 - 0.5 \sum \lvert p_i - q_i \rvert$ | `test_cpc_norm_1_minus_tvd` | `test_metrics.py` | `reference` |
| **T29** | Metrics | RMSE-log1p is $0.0$ for identical predictions | `test_rmse_log1p_zero_on_identical` | `test_metrics.py` | `reference` |
| **T30** | Metrics | Pearson $r$ is $1.0$ for linear transformation and bounded in $[-1, 1]$ | `test_pearson_r_bounds` | `test_metrics.py` | `reference` |
| **T31** | Metrics | `evaluate_all` returns all locked metrics simultaneously | `test_evaluate_all_contract` | `test_metrics.py` | `contract` |
| **T32** | RQ2 Inversion | Isotonic regression enforces non-decreasing curve $R_c(m)$ | `test_isotonic_monotonicity_enforcement` | `test_isotonic_inversion.py` | `reference` |
| **T33** | RQ2 Inversion | Plateau handling: leftmost crossing picks minimal sample size $m^*$ | `test_isotonic_plateau_leftmost_crossing` | `test_isotonic_inversion.py` | `reference` |
| **T34** | RQ2 Inversion | Boundary case: target $\le R(m_{\min}) \implies m^* = m_{\min}$, `below_min_grid` | `test_isotonic_boundary_below_min_grid` | `test_isotonic_inversion.py` | `reference` |
| **T35** | RQ2 Inversion | Boundary case: target $\ge R(m_{\max}) \implies m^* = T_{\text{total}}$, `at_oracle_reference` | `test_isotonic_boundary_at_oracle_reference` | `test_isotonic_inversion.py` | `reference` |
| **T36** | RQ2 Inversion | Observation equivalence ratio $q^* = m^* / T^{\text{inter, total}}$ | `test_qstar_ratio_computation` | `test_isotonic_inversion.py` | `contract` |
| **T37** | Integrity | Parameter freeze: $\theta^*$ frozen (`requires_grad=False`) before target inference | `test_model_freezing_theta_star` | `test_experiment_contracts.py` | `scientific` |
| **T38** | Integrity | Shared support $\Omega_c$ strictly identical across $M_0, M_1^{\text{oracle},+}, M_1^{\text{real},+}, M_m^+$ | `test_shared_support_omega_c_across_conditions` | `test_experiment_contracts.py` | `scientific` |
| **T39** | RQ1 Estimands | Marginal value $\Delta R^{\text{real},+}, \Delta R^{\text{oracle},+}$ and realization gap definitions | `test_delta_r_and_realization_gap_formulas` | `test_experiment_contracts.py` | `reference` |
| **T40** | Provenance | Experiment manifest verifies seed, split, and zero target leakage | `test_experiment_manifest_reproducibility` | `test_experiment_contracts.py` | `scientific` |
| **T41** | Smoke Test | End-to-end moving-bin target city experiment pipeline run on test city | `test_run_target_city_experiments_smoke` | `test_experiment_contracts.py` | `contract` |
