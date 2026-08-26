# GUIDELINE CHO CODING AGENT — DISTANCE-BINNED OD RECONSTRUCTION

You are modifying a frozen scientific experiment pipeline.

Your job is NOT merely to make the code run.
Your job is to preserve the experimental estimand, avoid target leakage,
preserve reproducibility, and make the implementation auditable.

LOCKED EXPERIMENT CONTRACT

1. Problem:
   Support-conditioned zero-shot OD intensity reconstruction.
   Missing OD pairs are UNKNOWN, not zero.
   Do not construct a full zero-filled OD matrix.

2. Target evaluation support:
   Positive observed interzonal support Omega_c^+:
       i != j
       geographic distance > 0
   All primary CPC claims are on this support.

3. Training:
   Current frozen backbone is trained on available positive observed OD pairs.
   Primary validation/evaluation is interzonal Omega_c^+.
   Do not change training support unless explicitly instructed.

4. Model:
   Urban features -> Urban spatial GNN -> node embeddings.
   GravityPrior supplies population/distance prior.
   Pairwise decoder consumes:
       h_i, h_j, log1p(distance), log gravity prior.
   Loss is ZTNB.
   Inference prediction is E[T | T >= 1].
   Do not add target OD information to M0.

5. Graph:
   Urban graph must be constructed only from observable geography.
   Never construct GNN edges from OD observations or flows.

6. Model seeds:
       [1, 10, 100]
   All three seeds are mandatory for a certified full run.

7. Cross-city folds:
       35 train
       5 validation
       10 test
   5 folds total.
   All 50 test cities must occur exactly once.
   Use the locked split manifest.
   Verify the split-manifest SHA256.

8. Distance bins:
   K=8 unless the current experiment is explicitly a K-sensitivity arm.
   Bin edges must be derived from TRAINING CITIES ONLY.
   Never derive bin edges from target/test OD.

9. Canonical calibration:
   src.calibration.bin_calibration.calibrate_kbins()

   This is the single source of truth.

   Any optimized/manual implementation MUST be numerically checked against
   calibrate_kbins on synthetic and real samples.

   Mandatory invariants:
       q=0 -> exact M0 identity
       q=1 -> target active-bin distribution matched
       predicted interzonal mass preserved
       intrazonal values unchanged

   Active bins are determined from candidate/model support,
   NOT from yd_target > 0.

10. Statistical unit:
   CITY.

   For stochastic experiments use:
       replicates -> mean within model seed
       model seeds -> mean within city
       cities -> inferential statistics

   Never treat replicates or model seeds as independent statistical samples.

11. Primary inference:
   Paired city-level Delta CPC.
   Fold-stratified bootstrap for mean CI.
   Paired Wilcoxon for city-level paired comparison.
   Use Holm correction when multiple inferential comparisons are made.

12. Checkpoints:
   Full runs must fail if a required checkpoint is missing.
   Never silently continue with fewer model seeds.

   Checkpoint metadata must match expected:
       split hash
       fold
       seed
       architecture
       graph config
       loss
       LR
       epochs
       patience
       scheduler
       weight decay
       relevant hyperparameters

13. Resume:
   Never mix stale and new outputs.
   Before resume, verify protocol version, seeds, fold, grid, B,
   checkpoint config and split hash.

14. Numerical failures:
   Full scientific runs must FAIL FAST.
   NaN/Inf loss must raise an exception.
   Do not silently skip a city.

15. Partial-OD experiment:
   Use one stable permutation per city/replicate.
   Reveal sets must be nested across p.
   The same mask must be used across model seeds.
   Partial Y_D must be fed through the SAME canonical calibrate_kbins operator.
   p=0 has undefined empirical Y_D discrepancy; do not report TV=0 by default.

16. Noise experiment:
   Requested TV epsilon must be achieved numerically.
   Validate abs(actual_TV - epsilon) < tolerance.
   Use deterministic seeds.

17. Sampling experiment:
   Empirical Y_D sampling must be based on observed trip counts.
   If sampling without replacement, use multivariate hypergeometric sampling.

18. Spatial-resolution experiment:
   Keep tract-level pair distances.
   Assign origin tract to county and compute grouped Y_D from tract pairs.
   Do NOT aggregate a county pair to one mean distance and then re-bin it.
   Mapping failures must be explicit and audited.

19. Terminology:
   Call M0 "support-conditioned zero-shot".
   Do not claim full-matrix zero-shot prediction.
   Do not claim decoder initialization equals the pure gravity prediction:
   current decoder uses softplus(log_gravity + residual).
   Describe GNN as distance-conditioned message passing,
   not distance-weighted aggregation unless actual weights are implemented.

20. Before editing:
   State:
       a) files that will change,
       b) invariants that must remain unchanged,
       c) whether this is reporting-only, analysis-only,
          calibration/prediction-changing, or backbone-changing,
       d) expected rerun scope.

21. After editing:
   Run relevant unit tests and smoke tests.
   Report:
       exact changes
       contract tests
       numerical equivalence checks
       row/count QA
       whether old results remain valid
       exactly which experiments, if any, need rerunning.

22. Decouple execution completion from certification:
    Runners write EXECUTION_COMPLETE.marker upon successful pipeline computation.
    FROZEN.marker indicates that all post-execution scientific certification gates and contract tests have passed.
    Never treat an unverified execution as certified.

When uncertain, preserve existing frozen scientific behavior and fail loudly
rather than silently changing the experiment.


# AUDIT CLARIFICATIONS (Pre-paper freeze)

23. Estimand boundary:
   All evaluations are SUPPORT-CONDITIONED.
   The estimand is OD intensity reconstruction conditional on known positive OD support.
   Formally: predict T_ij > 0 for (i,j) in Omega_c^+.
   This is NOT full N x N matrix reconstruction.
   Do NOT claim "predicts previously unknown OD links" or "reconstructs complete OD matrix."
   Acceptable claim: "support-conditioned OD intensity reconstruction" or
   "OD intensity reconstruction over the observed positive interzonal support."

24. Y_D provenance for M1 conditions:
   The canonical M1_city, M1_county, and M1_subzone conditions in run_experiment.py
   and run_e1.py use Y_D^{GT,+}: distance-binned aggregate histograms extracted
   DIRECTLY from the target city's own ground-truth OD flows.
   This is a deliberate target-information intervention.
   It is NOT an external Meta/GAMD observation.
   Output artifacts store yd_source = "target_ground_truth_positive_od" to confirm this.
   If future work uses externally observable Y_D (e.g., from Meta mobility data),
   use a separate runner and key suffix (e.g., M1_city_external_meta).

25. Bin edge weighting:
   Distance bin boundaries are computed using PAIR-WEIGHTED quantile pooling
   across training cities (via compute_kbin_edges in src/data/yd_extractor.py).
   Large cities contribute proportionally more OD pairs to the quantile estimation.
   This means bins represent the distribution of training OD pairs, not the
   city-average distribution.
   Paper Methods must state: "Bin boundaries were computed using pair-weighted distance
   quantiles pooled across training cities."

26. Statistical unit and seed averaging:
   The statistical unit is the CITY (N=50).
   Model seeds {1, 10, 100} are averaged within each city BEFORE any inference.
   Do NOT treat 50 cities x 3 seeds = 150 as independent observations.
   Bootstrap and Wilcoxon operate on N=50 city-level values.
   Paper Methods must state: "City was treated as the statistical unit; model-seed
   variability was averaged within each city before cross-city inference."

27. K sensitivity canonical grid:
   The canonical K-sensitivity analysis tests K in {2, 4, 6, 8, 10, 12, 14, 16, 18, 20}
   (10 resolution levels, defined as CANONICAL_K_VALUES in run_k_sensitivity_v1.py).
   This grid is frozen for paper submission.
   Paper Methods must cite exactly these 10 values.
   Do not run K-sensitivity with a different grid and call it canonical.
