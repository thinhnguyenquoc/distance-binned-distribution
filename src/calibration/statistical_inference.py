"""
Statistical Inference Suite for Cross-City Zero-Shot Transfer Experiments.

Architecture:
- Tier A: Seed Aggregation (average 3 seeds per source-target-model tuple).
- Tier B: Target-Level Primary Summary Inference (50 target-city aggregated summaries G_t;
          note: summaries share source models and are not assumed to be fully independent).
- Tier C: Source-Level Transfer Summary Analysis (50 source-city aggregated summaries G_s).
- Tier D: Crossed Random-Effects Robustness Model (Delta_CPC ~ 1 + (1|source) + (1|target),
          primary robustness check accounting for crossed source and target dependence).
- Heatmap Plotting: 50x50 transfer matrix visualization (diagonal = NA).
"""

from typing import Tuple, Dict, Optional, List
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import scipy.linalg as la
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Tier A: Seed Aggregation
# ---------------------------------------------------------------------------
def aggregate_seeds(df_calibration: pd.DataFrame) -> pd.DataFrame:
    """
    Averages metric gains across exactly 3 random seeds for each (source_city, target_city, model, K).
    Enforces strict invariant: each (source, target, model, K) group must contain exactly 3 seed runs.
    From 22,050 seed-level transfer runs overall (7,350 runs per model family),
    returns 2,450 seed-averaged source-target pairs per model family.
    """
    groupby_cols = ["source_city", "target_city", "model", "K"]
    metric_cols = [
        "CPC_before", "CPC_after", "delta_CPC",
        "MAE_before", "MAE_after", "delta_MAE",
        "MSE_before", "MSE_after", "delta_MSE"
    ]
    avail_metrics = [c for c in metric_cols if c in df_calibration.columns]

    # Verify that each pair has exactly 3 seed runs for stochastic models
    counts = df_calibration.groupby(groupby_cols).size()
    if not (counts == 3).all():
        bad_groups = counts[counts != 3]
        raise ValueError(f"Seed aggregation error: {len(bad_groups)} groups do not have exactly 3 seeds. "
                         f"Example: {bad_groups.head()}")

    agg_df = df_calibration.groupby(groupby_cols)[avail_metrics].agg(["mean", "std"]).reset_index()
    flat_cols = []
    for col in agg_df.columns:
        if col[1] == "mean" or col[1] == "":
            flat_cols.append(col[0])
        else:
            flat_cols.append(f"{col[0]}_{col[1]}")
    agg_df.columns = flat_cols
    return agg_df


# ---------------------------------------------------------------------------
# Tier B: Target-Level Analysis (Separating City Summary and Global Inference)
# ---------------------------------------------------------------------------
def compute_target_city_summary(
    df_seed_averaged: pd.DataFrame,
    metric_col: str = "delta_CPC",
) -> pd.DataFrame:
    r"""
    Level 1: Target-City Descriptive Summary.
    Computes descriptive statistics for each target city across its 49 source models.
        G_t = 1/49 * sum_{s != t} Delta_CPC_{s,t}
        
    Schema:
        target_city, model, G_t, median_source_gain, IQR_source_gain, std_source_gain,
        n_sources, positive_sources, positive_source_fraction
        
    Strict Invariant:
    - Exactly 50 rows per model.
    - Zero p-values, zero global bootstrap CIs in individual city rows.
    """
    models = df_seed_averaged["model"].unique()
    all_rows = []

    for m in models:
        m_df = df_seed_averaged[df_seed_averaged["model"] == m]
        for target_city, group in m_df.groupby("target_city"):
            gains = group[metric_col].values
            n_sources = len(gains)
            assert n_sources == 49, f"Target {target_city} has {n_sources} sources, expected 49"

            g_t = float(np.mean(gains))
            med = float(np.median(gains))
            q75, q25 = np.percentile(gains, [75, 25])
            iqr = float(q75 - q25)
            std_val = float(np.std(gains, ddof=1)) if n_sources > 1 else 0.0
            pos = int(np.sum(gains > 0))
            pos_frac = float(pos / n_sources)

            all_rows.append({
                "target_city": target_city,
                "model": m,
                "G_t": round(g_t, 6),
                "median_source_gain": round(med, 6),
                "IQR_source_gain": round(iqr, 6),
                "std_source_gain": round(std_val, 6),
                "n_sources": n_sources,
                "positive_sources": pos,
                "positive_source_fraction": round(pos_frac, 6),
            })

    return pd.DataFrame(all_rows)


def compute_global_target_inference(
    df_target_city_summary: pd.DataFrame,
    n_bootstraps: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    r"""
    Level 2: Global Across-Target Statistical Inference (Tier B).
    Operates on the 50 aggregated G_t values for each model:
        G_t = 1/49 * \sum_{s != t} \bar{\Delta CPC}_{s,t}

    Roles & Scientific Interpretation Rules:
    - Bootstrap (10,000 resamples on 50 G_t):
      Serves strictly as a *target-level descriptive uncertainty summary* reflecting
      empirical variability among target summaries.
      DO NOT describe as "cross-dependence-adjusted CI" or "cluster-robust CI".
      DO NOT describe as removing or accounting for source-target dependence.
    - Wilcoxon signed-rank test (on 50 G_t vs 0):
      Serves strictly as a *target-level summary test* evaluating whether the distribution
      of aggregated target gains systematically deviates from zero.
      DO NOT describe as accounting for dependence due to shared source models.
    - Independence Assumption:
      The 50 target summaries share source models and are NOT independent observations.
      Tier D crossed mixed-effects model (\Delta CPC ~ 1 + (1|source) + (1|target))
      is the primary dependence-aware robustness analysis addressing crossed dependence.
    - Seed:
      Fixed statistical seed (default seed=42) pre-specified and held constant across models.

    Schema:
        model, n_targets, mean_G, median_G, IQR_G, bootstrap_ci_low, bootstrap_ci_high,
        wilcoxon_stat, wilcoxon_p, positive_targets, positive_target_fraction

    Strict Invariants:
    - Exactly ONE row per model.
    - Resampling unit is strictly the 50 target cities (resamples 50 G_t values with replacement).
    - Never bootstrap 2,450 pair rows as independent observations.
    - Wilcoxon test is strictly Wilcoxon(G_1 ... G_50, 0).
    """
    models = df_target_city_summary["model"].unique()
    rng = np.random.RandomState(seed)
    rows = []

    for m in models:
        m_df = df_target_city_summary[df_target_city_summary["model"] == m]
        g_t = m_df["G_t"].values
        n_targets = len(g_t)
        assert n_targets == 50, f"Expected 50 targets, got {n_targets}"

        mean_G = float(np.mean(g_t))
        median_G = float(np.median(g_t))
        q75, q25 = np.percentile(g_t, [75, 25])
        iqr_G = float(q75 - q25)

        # Bootstrap 95% CI on the 50 G_t values
        boot_means = [np.mean(rng.choice(g_t, size=n_targets, replace=True)) for _ in range(n_bootstraps)]
        ci_low = float(np.percentile(boot_means, 2.5))
        ci_high = float(np.percentile(boot_means, 97.5))

        # Wilcoxon signed-rank test on 50 G_t values vs 0
        if np.all(g_t == 0) or len(np.unique(g_t)) <= 1:
            w_stat, p_val = 0.0, 1.0
        else:
            res_w = stats.wilcoxon(g_t, alternative="greater")
            w_stat, p_val = float(res_w.statistic), float(res_w.pvalue)

        pos_targets = int(np.sum(g_t > 0))
        pos_frac = float(pos_targets / n_targets)

        rows.append({
            "model": m,
            "n_targets": n_targets,
            "mean_G": round(mean_G, 6),
            "median_G": round(median_G, 6),
            "IQR_G": round(iqr_G, 6),
            "bootstrap_ci_low": round(ci_low, 6),
            "bootstrap_ci_high": round(ci_high, 6),
            "wilcoxon_stat": round(w_stat, 2),
            "wilcoxon_p": f"{p_val:.6e}",
            "positive_targets": pos_targets,
            "positive_target_fraction": round(pos_frac, 6),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tier C: Source-Level Transfer Summary Analysis
# ---------------------------------------------------------------------------
def compute_source_city_summary(
    df_seed_averaged: pd.DataFrame,
    metric_col: str = "delta_CPC",
) -> pd.DataFrame:
    r"""
    Tier C: Source-City Descriptive Summary.
    Computes descriptive statistics for each source city across its 49 target cities:
        G_s = 1/49 * \sum_{t != s} \bar{\Delta CPC}_{s,t}

    Roles & Scientific Interpretation Rules:
    - Purely a supporting descriptive diagnostic to evaluate source heterogeneity
      and check whether calibration gains are dominated by specific source models.
    - NOT a separate experiment (no Experiment F) and NOT a confirmatory hypothesis test.
    - No correlation analysis (e.g. held-out performance vs G_s) in main protocol.
    - No causal inferences (e.g. "better source models cause larger DBD gains").

    Schema:
        source_city, model, G_s, median_target_gain, IQR_target_gain, std_target_gain,
        n_targets, positive_targets, positive_target_fraction
    """
    models = df_seed_averaged["model"].unique()
    all_rows = []

    for m in models:
        m_df = df_seed_averaged[df_seed_averaged["model"] == m]
        for source_city, group in m_df.groupby("source_city"):
            gains = group[metric_col].values
            n_targets = len(gains)
            assert n_targets == 49, f"Source {source_city} has {n_targets} targets, expected 49"

            g_s = float(np.mean(gains))
            med = float(np.median(gains))
            q75, q25 = np.percentile(gains, [75, 25])
            iqr = float(q75 - q25)
            std_val = float(np.std(gains, ddof=1)) if n_targets > 1 else 0.0
            pos = int(np.sum(gains > 0))
            pos_frac = float(pos / n_targets)

            all_rows.append({
                "source_city": source_city,
                "model": m,
                "G_s": round(g_s, 6),
                "median_target_gain": round(med, 6),
                "IQR_target_gain": round(iqr, 6),
                "std_target_gain": round(std_val, 6),
                "n_targets": n_targets,
                "positive_targets": pos,
                "positive_target_fraction": round(pos_frac, 6),
            })

    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Tier D: Crossed Source-Target Random-Effects Robustness Model
# ---------------------------------------------------------------------------
def fit_crossed_random_effects(
    df_seed_averaged: pd.DataFrame,
    metric_col: str = "delta_CPC",
    optimizer_maxiter: int = 1000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    r"""
    Fits crossed random-effects model using exact Restricted Maximum Likelihood (REML)
    via Henderson's Mixed Model Equations (MME):
        \Delta CPC_{s,t} = \beta_0 + u_s + v_t + \epsilon_{s,t}
        u_s ~ N(0, \sigma_s^2), v_t ~ N(0, \sigma_t^2), \epsilon_{s,t} ~ N(0, \sigma_\epsilon^2)

    Strict Invariants:
    1. Input Data Contract:
       - Exactly 2,450 source-target pairs per model.
       - 50 unique source cities, 50 unique target cities.
       - No diagonal observations (source != target).
       - Zero missing values in metric_col.
    2. Fit separately per model family:
       - gravity_2param, pairwise_mlp, urban_gnn. Never pool across models.
    3. Estimation:
       - Pure REML with L-BFGS-B (maxiter=1000).
    4. Hard Failure Convergence Policy:
       - Raises ValueError if optimizer fails to converge. No silent fallback to OLS,
         no dropping random effects, no switching optimizers.
    5. Singular / Near-Zero Variance Preservation:
       - Never drops random intercepts even if variance is near boundary.
    6. Outputs:
       - crossed_effects_results.csv (1 row per model)
       - crossed_effects_diagnostics.csv (1 row per model)
    """
    models = df_seed_averaged["model"].unique()
    results_rows = []
    diag_rows = []

    for m in models:
        sub = df_seed_averaged[df_seed_averaged["model"] == m].copy()

        # 1. Input data validation
        n_pairs = len(sub)
        if n_pairs != 2450:
            raise ValueError(f"Model {m} has {n_pairs} observations, expected exactly 2450.")
        if sub["source_city"].nunique() != 50:
            raise ValueError(f"Model {m} has {sub['source_city'].nunique()} source cities, expected 50.")
        if sub["target_city"].nunique() != 50:
            raise ValueError(f"Model {m} has {sub['target_city'].nunique()} target cities, expected 50.")
        if (sub["source_city"] == sub["target_city"]).any():
            raise ValueError(f"Model {m} contains diagonal entries (source == target).")
        if sub[metric_col].isna().any():
            raise ValueError(f"Model {m} has missing values in {metric_col}.")

        # 2. Sort by canonical city ordering
        canonical_cities = sorted(sub["source_city"].unique())
        city_to_idx = {c: i for i, c in enumerate(canonical_cities)}
        sub = sub.sort_values(by=["source_city", "target_city"]).reset_index(drop=True)

        y = sub[metric_col].values.astype(np.float64)
        N = len(y)
        S = 50
        T = 50
        p = 1  # Intercept dimension

        # Construct Design Matrices X (N x 1) and Z (N x 100)
        X = np.ones((N, 1), dtype=np.float64)
        Z_s = np.zeros((N, S), dtype=np.float64)
        Z_t = np.zeros((N, T), dtype=np.float64)

        for row_idx, row in sub.iterrows():
            s_idx = city_to_idx[row["source_city"]]
            t_idx = city_to_idx[row["target_city"]]
            Z_s[row_idx, s_idx] = 1.0
            Z_t[row_idx, t_idx] = 1.0

        Z = np.hstack([Z_s, Z_t])  # (2450, 100)

        # Precompute mixed model equation blocks
        XtX = X.T @ X  # (1, 1) = [[2450.0]]
        XtZ = X.T @ Z  # (1, 100)
        ZtX = Z.T @ X  # (100, 1)
        ZtZ = Z.T @ Z  # (100, 100)
        Xty = X.T @ y  # (1,)
        Zty = Z.T @ y  # (100,)
        yty = float(np.dot(y, y))
        r = np.concatenate([Xty, Zty])  # (101,)

        # Two-way crossed ANOVA initial variance estimates for variance ratios
        grand_mean = float(np.mean(y))
        src_means = np.array([np.mean(y[sub["source_city"] == c]) for c in canonical_cities])
        tgt_means = np.array([np.mean(y[sub["target_city"] == c]) for c in canonical_cities])
        
        raw_var_y = float(np.var(y, ddof=1))
        var_s_init = max(1e-6, float(np.var(src_means, ddof=1) - raw_var_y / (T - 1)))
        var_t_init = max(1e-6, float(np.var(tgt_means, ddof=1) - raw_var_y / (S - 1)))
        var_e_init = max(1e-6, float(raw_var_y - var_s_init - var_t_init))

        init_log_gs = float(np.log(max(1e-5, var_s_init / var_e_init)))
        init_log_gt = float(np.log(max(1e-5, var_t_init / var_e_init)))

        # Profile REML Criterion over log variance ratios: params = (log(gamma_s), log(gamma_t))
        # where gamma_s = sigma_s^2 / sigma_e^2, gamma_t = sigma_t^2 / sigma_e^2
        def reml_objective(params):
            log_gs, log_gt = params
            gs, gt = np.exp(log_gs), np.exp(log_gt)

            diag_Ginv = np.concatenate([np.full(S, 1.0 / gs), np.full(T, 1.0 / gt)])
            C = np.empty((101, 101), dtype=np.float64)
            C[0, 0] = XtX[0, 0]
            C[0, 1:] = XtZ[0, :]
            C[1:, 0] = ZtX[:, 0]
            C[1:, 1:] = ZtZ + np.diag(diag_Ginv)

            try:
                L = la.cholesky(C, lower=True)
            except la.LinAlgError:
                return 1e12

            sol = la.cho_solve((L, True), r)
            quad = yty - float(np.dot(sol, r))
            if quad <= 0:
                return 1e12

            sigma_e2 = quad / (N - p)
            log_det_C = 2.0 * float(np.sum(np.log(np.diag(L))))
            log_det_G = float(S * np.log(gs) + T * np.log(gt))

            # -2 * log_likelihood_REML
            neg_2_loglik = (N - p) * np.log(sigma_e2) + log_det_G + log_det_C
            return neg_2_loglik

        opt_res = minimize(
            reml_objective,
            [init_log_gs, init_log_gt],
            method="L-BFGS-B",
            options={"maxiter": optimizer_maxiter}
        )

        # 3. Strict Convergence Check (Hard Failure Policy)
        if not opt_res.success:
            raise ValueError(f"Mixed-effects optimizer failed to converge for model {m}: {opt_res.message}")

        gs_hat, gt_hat = np.exp(opt_res.x)
        diag_Ginv = np.concatenate([np.full(S, 1.0 / gs_hat), np.full(T, 1.0 / gt_hat)])
        C = np.empty((101, 101), dtype=np.float64)
        C[0, 0] = XtX[0, 0]
        C[0, 1:] = XtZ[0, :]
        C[1:, 0] = ZtX[:, 0]
        C[1:, 1:] = ZtZ + np.diag(diag_Ginv)

        L = la.cholesky(C, lower=True)
        sol = la.cho_solve((L, True), r)
        beta0_hat = float(sol[0])
        quad = yty - float(np.dot(sol, r))
        sigma_e2_hat = float(quad / (N - p))
        sigma_s2_hat = float(gs_hat * sigma_e2_hat)
        sigma_t2_hat = float(gt_hat * sigma_e2_hat)

        # Asymptotic variance of beta0
        C_inv = la.cho_solve((L, True), np.eye(101))
        se_beta0 = float(np.sqrt(sigma_e2_hat * C_inv[0, 0]))
        ci_low = float(beta0_hat - 1.96 * se_beta0)
        ci_high = float(beta0_hat + 1.96 * se_beta0)

        # Full log-likelihood for reporting:
        # log_lik = -0.5 * [(N - p) * log(2*pi*sigma_e2) + log|G| + log|C| + (N - p)]
        log_det_C = 2.0 * float(np.sum(np.log(np.diag(L))))
        log_det_G = float(S * np.log(gs_hat) + T * np.log(gt_hat))
        log_lik = -0.5 * ((N - p) * (np.log(2 * np.pi * sigma_e2_hat) + 1.0) + log_det_G + log_det_C)

        # Diagnostics & warnings detection
        singular_warn = bool(C_inv[0, 0] < 1e-12 or np.isnan(C_inv[0, 0]))
        boundary_warn = bool(sigma_s2_hat < 1e-8 or sigma_t2_hat < 1e-8)
        hessian_warn = False
        other_warn = ""
        warnings_list = []
        if boundary_warn:
            warnings_list.append("boundary_variance")
        if singular_warn:
            warnings_list.append("singular_covariance")

        warnings_str = ";".join(warnings_list) if warnings_list else "none"

        results_rows.append({
            "model": m,
            "n_pairs": N,
            "estimation_method": "REML",
            "optimizer": "lbfgs",
            "converged": True,
            "beta0": round(beta0_hat, 6),
            "se_beta0": round(se_beta0, 6),
            "ci_low": round(ci_low, 6),
            "ci_high": round(ci_high, 6),
            "source_variance": round(sigma_s2_hat, 8),
            "target_variance": round(sigma_t2_hat, 8),
            "residual_variance": round(sigma_e2_hat, 8),
            "log_likelihood": round(log_lik, 4),
            "warnings": warnings_str,
        })

        diag_rows.append({
            "model": m,
            "converged": True,
            "n_iterations": int(opt_res.nit),
            "singular_warning": singular_warn,
            "boundary_warning": boundary_warn,
            "hessian_warning": hessian_warn,
            "other_warning": other_warn if other_warn else "none",
        })

    return pd.DataFrame(results_rows), pd.DataFrame(diag_rows)


# ---------------------------------------------------------------------------
# Tier Figure: 50x50 Transfer Matrix Heatmap
# ---------------------------------------------------------------------------
def plot_transfer_heatmap(
    df_seed_averaged: pd.DataFrame,
    output_path: Path,
    model_name: str = "GNN",
    metric_col: str = "delta_CPC",
    figsize: Tuple[int, int] = (16, 14)
):
    """
    Plots a 50x50 heatmap with source cities on rows, target cities on columns,
    and seed-averaged delta_CPC on cells (diagonal = NA).
    """
    sub = df_seed_averaged[df_seed_averaged["model"] == model_name]
    matrix = sub.pivot(index="source_city", columns="target_city", values=metric_col)
    
    # Fill diagonal with NaN for visual separation
    for c in matrix.index:
        if c in matrix.columns:
            matrix.loc[c, c] = np.nan

    plt.figure(figsize=figsize)
    data = matrix.values
    masked_data = np.ma.masked_invalid(data)
    
    cmap = plt.cm.coolwarm.copy()
    cmap.set_bad(color="lightgray")
    
    im = plt.imshow(masked_data, cmap=cmap, aspect="auto", interpolation="nearest")
    cbar = plt.colorbar(im)
    cbar.set_label(f"Seed-Averaged {metric_col}", fontsize=12)
    
    plt.title(f"Cross-City Transfer Gain Heatmap (50x50 Matrix, {model_name})", fontsize=14, pad=15)
    plt.xlabel("Target City (Unseen)", fontsize=12)
    plt.ylabel("Source City (30% Training Data)", fontsize=12)
    plt.xticks(ticks=range(len(matrix.columns)), labels=matrix.columns, rotation=90, fontsize=8)
    plt.yticks(ticks=range(len(matrix.index)), labels=matrix.index, rotation=0, fontsize=8)
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved heatmap to: {output_path}")
