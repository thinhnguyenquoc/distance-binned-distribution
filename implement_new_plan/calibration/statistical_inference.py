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
    rng = np.random.default_rng(seed)
    rows = []

    # Compute statistics and raw p-values for all models
    model_data = []
    for m in models:
        m_df = df_target_city_summary[df_target_city_summary["model"] == m]
        g_t = m_df["G_t"].values
        n_targets = len(g_t)
        assert n_targets == 50, f"Expected 50 targets, got {n_targets}"

        mean_G = float(np.mean(g_t))
        median_G = float(np.median(g_t))
        q75 = float(np.quantile(g_t, 0.75, method="linear"))
        q25 = float(np.quantile(g_t, 0.25, method="linear"))
        iqr_G = float(q75 - q25)

        # Bootstrap 95% CI on the 50 G_t values
        boot_means = [np.mean(rng.choice(g_t, size=n_targets, replace=True)) for _ in range(n_bootstraps)]
        ci_low = float(np.percentile(boot_means, 2.5))
        ci_high = float(np.percentile(boot_means, 97.5))

        # Locked Wilcoxon signed-rank test on 50 G_t values vs 0
        if np.all(g_t == 0) or len(np.unique(g_t)) <= 1:
            w_stat, p_val = 0.0, 1.0
        else:
            res_w = stats.wilcoxon(x=g_t, alternative="two-sided", zero_method="wilcox", correction=False, method="auto")
            w_stat, p_val = float(res_w.statistic), float(res_w.pvalue)

        pos_targets = int(np.sum(g_t > 0))
        pos_frac = float(pos_targets / n_targets)

        model_data.append({
            "model": m,
            "n_targets": n_targets,
            "mean_G": round(mean_G, 6),
            "median_G": round(median_G, 6),
            "IQR_G": round(iqr_G, 6),
            "bootstrap_ci_low": round(ci_low, 6),
            "bootstrap_ci_high": round(ci_high, 6),
            "wilcoxon_stat": round(w_stat, 2),
            "wilcoxon_p_raw": float(p_val),
            "positive_targets": pos_targets,
            "positive_target_fraction": round(pos_frac, 6),
        })

    # Holm-Bonferroni correction across models
    p_raws = [d["wilcoxon_p_raw"] for d in model_data]
    order = np.argsort(p_raws)
    p_holm = np.zeros(len(p_raws), dtype=np.float64)
    m_tests = len(p_raws)
    cum_max = 0.0
    for rank, orig_idx in enumerate(order):
        adj = min(1.0, p_raws[orig_idx] * (m_tests - rank))
        cum_max = max(cum_max, adj)
        p_holm[orig_idx] = cum_max

    for i, d in enumerate(model_data):
        d["wilcoxon_p_holm"] = float(p_holm[i])
        rows.append(d)

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
    Fits the canonical crossed source-target random-effects model using statsmodels MixedLM with REML:
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
       - statsmodels MixedLM.from_formula with groups=_all, re_formula="0",
         vc_formula={"source": "0 + C(source_city)", "target": "0 + C(target_city)"},
         method="lbfgs", reml=True, maxiter=1000.
    4. Hard Failure Convergence Policy:
       - Raises RuntimeError if optimizer fails to converge. No silent fallback,
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

    from statsmodels.regression.mixed_linear_model import MixedLM

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

        # 2. Canonical Crossed Random Effects MixedLM
        sub["_all"] = 1
        model = MixedLM.from_formula(
            f"{metric_col} ~ 1",
            groups=sub["_all"],
            re_formula="0",
            vc_formula={
                "source": "0 + C(source_city)",
                "target": "0 + C(target_city)",
            },
            data=sub,
        )

        fit = model.fit(method="lbfgs", maxiter=optimizer_maxiter, reml=True)

        # 3. Strict Convergence Check (Hard Failure Policy)
        if not fit.converged:
            raise RuntimeError(f"Mixed-effects optimizer failed to converge for model {m} after {optimizer_maxiter} iterations.")

        beta0_hat = float(fit.params["Intercept"])
        se_beta0 = float(fit.bse["Intercept"])
        ci_low = float(beta0_hat - 1.95996 * se_beta0)
        ci_high = float(beta0_hat + 1.95996 * se_beta0)

        vcomp = fit.vcomp
        sigma_s2_hat = float(vcomp[0]) if len(vcomp) > 0 else 0.0
        sigma_t2_hat = float(vcomp[1]) if len(vcomp) > 1 else 0.0
        sigma_e2_hat = float(fit.scale)
        log_lik = float(fit.llf)

        # Diagnostics & warnings detection
        singular_warn = bool(se_beta0 < 1e-12 or np.isnan(se_beta0))
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
            "n_pairs": n_pairs,
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
            "n_iterations": int(fit.iterations if hasattr(fit, "iterations") else 0),
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


# ---------------------------------------------------------------------------
# Tier RQ2: Experiment B Adjacent Contrasts & Scarcity Inference
# ---------------------------------------------------------------------------

ADJACENT_CONTRASTS = [
    ("10_vs_20", 0.10, 0.20),
    ("20_vs_30", 0.20, 0.30),
    ("30_vs_50", 0.30, 0.50),
    ("50_vs_100", 0.50, 1.00),
]

def compute_scarcity_contrasts(
    df_scarcity_mean: pd.DataFrame,
    metric_col: str = "delta_CPC_mean",
) -> pd.DataFrame:
    r"""
    Computes 4 adjacent paired contrasts for Experiment B:
        C^{a-b}_{s,t} = g_{s,t,a} - g_{s,t,b}
    where contrast_name in {"10_vs_20", "20_vs_30", "30_vs_50", "50_vs_100"}.
    
    Input: df_scarcity_mean with (source_city, target_city, model, train_fraction, delta_CPC_mean).
    Output: DataFrame with (source_city, target_city, model, contrast_name, contrast_value).
    Total rows = 2450 pairs * 3 models * 4 contrasts = 29,400 rows.
    """
    piv = df_scarcity_mean.pivot(
        index=["source_city", "target_city", "model"],
        columns="train_fraction",
        values=metric_col
    ).reset_index()

    records = []
    for contrast_name, f_a, f_b in ADJACENT_CONTRASTS:
        diff = piv[f_a] - piv[f_b]
        for idx, row in piv.iterrows():
            records.append({
                "source_city": row["source_city"],
                "target_city": row["target_city"],
                "model": row["model"],
                "contrast_name": contrast_name,
                "contrast_value": float(diff.iloc[idx]),
            })
    return pd.DataFrame(records)


def compute_scarcity_contrast_target_summary(
    df_contrasts: pd.DataFrame
) -> pd.DataFrame:
    r"""
    Computes target-level descriptive summary for 4 adjacent contrasts:
        C_t^{a-b} = 1/49 * sum_{s != t} C_{s,t}^{a-b}
    Schema:
        target_city, model, contrast_name, mean_contrast, median_contrast, IQR_contrast, std_contrast, n_sources, positive_sources
    Expected rows: 50 targets * 3 models * 4 contrasts = 600 rows.
    """
    rows = []
    for (m, c_name), group_mc in df_contrasts.groupby(["model", "contrast_name"]):
        for target_city, group in group_mc.groupby("target_city"):
            vals = group["contrast_value"].values
            n_sources = len(vals)
            assert n_sources == 49, f"Target {target_city} has {n_sources} sources, expected 49"

            m_val = float(np.mean(vals))
            med_val = float(np.median(vals))
            q75, q25 = np.percentile(vals, [75, 25])
            iqr_val = float(q75 - q25)
            std_val = float(np.std(vals, ddof=1)) if n_sources > 1 else 0.0
            pos_sources = int(np.sum(vals > 0))

            rows.append({
                "target_city": target_city,
                "model": m,
                "contrast_name": c_name,
                "mean_contrast": round(m_val, 6),
                "median_contrast": round(med_val, 6),
                "IQR_contrast": round(iqr_val, 6),
                "std_contrast": round(std_val, 6),
                "n_sources": n_sources,
                "positive_sources": pos_sources,
            })
    return pd.DataFrame(rows)


def compute_scarcity_contrast_inference(
    df_contrast_target_summary: pd.DataFrame,
    n_bootstraps: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    r"""
    Global across-target statistical inference for 4 adjacent contrasts with Holm correction:
    Operates on 50 aggregated C_t^{a-b} values.
    Schema:
        model, contrast_name, n_targets, mean_contrast, median_contrast, IQR_contrast,
        bootstrap_ci_low, bootstrap_ci_high, wilcoxon_stat, wilcoxon_p_raw, wilcoxon_p_holm,
        positive_targets, positive_target_fraction
    Expected rows: 3 models * 4 contrasts = 12 rows.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for m in df_contrast_target_summary["model"].unique():
        m_df = df_contrast_target_summary[df_contrast_target_summary["model"] == m]
        raw_p_list = []
        model_rows = []

        for c_name in ["10_vs_20", "20_vs_30", "30_vs_50", "50_vs_100"]:
            c_df = m_df[m_df["contrast_name"] == c_name]
            vals = c_df["mean_contrast"].values
            n_targets = len(vals)
            assert n_targets == 50, f"Expected 50 targets for {m}-{c_name}, got {n_targets}"

            mean_c = float(np.mean(vals))
            median_c = float(np.median(vals))
            q75 = float(np.quantile(vals, 0.75, method="linear"))
            q25 = float(np.quantile(vals, 0.25, method="linear"))
            iqr_c = float(q75 - q25)

            boot_means = [np.mean(rng.choice(vals, size=n_targets, replace=True)) for _ in range(n_bootstraps)]
            ci_low = float(np.percentile(boot_means, 2.5))
            ci_high = float(np.percentile(boot_means, 97.5))

            if np.all(vals == 0) or len(np.unique(vals)) <= 1:
                w_stat, p_raw = 0.0, 1.0
            else:
                res_w = stats.wilcoxon(x=vals, alternative="two-sided", zero_method="wilcox", correction=False, method="auto")
                w_stat, p_raw = float(res_w.statistic), float(res_w.pvalue)

            raw_p_list.append(p_raw)
            pos_targets = int(np.sum(vals > 0))
            pos_frac = float(pos_targets / n_targets)

            model_rows.append({
                "model": m,
                "contrast_name": c_name,
                "n_targets": n_targets,
                "mean_contrast": round(mean_c, 6),
                "median_contrast": round(median_c, 6),
                "IQR_contrast": round(iqr_c, 6),
                "bootstrap_ci_low": round(ci_low, 6),
                "bootstrap_ci_high": round(ci_high, 6),
                "wilcoxon_stat": round(w_stat, 2),
                "wilcoxon_p_raw": float(p_raw),
                "positive_targets": pos_targets,
                "positive_target_fraction": round(pos_frac, 6),
            })

        # Holm-Bonferroni correction over the 4 contrasts for this model
        p_raw_arr = np.array(raw_p_list)
        order = np.argsort(p_raw_arr)
        p_holm = np.zeros(len(p_raw_arr), dtype=np.float64)
        m_tests = len(p_raw_arr)  # 4
        cum_max = 0.0
        for rank, orig_idx in enumerate(order):
            adj = min(1.0, p_raw_arr[orig_idx] * (m_tests - rank))
            cum_max = max(cum_max, adj)
            p_holm[orig_idx] = cum_max

        for r_idx, r in enumerate(model_rows):
            r["wilcoxon_p_holm"] = float(p_holm[r_idx])
            rows.append(r)

    return pd.DataFrame(rows)


def fit_scarcity_contrast_mixed_effects(
    df_contrasts: pd.DataFrame
) -> pd.DataFrame:
    r"""
    Fits crossed random-effects models for each adjacent contrast on 2450 pairs:
        C_{s,t} = beta0 + u_s + v_t + epsilon_{s,t}
    Schema:
        model, contrast_name, n_pairs, estimation_method, optimizer, converged, beta0, se_beta0, ci_low, ci_high,
        source_variance, target_variance, residual_variance, log_likelihood, warnings
    Expected rows: 3 models * 4 contrasts = 12 rows.
    """
    results = []
    for (m, c_name), group in df_contrasts.groupby(["model", "contrast_name"]):
        # reuse fit_crossed_random_effects logic
        res_df, _ = fit_crossed_random_effects(group, metric_col="contrast_value")
        row = res_df.iloc[0].to_dict()
        row["contrast_name"] = c_name
        results.append(row)
    return pd.DataFrame(results)


def compute_gap_recovery(
    df_scarcity_results_mean: pd.DataFrame
) -> pd.DataFrame:
    r"""
    Computes GapRecovery for f in {0.10, 0.20, 0.30, 0.50, 1.00}:
        GapRecovery(f) = (CPC_{f+DBD} - CPC_f) / (CPC_100 - CPC_f)
    Strict rules:
    - For f = 1.00: gap_recovery = NaN, gap_recovery_valid = False, invalid_reason = 'full_supervision_reference'
    - For f in {0.10, 0.20, 0.30, 0.50}: valid if denominator > 0, else NaN with reason.
    Total rows = 2450 pairs * 3 models * 5 fractions = 36,750 rows.
    """
    piv_before = df_scarcity_results_mean.pivot(
        index=["source_city", "target_city", "model"],
        columns="train_fraction",
        values="CPC_before_mean"
    )
    piv_after = df_scarcity_results_mean.pivot(
        index=["source_city", "target_city", "model"],
        columns="train_fraction",
        values="CPC_after_mean"
    )

    rows = []
    for (s, t, m), row_before in piv_before.iterrows():
        row_after = piv_after.loc[(s, t, m)]
        cpc_full = float(row_before[1.00])

        for f in [0.10, 0.20, 0.30, 0.50, 1.00]:
            cpc_frac_base = float(row_before[f])
            cpc_frac_cal = float(row_after[f])
            gap_denom = cpc_full - cpc_frac_base

            if f == 1.00:
                gap_rec = np.nan
                valid = False
                inv_reason = "full_supervision_reference"
            elif not np.isfinite(gap_denom):
                gap_rec = np.nan
                valid = False
                inv_reason = "non_finite_denominator"
            elif gap_denom <= 0.0:
                gap_rec = np.nan
                valid = False
                inv_reason = "zero_or_negative_denominator"
            else:
                gap_rec = (cpc_frac_cal - cpc_frac_base) / gap_denom
                valid = True
                inv_reason = "none"

            rows.append({
                "source_city": s,
                "target_city": t,
                "model": m,
                "train_fraction": f,
                "baseline_CPC_fraction": round(cpc_frac_base, 6),
                "calibrated_CPC_fraction": round(cpc_frac_cal, 6),
                "baseline_CPC_full": round(cpc_full, 6),
                "gap_denominator": round(gap_denom, 6),
                "gap_recovery": round(gap_rec, 6) if np.isfinite(gap_rec) else np.nan,
                "gap_recovery_valid": valid,
                "invalid_reason": inv_reason,
            })
    return pd.DataFrame(rows)


def fit_scarcity_overall_mixed_effects(
    df_scarcity_mean: pd.DataFrame
) -> pd.DataFrame:
    r"""
    Fits overall dependence-aware scarcity mixed-effects model across all 5 fractions
    with train_fraction as categorical factor and locked reference level f=1.00:
        \Delta CPC_{s,t,f} = \beta_0 + \sum_{f \in {0.10, 0.20, 0.30, 0.50}} \beta_f * I(frac=f) + u_s + v_t + \epsilon_{s,t,f}
    
    Schema:
        model, n_obs, estimation_method, optimizer, converged, beta0, beta_10, beta_20, beta_30, beta_50,
        se_beta0, se_beta10, se_beta20, se_beta30, se_beta50, p_beta10, p_beta20, p_beta30, p_beta50,
        source_variance, target_variance, residual_variance, log_likelihood, warnings
    Expected rows: exactly 3 rows (1 per model family).
    """
    from statsmodels.regression.mixed_linear_model import MixedLM
    results = []

    for m in df_scarcity_mean["model"].unique():
        sub = df_scarcity_mean[df_scarcity_mean["model"] == m].copy()
        sub["_all"] = 1
        # Create categorical fraction with 1.00 as reference
        sub["frac_cat"] = sub["train_fraction"].astype(str)
        # Using formula with reference level '1.0' or '1.00'
        # Convert to categorical with explicit order: ['1.0', '0.1', '0.2', '0.3', '0.5']
        frac_levels = ['1.0', '0.1', '0.2', '0.3', '0.5']
        sub["frac_cat"] = pd.Categorical(sub["train_fraction"].astype(str), categories=frac_levels, ordered=False)

        model = MixedLM.from_formula(
            "delta_CPC_mean ~ C(frac_cat, Treatment(reference='1.0'))",
            groups=sub["_all"],
            re_formula="0",
            vc_formula={
                "source": "0 + C(source_city)",
                "target": "0 + C(target_city)",
            },
            data=sub,
        )
        fit = model.fit(method="lbfgs", maxiter=1000, reml=True)
        if not fit.converged:
            raise RuntimeError(
                f"MixedLM failed to converge for scarcity overall model on {m} after 1000 iterations (Hard Failure Policy)."
            )
        converged = bool(fit.converged)
        loglik = float(fit.llf)
        params = fit.params
        bse = fit.bse
        pvals = fit.pvalues
        vcomps = fit.vcomp
        scale = float(fit.scale)

        beta0 = float(params.get("Intercept", 0.0))
        se_beta0 = float(bse.get("Intercept", 0.0))

        beta10 = float(params.get("C(frac_cat, Treatment(reference='1.0'))[T.0.1]", 0.0))
        beta20 = float(params.get("C(frac_cat, Treatment(reference='1.0'))[T.0.2]", 0.0))
        beta30 = float(params.get("C(frac_cat, Treatment(reference='1.0'))[T.0.3]", 0.0))
        beta50 = float(params.get("C(frac_cat, Treatment(reference='1.0'))[T.0.5]", 0.0))

        se_beta10 = float(bse.get("C(frac_cat, Treatment(reference='1.0'))[T.0.1]", 0.0))
        se_beta20 = float(bse.get("C(frac_cat, Treatment(reference='1.0'))[T.0.2]", 0.0))
        se_beta30 = float(bse.get("C(frac_cat, Treatment(reference='1.0'))[T.0.3]", 0.0))
        se_beta50 = float(bse.get("C(frac_cat, Treatment(reference='1.0'))[T.0.5]", 0.0))

        p_beta10 = float(pvals.get("C(frac_cat, Treatment(reference='1.0'))[T.0.1]", 1.0))
        p_beta20 = float(pvals.get("C(frac_cat, Treatment(reference='1.0'))[T.0.2]", 1.0))
        p_beta30 = float(pvals.get("C(frac_cat, Treatment(reference='1.0'))[T.0.3]", 1.0))
        p_beta50 = float(pvals.get("C(frac_cat, Treatment(reference='1.0'))[T.0.5]", 1.0))

        var_s = float(vcomps[0]) if len(vcomps) > 0 else 0.0
        var_t = float(vcomps[1]) if len(vcomps) > 1 else 0.0
        warnings = "boundary_variance" if (var_s < 1e-8 or var_t < 1e-8) else "none"

        results.append({
            "model": m,
            "n_obs": len(sub),
            "estimation_method": "REML",
            "optimizer": "lbfgs",
            "converged": converged,
            "beta0": round(beta0, 6),
            "beta_10": round(beta10, 6),
            "beta_20": round(beta20, 6),
            "beta_30": round(beta30, 6),
            "beta_50": round(beta50, 6),
            "se_beta0": round(se_beta0, 6),
            "se_beta10": round(se_beta10, 6),
            "se_beta20": round(se_beta20, 6),
            "se_beta30": round(se_beta30, 6),
            "se_beta50": round(se_beta50, 6),
            "p_beta10": float(p_beta10),
            "p_beta20": float(p_beta20),
            "p_beta30": float(p_beta30),
            "p_beta50": float(p_beta50),
            "source_variance": round(var_s, 8),
            "target_variance": round(var_t, 8),
            "residual_variance": round(scale, 8),
            "log_likelihood": round(loglik, 4),
            "warnings": warnings,
        })

    return pd.DataFrame(results)

