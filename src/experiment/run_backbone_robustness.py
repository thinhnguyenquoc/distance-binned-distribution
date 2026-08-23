"""
Backbone Robustness Evaluation Experiment.
Evaluates the Calibration Operator across multiple zero-shot backbones:
    1. Classical 2-Parameter Gravity Baseline: T_ij^grav = exp(G) * P_i * P_j * D_ij^(-alpha)
    2. Proposed Gravity-Informed Urban GNN: f_theta(X_i, X_j, D_ij, T_ij^grav)

For each backbone b, computes:
    - Delta R_b (CPC_inter)
    - Delta RMSE
    - Delta Spearman rho_s
Across untouched Confirmatory Folds 2-5 (n=40) and Full Out-of-fold benchmark (N=50).
"""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import sys
import json
import torch
import numpy as np
from pathlib import Path
from scipy import stats
from typing import Dict, Any, List

# Ensure repo root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.city_splits import generate_5fold_splits
from src.data.dataset import load_city, load_raw_city
from src.data.yd_extractor import compute_kbin_edges, extract_yd_kbins
from src.calibration.bin_calibration import calibrate_kbins
from src.training.evaluate import compute_cpc_pair, compute_spearman_pair


def fit_gravity_parameters(train_cities: List[str], data_root: str = "data") -> tuple[float, float]:
    """Fits global classical gravity parameters G and alpha via log-linear regression on training cities."""
    log_pi_pj = []
    log_dist = []
    log_flow = []

    for c in train_cities:
        raw = load_raw_city(c, data_root=data_root)
        dist_km = raw.dist_km
        mask = (raw.pair_o_idx.numpy() != raw.pair_d_idx.numpy()) & (dist_km > 0.0) & (raw.pair_trips.numpy() > 0)
        if np.sum(mask) == 0:
            continue
        p = raw.population.numpy()
        p_i = np.clip(p[raw.pair_o_idx.numpy()[mask]], 1.0, None)
        p_j = np.clip(p[raw.pair_d_idx.numpy()[mask]], 1.0, None)
        d = np.clip(dist_km[mask], 0.1, None)
        f = raw.pair_trips.numpy()[mask]

        log_pi_pj.extend(np.log(p_i) + np.log(p_j))
        log_dist.extend(np.log(d))
        log_flow.extend(np.log(f))

    # OLS: log_flow = G + 1.0 * log_pi_pj - alpha * log_dist
    y = np.array(log_flow) - np.array(log_pi_pj)
    X = np.column_stack([np.ones(len(y)), -np.array(log_dist)])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    G = float(beta[0])
    alpha = float(beta[1])
    return G, alpha


def run_backbone_robustness(
    data_root: str = "data",
    output_dir: str = "results/tables",
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    splits = generate_5fold_splits(data_root=data_root)

    results_file = Path("results/5fold_results.json")
    if not results_file.exists():
        raise FileNotFoundError(f"Missing {results_file}. Run 5-fold experiment first.")

    with open(results_file, "r") as f:
        full_res = json.load(f)

    city_map = {r["city"]: r for r in full_res["city_level_results"]}

    results_by_backbone: Dict[str, List[Dict[str, Any]]] = {
        "classical_gravity": [],
        "urban_gnn": [],
    }

    print("Running Backbone Robustness across 5 folds...")

    for fold_id in range(1, 6):
        train_cities = splits[fold_id]["train"]
        test_cities = splits[fold_id]["test"]

        # 1. Fit Classical Gravity on Fold training cities
        G_fit, alpha_fit = fit_gravity_parameters(train_cities, data_root=data_root)
        print(f"Fold {fold_id} Classical Gravity: G={G_fit:.3f}, alpha={alpha_fit:.3f}")

        # Compute K=8 bin edges from training cities
        bin_edges, _ = compute_kbin_edges(train_cities, K=8, data_root=data_root)

        for city_name in test_cities:
            raw = load_raw_city(city_name, data_root=data_root)
            existing_r = city_map.get(city_name)
            if existing_r is None:
                continue

            dist_km = raw.dist_km
            inter_mask = (raw.pair_o_idx.numpy() != raw.pair_d_idx.numpy()) & (dist_km > 0.0)
            t_true_inter = raw.pair_trips.numpy()[inter_mask]

            # Extract Oracle Target Y_D
            yd_target = extract_yd_kbins(dist_km, raw.pair_trips.numpy(), bin_edges, inter_mask)

            # --- Backbone 1: Classical Gravity ---
            p = raw.population.numpy()
            p_i = np.clip(p[raw.pair_o_idx.numpy()], 1.0, None)
            p_j = np.clip(p[raw.pair_d_idx.numpy()], 1.0, None)
            d = np.clip(dist_km, 0.1, None)
            t_grav = np.exp(G_fit) * p_i * p_j * (d ** (-alpha_fit))
            t_grav_inter = t_grav[inter_mask]

            # Evaluate M0_grav
            m0_cpc_grav = float(compute_cpc_pair(t_true_inter, t_grav_inter))
            m0_rmse_grav = float(np.sqrt(np.mean((t_true_inter - t_grav_inter) ** 2)))
            m0_spr_grav = float(compute_spearman_pair(t_true_inter, t_grav_inter))

            # Apply K=8 calibration on Gravity
            t_grav_cal = calibrate_kbins(t_grav, dist_km, inter_mask, yd_target, bin_edges, q=1.0)
            t_grav_cal_inter = t_grav_cal[inter_mask]

            m1_cpc_grav = float(compute_cpc_pair(t_true_inter, t_grav_cal_inter))
            m1_rmse_grav = float(np.sqrt(np.mean((t_true_inter - t_grav_cal_inter) ** 2)))
            m1_spr_grav = float(compute_spearman_pair(t_true_inter, t_grav_cal_inter))

            results_by_backbone["classical_gravity"].append({
                "city": city_name,
                "fold": fold_id,
                "m0_cpc_inter": m0_cpc_grav,
                "m1_cpc_inter": m1_cpc_grav,
                "delta_r": m1_cpc_grav - m0_cpc_grav,
                "m0_rmse_inter": m0_rmse_grav,
                "m1_rmse_inter": m1_rmse_grav,
                "delta_rmse": m1_rmse_grav - m0_rmse_grav,
                "m0_spearman_inter": m0_spr_grav,
                "m1_spearman_inter": m1_spr_grav,
                "delta_spearman": m1_spr_grav - m0_spr_grav,
            })

            # --- Backbone 2: Gravity-Informed Urban GNN (Main) ---
            m0_gnn = existing_r["M0"]
            m1_gnn = existing_r.get("M1_city_oracle_obs", existing_r.get("M1_real_plus", {}))
            
            m0_cpc_gnn = m0_gnn["cpc_inter"]
            m1_cpc_gnn = m1_gnn["cpc_inter"]
            delta_gnn = m1_cpc_gnn - m0_cpc_gnn

            m0_rmse_gnn = m0_gnn.get("rmse_inter", 0.0)
            m1_rmse_gnn = m1_gnn.get("rmse_inter", 0.0)
            m0_spr_gnn = m0_gnn.get("spearman_inter", 0.0)
            m1_spr_gnn = m1_gnn.get("spearman_inter", 0.0)

            results_by_backbone["urban_gnn"].append({
                "city": city_name,
                "fold": fold_id,
                "m0_cpc_inter": m0_cpc_gnn,
                "m1_cpc_inter": m1_cpc_gnn,
                "delta_r": delta_gnn,
                "m0_rmse_inter": m0_rmse_gnn,
                "m1_rmse_inter": m1_rmse_gnn,
                "delta_rmse": m1_rmse_gnn - m0_rmse_gnn,
                "m0_spearman_inter": m0_spr_gnn,
                "m1_spearman_inter": m1_spr_gnn,
                "delta_spearman": m1_spr_gnn - m0_spr_gnn,
            })

    # Summarize across Confirmatory Fold 2-5 (n=40) and Full (n=50)
    def summarize_backbone(records: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
        conf_recs = [r for r in records if r["fold"] in [2, 3, 4, 5]]
        all_recs = records

        def get_block(sub: List[Dict[str, Any]]):
            n = len(sub)
            m0_cpc = np.array([r["m0_cpc_inter"] for r in sub])
            m1_cpc = np.array([r["m1_cpc_inter"] for r in sub])
            dr = np.array([r["delta_r"] for r in sub])
            d_rmse = np.array([r["delta_rmse"] for r in sub])
            d_sp = np.array([r["delta_spearman"] for r in sub])

            # Stratified bootstrap CI
            delta_by_fold = {}
            for f in (range(1, 6) if n == 50 else range(2, 6)):
                delta_by_fold[f] = [r["delta_r"] for r in sub if r["fold"] == f]

            rng = np.random.default_rng(42)
            boot_means = []
            for _ in range(5000):
                samp = []
                for f, vals in delta_by_fold.items():
                    if len(vals) > 0:
                        samp.extend(rng.choice(vals, size=len(vals), replace=True))
                boot_means.append(np.mean(samp))
            ci_l, ci_h = np.percentile(boot_means, [2.5, 97.5])

            _, w_p = stats.wilcoxon(m1_cpc, m0_cpc, alternative="greater")

            return {
                "n": n,
                "m0_cpc_mean": float(np.mean(m0_cpc)),
                "m0_cpc_std": float(np.std(m0_cpc, ddof=1)),
                "m1_cpc_mean": float(np.mean(m1_cpc)),
                "m1_cpc_std": float(np.std(m1_cpc, ddof=1)),
                "delta_r_mean": float(np.mean(dr)),
                "delta_r_std": float(np.std(dr, ddof=1)),
                "delta_r_median": float(np.median(dr)),
                "delta_r_iqr": float(np.percentile(dr, 75) - np.percentile(dr, 25)),
                "bootstrap_95_ci": [float(ci_l), float(ci_h)],
                "p_improved": float(np.mean(dr > 0)),
                "n_improved": f"{int(np.sum(dr > 0))}/{n}",
                "wilcoxon_p": float(w_p),
                "delta_rmse_mean": float(np.mean(d_rmse)),
                "delta_spearman_mean": float(np.mean(d_sp)),
            }

        return {
            "backbone": label,
            "confirmatory_fold2_5": get_block(conf_recs),
            "full_50_cities": get_block(all_recs),
        }

    summary = {
        "classical_gravity": summarize_backbone(results_by_backbone["classical_gravity"], "Classical 2-Parameter Gravity"),
        "urban_gnn": summarize_backbone(results_by_backbone["urban_gnn"], "Gravity-Informed Urban GNN"),
    }

    # Generate Markdown Table
    t7_md = []
    t7_md.append("# Backbone Robustness — Marginal Value of Calibration Across Model Architectures")
    t7_md.append("")
    t7_md.append("> **Evaluation Scope**: Assesses whether distance-binned aggregate information ($Y_D^{\\text{target}}$) improves interzonal reconstruction across different zero-shot model families.")
    t7_md.append("")
    t7_md.append("## Part A: Confirmatory Evaluation Set (Folds 2–5, $n=40$ Cities)")
    t7_md.append("| Backbone Architecture | Zero-Shot $M_0$ CPC | Calibrated $M_1$ CPC | Marginal Gain $\\Delta R$ | 95% Fold-Stratified Bootstrap CI | $P(\\Delta R > 0)$ | Wilcoxon $p$ | $\\Delta \\text{RMSE}$ |")
    t7_md.append("|---|---|---|---|---|---|---|---|")

    for k, v in summary.items():
        b_name = v["backbone"]
        c_stats = v["confirmatory_fold2_5"]
        m0_str = f"{c_stats['m0_cpc_mean']:.4f} +- {c_stats['m0_cpc_std']:.4f}"
        m1_str = f"**{c_stats['m1_cpc_mean']:.4f} +- {c_stats['m1_cpc_std']:.4f}**"
        dr_str = f"**{c_stats['delta_r_mean']:+.4f} +- {c_stats['delta_r_std']:.4f}**"
        ci_str = f"[{c_stats['bootstrap_95_ci'][0]:+.4f}, {c_stats['bootstrap_95_ci'][1]:+.4f}]"
        p_imp = f"{c_stats['p_improved']*100:.1f}% ({c_stats['n_improved']})"
        w_p = f"{c_stats['wilcoxon_p']:.4e}"
        rmse_str = f"{c_stats['delta_rmse_mean']:+.4f}"
        t7_md.append(f"| **{b_name}** | {m0_str} | {m1_str} | {dr_str} | {ci_str} | {p_imp} | p = {w_p} | {rmse_str} |")

    t7_md.append("")
    t7_md.append("## Part B: Full Descriptive Set ($N=50$ Cities)")
    t7_md.append("| Backbone Architecture | Zero-Shot $M_0$ CPC | Calibrated $M_1$ CPC | Marginal Gain $\\Delta R$ | 95% Fold-Stratified Bootstrap CI | $P(\\Delta R > 0)$ | Wilcoxon $p$ |")
    t7_md.append("|---|---|---|---|---|---|---|")

    for k, v in summary.items():
        b_name = v["backbone"]
        a_stats = v["full_50_cities"]
        m0_str = f"{a_stats['m0_cpc_mean']:.4f} +- {a_stats['m0_cpc_std']:.4f}"
        m1_str = f"**{a_stats['m1_cpc_mean']:.4f} +- {a_stats['m1_cpc_std']:.4f}**"
        dr_str = f"**{a_stats['delta_r_mean']:+.4f} +- {a_stats['delta_r_std']:.4f}**"
        ci_str = f"[{a_stats['bootstrap_95_ci'][0]:+.4f}, {a_stats['bootstrap_95_ci'][1]:+.4f}]"
        p_imp = f"{a_stats['p_improved']*100:.1f}% ({a_stats['n_improved']})"
        w_p = f"{a_stats['wilcoxon_p']:.4e}"
        t7_md.append(f"| **{b_name}** | {m0_str} | {m1_str} | {dr_str} | {ci_str} | {p_imp} | p = {w_p} |")

    t7_md_content = "\n".join(t7_md)
    with open(Path(output_dir) / "table7_backbone_robustness.md", "w", encoding="utf-8") as f:
        f.write(t7_md_content)

    with open("results/backbone_robustness_results.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_city_records": results_by_backbone}, f, indent=2)

    print(f"Backbone robustness table generated at {output_dir}/table7_backbone_robustness.md")
    return summary


if __name__ == "__main__":
    run_backbone_robustness()
