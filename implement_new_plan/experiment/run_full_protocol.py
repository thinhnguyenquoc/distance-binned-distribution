"""
Master Protocol Execution Script for Experiments A, B, C, D.
Adheres 100% to new_approve/new_plan.md.
"""

import os
import sys
import time
import math
from pathlib import Path

# Ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import torch

from implement_new_plan.loss.log1p_mse import log1p_mse_loss
from implement_new_plan.models.od_models import TwoParameterGravity, PairwiseMLP, UrbanGNN
from implement_new_plan.data.dataset import load_raw_city, RawCityData, NODE_FEATURE_COLUMNS
from implement_new_plan.data.urban_graph import build_radius_graph
from implement_new_plan.calibration.source_bins import (
    assign_to_source_bins,
    build_target_dbd,
    calibrate_dbd,
    compute_pure_calibration_ratios,
    apply_pure_dbd_calibration,
)
from implement_new_plan.calibration.tv_noise import generate_exact_tv_noise, derive_noise_seed
from implement_new_plan.calibration.dose_matching import (
    compute_source_binned_dbd,
    compute_rms_log_ratio_dose,
    find_dose_matching_lambda,
    reconstruct_scaled_donor,
    fit_crossed_mixed_effects,
)
from implement_new_plan.training.evaluate import (
    compute_cpc_pair,
    compute_cpc_norm_pair,
    compute_mae_pair,
    compute_mse_pair,
    compute_rmse_pair,
    evaluate_calibration_transfer,
)
from implement_new_plan.calibration.statistical_inference import (
    aggregate_seeds,
    compute_target_city_summary,
    compute_global_target_inference,
    compute_source_city_summary,
    fit_crossed_random_effects,
    compute_scarcity_contrasts,
    compute_scarcity_contrast_target_summary,
    compute_scarcity_contrast_inference,
    fit_scarcity_contrast_mixed_effects,
    compute_gap_recovery,
    fit_scarcity_overall_mixed_effects,
    plot_transfer_heatmap,
)
from implement_new_plan.experiment.master_protocol_runner import (
    set_seed,
    get_city_splits,
    load_source_scaler,
    transform_nodes_with_scaler,
    impute_pop_with_scaler,
    transform_dist_with_scaler,
    train_gravity,
    train_mlp,
    train_gnn,
    CANONICAL_SEEDS,
    FRACTIONS,
    K_GRID,
    EPS_GRID,
    GLOBAL_NOISE_SEED,
)

MODELS = ["gravity_2param", "pairwise_mlp", "urban_gnn"]


def run_full_pipeline(max_cities: Optional[int] = None, output_dir: str = "new_plan_result"):
    print("=" * 80)
    print(f"STARTING FULL MASTER PROTOCOL PIPELINE (new_plan.md) -> {output_dir}")
    print("=" * 80)

    # 1. Load canonical cities
    cities_file = Path("manifests/cities_canonical.txt")
    with open(cities_file, "r") as f:
        all_canonical_cities = [c.strip() for c in f if c.strip()]
    if max_cities is not None:
        cities = all_canonical_cities[:max_cities]
    else:
        cities = all_canonical_cities
    n_cities = len(cities)
    print(f"Running pipeline on {n_cities} source cities (total pool: {len(all_canonical_cities)}).")

    scalers_df = pd.read_csv("manifests/source_feature_scalers.csv")
    bins_df = pd.read_csv("manifests/source_distance_bins.csv")
    donor_map_df = pd.read_csv("manifests/donor_mapping.csv").set_index(["source_city", "target_city"])

    # 2. Pre-load raw city data, positive supports, and radius graphs
    print("Pre-loading raw city representations into memory...")
    raw_cities: Dict[str, RawCityData] = {}
    city_splits: Dict[str, Dict[str, np.ndarray]] = {}
    city_graphs: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    source_scalers: Dict[str, Dict[str, Any]] = {}
    source_bins: Dict[Tuple[str, int], np.ndarray] = {}

    for c in all_canonical_cities:
        rc = load_raw_city(c)
        raw_cities[c] = rc
        city_splits[c] = get_city_splits(rc)
        coords = np.asarray(rc.lon_lat)
        edge_index, edge_dist = build_radius_graph(coords, radius_km=5.0)
        city_graphs[c] = (edge_index, edge_dist)
        source_scalers[c] = load_source_scaler(c, scalers_df)
        for K in K_GRID:
            b_sub = bins_df[(bins_df["source_city"] == c) & (bins_df["K"] == K)].sort_values("bin_id")
            edges = [float(b_sub.iloc[0]["lower_km"])]
            for _, r in b_sub.iterrows():
                edges.append(float(r["upper_km"]) if r["upper_km"] != "inf" else np.inf)
            source_bins[(c, K)] = np.array(edges, dtype=np.float64)

    print("All city structures initialized.")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir = out_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Storage for artifacts
    gravity_params_records = []
    gravity_trace_records = []
    source_city_results_records = []

    # Storage for experiments
    zero_shot_baseline_records = []
    exp_a_records = []
    exp_b_records = []
    master_c_records = []
    exp_d_records = []

    # Check for existing completed cities to resume seamlessly
    completed_cities = set()
    for cp_file in checkpoint_dir.glob("done_*.txt"):
        completed_cities.add(cp_file.stem.replace("done_", ""))

    if completed_cities:
        print(f"Detected {len(completed_cities)} already completed cities in checkpoint: {sorted(completed_cities)}")
        # Load accumulated records from checkpoint parquet/csv if present
        for rec_name, rec_list in [
            ("gravity_params", gravity_params_records),
            ("gravity_trace", gravity_trace_records),
            ("source_city_results", source_city_results_records),
            ("zero_shot_baseline", zero_shot_baseline_records),
            ("exp_a", exp_a_records),
            ("exp_b", exp_b_records),
            ("master_c", master_c_records),
            ("exp_d", exp_d_records),
        ]:
            rec_path = checkpoint_dir / f"{rec_name}.csv"
            if rec_path.exists():
                rec_list.extend(pd.read_csv(rec_path).to_dict("records"))
                print(f"Resumed {len(rec_list)} records from {rec_path.name}")

    # Iterate through source cities
    total_start = time.time()
    for s_idx, source_city in enumerate(cities, 1):
        if source_city in completed_cities:
            print(f"[{s_idx}/{n_cities}] Skipping already completed Source City: {source_city}")
            continue

        s_time = time.time()
        print(f"\n[{s_idx}/{n_cities}] Processing Source City: {source_city}...")
        rc_s = raw_cities[source_city]
        splits_s = city_splits[source_city]
        s_scaler = source_scalers[source_city]
        edge_index_s, edge_dist_s = city_graphs[source_city]

        # Source raw tensors
        s_o = np.asarray(rc_s.pair_o_idx)
        s_d = np.asarray(rc_s.pair_d_idx)
        s_dist = np.asarray(rc_s.dist_km)
        s_trips = np.asarray(rc_s.pair_trips)
        s_pop = impute_pop_with_scaler(np.asarray(rc_s.population), s_scaler)
        s_x = transform_nodes_with_scaler(rc_s.X_raw, s_scaler)
        s_dist_std = transform_dist_with_scaler(s_dist, s_scaler)

        s_o_t = torch.from_numpy(s_o).long()
        s_d_t = torch.from_numpy(s_d).long()
        s_dist_t = torch.from_numpy(s_dist).float()
        s_trips_t = torch.from_numpy(s_trips).float()
        s_pop_t = torch.from_numpy(s_pop).float()
        s_x_t = torch.from_numpy(s_x).float()
        s_dist_std_t = torch.from_numpy(s_dist_std).float().unsqueeze(-1)

        # Dictionary to hold models: model_dict[(model_name, seed, fraction)]
        trained_models = {}

        # -------------------------------------------------------------
        # STEP 1: TRAIN MODELS ACROSS ACTIVE FRACTIONS AND SEEDS
        # -------------------------------------------------------------
        for f in FRACTIONS:
            f_str = f"{f:.2f}"
            train_idx = splits_s[f_str]
            train_o = s_o_t[train_idx]
            train_d = s_d_t[train_idx]
            train_dist = s_dist_t[train_idx]
            train_trips = s_trips_t[train_idx]
            train_dist_std = s_dist_std_t[train_idx]
            train_pop_o = s_pop_t[train_o]
            train_pop_d = s_pop_t[train_d]

            for seed in CANONICAL_SEEDS:
                # 1. Gravity Model
                grav_m, grav_tr = train_gravity(
                    pop_raw_o=train_pop_o,
                    pop_raw_d=train_pop_d,
                    dist_raw=train_dist,
                    true_flow=train_trips,
                    seed=seed,
                    epochs=40,
                    lr=2e-3,
                )
                trained_models[("gravity_2param", seed, f)] = grav_m

                # Record traces for main fraction f=0.30
                if f == 0.30:
                    for tr_row in grav_tr:
                        gravity_trace_records.append({
                            "source_city": source_city,
                            "seed": seed,
                            "epoch": tr_row["epoch"],
                            "G": tr_row["G"],
                            "alpha": tr_row["alpha"],
                            "train_loss": tr_row["train_loss"],
                        })
                    # Loss eval on heldout 70%
                    heldout_idx = splits_s["heldout_30"]
                    with torch.no_grad():
                        ho_pred = grav_m(s_pop_t[s_o_t[heldout_idx]], s_pop_t[s_d_t[heldout_idx]], s_dist_t[heldout_idx])
                        ho_loss = float(log1p_mse_loss(ho_pred, s_trips_t[heldout_idx]).item())
                    gravity_params_records.append({
                        "source_city": source_city,
                        "seed": seed,
                        "stochastic_training": False,
                        "G": round(float(grav_m.G.item()), 6),
                        "alpha": round(float(grav_m.alpha.item()), 6),
                        "loss_eval": round(ho_loss, 6),
                    })

                # 2. Pairwise MLP
                mlp_m = train_mlp(
                    x_o=s_x_t[train_o],
                    x_d=s_x_t[train_d],
                    dist_std=train_dist_std,
                    true_flow=train_trips,
                    seed=seed,
                    epochs=40,
                )
                trained_models[("pairwise_mlp", seed, f)] = mlp_m

                # 3. Urban-GNN
                gnn_m = train_gnn(
                    x_all=s_x_t,
                    spatial_edge_index=edge_index_s,
                    spatial_edge_dist_raw=edge_dist_s,
                    pair_o=train_o,
                    pair_d=train_d,
                    dist_raw=train_dist,
                    pop_raw=s_pop_t,
                    true_flow=train_trips,
                    seed=seed,
                    epochs=40,
                )
                trained_models[("urban_gnn", seed, f)] = gnn_m

        # -------------------------------------------------------------
        # STEP 2: WITHIN-CITY EVALUATION ON 70% HELDOUT (f=0.30)
        # -------------------------------------------------------------
        ho_30_idx = splits_s["heldout_30"]
        y_ho = s_trips[ho_30_idx]
        ho_o = s_o_t[ho_30_idx]
        ho_d = s_d_t[ho_30_idx]
        ho_dist = s_dist_t[ho_30_idx]
        ho_dist_std = s_dist_std_t[ho_30_idx]
        ho_pop_o = s_pop_t[ho_o]
        ho_pop_d = s_pop_t[ho_d]

        for m_name in MODELS:
            is_stoch = (m_name != "gravity_2param")
            for seed in CANONICAL_SEEDS:
                m_obj = trained_models[(m_name, seed, 0.30)]
                with torch.no_grad():
                    if m_name == "gravity_2param":
                        p_ho = m_obj(ho_pop_o, ho_pop_d, ho_dist).numpy()
                    elif m_name == "pairwise_mlp":
                        p_ho = m_obj(s_x_t[ho_o], s_x_t[ho_d], ho_dist_std).numpy()
                    else:
                        p_ho = m_obj(s_x_t, edge_index_s, edge_dist_s, ho_o, ho_d, ho_dist, s_pop_t).numpy()

                cpc = compute_cpc_pair(y_ho, p_ho)
                cpc_n = compute_cpc_norm_pair(y_ho, p_ho)
                mae = compute_mae_pair(y_ho, p_ho)
                mse = compute_mse_pair(y_ho, p_ho)
                rmse = compute_rmse_pair(y_ho, p_ho)

                source_city_results_records.append({
                    "source_city": source_city,
                    "model": m_name,
                    "seed": seed,
                    "stochastic_training": is_stoch,
                    "train_ratio": 0.30,
                    "CPC_eval": round(cpc, 6),
                    "CPC_norm_eval": round(cpc_n, 6),
                    "MAE_eval": round(mae, 4),
                    "MSE_eval": round(mse, 2),
                    "RMSE_eval": round(rmse, 4),
                })

        # -------------------------------------------------------------
        # STEP 3: ZERO-SHOT TRANSFER & EXPERIMENTS TO ALL TARGET CITIES
        # -------------------------------------------------------------
        source_bin_k8 = source_bins[(source_city, 8)]

        for target_city in all_canonical_cities:
            if target_city == source_city:
                continue

            rc_t = raw_cities[target_city]
            edge_index_t, edge_dist_t = city_graphs[target_city]
            t_supp_idx = city_splits[target_city]["support_idx"]

            # Target arrays on positive support
            t_o = np.asarray(rc_t.pair_o_idx)[t_supp_idx]
            t_d = np.asarray(rc_t.pair_d_idx)[t_supp_idx]
            t_dist_raw = np.asarray(rc_t.dist_km)[t_supp_idx]
            t_trips = np.asarray(rc_t.pair_trips)[t_supp_idx]
            true_total = float(np.sum(t_trips))

            # Apply SOURCE scaler to target
            t_pop = impute_pop_with_scaler(np.asarray(rc_t.population), s_scaler)
            t_x = transform_nodes_with_scaler(rc_t.X_raw, s_scaler)
            t_dist_std = transform_dist_with_scaler(t_dist_raw, s_scaler)

            # PyTorch tensors for inference
            t_o_t = torch.from_numpy(t_o).long()
            t_d_t = torch.from_numpy(t_d).long()
            t_dist_raw_t = torch.from_numpy(t_dist_raw).float()
            t_pop_t = torch.from_numpy(t_pop).float()
            t_x_t = torch.from_numpy(t_x).float()
            t_dist_std_t = torch.from_numpy(t_dist_std).float().unsqueeze(-1)
            t_pop_o = t_pop_t[t_o_t]
            t_pop_d = t_pop_t[t_d_t]

            # GNN node embeddings for target (computed once per source model)
            # Compute target DBD on source K=8 bins
            target_p_k8 = build_target_dbd(t_trips, t_dist_raw, source_bin_k8)

            # Precompute perturbed target DBDs for Experiment C across K and eps
            target_p_c_perturbed: Dict[Tuple[int, float, int], Tuple[np.ndarray, float, int]] = {}
            for K_val in K_GRID:
                edges_k = source_bins[(source_city, K_val)]
                p_true_k = build_target_dbd(t_trips, t_dist_raw, edges_k)
                for eps in EPS_GRID:
                    realizations = [0] if (eps == 0.0 or abs(eps) < 1e-12) else list(range(20))
                    for r_id in realizations:
                        noise_seed = derive_noise_seed(
                            global_noise_seed=GLOBAL_NOISE_SEED,
                            source_city=source_city,
                            target_city=target_city,
                            K=K_val,
                            epsilon=eps,
                            realization_id=r_id,
                        )
                        p_tilde, actual_tv, _ = generate_exact_tv_noise(
                            p=p_true_k,
                            epsilon=eps,
                            rng_or_seed=noise_seed,
                            source_city=source_city,
                            target_city=target_city,
                            K=K_val,
                            realization_id=r_id,
                        )
                        target_p_c_perturbed[(K_val, eps, r_id)] = (p_tilde, actual_tv, noise_seed)

            # ---------------------------------------------------------
            # EXPERIMENT B RUNS (5 fractions x 3 models x 3 seeds)
            # ---------------------------------------------------------
            for f in FRACTIONS:
                for m_name in MODELS:
                    for seed in CANONICAL_SEEDS:
                        m_obj = trained_models[(m_name, seed, f)]
                        with torch.no_grad():
                            if m_name == "gravity_2param":
                                pred_before = m_obj(t_pop_o, t_pop_d, t_dist_raw_t).numpy()
                            elif m_name == "pairwise_mlp":
                                pred_before = m_obj(t_x_t[t_o_t], t_x_t[t_d_t], t_dist_std_t).numpy()
                            else:
                                h_t = m_obj.node_encoder(t_x_t, edge_index_t, edge_dist_t)
                                log_grav_t = m_obj.gravity(t_pop_o, t_pop_d, t_dist_raw_t)
                                pred_before = m_obj.decoder(h_t[t_o_t], h_t[t_d_t], torch.log1p(t_dist_raw_t), log_grav_t).numpy()

                        pred_after = calibrate_dbd(
                            pred_flow=pred_before,
                            distance_km=t_dist_raw,
                            bin_edges=source_bin_k8,
                            target_dbd_p=target_p_k8,
                        )

                        metrics_dict = evaluate_calibration_transfer(
                            true_flow=t_trips,
                            pred_before=pred_before,
                            pred_after=pred_after,
                        )

                        # Record for Exp B
                        exp_b_records.append({
                            "source_city": source_city,
                            "target_city": target_city,
                            "model": m_name,
                            "seed": seed,
                            "train_fraction": f,
                            "K": 8,
                            "CPC_before": round(metrics_dict["CPC_before"], 6),
                            "CPC_after": round(metrics_dict["CPC_after"], 6),
                            "delta_CPC": round(metrics_dict["delta_CPC"], 6),
                            "MAE_before": round(metrics_dict["MAE_before"], 4),
                            "MAE_after": round(metrics_dict["MAE_after"], 4),
                            "delta_MAE": round(metrics_dict["delta_MAE"], 4),
                            "MSE_before": round(metrics_dict["MSE_before"], 2),
                            "MSE_after": round(metrics_dict["MSE_after"], 2),
                            "delta_MSE": round(metrics_dict["delta_MSE"], 2),
                            "RMSE_before": round(metrics_dict["RMSE_before"], 4),
                            "RMSE_after": round(metrics_dict["RMSE_after"], 4),
                            "delta_RMSE": round(metrics_dict["delta_RMSE"], 4),
                        })

                        # If f == 0.30: Main setting for zero_shot_baseline, Exp A, Exp C, Exp D
                        if f == 0.30:
                            pred_total = float(np.sum(pred_before))
                            r_vol = float(pred_total / true_total)

                            zero_shot_baseline_records.append({
                                "source_city": source_city,
                                "target_city": target_city,
                                "model": m_name,
                                "seed": seed,
                                "CPC_before": round(metrics_dict["CPC_before"], 6),
                                "CPC_norm_before": round(metrics_dict["CPC_norm_before"], 6),
                                "MAE_before": round(metrics_dict["MAE_before"], 4),
                                "MSE_before": round(metrics_dict["MSE_before"], 2),
                                "RMSE_before": round(metrics_dict["RMSE_before"], 4),
                                "pred_total": round(pred_total, 2),
                                "true_total": round(true_total, 2),
                                "R_vol": round(r_vol, 6),
                            })

                            # Coverage audit on K=8
                            bin_ids_k8 = assign_to_source_bins(t_dist_raw, source_bin_k8)
                            q_k8 = np.zeros(8, dtype=np.float64)
                            for b in range(8):
                                mask = (bin_ids_k8 == b)
                                if mask.any():
                                    q_k8[b] = float(np.sum(pred_before[mask])) / pred_total
                            pos_b = (q_k8 > 0.0)
                            cov_mass = float(np.sum(target_p_k8[pos_b]))
                            uncov_mass = float(1.0 - cov_mass)
                            n_uncov_bins = int(np.sum((q_k8 == 0) & (target_p_k8 > 0)))

                            exp_a_records.append({
                                "source_city": source_city,
                                "target_city": target_city,
                                "model": m_name,
                                "seed": seed,
                                "train_fraction": 0.30,
                                "K": 8,
                                "epsilon": 0.0,
                                "CPC_before": round(metrics_dict["CPC_before"], 6),
                                "CPC_after": round(metrics_dict["CPC_after"], 6),
                                "delta_CPC": round(metrics_dict["delta_CPC"], 6),
                                "CPC_norm_before": round(metrics_dict["CPC_norm_before"], 6),
                                "CPC_norm_after": round(metrics_dict["CPC_norm_after"], 6),
                                "delta_CPC_norm": round(metrics_dict["delta_CPC_norm"], 6),
                                "MAE_before": round(metrics_dict["MAE_before"], 4),
                                "MAE_after": round(metrics_dict["MAE_after"], 4),
                                "delta_MAE": round(metrics_dict["delta_MAE"], 4),
                                "MSE_before": round(metrics_dict["MSE_before"], 2),
                                "MSE_after": round(metrics_dict["MSE_after"], 2),
                                "delta_MSE": round(metrics_dict["delta_MSE"], 2),
                                "RMSE_before": round(metrics_dict["RMSE_before"], 4),
                                "RMSE_after": round(metrics_dict["RMSE_after"], 4),
                                "delta_RMSE": round(metrics_dict["delta_RMSE"], 4),
                                "covered_target_mass": round(cov_mass, 6),
                                "uncovered_target_mass": round(uncov_mass, 6),
                                "n_uncovered_bins": n_uncov_bins,
                            })

                            # -------------------------------------------------
                            # EXPERIMENT D: DOSE-MATCHED SCALED DONOR CONTROL
                            # -------------------------------------------------
                            donor_city = donor_map_df.loc[(source_city, target_city), "donor_city"]
                            rc_d = raw_cities[donor_city]
                            d_supp_idx = city_splits[donor_city]["support_idx"]
                            d_dist_raw = np.asarray(rc_d.dist_km)[d_supp_idx]
                            d_trips = np.asarray(rc_d.pair_trips)[d_supp_idx]
                            donor_p_k8 = build_target_dbd(d_trips, d_dist_raw, source_bin_k8)

                            root_res = find_dose_matching_lambda(
                                p_s_target=target_p_k8,
                                p_s_donor=donor_p_k8,
                                q_s_target=q_k8,
                                tolerance=1e-6,
                                eps=1e-9,
                                source_city=source_city,
                                target_city=target_city,
                                donor_city=donor_city,
                            )

                            p_donor_control = root_res["p_scaled_donor"]
                            pred_donor_after = calibrate_dbd(
                                pred_flow=pred_before,
                                distance_km=t_dist_raw,
                                bin_edges=source_bin_k8,
                                target_dbd_p=p_donor_control,
                            )
                            cpc_donor = compute_cpc_pair(t_trips, pred_donor_after)
                            cpc_target = metrics_dict["CPC_after"]

                            exp_d_records.append({
                                "source_city": source_city,
                                "target_city": target_city,
                                "donor_city": donor_city,
                                "model": m_name,
                                "seed": seed,
                                "K": 8,
                                "lambda_low": round(root_res["lambda_low"], 6),
                                "lambda_high": round(root_res["lambda_high"], 6),
                                "lambda_star": round(root_res["lambda_star"], 6),
                                "target_dose": round(root_res["target_dose"], 6),
                                "donor_raw_dose": round(root_res["donor_raw_dose"], 6),
                                "donor_scaled_dose": round(root_res["donor_scaled_dose"], 6),
                                "dose_error": round(root_res["dose_error"], 8),
                                "root_iterations": root_res["root_iterations"],
                                "root_converged": root_res["root_converged"],
                                "CPC_before": round(metrics_dict["CPC_before"], 6),
                                "CPC_target": round(cpc_target, 6),
                                "CPC_donor_control": round(cpc_donor, 6),
                                "delta_CPC_target": round(cpc_target - metrics_dict["CPC_before"], 6),
                                "delta_CPC_donor_control": round(cpc_donor - metrics_dict["CPC_before"], 6),
                            })

                            # -------------------------------------------------
                            # EXPERIMENT C: MASTER SENSITIVITY GRID
                            # -------------------------------------------------
                            for K_val in K_GRID:
                                edges_k = source_bins[(source_city, K_val)]
                                bin_ids_k = assign_to_source_bins(t_dist_raw, edges_k)

                                q_k = np.zeros(K_val, dtype=np.float64)
                                for b in range(K_val):
                                    mask = (bin_ids_k == b)
                                    if mask.any():
                                        q_k[b] = float(np.sum(pred_before[mask])) / pred_total

                                for eps in EPS_GRID:
                                    realizations = [0] if (eps == 0.0 or abs(eps) < 1e-12) else list(range(20))

                                    for r_id in realizations:
                                        p_tilde, actual_tv, noise_seed = target_p_c_perturbed[(K_val, eps, r_id)]

                                        pred_cal_c = calibrate_dbd(
                                            pred_flow=pred_before,
                                            distance_km=t_dist_raw,
                                            bin_edges=edges_k,
                                            target_dbd_p=p_tilde,
                                        )

                                        m_c = evaluate_calibration_transfer(
                                            true_flow=t_trips,
                                            pred_before=pred_before,
                                            pred_after=pred_cal_c,
                                        )

                                        pos_b_c = (q_k > 0.0)
                                        cov_c = float(np.sum(p_tilde[pos_b_c]))
                                        uncov_c = float(1.0 - cov_c)
                                        n_uncov_c = int(np.sum((q_k == 0) & (p_tilde > 0)))

                                        master_c_records.append({
                                            "source_city": source_city,
                                            "target_city": target_city,
                                            "model": m_name,
                                            "model_seed": seed,
                                            "K": K_val,
                                            "epsilon": eps,
                                            "realization_id": r_id,
                                            "noise_seed": noise_seed,
                                            "actual_TV": round(actual_tv, 6),
                                            "CPC_before": round(m_c["CPC_before"], 6),
                                            "CPC_after": round(m_c["CPC_after"], 6),
                                            "delta_CPC": round(m_c["delta_CPC"], 6),
                                            "CPC_norm_before": round(m_c["CPC_norm_before"], 6),
                                            "CPC_norm_after": round(m_c["CPC_norm_after"], 6),
                                            "delta_CPC_norm": round(m_c["delta_CPC_norm"], 6),
                                            "MAE_before": round(m_c["MAE_before"], 4),
                                            "MAE_after": round(m_c["MAE_after"], 4),
                                            "delta_MAE": round(m_c["delta_MAE"], 4),
                                            "MSE_before": round(m_c["MSE_before"], 2),
                                            "MSE_after": round(m_c["MSE_after"], 2),
                                            "delta_MSE": round(m_c["delta_MSE"], 2),
                                            "RMSE_before": round(m_c["RMSE_before"], 4),
                                            "RMSE_after": round(m_c["RMSE_after"], 4),
                                            "delta_RMSE": round(m_c["delta_RMSE"], 4),
                                            "covered_target_mass": round(cov_c, 6),
                                            "uncovered_target_mass": round(uncov_c, 6),
                                            "n_uncovered_bins": n_uncov_c,
                                        })

        print(f"Finished {source_city} in {time.time() - s_time:.2f}s")

        # Save incremental checkpoint
        for rec_name, rec_list in [
            ("gravity_params", gravity_params_records),
            ("gravity_trace", gravity_trace_records),
            ("source_city_results", source_city_results_records),
            ("zero_shot_baseline", zero_shot_baseline_records),
            ("exp_a", exp_a_records),
            ("exp_b", exp_b_records),
            ("master_c", master_c_records),
            ("exp_d", exp_d_records),
        ]:
            if rec_list:
                pd.DataFrame(rec_list).to_csv(checkpoint_dir / f"{rec_name}.csv", index=False)

        (checkpoint_dir / f"done_{source_city}.txt").write_text("OK", encoding="utf-8")
        print(f"Checkpoint saved for {source_city}.")

    print(f"\nALL CITIES EVALUATION COMPLETED in {time.time() - total_start:.2f}s!")
    print("Exporting raw results and executing 4-tier statistical inference...")

    # -----------------------------------------------------------------
    # EXPORT PRIMARY ARTIFACTS AND TRACES
    # -----------------------------------------------------------------

    # 1. source_city_results.csv & source_city_results_mean.csv
    df_src = pd.DataFrame(source_city_results_records)
    df_src.to_csv(out_dir / "source_city_results.csv", index=False)
    df_src_mean = (
        df_src.groupby(["source_city", "model"])[["CPC_eval", "CPC_norm_eval", "MAE_eval", "MSE_eval", "RMSE_eval"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    df_src_mean.columns = [f"{c[0]}_{c[1]}" if c[1] else c[0] for c in df_src_mean.columns]
    df_src_mean.to_csv(out_dir / "source_city_results_mean.csv", index=False)
    print(f"Saved source_city_results.csv and source_city_results_mean.csv to {out_dir}")

    # 2. gravity_parameters.csv & gravity_training_trace.csv
    pd.DataFrame(gravity_params_records).to_csv(manifests_dir / "gravity_parameters.csv", index=False)
    pd.DataFrame(gravity_trace_records).to_csv(manifests_dir / "gravity_training_trace.csv", index=False)
    print(f"Saved gravity_parameters.csv and gravity_training_trace.csv to {manifests_dir}")

    # 3. zero_shot_baseline.csv
    df_zs = pd.DataFrame(zero_shot_baseline_records)
    df_zs.to_csv(out_dir / "zero_shot_baseline.csv", index=False)
    print(f"Saved zero_shot_baseline.csv to {out_dir}")

    # 4. calibration_results.csv (Experiment A) & calibration_results_mean.csv
    df_exp_a = pd.DataFrame(exp_a_records)
    df_exp_a.to_csv(out_dir / "calibration_results.csv", index=False)
    df_exp_a_mean = aggregate_seeds(df_exp_a)
    df_exp_a_mean.to_csv(out_dir / "calibration_results_mean.csv", index=False)
    print(f"Saved calibration_results.csv and calibration_results_mean.csv to {out_dir}")

    # Generate heatmaps for all models
    for m in MODELS:
        try:
            plot_transfer_heatmap(
                df_exp_a_mean,
                output_path=figures_dir / f"transfer_heatmap_{m}.png",
                model_name=m
            )
        except Exception as e:
            print(f"Warning: could not plot heatmap for {m}: {e}")

    # Tier B, C, D inference on Experiment A
    if n_cities == 50:
        df_target_sum = compute_target_city_summary(df_exp_a_mean, metric_col="delta_CPC")
        df_target_sum.to_csv(out_dir / "target_city_summary.csv", index=False)
        df_target_inf = compute_global_target_inference(df_target_sum)
        df_target_inf.to_csv(out_dir / "target_level_inference.csv", index=False)
        df_source_sum = compute_source_city_summary(df_exp_a_mean, metric_col="delta_CPC")
        df_source_sum.to_csv(out_dir / "source_city_summary.csv", index=False)
        df_crossed, df_diag = fit_crossed_random_effects(df_exp_a_mean, metric_col="delta_CPC")
        df_crossed.to_csv(out_dir / "crossed_effects_results.csv", index=False)
        df_diag.to_csv(out_dir / "crossed_effects_diagnostics.csv", index=False)
        print(f"Saved Experiment A statistical inference files to {out_dir}")
    else:
        print(f"Skipping 50-city statistical inference for Experiment A (running on subset: {n_cities}/50 cities).")

    # 5. scarcity_results.csv (Experiment B) and related analyses
    df_exp_b = pd.DataFrame(exp_b_records)
    df_exp_b.to_csv(out_dir / "scarcity_results.csv", index=False)
    df_b_mean = (
        df_exp_b.groupby(["source_city", "target_city", "model", "train_fraction"])[["CPC_before", "CPC_after", "delta_CPC"]]
        .mean()
        .reset_index()
        .rename(columns={
            "CPC_before": "CPC_before_mean",
            "CPC_after": "CPC_after_mean",
            "delta_CPC": "delta_CPC_mean"
        })
    )
    df_b_mean.to_csv(out_dir / "scarcity_results_mean.csv", index=False)

    df_scarcity_target = (
        df_b_mean.groupby(["target_city", "model", "train_fraction"])["delta_CPC_mean"]
        .agg(mean_delta_CPC="mean", median_delta_CPC="median", n_sources="count", positive_sources=lambda x: int((x > 0).sum()))
        .reset_index()
    )
    df_scarcity_target.to_csv(out_dir / "scarcity_target_summary.csv", index=False)

    df_gap = compute_gap_recovery(df_b_mean)
    df_gap.to_csv(out_dir / "scarcity_gap_recovery.csv", index=False)

    if n_cities == 50:
        df_contrasts = compute_scarcity_contrasts(df_b_mean, metric_col="delta_CPC_mean")
        df_c_target_sum = compute_scarcity_contrast_target_summary(df_contrasts)
        df_c_target_sum.to_csv(out_dir / "scarcity_contrast_target_summary.csv", index=False)
        df_c_inf = compute_scarcity_contrast_inference(df_c_target_sum)
        df_c_inf.to_csv(out_dir / "scarcity_contrast_inference.csv", index=False)
        df_c_mixed = fit_scarcity_contrast_mixed_effects(df_contrasts)
        df_c_mixed.to_csv(out_dir / "scarcity_contrast_mixed_effects.csv", index=False)
        df_scarcity_overall_mixed = fit_scarcity_overall_mixed_effects(df_b_mean)
        df_scarcity_overall_mixed.to_csv(out_dir / "scarcity_overall_mixed_effects.csv", index=False)
        print(f"Saved Experiment B scarcity results and contrast inferences to {out_dir}")
    else:
        print(f"Skipping 50-city scarcity contrast inferences (running on subset: {n_cities}/50 cities).")

    # 6. noise_robustness_results.csv (Experiment C Master Grid) and Derived Views C1, C2, C3
    df_master_c = pd.DataFrame(master_c_records)
    df_master_c.to_csv(out_dir / "noise_robustness_results.csv", index=False)

    # Derived View C1 (Resolution Sensitivity: epsilon == 0)
    df_c1 = (
        df_master_c[df_master_c["epsilon"] == 0.0]
        .groupby(["model", "K"])[["delta_CPC", "delta_CPC_norm", "delta_MAE", "delta_MSE", "delta_RMSE"]]
        .agg(["mean", "median", "std"])
        .reset_index()
    )
    df_c1.to_csv(out_dir / "experiment_c1_resolution_summary.csv", index=False)

    # Derived View C2 (Error Robustness: K == 8)
    df_c2 = (
        df_master_c[df_master_c["K"] == 8]
        .groupby(["model", "epsilon"])[["delta_CPC", "delta_CPC_norm", "delta_MAE", "delta_MSE", "delta_RMSE"]]
        .agg(["mean", "median", "std"])
        .reset_index()
    )
    df_c2.to_csv(out_dir / "experiment_c2_error_summary.csv", index=False)

    # Derived View C3 (Interaction: K x epsilon)
    df_c3 = (
        df_master_c.groupby(["model", "K", "epsilon"])[["delta_CPC", "delta_CPC_norm", "delta_MAE", "delta_MSE", "delta_RMSE"]]
        .agg(["mean", "median", "std"])
        .reset_index()
    )
    df_c3.to_csv(out_dir / "experiment_c3_interaction_summary.csv", index=False)
    print(f"Saved Experiment C noise robustness master grid and derived views C1, C2, C3 to {out_dir}")

    # 7. structural_control_results.csv (Experiment D) and inferences
    df_exp_d = pd.DataFrame(exp_d_records)
    df_exp_d.to_csv(out_dir / "structural_control_results.csv", index=False)

    df_d_mean = (
        df_exp_d.groupby(["source_city", "target_city", "model", "K"])
        .agg(
            target_delta_CPC_mean=("delta_CPC_target", "mean"),
            control_delta_CPC_mean=("delta_CPC_donor_control", "mean"),
        )
        .reset_index()
    )
    df_d_mean["structural_advantage_delta"] = df_d_mean["target_delta_CPC_mean"] - df_d_mean["control_delta_CPC_mean"]
    df_d_mean.to_csv(out_dir / "structural_control_mean.csv", index=False)

    df_d_target_sum = (
        df_d_mean.groupby(["target_city", "model"])["structural_advantage_delta"]
        .agg(
            mean_structural_advantage="mean",
            median_structural_advantage="median",
            n_sources="count",
            positive_sources=lambda x: int((x > 0).sum())
        )
        .reset_index()
    )
    df_d_target_sum.to_csv(out_dir / "structural_control_target_summary.csv", index=False)

    # Global inference for Experiment D
    if n_cities == 50:
        from scipy.stats import wilcoxon
        d_inf_records = []
        for m in MODELS:
            sub_d = df_d_target_sum[df_d_target_sum["model"] == m]
            H_vals = sub_d["mean_structural_advantage"].values
            n_t = len(H_vals)

            mean_H = float(np.mean(H_vals))
            median_H = float(np.median(H_vals))
            q75, q25 = np.percentile(H_vals, [75, 25])
            iqr_H = float(q75 - q25)
            pos_t = int(np.sum(H_vals > 0))
            pos_f = float(pos_t / n_t)

            rng = np.random.RandomState(42)
            b_means = [np.mean(rng.choice(H_vals, size=n_t, replace=True)) for _ in range(10000)]
            b_low = float(np.percentile(b_means, 2.5))
            b_high = float(np.percentile(b_means, 97.5))

            if np.all(H_vals == 0) or len(np.unique(H_vals)) <= 1:
                w_stat, w_p = 0.0, 1.0
            else:
                res_w = wilcoxon(x=H_vals, alternative="two-sided", zero_method="wilcox", correction=False, method="auto")
                w_stat, w_p = float(res_w.statistic), float(res_w.pvalue)

            pair_d_m = df_d_mean[df_d_mean["model"] == m]
            beta0, se, ci_l, ci_h, var_s, var_t, var_eps = fit_crossed_mixed_effects(
                pair_d_m, outcome_col="structural_advantage_delta"
            )

            d_inf_records.append({
                "model": m,
                "mean_H": round(mean_H, 6),
                "median_H": round(median_H, 6),
                "IQR_H": round(iqr_H, 6),
                "bootstrap_ci_low": round(b_low, 6),
                "bootstrap_ci_high": round(b_high, 6),
                "wilcoxon_stat": round(w_stat, 2),
                "wilcoxon_p_raw": float(w_p),
                "positive_targets": pos_t,
                "positive_target_fraction": round(pos_f, 6),
                "n_targets": n_t,
                "mixed_beta0": round(beta0, 6),
                "mixed_se": round(se, 6),
                "mixed_ci_low": round(ci_l, 6),
                "mixed_ci_high": round(ci_h, 6),
                "source_variance": round(var_s, 6),
                "target_variance": round(var_t, 6),
                "residual_variance": round(var_eps, 6),
            })

        # Holm correction
        p_raws = [r["wilcoxon_p_raw"] for r in d_inf_records]
        order = np.argsort(p_raws)
        p_holm = np.zeros(len(p_raws), dtype=np.float64)
        cum_max = 0.0
        for rank, orig_idx in enumerate(order):
            adj = min(1.0, p_raws[orig_idx] * (len(p_raws) - rank))
            cum_max = max(cum_max, adj)
            p_holm[orig_idx] = cum_max

        for i, r in enumerate(d_inf_records):
            r["wilcoxon_p_holm"] = float(p_holm[i])

        pd.DataFrame(d_inf_records).to_csv(out_dir / "structural_control_inference.csv", index=False)
        print(f"Saved Experiment D structural control outputs to {out_dir}")
    else:
        print(f"Skipping 50-city structural control inference for Experiment D (running on subset: {n_cities}/50 cities).")

    print("\n" + "=" * 80)
    print(f"MASTER EXECUTION AND STATISTICAL PIPELINE SUCCESSFULLY FINISHED! Output: {out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    max_c = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else None
    out_dir_arg = sys.argv[2] if len(sys.argv) > 2 else "new_plan_result"
    run_full_pipeline(max_cities=max_c, output_dir=out_dir_arg)
