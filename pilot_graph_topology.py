"""
Graph Topology Pilot Experiment.

Systematically evaluates and compares urban spatial graph constructions for G^urban:
    1. k-NN with k = 5
    2. k-NN with k = 10
    3. k-NN with k = 20
    4. Radius graph with normalized radius (median neighbor distance)

Trains on source cities (Raleigh + Denver) -> Evaluates zero-shot on held-out city (Philadelphia).
Logs CPC, RMSE-log1p, training loss, and runtime to determine and lock the optimal G^urban rule.
"""

import time
import torch
import numpy as np

from src.data.dataset import load_cities, load_city
from src.data.urban_graph import build_knn_graph, build_radius_graph
from src.models.zero_shot_model import ZeroShotODModel
from src.training.train import train_zero_shot_model, infer_zero_shot
from src.training.evaluate import evaluate_all


def run_graph_topology_pilot():
    print("=" * 80)
    print("RUNNING URBAN GRAPH TOPOLOGY PILOT (G^urban)")
    print("Comparing: k-NN (k in {5, 10, 20}) vs Radius Graph")
    print("Train cities: ['Raleigh', 'Denver'] | Held-out target: 'Philadelphia'")
    print("=" * 80)

    train_cities = ["Raleigh", "Denver"]
    target_city = "Philadelphia"

    configurations = [
        ("k-NN (k=5)", {"type": "knn", "k": 5}),
        ("k-NN (k=10)", {"type": "knn", "k": 10}),
        ("k-NN (k=20)", {"type": "knn", "k": 20}),
        ("Radius Graph (r=5km)", {"type": "radius", "radius": 5.0}),
    ]

    results = []

    for name, cfg in configurations:
        print(f"\n--- Training with {name} ---")
        t0 = time.time()
        k_val = cfg.get("k", 10)

        # Train model with current graph configuration
        model, scaler = train_zero_shot_model(
            train_city_names=train_cities,
            data_root="data",
            epochs=15,
            lr=2e-3,
            hidden_dim=48,
            knn_k=k_val,
            verbose=False,
        )
        train_time = time.time() - t0

        # Build target city graph according to configuration
        target_data = load_city(target_city, data_root="data", feature_scaler=scaler, fit_scaler=False)
        coords = target_data.lon_lat.numpy()

        if cfg["type"] == "knn":
            ei, ed = build_knn_graph(coords, k=cfg["k"])
        else:
            ei, ed = build_radius_graph(coords, radius_km=cfg["radius"])

        # Infer on held-out city
        t_pred = infer_zero_shot(model, target_data, ei, ed)
        metrics = evaluate_all(target_data.pair_trips, t_pred)

        n_edges = ei.shape[1]
        avg_degree = n_edges / target_data.n_tracts

        res_dict = {
            "name": name,
            "target_cpc": metrics["cpc"],
            "target_rmse": metrics["rmse_log1p"],
            "target_pearson": metrics["pearson_r"],
            "graph_edges": n_edges,
            "avg_degree": avg_degree,
            "train_time_sec": train_time,
        }
        results.append(res_dict)

        print(f"  Result: Held-out CPC = {metrics['cpc']:.4f} | RMSE = {metrics['rmse_log1p']:.4f} | Avg degree = {avg_degree:.1f} | Train time = {train_time:.1f}s")

    print("\n" + "=" * 80)
    print("GRAPH TOPOLOGY PILOT SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Graph Configuration':<25} {'Held-out CPC':>14} {'RMSE-log1p':>12} {'Avg Degree':>12} {'Train Time (s)':>16}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<25} {r['target_cpc']:>14.4f} {r['target_rmse']:>12.4f} {r['avg_degree']:>12.1f} {r['train_time_sec']:>16.1f}")
    print("=" * 80)


if __name__ == "__main__":
    run_graph_topology_pilot()
