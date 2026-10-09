"""Numerical and selection checks for source validation in the v3 runner."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from implement_new_plan.experiment import run_new_plan_v4_budgeted_validation as runner


def _small_inputs() -> tuple[runner.ModelInputs, SimpleNamespace]:
    origin = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=np.int64)
    destination = np.array([1, 2, 0, 3, 1, 4, 0, 5, 1, 5], dtype=np.int64)
    distance = np.linspace(0.5, 5.0, len(origin)).astype(np.float64)
    flow = np.linspace(1.0, 10.0, len(origin)).astype(np.float64)
    population = np.linspace(100.0, 600.0, 6).astype(np.float32)
    features = np.stack([np.log1p(population), np.arange(6, dtype=np.float32)], axis=1)
    inputs = runner.ModelInputs(
        origin=origin,
        destination=destination,
        distance=distance,
        true_flow=flow,
        origin_tensor=torch.from_numpy(origin),
        destination_tensor=torch.from_numpy(destination),
        distance_tensor=torch.from_numpy(distance.astype(np.float32)),
        population_tensor=torch.from_numpy(population),
        node_features=torch.from_numpy(features),
        distance_std=torch.from_numpy(np.log1p(distance).astype(np.float32)).unsqueeze(-1),
    )
    nodes = torch.arange(6, dtype=torch.long)
    prepared = SimpleNamespace(edge_index=torch.stack([nodes, nodes]), edge_distance=torch.ones(6))
    return inputs, prepared


@pytest.mark.parametrize("model_name", ["gravity_2param", "pairwise_mlp", "urban_gnn"])
def test_validation_selects_and_restores_best_epoch(monkeypatch, model_name):
    inputs, prepared = _small_inputs()
    monkeypatch.setitem(runner.TRAINING_CONFIG, "max_epochs", 12)
    monkeypatch.setitem(runner.TRAINING_CONFIG, "early_stopping_patience", 3)
    monkeypatch.setitem(runner.TRAINING_CONFIG, "prediction_chunk_size", 3)
    model, detail = runner._train_with_source_validation(model_name, prepared, inputs, inputs, seed=1)

    trace = detail["training_trace"]
    assert 1 <= detail["best_epoch"] <= detail["epochs_run"] <= 12
    assert detail["best_epoch"] == min(trace, key=lambda row: row["validation_loss"])["epoch"]
    assert detail["best_validation_loss"] == min(row["validation_loss"] for row in trace)
    assert detail["validation_pairs"] == len(inputs.true_flow)
    restored = runner._predict(model_name, model, inputs, prepared)
    restored_loss = np.mean((np.log1p(restored) - np.log1p(inputs.true_flow)) ** 2)
    assert restored_loss == pytest.approx(detail["best_validation_loss"], rel=1e-6)
    assert not any(parameter.requires_grad for parameter in model.parameters())


@pytest.mark.parametrize("model_name", ["gravity_2param", "pairwise_mlp", "urban_gnn"])
def test_chunked_inference_matches_single_chunk(monkeypatch, model_name):
    inputs, prepared = _small_inputs()
    torch.manual_seed(5)
    model = runner._new_model(model_name, inputs.node_features.shape[-1]).eval()
    monkeypatch.setitem(runner.TRAINING_CONFIG, "prediction_chunk_size", 3)
    chunked = runner._predict(model_name, model, inputs, prepared)
    monkeypatch.setitem(runner.TRAINING_CONFIG, "prediction_chunk_size", 100)
    full = runner._predict(model_name, model, inputs, prepared)
    np.testing.assert_allclose(chunked, full, rtol=1e-6, atol=1e-6)
