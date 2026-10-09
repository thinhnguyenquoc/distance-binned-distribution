"""Fast contracts for the focused source-scarcity protocol.

These tests intentionally use synthetic OD arrays: they lock the split and
execution-matrix semantics without requiring a model fit or a city dataset.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from implement_new_plan.experiment.run_new_plan_v4_budgeted_validation import (
    FRACTIONS,
    GRAVITY_SEED,
    MAIN_FRACTION,
    NEURAL_SEEDS,
    VALIDATION_FRACTION,
    _model_seed_plan,
    build_v4_city_split,
    expected_source_row_counts,
)


def _synthetic_city(order=None):
    """Return a city with valid support plus pairs excluded by the contract."""
    n_valid = 123
    origin = np.arange(n_valid, dtype=np.int64)
    destination = np.arange(n_valid, dtype=np.int64) + 100
    distance = np.linspace(0.5, 23.5, n_valid, dtype=np.float64)
    flow = np.arange(1, n_valid + 1, dtype=np.float64)

    # Intrazonal, zero-distance, and zero-flow pairs must not enter Omega^+.
    origin = np.concatenate([origin, np.array([7, 30, 31])])
    destination = np.concatenate([destination, np.array([7, 130, 131])])
    distance = np.concatenate([distance, np.array([2.0, 0.0, 3.0])])
    flow = np.concatenate([flow, np.array([8.0, 9.0, 0.0])])

    if order is not None:
        origin = origin[order]
        destination = destination[order]
        distance = distance[order]
        flow = flow[order]

    return SimpleNamespace(
        city_name="Synthetic",
        pair_o_idx=origin,
        pair_d_idx=destination,
        dist_km=distance,
        pair_trips=flow,
    )


def _manifest_key_view(split):
    columns = [
        "origin",
        "destination",
        "permutation_rank",
        "in_budget_10",
        "in_budget_20",
        "in_budget_30",
        "in_budget_40",
        "in_train_10",
        "in_train_20",
        "in_train_30",
        "in_train_40",
        "in_source_validation",
        "in_source_test",
        "split_seed",
    ]
    return split["manifest"][columns].sort_values(["origin", "destination"]).reset_index(drop=True)


def test_v4_split_is_deterministic_and_independent_of_input_row_order():
    city = _synthetic_city()
    first = build_v4_city_split(city)
    repeated = build_v4_city_split(city)

    # Reordering raw CSV-like rows must not alter the pair-key manifest.
    reordered = build_v4_city_split(_synthetic_city(np.random.default_rng(9).permutation(126)))

    assert first["support_hash"] == repeated["support_hash"] == reordered["support_hash"]
    assert first["counts"] == repeated["counts"] == reordered["counts"]
    pd.testing.assert_frame_equal(_manifest_key_view(first), _manifest_key_view(repeated))
    pd.testing.assert_frame_equal(_manifest_key_view(first), _manifest_key_view(reordered))


def test_v4_split_respects_total_label_budget_and_independent_test():
    split = build_v4_city_split(_synthetic_city())
    manifest = split["manifest"]
    n_support = 123

    assert set(FRACTIONS) == {0.10, 0.20, 0.30, 0.40}
    assert split["counts"] == {fraction: int(np.floor(fraction * n_support)) for fraction in FRACTIONS}
    assert len(split["source_validation_indices"]) == int(np.floor(VALIDATION_FRACTION * n_support))
    assert len(split["source_test_indices"]) == n_support - int(np.floor(0.40 * n_support))

    train_sets = {
        fraction: set(np.asarray(indices, dtype=np.int64).tolist())
        for fraction, indices in split["train_indices"].items()
    }
    source_validation = set(np.asarray(split["source_validation_indices"], dtype=np.int64).tolist())
    source_test = set(np.asarray(split["source_test_indices"], dtype=np.int64).tolist())
    support = set(np.asarray(split["support_indices"], dtype=np.int64).tolist())

    previous = set()
    for fraction in FRACTIONS:
        assert previous <= train_sets[fraction]
        assert not (train_sets[fraction] & source_validation)
        assert not (train_sets[fraction] & source_test)
        assert len(train_sets[fraction]) + len(source_validation) == split["counts"][fraction]
        previous = train_sets[fraction]

    assert train_sets[MAIN_FRACTION] | source_validation | source_test == support
    assert int(manifest["in_source_validation"].sum()) == len(source_validation)
    assert int(manifest["in_source_test"].sum()) == len(source_test)
    for fraction in FRACTIONS:
        column = f"in_train_{int(round(fraction * 100)):02d}"
        budget_column = f"in_budget_{int(round(fraction * 100)):02d}"
        assert int(manifest[column].sum()) == split["counts"][fraction] - len(source_validation)
        assert int(manifest[budget_column].sum()) == split["counts"][fraction]
        assert not (manifest[column] & manifest["in_source_validation"]).any()
        assert not (manifest[column] & manifest["in_source_test"]).any()


def test_v4_split_filters_to_positive_interzonal_support():
    split = build_v4_city_split(_synthetic_city())
    manifest = split["manifest"]

    assert len(manifest) == 123
    assert (manifest["origin"] != manifest["destination"]).all()
    assert (manifest["distance_km"] > 0.0).all()
    assert "raw_pair_index" in manifest.columns


def test_v4_split_rejects_ambiguous_or_too_small_support():
    duplicate = _synthetic_city()
    duplicate.pair_o_idx = np.append(duplicate.pair_o_idx, 0)
    duplicate.pair_d_idx = np.append(duplicate.pair_d_idx, 100)
    duplicate.dist_km = np.append(duplicate.dist_km, 1.0)
    duplicate.pair_trips = np.append(duplicate.pair_trips, 1.0)
    with pytest.raises(ValueError, match="duplicate positive OD keys"):
        build_v4_city_split(duplicate)

    small = SimpleNamespace(
        city_name="TooSmall",
        pair_o_idx=np.arange(9, dtype=np.int64),
        pair_d_idx=np.arange(9, dtype=np.int64) + 20,
        dist_km=np.ones(9, dtype=np.float64),
        pair_trips=np.ones(9, dtype=np.float64),
    )
    with pytest.raises(ValueError, match="invalid v4 split sizes"):
        build_v4_city_split(small)


def test_v4_model_seed_plan_matches_the_focused_coverage_matrix():
    plan = _model_seed_plan()
    assert len(plan) == len(set(plan)) == 19

    gravity = [(fraction, seed, stochastic) for model, fraction, seed, stochastic in plan if model == "gravity_2param"]
    gnn = [(fraction, seed, stochastic) for model, fraction, seed, stochastic in plan if model == "urban_gnn"]
    mlp = [(fraction, seed, stochastic) for model, fraction, seed, stochastic in plan if model == "pairwise_mlp"]

    assert gravity == [(fraction, GRAVITY_SEED, False) for fraction in FRACTIONS]
    assert set(gnn) == {(fraction, seed, True) for fraction in FRACTIONS for seed in NEURAL_SEEDS}
    assert set(mlp) == {(MAIN_FRACTION, seed, True) for seed in NEURAL_SEEDS}
    assert len(gnn) == 12
    assert len(mlp) == 3


def test_v4_expected_per_source_result_counts_follow_the_protocol_matrix():
    n_targets = 49
    assert expected_source_row_counts(n_targets) == {
        "source_validation_metrics.csv": 19,
        "source_test_metrics.csv": 19,
        "main_transfer_metrics.csv": n_targets * 7,
        "scarcity_transfer_metrics.csv": n_targets * 16,
        "resolution_metrics.csv": n_targets * 3 * 3,
        "noise_metrics.csv": n_targets * 3 * (1 + 2 * 5),
        "specificity_metrics.csv": n_targets * 3,
    }
