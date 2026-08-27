import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from src.data.dataset import (
    NODE_FEATURE_COLUMNS,
    get_scaler_fingerprint,
    load_cities,
    load_city,
    validate_feature_scaler,
)
from src.models.zero_shot_model import ZeroShotMLPModel
from src.training.train import load_checkpoint, save_checkpoint


def test_scaler_input_guards():
    with pytest.raises(ValueError, match="At least one training city"):
        load_cities([], data_root="data")

    with pytest.raises(ValueError, match="must be unique"):
        load_cities(["Raleigh", "Raleigh"], data_root="data")

    _, scaler = load_cities(["Raleigh", "Denver"], data_root="data")
    with pytest.raises(ValueError, match="either feature_scaler or fit_scaler"):
        load_city(
            "Raleigh",
            data_root="data",
            feature_scaler=scaler,
            fit_scaler=True,
        )


def test_scaler_schema_validation():
    malformed = StandardScaler()
    malformed.mean_ = np.zeros(25)
    malformed.var_ = np.ones(25)
    malformed.scale_ = np.ones(25)
    malformed.n_features_in_ = 25

    with pytest.raises(ValueError, match=r"expected \(26,\)"):
        validate_feature_scaler(malformed)


def test_checkpoint_scaler_provenance_and_tamper_detection(tmp_path):
    rng = np.random.default_rng(7)
    scaler = StandardScaler().fit(
        rng.normal(size=(31, len(NODE_FEATURE_COLUMNS)))
    )
    model = ZeroShotMLPModel(
        node_in_dim=26,
        node_hidden_dim=8,
        node_out_dim=8,
        decoder_hidden_dim=8,
        num_gnn_layers=1,
    )
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(
        checkpoint,
        model,
        scaler,
        train_info={},
        hyperparams={
            "node_in_dim": 26,
            "hidden_dim": 8,
            "num_gnn_layers": 1,
            "backbone": "mlp",
        },
    )

    _, loaded_scaler, metadata = load_checkpoint(checkpoint)
    assert np.array_equal(loaded_scaler.mean_, scaler.mean_)
    assert loaded_scaler.n_samples_seen_ == 31
    assert metadata["scaler_provenance"]["fingerprint"] == get_scaler_fingerprint(scaler)
    assert metadata["scaler_provenance"]["feature_columns"] == list(NODE_FEATURE_COLUMNS)

    bundle = torch.load(checkpoint, map_location="cpu", weights_only=False)
    bundle["scaler_mean_"][0] += 1.0
    torch.save(bundle, checkpoint)
    with pytest.raises(ValueError, match="scaler fingerprint mismatch"):
        load_checkpoint(checkpoint)


def test_existing_checkpoint_scaler_backward_compatibility():
    for checkpoint in (
        "results/checkpoints/5fold_fold1_seed1.pt",
        "results/checkpoints/mlp_fold1_seed1.pt",
    ):
        _, scaler, _ = load_checkpoint(checkpoint)
        validate_feature_scaler(scaler)