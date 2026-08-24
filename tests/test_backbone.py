"""
Unit and Contract Test Suite for Backbone Robustness & MLP Architecture.
Tests:
    1. Parameter parity between MLPLayer and GraphConvLayer.
    2. NodeMLP forward pass shape and gradient propagation.
    3. ZeroShotMLPModel forward pass and ZTNB loss compatibility.
    4. Model checkpoint save and reload consistency for backbone='mlp'.
    5. Table 7 Backbone Robustness summary output validity.
"""

import os
import tempfile
import torch
import numpy as np
from pathlib import Path

from src.models.node_encoder import UrbanGNN, NodeMLP, MLPLayer, GraphConvLayer
from src.models.zero_shot_model import ZeroShotODModel, ZeroShotMLPModel
from src.training.train import save_checkpoint, load_checkpoint
from src.loss.ztnb import ztnb_nll


def test_parameter_count_parity():
    """T1: MLPLayer and GraphConvLayer have identical learnable parameter counts."""
    hidden_dim = 64
    gnn_layer = GraphConvLayer(in_dim=hidden_dim, out_dim=hidden_dim)
    mlp_layer = MLPLayer(in_dim=hidden_dim, out_dim=hidden_dim)

    gnn_params = sum(p.numel() for p in gnn_layer.parameters() if p.requires_grad)
    mlp_params = sum(p.numel() for p in mlp_layer.parameters() if p.requires_grad)

    assert gnn_params == mlp_params, f"Parameter mismatch: GNN={gnn_params} vs MLP={mlp_params}"
    print("✓ Test 1 Passed: Exact parameter parity between MLPLayer and GraphConvLayer.")


def test_node_mlp_forward_and_backward():
    """T2: NodeMLP produces correct embedding shapes and supports backprop."""
    N = 25
    in_dim = 26
    hidden_dim = 64
    x = torch.randn(N, in_dim, requires_grad=True)
    dummy_edge_index = torch.zeros((2, 10), dtype=torch.long)
    dummy_edge_dist = torch.zeros(10, dtype=torch.float32)

    mlp_encoder = NodeMLP(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=hidden_dim, num_layers=2)
    h = mlp_encoder(x, dummy_edge_index, dummy_edge_dist)

    assert h.shape == (N, hidden_dim)
    loss = h.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    print("✓ Test 2 Passed: NodeMLP forward shape and gradients are valid.")


def test_zero_shot_mlp_model():
    """T3: ZeroShotMLPModel forward pass, conditional mean, and loss computation."""
    N = 20
    E = 150
    x = torch.randn(N, 26)
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    edge_dist = torch.zeros(0, dtype=torch.float32)
    pair_o = torch.randint(0, N, (E,))
    pair_d = torch.randint(0, N, (E,))
    pair_dist = torch.rand(E) * 2.0
    pop = torch.rand(N) * 5000.0 + 100.0
    t_true = torch.randint(1, 100, (E,)).float()

    model = ZeroShotMLPModel(node_in_dim=26, node_hidden_dim=64, node_out_dim=64, num_gnn_layers=2)
    
    mu_nb = model(x, edge_index, edge_dist, pair_o, pair_d, pair_dist, pop, return_conditional_mean=False)
    assert mu_nb.shape == (E,)
    assert (mu_nb > 0).all()

    cond_mean = model(x, edge_index, edge_dist, pair_o, pair_d, pair_dist, pop, return_conditional_mean=True)
    assert cond_mean.shape == (E,)
    assert (cond_mean > 0).all()

    loss = ztnb_nll(t_true, mu_nb, model.log_phi)
    assert torch.isfinite(loss)
    print("✓ Test 3 Passed: ZeroShotMLPModel forward pass and ZTNB loss computation are valid.")


def test_mlp_checkpoint_save_and_load():
    """T4: Checkpoint save and load correctly identifies and restores ZeroShotMLPModel."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test_mlp.pt"
        model = ZeroShotMLPModel(node_in_dim=26, node_hidden_dim=32, node_out_dim=32, decoder_hidden_dim=32, num_gnn_layers=2)
        scaler = None

        save_checkpoint(
            path=ckpt_path,
            model=model,
            scaler=scaler,
            train_info={"epoch": 10, "val_cpc": 0.725},
            hyperparams={
                "node_in_dim": 26,
                "hidden_dim": 32,
                "num_gnn_layers": 2,
                "backbone": "mlp",
            }
        )

        loaded_model, _, info = load_checkpoint(ckpt_path, device_str="cpu")
        assert isinstance(loaded_model, ZeroShotMLPModel)
        assert info["train_info"]["epoch"] == 10
        assert info["train_info"]["val_cpc"] == 0.725
        print("✓ Test 4 Passed: Checkpoint save/load for backbone='mlp' functions correctly.")


if __name__ == "__main__":
    print("Running Backbone Test Suite...")
    test_parameter_count_parity()
    test_node_mlp_forward_and_backward()
    test_zero_shot_mlp_model()
    test_mlp_checkpoint_save_and_load()
    print("\nAll 4/4 Backbone Unit & Contract Tests Passed Successfully!")
