"""Test the MLP model."""


import sys
from pathlib import Path

import pytest
from torch_geometric.data import Data
import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))
from matgnn.models.mlp import MLP, MLPParameters  # noqa: E402


def test_mlp_parameters_invalid_inputs() -> None:
    """Test the MLPParameters with invalid inputs."""
    params = MLPParameters(
        n_hidden_layers="a",
        base2=True,
        activation="relu",
        activation_parameters={},
        layer_norm=True
    )
    with pytest.raises(ValueError):
        MLP(input_size=256, mlp_params=params)

def test_mlp_parameters_incompatible_inputs() -> None:
    """Test the MLPParameters with invalid inputs."""
    params = MLPParameters(
        n_hidden_layers=8,
        base2=True,
        activation="relu",
        activation_parameters={},
        layer_norm=True
    )
    with pytest.raises(ValueError):
        MLP(input_size=16, mlp_params=params)

def test_mlp_parameters_valid_inputs() -> None:
    """Test the MLPParameters with invalid inputs."""
    params = MLPParameters(
        n_hidden_layers=4,
        base2=True,
        activation="relu",
        activation_parameters={},
        layer_norm=True
    )

    mlp = MLP(input_size=256, mlp_params=params)
    assert isinstance(mlp, MLP)

def test_mlp_forward() -> None:
    """Test the MLP forward pass."""
    params = MLPParameters(
        n_hidden_layers=4,
        base2=True,
        activation="relu",
        activation_parameters={},
        layer_norm=True
    )
    mlp = MLP(input_size=256, mlp_params=params)
    data = Data(x=torch.randn(256))
    output = mlp(data)
    assert output.shape == torch.Size([1])
