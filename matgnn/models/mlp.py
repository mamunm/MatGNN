"""Multi Layer Perceptron for MatGNN."""

import math
from typing import Any, Dict, NamedTuple

import torch
from torch import Tensor, nn

from .utils import get_activation, get_prev_pow_2


class MLPParameters(NamedTuple):
    """Inpute parameters for the MatGNN MLP model.

    Args:
        n_features (int): number of features.
        n_hidden_layers (int): number of the hidden layers.
        base2 (bool): should the number of neurons be halved in each layer.
        activation (str): which activation to use.
        activation_parameters (dict): parameters for the activation. Default to {}.
        layer_norm (bool): should the layer norm be used. Defaults to False.
        dtype (str): data type of the model. Defaults to "64".
    """

    n_features: int
    n_hidden_layers: int
    base2: bool
    activation: str
    activation_parameters: Dict[str, Any] = {}
    layer_norm: bool = False
    dtype: str = "64"

    def to_dict(self) -> Dict[str, Any]:
        """Returns a dictionary representation of the MLPParameters.

        Returns:
            dict: A dictionary representation of the MLPParameters.
        """
        return {
            "n_features": self.n_features,
            "n_hidden_layers": self.n_hidden_layers,
            "base2": self.base2,
            "activation": self.activation,
            "activation_parameters": self.activation_parameters,
            "layer_norm": self.layer_norm,
            "dtype": self.dtype,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MLPParameters":
        """Creates a MLPParameters object from a dictionary.

        Args:
            data (dict): A dictionary representation of the MLPParameters.

        Returns:
            MPLParameters: A MLPParameters object.
        """
        return cls(
            n_features=data.get("n_features"),  # type: ignore
            n_hidden_layers=data.get("n_hidden_layers"),  # type: ignore
            base2=data.get("base2"),  # type: ignore
            activation=data.get("activation"),  # type: ignore
            activation_parameters=data.get("activation_parameters"),  # type: ignore
            layer_norm=data.get("layer_norm"),  # type: ignore
            dtype=data.get("dtype"),  # type: ignore
        )


class MLP(nn.Module):
    """
    Multi Layer Perceptron

    Args:
        mlp_parameters: parameters defining the MLP output layer.
    """

    def __init__(self, mlp_params: MLPParameters) -> None:
        super().__init__()
        validate_mlp_parameters(mlp_params)
        input_size = mlp_params.n_features
        activation = get_activation(
            mlp_params.activation, mlp_params.activation_parameters
        )
        layers = torch.nn.ModuleList()
        output_size = get_prev_pow_2(input_size) if mlp_params.base2 else input_size
        if mlp_params.base2 and math.log2(output_size) < mlp_params.n_hidden_layers:
            raise ValueError(
                "The number of hidden layers is not compatible with base2 setting."
            )
        layers.append(nn.Linear(input_size, output_size))
        if mlp_params.layer_norm is True:
            layers.append(nn.LayerNorm(output_size))
        layers.append(activation)
        input_size = output_size
        for _ in range(mlp_params.n_hidden_layers):
            output_size = int(input_size / 2) if mlp_params.base2 else input_size
            layers.append(nn.Linear(input_size, output_size))
            if mlp_params.layer_norm is True:
                layers.append(nn.LayerNorm(output_size))
            layers.append(activation)
            input_size = output_size
        layers.append(nn.Linear(input_size, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, X: Tensor) -> Tensor:
        """Forward pass through the layers.

        Args:
            data (Data): input data.
        """
        out: Tensor = self.model(X)
        return out


def validate_mlp_parameters(params: MLPParameters) -> None:
    """Validate parameters for MLP.

    Args:
        params (MLPParameters): Input parameters.
    """

    if params.n_features is None:
        raise ValueError("n_features must be specified.")
    if params.n_hidden_layers is None:
        raise ValueError("n_hidden_layers must be specified.")
    if not isinstance(params.n_hidden_layers, int):
        raise ValueError("n_hidden_layers must be an integer.")
    if not isinstance(params.base2, bool):
        raise ValueError("base2 must be a boolean.")
    if params.activation is None:
        raise ValueError("activation must be specified.")
    if params.activation_parameters is None:
        raise ValueError("activation_parameters must be specified.")
    if not isinstance(params.layer_norm, bool):
        raise ValueError("layer_norm must be a boolean.")
    if params.dtype not in ["f64", "f32", "f16", "bf16"]:
        raise ValueError("dtype must be one of f64, f32, f16, bf16.")
