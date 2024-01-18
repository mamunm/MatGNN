"""Multi Layer Perceptron for MatGNN."""

import math

import torch
from torch import Tensor, nn
from torch_geometric.data import Data

from .input_parameters import MLPParameters
from .utils import get_activation, get_prev_pow_2
from .validation import validate_mlp_parameters


class MLP(nn.Module):
    """
    Multi Layer Perceptron

    Args:
        input_size: size of the input
        mlp_parameters: parameters defining the MLP output layer.
    """

    def __init__(self, input_size: int, mlp_params: MLPParameters) -> None:
        super().__init__()
        validate_mlp_parameters(mlp_params)
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

    def forward(self, data: Data) -> Tensor:
        """Forward pass through the layers.

        Args:
            data (Data): input data.
        """
        out: Tensor = self.model(data.x)
        return out
