"""Helper function to create model instances."""

from typing import Dict, Union

import torch
from torch import nn

from .conv_nn import GraphConvolution, GraphConvolutionParameters
from .mlp import MLP, MLPParameters, validate_mlp_parameters
from .mpnn import MPNN, MPNNParameters

ModelParams = Union[MLPParameters, MPNNParameters, GraphConvolutionParameters]

DTYPE_MAP: Dict[str, torch.dtype] = {
    "64": torch.float64,
    "32": torch.float32,
    "16": torch.float16,
    "bf16": torch.bfloat16,
}


def validate_model_parameters(model_params: ModelParams) -> None:
    """Validate the input parameters for the model.

    Args:
        model_params (ModelParams): Input parameters for the model.

    Raises:
        ValueError: If the input parameters are invalid.
    """
    if isinstance(model_params, MLPParameters):
        validate_mlp_parameters(model_params)


def create_model(model_params: ModelParams) -> nn.Module:
    """Creates a model instance.

    Args:
        model_params (ModelParams): Input parameters for the model.

    Returns:
        Model: A model instance.
    """
    dtype = DTYPE_MAP[model_params.dtype]
    validate_model_parameters(model_params)
    if isinstance(model_params, MLPParameters):
        return MLP(model_params).to(dtype=dtype)
    elif isinstance(model_params, MPNNParameters):
        return MPNN(model_params).to(dtype=dtype)
    else:
        return GraphConvolution(model_params).to(dtype=dtype)
