"""Helper function to create model instances."""

from typing import Dict, Union

import torch
from torch import nn

from .mlp import MLP, MLPParameters, validate_mlp_parameters

ModelParams = Union[MLPParameters, None]

DTYPE_MAP: Dict[str, torch.dtype] = {
    "f64": torch.float64,
    "f32": torch.float32,
    "f16": torch.float16,
    "bf16": torch.bfloat16,
}


def validate_model_parameters(model_params: ModelParams) -> None:
    """Validate the input parameters for the model.

    Args:
        model_params (ModelParams): Input parameters for the model.

    Raises:
        ValueError: If the input parameters are invalid.
    """
    if model_params is not None:
        validate_mlp_parameters(model_params)


def create_model(model_params: ModelParams) -> nn.Module:
    """Creates a model instance.

    Args:
        model_params (ModelParams): Input parameters for the model.

    Returns:
        Model: A model instance.
    """
    dtype = DTYPE_MAP[model_params.dtype]  # type: ignore
    validate_model_parameters(model_params)
    if model_params is not None:
        return MLP(model_params).to(dtype=dtype)
    else:
        return None  # type: ignore