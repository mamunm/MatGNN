"""Utility functions for models."""

import math
from typing import Any, Dict

import numpy as np
import torch
from torch import nn

ACTIVATION_FUNCTIONS: Dict[str, Any] = {
    "relu": torch.nn.ReLU,
    "selu": torch.nn.SELU,
    "celu": torch.nn.CELU,
    "gelu": torch.nn.GELU,
    "sigmoid": torch.nn.Sigmoid,
    "tanh": torch.nn.Tanh,
}


def get_prev_pow_2(number: int) -> int:
    """Returns the previous power of 2 of a number.

    Args:
        number (int): The number.

    Returns:
        int: The previous power of 2 of the number.
    """
    if math.log2(number).is_integer():
        return int(2 ** (math.log2(number) - 1))
    else:
        return int(2 ** (np.floor(math.log2(number))))


def get_activation(activation: str, activation_parameters: Dict[str, Any]) -> nn.Module:
    """Returns the activation function.

    Args:
        activation (str): The activation function.
        activation_parameters (dict): parameters for the activation.

    Returns:
        nn.Module: The activation function.
    """
    if activation not in ACTIVATION_FUNCTIONS:
        raise ValueError(f"Activation function {activation} not supported.")
    act_func: nn.Module = ACTIVATION_FUNCTIONS[activation](**activation_parameters)
    return act_func
