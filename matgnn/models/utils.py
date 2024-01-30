"""Utility functions for models."""

import math
from typing import Any, Callable, Dict

import numpy as np
import torch
from torch import Tensor, nn
from torch_geometric.nn import (
    Set2Set,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

ACTIVATION_FUNCTIONS: Dict[str, Any] = {
    "relu": torch.nn.ReLU,
    "selu": torch.nn.SELU,
    "celu": torch.nn.CELU,
    "gelu": torch.nn.GELU,
    "sigmoid": torch.nn.Sigmoid,
    "tanh": torch.nn.Tanh,
}

POOL_FUNCTIONS: Dict[str, Any] = {
    "max": global_max_pool,
    "avg": global_mean_pool,
    "add": global_add_pool,
    "set2set": Set2Set,
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


def get_activation(
    activation: str, activation_parameters: Dict[str, Any] = {}
) -> nn.Module:
    """Returns the activation function.

    Args:
        activation (str): The activation function.
        activation_parameters (dict): parameters for the activation.
            Defaults to an empty dict.

    Returns:
        nn.Module: The activation function.
    """
    if activation not in ACTIVATION_FUNCTIONS:
        raise ValueError(f"Activation function {activation} not supported.")
    act_func: nn.Module = ACTIVATION_FUNCTIONS[activation](**activation_parameters)
    return act_func


def get_pool(
    pool: str, pool_parameters: Dict[str, Any] = {}
) -> Callable[[Tensor, Tensor], Tensor]:
    """Returns the pooling function.

    Args:
        pool (str): The pooling function.
        pool_parameters (dict): parameters for the pooling.

    Returns:
        nn.Module: The pooling function.
    """
    if pool not in POOL_FUNCTIONS:
        raise ValueError(f"Pooling function {pool} not supported.")
    if pool == "set2set":
        pool_func = POOL_FUNCTIONS[pool](**pool_parameters)
    else:
        pool_func = POOL_FUNCTIONS[pool]
    return pool_func  # type: ignore
