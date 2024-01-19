"""Utility functions for models."""

from typing import Any, Dict, Union

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler

OPTIMIZER: Dict[str, Any] = {
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
}

SCHEDULER: Dict[str, Any] = {
    "constant_lr": torch.optim.lr_scheduler.ConstantLR,
    "reduce_lr_on_plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
}

SchedulerType = Union[_LRScheduler, ReduceLROnPlateau]


def get_optimizer(optimizer: str, optimizer_parameters: Dict[str, Any]) -> Optimizer:
    """Returns the activation function.

    Args:
        activation (str): The activation function.
        activation_parameters (Dict[str, Any]): parameters for the activation.

    Returns:
        Optimizer: The activation function.
    """
    if optimizer not in OPTIMIZER:
        raise ValueError(f"Optimizer function {optimizer} not supported.")
    opt_func: Optimizer = OPTIMIZER[optimizer](**optimizer_parameters)
    return opt_func


def get_scheduler(
    scheduler: str, scheduler_parameters: Dict[str, Any]
) -> SchedulerType:
    """Returns the scheduler function.

    Args:
        scheduler (str): The scheduler function.
        scheduler_parameters (Dict[str, Any]): parameters for the scheduler.

    Returns:
        SchedulerType: The scheduler function.
    """
    if scheduler not in SCHEDULER:
        raise ValueError(f"Scheduler function {scheduler} not supported.")
    scheduler_func: SchedulerType = SCHEDULER[scheduler](**scheduler_parameters)
    return scheduler_func
