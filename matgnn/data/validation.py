"""Validate input parameters."""

from .dataset.validation import validate_dataset_parameters
from .input_parameters import DataModuleParameters


def validate_data_module_parameters(
    params: DataModuleParameters,
) -> None:
    """Validate input parameters for MatGNN data module.

    Args:
        params (DataModuleParameters): Input parameters.
    """

    if params.dataset_params is None:
        raise ValueError("dataset_params must be specified.")
    validate_dataset_parameters(params.dataset_params)
    if not isinstance(params.batch_size, int):
        raise ValueError("batch_size must be an integer.")
    if not isinstance(params.num_workers, int):
        raise ValueError("num_workers must be an integer.")
    if not 0 <= params.test_ratio <= 1:
        raise ValueError("test_ratio must be between 0 and 1.")
