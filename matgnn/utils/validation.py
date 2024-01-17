"""Validate input parameters."""

from pathlib import Path

from .input_parameters import DataModuleParameters, DatasetParameters


def validate_dataset_parameters(params: DatasetParameters) -> None:
    """Validate input parameters for MatGNN dataset.

    Args:
        params (DatasetParameters): Input parameters.
    """

    if params.feature_type not in ["CM", "SM"]:
        raise ValueError("feature_type must be one of the following: CM, SM.")
    if not isinstance(params.ase_db_loc, str):
        raise ValueError("ase_db_loc must be a string.")
    if not Path(params.ase_db_loc).exists():
        raise ValueError("ase_db_loc must be a valid file.")


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
