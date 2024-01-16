"""Validate input parameters."""

from torch_geometric.data import Dataset, InMemoryDataset

from .input_parameters import DataModuleParameters


def validate_data_module_parameters(
    params: DataModuleParameters,
) -> None:
    """Validate input parameters for MatGNN data module.

    Args:
        params (DataModuleParameters): Input parameters.
    """

    if params.data is None:
        raise ValueError(
            "The data module requires a valid dataset. "
            "Please pass a valid dataset to the data module."
        )
    if not isinstance(params.data, Dataset) and not isinstance(
        params.data, InMemoryDataset
    ):
        raise ValueError(
            "data must be a pytorch geometric Dataset or InMemoryDataset object."
        )
    if not isinstance(params.batch_size, int):
        raise ValueError("batch_size must be an integer.")
    if not isinstance(params.num_workers, int):
        raise ValueError("num_workers must be an integer.")
    if not 0 <= params.test_ratio <= 1:
        raise ValueError("test_ratio must be between 0 and 1.")
