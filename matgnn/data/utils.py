"""Utility funtions for datasets."""
from typing import Union

from torch_geometric.data import Dataset, InMemoryDataset

from ..utils.input_parameters import DatasetParameters
from .dataset.in_memory_dataset import MolGraphInMemoryDataset


def get_dir_name(dataset_params: DatasetParameters) -> str:
    """Get directory name.

    Args:
        dataset_params (DatasetParameters): Dataset parameters.

    Returns:
        str: Directory name.
    """
    return f"{dataset_params.feature_type}"


def create_dataset(
    dataset_params: DatasetParameters
) -> Union[Dataset, InMemoryDataset]:
    """Create dataset.

    Args:
        dataset_params (DatasetParameters): Dataset parameters.

    Returns:
        Union[Dataset, InMemoryDataset]: Dataset.
    """
    return MolGraphInMemoryDataset(dataset_params)
