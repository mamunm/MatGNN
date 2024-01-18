"""Utility funtions for data module."""
from typing import Union

from torch_geometric.data import Dataset, InMemoryDataset

from .dataset.in_memory_dataset import MolGraphInMemoryDataset
from .dataset.input_parameters import DatasetParameters


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
