"""Utility funtions for data module."""
from typing import Union

from torch_geometric.data import Dataset, InMemoryDataset

from .dataset.dataset import MolGraphDataset
from .dataset.in_memory_dataset import MolGraphInMemoryDataset
from .dataset.utils import DatasetParameters


def create_dataset(
    in_memory: bool, dataset_params: DatasetParameters
) -> Union[Dataset, InMemoryDataset]:
    """Create dataset.

    Args:
        in_memory (bool): Whether to use in memory dataset.
        dataset_params (DatasetParameters): Dataset parameters.

    Returns:
        Union[Dataset, InMemoryDataset]: Dataset.
    """
    if in_memory:
        return MolGraphInMemoryDataset(dataset_params)
    else:
        return MolGraphDataset(dataset_params)
