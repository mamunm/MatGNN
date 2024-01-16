"""Input parameters for MatGNN package."""

from typing import NamedTuple, Optional, Union

from torch_geometric.data import Dataset, InMemoryDataset


class DataModuleParameters(NamedTuple):
    """Inpute parameters for the MatGNN data module.

    Args:
        data (Optional[Union[Dataset, InMemoryDataset]]): pytorch geometric
            Dataset or InMemoryDataset. Default to None.
        batch_size (Optional[int]): The batch size. Defaults to 64.
        num_workers (Optional[int]): The number of workers. Defaults to 1.
        test_ratio (float): The ratio of testing to training.
    """

    data: Optional[Union[Dataset, InMemoryDataset]] = None
    batch_size: int = 64
    num_workers: int = 1
    test_ratio: float = 0.2
