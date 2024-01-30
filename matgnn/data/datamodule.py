"""Data Module for MatGNN"""

from typing import NamedTuple

import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.data.dataset import Dataset
from torch_geometric.loader import DataLoader as GeoDataLoader

from .create_dataset import create_dataset
from .dataset.utils import DatasetParameters, validate_dataset_parameters


class DataModuleParameters(NamedTuple):
    """Inpute parameters for the MatGNN data module.

    Args:
        in_memory (bool): Whether to use in memory dataset.
        dataset_params (DatasetParameters): dataset parameters.
        batch_size (Optional[int]): The batch size. Defaults to 64.
        num_workers (Optional[int]): The number of workers. Defaults to 1.
        val_ratio (float): The ratio of validation to training. Defaults to 0.2.
    """

    in_memory: bool
    dataset_params: DatasetParameters
    batch_size: int = 64
    num_workers: int = 12
    val_ratio: float = 0.2


class MaterialsGraphDataModule(pl.LightningDataModule):
    """
    Lightning DataModule constructor for materials graph data.

    Args:
    params (DataModuleParameters): inputs to the data module

    Raises:
        ValueError: if the input parameters are not valid.
    """

    def __init__(self, params: DataModuleParameters):
        super().__init__()
        validate_data_module_parameters(params)
        self.dataset = create_dataset(params.in_memory, params.dataset_params)
        self.batch_size = params.batch_size
        self.num_workers = params.num_workers
        self.val_ratio = params.val_ratio

    def setup(self, stage: str = "train") -> None:
        """Sets up the data loaders for graph learning.

        Args:
        stage (str): The stage of training to set up the data loader
            for. Defaults to train.

        Returns:
            None
        """

        if stage not in ["train", "predict"]:
            raise ValueError(
                "The stage must be one of the following: " "train, predict."
            )
        size = self.dataset.len()
        if stage == "train":
            train_indices, val_indices = train_test_split(
                range(size), test_size=self.val_ratio
            )
            self.train_data = torch.utils.data.Subset(
                self.dataset,
                train_indices,
            )
            self.val_data = torch.utils.data.Subset(
                self.dataset,
                val_indices,
            )
        if stage == "predict":
            self.prediction_data: Dataset = self.dataset

    def train_dataloader(self) -> DataLoader[Data]:
        """Returns the training dataloader."""
        return GeoDataLoader(  # type: ignore
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader[Data]:
        """Returns the val dataloader."""
        return GeoDataLoader(  # type: ignore
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self) -> DataLoader[Data]:
        """Returns the test dataloader."""
        return GeoDataLoader(  # type: ignore
            self.prediction_data,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
        )


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
    if not 0 <= params.val_ratio <= 1:
        raise ValueError("val_ratio must be between 0 and 1.")
