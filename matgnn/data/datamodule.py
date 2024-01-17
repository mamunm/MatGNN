"""Data Module for MatGNN"""


import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.data.dataset import Dataset

from ..utils.input_parameters import DataModuleParameters
from ..utils.validation import validate_data_module_parameters
from .utils import create_dataset


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
        self.dataset = create_dataset(params.dataset_params)
        self.batch_size = params.batch_size
        self.num_workers = params.num_workers
        self.test_ratio = params.test_ratio

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
        size = self.dataset.size
        if stage == "train":
            train_indices, test_indices = train_test_split(
                range(size), test_size=self.test_ratio
            )
            self.training_data = torch.utils.data.Subset(
                self.dataset,
                train_indices,
            )
            self.testing_data = torch.utils.data.Subset(
                self.dataset,
                test_indices,
            )
        if stage == "predict":
            self.prediction_data: Dataset = self.dataset

    def train_dataloader(self) -> DataLoader[Data]:
        """Returns the training dataloader."""
        return DataLoader(
            self.training_data,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def test_dataloader(self) -> DataLoader[Data]:
        """Returns the test dataloader."""
        return DataLoader(
            self.testing_data,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self) -> DataLoader[Data]:
        """Returns the test dataloader."""
        return DataLoader(
            self.prediction_data,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
        )
