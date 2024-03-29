"""Trainer class for training graph neural network."""

from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import pytorch_lightning as pl
from loguru import logger
from torch import Tensor
from torch.nn import L1Loss
from torch.optim import Optimizer
from torch_geometric.data import Data

from .models.conv_nn import GraphConvolutionParameters
from .models.create_model import ModelParams, create_model
from .models.mlp import MLP, MLPParameters
from .models.mpnn import MPNN, MPNNParameters
from .utils import get_optimizer, get_scheduler

AVAILABLE_OPTIMIZERS: List[str] = ["adam", "sgd"]
AVAILABLE_SCHEDULERS: List[str] = ["constant_lr", "reduce_lr_on_plateau"]


class MatGNNParameters(NamedTuple):
    """Input parameters for the Trainer class.

    Args:
        model_params (ModelParams): Model parameters.
        optimizer (str): Optimizer.
        scheduler (Optional[str]): Scheduler.
        optimizer_parameters (Dict[str, Any]): Parameters for the optimizer.
        scheduler_parameters (Dict[str, Any]): Parameters for the scheduler.
    """

    model_params: ModelParams
    optimizer: str
    scheduler: Optional[str] = None
    optimizer_parameters: Dict[str, Any] = {}
    scheduler_parameters: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert the input parameters to a dictionary."""
        return {
            "model_params": self.model_params.to_dict(),
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "optimizer_parameters": self.optimizer_parameters,
            "scheduler_parameters": self.scheduler_parameters,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MatGNNParameters":
        """Convert the input parameters from a dictionary to as class.

        Args:
            d (Dict[str, Any]): Input parameters.

        Returns:
            TrainerParameters: Input parameters.
        """
        if "gcn_type" in d["model_params"]:
            model_params = GraphConvolutionParameters.from_dict(d["model_params"])
        elif "gcn_hidden_size" in d["model_params"]:
            model_params = MPNNParameters.from_dict(**d)  # type: ignore
        else:
            model_params = MLPParameters.from_dict(**d)  # type: ignore
        return cls(
            model_params=model_params,
            optimizer=d["optimizer"],
            optimizer_parameters=d["optimizer_parameters"],
            scheduler=d["scheduler"],
            scheduler_parameters=d["scheduler_parameters"],
        )


class MatGNN(pl.LightningModule):
    """MatGNN class for training graph neural network.

    Args:
        params (MatGNNParameters): Parameters
    """

    def __init__(self, params: MatGNNParameters):
        super(MatGNN, self).__init__()
        self.save_hyperparameters()
        self.input_params = params
        self.model = create_model(params.model_params)
        self.loss = L1Loss()
        self.train_outputs: List[Tuple[int | float]] = []
        self.val_outputs: List[Tuple[int | float]] = []

    def configure_optimizers(self) -> Union[Optimizer, Dict[str, Any]]:
        """
        configure optimizer and scheduler for training.

        Returns:
            the initialized optimizer and scheduler.
        """
        optimizer_params = {
            k: v for k, v in self.input_params.optimizer_parameters.items()
        }
        optimizer_params["params"] = self.parameters()
        optimizer = get_optimizer(self.input_params.optimizer, optimizer_params)
        if not self.input_params.scheduler:
            return optimizer
        else:
            scheduler_parameters = {
                k: v for k, v in self.input_params.scheduler_parameters.items()
            }
            scheduler_parameters["optimizer"] = optimizer
            scheduler_configs = {
                "scheduler": get_scheduler(
                    self._input_parameters.scheduler, scheduler_parameters
                ),
                "interval": "epoch",
            }
            if self.input_params.scheduler == "reduce_lr_on_plateau":
                scheduler_configs["monitor"] = "val_loss"
            return {"optimizer": optimizer, "lr_scheduler": scheduler_configs}

    def forward(
        self,
        X: Tensor,
        edge_idx: Optional[Tensor] = None,
        edge_weight: Optional[Tensor] = None,
        edge_attr: Optional[Tensor] = None,
        batch_map: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Get the prediction of a batch of features.

        Args:
            X (Tensor): The input features.
            edge_idx (Optional[Tensor], optional): The edge index.
                Defaults to None.
            edge_weight (Optional[Tensor], optional): The edge weights.
                Defaults to None.
            edge_attr (Optional[Tensor], optional): The edge attributes.
                Defaults to None.
            batch_map (Optional[Tensor], optional): The batch map.
                Defaults to None.

        Returns:
            Tensor: The prediction of the model.
        """

        if isinstance(self.model, MLP):
            predictions: Tensor = self.model(X)
        elif isinstance(self.model, MPNN):
            predictions = self.model(X, edge_idx, edge_attr, batch_map)
        else:
            predictions = self.model(X, edge_idx, edge_weight, edge_attr, batch_map)

        return predictions

    def training_step(self, batch: Tuple[Data], batch_idx: int) -> Tensor:
        """
        Train the model for one batch.

        Args:
            batch (Tuple[Data]): a batch of features
            batch_idx (int): the index of the batch

        Returns:
            Tensor: The prediction of the model.
        """

        if isinstance(self.model, MLP):
            predictions = self.model(batch.x).view(-1)  # type: ignore
        if isinstance(self.model, MPNN):
            predictions = self.model(
                batch.x,  # type: ignore
                batch.edge_index,  # type: ignore
                batch.edge_attr,  # type: ignore
                batch.batch,  # type: ignore
            ).view(-1)
        else:
            predictions = self.model(
                batch.x,  # type: ignore
                batch.edge_index,  # type: ignore
                batch.edge_weight,  # type: ignore
                batch.edge_attr,  # type: ignore
                batch.batch,  # type: ignore
            ).view(-1)

        loss: Tensor = self.loss(predictions, batch.y)  # type: ignore
        self.train_outputs.append((loss.item(), len(batch.y)))  # type: ignore
        return loss

    def on_train_epoch_end(self) -> None:
        """
        Compute the loss per molecule for the entire epoch and log.

        Args:
            outputs (List[Dict[str, Tensor]]): the loss of each batch in the epoch
        """
        total_loss = sum(item[0] * item[1] for item in self.train_outputs)  # type: ignore
        n_data = sum(item[1] for item in self.train_outputs)  # type: ignore
        mean_loss = total_loss / n_data
        self.train_outputs.clear()
        self.log("train_loss", mean_loss, on_epoch=True, prog_bar=False)
        logger.info(f"Epoch: {self.current_epoch} | Training loss: {mean_loss:.4f}")

    def validation_step(self, batch: Tuple[Data], batch_idx: int) -> Dict[str, float]:
        """
        Validate the model for one batch.

        Args:
            batch (Tuple[Data]): a batch of features
            batch_idx (int): the index of the batch

        Returns:
            Tensor: The prediction of the model.
        """

        if isinstance(self.model, MLP):
            predictions = self.model(batch.x).view(-1)  # type: ignore
        if isinstance(self.model, MPNN):
            predictions = self.model(
                batch.x,  # type: ignore
                batch.edge_index,  # type: ignore
                batch.edge_attr,  # type: ignore
                batch.batch,  # type: ignore
            ).view(-1)
        else:
            predictions = self.model(
                batch.x,  # type: ignore
                batch.edge_index,  # type: ignore
                batch.edge_weight,  # type: ignore
                batch.edge_attr,  # type: ignore
                batch.batch,  # type: ignore
            ).view(-1)

        val_loss: Tensor = self.loss(predictions, batch.y)  # type: ignore
        self.val_outputs.append((val_loss.item(), len(batch.y)))  # type: ignore

        return {"val_loss": val_loss.item()}

    def on_validation_epoch_end(self) -> None:  # type: ignore
        """
        Compute the loss per molecule for the validation set and log.
        """
        total_loss = sum(item[0] * item[1] for item in self.val_outputs)  # type: ignore
        n_data = sum(item[1] for item in self.val_outputs)  # type: ignore
        val_loss = total_loss / n_data
        self.val_outputs.clear()

        self.log("val_loss", val_loss, on_epoch=True, prog_bar=False)
        logger.info(f"Epoch: {self.current_epoch} | Validation loss: {val_loss:.4f}")


def validate_trainer_parameters(params: MatGNNParameters) -> None:
    """Validate the input parameters for the MatGNN class.

    Args:
        params (TrainerParameters): Input parameters.

    Raises:
        ValueError: If the input parameters are invalid.
    """
    if params.optimizer not in AVAILABLE_OPTIMIZERS:
        raise ValueError(f"Invalid optimizer: {params.optimizer}")
    if params.scheduler is not None and params.scheduler not in AVAILABLE_SCHEDULERS:
        raise ValueError(f"Invalid scheduler: {params.scheduler}")
