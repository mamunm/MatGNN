"""Utils for datasets."""

import functools
from pathlib import Path
from typing import Any, Dict, Literal, NamedTuple

import torch
from ase import db
from torch import Tensor


class AtomGraphParameters(NamedTuple):
    """Input parameters for the dataset.

    Args:
        feature_type (Literal["atomic_number", "atomic_symbol"]): The type of the feature.
        self_loop (bool, optional): whether to add self loop. Defaults to False.
        graph_radius (float, optional): graph radius. Defaults to 5.
        max_neighbors (int, optional): maximum number of neighbors. Defaults to 5.
        edge_feature (bool, optional): whether to add edge features. Defaults to False.
        edge_resolution (float, optional): edge resolution. Defaults to 50.
        add_node_degree (bool, optional): whether to add node degree. Defaults to False.
    """

    feature_type: Literal["atomic_number", "atomic_symbol"]
    self_loop: bool = False
    graph_radius: float = 5
    max_neighbors: int = 5
    edge_feature: bool = False
    edge_resolution: int = 50
    add_node_degree: bool = False


class DatasetParameters(NamedTuple):
    """Input parameters for the dataset.

    Args:
        feature_type (Literal["CM", "SM", "SOAP", "ACSF"]): The type of the feature.
        ase_db_loc (str): The location of the ASE database.
        target (str): The target property.
        dtype (str): The data type. Default is 64.
    """

    feature_type: Literal["CM", "SM", "SOAP", "ACSF", "AtomGraph"]
    ase_db_loc: str
    target: str
    dtype: str = "64"
    extra_parameters: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
        """Returns a dictionary representation of the DatasetParameters.

        Returns:
            dict: A dictionary representation of the DatasetParameters.
        """
        return {
            "feature_type": self.feature_type,
            "ase_db_loc": self.ase_db_loc,
            "target": self.target,
            "dtype": self.dtype,
            "extra_parameters": self.extra_parameters,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetParameters":
        """Creates a DatasetParameters object from a dictionary.

        Args:
            data (dict): A dictionary representation of the DatasetParameters.

        Returns:
            DatasetParameters: A DatasetParameters object.
        """
        return cls(
            feature_type=data.get("feature_type", None),
            ase_db_loc=data.get("ase_db_loc", None),
            target=data.get("target", None),
            dtype=data.get("dtype", None),
            extra_parameters=data.get("extra_parameters", {}),
        )


def get_dir_name(dataset_params: DatasetParameters) -> str:
    """Get directory name.

    Args:
        dataset_params (DatasetParameters): Dataset parameters.

    Returns:
        str: Directory name.
    """
    if dataset_params.feature_type in ("CM", "SM"):
        return f"{dataset_params.feature_type}_{dataset_params.dtype}"
    elif dataset_params.feature_type == "SOAP":
        extra_name = ""
        for key in ["n_max", "l_max", "r_cut", "sigma"]:
            extra_name += str(dataset_params.extra_parameters[key]) + "_"
        extra_name += "true" if dataset_params.extra_parameters["periodic"] else "false"
        return f"{dataset_params.feature_type}_{extra_name}_{dataset_params.dtype}"
    elif dataset_params.feature_type == "ACSF":
        extra_name = f"{dataset_params.extra_parameters['r_cut']}"
        g_params = tuple(
            tuple(
                tuple(
                    ele,
                )
                for ele in dataset_params.extra_parameters[g_data]
            )
            for g_data in ["g2_params", "g3_params", "g4_params", "g5_params"]
        )
        extra_name += f"_{hash(g_params)}"
        return f"{dataset_params.feature_type}_{extra_name}_{dataset_params.dtype}"
    else:
        return "dummy"


def get_data_statistics(ase_db_loc: str) -> Dict[str, Any]:
    """Get data statistics.

    Args:
        ase_db_loc (str): location of the ASE database.
    """
    ase_db = db.connect(ase_db_loc)
    max_natoms = max(d.natoms for d in ase_db.select())
    species = [
        *functools.reduce(
            lambda x, y: {*x, *y}, (data.symbols for data in ase_db.select())
        )
    ]
    return {"max_natoms": max_natoms, "species": species}


def validate_dataset_parameters(params: DatasetParameters) -> None:
    """Validate input parameters for MatGNN dataset.

    Args:
        params (DatasetParameters): Input parameters.
    """

    if params.feature_type not in ["CM", "SM", "SOAP", "ACSF", "AtomGraph"]:
        raise ValueError(
            "feature_type must be one of the following: "
            "CM, SM, SOAP, ACSF, AtomGraph."
        )
    if not isinstance(params.ase_db_loc, str):
        raise ValueError("ase_db_loc must be a string.")
    if not Path(params.ase_db_loc).exists():
        raise ValueError("ase_db_loc must be a valid file.")
    if params.dtype not in ["f64", "f32", "f16", "bf16"]:
        raise ValueError("dtype must be one of the following: f64, f32, f16, bf16.")
    if params.feature_type == "SOAP":
        if not isinstance(params.extra_parameters, dict):
            raise ValueError("extra_parameters must be a dictionary.")
        if "periodic" not in params.extra_parameters:
            raise ValueError("periodic must be specified in extra_parameters.")
        if "n_max" not in params.extra_parameters:
            raise ValueError("n_max must be specified in extra_parameters.")
        if "l_max" not in params.extra_parameters:
            raise ValueError("l_max must be specified in extra_parameters.")
        if "r_cut" not in params.extra_parameters:
            raise ValueError("r_cut must be specified in extra_parameters.")
        if "sigma" not in params.extra_parameters:
            raise ValueError("sigma must be specified in extra_parameters.")
    if params.feature_type == "ACSF":
        if not isinstance(params.extra_parameters, dict):
            raise ValueError("extra_parameters must be a dictionary.")
        if "r_cut" not in params.extra_parameters:
            raise ValueError("r_cut must be specified in extra_parameters.")
        if "g2_params" not in params.extra_parameters:
            raise ValueError("g2_params must be specified in extra_parameters.")
        if "g3_params" not in params.extra_parameters:
            raise ValueError("g3_params must be specified in extra_parameters.")
        if "g4_params" not in params.extra_parameters:
            raise ValueError("g4_params must be specified in extra_parameters.")
        if "g5_params" not in params.extra_parameters:
            raise ValueError("g5_params must be specified in extra_parameters.")


class GaussianSmearing(torch.nn.Module):
    """Gaussian smearing.

    Args:
        start (float, optional): Start value. Defaults to 0.0.
        stop (float, optional): Stop value. Defaults to 5.0.
        resolution (int, optional): Resolution. Defaults to 50.
        width (float, optional): Width. Defaults to 0.05.
    """

    def __init__(
        self,
        start: float = 0.0,
        stop: float = 5.0,
        resolution: int = 50,
        width: float = 0.05,
    ):
        super(GaussianSmearing, self).__init__()
        offset = torch.linspace(start, stop, resolution)
        self.coeff = -0.5 / ((stop - start) * width) ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist: Tensor) -> Tensor:
        """Forward pass.

        Args:
            dist (torch.Tensor): Distance.

        Returns:
            torch.Tensor: Smoothed distance.
        """
        dist = dist.unsqueeze(-1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))
