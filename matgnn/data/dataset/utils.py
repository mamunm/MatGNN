"""Utils for datasets."""

import functools
from pathlib import Path
from typing import Any, Dict, Literal, NamedTuple, Optional, Union

import torch
from ase import db
from torch import Tensor


class AtomGraphParameters(NamedTuple):
    """Input parameters for the atom graph.

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


class SOAPParameters(NamedTuple):
    """Input parameters for the SOAP descriptor.

    Args:
        periodic (bool): whether to use periodic boundary conditions.
        n_max (int): number of radial basis functions.
        l_max (int): maximum degree of spherical harmonics.
        r_cut (float): cut-off radius.
        sigma (float): width of Gaussian kernel.
    """

    periodic: bool
    n_max: int
    l_max: int
    r_cut: float
    sigma: float


class DatasetParameters(NamedTuple):
    """Input parameters for the dataset.

    Args:
        feature_type (Literal["CM", "SM", "SOAP", "AtomGraph"]): The type of the feature.
        ase_db_loc (str): The location of the ASE database.
        target (str): The target property.
        dtype (str): The data type. Default is 64.
    """

    feature_type: Literal["CM", "SM", "SOAP", "AtomGraph"]
    ase_db_loc: str
    target: str
    dtype: str = "64"
    extra_parameters: Optional[Union[SOAPParameters, AtomGraphParameters]] = None


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
            extra_name += str(getattr(dataset_params.extra_parameters, key)) + "_"
        extra_name += "true" if dataset_params.extra_parameters.periodic else "false"  # type: ignore
        return f"{dataset_params.feature_type}_{extra_name}_{dataset_params.dtype}"
    else:
        extra_name = f"{dataset_params.extra_parameters.feature_type}"  # type: ignore
        extra_name += f"_{dataset_params.extra_parameters.self_loop}"  # type: ignore
        extra_name += f"_{dataset_params.extra_parameters.graph_radius}"  # type: ignore
        extra_name += f"_{dataset_params.extra_parameters.max_neighbors}"  # type: ignore
        extra_name += f"_{dataset_params.extra_parameters.edge_feature}"  # type: ignore
        extra_name += f"_{dataset_params.extra_parameters.edge_resolution}"  # type: ignore
        extra_name += f"_{dataset_params.extra_parameters.add_node_degree}"  # type: ignore
        return f"{dataset_params.feature_type}_{extra_name}_{dataset_params.dtype}"


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
