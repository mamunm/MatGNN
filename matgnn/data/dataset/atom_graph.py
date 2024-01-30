"""Atom graph feature constructor."""
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from ase import db
from numpy.typing import NDArray
from scipy.stats import rankdata
from sklearn.preprocessing import LabelBinarizer
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, degree, dense_to_sparse

from .utils import AtomGraphParameters, GaussianSmearing, get_data_statistics


def get_atom_graph_data(
    ase_db_loc: str, target: str, params: AtomGraphParameters, dtype: torch.dtype
) -> List[Data]:
    """Generate atom graph features.

    Args:
        ase_db_loc (str): location of the ASE database.
        target (str): target attribute.
        params (AtomGraphParameters): parameters.
        dtype (torch.dtype): data type.

    Returns:
        List[Data]: List of Data objects.
    """
    if params.feature_type == "atomic_number":
        with open(Path(__file__).parent / "dictionary_default.json", "r") as f:
            atomic_number_dict = json.load(f)
    else:
        data_statistics = get_data_statistics(ase_db_loc)
        lb = LabelBinarizer()
        lb.fit(data_statistics["species"])

    if params.edge_feature:
        distance_gaussian = GaussianSmearing(0, 1, params.edge_resolution, 0.2)
    ase_db = db.connect(ase_db_loc)

    data_list = []

    for struct in ase_db.select():
        y = getattr(struct, target)
        atom_obj = struct.toatoms()
        edge_index, edge_weight = get_edge_from_positions(
            atom_obj.get_all_distances(mic=True),
            params.self_loop,
            params.graph_radius,
            params.max_neighbors,
        )
        if params.feature_type == "atomic_number":
            atom_features = np.vstack(
                [atomic_number_dict[str(i)] for i in atom_obj.get_atomic_numbers()]
            )
        elif params.feature_type == "atomic_symbol":
            atom_features = lb.transform(atom_obj.get_chemical_symbols())
        else:
            raise ValueError(
                "feature_type must be one of the following: "
                "atomic_number, atomic_symbol."
            )
        atom_features_tensor: Tensor = torch.tensor(atom_features, dtype=dtype)
        if params.add_node_degree:
            node_degree = degree(
                edge_index[0], atom_obj.get_global_number_of_atoms(), dtype=torch.long
            )
            node_degree = F.one_hot(
                node_degree, num_classes=params.max_neighbors + 1
            ).to(dtype)
            atom_features_tensor = torch.cat(
                [atom_features_tensor, node_degree], dim=-1
            )
        data = Data(
            x=atom_features_tensor,
            y=torch.tensor(y, dtype=dtype),
            edge_index=edge_index.to(dtype=torch.long),
            edge_weight=edge_weight.to(dtype=dtype),
            edge_attr=edge_weight.to(dtype=dtype),
        )
        data_list.append(data)

        if params.edge_feature:
            data_list = normalize_edge_features(data_list)
            for data in data_list:
                data.edge_attr = distance_gaussian(data.edge_attr)

    return data_list


def get_edge_from_positions(
    positions: NDArray[np.float64],
    self_loop: bool = False,
    graph_radius: float = 5,
    max_neighbors: int = 5,
) -> Tuple[Tensor, Tensor]:
    """Get edge index and edge weight from positions.

    Args:
        positions (NDArray[np.float64]): positions.
        self_loop (bool, optional): whether to add self loop. Defaults to False.
        graph_radius (float, optional): graph radius. Defaults to 5.
        max_neighbors (int, optional): maximum number of neighbors. Defaults to 5.

    Returns:
        Tuple[Tensor, Tensor]: edge index and edge weight.
    """
    filtered_positions = torch.tensor(
        filter_positions(positions, graph_radius, max_neighbors)
    )
    edge_index, edge_weight = dense_to_sparse(filtered_positions)

    if self_loop:
        edge_index, edge_weight = add_self_loops(
            edge_index, edge_weight, num_nodes=positions.shape[0], fill_value=0
        )
    return edge_index, edge_weight


def filter_positions(
    positions: NDArray[np.float64], graph_radius: float, max_neighbors: float
) -> NDArray[np.float64]:
    """Filter positions based on distance and number of neighbor threshold.

    Args:
        positions (NDArray[np.float64]): positions.
        graph_radius (float): graph radius.
        max_neighbors (float): maximum number of neighbors.

    Returns:
        NDArray[np.float64]: filtered positions.
    """
    neighbor_mask = rankdata(positions, method="ordinal", axis=1) <= max_neighbors
    positions = np.where(neighbor_mask, positions, np.nan)
    distance_mask = positions <= graph_radius
    positions = np.where(distance_mask, positions, np.nan)
    return np.nan_to_num(positions)


def normalize_edge_features(data_list: List[Data]) -> List[Data]:
    """Normalize edge features.

    Args:
        data_list (List[Data]): List of Data objects.

    Returns:
        List[Data]: List of Data objects.
    """
    min_feature, max_feature = np.inf, -np.inf
    for data in data_list:
        min_feature = min(min_feature, data.edge_attr.min())
        max_feature = max(max_feature, data.edge_attr.max())
    feature_range = max_feature - min_feature
    for data in data_list:
        data.edge_attr = (data.edge_attr - min_feature) / feature_range
    return data_list
