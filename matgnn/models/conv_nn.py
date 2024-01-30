"""Convolution GNN for MatGNN."""

from typing import Any, Dict, Literal, NamedTuple

import torch_geometric.nn as pyg_nn
from torch import Tensor, nn
from torch_geometric.nn.models.schnet import InteractionBlock

from .utils import get_activation, get_pool


class ConvolutionGNNParameters(NamedTuple):
    """Inpute parameters for the ConvolutionGNN model.

    Args:
        n_features (int): number of features.
        n_edge_features (int): number of edge features.
        batch_size (int): batch size.
        pre_hidden_size (int): size of the pre-GCN hidden layers.
        post_hidden_size (int): size of the post-GCN hidden layers.
        gcn_type (Literal["mpnn", "gcn", "cgcnn", "schnet"]): type of the
            graph convolution.
        gcn_hidden_size (int): size of the GCN hidden layers.
        n_pre_gcn_layers (int): number of the pre-GCN layers.
        n_post_gcn_layers (int): number of the post-GCN layers.
        n_gcn (int): number of Graph convolution.
        pool (str): which pooling to use.
        pool_order (str): which pooling order to use.
        batch_norm (bool): should the batch norm be used.
        track_running_stats (bool): should the running stats be tracked.
        activation (str): which activation to use.
        dropout (float): dropout rate.
        schnet_cutoff (float): SchNet cutoff. Default is 8.
    """

    n_features: int
    n_edge_features: int
    batch_size: int
    pre_hidden_size: int
    post_hidden_size: int
    gcn_type: Literal["mpnn", "gcn", "cgcnn", "schnet"]
    gcn_hidden_size: int
    n_pre_gcn_layers: int
    n_post_gcn_layers: int
    n_gcn: int
    pool: str
    pool_order: str
    batch_norm: bool
    track_running_stats: bool
    activation: str
    dropout: float = 0.0
    schnet_cutoff: float = 8

    def to_dict(self) -> Dict[str, Any]:
        """Returns a dictionary representation of the ConvolutionGNNParameters.

        Returns:
            dict: A dictionary representation of the ConvolutionGNNParameters.
        """
        return {
            "n_features": self.n_features,
            "n_edge_features": self.n_edge_features,
            "batch_size": self.batch_size,
            "pre_hidden_size": self.pre_hidden_size,
            "post_hidden_size": self.post_hidden_size,
            "gcn_type": self.gcn_type,
            "gcn_hidden_size": self.gcn_hidden_size,
            "n_pre_gcn_layers": self.n_pre_gcn_layers,
            "n_post_gcn_layers": self.n_post_gcn_layers,
            "n_gcn": self.n_gcn,
            "pool": self.pool,
            "pool_order": self.pool_order,
            "batch_norm": self.batch_norm,
            "track_running_stats": self.track_running_stats,
            "activation": self.activation,
            "dropout": self.dropout,
            "schnet_cutoff": self.schnet_cutoff,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConvolutionGNNParameters":
        """Constructs a ConvolutionGNNParameters from a dictionary.

        Args:
            data (Dict[str, Any]): A dictionary representation of the
                ConvolutionGNNParameters.

        Returns:
            ConvolutionGNNParameters: A ConvolutionGNNParameters object.
        """
        return cls(**data)


class GraphConvolution(nn.Module):
    """
    Message passing GNN

    Args:
        mpnn_params (ConvolutionGNNParameters): parameters defining the GCN.
    """

    def __init__(self, gcn_params: ConvolutionGNNParameters) -> None:
        super(GraphConvolution, self).__init__()
        self.params = gcn_params
        self.pre_GC_layers = self.construct_pre_GC_layers()
        self.post_GC_layers = self.construct_post_GC_layers()
        self.gcn_layers = [
            self.construct_gcn_layers() for _ in range(self.params.n_gcn)
        ]
        pool_parameters = self.construct_pool_parameters()
        self.pool_func = get_pool(gcn_params.pool, pool_parameters)
        if self.params.pool_order == "late" and self.params.pool == "set2set":
            self.post_linear = nn.Linear(2, 1)
        else:
            self.post_linear = nn.Identity()  # type: ignore

    def forward(
        self, X: Tensor, edge_idx: Tensor, edge_attr: Tensor, batch_map: Tensor
    ) -> Tensor:
        """
        Get the prediction of a batch of features.

        Args:
            X (Tensor): The input features.
            edge_idx (Tensor): The input edge indices.
            edge_attr (Tensor): The input edge attributes.
            batch_map (Tensor): The batch map.

        Returns:
            Tensor: The prediction of the model.
        """
        out: Tensor = self.pre_GC_layers(X)
        out = out.unsqueeze(0)
        for gcn_layer in self.gcn_layers:
            out = gcn_layer(out, edge_idx, edge_attr)
            out = out.squeeze(0)
        if self.params.pool_order == "early":
            out = self.pool_func(out, batch_map)
        out = self.post_GC_layers(out)
        if self.params.pool_order == "late":
            out = self.pool_func(out, batch_map)
            out = self.post_linear(out)
        return out

    def construct_pre_GC_layers(self) -> nn.Module:
        """Construct the pre-GNN layers.

        Returns:
            nn.Module: torch.nn.Module of the pre-GNN layers.
        """
        layers = nn.ModuleList()
        act = get_activation(self.params.activation)

        if self.params.n_pre_gcn_layers == 0:
            layers.append(nn.Identity())
            return nn.Sequential(*layers)

        input_size = self.params.n_features
        output_size = self.params.pre_hidden_size
        for _ in range(self.params.n_pre_gcn_layers):
            layers.append(nn.Linear(input_size, output_size))
            layers.append(act)
            input_size = output_size
            output_size = self.params.pre_hidden_size
        return nn.Sequential(*layers)

    def construct_post_GC_layers(self) -> nn.Module:
        """Construct the post-GNN layers.

        Returns:
            nn.Module: torch.nn.Module of the post-GNN layers.
        """
        layers = nn.ModuleList()
        act = get_activation(self.params.activation)
        input_size = (
            2 * self.params.post_hidden_size
            if self.params.pool == "set2set"
            else self.params.post_hidden_size
        )
        output_size = self.params.post_hidden_size
        if self.params.n_post_gcn_layers == 0:
            layers.append(nn.Linear(input_size, 1))
            return nn.Sequential(*layers)
        for _ in range(self.params.n_post_gcn_layers):
            layers.append(nn.Linear(input_size, output_size))
            layers.append(act)
            input_size = self.params.post_hidden_size
            output_size = self.params.post_hidden_size
        layers.append(nn.Linear(input_size, 1))
        return nn.Sequential(*layers)

    def construct_gcn_layers(self) -> nn.Module:
        """Construct the GCN layers.

        Returns:
            nn.ModuleList: torch.nn.ModuleList of the GCN layers.
        """
        layers = []
        act = get_activation(self.params.activation)
        gc_dim = (
            self.params.n_features
            if self.params.n_pre_gcn_layers == 0
            else self.params.pre_hidden_size
        )
        if self.params.gcn_type == "mpnn":
            edge_nn = nn.Sequential(
                nn.Linear(self.params.n_edge_features, self.params.gcn_hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.params.gcn_hidden_size, gc_dim * gc_dim),
            )
            layers.append(
                (
                    pyg_nn.NNConv(gc_dim, gc_dim, edge_nn, aggr="mean"),
                    "x, edge_index, edge_attr -> x",
                )
            )
        elif self.params.gcn_type == "gcn":
            layers.append(
                (
                    pyg_nn.GCNConv(gc_dim, gc_dim, improved=True, add_self_loops=False),
                    "x, edge_index, edge_attr -> x",
                )
            )
        elif self.params.gcn_type == "cgcnn":
            layers.append(
                (
                    pyg_nn.CGConv(
                        gc_dim,
                        self.params.n_edge_features,
                        aggr="mean",
                        batch_norm=False,
                    ),
                    "x, edge_index, edge_attr -> x",
                )
            )
        elif self.params.gcn_type == "schnet":
            layers.append(
                (
                    InteractionBlock(
                        gc_dim,
                        self.params.n_edge_features,
                        self.params.gcn_hidden_size,
                        self.params.schnet_cutoff,
                    ),
                    "x, edge_index, edge_attr -> x",
                )
            )
        else:
            raise ValueError(f"GCN type {self.params.gcn_type} not supported.")
        if self.params.batch_norm:
            layers.append(
                (
                    nn.BatchNorm1d(
                        gc_dim, track_running_stats=self.params.track_running_stats
                    ),
                    "x -> x",
                )
            )
        layers.append(act)  # type: ignore
        if self.params.dropout > 0:
            layers.append(nn.Dropout(self.params.dropout, inplace=True))  # type: ignore
        layers.append((nn.GRU(gc_dim, gc_dim), "x -> x"))
        gc_model: nn.Module = pyg_nn.Sequential("x, edge_index, edge_attr", layers)
        return gc_model

    def construct_pool_parameters(self) -> Dict[str, Any]:
        """Construct the pooling parameters.

        Returns:
            Dict[str, Any]: parameters for the pooling.
        """
        if self.params.pool == "set2set":
            if self.params.pool_order == "early":
                return {
                    "in_channels": self.params.post_hidden_size,
                    "processing_steps": 3,
                }
            else:
                return {"in_channels": 1, "processing_steps": 3, "num_layers": 1}
        else:
            return {}
