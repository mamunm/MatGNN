"""Convolution GNN for MatGNN."""

from typing import Any, Dict, Literal, NamedTuple

import torch
import torch_geometric.nn as pyg_nn
from torch import Tensor, nn
from torch_geometric.nn.models.schnet import InteractionBlock

from .utils import get_activation, get_pool

DTYPE_MAP: Dict[str, torch.dtype] = {
    "64": torch.float64,
    "32": torch.float32,
    "16": torch.float16,
    "bf16": torch.bfloat16,
}


class GraphConvolutionParameters(NamedTuple):
    """Inpute parameters for the GraphConvolution model.

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
        dtype (Literal["f64", "f32", "f16", "bf16"]): data type of the model.
            Defaults to "f64".
        device (Literal["cpu", "gpu"]): device to use. Defaults to "cpu".
    """

    n_features: int
    n_edge_features: int
    batch_size: int
    pre_hidden_size: int = 64
    post_hidden_size: int = 64
    gcn_type: Literal["gcn", "cgcnn", "schnet"] = "gcn"
    gcn_hidden_size: int = 64
    n_pre_gcn_layers: int = 3
    n_post_gcn_layers: int = 3
    n_gcn: int = 3
    pool: str = "set2set"
    pool_order: str = "early"
    batch_norm: bool = False
    track_running_stats: bool = False
    activation: str = "relu"
    dropout: float = 0.0
    schnet_cutoff: float = 8
    dtype: Literal["f64", "f32", "f16", "bf16"] = "f64"
    device: Literal["cpu", "gpu"] = "cpu"

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
            "dtype": self.dtype,
            "device": self.device,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphConvolutionParameters":
        """Constructs a GraphConvolutionParameters from a dictionary.

        Args:
            data (Dict[str, Any]): A dictionary representation of the
                GraphConvolutionParameters.

        Returns:
            GraphConvolutionParameters: A GraphConvolutionParameters object.
        """
        return cls(**data)


class GraphConvolution(nn.Module):
    """
    Graph Convolution Network.

    Args:
        mpnn_params (GraphConvolutionParameters): parameters defining the GCN.
    """

    def __init__(self, gcn_params: GraphConvolutionParameters) -> None:
        super(GraphConvolution, self).__init__()
        self.params = gcn_params
        self.pre_GC_layers = self.construct_pre_GC_layers()
        self.post_GC_layers = self.construct_post_GC_layers()
        self.gcn_layers = [
            self.construct_gcn_layers() for _ in range(self.params.n_gcn)
        ]
        for gcn_layer in self.gcn_layers:
            dtype = DTYPE_MAP[gcn_params.dtype]
            device = "cpu" if gcn_params.device == "cpu" else "cuda:0"
            gcn_layer = gcn_layer.to(dtype=dtype, device=device)  # type: ignore
        pool_parameters = self.construct_pool_parameters()
        self.pool_func = get_pool(gcn_params.pool, pool_parameters)
        if self.params.pool_order == "late" and self.params.pool == "set2set":
            self.post_linear = nn.Linear(2, 1)
        else:
            self.post_linear = nn.Identity()  # type: ignore

    def forward(
        self,
        X: Tensor,
        edge_idx: Tensor,
        edge_weight: Tensor,
        edge_attr: Tensor,
        batch_map: Tensor,
    ) -> Tensor:
        """
        Get the prediction of a batch of features.

        Args:
            X (Tensor): The input features.
            edge_idx (Tensor): The input edge indices.
            edge_weight (Tensor): The input edge weights.
            edge_attr (Tensor): The input edge attributes.
            batch_map (Tensor): The batch map.

        Returns:
            Tensor: The prediction of the model.
        """
        out: Tensor = self.pre_GC_layers(X)
        for gcn_layer in self.gcn_layers:
            if self.params.gcn_type in ["schnet"]:
                out = gcn_layer(out, edge_idx, edge_weight, edge_attr.view(-1, 1))
            else:
                out = gcn_layer(out, edge_idx, edge_attr.view(-1, 1))
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
            2 * self.params.pre_hidden_size
            if self.params.pool == "set2set"
            else self.params.pre_hidden_size
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
        if self.params.gcn_type == "gcn":
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
                    "x, edge_index, edge_weight, edge_attr -> x",
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
        if self.params.gcn_type == "schnet":
            gc_model: nn.Module = pyg_nn.Sequential(
                "x, edge_index, edge_weight, edge_attr", layers
            )
        else:
            gc_model = pyg_nn.Sequential("x, edge_index, edge_attr", layers)
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
