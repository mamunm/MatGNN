"""Input Parameters for models Module."""

from typing import Any, Dict, NamedTuple


class MLPParameters(NamedTuple):
    """Inpute parameters for the MatGNN MLP model.

    Args:
        n_hidden_layers (int): number of the hidden layers.
        base2 (bool): should the number of neurons be halved in each layer.
        activation (str): which activation to use.
        activation_parameters (dict): parameters for the activation.
        layer_norm (bool): should the layer norm be used.
    """

    n_hidden_layers: int
    base2: bool
    activation: str
    activation_parameters: Dict[str, Any]
    layer_norm: bool

    def to_dict(self) -> Dict[str, Any]:
        """Returns a dictionary representation of the MLPParameters.

        Returns:
            dict: A dictionary representation of the MLPParameters.
        """
        return {
            "n_hidden_layers": self.n_hidden_layers,
            "base2": self.base2,
            "activation": self.activation,
            "activation_parameters": self.activation_parameters,
            "layer_norm": self.layer_norm,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MLPParameters":
        """Creates a MLPParameters object from a dictionary.

        Args:
            data (dict): A dictionary representation of the MLPParameters.

        Returns:
            MPLParameters: A MLPParameters object.
        """
        return cls(
            n_hidden_layers=data.get("n_hidden_layers"),  # type: ignore
            base2=data.get("base2"),  # type: ignore
            activation=data.get("activation"),  # type: ignore
            activation_parameters=data.get("activation_parameters"),  # type: ignore
            layer_norm=data.get("layer_norm"),  # type: ignore
        )
