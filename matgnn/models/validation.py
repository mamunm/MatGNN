from .input_parameters import MLPParameters


def validate_mlp_parameters(params: MLPParameters) -> None:
    """Validate parameters for MLP.

    Args:
        params (MLPParameters): Input parameters.
    """

    if params.n_hidden_layers is None:
        raise ValueError("n_hidden_layers must be specified.")
    if not isinstance(params.n_hidden_layers, int):
        raise ValueError("n_hidden_layers must be an integer.")
    if not isinstance(params.base2, bool):
        raise ValueError("base2 must be a boolean.")
    if params.activation is None:
        raise ValueError("activation must be specified.")
    if params.activation_parameters is None:
        raise ValueError("activation_parameters must be specified.")
    if not isinstance(params.layer_norm, bool):
        raise ValueError("layer_norm must be a boolean.")
