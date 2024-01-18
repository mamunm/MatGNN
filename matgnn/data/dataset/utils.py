"""Utility funtions for datasets."""

from .input_parameters import DatasetParameters


def get_dir_name(dataset_params: DatasetParameters) -> str:
    """Get directory name.

    Args:
        dataset_params (DatasetParameters): Dataset parameters.

    Returns:
        str: Directory name.
    """
    return f"{dataset_params.feature_type}"
