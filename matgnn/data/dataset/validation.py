"""Validate dataset parameters."""

from pathlib import Path

from .input_parameters import DatasetParameters


def validate_dataset_parameters(params: DatasetParameters) -> None:
    """Validate input parameters for MatGNN dataset.

    Args:
        params (DatasetParameters): Input parameters.
    """

    if params.feature_type not in ["CM", "SM"]:
        raise ValueError("feature_type must be one of the following: CM, SM.")
    if not isinstance(params.ase_db_loc, str):
        raise ValueError("ase_db_loc must be a string.")
    if not Path(params.ase_db_loc).exists():
        raise ValueError("ase_db_loc must be a valid file.")
