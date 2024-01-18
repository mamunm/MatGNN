"""Test the MolGraphInMemoryDataset class."""


import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[2]))
from matgnn.data.dataset.input_parameters import DatasetParameters  # noqa: E402
from matgnn.data.dataset.in_memory_dataset import MolGraphInMemoryDataset  # noqa: E402


def test_dataset_parameters_invalid_inputs() -> None:
    """Test the DatasetParameters with invalid inputs."""
    params = DatasetParameters(
        feature_type="CM",
        ase_db_loc=5
    )
    with pytest.raises(ValueError):
        MolGraphInMemoryDataset(params=params)

def test_dataset_parameters_valid_inputs() -> None:
    """Test the DatasetParameters with invalid inputs."""
    params = DatasetParameters(
        feature_type="CM",
        ase_db_loc=str(Path(__file__).parent / "dummy.db")
    )

    dataset = MolGraphInMemoryDataset(params=params)
    assert isinstance(dataset, MolGraphInMemoryDataset)
    assert dataset.len() == 1
