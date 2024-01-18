"""In memory dataset for molecular graph data."""

from pathlib import Path
from typing import Any, Dict, List, Union

from ase import db
from dscribe.descriptors import CoulombMatrix, SineMatrix
from torch_geometric.data import Data, InMemoryDataset

from .input_parameters import DatasetParameters
from .utils import get_dir_name
from .validation import validate_dataset_parameters


class MolGraphInMemoryDataset(InMemoryDataset):  # type: ignore
    """InMemory dataset for molecular graph data.

    Args:
        params (DatasetParameters): inputs to the dataset
    """

    def __init__(self, params: DatasetParameters) -> None:
        validate_dataset_parameters(params)
        self.params = params
        root = Path(params.ase_db_loc).parent / get_dir_name(params)
        if not root.exists():
            root.mkdir()
        super(InMemoryDataset, self).__init__(root=root)
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self) -> List[str]:
        """Processed file names."""
        return ["data.pt"]

    def process(self) -> None:
        """Process data."""
        data_list = []

        ase_db = db.connect(self.params.ase_db_loc)

        feature_constructor = self.get_feature_constructor()

        for struct in ase_db.select():
            x = feature_constructor.create(struct.toatoms())
            data_list.append(Data(x=x))

        self.save(data_list, self.processed_paths[0])

    def get_data_statistics(self) -> Dict[str, Any]:
        """Get data statistics.

        Args:
            ase_db (db): ASE database.
        """
        ase_db = db.connect(self.params.ase_db_loc)
        max_natoms = max(d.natoms for d in ase_db.select())
        return {"max_natoms": max_natoms}

    def get_feature_constructor(self) -> Union[SineMatrix, CoulombMatrix]:
        """Get feature.

        Returns:
            Tensor: Feature.
        """
        data_statistics = self.get_data_statistics()
        if self.params.feature_type == "CM":
            return CoulombMatrix(
                n_atoms_max=data_statistics["max_natoms"],
                permutation="eigenspectrum",
                sparse=False,
            )
        else:
            return SineMatrix(
                n_atoms_max=data_statistics["max_natoms"],
                permutation="eigenspectrum",
                sparse=False,
            )
