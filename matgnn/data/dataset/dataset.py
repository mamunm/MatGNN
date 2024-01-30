"""Dataset for molecular graph data."""

from pathlib import Path
from typing import Dict, List, Union

import torch
from ase import db
from dscribe.descriptors import SOAP, CoulombMatrix, SineMatrix
from torch_geometric.data import Data, Dataset

from .atom_graph import get_atom_graph_data
from .utils import (
    DatasetParameters,
    get_data_statistics,
    get_dir_name,
    validate_dataset_parameters,
)

DTYPE_MAP: Dict[str, torch.dtype] = {
    "f64": torch.float64,
    "f32": torch.float32,
    "f16": torch.float16,
    "bf16": torch.bfloat16,
}


class MolGraphDataset(Dataset):  # type: ignore
    """Dataset for molecular graph data.

    Args:
        params (DatasetParameters): inputs to the dataset
    """

    def __init__(self, params: DatasetParameters) -> None:
        validate_dataset_parameters(params)
        self.params = params
        root = Path(params.ase_db_loc).parent / get_dir_name(params)
        if not root.exists():
            root.mkdir()
        self.process_has_been_called = False
        super(MolGraphDataset, self).__init__(root=root)
        if not self.process_has_been_called:
            print("Processing...")
            self.process()
            print("Done!")

    @property
    def processed_file_names(self) -> List[str]:
        """Processed file names."""
        return list(map(str, Path(self.processed_dir).glob("*.pt")))

    def len(self) -> int:
        """Length of dataset."""
        return len(self.processed_file_names)

    def get(self, idx: int) -> Data:
        """Get data by index."""
        return torch.load(Path(self.processed_dir) / f"data_{idx}.pt")

    def process(self) -> None:
        """Process data."""
        self.process_has_been_called = True
        if self.params.feature_type in ["CM", "SM", "SOAP"]:
            data_list = []
            dtype = DTYPE_MAP[self.params.dtype]

            ase_db = db.connect(self.params.ase_db_loc)

            feature_constructor = self.get_feature_constructor()

            for i, struct in enumerate(ase_db.select()):
                y = getattr(struct, self.params.target)
                x = feature_constructor.create(struct.toatoms())
                data_list.append(
                    Data(
                        x=torch.tensor(x, dtype=dtype).reshape(1, -1),
                        y=torch.tensor(y, dtype=dtype),
                    )
                )
        else:
            data_list = get_atom_graph_data(
                self.params.ase_db_loc,
                self.params.target,
                self.params.extra_parameters,  # type: ignore
                dtype=DTYPE_MAP[self.params.dtype],
            )

        for idx, data in enumerate(data_list):
            torch.save(data, Path(self.processed_dir) / f"data_{idx}.pt")

    def get_feature_constructor(self) -> Union[SineMatrix, CoulombMatrix]:
        """Get feature.

        Returns:
            Tensor: Feature.
        """
        data_statistics = get_data_statistics(self.params.ase_db_loc)
        if self.params.feature_type == "CM":
            return CoulombMatrix(
                n_atoms_max=data_statistics["max_natoms"],
                permutation="eigenspectrum",
                sparse=False,
            )
        elif self.params.feature_type == "SM":
            return SineMatrix(
                n_atoms_max=data_statistics["max_natoms"],
                permutation="eigenspectrum",
                sparse=False,
            )
        else:
            return SOAP(
                species=data_statistics["species"],
                periodic=self.params.extra_parameters.periodic,  # type: ignore
                n_max=self.params.extra_parameters.n_max,  # type: ignore
                l_max=self.params.extra_parameters.l_max,  # type: ignore
                r_cut=self.params.extra_parameters.r_cut,  # type: ignore
                sigma=self.params.extra_parameters.sigma,  # type: ignore
                sparse=False,
                average="inner",
                rbf="gto",
            )
