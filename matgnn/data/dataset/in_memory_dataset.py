"""In memory dataset for molecular graph data."""

from pathlib import Path
from typing import Dict, List, Union

import torch
from ase import db
from dscribe.descriptors import ACSF, SOAP, CoulombMatrix, SineMatrix
from torch_geometric.data import Data, InMemoryDataset

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
        self.n_features = self._data.x.shape[1]

    @property
    def processed_file_names(self) -> List[str]:
        """Processed file names."""
        return ["data.pt"]

    def process(self) -> None:
        """Process data."""
        data_list = []
        dtype = DTYPE_MAP[self.params.dtype]

        ase_db = db.connect(self.params.ase_db_loc)

        feature_constructor = self.get_feature_constructor()

        for struct in ase_db.select():
            y = getattr(struct, self.params.target)
            x = feature_constructor.create(struct.toatoms())
            data_list.append(
                Data(
                    x=torch.tensor(x, dtype=dtype).reshape(1, -1),
                    y=torch.tensor(y, dtype=dtype),
                )
            )

        self.save(data_list, self.processed_paths[0])

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
        elif self.params.feature_type == "SOAP":
            return SOAP(
                species=data_statistics["species"],
                periodic=self.params.extra_parameters["periodic"],
                n_max=self.params.extra_parameters["n_max"],
                l_max=self.params.extra_parameters["l_max"],
                r_cut=self.params.extra_parameters["r_cut"],
                sigma=self.params.extra_parameters["sigma"],
                sparse=False,
                average="inner",
                rbf="gto",
            )
        else:
            return ACSF(
                species=data_statistics["species"],
                periodic=self.params.extra_parameters["periodic"],
                r_cut=self.params.extra_parameters["r_cut"],
                g2_params=self.params.extra_parameters["g2_params"],
                g3_params=self.params.extra_parameters["g3_params"],
                g4_params=self.params.extra_parameters["g4_params"],
                g5_params=self.params.extra_parameters["g5_params"],
                sparse=False,
            )
