"""Input parameters for dataset."""

from typing import Any, Dict, Literal, NamedTuple


class DatasetParameters(NamedTuple):
    """Input parameters for the dataset.

    Args:
        feature_type (Literal["CM", "SM"]): The type of the feature.
        ase_db_loc (str): The location of the ASE database.
    """

    feature_type: Literal["CM", "SM"]
    ase_db_loc: str

    def to_dict(self) -> Dict[str, Any]:
        """Returns a dictionary representation of the DatasetParameters.

        Returns:
            dict: A dictionary representation of the DatasetParameters.
        """
        return {
            "feature_type": self.feature_type,
            "ase_db_loc": self.ase_db_loc,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetParameters":
        """Creates a DatasetParameters object from a dictionary.

        Args:
            data (dict): A dictionary representation of the DatasetParameters.

        Returns:
            DatasetParameters: A DatasetParameters object.
        """
        return cls(
            feature_type=data.get("feature_type", None),
            ase_db_loc=data.get("ase_db_loc", None),
        )
