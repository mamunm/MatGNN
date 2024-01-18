"""Input parameters for MatGNN data module."""

from typing import Any, Dict, NamedTuple

from .dataset.input_parameters import DatasetParameters


class DataModuleParameters(NamedTuple):
    """Inpute parameters for the MatGNN data module.

    Args:
        dataset_params (DatasetParameters): dataset parameters.
        batch_size (Optional[int]): The batch size. Defaults to 64.
        num_workers (Optional[int]): The number of workers. Defaults to 1.
        test_ratio (float): The ratio of testing to training.
    """

    dataset_params: DatasetParameters
    batch_size: int = 64
    num_workers: int = 1
    test_ratio: float = 0.2

    def to_dict(self) -> Dict[str, Any]:
        """Returns a dictionary representation of the DataModuleParameters.

        Returns:
            dict: A dictionary representation of the DataModuleParameters.
        """
        return {
            "dataset_params": self.dataset_params.to_dict(),
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "test_ratio": self.test_ratio,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataModuleParameters":
        """Creates a DataModuleParameters object from a dictionary.

        Args:
            data (dict): A dictionary representation of the DataModuleParameters.

        Returns:
            DataModuleParameters: A DataModuleParameters object.
        """
        return cls(
            dataset_params=DatasetParameters.from_dict(
                data.get("dataset_params", {"feature_type": "CM"})
            ),
            batch_size=data.get("batch_size", 64),
            num_workers=data.get("num_workers", 1),
            test_ratio=data.get("test_ratio", 0.2),
        )
