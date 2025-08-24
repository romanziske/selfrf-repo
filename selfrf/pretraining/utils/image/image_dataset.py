
from pathlib import Path
from typing import Tuple, TYPE_CHECKING
import yaml
import numpy as np
from torch.utils.data import Dataset
from torchsig.signals.signal_types import DatasetSignal, DatasetDict
from torchsig.utils.verify import verify_transforms, verify_target_transforms
from torchsig.utils.file_handlers.base_handler import TorchSigFileHandler
from torchsig.utils.file_handlers.zarr import ZarrFileHandler
from torchsig.datasets.dataset_utils import dataset_yaml_name, writer_yaml_name, to_dataset_metadata

from selfrf.pretraining.utils.image.image_file_handler import ImageFileHandler


class CustomStaticTorchSigDataset(Dataset):

    def __init__(
        self,
        root: str,
        transforms: list = [],
        target_transforms: list = [],
        batch_size: int = 32,
        # **kwargs
    ):
        self.root = Path(root)
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.file_handler = ImageFileHandler(
            root=str(self.root),
            batch_size=batch_size  # Single sample loading
        )

        # dataset size
        self.num_samples = self.file_handler.size(
            self.file_handler.images_path)
        print(self.num_samples, "samples found in dataset")

        self.raw = True  # Assume raw data by default

        self._verify()

    def _verify(self):
        # Transforms
        self.transforms = verify_transforms(self.transforms)

        # Target Transforms
        self.target_transforms = verify_target_transforms(
            self.target_transforms)

    def __len__(self) -> int:
        """Returns the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Tuple]:
        """Retrieves a sample from the dataset by index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            Tuple[np.ndarray, Tuple]: The data and targets for the sample.

        Raises:
            IndexError: If the index is out of bounds.
        """
        if idx >= 0 and idx < self.__len__():
            # load data and metadata
            # data: np.ndarray
            # signal_metadatas: List[dict]
            if self.raw:
                # loading in raw IQ data and signal metadata
                data, signal_metadatas = self.file_handler.static_load(
                    self.root, idx)

                # convert to DatasetSignal
                sample = DatasetSignal(
                    data=data,
                    signals=signal_metadatas,
                    dataset_metadata=None,  # No dataset metadata for custom datasets
                )

                # apply user transforms
                for t in self.transforms:
                    sample = t(sample)

                # convert to DatasetDict
                sample = DatasetDict(signal=sample)

                # apply target transforms
                targets = [0]

                return sample.data, targets
            # else:
            # loading in transformed data and targets from target transform
            data, targets = self.file_handler.static_load(self.root, idx)

            return data, targets

        else:
            raise IndexError(
                f"Index {idx} is out of bounds. Must be [0, {self.__len__()}]")

    def __str__(self) -> str:
        return f"{self.__class__.__name__}: {self.root}"

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(root={self.root}, "
            f"transforms={self.transforms.__repr__()}, "
            f"target_transforms={self.target_transforms.__repr__()}, "
            f"file_handler_class={self.file_handler}, "
        )
