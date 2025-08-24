import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Callable, List
from pathlib import Path

from selfrf.pretraining.utils.image.image_dataset import CustomStaticTorchSigDataset
from selfrf.pretraining.utils.image.image_file_handler import ImageFileHandler
from torchsig.datasets.narrowband import StaticNarrowband


class ImageDataModule(pl.LightningDataModule):

    def __init__(
        self,
        root: str,
        batch_size: int = 32,
        num_workers: int = 4,
        transforms: List[Callable] = None,
        target_transforms: List[Callable] = None,
        collate_fn: Callable = None,
    ):
        super().__init__()

        self.root = Path(root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transforms = transforms or []
        self.target_transforms = target_transforms or []
        self.collate_fn = collate_fn

        # Initialize file handler
        self.file_handler = ImageFileHandler(
            root=str(self.root),
            batch_size=batch_size
        )

        # Will be set in setup()
        self.train_dataset: StaticNarrowband = None

        # Check if dataset exists
        if not self.file_handler.exists():
            raise FileNotFoundError(
                f"No images found in {self.file_handler.images_path}")

    def prepare_data(self) -> None:

        if not self.file_handler.exists():
            raise FileNotFoundError(
                f"Images not found at {self.file_handler.images_path}")

        num_images = len(
            [f for f in Path(self.file_handler.images_path).glob("*.png")])
        print(f"Found {num_images} images in {self.file_handler.images_path}")

    def setup(self, stage: str = None) -> None:

        # Create datasets
        self.train_dataset = CustomStaticTorchSigDataset(
            root=str(self.root),
            transforms=self.transforms,
            target_transforms=self.target_transforms,
            batch_size=self.batch_size
        )

    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )
