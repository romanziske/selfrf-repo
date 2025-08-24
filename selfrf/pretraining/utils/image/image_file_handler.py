import os
from typing import Dict, Tuple
import cv2
from git import List
import numpy as np
from torchsig.utils.file_handlers.base_handler import TorchSigFileHandler
from traitlets import Any


class ImageFileHandler(TorchSigFileHandler):

    file_ext = ".png"

    def __init__(
        self,
        root: str,
        batch_size: int = 1,
    ):
        super().__init__(
            root=root,
            batch_size=batch_size
        )
        self.images_path = f"{self.root}/images/train"

    def _setup(self) -> None:
        pass

    def write(self, batch_idx, batch):
        pass

    def exists(self) -> bool:

        images_exist = os.path.exists(self.images_path) and len(
            os.listdir(self.images_path)) > 0

        return images_exist

    @staticmethod
    def size(dataset_path: str) -> int:
        return len(os.listdir(dataset_path))

    @staticmethod
    def static_load(filename: str, idx: int) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        # Format idx with zero-padding to match COCO format (e.g., 13993 -> 000000013993)
        formatted_idx = f"{idx:012d}"  # 12 digits with leading zeros

        image = cv2.imread(
            f"{filename}/images/train/{formatted_idx}{ImageFileHandler.file_ext}",
            cv2.IMREAD_GRAYSCALE
        )
        # image = cv2.cvtColor(image, cv2.)

        image = image.astype(np.float32) / 255.0

        targets = []  # targets are not used for ssl pretraining
        return image, targets
