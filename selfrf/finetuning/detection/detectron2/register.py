"""
Dataset registration utilities for Detectron2 COCO format datasets.
"""
from pathlib import Path
from detectron2.data.datasets import register_coco_instances

from selfrf.finetuning.detection.detectron2.config import Detectron2Config


def register_dataset(config: Detectron2Config):
    """
    Registers custom COCO format RF signal dataset with Detectron2.

    :param config: Configuration object containing dataset paths
    :type config: Detectron2Config
    :raises FileNotFoundError: If dataset annotation or image files not found
    :raises ValueError: If dataset registration fails
    """
    root = Path(config.root)
    dataset_path = Path(config.dataset_path)
    path_to_coco = root / dataset_path / "coco"
    dataset_name = "wideband"

    # Register datasets
    register_coco_instances(
        f"{dataset_name}_train",
        {},
        str(path_to_coco / "train" / "_annotations.coco.json"),
        str(path_to_coco / "train"),
    )

    register_coco_instances(
        f"{dataset_name}_val",
        {},
        str(path_to_coco / "valid" / "_annotations.coco.json"),
        str(path_to_coco / "valid"),
    )

    print("Dataset registered successfully!")
