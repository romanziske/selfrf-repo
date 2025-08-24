"""
Dataset visualization utilities for Detectron2-based RF signal detection.
"""
from pathlib import Path
import random

import matplotlib.pyplot as plt
import torch

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.data import detection_utils as utils
from selfrf.finetuning.detection.detectron2.config import build_detectron2_config
from selfrf.finetuning.detection.detectron2.config import Detectron2Config
from selfrf.finetuning.detection.detectron2.mapper import SelfrfDatasetMapper


def visualize_dataset(config: Detectron2Config):
    """
    Visualize random samples from the wideband training dataset with bounding box annotations.

    :param config: Configuration object containing dataset paths and visualization parameters
    :type config: Detectron2Config
    :raises FileNotFoundError: If dataset catalog or metadata cannot be accessed
    :raises ValueError: If process_n_samples is invalid or dataset is empty
    """

    metadata = MetadataCatalog.get("wideband_train")
    dataset_dicts = DatasetCatalog.get("wideband_train")

    output_dir = Path(config.root) / config.dataset_path / "visualization"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Make sure we don't try to sample more than what's available
    n_samples = min(config.process_n_samples, len(dataset_dicts))
    samples = random.sample(dataset_dicts, n_samples)
    print(f"Visualizing {n_samples} samples to {output_dir}...")

    cfg = build_detectron2_config(config)

    mapper = SelfrfDatasetMapper(
        augmentations=utils.build_augmentation(
            cfg, is_train=False),
        is_train=True,
        img_format=cfg.INPUT.FORMAT,
    )

    for d in samples:
        processed_dict = mapper(d)
        img: torch.Tensor = processed_dict["image"]
        # print min max
        print(img.min(), img.max())
        print(img.shape)

        # Convert (C,H,W) to (H,W,C)
        img = img.permute(1, 2, 0)

        # **CRITICAL: Convert single channel to RGB BEFORE passing to Visualizer**
        if img.shape[2] == 1:
            img = img.repeat(1, 1, 3)  # [512, 512, 3] - now it's proper RGB
            print(f"Converted to RGB: {img.shape}")

        img_uint8 = (img*255).to(torch.uint8)

        visualizer = Visualizer(
            img_uint8,
            metadata=metadata,
            scale=1.0,
        )

        target_fields = processed_dict["instances"].get_fields()
        labels = [metadata.thing_classes[i]
                  for i in target_fields["gt_classes"]]

        vis = visualizer.overlay_instances(
            labels=labels,
            boxes=target_fields.get("gt_boxes", None),
        )

        # Save visualization
        vis_img = vis.get_image()
        plt.imsave(
            output_dir / processed_dict["file_name"].split("/")[-1], vis_img)
        print(processed_dict["file_name"].split("/")[-1])
