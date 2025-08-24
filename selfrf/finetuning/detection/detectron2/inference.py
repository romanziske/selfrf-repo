"""
Inference utilities for running detection models on RF signal datasets.
"""
from pathlib import Path
import random
import logging

import matplotlib.pyplot as plt
import torch
import numpy as np

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.modeling import build_model

from selfrf.finetuning.detection.detectron2.config import Detectron2Config, build_detectron2_config
from selfrf.finetuning.detection.detectron2.register import register_dataset
from selfrf.finetuning.detection.detectron2.mapper import SelfrfDatasetMapper

logger = logging.getLogger("detectron2")


def inference_dataset(config: Detectron2Config):
    """
    Runs inference on validation dataset and creates side-by-side visualizations of predictions vs ground truth.

    :param config: Configuration object containing model paths, dataset parameters, and inference settings
    :type config: Detectron2Config
    :raises FileNotFoundError: If model checkpoint file is not found
    :raises RuntimeError: If model loading or inference execution fails
    :raises ValueError: If dataset registration or sample processing fails
    """
    register_dataset(config)

    cfg = build_detectron2_config(config)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config.inference_threshold
    model = build_model(cfg)

    # move model to GPU
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model.to(device)
    model.eval()

    metadata = MetadataCatalog.get("wideband_val")
    dataset_dicts = DatasetCatalog.get("wideband_val")

    output_dir = Path(config.root) / \
        Path(config.dataset_path) / "inference"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Make sure we don't try to sample more than what's available
    n_samples = min(config.process_n_samples, len(dataset_dicts))
    samples = random.sample(dataset_dicts, k=n_samples)
    print(f"Visualizing {n_samples} with inference samples to {output_dir}...")

    mapper = SelfrfDatasetMapper(
        is_train=True,
        img_format=cfg.INPUT.FORMAT,
        normalize=True,
    )

    for d in samples:
        processed_dict = mapper(d)
        img: torch.Tensor = processed_dict["image"]

        # Get original ground truth annotations
        gt_boxes = processed_dict.get("instances", None)

        # Prepare input in the format expected by the model
        height, width = img.shape[1], img.shape[2]  # Get image dimensions

        # Create input dict in the format expected by Detectron2 models
        inputs = [{
            "image": img,
            "height": height,
            "width": width
        }]

        # Run inference
        with torch.no_grad():
            outputs = model(inputs)[0]

        # Convert (C,H,W) to (H,W,C)
        img_np = img.permute(1, 2, 0).cpu().numpy()
        img_uint8 = (img_np * 255).astype(np.uint8)

        # Create a figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Plot predictions on the left
        visualizer_pred = Visualizer(
            img_uint8.copy(),
            metadata=metadata,
            scale=1.0,
        )
        out_pred = visualizer_pred.draw_instance_predictions(
            outputs["instances"].to("cpu"))
        pred_img = out_pred.get_image()
        ax1.imshow(pred_img)
        ax1.set_title("Predictions", fontsize=16)
        ax1.axis('off')

        # Plot ground truth on the right
        visualizer_gt = Visualizer(
            img_uint8.copy(),
            metadata=metadata,
            scale=1.0,
        )

        if gt_boxes is not None:
            out_gt = visualizer_gt.draw_dataset_dict(processed_dict)
            gt_img = out_gt.get_image()
        else:
            # If no ground truth is available, just show the original image
            gt_img = img_uint8

        ax2.imshow(gt_img)
        ax2.set_title("Ground Truth", fontsize=16)
        ax2.axis('off')

        # Add a main title
        fig.suptitle(
            f"Detection Results - {processed_dict['file_name'].split('/')[-1]}", fontsize=18)
        plt.tight_layout()

        # Save the combined visualization
        output_filename = output_dir / \
            f"comparison_{processed_dict['file_name'].split('/')[-1]}"
        plt.savefig(output_filename, dpi=150)
        plt.close(fig)

        print(f"Saved comparison: {output_filename}")
