"""
Converts TorchSig datasets to COCO format for object detection training on RF spectrograms.

TorchSig provides PyTorch-based RF signal datasets with realistic modulation schemes 
and channel effects. This converter transforms TorchSig's tensor-based format into 
COCO-compatible annotations for training YOLO and other detection models.

Conversion Modes
----------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Mode
     - Description
   * - **detection**
     - Binary signal detection. All modulated signals mapped to single 'signal' class.
   * - **modulation**
     - Multi-class modulation recognition (BPSK, QPSK, QAM16, OFDM, etc.).
   * - **protocol**
     - Protocol-level classification (WiFi, Bluetooth, LTE, 5G-NR).
   * - **interference**
     - Detection and classification of interference patterns.

CLI Usage
---------

.. code-block:: bash

    # Convert for modulation recognition
    python torchsig_to_coco.py \
           --torchsig_dir ./torchsig_dataset \
           --out_dir ./coco_output \
           --mode modulation \
           --train_split 0.8

    # Convert for signal detection with custom spectrogram parameters
    python torchsig_to_coco.py \
           --torchsig_dir ./torchsig_dataset \
           --out_dir ./coco_output \
           --mode detection \
           --fft_size 1024 \
           --overlap 0.5 \
           --generate_spectrograms

    # Convert for protocol classification
    python torchsig_to_coco.py \
           --torchsig_dir ./torchsig_dataset \
           --out_dir ./coco_output \
           --mode protocol \
           --min_snr -10 \
           --max_snr 30

API Reference
-------------
"""
import os
import json
from pathlib import Path
from typing import Literal
import concurrent.futures
from functools import partial

from matplotlib import pyplot as plt
import torch
from torch.utils.data import random_split
from tqdm import tqdm

from torchsig.datasets.datamodules import WidebandDataModule
from torchsig.datasets.wideband import StaticWideband, StaticTorchSigDataset
from torchsig.datasets.dataset_utils import to_dataset_metadata

from torchsig.transforms.target_transforms import (
    ClassName,
    FamilyName,
    ClassIndex,
    FamilyIndex,
    SNR,
)
from selfrf.pretraining.config.base_config import BaseConfig

from selfrf.pretraining.utils.enums import DatasetType
from selfrf.transforms.extra.target_transforms import BBOXLabel, ConstantFamilyName, ConstantSignalIndex, ConstantSignalName
from selfrf.pretraining.factories.transform_factory import TransformFactory
from selfrf.pretraining.factories.dataset_factory import get_wideband_metadata


def get_target_transforms(
    mode: Literal["detection", "recognition",
                  "family_recognition"] = "detection",
) -> list:
    """Get target transform list for detectron2 based on the specified mode.

    :param mode: The task mode determining target transforms
    :returns: List of target transform objects
    """

    if mode == "detection":
        return [
            BBOXLabel(),  # bbox
            ConstantSignalName("signal"),  # category name
            ConstantSignalIndex(0),  # category index (now starts at 0)
            ConstantFamilyName("signal"),  # super category name
            SNR(),  # SNR
        ]
    elif mode == "recognition":
        return [
            BBOXLabel(),  # bbox
            ClassName(),  # category name
            ClassIndex(),  # category index (will be adjusted to start at 0)
            FamilyName(),  # super category name
            SNR(),  # SNR
        ]
    elif mode == "family_recognition":
        return [
            BBOXLabel(),  # bbox
            FamilyName(),  # category name
            FamilyIndex(),  # category index (will be adjusted to start at 0)
            FamilyName(),  # super category name
            SNR(),  # SNR
        ]


def process_sample(
        idx,
        dataset: StaticTorchSigDataset,
        split_dir: Path,
        split: Literal["train", "valid", "test"]
) -> tuple[dict, list, dict]:
    """Process a single sample from the dataset for COCO format conversion.

    :param idx: Index of the sample in the dataset
    :param dataset: The source dataset containing spectrograms and labels
    :param split_dir: Directory path where the spectrogram image will be saved
    :param split: Dataset split identifier used in filename
    :returns: Tuple containing image info, sample annotations, and sample categories
    """
    spectrogram, labels = dataset.__getitem__(idx)

    filename = f"{idx:010d}.png"
    image_path = split_dir / filename

    plt.imsave(str(image_path), spectrogram, cmap='gray')

    height, width = spectrogram.shape
    # Prepare image info
    image_info = {
        "id": idx,
        "file_name": filename,  # Just the filename, no directory prefix
        "width": width,
        "height": height
    }

    # Prepare annotations
    sample_annotations = []
    sample_categories = {}

    for label in labels:
        bbox = label[0]
        class_name = label[1]
        class_id = label[2]  # Keep as 0-indexed (removed +1)
        family_name = label[3]
        snr = label[4]

        sample_categories[class_id] = {
            "id": class_id,
            "name": class_name,
            "supercategory": family_name
        }

        # convert bbox to pixel coordinates
        x = bbox[0] * width
        y = bbox[1] * height
        w = bbox[2] * width
        h = bbox[3] * height
        bbox = [x, y, w, h]
        area = w * h

        sample_annotations.append({
            "bbox": bbox,
            "category_id": class_id,
            "area": area,
            "snr": snr,
        })

    return image_info, sample_annotations, sample_categories


def store_spectrograms(
    dataset: StaticWideband,
    split_dir: Path,
    split: Literal["train", "valid", "test"],
    max_workers: int = 8
) -> tuple[list[dict], list[dict], dict[int, dict]]:
    """Process a dataset of RF signals and convert them to COCO format spectrograms.

    :param dataset: The RF signal dataset to process
    :param split_dir: Split-specific directory path
    :param split: Dataset split identifier for the output files
    :param max_workers: Maximum number of worker threads for parallel processing
    :returns: Tuple containing images, annotations, and categories dictionaries
    """
    # Ensure split directory exists
    split_dir.mkdir(parents=True, exist_ok=True)

    images = []
    annotations = []
    categories = {}
    global_annotation_idx = 0  # global index for annotations

    # Create a partial function with fixed parameters
    process_func = partial(process_sample, dataset=dataset,
                           split_dir=split_dir, split=split)

    # Process samples in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_func, idx)
                                   : idx for idx in range(len(dataset))}

        for future in tqdm(concurrent.futures.as_completed(futures),
                           total=len(futures),
                           desc=f"Processing {split} samples"):
            idx = futures[future]
            image_info, sample_annotations, sample_categories = future.result()

            if image_info is None:
                continue

            images.append(image_info)

            # Add annotations with proper IDs
            for ann in sample_annotations:
                ann["id"] = global_annotation_idx
                ann["image_id"] = idx
                ann["iscrowd"] = 0
                annotations.append(ann)
                global_annotation_idx += 1

            # Update categories
            categories.update(sample_categories)

    return images, annotations, categories


def convert_dataset_to_coco(
    dataset: StaticWideband,
    path_to_coco: Path,
    split: Literal["train", "valid", "test"],
):
    """Convert a StaticWideband dataset to COCO format for object detection.

    :param dataset: The source dataset containing signal data to be converted
    :param path_to_coco: The output directory path where COCO format files will be saved
    :param split: The dataset split type: "train", "valid", or "test"
    """
    print(f"Converting {split} dataset to COCO format at", path_to_coco)

    # Create split-specific directory
    split_dir = path_to_coco / split
    split_dir.mkdir(parents=True, exist_ok=True)

    # Initialize COCO JSON structure
    coco_json = {
        "info": {},
        "categories": [],
        "images": [],
        "annotations": [],
    }

    # Store images and process annotations
    images, annotations, categories = store_spectrograms(
        dataset, split_dir, split)
    coco_json["images"] = images
    coco_json["annotations"] = annotations
    coco_json["categories"] = list(categories.values())

    # Save COCO JSON in the split directory with the specific name
    annotations_file = split_dir / "_annotations.coco.json"
    with open(annotations_file, "w") as f:
        json.dump(coco_json, f, indent=4)

    print(f"✓ Created {len(images)} images in {split}/")
    print(
        f"✓ Created {len(annotations)} annotations in {split}/_annotations.coco.json")


def convert_datamodule_to_coco(
    datamodule: WidebandDataModule,
    force: bool = False,
    include_test: bool = False,
) -> Path:
    """Convert a WidebandDataModule to COCO format for object detection.

    :param datamodule: The datamodule containing train and validation datasets to be converted
    :param force: If True, forces conversion even if COCO format already exists
    :param include_test: If True, creates a test split by splitting validation 50/50
    :returns: Path to the created COCO format directory
    """
    path_to_coco = datamodule.root / "coco"

    if not force and path_to_coco.exists():
        print("COCO format already exists at", path_to_coco)
        return path_to_coco

    path_to_coco.mkdir(parents=True, exist_ok=True)
    print("Converting datamodule to COCO format at", path_to_coco)

    # Convert training set
    convert_dataset_to_coco(datamodule.train, path_to_coco, split="train")

    if include_test:
        # Split validation set 50/50 into val and test
        val_dataset = datamodule.val
        val_size = len(val_dataset)

        # Calculate split sizes (50/50)
        test_size = val_size // 2
        val_new_size = val_size - test_size

        print(
            f"Splitting validation set: {val_size} samples -> {val_new_size} val + {test_size} test")

        # Create random split
        generator = torch.Generator().manual_seed(42)  # For reproducibility
        val_subset, test_subset = random_split(
            val_dataset,
            [val_new_size, test_size],
            generator=generator
        )

        # Convert the subsets
        convert_dataset_to_coco(val_subset, path_to_coco, split="valid")
        convert_dataset_to_coco(test_subset, path_to_coco, split="test")
    else:
        # Use entire validation set as validation
        convert_dataset_to_coco(datamodule.val, path_to_coco, split="valid")

    return path_to_coco


def torchsig_to_coco(
    input_dir: Path,
    nfft: int,
    mode: Literal["detection", "recognition",
                  "family_recognition"] = "detection",
    include_test: bool = True,
    val_test_split: float = 0.5,
):
    """Convert TorchSig wideband dataset to COCO format for object detection tasks.

    :param input_dir: Path to the input directory containing the TorchSig dataset
    :param nfft: Number of FFT points for spectrogram computation
    :param mode: The task mode that determines target transforms
    :param include_test: If True, creates a test split by splitting validation
    :param val_test_split: Fraction of validation data to use for test (0.5 = 50/50 split)
    """
    base_config = BaseConfig(
        dataset=DatasetType.TORCHSIG_WIDEBAND,
        nfft=nfft,
        spectrogram=True,
    )

    metadata = get_wideband_metadata(base_config)
    print(json.dumps(metadata, indent=4))

    metadata = to_dataset_metadata(metadata)

    spectrogram_transform = TransformFactory.create_spectrogram_transform(
        config=base_config,
        to_tensor=False,
    )

    datamodule = WidebandDataModule(
        root=input_dir,
        dataset_metadata=metadata,
        num_samples_train=base_config.num_samples,
        transforms=spectrogram_transform,
        target_transforms=get_target_transforms(mode),
    )

    datamodule.prepare_data()
    datamodule.setup("fit")

    convert_datamodule_to_coco(datamodule, include_test=include_test)
