"""
Data transformation utilities for Detectron2 RF signal detection datasets.
"""
import copy
import logging
import numpy as np
import torch

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

logger = logging.getLogger("detectron2")


class SelfrfDatasetMapper:

    def __init__(
        self,
        augmentations: list = [],
        is_train=True,
        img_format="L",
        normalize=True,
    ):
        self.augmentations = augmentations
        self.img_format = img_format
        self.is_train = is_train
        self.normalize = normalize

        mode = "training" if is_train else "inference"
        logging.getLogger(__name__).info(
            f"Augmentations used in mode {mode}: {self.augmentations}"
        )

    def __call__(self, dataset_dict):

        dataset_dict = copy.deepcopy(
            dataset_dict)  # it will be modified by code below

        image = utils.read_image(
            dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        # Convert H×W to H×W×1 if necessary for grayscale images
        if image.ndim == 2:
            image = np.expand_dims(image, -1)

        # Apply augmentations
        auginput = T.AugInput(image)
        transforms = T.AugmentationList(self.augmentations)(auginput)
        image = auginput.image
        image_shape = image.shape[:2]

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1)).copy())

        if self.normalize:
            # Normalize the image to [0, 1] range
            dataset_dict["image"] = dataset_dict["image"].float() / 255.0

        if not self.is_train:
            # Remove annotations for inference
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # Transform annotations to match augmented image
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]

            instances = utils.annotations_to_instances(annos, image_shape)
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        return dataset_dict
