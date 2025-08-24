from detectron2.evaluation import COCOEvaluator
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.data.datasets import register_coco_instances
from omegaconf import OmegaConf
import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from selfrf.finetuning.detection.detectron2.mapper import SelfrfDatasetMapper
# Data using LSJ
image_size = 1024


train_annotations = "/home/sigence/repos/selfRF/datasets/50k/wideband_impaired/coco/annotations/instances_train.json"
train_images = "/home/sigence/repos/selfRF/datasets/50k/wideband_impaired/coco/images"
# Register datasets
register_coco_instances(
    "wideband_train",
    {},
    train_annotations,
    train_images,
)

# hardcoded validation set for now
val_annotations = "/home/sigence/repos/selfRF/datasets/100k/wideband_impaired/coco/annotations/instances_val.json"
val_images = "/home/sigence/repos/selfRF/datasets/100k/wideband_impaired/coco/images"
register_coco_instances(
    "wideband_val",
    {},
    val_annotations,
    val_images,
)

dataloader = OmegaConf.create()

dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="wideband_train"),
    mapper=L(SelfrfDatasetMapper)(
        is_train=True,
        augmentations=[
            L(T.RandomFlip)(horizontal=True),  # flip first
            L(T.ResizeScale)(
                min_scale=0.1, max_scale=2.0, target_height=image_size, target_width=image_size
            ),
            L(T.FixedSizeCrop)(crop_size=(image_size, image_size), pad=False),
        ],
        img_format="L",
    ),
    total_batch_size=2,
    num_workers=4,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(
        names="wideband_val", filter_empty=False),
    mapper=L(SelfrfDatasetMapper)(
        is_train=False,
        augmentations=[
            L(T.ResizeShortestEdge)(
                short_edge_length=image_size, max_size=image_size),
        ],
        img_format="${...train.mapper.img_format}",
    ),
    num_workers=4,
)

dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
)


dataloader.train.total_batch_size = 2

# recompute boxes due to cropping
# dataloader.train.mapper.recompute_boxes = True
