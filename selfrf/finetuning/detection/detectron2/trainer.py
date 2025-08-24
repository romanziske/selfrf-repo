"""
Detectron2 training utilities for RF signal detection models.
"""
import os
import logging

from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.data import detection_utils as utils
from detectron2.checkpoint import DetectionCheckpointer

from selfrf.finetuning.detection.detectron2.config import Detectron2Config, build_detectron2_config
from selfrf.finetuning.detection.detectron2.export import export_onnx_model
from selfrf.finetuning.detection.detectron2.util import create_output_directory

from .mapper import SelfrfDatasetMapper

logger = logging.getLogger("detectron2")


class Trainer(DefaultTrainer):
    """
    Custom Detectron2 trainer for RF signal detection with COCO evaluation.
    """

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Builds training data loader with custom RF signal mapper.

        :param cfg: Detectron2 configuration object
        :type cfg: detectron2.config.CfgNode
        :returns: Training data loader
        :rtype: torch.utils.data.DataLoader
        """
        return build_detection_train_loader(
            cfg,
            mapper=SelfrfDatasetMapper(
                augmentations=utils.build_augmentation(
                    cfg, is_train=True),
                is_train=True,
                img_format=cfg.INPUT.FORMAT,
                normalize=True,
            )
        )

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Creates COCO evaluator for object detection evaluation.

        :param cfg: Detectron2 configuration object
        :type cfg: detectron2.config.CfgNode
        :param dataset_name: Name of registered dataset
        :type dataset_name: str
        :param output_folder: Directory for evaluation files
        :type output_folder: str
        :returns: COCO evaluator instance
        :rtype: detectron2.evaluation.COCOEvaluator
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            os.makedirs(output_folder, exist_ok=True)

        return COCOEvaluator(
            dataset_name,
            output_dir=output_folder,
            tasks=("bbox",),  # Only evaluate bounding boxes
            use_fast_impl=True
        )

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Builds test data loader with non-training mapper for evaluation.

        :param cfg: Detectron2 configuration object
        :type cfg: detectron2.config.CfgNode
        :param dataset_name: Name of registered dataset
        :type dataset_name: str
        :returns: Test data loader
        :rtype: torch.utils.data.DataLoader
        """
        return build_detection_test_loader(
            cfg,
            dataset_name,
            mapper=SelfrfDatasetMapper(
                augmentations=utils.build_augmentation(
                    cfg, is_train=False),
                is_train=False,
                img_format=cfg.INPUT.FORMAT,
                normalize=True,
            )
        )

    def export_model(self, export_path: str, input_shape: tuple = (1, 1, 512, 512)):
        """
        Exports trained model to ONNX format.

        :param export_path: Path to save ONNX model
        :type export_path: str
        :param input_shape: Input tensor shape (batch_size, channels, height, width)
        :type input_shape: tuple
        :raises RuntimeError: If ONNX export fails
        """
        export_onnx_model(self.cfg, self.model, export_path, input_shape)


def do_train(config: Detectron2Config):
    """
    Trains Detectron2 model with optional backbone preloading and freezing.

    :param config: Training configuration containing model and data parameters
    :type config: Detectron2Config
    :raises ValueError: If both model_path and backbone_path are specified
    :raises FileNotFoundError: If specified checkpoint files do not exist
    :raises RuntimeError: If training execution fails
    """

    cfg = build_detectron2_config(config)

    if config.run_name:
        # Use the provided run name for the output directory
        cfg.OUTPUT_DIR = os.path.join(config.output_dir, config.run_name)
    else:
        cfg.OUTPUT_DIR = create_output_directory(config)

    if config.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=config.resume
        )
        res = Trainer.test(cfg, model)
        print(res)
        return

    trainer = Trainer(cfg)

    # Load checkpoint if available - use resume flag from config
    trainer.resume_or_load(resume=config.resume)

    if config.export_onnx:
        onnx_path = os.path.join(cfg.OUTPUT_DIR, "model.onnx")
        trainer.export_model(onnx_path, input_shape=(1, 1, 512, 512))
        logger.info(f"Model exported to ONNX: {onnx_path}")
        return

    # Run training
    trainer.train()
