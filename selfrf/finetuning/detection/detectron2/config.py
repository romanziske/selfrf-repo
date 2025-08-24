"""
Configuration management for Detectron2-based RF signal detection training.
"""
from typing import Optional
from dataclasses import fields, dataclass
import argparse
import torch

from detectron2.model_zoo import model_zoo
from detectron2.config import get_cfg, CfgNode

from selfrf.finetuning.detection.detectron2.util import ModelType

# Default values as constants
DEFAULT_MODEL_TYPE = ModelType.FASTER_RCNN_R50_FPN
DEFAULT_MODEL_PATH = ""
DEFAULT_NUM_CLASSES = 10
DEFAULT_MAX_ITER = 90_000
DEFAULT_BASE_LR = 0.025
DEFAULT_IMS_PER_BATCH = 8
DEFAULT_CHECKPOINT_PERIOD = 1000
DEFAULT_OUTPUT_DIR = "./train/detection"


@dataclass
class Detectron2Config:
    """
    Configuration container for Detectron2 model training parameters.

    :param root: Root directory path for dataset location
    :type root: str
    :param dataset_path: Dataset directory path relative to root
    :type dataset_path: str
    :param run_name: Specific run identifier for output organization
    :type run_name: str
    :param resume: Whether to resume training from the last checkpoint
    :type resume: bool
    :param output_dir: Directory for training outputs and checkpoints
    :type output_dir: str
    :param visualize: Whether to generate visualization outputs
    :type visualize: bool
    :param inference: Whether to run inference mode
    :type inference: bool
    :param inference_threshold: Confidence threshold for inference predictions
    :type inference_threshold: float
    :param process_n_samples: Number of samples to process for inference/visualization
    :type process_n_samples: int
    :param model_type: Architecture type for the detection model
    :type model_type: ModelType
    :param model_path: Path to existing model weights file
    :type model_path: str
    :param num_classes: Number of object classes to detect
    :type num_classes: int
    :param max_iter: Maximum training iterations
    :type max_iter: int
    :param base_lr: Base learning rate for optimization
    :type base_lr: float
    :param ims_per_batch: Number of images per training batch
    :type ims_per_batch: int
    :param checkpoint_period: Frequency of checkpoint saving
    :type checkpoint_period: int
    :param eval_period: Frequency of validation evaluation
    :type eval_period: int
    :param export_onnx: Whether to export trained model to ONNX format
    :type export_onnx: bool
    """
    root: str = ""
    dataset_path: str = ""
    run_name: str = ""
    resume: bool = False

    output_dir: str = DEFAULT_OUTPUT_DIR
    visualize: bool = False
    inference: bool = False
    eval_only: bool = False
    inference_threshold: float = 0.7
    process_n_samples: int = 100

    model_type: ModelType = DEFAULT_MODEL_TYPE
    model_path: str = DEFAULT_MODEL_PATH
    num_classes: int = DEFAULT_NUM_CLASSES
    max_iter: int = DEFAULT_MAX_ITER
    base_lr: float = DEFAULT_BASE_LR
    ims_per_batch: int = DEFAULT_IMS_PER_BATCH
    checkpoint_period: int = DEFAULT_CHECKPOINT_PERIOD
    eval_period: int = 3000

    export_onnx: bool = False


def add_detectron2_config_args(parser: argparse.ArgumentParser) -> None:
    """
    Adds Detectron2-specific command-line arguments to argument parser.

    :param parser: ArgumentParser instance to extend with Detectron2 options
    :type parser: argparse.ArgumentParser
    :raises AttributeError: If parser is not a valid ArgumentParser instance
    """
    parser.add_argument(
        '--root',
        type=str,
        default='',
        help='Root directory for dataset'
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        default="",
        help='path to dataset directory, relative to root'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help='Output directory for training results'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        default=False,
        help='Visualize training results'
    )
    parser.add_argument(
        '--inference',
        action='store_true',
        default=False,
        help='Run inference on the model'
    )
    parser.add_argument(
        '--eval-only',
        action='store_true',
        default=False,
        help='Run evaluation only without training'
    )
    parser.add_argument(
        '--inference-threshold',
        type=float,
        default=0.7,
        help='Threshold for inference'
    )
    parser.add_argument(
        '--process-n-samples',
        type=int,
        default=100,
        help='Number of samples to process for inference or visualization'
    )
    parser.add_argument(
        '--model-type',
        type=ModelType.from_string,
        choices=list(ModelType),
        default=DEFAULT_MODEL_TYPE,
        help='Model architecture type (e.g., vitdet-vit-l, vitdet-vit-b)'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=DEFAULT_MODEL_PATH,
        help='Path to existing model weights'
    )
    parser.add_argument(
        '--num-classes',
        type=int,
        default=DEFAULT_NUM_CLASSES,
        help='Number of classes to detect'
    )
    parser.add_argument(
        '--max-iter',
        type=int,
        default=DEFAULT_MAX_ITER,
        help='Maximum number of training iterations'
    )
    parser.add_argument(
        '--base-lr',
        type=float,
        default=DEFAULT_BASE_LR,
        help='Base learning rate'
    )
    parser.add_argument(
        '--ims-per-batch',
        type=int,
        default=DEFAULT_IMS_PER_BATCH,
        help='Images per batch'
    )
    parser.add_argument(
        '--checkpoint-period',
        type=int,
        default=DEFAULT_CHECKPOINT_PERIOD,
        help='Checkpoint save frequency'
    )
    parser.add_argument(
        '--eval-period',
        type=int,
        default=3000,
        help='Evaluation frequency'
    )
    parser.add_argument(
        '--export-onnx',
        action='store_true',
        default=False,
        help='Export model to ONNX format after training'
    )
    parser.add_argument(
        '--run-name',
        type=str,
        default=None,
        help='Specific run name to use for output directory'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='Resume training from the last checkpoint'
    )


def print_config(config: Detectron2Config) -> None:
    """
    Prints configuration parameters in structured format for debugging.

    :param config: Configuration object to display
    :type config: Detectron2Config
    """
    print("\nDetectron2 Configuration:")
    for field in fields(config):
        value = getattr(config, field.name)
        print(f"  {field.name}: {value}")


def build_detectron2_config(config: Detectron2Config = Detectron2Config()) -> CfgNode:
    """
    Constructs Detectron2 CfgNode with custom parameters for RF signal detection.

    :param config: Custom configuration parameters
    :type config: Detectron2Config
    :returns: Detectron2 configuration node with applied settings
    :rtype: CfgNode
    :raises RuntimeError: If CUDA configuration fails or model zoo config unavailable
    """
    # ───────────  Dataset + Config  ──────────────────────────────
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("wideband_train",)
    cfg.DATASETS.TEST = ("wideband_val",)
    cfg.DATALOADER.NUM_WORKERS = 16
    cfg.TEST.EVAL_PERIOD = config.eval_period

    # ───────────  Model  ────────────────────────────────────────
    cfg.MODEL.WEIGHTS = config.model_path
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = config.num_classes

    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.INPUT.FORMAT = "L"

    cfg.INPUT.MIN_SIZE_TRAIN = (512, 640, 672, 704, 736, 768, 800)
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"  # Randomly pick one size per image
    cfg.INPUT.MAX_SIZE_TRAIN = 1333

    cfg.INPUT.MIN_SIZE_TEST = 0  # Set to zero to disable resize in testing.

    cfg.MODEL.PIXEL_MEAN = [0.0]
    cfg.MODEL.PIXEL_STD = [1.0]

    # ───────────  SOLVER  ────────────────────────────────────────
    cfg.SOLVER.AMP.ENABLED = True
    cfg.SOLVER.MAX_ITER = config.max_iter
    cfg.SOLVER.IMS_PER_BATCH = config.ims_per_batch
    cfg.SOLVER.BASE_LR = config.base_lr
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000

    decay1 = int(0.9 * cfg.SOLVER.MAX_ITER)
    decay2 = int(0.97 * cfg.SOLVER.MAX_ITER)
    cfg.SOLVER.STEPS = (decay1, decay2)
    cfg.SOLVER.GAMMA = 0.1

    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0

    cfg.SOLVER.CHECKPOINT_PERIOD = 10000

    return cfg
