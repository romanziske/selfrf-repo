"""
Training-specific configuration management extending base configuration for SSL pretraining workflows.

This module provides specialized configuration handling for self-supervised learning pretraining experiments, building upon the BaseConfig foundation with training-specific parameters like SSL model selection, epoch counts, and checkpoint management. It serves as the primary configuration interface for training scripts and handles the complex parameter inheritance from base configuration while adding training-specific arguments and validation. The TrainingConfig class manages essential training parameters including SSL method selection, online evaluation settings, checkpoint paths, and training duration specifications. It integrates with the pretraining pipeline by providing a standardized configuration interface that training scripts can use to configure SSL models, data loaders, and training loops consistently across different experimental setups. The module supports both command-line argument parsing and programmatic configuration creation, enabling flexible usage in both automated experiment pipelines and interactive development workflows.
"""
import argparse
from dataclasses import dataclass

from selfrf.pretraining.utils.enums import SSLModelType
from .base_config import BaseConfig, add_base_config_args, parse_base_config

DEFAULT_ONLINE_LINEAR_EVAL = False
DEFAULT_SSL_MODEL = SSLModelType.VICREGL
DEFAULT_TRAINING_PATH = './train'
DEFAULT_NUM_EPOCHS = 100


@dataclass
class TrainingConfig(BaseConfig):
    """
    Configuration class for SSL pretraining workflows extending BaseConfig with training-specific parameters.

    Inherits all base configuration parameters and adds SSL method selection, training duration, and checkpoint management capabilities.

    :param online_linear_eval: Whether to perform online linear evaluation during training
    :type online_linear_eval: bool
    :param ssl_model: Self-supervised learning method to use for pretraining
    :type ssl_model: SSLModelType
    :param training_path: Directory path for saving training outputs and checkpoints
    :type training_path: str
    :param num_epochs: Number of training epochs to execute
    :type num_epochs: int
    :param resume_from_checkpoint: Path to checkpoint file for resuming interrupted training
    :type resume_from_checkpoint: str
    """
    online_linear_eval: bool = DEFAULT_ONLINE_LINEAR_EVAL
    ssl_model: SSLModelType = DEFAULT_SSL_MODEL
    training_path: str = DEFAULT_TRAINING_PATH
    num_epochs: int = DEFAULT_NUM_EPOCHS
    resume_from_checkpoint: str = None


def add_training_config_args(parser: argparse.ArgumentParser) -> None:
    """
    Adds training-specific configuration arguments to an ArgumentParser instance.

    Extends base configuration arguments with SSL training parameters including model selection, evaluation settings, and checkpoint management.

    :param parser: ArgumentParser instance to add training configuration arguments to
    :type parser: argparse.ArgumentParser
    """
    add_base_config_args(parser)
    parser.add_argument(
        '--online-linear-eval',
        type=lambda x: x.lower() == 'true',
        default=DEFAULT_ONLINE_LINEAR_EVAL
    )
    parser.add_argument(
        "--ssl-model",
        type=lambda x: SSLModelType.from_string(x),
        default=SSLModelType.VICREGL,
        choices=list(SSLModelType),
        help=f"SSL model to use for pretraining {[model_type.value for model_type in SSLModelType]}"
    )
    parser.add_argument(
        '--training-path',
        type=str,
        default=DEFAULT_TRAINING_PATH
    )
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=DEFAULT_NUM_EPOCHS
    )
    parser.add_argument(
        '--resume-from-checkpoint',
        type=str,
        default=None,
        help='Path to the checkpoint file to resume training from.'
    )


def parse_training_config() -> TrainingConfig:
    """
    Parses command line arguments into a TrainingConfig object with proper inheritance handling.

    Creates base configuration first then extends it with training-specific parameters to ensure proper field inheritance and validation.

    :returns: Fully configured TrainingConfig instance with parsed argument values
    :rtype: TrainingConfig
    """
    # Create parser with description
    parser = argparse.ArgumentParser(description="Training Config")
    add_training_config_args(parser)

    # First parse the base config (handles num_iq_samples properly)
    base_config = parse_base_config(parser)

    # Get the args again to extract training-specific fields
    args = parser.parse_args()

    # Create TrainingConfig by combining base config and training args
    training_config = TrainingConfig(
        **vars(base_config),

        # Add training fields
        online_linear_eval=args.online_linear_eval,
        ssl_model=args.ssl_model,
        training_path=args.training_path,
        num_epochs=args.num_epochs,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

    return training_config
