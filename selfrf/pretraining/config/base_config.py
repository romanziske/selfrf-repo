"""
Core configuration management for selfRF pretraining experiments with unified parameter handling.

This module provides the foundational configuration infrastructure used across all pretraining workflows in the selfRF library, establishing consistent parameter management for datasets, model architectures, and training settings. It serves as the base class for specialized configuration classes (TrainingConfig, EvaluationConfig) and handles the complex logic of dataset-specific defaults, command-line argument parsing, and configuration validation. The BaseConfig class manages essential parameters like dataset selection, backbone architecture, batch sizes, and device configuration while providing intelligent defaults based on the chosen dataset type. It integrates seamlessly with the broader pretraining ecosystem by providing a standardized interface for configuration management that ensures reproducible experiments and consistent parameter handling across different SSL methods. The module supports both programmatic configuration creation and command-line argument parsing, making it suitable for both interactive development and automated experiment pipelines.
"""
from dataclasses import dataclass, fields
import argparse
from typing import Optional, Tuple

import torch

from selfrf.pretraining.utils.enums import BackboneType, DatasetType

# Default values as constants
DEFAULT_DATASET = DatasetType.TORCHSIG_NARROWBAND
DEFAULT_ROOT = './datasets'
DEFAULT_FAMILY = False
DEFAULT_SPECTROGRAM = False
DEFAULT_NFFT = 512
DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_WORKERS = 8
DEFAULT_NUM_SAMPLES = 5000
DEFAULT_IMPAIRMENT_LEVEL = 2
DEFAULT_BACKBONE = BackboneType.RESNET50

DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATASET_IQ_SAMPLES = {
    DatasetType.TORCHSIG_WIDEBAND: 512**2,
    DatasetType.TORCHSIG_NARROWBAND: 512**2,
}


@dataclass
class BaseConfig:
    """
    Base configuration class containing common parameters for all pretraining workflows.

    Provides intelligent defaults based on dataset type and manages dataset-specific parameter overrides through property-based access patterns.

    :param dataset: Type of RF dataset to use for pretraining
    :type dataset: DatasetType
    :param root: Root directory path containing the dataset files
    :type root: str
    :param family: Whether to use family-level classification instead of signal-level
    :type family: bool
    :param impairment_level: Level of channel impairments to apply during data loading
    :type impairment_level: int
    :param num_samples: Total number of samples to use from the dataset
    :type num_samples: int
    :param _custom_iq_samples: Private field for storing user-specified IQ sample count
    :type _custom_iq_samples: Optional[int]
    :param spectrogram: Whether to convert IQ data to spectrogram representation
    :type spectrogram: bool
    :param nfft: Number of FFT points for spectrogram computation
    :type nfft: int
    :param resize: Target dimensions for spectrogram resizing as (height, width)
    :type resize: Optional[Tuple[int, int]]
    :param batch_size: Number of samples per training batch
    :type batch_size: int
    :param num_workers: Number of worker processes for data loading
    :type num_workers: int
    :param backbone: Neural network backbone architecture to use
    :type backbone: BackboneType
    :param device: PyTorch device specification for computation
    :type device: torch.device
    """
    dataset: DatasetType = DEFAULT_DATASET
    dataset_name: str = None
    root: str = DEFAULT_ROOT
    family: bool = DEFAULT_FAMILY
    impairment_level: int = DEFAULT_IMPAIRMENT_LEVEL
    num_samples: int = DEFAULT_NUM_SAMPLES

    # Add private storage field
    _custom_iq_samples: Optional[int] = None

    @property
    def num_iq_samples(self) -> int:
        """
        Returns the number of IQ samples based on dataset type or user override.

        Uses dataset-specific defaults unless explicitly overridden by the user through the setter.

        :returns: Number of IQ samples to use for signal processing
        :rtype: int
        """
        if self._custom_iq_samples is not None:
            return self._custom_iq_samples
        return DATASET_IQ_SAMPLES.get(self.dataset)

    @num_iq_samples.setter
    def num_iq_samples(self, value: int):
        """
        Sets a custom number of IQ samples, overriding dataset defaults.

        :param value: Custom number of IQ samples to use
        :type value: int
        """
        self._custom_iq_samples = value

    spectrogram: bool = DEFAULT_SPECTROGRAM
    nfft: int = DEFAULT_NFFT
    resize: Optional[Tuple[int, int]] = None

    batch_size: int = DEFAULT_BATCH_SIZE
    num_workers: int = DEFAULT_NUM_WORKERS

    backbone: BackboneType = DEFAULT_BACKBONE

    device: torch.device = DEFAULT_DEVICE


def add_base_config_args(parser: argparse.ArgumentParser) -> None:
    """
    Adds base configuration arguments to an ArgumentParser instance.

    Defines all command-line arguments needed for basic pretraining configuration including dataset selection, model parameters, and training settings.

    :param parser: ArgumentParser instance to add base configuration arguments to
    :type parser: argparse.ArgumentParser
    """
    parser.add_argument(
        '--dataset',
        type=lambda x: DatasetType(x),
        choices=list(DatasetType),
        default=DEFAULT_DATASET,
    )
    parser.add_argument(
        '--root',
        type=str,
        default=DEFAULT_ROOT,
    )
    parser.add_argument(
        '--family',
        type=lambda x: x.lower() == 'true',
        default=DEFAULT_FAMILY,
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=DEFAULT_NUM_SAMPLES,
    )
    parser.add_argument(
        '--impairment-level',
        type=int,
        default=DEFAULT_IMPAIRMENT_LEVEL,
    )
    parser.add_argument(
        '--num-iq-samples',
        type=int,
        default=None,
        help=(f'Number of IQ samples to use. Defaults based on dataset: '
              f'Wideband={DATASET_IQ_SAMPLES[DatasetType.TORCHSIG_WIDEBAND]}, '
              f'Narrowband={DATASET_IQ_SAMPLES[DatasetType.TORCHSIG_NARROWBAND]}. '
              f'Set this to override the default for your dataset.')
    )
    parser.add_argument(
        '--spectrogram',
        type=lambda x: x.lower() == 'true',
        default=DEFAULT_SPECTROGRAM,
    )
    parser.add_argument(
        '--nfft',
        type=int,
        default=DEFAULT_NFFT,
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=DEFAULT_NUM_WORKERS,
    )
    parser.add_argument(
        '--backbone',
        type=BackboneType.from_string,
        choices=list(BackboneType),
        default=DEFAULT_BACKBONE,
    )
    parser.add_argument(
        '--device',
        type=lambda x: torch.device(x),
        default=DEFAULT_DEVICE,
    )
    parser.add_argument(
        '--resize',
        type=int,
        nargs=2,  # Expect exactly two arguments
        metavar=('HEIGHT', 'WIDTH'),  # Help text for arguments
        default=None,  # Default is None
        help='Resize spectrogram to (HEIGHT, WIDTH). Example: --resize 224 224'
    )


def parse_base_config(parser: argparse.ArgumentParser) -> BaseConfig:
    """
    Parses command line arguments into a BaseConfig object with proper field filtering.

    Handles the complex logic of filtering parsed arguments to match BaseConfig fields and properly initializing the num_iq_samples property.

    :param parser: Configured ArgumentParser instance ready for parsing
    :type parser: argparse.ArgumentParser
    :returns: Initialized BaseConfig instance with parsed argument values
    :rtype: BaseConfig
    """
    args = parser.parse_args()
    args_dict = vars(args)

    # Get all field names from BaseConfig
    base_field_names = {f.name for f in fields(BaseConfig)}

    # Handle num_iq_samples specially
    custom_samples = args_dict.pop('num_iq_samples', None)

    # Filter args_dict to only include fields in BaseConfig
    filtered_args = {k: v for k,
                     v in args_dict.items() if k in base_field_names}

    # Create config with only the fields BaseConfig knows about
    config = BaseConfig(**filtered_args)

    # Set custom samples if provided
    if custom_samples is not None:
        config.num_iq_samples = custom_samples

    return config


def print_config(config: BaseConfig) -> None:
    """
    Prints configuration parameters in a structured, human-readable format.

    Displays all non-private configuration fields with proper handling of property-based fields like num_iq_samples.

    :param config: Configuration instance to display
    :type config: BaseConfig
    """
    print("\nConfiguration:")

    # Get fields directly from config instance
    for field in fields(config):
        # Skip private fields that start with underscore
        if field.name.startswith('_'):
            continue

        if field.name == 'num_iq_samples':
            # Get property value for num_iq_samples
            value = config.num_iq_samples
        else:
            value = getattr(config, field.name)
        print(f"  {field.name}: {value}")
