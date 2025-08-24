"""
Enumerations and specification classes for backbone architectures, SSL models, and dataset types in selfRF.

This module defines standardized enums and data structures to represent backbone architectures, model sizes, SSL methods, dataset types, and collate strategies used throughout the pretraining and evaluation pipelines. Its main responsibilities include providing type-safe identifiers for model selection, configuration parsing, and experiment reproducibility. Typical use-cases involve specifying model backbones in configuration files, selecting SSL methods for training, and ensuring consistent handling of dataset and transform types across the library. The module integrates with configuration management, model builders, and training scripts to enforce consistency and reduce errors in experiment setup.
"""

from enum import Enum, auto
from dataclasses import dataclass


class BackboneArchitecture(Enum):
    """
    Enumeration of supported backbone architecture types.

    Used to specify the core neural network family for model construction and selection.

    :returns: String identifier for the backbone architecture
    :rtype: str
    """
    RESNET = "resnet"
    VIT = "vit"
    XCIT = "xcit"

    def __str__(self) -> str:
        """
        Returns the name of the backbone architecture.

        :returns: Name of the backbone architecture as a string
        :rtype: str
        """
        return self.name


class BackboneSize(Enum):
    """
    Enumeration of size variants for different backbone architectures.

    Used to specify the depth or scale of a given backbone model.

    :returns: String identifier for the backbone size
    :rtype: str
    """
    # ResNet variants
    RESNET_18 = "18"
    RESNET_34 = "34"
    RESNET_50 = "50"
    RESNET_101 = "101"
    RESNET_152 = "152"

    # ViT variants
    VIT_TINY = "vit_tiny_patch16_224"
    VIT_SMALL = "vit_small_patch16_224"
    VIT_BASE = "vit_base_patch16_224"
    VIT_LARGE = "vit_large_patch16_224"
    VIT_HUGE = "vit_huge_patch14_224"

    # XCiT variants
    XCIT_NANO_12 = "xcit_nano_12_p16_224"
    XCIT_TINY_12 = "xcit_tiny_12_p16_224"
    XCIT_SMALL_12 = "xcit_small_12_p16_224"
    XCIT_MEDIUM_24 = "xcit_medium_24_p16_224"
    XCIT_LARGE_24 = "xcit_large_24_p16_224"

    def __str__(self) -> str:
        """
        Returns the name of the backbone size.

        :returns: Name of the backbone size as a string
        :rtype: str
        """
        return self.name


@dataclass
class BackboneSpec:
    """
    Specification of a backbone model architecture and size.

    Combines architecture and size into a single specification for model selection and configuration.

    :param architecture: Backbone architecture type (e.g., RESNET, VIT)
    :type architecture: BackboneArchitecture
    :param size: Backbone size variant (e.g., RESNET_50, VIT_BASE)
    :type size: BackboneSize
    :returns: String representation of the backbone specification
    :rtype: str
    """
    architecture: BackboneArchitecture
    size: BackboneSize

    def __str__(self) -> str:
        """
        Returns a human-readable string representation of the backbone specification.

        :returns: String representation in the format 'architecture-size'
        :rtype: str
        """
        return f"{self.architecture.value}-{self.size.value}"

    @classmethod
    def from_string(cls, spec_string: str) -> 'BackboneSpec':
        """
        Parses a backbone specification from a string.

        Expects the format 'architecture-size' (e.g., 'resnet-50') and returns a BackboneSpec instance.

        :param spec_string: String specifying the backbone architecture and size
        :type spec_string: str
        :returns: Parsed BackboneSpec instance
        :rtype: BackboneSpec
        :raises ValueError: If the string format or values are invalid
        """
        parts = spec_string.lower().split('-')
        if len(parts) != 2:
            raise ValueError(f"Invalid backbone spec format: {spec_string}. "
                             f"Expected format: 'architecture-size'")

        arch_str, size_str = parts[0], parts[1]

        try:
            architecture = BackboneArchitecture(arch_str)
        except ValueError:
            raise ValueError(f"Unknown architecture: {arch_str}")

        size = None
        for size_enum in BackboneSize:
            if size_enum.name.lower().startswith(f"{arch_str}_{size_str}"):
                size = size_enum
                break

        if size is None:
            raise ValueError(
                f"Unknown size '{size_str}' for architecture '{arch_str}'")

        return cls(architecture, size)


class BackboneType(Enum):
    """
    Enumeration of backbone types for model training and selection.

    Provides type-safe identifiers for supported backbone architectures and sizes.

    :returns: BackboneSpec instance representing the architecture and size
    :rtype: BackboneSpec
    """
    RESNET18 = BackboneSpec(BackboneArchitecture.RESNET,
                            BackboneSize.RESNET_18)
    RESNET34 = BackboneSpec(BackboneArchitecture.RESNET,
                            BackboneSize.RESNET_34)
    RESNET50 = BackboneSpec(BackboneArchitecture.RESNET,
                            BackboneSize.RESNET_50)
    RESNET101 = BackboneSpec(
        BackboneArchitecture.RESNET, BackboneSize.RESNET_101)

    VIT_TINY = BackboneSpec(BackboneArchitecture.VIT, BackboneSize.VIT_TINY)
    VIT_SMALL = BackboneSpec(BackboneArchitecture.VIT, BackboneSize.VIT_SMALL)
    VIT_BASE = BackboneSpec(BackboneArchitecture.VIT, BackboneSize.VIT_BASE)
    VIT_LARGE = BackboneSpec(BackboneArchitecture.VIT, BackboneSize.VIT_LARGE)
    VIT_HUGE = BackboneSpec(BackboneArchitecture.VIT, BackboneSize.VIT_HUGE)

    XCIT_NANO_12 = BackboneSpec(
        BackboneArchitecture.XCIT, BackboneSize.XCIT_NANO_12)
    XCIT_TINY_12 = BackboneSpec(
        BackboneArchitecture.XCIT, BackboneSize.XCIT_TINY_12)
    XCIT_SMALL_12 = BackboneSpec(
        BackboneArchitecture.XCIT, BackboneSize.XCIT_SMALL_12)
    XCIT_MEDIUM_24 = BackboneSpec(
        BackboneArchitecture.XCIT, BackboneSize.XCIT_MEDIUM_24)
    XCIT_LARGE_24 = BackboneSpec(
        BackboneArchitecture.XCIT, BackboneSize.XCIT_LARGE_24)

    def get_architecture(self) -> BackboneArchitecture:
        """
        Returns the backbone architecture type.

        :returns: BackboneArchitecture enum value
        :rtype: BackboneArchitecture
        """
        return self.value.architecture

    def get_size(self) -> BackboneSize:
        """
        Returns the backbone size variant.

        :returns: BackboneSize enum value
        :rtype: BackboneSize
        """
        return self.value.size

    @classmethod
    def from_string(cls, name: str) -> 'BackboneType':
        """
        Converts a string to a BackboneType enum.

        :param name: String name of the backbone type
        :type name: str
        :returns: Corresponding BackboneType enum value
        :rtype: BackboneType
        :raises ValueError: If the name does not match any backbone type
        """
        try:
            return cls[name.upper()]
        except KeyError:
            for bt in cls:
                if bt.name.lower() == name.lower():
                    return bt
            raise ValueError(
                f"Unknown backbone type: {name}. Valid values are: {[t.name for t in cls]}")

    def __str__(self) -> str:
        """
        Returns the name of the backbone type.

        :returns: Name of the backbone type as a string
        :rtype: str
        """
        return self.name


class CollateType(Enum):
    """
    Enumeration of collate function types for data loading.

    Used to specify the data collation strategy for SSL model training.

    :returns: Enum value representing the collate type
    :rtype: CollateType
    """
    MULTI_VIEW = auto()
    SINGLE_VIEW = auto()
    MULTI_VIEW_GRID = auto()


class SSLModelType(Enum):
    """
    Enumeration of self-supervised learning (SSL) model types and their collate strategies.

    Associates each SSL model type with its required data collation method for training.

    :returns: Tuple of (string identifier, CollateType)
    :rtype: tuple
    """
    DENSECL = ("densecl", CollateType.MULTI_VIEW)
    MAE = ("mae", CollateType.SINGLE_VIEW)
    VICREGL = ("vicregl", CollateType.MULTI_VIEW_GRID)

    def __init__(self, value, collate_type):
        self._value_ = value
        self.collate_type = collate_type

    @classmethod
    def from_string(cls, name: str) -> 'SSLModelType':
        """
        Converts a string to an SSLModelType enum.

        :param name: String name of the SSL model type
        :type name: str
        :returns: Corresponding SSLModelType enum value
        :rtype: SSLModelType
        :raises ValueError: If the name does not match any SSL model type
        """
        name_lower = name.lower()
        for model_type in cls:
            if model_type.value.lower() == name_lower:
                return model_type
        try:
            return cls[name.upper()]
        except KeyError:
            valid_types = [f"{t.name} ({t.value})" for t in cls]
            raise ValueError(
                f"Unknown SSL model type: '{name}'. "
                f"Valid types are: {', '.join(valid_types)}"
            )


class DatasetType(Enum):
    """
    Enumeration of supported dataset types for pretraining.

    Used to specify the RF dataset variant for experiment configuration.

    :returns: String identifier for the dataset type
    :rtype: str
    """
    TORCHSIG_NARROWBAND = "narrowband"
    TORCHSIG_WIDEBAND = "wideband"
    CUSTOM_SPECTROGRAM = "custom_spectrogram"
    SIGMF_NARROWBAND = "sigmf_narrowband"


class TransformType(Enum):
    """
    Enumeration of transform types for input data representation.

    Used to specify whether to use spectrogram or IQ data for model input.

    :returns: String identifier for the transform type
    :rtype: str
    """
    SPECTROGRAM = "spectrogram"
    IQ = "iq"
