"""
This module provides factory classes and utility functions for constructing backbone and self-supervised learning (SSL) models in selfRF pretraining workflows. Its role is to centralize and standardize the instantiation logic for all supported backbone architectures (such as ResNet, ViT, and XCiT) and SSL algorithms (including DenseCL, MAE, and VICRegL), ensuring consistency and reproducibility across experiments. The main responsibilities include selecting the appropriate model type based on configuration, managing registries for backbones and SSL models, and abstracting away configuration-dependent construction details. Typical use cases involve building backbone or SSL model objects from configuration files for use in training pipelines, evaluation scripts, or automated research workflows. This module integrates with the broader selfRF pretraining pipeline by providing a unified interface for model selection and instantiation, ensuring compatibility with configuration management, dataset factories, and training utilities.
"""

from dataclasses import dataclass
from typing import Dict, Type, Callable, Union
import torch

from selfrf.pretraining.config import TrainingConfig, BaseConfig
from selfrf.pretraining.config.evaluation_config import EvaluationConfig
from selfrf.pretraining.utils.utils import get_class_list
from selfrf.models.iq_models import build_resnet1d
from selfrf.models.spectrogram_models import build_resnet2d, build_vit2d

from selfrf.models.ssl_models import DenseCL, MAE, VICRegL
from selfrf.pretraining.utils.enums import BackboneArchitecture, SSLModelType


@dataclass(frozen=True)
class BackboneConfig:
    """
    Immutable configuration for specifying a backbone architecture and data modality.
    :param backbone_arch: Enum specifying the backbone architecture type.
    :param is_spectrogram: Boolean indicating if the input is a spectrogram.
    """
    backbone_arch: BackboneArchitecture
    is_spectrogram: bool


class ModelFactory:
    """
    Factory class for constructing backbone and SSL models based on configuration.
    This class manages registries for supported backbone and SSL model types and provides unified interfaces for instantiation.
    """
    _backbone_registry: Dict[BackboneConfig, Callable] = {
        BackboneConfig(BackboneArchitecture.RESNET, False): lambda **kwargs: build_resnet1d(input_channels=2, **kwargs),
        BackboneConfig(BackboneArchitecture.RESNET, True): lambda **kwargs: build_resnet2d(input_channels=1, **kwargs),
        BackboneConfig(BackboneArchitecture.VIT, True): lambda **kwargs: build_vit2d(input_channels=1, **kwargs),
    }

    _ssl_registry: Dict[SSLModelType, Type] = {
        SSLModelType.DENSECL: DenseCL,
        SSLModelType.MAE: MAE,
        SSLModelType.VICREGL: VICRegL,
    }

    @classmethod
    def create_backbone(cls, config: Union[TrainingConfig, EvaluationConfig]) -> torch.nn.Module:
        """
        Creates and returns a backbone model instance from the provided configuration.
        :param config: Training or evaluation configuration object specifying backbone parameters.
        :returns: Instantiated torch.nn.Module representing the backbone model.
        :raises ValueError: If the backbone configuration is unknown or unsupported.
        """
        backbone_arch = config.backbone.get_architecture()
        is_spectrogram = config.spectrogram
        backbone_config = BackboneConfig(
            backbone_arch=backbone_arch,
            is_spectrogram=is_spectrogram
        )

        try:
            builder = cls._backbone_registry[backbone_config]
        except KeyError:
            data_type = "spectrogram" if is_spectrogram else "IQ"
            available_configs = []
            for cfg in cls._backbone_registry.keys():
                data_str = "spectrogram" if cfg.is_spectrogram else "IQ"
                available_configs.append(
                    f"{cfg.backbone_arch.value} with {data_str} data")
            raise ValueError(
                f"No backbone implementation found for '{backbone_arch.name}' architecture with {data_type} data.\n"
                f"Make sure 'spectrogram={is_spectrogram}' is compatible with your backbone choice.\n"
                f"Available combinations:\n" +
                "\n".join(f"- {c}" for c in available_configs)
            )

        if isinstance(config, EvaluationConfig):
            return builder(
                version=config.backbone.get_size().value,
            )

        return builder(
            version=config.backbone.get_size().value,
            features_only=True if config.ssl_model == SSLModelType.DENSECL or config.ssl_model == SSLModelType.VICREGL else False,
        )

    @classmethod
    def create_ssl_model(cls, config: TrainingConfig) -> torch.nn.Module:
        """
        Creates and returns an SSL model instance from the provided training configuration.
        :param config: Training configuration object specifying SSL model parameters.
        :returns: Instantiated torch.nn.Module representing the SSL model.
        :raises ValueError: If the SSL model type is unknown or unsupported.
        """
        backbone = cls.create_backbone(config)

        if isinstance(config.ssl_model, SSLModelType):
            ssl_type = config.ssl_model
        else:
            ssl_type = SSLModelType.from_string(config.ssl_model)

        ssl_model_class = cls._ssl_registry[ssl_type]

        model_args = {
            "backbone": backbone,
            "batch_size_per_device": config.batch_size,
            "use_online_linear_eval": config.online_linear_eval,
            "num_classes": len(get_class_list(config))
        }

        return ssl_model_class(**model_args)


def build_backbone(config: BaseConfig) -> torch.nn.Module:
    """
    Builds and returns a backbone model from the provided configuration.
    :param config: Configuration object specifying backbone parameters.
    :returns: Instantiated torch.nn.Module representing the backbone model.
    """
    return ModelFactory.create_backbone(config)


def build_ssl_model(config: TrainingConfig) -> torch.nn.Module:
    """
    Builds and returns an SSL model from the provided training configuration.
    :param config: Training configuration object specifying SSL model parameters.
    :returns: Instantiated torch.nn.Module representing the SSL model.
    """
    return ModelFactory.create_ssl_model(config)
