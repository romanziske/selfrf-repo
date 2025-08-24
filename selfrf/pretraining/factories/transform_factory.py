"""
This module provides factory classes and utility functions for constructing data transforms used in selfRF pretraining workflows. Its role is to centralize and standardize the creation of input and target transforms for both IQ and spectrogram data, supporting a variety of self-supervised learning (SSL) algorithms and RF data modalities. The main responsibilities include selecting and composing the appropriate normalization, augmentation, and tensor conversion routines based on configuration, as well as managing SSL-specific transform wrappers for methods like DenseCL, MAE, and VICRegL. Typical use cases involve building input and target transforms for PyTorch dataloaders in SSL training or evaluation pipelines, dynamically adapting preprocessing for different model architectures, and supporting automated experiment orchestration. This module integrates with the broader selfRF pretraining pipeline by providing a unified interface for transform selection and construction, ensuring compatibility with configuration management, dataset factories, and collate function utilities.
"""

from typing import Dict, Callable, Union

import numpy as np
from torchsig.transforms.dataset_transforms import ComplexTo2D, Transform
from torchsig.transforms.base_transforms import Compose, Normalize
from torchsig.transforms.target_transforms import ClassIndex, FamilyIndex
from torchsig.signals.signal_lists import TorchSigSignalLists

from selfrf.transforms import (
    Identity,
    ToTensor,
    SpectrogramImageHighQuality,
    DenseCLTransform,
    MAETransform,
    OptimizedVICRegLTransform,
)
from selfrf.pretraining.config import BaseConfig, TrainingConfig, EvaluationConfig
from selfrf.pretraining.utils.enums import TransformType, SSLModelType, DatasetType


class TransformFactory:
    """
    Factory class for constructing input and target transforms for selfRF pretraining.
    This class manages registries for supported transform types and SSL-specific wrappers, providing unified interfaces for transform instantiation.
    """

    @staticmethod
    def create_spectrogram_transform(config: BaseConfig, to_tensor: bool = True) -> Transform:
        """
        Creates and returns a composed transform for spectrogram data, including normalization and tensor conversion.
        :param config: Configuration object specifying transform parameters.
        :param to_tensor: Whether to convert the output to a tensor.
        :returns: Composed Transform object for spectrogram preprocessing.
        """

        if config.dataset == DatasetType.CUSTOM_SPECTROGRAM:
            # Custom spectrogram dataset handling
            return Compose([])

        return Compose([
            Normalize(norm=np.inf),
            SpectrogramImageHighQuality(
                nfft=config.nfft,
                to_tensor=to_tensor,
                resize=config.resize,
            ),
        ])

    @staticmethod
    def create_iq_transform(config: BaseConfig) -> Transform:
        """
        Creates and returns a composed transform for IQ data, including normalization and conversion to 2D tensor.
        :param config: Configuration object specifying transform parameters.
        :returns: Composed Transform object for IQ preprocessing.
        """
        return Compose([
            Normalize(norm=np.inf),
            ComplexTo2D(),
            ToTensor(),
        ])

    _transform_registry: Dict[TransformType, Callable[[BaseConfig], Transform]] = {
        TransformType.SPECTROGRAM: create_spectrogram_transform,
        TransformType.IQ: create_iq_transform
    }

    _ssl_transform_registry: Dict[SSLModelType, Callable] = {
        SSLModelType.DENSECL: DenseCLTransform,
        SSLModelType.MAE: MAETransform,
        SSLModelType.VICREGL: OptimizedVICRegLTransform,
    }

    @classmethod
    def create_tensor_transform(cls, config: Union[TrainingConfig, EvaluationConfig]) -> Transform:
        """
        Selects and returns the appropriate tensor transform based on the data modality in the configuration.
        :param config: Training or evaluation configuration object.
        :returns: Transform object for input tensor preprocessing.
        """
        transform_type = TransformType.SPECTROGRAM if config.spectrogram else TransformType.IQ
        return cls._transform_registry[transform_type](config)

    @classmethod
    def create_transform(cls, config: Union[TrainingConfig, EvaluationConfig]) -> Transform:
        """
        Builds and returns the full transform pipeline, including SSL-specific wrappers if needed.
        :param config: Training or evaluation configuration object.
        :returns: Transform object for use in dataloaders.
        """
        tensor_transform = cls.create_tensor_transform(config)

        if isinstance(config, EvaluationConfig):
            return tensor_transform

        return cls._ssl_transform_registry[config.ssl_model](
            tensor_transform=tensor_transform,
            apply_on="spectrogram" if config.spectrogram else "iq",
        )

    @classmethod
    def create_target_transform(cls, config: BaseConfig) -> Transform:
        """
        Builds and returns the appropriate target transform based on the dataset type and configuration.
        :param config: Configuration object specifying dataset and target parameters.
        :returns: Transform or list of transforms for target preprocessing.
        """
        if config.dataset == DatasetType.TORCHSIG_NARROWBAND or config.dataset == DatasetType.SIGMF_NARROWBAND:
            if config.family:

                return [FamilyIndex(class_family_dict=TorchSigSignalLists.family_dict, family_list=TorchSigSignalLists.family_list)]
            return [ClassIndex()]

        if config.dataset == DatasetType.TORCHSIG_WIDEBAND:
            return [Identity()]

        if config.dataset == DatasetType.CUSTOM_SPECTROGRAM:
            return [Identity()]


def build_transform(config: Union[TrainingConfig, EvaluationConfig]) -> Transform:
    """
    Builds and returns the full input transform pipeline for the given configuration.
    :param config: Training or evaluation configuration object.
    :returns: Transform object for input preprocessing.
    """
    return TransformFactory.create_transform(config)


def build_target_transform(config: BaseConfig) -> Transform:
    """
    Builds and returns the target transform pipeline for the given configuration.
    :param config: Configuration object specifying dataset and target parameters.
    :returns: Transform or list of transforms for target preprocessing.
    """
    return TransformFactory.create_target_transform(config)
