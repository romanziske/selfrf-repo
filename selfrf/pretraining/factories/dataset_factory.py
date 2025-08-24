"""
This module provides factory classes and utility functions for constructing datasets and dataloaders used in selfRF pretraining workflows. Its role is to abstract and standardize the instantiation of TorchSig-based data modules, metadata, and transforms, enabling flexible and reproducible dataset creation for self-supervised learning (SSL) experiments. The main responsibilities include selecting the appropriate dataset type, building dataset metadata from configuration, and assembling all necessary transforms and collate functions for training and evaluation. Typical use cases involve creating dataloaders for SSL training pipelines, dynamically configuring datasets for different RF modalities, and supporting automated experiment orchestration. This module integrates with the broader selfRF pretraining pipeline by providing a unified interface for dataset and dataloader construction, ensuring compatibility with configuration management, transform factories, and collate function utilities.
"""

from typing import Dict, Type

from torchsig.datasets.datamodules import TorchSigDataModule, NarrowbandDataModule, WidebandDataModule
from torchsig.datasets.dataset_metadata import DatasetMetadata, NarrowbandMetadata, WidebandMetadata
from torchsig.datasets.datamodules import WidebandDataModule
from torchsig.datasets.default_configs.loader import get_default_yaml_config
from torchsig.datasets.dataset_utils import to_dataset_metadata

from selfrf.pretraining.config import BaseConfig
from selfrf.pretraining.utils.image.image_datamodule import ImageDataModule
from selfrf.pretraining.utils.enums import DatasetType
from selfrf.pretraining.factories.collate_fn_factory import build_collate_fn
from selfrf.pretraining.factories.transform_factory import build_transform, build_target_transform
from selfrf.pretraining.utils.sigmf.sigmf_datamodule import SigmfDataModule


class DatasetFactory:
    """
    Factory class for constructing TorchSigDataModule datasets based on configuration.
    This class manages a registry of supported dataset types and provides a unified interface for dataset instantiation.
    """
    _dataset_registry: Dict[DatasetType, Type[TorchSigDataModule]] = {
        DatasetType.TORCHSIG_NARROWBAND: NarrowbandDataModule,
        DatasetType.TORCHSIG_WIDEBAND: WidebandDataModule,
        DatasetType.CUSTOM_SPECTROGRAM: NarrowbandDataModule,
        DatasetType.SIGMF_NARROWBAND: SigmfDataModule,
    }

    @classmethod
    def create_dataset(cls, config: BaseConfig) -> TorchSigDataModule:
        """
        Create and return a TorchSigDataModule instance from the provided configuration.

        :param config: Configuration object specifying dataset parameters.

        :returns: Instantiated TorchSigDataModule for use in training or evaluation.

        """
        dataset_type = DatasetType(config.dataset)

        if dataset_type == DatasetType.CUSTOM_SPECTROGRAM:
            # Custom spectrogram dataset handling
            return ImageDataModule(
                root=config.root,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                transforms=[build_transform(config)],
                target_transforms=build_target_transform(config),
                collate_fn=build_collate_fn(config),
            )

        if dataset_type == DatasetType.SIGMF_NARROWBAND:
            # SigMF narrowband dataset handling
            return SigmfDataModule(
                root=config.root,
                dataset="narrowband",
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                transforms=[build_transform(config)],
                target_transforms=build_target_transform(config),
                collate_fn=build_collate_fn(config),
                num_iq_samples=4096
            )

        dataset_class = cls._dataset_registry[dataset_type]

        return dataset_class(
            root=config.root,
            dataset_metadata=get_dataset_metadata(config),
            num_samples_train=config.num_samples,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            transforms=[build_transform(config)],
            target_transforms=build_target_transform(config),
            collate_fn=build_collate_fn(config),
        )


def get_dataset_metadata(config: BaseConfig) -> DatasetMetadata:
    """
    Returns the appropriate DatasetMetadata object based on the dataset type in the configuration.

    :param config: Configuration object specifying dataset parameters.
    :returns: DatasetMetadata instance for the selected dataset type.

    """
    if config.dataset == DatasetType.TORCHSIG_NARROWBAND:
        return get_narrowband_metadata(config)
    elif config.dataset == DatasetType.TORCHSIG_WIDEBAND:
        return get_wideband_metadata(config)
    else:
        raise ValueError(f"Unknown dataset type: {config.dataset}")


def get_narrowband_metadata(config: BaseConfig) -> NarrowbandMetadata:
    """
    Builds and returns metadata for a TorchSig narrowband dataset.

    :param config: Configuration object specifying dataset parameters.
    :returns: NarrowbandMetadata instance with overrides applied.
    """
    metadata = get_default_yaml_config(
        dataset_type="narrowband",
        impairment_level=2,
        train=True,
    )
    metadata["overrides"]["snr_db_min"] = 10
    metadata["overrides"]["impairment_level"] = 2
    metadata["overrides"]["num_iq_samples_dataset"] = config.nfft**2
    metadata["overrides"]["fft_size"] = config.nfft

    metadata = to_dataset_metadata(metadata)
    return metadata


def get_wideband_metadata(config: BaseConfig) -> WidebandMetadata:
    """
    Builds and returns metadata for a TorchSig wideband dataset.

    :param config: Configuration object specifying dataset parameters.
    :returns: WidebandMetadata instance with overrides applied.
    """
    metadata = get_default_yaml_config(
        dataset_type="wideband",
        impairment_level=2,
        train=True,
    )
    metadata["overrides"]["snr_db_min"] = 0
    metadata["overrides"]["impairment_level"] = 2
    metadata["overrides"]["num_iq_samples_dataset"] = config.nfft**2
    metadata["overrides"]["fft_size"] = config.nfft

    # metadata = to_dataset_metadata(metadata)
    return metadata


def build_dataloader(config: BaseConfig) -> TorchSigDataModule:
    """
    Builds and returns a TorchSigDataModule dataloader from the provided configuration.

    :param config: Configuration object specifying dataset parameters.
    :returns: Instantiated TorchSigDataModule for use in training or evaluation.
    """
    return DatasetFactory.create_dataset(config)
