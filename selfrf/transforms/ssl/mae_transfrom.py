"""
MAE (Masked Autoencoder) transformation pipeline for RF signal self-supervised learning.

This module implements the data augmentation strategy used in Masked Autoencoder (MAE) training, adapted specifically for RF signal spectrograms. MAE is a self-supervised learning method that masks random patches of input images and trains a model to reconstruct the missing content, learning robust representations in the process. The transform applies standard computer vision augmentations like random cropping and horizontal flipping to RF spectrograms generated from complex IQ data. It integrates with the selfRF SSL framework by providing the preprocessing pipeline required for MAE-based representation learning on RF signals. The module works exclusively with spectrogram representations, converting time-domain signals to frequency-domain images suitable for vision transformer architectures.
"""
from typing import Literal, Tuple, Union


import torch
from lightly.transforms.torchvision_v2_compatibility import torchvision_transforms as T

from torchsig.signals import DatasetSignal
from torchsig.transforms.dataset_transforms import DatasetTransform

from selfrf.transforms.extra.transforms import SpectrogramImageHighQuality, PyTorchImageTransformWrapper


class ClampToRange(DatasetTransform):
    """Clamp values to [0,1] range."""

    def __init__(self, min_val=0.0, max_val=1.0, **kwargs):
        super().__init__(**kwargs)
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, signal: DatasetSignal) -> DatasetSignal:
        signal.data = torch.clamp(signal.data, self.min_val, self.max_val)
        self.update(signal)
        return signal


class MAETransform:
    """
    Implements the view augmentation for MAE training on RF signal spectrograms.

    Applies random resized crop and horizontal flip augmentations following the standard MAE preprocessing pipeline adapted for RF data.

    :param tensor_transform: Transform to convert RF signals to spectrogram tensors
    :type tensor_transform: DatasetTransform
    :param input_size: Target size of the output spectrogram image in pixels
    :type input_size: Union[int, Tuple[int, int]]
    :param min_scale: Minimum size of the randomized crop relative to the input size
    :type min_scale: float
    :param apply_on: Data representation to apply transforms on, only "spectrogram" supported
    :type apply_on: Literal["spectrogram", "iq"]
    :raises ValueError: If apply_on is set to "iq" which is not supported
    """

    def __init__(
        self,
        tensor_transform: DatasetTransform = SpectrogramImageHighQuality,
        input_size: Union[int, Tuple[int, int]] = 224,
        min_scale: float = 0.2,
        apply_on: Literal["spectrogram", "iq"] = "spectrogram",
    ):

        if apply_on == "iq":
            raise ValueError(
                "MAETransform is not supported for IQ data. Use MAETransformIQ instead."
            )

        transforms = [
            tensor_transform,
            PyTorchImageTransformWrapper(
                torch_transform=T.RandomResizedCrop(
                    input_size, scale=(min_scale, 1.0), interpolation=3
                ),
            ),
            PyTorchImageTransformWrapper(
                torch_transform=T.RandomHorizontalFlip(),
            ),
            ClampToRange(min_val=0.0, max_val=1.0),
        ]

        self.transform = T.Compose(transforms)

    def __call__(self, signal: DatasetSignal) -> DatasetSignal:
        """
        Applies the complete MAE transformation pipeline to the input RF signal.

        :param signal: Input RF signal to transform into augmented spectrogram
        :type signal: DatasetSignal
        :returns: Transformed signal with applied MAE augmentations
        :rtype: DatasetSignal
        """

        return self.transform(signal)
