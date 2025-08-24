"""
DenseCL transformation pipeline for dense contrastive learning on RF signal spectrograms.

This module implements the data augmentation strategy for DenseCL (Dense Contrastive Learning), a self-supervised learning method that learns dense representations by contrasting features at multiple spatial locations within spectrograms. DenseCL extends traditional contrastive learning from global image representations to pixel-level features, making it particularly suitable for RF signal analysis where spatial frequency patterns are crucial. The transform applies RF-specific augmentations including time reversal, spectral inversion, and additive noise, followed by spectrogram conversion and image-like transformations. It generates two augmented views of each input signal for contrastive learning, enabling the model to learn spatially-aware representations. This integrates with the selfRF SSL framework by providing the multi-view data preprocessing required for dense contrastive representation learning on RF signals.
"""
import torchvision.transforms.v2 as T

from torchsig.signals import DatasetSignal
from torchsig.transforms.base_transforms import RandomApply, Compose
from torchsig.transforms.dataset_transforms import (
    TimeReversal,
    SpectralInversionDatasetTransform,
    DatasetTransform,
    CutOut,
    RandomMagRescale,
)

from selfrf.transforms.extra.transforms import (
    MultiViewTransform, PyTorchImageTransformWrapper, RandomAWGN, SpectrogramImageHighQuality)


class DenseCLTransform(MultiViewTransform):
    """
    Implements the dual-view augmentation strategy for DenseCL training on RF signal data.

    Creates two differently augmented views of each input signal to enable dense contrastive learning at the pixel level.

    :param nfft: Number of FFT points for spectrogram generation
    :type nfft: int
    :param tensor_transform: Transform to convert RF signals to spectrogram tensors
    :type tensor_transform: DatasetTransform
    :param crop_min_scale: Minimum scale factor for random resized cropping
    :type crop_min_scale: float
    :param crop_max_scale: Maximum scale factor for random resized cropping
    :type crop_max_scale: float
    :param hf_prob: Probability of applying time reversal augmentation
    :type hf_prob: float
    :param vf_prob: Probability of applying spectral inversion augmentation
    :type vf_prob: float
    :param cutout_duration: Duration range for cutout augmentation as fraction of signal length
    :type cutout_duration: tuple
    :param min_amplitude_scale: Minimum amplitude scaling factor for magnitude rescaling
    :type min_amplitude_scale: float
    :param max_amplitude_scale: Maximum amplitude scaling factor for magnitude rescaling
    :type max_amplitude_scale: float
    :param noise_power_db: Range of noise power in dB for AWGN augmentation
    :type noise_power_db: tuple
    """

    def __init__(
        self,
        nfft: int = 512,
        tensor_transform: DatasetTransform = SpectrogramImageHighQuality,
        crop_min_scale: float = 0.2,
        crop_max_scale: float = 1,
        hf_prob: float = 0.5,
        vf_prob: float = 0.5,
        cutout_duration: tuple = (0, 0.05),
        min_amplitude_scale: float = 0.7,
        max_amplitude_scale: float = 1.5,
        noise_power_db: tuple = (-30, 20),
        **kwargs,
    ):
        view_transform1 = DenseCLViewTransform(
            nfft=nfft,
            spectrogram_transform=tensor_transform,
            crop_min_scale=crop_min_scale,
            crop_max_scale=crop_max_scale,
            tr_prob=hf_prob,
            si_prob=vf_prob,
            cutout_duration=cutout_duration,
            min_amplitude_scale=min_amplitude_scale,
            max_amplitude_scale=max_amplitude_scale,
            noise_power_db=noise_power_db,
        )

        view_transform2 = DenseCLViewTransform(
            nfft=nfft,
            spectrogram_transform=tensor_transform,
            crop_min_scale=crop_min_scale,
            crop_max_scale=crop_max_scale,
            tr_prob=hf_prob,
            si_prob=vf_prob,
            cutout_duration=cutout_duration,
            min_amplitude_scale=min_amplitude_scale,
            max_amplitude_scale=max_amplitude_scale,
            noise_power_db=noise_power_db,
        )
        super().__init__(transforms=[view_transform1, view_transform2])


class DenseCLViewTransform:
    """
    Creates a single augmented view for DenseCL training with RF-specific transformations.

    Applies a sequence of RF domain augmentations followed by spectrogram conversion and image-like transformations.

    :param nfft: Number of FFT points for spectrogram generation and output size
    :type nfft: int
    :param crop_min_scale: Minimum scale factor for random resized cropping
    :type crop_min_scale: float
    :param crop_max_scale: Maximum scale factor for random resized cropping
    :type crop_max_scale: float
    :param tr_prob: Probability of applying time reversal augmentation
    :type tr_prob: float
    :param si_prob: Probability of applying spectral inversion augmentation
    :type si_prob: float
    :param cutout_duration: Duration range for cutout augmentation as fraction of signal length
    :type cutout_duration: tuple
    :param min_amplitude_scale: Minimum amplitude scaling factor for magnitude rescaling
    :type min_amplitude_scale: float
    :param max_amplitude_scale: Maximum amplitude scaling factor for magnitude rescaling
    :type max_amplitude_scale: float
    :param noise_power_db: Range of noise power in dB for AWGN augmentation
    :type noise_power_db: tuple
    :param spectrogram_transform: Transform to convert signal to spectrogram representation
    :type spectrogram_transform: DatasetTransform
    """

    def __init__(
        self,
        nfft: int,
        crop_min_scale: float,
        crop_max_scale: float,
        tr_prob: float,
        si_prob: float,
        cutout_duration: tuple,
        min_amplitude_scale: float,
        max_amplitude_scale: float,
        noise_power_db: tuple,
        spectrogram_transform: DatasetTransform = SpectrogramImageHighQuality(
            nfft=512),
    ):

        transform = [
            RandomAWGN(noise_power_db),
            RandomApply(TimeReversal(), tr_prob),
            RandomApply(SpectralInversionDatasetTransform(), si_prob),
            CutOut(
                duration=cutout_duration,
                cut_type=["low_noise"],
            ),
            RandomMagRescale(
                scale=(min_amplitude_scale, max_amplitude_scale),
            ),
            spectrogram_transform,
            PyTorchImageTransformWrapper(
                T.RandomResizedCrop(size=nfft, scale=(
                    crop_min_scale, crop_max_scale))
            ),
        ]
        self.transform = Compose(transform)

    def __call__(self, signal: DatasetSignal):
        """
        Applies the complete DenseCL view transformation pipeline to the input signal.

        :param signal: Input RF signal to transform into augmented spectrogram view
        :type signal: DatasetSignal
        :returns: Transformed signal with applied DenseCL augmentations
        :rtype: DatasetSignal
        """

        return self.transform(signal)
