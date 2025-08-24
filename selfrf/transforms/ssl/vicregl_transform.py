"""
Adapted VICRegLTransform for RF data using DatasetSignal.

This module implements VICRegL (Variance-Invariance-Covariance Regularization with Local features) transformations specifically designed for RF signal data. VICRegL is a self-supervised learning method that creates multiple views of input data to learn robust representations without requiring labeled data. The module provides RF-aware data augmentations including time reversal, spectral inversion, and additive noise, followed by spectrogram conversion and image-like transformations. It generates both global and local views of RF spectrograms with corresponding spatial grids for the VICRegL loss computation. This integrates with the broader selfRF SSL pipeline by providing the data augmentation strategy required for contrastive representation learning on RF signals.
"""
import warnings
from typing import Literal, Tuple

from lightly.transforms.random_crop_and_flip_with_grid import RandomResizedCropAndFlip


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
    MultiViewAndGridTransform,
    PyTorchImageTransformWrapper,
    RandomAWGN,
    SpectrogramImageHighQuality,
)


# suppress all torchvision warnings
warnings.filterwarnings("ignore", message=".*get_image_size.*")


class VICRegLViewTransform(DatasetTransform):
    """
    Transforms a DatasetSignal for a single view in VICRegL style for RF data.

    Applies RF-specific augmentations, converts to spectrogram, then applies image-like transformations with spatial grid generation.

    :param crop_size: Output size of the cropped spectrogram view
    :type crop_size: int
    :param crop_min_scale: Minimum scale factor for random cropping
    :type crop_min_scale: float
    :param crop_max_scale: Maximum scale factor for random cropping
    :type crop_max_scale: float
    :param grid_size: Size of spatial grid for VICRegL loss computation
    :type grid_size: int
    :param spectrogram_transform: Transform to convert signal to spectrogram
    :type spectrogram_transform: DatasetTransform
    :param noise_power_db: Range of noise power in dB for AWGN augmentation
    :type noise_power_db: Tuple[float, float]
    :param tr_prob: Probability of applying time reversal augmentation
    :type tr_prob: float
    :param si_prob: Probability of applying spectral inversion augmentation
    :type si_prob: float
    :param cutout_duration: Duration range for cutout augmentation as fraction of signal length
    :type cutout_duration: tuple
    :param min_amplitude_scale: Minimum amplitude scaling factor
    :type min_amplitude_scale: float
    :param max_amplitude_scale: Maximum amplitude scaling factor
    :type max_amplitude_scale: float
    """

    def __init__(
        self,
        # Spectrogram parameters
        crop_size: int,
        crop_min_scale: float,
        crop_max_scale: float,
        grid_size: int,
        spectrogram_transform: DatasetTransform = SpectrogramImageHighQuality(
            nfft=512),
        noise_power_db: Tuple[float, float] = (-30, 20),
        tr_prob: float = 0.5,
        si_prob: float = 0.5,
        cutout_duration: tuple = (0, 0.20),
        min_amplitude_scale: float = 0.7,
        max_amplitude_scale: float = 1.5,
        **kwargs,  # for DatasetTransform
    ):
        super().__init__(**kwargs)

        transform = [
            RandomAWGN(noise_power_db),
            RandomApply(TimeReversal(), tr_prob),
            RandomApply(SpectralInversionDatasetTransform(), si_prob),
            CutOut(
                duration=cutout_duration,
                cut_type=["avg_noise"],
            ),
            RandomMagRescale(
                scale=(min_amplitude_scale, max_amplitude_scale),
            ),
            spectrogram_transform,
            PyTorchImageTransformWrapper(
                RandomResizedCropAndFlip(
                    crop_size=crop_size,
                    crop_min_scale=crop_min_scale,
                    crop_max_scale=crop_max_scale,
                    hf_prob=0,
                    vf_prob=0,
                    grid_size=grid_size,
                ))
        ]
        self.transform = Compose(transform)

    def __call__(self, signal: DatasetSignal) -> DatasetSignal:
        """
        Applies the complete VICRegL view transformation pipeline to input signal.

        :param signal: Input RF signal to transform
        :type signal: DatasetSignal
        :returns: Transformed signal with spectrogram and spatial grid
        :rtype: DatasetSignal
        """
        return self.transform(signal)


class VICRegLTransform(MultiViewAndGridTransform):
    """
    Adapted VICRegLTransform for RF data using DatasetSignal.

    Generates multiple views (global and local) of spectrograms from input signals with corresponding grids for VICRegL loss computation.

    :param tensor_transform: Transform to convert signals to tensor spectrograms
    :type tensor_transform: DatasetTransform
    :param n_global_views: Number of global view spectrograms to generate
    :type n_global_views: int
    :param n_local_views: Number of local view spectrograms to generate
    :type n_local_views: int
    :param global_crop_size: Output size of global view spectrograms
    :type global_crop_size: int
    :param local_crop_size: Output size of local view spectrograms
    :type local_crop_size: int
    :param global_crop_scale: Scale range for global view random cropping
    :type global_crop_scale: Tuple[float, float]
    :param local_crop_scale: Scale range for local view random cropping
    :type local_crop_scale: Tuple[float, float]
    :param global_grid_size: Grid size for global views (e.g., for ResNet50 output)
    :type global_grid_size: int
    :param local_grid_size: Grid size for local views
    :type local_grid_size: int
    :param noise_power_db: Range of noise power in dB for AWGN augmentation
    :type noise_power_db: Tuple[float, float]
    :param global_time_reversal_prob: Probability of time reversal for global views
    :type global_time_reversal_prob: float
    :param local_time_reversal_prob: Probability of time reversal for local views
    :type local_time_reversal_prob: float
    :param global_spectral_inversion_prob: Probability of spectral inversion for global views
    :type global_spectral_inversion_prob: float
    :param local_spectral_inversion_prob: Probability of spectral inversion for local views
    :type local_spectral_inversion_prob: float
    :param apply_on: Whether to apply transformations on spectrogram
    :type apply_on: Literal["spectrogram"]
    """

    def __init__(
        self,
        # Spectrogram generation
        tensor_transform: DatasetTransform,
        # View counts
        n_global_views: int = 2,
        n_local_views: int = 6,

        # Crop and Grid parameters
        global_crop_size: int = 224,  # Output size of global view spectrograms
        local_crop_size: int = 96,   # Output size of local view spectrograms
        global_crop_scale: Tuple[float, float] = (0.5, 1),
        local_crop_scale: Tuple[float, float] = (0.1, 0.5),

        # Grid size for global views (e.g., for ResNet50 output)
        global_grid_size: int = 7,
        local_grid_size: int = 3,  # Grid size for local views

        # RF-specific augmentation probabilities
        noise_power_db: Tuple[float, float] = (-45, -25),
        global_time_reversal_prob: float = 0.5,
        local_time_reversal_prob: float = 0.0,
        global_spectral_inversion_prob: float = 0.2,
        local_spectral_inversion_prob: float = 0.0,
        apply_on: Literal["spectrogram"] = "spectrogram",
    ):
        global_view = VICRegLViewTransform(
            spectrogram_transform_cls=tensor_transform,
            crop_size=global_crop_size,
            crop_min_scale=global_crop_scale[0],
            crop_max_scale=global_crop_scale[1],
            grid_size=global_grid_size,
            noise_power_db=noise_power_db,
            time_reversal_prob=global_time_reversal_prob,
            spectral_inversion_prob=global_spectral_inversion_prob,

        )

        local_view = VICRegLViewTransform(
            spectrogram_transform_cls=tensor_transform,
            crop_size=local_crop_size,
            crop_min_scale=local_crop_scale[0],
            crop_max_scale=local_crop_scale[1],
            grid_size=local_grid_size,
            noise_power_db=noise_power_db,
            time_reversal_prob=local_time_reversal_prob,
            spectral_inversion_prob=local_spectral_inversion_prob,
        )

        transforms = [global_view] * n_global_views + \
            [local_view] * n_local_views

        super().__init__(transforms=transforms)


class OptimizedVICRegLTransform(DatasetTransform):
    """
    Memory-optimized VICRegL transformation for RF signal data.

    This class implements an efficient version of VICRegL (Variance-Invariance-Covariance 
    Regularization with Local features) transformations specifically designed for RF signals.
    Unlike the standard VICRegLTransform, this optimized version applies IQ-domain augmentations
    once to the input signal, converts to spectrogram once, then generates multiple views through 
    cropping operations only. This approach significantly reduces memory usage and computational 
    overhead by avoiding expensive deepcopy operations and redundant spectrogram computations.

    The transformation pipeline consists of:
    1. IQ-domain augmentations (noise, time reversal, spectral inversion, cutout, amplitude scaling)
    2. Single spectrogram conversion
    3. Multiple view generation via random cropping with spatial grids

    This generates both global views (larger crops for overall structure) and local views 
    (smaller crops for fine-grained features) required for VICRegL's multi-scale contrastive learning.

    :param tensor_transform: Transform to convert IQ signals to spectrogram tensors
    :type tensor_transform: DatasetTransform
    :param n_global_views: Number of global view spectrograms to generate (typically 2)
    :type n_global_views: int
    :param n_local_views: Number of local view spectrograms to generate (typically 6)
    :type n_local_views: int
    :param global_crop_size: Output size of global view spectrograms in pixels
    :type global_crop_size: int
    :param local_crop_size: Output size of local view spectrograms in pixels
    :type local_crop_size: int
    :param global_crop_scale: Scale range (min, max) for global view random cropping
    :type global_crop_scale: Tuple[float, float]
    :param local_crop_scale: Scale range (min, max) for local view random cropping  
    :type local_crop_scale: Tuple[float, float]
    :param global_grid_size: Spatial grid size for global views (e.g., 7 for ResNet50 features)
    :type global_grid_size: int
    :param local_grid_size: Spatial grid size for local views (e.g., 3 for fine-grained features)
    :type local_grid_size: int
    :param noise_power_db: AWGN noise power range in dB for signal augmentation
    :type noise_power_db: Tuple[float, float]
    :param tr_prob: Probability of applying time reversal augmentation
    :type tr_prob: float
    :param si_prob: Probability of applying spectral inversion augmentation
    :type si_prob: float
    :param cutout_duration: Duration range for cutout augmentation as fraction of signal length
    :type cutout_duration: tuple
    :param min_amplitude_scale: Minimum factor for random amplitude scaling
    :type min_amplitude_scale: float
    :param max_amplitude_scale: Maximum factor for random amplitude scaling
    :type max_amplitude_scale: float
    :param apply_on: Domain where transformations are applied (always "spectrogram")
    :type apply_on: Literal["spectrogram"]

    """

    def __init__(
        self,
        tensor_transform: DatasetTransform,
        n_global_views: int = 2,
        n_local_views: int = 6,
        global_crop_size: int = 224,
        local_crop_size: int = 96,
        global_crop_scale: Tuple[float, float] = (0.5, 1),
        local_crop_scale: Tuple[float, float] = (0.1, 0.5),
        global_grid_size: int = 7,
        local_grid_size: int = 3,
        noise_power_db: Tuple[float, float] = (-30, 20),
        tr_prob: float = 0.5,
        si_prob: float = 0.5,
        cutout_duration: tuple = (0, 0.20),
        min_amplitude_scale: float = 0.7,
        max_amplitude_scale: float = 1.5,
        apply_on: Literal["spectrogram"] = "spectrogram",
        **kwargs
    ):
        super().__init__(**kwargs)

        # Store parameters
        self.n_global_views = n_global_views
        self.n_local_views = n_local_views
        self.tensor_transform = tensor_transform

        # **OPTIMIZED: Create a single IQ augmentation pipeline (NO deepcopy)**
        self.iq_augmentations = Compose([
            RandomAWGN(noise_power_db),
            RandomApply(TimeReversal(), tr_prob),
            RandomApply(SpectralInversionDatasetTransform(), si_prob),
            CutOut(duration=cutout_duration, cut_type=["avg_noise"]),
            RandomMagRescale(scale=(min_amplitude_scale, max_amplitude_scale)),
        ])

        # Crop transforms (applied per view)
        self.global_crop = RandomResizedCropAndFlip(
            crop_size=global_crop_size,
            crop_min_scale=global_crop_scale[0],
            crop_max_scale=global_crop_scale[1],
            hf_prob=0,
            vf_prob=0,
            grid_size=global_grid_size,
        )

        self.local_crop = RandomResizedCropAndFlip(
            crop_size=local_crop_size,
            crop_min_scale=local_crop_scale[0],
            crop_max_scale=local_crop_scale[1],
            hf_prob=0,
            vf_prob=0,
            grid_size=local_grid_size,
        )

    def __call__(self, signal: DatasetSignal) -> DatasetSignal:
        """
        Applies the complete optimized VICRegL transformation pipeline to an RF signal.

        This method efficiently generates multiple augmented views by applying IQ-domain 
        augmentations once, converting to spectrogram once, then creating multiple views 
        through random cropping operations. Each view comes with a corresponding spatial 
        grid for VICRegL loss computation.

        :param signal: Input RF signal in IQ format to transform
        :type signal: DatasetSignal
        :returns: Transformed signal containing (views_list, grids_list) tuple in data attribute
        :rtype: DatasetSignal

        The returned signal.data contains:
            - views_list: List of tensors, each representing a cropped spectrogram view
            - grids_list: List of spatial grid tensors corresponding to each view

        Views are ordered as: [global_view_0, global_view_1, local_view_0, ..., local_view_N]
        where N = n_local_views - 1.

        """
        # **STEP 1: Apply IQ augmentations once (NO deepcopy)**
        augmented_signal = self.iq_augmentations(signal)

        # **STEP 2: Convert to spectrogram once**
        spectrogram_signal = self.tensor_transform(augmented_signal)
        spectrogram = spectrogram_signal.data  # Extract tensor

        views_list = []
        grids_list = []

        # **STEP 3: Generate global views through cropping only**
        for _ in range(self.n_global_views):
            view, grid = self.global_crop(spectrogram)
            views_list.append(view)
            grids_list.append(grid)

        # **STEP 4: Generate local views through cropping only**
        for _ in range(self.n_local_views):
            view, grid = self.local_crop(spectrogram)
            views_list.append(view)
            grids_list.append(grid)

        # **Return signal with views and grids**
        signal.data = (views_list, grids_list)
        return signal
