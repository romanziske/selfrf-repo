"""
Extended transformation utilities for RF signal processing and multi-view augmentation pipelines.

This module provides essential building blocks for RF signal data augmentation, spectrogram generation, and multi-view learning transformations used throughout the selfRF library. It extends TorchSig's transformation framework with RF-specific augmentations like additive white Gaussian noise, amplitude scaling, and high-quality spectrogram conversion optimized for self-supervised learning. The module serves as the foundation for complex augmentation strategies in SSL methods like VICRegL, MAE, and DenseCL by providing multi-view generation capabilities and seamless integration between RF domain processing and computer vision transforms. It bridges the gap between raw RF signal processing and modern deep learning pipelines by offering composable transforms that can be chained together for sophisticated data preprocessing workflows. The transforms integrate with both TorchSig's DatasetSignal format and PyTorch's tensor operations, enabling flexible deployment across different stages of the ML pipeline.
"""
from typing import List, Sequence, Tuple, Union
from copy import deepcopy
import numpy as np
import torch

import torchaudio
from torchsig.signals.signal_types import DatasetSignal
from torchsig.transforms.base_transforms import Transform, Compose
from torchsig.transforms.dataset_transforms import DatasetTransform
import torchsig.transforms.functional as torchsig_F

from . import functional as F

__all__ = [
    "MultiViewTransform",
    "MultiViewAndGridTransform",
    "RandomAWGN",
    "AmplitudeScale",
    "ToDtype",
    "SpectrogramImageHighQuality",
    "ToTensor",
    "ToSpectrogramTensor",
    "PyTorchImageTransformWrapper",
]


class MultiViewTransform(Transform):
    """
    Transforms a single signal into multiple independent views for contrastive learning.

    Creates separate copies of input signals and applies different augmentation pipelines to generate diverse views required by self-supervised learning methods.

    :param transforms: Sequence of transform pipelines to apply to create different views
    :type transforms: Sequence[Compose]
    """

    def __init__(self, transforms: Sequence[Compose]) -> None:
        super().__init__()
        self.transforms = transforms

    def __call__(self, signal: DatasetSignal) -> DatasetSignal:
        """
        Creates independent views with separate data copies and returns all views.

        Each transform pipeline receives a fresh copy of the input signal to prevent interference between augmentations.

        :param signal: Input RF signal to transform into multiple views
        :type signal: DatasetSignal
        :returns: Signal with data field containing list of transformed views
        :rtype: DatasetSignal
        """
        views = []
        for transform in self.transforms:
            # Create fresh copy for each transform pipeline
            data_copy = deepcopy(signal)
            transformed_view = transform(data_copy)
            views.append(transformed_view)

        signal.data = [view.data for view in views]
        return signal


class MultiViewAndGridTransform(Transform):
    """
    Transforms a signal into multiple (view, grid) pairs for spatial contrastive learning methods like VICRegL.

    Each transform generates both a view tensor and corresponding spatial grid tensor required for dense feature learning.

    :param transforms: Sequence of transforms that produce (view_tensor, grid_tensor) pairs
    :type transforms: Sequence[Transform]
    """

    def __init__(self, transforms: Sequence[Transform]) -> None:
        super().__init__()
        self.transforms = transforms

    def __call__(self, signal: DatasetSignal) -> DatasetSignal:
        """
        Creates independent (view, grid) pairs and returns them as separate lists.

        Each transform receives a fresh copy and must return a signal with (view_tensor, grid_tensor) data format.

        :param signal: Input RF signal to transform into view-grid pairs
        :type signal: DatasetSignal
        :returns: Signal with data field containing (views_list, grids_list) tuple
        :rtype: DatasetSignal
        """
        views_list = []
        grids_list = []
        for transform_callable in self.transforms:
            # Create fresh copy for each transform pipeline
            signal_copy = deepcopy(signal)
            # Each transform_callable is now expected to return (view_tensor, grid_tensor)
            transformed_signal = transform_callable(signal_copy)
            view_tensor, grid_tensor = transformed_signal.data

            views_list.append(view_tensor)
            grids_list.append(grid_tensor)

        signal.data = (views_list, grids_list)
        return signal


class RandomAWGN(DatasetTransform):
    """
    Adds white Gaussian noise to RF signals with randomized noise power for data augmentation.

    Simulates realistic channel conditions and receiver noise to improve model robustness during training.

    :param noise_power_db: Noise power range in dB, can be fixed value, list, tuple range, or callable
    :type noise_power_db: Union[List, Tuple]
    :param kwargs: Additional arguments passed to DatasetTransform parent class
    """

    def __init__(
        self,
        noise_power_db: Union[List, Tuple] = (0, 20.0),
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.noise_power_db_distribution = self.get_distribution(
            noise_power_db)

    def __call__(self, signal: DatasetSignal) -> DatasetSignal:
        """
        Apply random AWGN to the signal with sampled noise power.

        :param signal: Input RF signal to add noise to
        :type signal: DatasetSignal
        :returns: Signal with added white Gaussian noise
        :rtype: DatasetSignal
        """
        noise_power_db = self.noise_power_db_distribution()
        signal.data = torchsig_F.awgn(
            signal.data,
            noise_power_db=noise_power_db,
            rng=self.random_generator
        )
        # signal.data = signal.data.astype(torchsig_complex_data_type)

        self.update(signal)
        return signal


class AmplitudeScale(DatasetTransform):
    """
    Scales the amplitude of RF signals by a random factor for gain variation augmentation.

    Simulates automatic gain control variations and signal strength changes in realistic RF environments.

    :param scale: Scaling factor range, can be fixed value, list, tuple range, or callable
    :type scale: Union[List, Tuple]
    :param kwargs: Additional arguments passed to DatasetTransform parent class
    """

    def __init__(
        self,
        scale: Union[List, Tuple] = (0.5, 2.0),
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.scale_distribution = self.get_distribution(scale)

    def __call__(self, signal: DatasetSignal) -> DatasetSignal:
        """
        Applies random amplitude scaling to the input signal.

        :param signal: Input RF signal to scale
        :type signal: DatasetSignal
        :returns: Signal with scaled amplitude values
        :rtype: DatasetSignal
        """
        scale_value = self.scale_distribution()
        signal.data = F.amplitude_scale(signal.data, scale_value)
        self.update(signal)
        return signal


class ToDtype(DatasetTransform):
    """
    Converts signal data arrays to specified NumPy dtype for memory optimization or precision control.

    Useful for reducing memory usage by converting to lower precision types or ensuring consistent data types across pipeline stages.

    :param dtype: Target NumPy data type for conversion
    :type dtype: np.dtype
    :param kwargs: Additional arguments passed to DatasetTransform parent class
    """

    def __init__(
            self,
            dtype: np.dtype,
            **kwargs) -> None:
        super().__init__(**kwargs)
        self.dtype = dtype

    def __call__(self, signal: DatasetSignal) -> DatasetSignal:
        """
        Converts the signal data to the specified dtype.

        :param signal: Input signal with data to convert
        :type signal: DatasetSignal
        :returns: Signal with data converted to target dtype
        :rtype: DatasetSignal
        """
        # Use astype to convert the NumPy array to the new dtype.
        signal.data = signal.data.astype(self.dtype)
        self.update(signal)
        return signal


class ToTensor(DatasetTransform):
    """
    Converts NumPy arrays to PyTorch tensors for neural network compatibility.

    Essential bridge between NumPy-based RF processing and PyTorch-based deep learning models.
    """

    def __init__(
        self,
    ) -> None:
        super().__init__()

    def __call__(self, signal: DatasetSignal) -> DatasetSignal:
        """
        Converts signal data from NumPy array to PyTorch tensor.

        :param signal: Input signal with NumPy array data
        :type signal: DatasetSignal
        :returns: Signal with PyTorch tensor data
        :rtype: DatasetSignal
        """

        # convert to torch tensor
        tensor = torch.from_numpy(signal.data)

        signal.data = tensor
        self.update(signal)
        return signal


class ToSpectrogramTensor(DatasetTransform):
    """
    Converts 2D spectrogram arrays to 3D tensors with channel dimension for CNN compatibility.

    Reshapes spectrogram data to (C, H, W) format expected by computer vision models where C=1 for grayscale spectrograms.

    :param to_float_32: Whether to convert tensor to float32 precision
    :type to_float_32: bool
    :raises ValueError: If input data is not in 2D spectrogram format
    """

    def __init__(
        self,
        to_float_32: bool = False
    ) -> None:
        super().__init__()
        self.to_float_32 = to_float_32

    def __call__(self, signal: DatasetSignal) -> DatasetSignal:
        """
        Converts 2D spectrogram to 3D tensor with added channel dimension.

        :param signal: Input signal with 2D spectrogram data
        :type signal: DatasetSignal
        :returns: Signal with 3D tensor data in (C, H, W) format
        :rtype: DatasetSignal
        :raises ValueError: If signal data is not 2-dimensional
        """
        # check if data is in spectrogram format
        if len(signal.data.shape) != 2:
            raise ValueError("Data must be in spectrogram format (2D)")

        # Make sure the array is contiguous before converting to torch tensor
        # This fixes the negative stride issue
        if not signal.data.flags.c_contiguous:
            signal.data = np.ascontiguousarray(signal.data)

        # convert to torch tensor
        tensor = torch.from_numpy(signal.data)

        # convert to float32 if requested
        if self.to_float_32:
            tensor = tensor.float()

        # add channel dimension
        tensor = tensor.unsqueeze(0)

        signal.data = tensor
        self.update(signal)
        return signal


class SpectrogramImageHighQuality(DatasetTransform):
    """
    High-quality RF spectrogram transformation optimized for self-supervised learning applications.

    Converts complex IQ data into normalized grayscale spectrogram images using research-grade processing with proper frequency domain orientation and scaling.

    :param nfft: Number of FFT points for spectrogram computation
    :type nfft: int
    :param db_scale: Whether to apply logarithmic dB scaling to power spectrum
    :type db_scale: bool
    :param normalize: Whether to apply infinity norm normalization before dB conversion
    :type normalize: bool
    :param invert: Whether to invert colors for black-hot visualization
    :type invert: bool
    :param to_tensor: Whether to add channel dimension for tensor format
    :type to_tensor: bool
    :param resize: Target dimensions for output spectrogram as (height, width)
    :type resize: Union[Tuple[int, int], List[int]]
    :param kwargs: Additional arguments passed to DatasetTransform parent class
    """

    def __init__(
        self,
        nfft: int = 512,
        db_scale: bool = True,
        normalize: bool = True,
        invert: bool = False,
        to_tensor=False,
        resize: Union[Tuple[int, int], List[int]] = None,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.nfft = nfft
        self.db_scale = db_scale
        self.normalize = normalize
        self.invert = invert
        self.to_tensor = to_tensor
        self.resize = resize

        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=self.nfft,
            win_length=self.nfft,
            hop_length=self.nfft,
            window_fn=torch.blackman_window,
            normalized=False,
            center=False,
            onesided=False,
            power=2,
        )

    def __call__(self, signal: DatasetSignal) -> DatasetSignal:
        """
        Convert complex IQ data to high-quality grayscale spectrogram image.

        Applies research-grade spectrogram processing with proper frequency domain alignment and normalization for optimal ML performance.

        :param signal: Input signal with complex IQ data
        :type signal: DatasetSignal
        :returns: Signal with processed spectrogram image data
        :rtype: DatasetSignal
        """
        # Convert to torch tensor
        data = torch.from_numpy(signal.data)

        # Apply spectrogram transform
        x = self.spectrogram(data)

        # Normalize by infinity norm as in paper
        if self.normalize:
            norm_val = torch.norm(x.flatten(), p=float("inf"))
            x = x / (norm_val + 1e-12)

        # Apply FFT shift and flip for proper orientation
        # pylint: disable=not-callable
        x = torch.fft.fftshift(x, dim=0)
        x = torch.flip(x, dims=[0])  # same as flipud in the paper

        # Convert to dB scale
        if self.db_scale:
            x = 10 * torch.log10(x + 1e-12)

        # min–max to [0,1]
        x_min = x.min()
        x_max = x.max()
        if (x_max - x_min) < 1e-12:  # Avoid division by zero if x is flat
            x = np.zeros_like(x)
        else:
            x = (x - x_min) / (x_max - x_min)

        # Invert colors if requested (common in RF visualization)
        if self.invert:
            x = 1.0 - x

        # Resize if specified
        if self.resize is not None:
            # x has shape [H, W]
            x = x.unsqueeze(0).unsqueeze(0)  # → [1,1,H,W]
            x = torch.nn.functional.interpolate(
                x,
                size=self.resize,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            x = x.squeeze(0).squeeze(0)  # → [H',W']

        # add channel dimension
        if self.to_tensor:
            x = x.unsqueeze(0)

        signal.data = x

        self.update(signal)
        return signal


class PyTorchImageTransformWrapper(DatasetTransform):
    """
    Wrapper to apply PyTorch image transforms to TorchSig spectrogram outputs with proper format handling.

    Bridges TorchSig's DatasetSignal format with PyTorch's computer vision transforms by ensuring proper tensor dimensions and data types.

    :param torch_transform: PyTorch transform to apply to spectrogram data
    :param kwargs: Additional arguments passed to DatasetTransform parent class
    :raises ValueError: If input data is not in expected 3D tensor format after channel dimension handling
    """

    def __init__(self, torch_transform, **kwargs):
        super().__init__(**kwargs)
        self.torch_transform = torch_transform

    def __call__(self, signal: DatasetSignal) -> DatasetSignal:
        """
        Applies PyTorch image transform to spectrogram data with automatic dimension handling.

        :param signal: Input signal with spectrogram tensor data
        :type signal: DatasetSignal
        :returns: Signal with transformed spectrogram data
        :rtype: DatasetSignal
        :raises ValueError: If data cannot be formatted as 3D [C, H, W] tensor
        """
        data = signal.data

        # Check if data is tensor, if not convert to tensor
        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(data)

       # add channel dimension if not present
        if data.ndim == 2:
            data = data.unsqueeze(0)

        # throw error if data is not a 3D tensor
        if data.ndim != 3:
            raise ValueError("Data must be a  [C, H, W] tensor.")

        data = self.torch_transform(data)

        # Convert back to DatasetSignal
        signal.data = data
        # Ensure the signal is still a DatasetSignal
        self.update(signal)
        return signal
