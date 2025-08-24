"""
Legacy TorchSig transforms preserved from v0.6.0 for backward compatibility and specialized RF signal processing.

This module maintains compatibility with older TorchSig transform implementations that have been deprecated or significantly modified in newer versions of the library. These legacy transforms are essential for reproducing results from earlier research work and provide specialized RF signal processing operations that may not be available in the current TorchSig API. The transforms focus on fundamental RF domain operations including time shifting, frequency shifting, phase manipulation, amplitude scaling, and noise addition with specific behavioral characteristics that differ from their modern counterparts. They are particularly important for maintaining consistency with existing datasets, reproducing published results, and supporting research workflows that depend on the exact algorithmic behavior of the v0.6.0 implementation. This module integrates with the broader selfRF ecosystem by providing the transform building blocks used in data augmentation pipelines, preprocessing workflows, and signal conditioning operations throughout the library, ensuring that legacy code and research can continue to function correctly while new development can leverage updated APIs.
"""

from typing import Tuple, Union
import numpy as np

from torchsig.signals.signal_types import DatasetSignal
from torchsig.transforms.dataset_transforms import DatasetTransform
from torchsig.utils.dsp import torchsig_complex_data_type
import torchsig.transforms.functional as torchsig_F

import selfrf.transforms.extra.torchsig_legacy_functional as F_LEGACY

__all__ = [
    "Identity",
    "RandomTimeShift",
    "AmplitudeReversal",
    "RandomFrequencyShift",
    "RandomPhaseShift",
    "TargetSNR",
]


class Identity(DatasetTransform):
    """
    Pass-through transform that returns input signal unchanged for pipeline compatibility.

    Useful as a placeholder in transform pipelines or for conditional processing where no modification is needed.
    """

    def __call__(self, signal: DatasetSignal) -> DatasetSignal:
        """
        Returns the input signal without any modifications.

        :param signal: Input RF signal to pass through unchanged
        :type signal: DatasetSignal
        :returns: The same signal without any transformations applied
        :rtype: DatasetSignal
        """
        return signal


class RandomTimeShift(DatasetTransform):
    """
    Shifts RF signal in time by a random percentage of its length with zero-padding.

    Simulates timing synchronization errors and variable signal arrival times in realistic RF environments.

    :param shift_pct_range: Range of shift percentages as (min_shift_pct, max_shift_pct) where values are between -1.0 and 1.0
    :type shift_pct_range: Tuple[float, float]
    :param kwargs: Additional arguments passed to DatasetTransform parent class
    :raises ValueError: If shift percentages are outside [-1.0, 1.0] range or min > max
    """

    def __init__(
        self,
        shift_pct_range: Tuple[float, float] = (-0.1, 0.1),
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        min_shift_pct, max_shift_pct = shift_pct_range

        if not -1.0 <= min_shift_pct <= 1.0 or not -1.0 <= max_shift_pct <= 1.0:
            raise ValueError("Shift percentages must be between -1.0 and 1.0")
        if min_shift_pct > max_shift_pct:
            raise ValueError(
                "min_shift_pct cannot be greater than max_shift_pct")

        self.min_shift_pct = min_shift_pct
        self.max_shift_pct = max_shift_pct

    def __call__(self, signal: DatasetSignal) -> DatasetSignal:
        """
        Applies random time shift to the input signal based on configured percentage range.

        :param signal: Input RF signal to shift in time
        :type signal: DatasetSignal
        :returns: Time-shifted signal with zero-padding to maintain original length
        :rtype: DatasetSignal
        """
        signal_length = signal.data.shape[-1]
        # Calculate absolute shift range based on percentages
        min_absolute_shift = int(signal_length * self.min_shift_pct)
        max_absolute_shift = int(signal_length * self.max_shift_pct)

        # Ensure min is not greater than max after integer conversion
        if min_absolute_shift > max_absolute_shift:
            # This can happen if pct range is small and signal length is small
            # Default to no shift in this edge case
            return signal

        if min_absolute_shift == 0 and max_absolute_shift == 0:
            # No shift possible or needed
            return signal

        # Generate random shift within the calculated absolute range (inclusive)
        shift = self.random_generator.integers(
            min_absolute_shift, max_absolute_shift + 1
        )

        if shift != 0:
            # Apply the legacy functional time shift
            # Ensure data is contiguous for potential C-extensions in functional
            signal.data = np.ascontiguousarray(
                F_LEGACY.time_shift(signal.data, shift))
            signal.data = signal.data.astype(
                torchsig_complex_data_type)  # Ensure correct dtype
            self.update(signal)  # Update metadata if necessary

        return signal


class AmplitudeReversal(DatasetTransform):
    """
    Applies amplitude reversal by multiplying signal by -1, equivalent to π phase shift.

    Simulates signal polarity inversion that can occur in RF systems due to hardware configuration or channel effects.
    """

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, signal: DatasetSignal) -> DatasetSignal:
        """
        Applies amplitude reversal transformation to the input signal.

        :param signal: Input RF signal to reverse amplitude
        :type signal: DatasetSignal
        :returns: Signal with reversed amplitude (multiplied by -1)
        :rtype: DatasetSignal
        """
        # Apply the transformation to the signal data
        signal.data = F_LEGACY.amplitude_reversal(signal.data)
        # Ensure the data type is preserved
        signal.data = signal.data.astype(torchsig_complex_data_type)
        self.update(signal)
        return signal


class RandomFrequencyShift(DatasetTransform):
    """
    Shifts RF signal in frequency domain by random amount within specified range.

    Simulates carrier frequency offset, Doppler effects, and local oscillator drift in realistic RF communication systems.

    :param freq_shift_range: Frequency shift range as (min_freq, max_freq) relative to sample rate
    :type freq_shift_range: Union[list, tuple]
    :param kwargs: Additional arguments passed to DatasetTransform parent class
    """

    def __init__(
        self,
        freq_shift_range: Union[list, tuple] = (-0.3, 0.3),
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        # Store range directly
        self.min_freq = freq_shift_range[0]
        self.max_freq = freq_shift_range[1]

    def __call__(self, signal: DatasetSignal) -> DatasetSignal:
        """
        Applies random frequency shift to the input signal within configured range.

        :param signal: Input RF signal to frequency shift
        :type signal: DatasetSignal
        :returns: Frequency-shifted signal with random offset
        :rtype: DatasetSignal
        """
        freq_shift = np.random.uniform(self.min_freq, self.max_freq)

        # Apply frequency shift
        signal.data = np.ascontiguousarray(
            F_LEGACY.freq_shift(signal.data, freq_shift))

        # Ensure the data type is preserved
        signal.data = signal.data.astype(torchsig_complex_data_type)

        self.update(signal)
        return signal


class RandomPhaseShift(DatasetTransform):
    """
    Applies random phase shift to RF signal by multiplying with complex exponential.

    Simulates phase offset introduced by RF components, propagation delays, and local oscillator phase noise.

    :param phase_range: Range of phase shift in radians as (min, max) tuple
    :type phase_range: Tuple[float, float]
    :param kwargs: Additional arguments passed to DatasetTransform parent class
    """

    def __init__(
        self,
        phase_range: Tuple[float, float] = (0, np.pi/4),
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.phase_range_distribution = self.get_distribution(phase_range)

    def __call__(self, signal: DatasetSignal) -> DatasetSignal:
        """
        Applies random phase shift to the input signal within configured range.

        :param signal: Input RF signal to phase shift
        :type signal: DatasetSignal
        :returns: Phase-shifted signal rotated in complex plane
        :rtype: DatasetSignal
        """
        # Get random phase shift value
        phase_shift = self.phase_range_distribution()

        # Create phase shift factor e^(jθ)
        phase_factor = np.exp(1j * phase_shift)

        # Apply the phase shift by multiplication
        signal.data = signal.data * phase_factor

        # Ensure the data type is preserved
        signal.data = signal.data.astype(torchsig_complex_data_type)

        self.update(signal)
        return signal


class TargetSNR(DatasetTransform):
    """
    Sets target signal-to-noise ratio by adding appropriate white Gaussian noise.

    Simulates controlled noise conditions for testing algorithm performance under various SNR scenarios.

    :param target_snr_range: Target SNR range in dB as (min, max) tuple
    :type target_snr_range: Tuple[float, float]
    :param linear: If True, SNR values are in linear scale instead of dB
    :type linear: bool
    :param debug: If True, prints debugging information about achieved SNR
    :type debug: bool
    :param kwargs: Additional arguments passed to DatasetTransform parent class
    """

    def __init__(
        self,
        target_snr_range: Tuple[float, float] = (0, 30),
        linear: bool = False,
        debug: bool = False,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.linear = linear
        self.debug = debug
        self.min_snr = target_snr_range[0]
        self.max_snr = target_snr_range[1]

    def __call__(self, signal: DatasetSignal) -> DatasetSignal:
        """
        Adds white Gaussian noise to achieve the specified target SNR.

        :param signal: Input RF signal to add noise to
        :type signal: DatasetSignal
        :returns: Signal with added noise to achieve target SNR
        :rtype: DatasetSignal
        """
        # Get target SNR from distribution
        target_snr_db = np.random.uniform(self.min_snr, self.max_snr)

        # Convert from linear if needed
        if self.linear:
            target_snr_db = 10 * np.log10(target_snr_db)

        # Store original signal for SNR verification
        original_signal = signal.data.copy()

        # Calculate current signal power in dB
        signal_power = np.mean(np.abs(signal.data)**2)
        signal_power_db = 10 * np.log10(signal_power)

        # Calculate required noise power to achieve target SNR
        noise_power_db = signal_power_db - target_snr_db

        # Add noise using torchsig function
        signal.data = torchsig_F.awgn(
            signal.data,
            noise_power_db=noise_power_db,
        )

        # Calculate actual SNR achieved
        if self.debug:
            # Calculate added noise by subtracting original signal
            noise = signal.data - original_signal
            noise_power = np.mean(np.abs(noise)**2)

            # Calculate actual SNR in dB
            actual_snr_db = 10 * \
                np.log10(signal_power /
                         noise_power) if noise_power > 0 else float('inf')

            # Print debugging information
            print(
                f"TargetSNR: Target={target_snr_db:.2f}dB, Actual={actual_snr_db:.2f}dB, Difference={actual_snr_db-target_snr_db:.2f}dB")

        # Ensure data type is preserved
        signal.data = signal.data.astype(torchsig_complex_data_type)

        # Update any relevant signal metadata
        self.update(signal)
        return signal
