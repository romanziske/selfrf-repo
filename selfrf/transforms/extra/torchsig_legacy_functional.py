"""
Legacy functional transforms preserved from TorchSig v0.6.0 for backward compatibility and specialized operations.

This module contains functional implementations of signal processing transforms that were available in earlier versions of TorchSig but may have been deprecated or modified in newer releases. It provides essential low-level operations for RF signal manipulation including noise addition, frequency shifting, time-domain operations, and spectrogram processing that are required by various parts of the selfRF library. The functions serve as building blocks for more complex transform pipelines and maintain compatibility with existing code that depends on specific behavioral characteristics of the legacy implementations. These utilities are particularly important for research reproducibility where exact algorithmic consistency with previous work is required. The module integrates with the broader selfRF ecosystem by providing the computational primitives used in data augmentation, preprocessing, and signal analysis workflows throughout the library.
"""
from typing import Callable, List, Optional, Tuple, Union
from torchsig.utils.dsp import low_pass
from numba import njit
from scipy import signal as sp
from functools import partial
import numpy as np
import pywt
import os
import cv2

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


__all__ = [
    "FloatParameter",
    "IntParameter",
    "NumericParameter",
    "uniform_discrete_distribution",
    "uniform_continuous_distribution",
    "to_distribution",
    "resample",
    "make_sinc_filter",
    "awgn",
    "impulsive_interference",
    "interleave_complex",
    "real",
    "imag",
    "complex_magnitude",
    "wrapped_phase",
    "discrete_fourier_transform",
    "continuous_wavelet_transform",
    "time_shift",
    "time_crop",
    "freq_shift",
    "freq_shift_avoid_aliasing",
    "_fractional_shift_helper",
    "fractional_shift",
    "amplitude_reversal",
    "amplitude_scale",
    "roll_off",
    "clip",
    "random_convolve",
    "drop_spec_samples",
    "spec_patch_shuffle",
    "spec_translate",
    "spectrogram_image",
]

FloatParameter = Union[Callable[[int], float],
                       float, Tuple[float, float], List]
IntParameter = Union[Callable[[int], int], int, Tuple[int, int], List]
NumericParameter = Union[FloatParameter, IntParameter]


def uniform_discrete_distribution(
    choices: List, random_generator: Optional[np.random.Generator] = None
):
    """
    Creates a uniform discrete distribution sampler from a list of discrete choices.

    :param choices: List of discrete variables to sample from uniformly
    :type choices: List
    :param random_generator: Random generator instance for sampling
    :type random_generator: Optional[np.random.Generator]
    :returns: Partial function that samples uniformly from choices
    :rtype: Callable
    """
    random_generator = random_generator if random_generator else np.random.default_rng()
    return partial(random_generator.choice, choices)


def uniform_continuous_distribution(
    lower: Union[int, float],
    upper: Union[int, float],
    random_generator: Optional[np.random.Generator] = None,
):
    """
    Creates a uniform continuous distribution sampler between lower and upper bounds.

    :param lower: Lowest value possible in the distribution
    :type lower: Union[int, float]
    :param upper: Highest value possible in the distribution
    :type upper: Union[int, float]
    :param random_generator: Random generator instance for sampling
    :type random_generator: Optional[np.random.Generator]
    :returns: Partial function that samples uniformly between bounds
    :rtype: Callable
    """
    random_generator = random_generator if random_generator else np.random.default_rng()
    return partial(random_generator.uniform, lower, upper)


def to_distribution(
    param: Union[
        int,
        float,
        str,
        Callable,
        List[int],
        List[float],
        List[str],
        Tuple[int, int],
        Tuple[float, float],
    ],
    random_generator: Optional[np.random.Generator] = None,
):
    """
    Converts various parameter types into numpy random distribution functions.

    Handles conversion of fixed values, ranges, lists, and callables into consistent distribution interfaces.

    :param param: Parameter specification for the random distribution
    :type param: Union[int, float, str, Callable, List, Tuple]
    :param random_generator: Random generator instance to use
    :type random_generator: Optional[np.random.Generator]
    :returns: Distribution function that can be called to sample values
    :rtype: Callable
    """
    random_generator = random_generator if random_generator else np.random.default_rng()
    if isinstance(param, Callable):  # type: ignore
        return param

    if isinstance(param, list):
        #######################################################################
        # [BUG ALERT]: Nested tuples within lists does not function as desired.
        # Below will instantiate a random distribution from the list; however,
        # each call will only come from the previously randomized selection,
        # but the desired behavior would be for this to randomly select each
        # region at call time. Commenting out for now, but should revisit in
        # the future to add back the functionality.
        #######################################################################
        # if isinstance(param[0], tuple):
        #     tuple_from_list = param[random_generator.randint(len(param))]
        #     return uniform_continuous_distribution(
        #         tuple_from_list[0],
        #         tuple_from_list[1],
        #         random_generator,
        #     )
        return uniform_discrete_distribution(param, random_generator)

    if isinstance(param, tuple):
        return uniform_continuous_distribution(
            param[0],
            param[1],
            random_generator,
        )

    if isinstance(param, int) or isinstance(param, float):
        return uniform_discrete_distribution([param], random_generator)

    return param


@njit(cache=False)
def make_sinc_filter(beta, tap_cnt, sps, offset=0):
    """
    Creates sinc filter taps for interpolation and filtering operations.

    :param beta: Filter bandwidth parameter controlling cutoff frequency
    :type beta: float
    :param tap_cnt: Number of filter taps to generate
    :type tap_cnt: int
    :param sps: Samples per symbol for timing calculations
    :type sps: float
    :param offset: Phase offset for fractional delay applications
    :type offset: float
    :returns: Normalized sinc filter coefficients
    :rtype: np.ndarray
    """
    ntap_cnt = tap_cnt + ((tap_cnt + 1) % 2)
    t_index = np.arange(-(ntap_cnt - 1) // 2,
                        (ntap_cnt - 1) // 2 + 1) / np.double(sps)

    taps = np.sinc(beta * t_index + offset)
    taps /= np.sum(taps)

    return taps[:tap_cnt]


def awgn(tensor: np.ndarray, noise_power_db: float) -> np.ndarray:
    """
    Adds zero-mean complex additive white Gaussian noise to the input tensor.

    :param tensor: Input signal tensor to add noise to
    :type tensor: np.ndarray
    :param noise_power_db: Noise power in dB defined as 10*log10(E[|n|^2])
    :type noise_power_db: float
    :returns: Signal tensor with added AWGN
    :rtype: np.ndarray
    """
    real_noise = np.random.randn(*tensor.shape)
    imag_noise = np.random.randn(*tensor.shape)
    return tensor + (10.0 ** (noise_power_db / 20.0)) * (real_noise + 1j * imag_noise) / np.sqrt(2)


@njit(cache=False)
def impulsive_interference(
    tensor: np.ndarray,
    amp: float,
    per_offset: float,
) -> np.ndarray:
    """
    Applies impulsive interference to simulate pulsed jamming or interference signals.

    :param tensor: Input signal tensor to add interference to
    :type tensor: np.ndarray
    :param amp: Maximum vector magnitude of the complex interferer signal
    :type amp: float
    :param per_offset: Interferer offset as fraction of tensor length
    :type per_offset: float
    :returns: Signal tensor with added impulsive interference
    :rtype: np.ndarray
    """
    beta = 0.3
    num_samps = len(tensor)
    sinc_pulse = make_sinc_filter(beta, num_samps, 0.1, 0)
    imp = amp * np.roll(sinc_pulse / np.max(sinc_pulse),
                        int(per_offset * num_samps))
    rand_phase = np.random.uniform(0, 2 * np.pi)
    imp = np.exp(1j * rand_phase) * imp
    output: np.ndarray = tensor + imp
    return output


def interleave_complex(tensor: np.ndarray) -> np.ndarray:
    """
    Converts complex IQ vectors to real interleaved format alternating I and Q samples.

    :param tensor: Complex-valued input tensor
    :type tensor: np.ndarray
    :returns: Real-valued tensor with interleaved I and Q components
    :rtype: np.ndarray
    """
    new_tensor = np.empty((tensor.shape[0] * 2,))
    new_tensor[::2] = np.real(tensor)
    new_tensor[1::2] = np.imag(tensor)
    return new_tensor


def real(tensor: np.ndarray) -> np.ndarray:
    """
    Extracts the real (in-phase) component from complex IQ data.

    :param tensor: Complex-valued input tensor
    :type tensor: np.ndarray
    :returns: Real component of the input tensor
    :rtype: np.ndarray
    """
    return np.real(tensor)


def imag(tensor: np.ndarray) -> np.ndarray:
    """
    Extracts the imaginary (quadrature) component from complex IQ data.

    :param tensor: Complex-valued input tensor
    :type tensor: np.ndarray
    :returns: Imaginary component of the input tensor
    :rtype: np.ndarray
    """
    return np.imag(tensor)


def complex_magnitude(tensor: np.ndarray) -> np.ndarray:
    """
    Computes the magnitude (absolute value) of complex IQ samples.

    :param tensor: Complex-valued input tensor
    :type tensor: np.ndarray
    :returns: Magnitude values of the complex input
    :rtype: np.ndarray
    """
    return np.abs(tensor)


def wrapped_phase(tensor: np.ndarray) -> np.ndarray:
    """
    Computes the wrapped phase angle of complex IQ samples in radians.

    :param tensor: Complex-valued input tensor
    :type tensor: np.ndarray
    :returns: Phase angles wrapped to [-π, π] range
    :rtype: np.ndarray
    """
    return np.angle(tensor)


def discrete_fourier_transform(tensor: np.ndarray) -> np.ndarray:
    """
    Computes the orthonormalized discrete Fourier transform of the input signal.

    :param tensor: Input signal tensor for frequency domain transformation
    :type tensor: np.ndarray
    :returns: DFT coefficients with 1/sqrt(n) normalization
    :rtype: np.ndarray
    """
    return np.fft.fft(tensor, norm="ortho")


def continuous_wavelet_transform(
    tensor: np.ndarray, wavelet: str, nscales: int, sample_rate: float
) -> np.ndarray:
    """
    Computes continuous wavelet transform to generate time-frequency scalogram representation.

    :param tensor: Input signal tensor for wavelet analysis
    :type tensor: np.ndarray
    :param wavelet: Name of the mother wavelet function
    :type wavelet: str
    :param nscales: Number of scales for the scalogram
    :type nscales: int
    :param sample_rate: Sampling rate of the input signal
    :type sample_rate: float
    :returns: Magnitude scalogram of the input signal
    :rtype: np.ndarray
    """
    scales = np.arange(1, nscales)
    cwtmatr, _ = pywt.cwt(
        tensor, scales=scales, wavelet=wavelet, sampling_period=1.0 / sample_rate
    )

    # if the dtype is complex then return the magnitude
    if np.iscomplexobj(cwtmatr):
        cwtmatr = abs(cwtmatr)

    return cwtmatr


def time_shift(tensor: np.ndarray, t_shift: int) -> np.ndarray:
    """
    Shifts signal in time domain by specified number of samples with zero-padding.

    :param tensor: Input signal tensor to shift in time
    :type tensor: np.ndarray
    :param t_shift: Number of samples to shift (positive=right, negative=left)
    :type t_shift: int
    :returns: Time-shifted signal with original dimensions
    :rtype: np.ndarray
    """
    # Valid Range Error Checking
    if np.max(np.abs(t_shift)) >= tensor.shape[0]:
        return np.zeros_like(tensor, dtype=np.complex64)

    # This overwrites tensor as side effect, modifies inplace
    if t_shift > 0:
        tmp = tensor[:-t_shift]  # I'm sure there's a more compact way.
        tensor = np.pad(tmp, (t_shift, 0), "constant", constant_values=0 + 0j)
    elif t_shift < 0:
        tmp = tensor[-t_shift:]  # I'm sure there's a more compact way.
        tensor = np.pad(tmp, (0, -t_shift), "constant", constant_values=0 + 0j)
    return tensor


def time_crop(tensor: np.ndarray, start: int, length: int) -> np.ndarray:
    """
    Crops signal tensor in time dimension from specified start index for given length.

    :param tensor: Input signal tensor to crop
    :type tensor: np.ndarray
    :param start: Starting index for cropping operation
    :type start: int
    :param length: Number of samples to include in cropped output
    :type length: int
    :returns: Cropped signal tensor
    :rtype: np.ndarray
    :raises ValueError: If length is negative or start is negative
    """
    # Type and Size checking
    if length < 0:
        raise ValueError("Length must be greater than 0")

    if np.any(start < 0):
        raise ValueError("Start must be greater than 0")

    if np.max(start) >= tensor.shape[0] or length == 0:
        return np.empty(shape=(1, 1))

    return tensor[start: start + length]


def freq_shift(tensor: np.ndarray, f_shift: float) -> np.ndarray:
    """
    Shifts signal in frequency domain by mixing with complex exponential.

    :param tensor: Input signal tensor to frequency shift
    :type tensor: np.ndarray
    :param f_shift: Frequency shift relative to sample rate in range [-0.5, 0.5]
    :type f_shift: float
    :returns: Frequency-shifted signal tensor
    :rtype: np.ndarray
    """
    sinusoid = np.exp(
        2j * np.pi * f_shift * np.arange(tensor.shape[0], dtype=np.float64)
    )
    mult = np.multiply(tensor, np.asarray(sinusoid))
    return mult


def freq_shift_avoid_aliasing(
    tensor: np.ndarray,
    f_shift: float,
) -> np.ndarray:
    """
    Performs frequency shifting with upsampling and filtering to prevent aliasing artifacts.

    :param tensor: Input signal tensor to frequency shift
    :type tensor: np.ndarray
    :param f_shift: Frequency shift relative to sample rate in range [-0.5, 0.5]
    :type f_shift: float
    :returns: Anti-aliased frequency-shifted signal tensor
    :rtype: np.ndarray
    """
    # Match output size to input
    num_iq_samples = tensor.shape[0]

    # Interpolate up to avoid frequency wrap around during shift
    up = 2
    down = 1
    tensor = sp.resample_poly(tensor, up, down)

    taps = low_pass(cutoff=1 / 4, transition_bandwidth=(0.5 - 1 / 4) / 4)
    convolve_out = sp.convolve(tensor, taps, mode="full")
    lidx = (len(convolve_out) - len(tensor)) // 2
    ridx = lidx + len(tensor)
    tensor = convolve_out[lidx:ridx]

    # Freq shift to desired center freq
    time_vector = np.arange(tensor.shape[0], dtype=np.float64)
    tensor = tensor * np.exp(2j * np.pi * f_shift / up * time_vector)

    # Filter to remove out-of-band regions
    taps = low_pass(cutoff=1 / 4, transition_bandwidth=(0.5 - 1 / 4) / 4)
    convolve_out = sp.convolve(tensor, taps, mode="full")
    lidx = (len(convolve_out) - int(num_iq_samples * up)) // 2
    ridx = lidx + len(tensor)
    # prune to be correct size out of filter
    tensor = convolve_out[: int(num_iq_samples * up)]

    # Decimate back down to correct sample rate
    tensor = sp.resample_poly(tensor, down, up)

    return tensor[:num_iq_samples]


@njit(cache=False)
def _fractional_shift_helper(taps: np.ndarray, raw_iq: np.ndarray, stride: int, offset: int) -> np.ndarray:
    """
    Efficient polyphase implementation for fractional sample delay using decimated filter taps.

    :param taps: Filter coefficients for polyphase interpolation
    :type taps: np.ndarray
    :param raw_iq: Input IQ signal data
    :type raw_iq: np.ndarray
    :param stride: Interpolation stride for polyphase filtering
    :type stride: int
    :param offset: Phase offset within the polyphase structure
    :type offset: int
    :returns: Fractionally delayed signal samples
    :rtype: np.ndarray
    """
    # We purposely do not calculate values within the group delay.
    group_delay = ((taps.shape[0] - 1) // 2 - (stride - 1)) // stride + 1
    if offset < 0:
        offset += stride
        group_delay -= 1

    # Decimate the taps.
    taps = taps[offset::stride]

    # Determine output size
    num_taps = taps.shape[0]
    num_raw_iq = raw_iq.shape[0]
    output = np.zeros(
        ((num_taps + num_raw_iq - 1 - group_delay),), dtype=np.complex128)

    # This is a just convolution of taps and raw_iq
    for o_idx in range(output.shape[0]):
        idx_mn = o_idx - (num_raw_iq - 1) if o_idx >= num_raw_iq - 1 else 0
        idx_mx = o_idx if o_idx < num_taps - 1 else num_taps - 1
        for f_idx in range(idx_mn, idx_mx):
            output[o_idx - group_delay] += taps[f_idx] * raw_iq[o_idx - f_idx]
    return output


def fractional_shift(
    tensor: np.ndarray, taps: np.ndarray, stride: int, delay: float
) -> np.ndarray:
    """
    Applies fractional sample delay using polyphase interpolation filters.

    :param tensor: Input signal tensor to apply fractional delay
    :type tensor: np.ndarray
    :param taps: Polyphase filter coefficients
    :type taps: np.ndarray
    :param stride: Interpolation rate for the internal filter
    :type stride: int
    :param delay: Fractional delay in samples within range [-1, 1]
    :type delay: float
    :returns: Signal with applied fractional delay
    :rtype: np.ndarray
    """
    real_part = _fractional_shift_helper(
        taps, tensor.real, stride, int(stride * float(delay)))
    imag_part = _fractional_shift_helper(
        taps, tensor.imag, stride, int(stride * float(delay)))
    tensor = real_part[: tensor.shape[0]] + 1j * imag_part[: tensor.shape[0]]
    zero_idx = -1 if delay < 0 else 0  # do not extrapolate, zero-pad.
    tensor[zero_idx] = 0
    return tensor


def amplitude_reversal(tensor: np.ndarray) -> np.ndarray:
    """
    Applies amplitude reversal by multiplying signal by -1.

    :param tensor: Input signal tensor to reverse amplitude
    :type tensor: np.ndarray
    :returns: Signal with reversed amplitude
    :rtype: np.ndarray
    """
    return tensor * -1


def amplitude_scale(tensor: np.ndarray, scale: float) -> np.ndarray:
    """
    Scales signal amplitude by specified multiplicative factor.

    :param tensor: Input signal tensor to scale
    :type tensor: np.ndarray
    :param scale: Multiplicative scaling factor
    :type scale: float
    :returns: Amplitude-scaled signal tensor
    :rtype: np.ndarray
    """
    return tensor * scale


def roll_off(
    tensor: np.ndarray,
    lowercutfreq: float,
    uppercutfreq: float,
    num_taps: int,
) -> np.ndarray:
    """
    Applies bandpass filtering with configurable roll-off characteristics to simulate front-end filters.

    :param tensor: Input signal tensor to filter
    :type tensor: np.ndarray
    :param lowercutfreq: Lower bandwidth cutoff for linear roll-off
    :type lowercutfreq: float
    :param uppercutfreq: Upper bandwidth cutoff for linear roll-off
    :type uppercutfreq: float
    :param num_taps: Order of the FIR filter to apply
    :type num_taps: int
    :returns: Filtered signal with bandwidth roll-off applied
    :rtype: np.ndarray
    """
    if (lowercutfreq == 0) & (uppercutfreq == 1):
        return tensor

    elif uppercutfreq == 1:
        if num_taps % 2 == 0:
            num_taps += 1
    bandwidth = uppercutfreq - lowercutfreq
    center_freq = lowercutfreq - 0.5 + bandwidth / 2
    sinusoid = np.exp(2j * np.pi * center_freq *
                      np.linspace(0, num_taps - 1, num_taps))
    taps = sp.firwin(
        num_taps,
        bandwidth,
        width=bandwidth * 0.02,
        window=sp.get_window("blackman", num_taps),
        scale=True,
    )
    taps = taps * sinusoid
    convolve_out = sp.convolve(tensor, taps, mode="full")
    lidx = (len(convolve_out) - len(tensor)) // 2
    ridx = lidx + len(tensor)
    return convolve_out[lidx:ridx]


def clip(tensor: np.ndarray, clip_percentage: float) -> np.ndarray:
    """
    Clips signal values above and below specified percentage of maximum and minimum values.

    :param tensor: Input signal tensor to clip
    :type tensor: np.ndarray
    :param clip_percentage: Percentage of max/min values to use as clipping thresholds
    :type clip_percentage: float
    :returns: Clipped signal tensor
    :rtype: np.ndarray
    """
    real_tensor = tensor.real
    max_val = np.max(real_tensor) * clip_percentage
    min_val = np.min(real_tensor) * clip_percentage
    real_tensor[real_tensor > max_val] = max_val
    real_tensor[real_tensor < min_val] = min_val

    imag_tensor = tensor.imag
    max_val = np.max(imag_tensor) * clip_percentage
    min_val = np.min(imag_tensor) * clip_percentage
    imag_tensor[imag_tensor > max_val] = max_val
    imag_tensor[imag_tensor < min_val] = min_val

    new_tensor = real_tensor + 1j * imag_tensor
    return new_tensor


def random_convolve(
    tensor: np.ndarray,
    num_taps: int,
    alpha: float,
) -> np.ndarray:
    """
    Convolves signal with random complex filter and blends with original using alpha weighting.

    :param tensor: Input signal tensor to process
    :type tensor: np.ndarray
    :param num_taps: Number of taps in the random filter
    :type num_taps: int
    :param alpha: Weighting factor for blending original and filtered signals
    :type alpha: float
    :returns: Weighted combination of original and randomly filtered signal
    :rtype: np.ndarray
    """
    filter_taps = np.random.rand(num_taps) + 1j * np.random.rand(num_taps)
    convolve_out = sp.convolve(tensor, filter_taps, mode="full")
    lidx = (len(convolve_out) - len(tensor)) // 2
    ridx = lidx + len(tensor)
    return (1 - alpha) * tensor + alpha * convolve_out[lidx:ridx]


def drop_spec_samples(
    tensor: np.ndarray,
    drop_starts: np.ndarray,
    drop_sizes: np.ndarray,
    fill: str,
) -> np.ndarray:
    """
    Drops samples at specified locations in spectrogram and fills with chosen replacement strategy.

    :param tensor: Input spectrogram tensor to modify
    :type tensor: np.ndarray
    :param drop_starts: Starting indices for drop regions
    :type drop_starts: np.ndarray
    :param drop_sizes: Duration of each drop instance
    :type drop_sizes: np.ndarray
    :param fill: Fill strategy for dropped samples (ffill, bfill, mean, zero, min, max, low, ones)
    :type fill: str
    :returns: Modified spectrogram with dropped and filled regions
    :rtype: np.ndarray
    :raises ValueError: If fill strategy is not recognized
    """
    flat_spec = tensor.reshape(
        tensor.shape[0], tensor.shape[1] * tensor.shape[2])
    for idx, drop_start in enumerate(drop_starts):
        if fill == "ffill":
            drop_region_real = np.ones(
                drop_sizes[idx]) * flat_spec[0, drop_start - 1]
            drop_region_complex = (
                np.ones(drop_sizes[idx]) * flat_spec[1, drop_start - 1]
            )
            flat_spec[0, drop_start: drop_start +
                      drop_sizes[idx]] = drop_region_real
            flat_spec[
                1, drop_start: drop_start + drop_sizes[idx]
            ] = drop_region_complex
        elif fill == "bfill":
            drop_region_real = (
                np.ones(drop_sizes[idx]) * flat_spec[0,
                                                     drop_start + drop_sizes[idx]]
            )
            drop_region_complex = (
                np.ones(drop_sizes[idx]) * flat_spec[1,
                                                     drop_start + drop_sizes[idx]]
            )
            flat_spec[0, drop_start: drop_start +
                      drop_sizes[idx]] = drop_region_real
            flat_spec[
                1, drop_start: drop_start + drop_sizes[idx]
            ] = drop_region_complex
        elif fill == "mean":
            drop_region_real = np.ones(drop_sizes[idx]) * np.mean(flat_spec[0])
            drop_region_complex = np.ones(
                drop_sizes[idx]) * np.mean(flat_spec[1])
            flat_spec[0, drop_start: drop_start +
                      drop_sizes[idx]] = drop_region_real
            flat_spec[
                1, drop_start: drop_start + drop_sizes[idx]
            ] = drop_region_complex
        elif fill == "zero":
            drop_region = np.zeros(drop_sizes[idx])
            flat_spec[:, drop_start: drop_start +
                      drop_sizes[idx]] = drop_region
        elif fill == "min":
            drop_region_real = np.ones(
                drop_sizes[idx]) * np.min(np.abs(flat_spec[0]))
            drop_region_complex = np.ones(drop_sizes[idx]) * np.min(
                np.abs(flat_spec[1])
            )
            flat_spec[0, drop_start: drop_start +
                      drop_sizes[idx]] = drop_region_real
            flat_spec[
                1, drop_start: drop_start + drop_sizes[idx]
            ] = drop_region_complex
        elif fill == "max":
            drop_region_real = np.ones(
                drop_sizes[idx]) * np.max(np.abs(flat_spec[0]))
            drop_region_complex = np.ones(drop_sizes[idx]) * np.max(
                np.abs(flat_spec[1])
            )
            flat_spec[0, drop_start: drop_start +
                      drop_sizes[idx]] = drop_region_real
            flat_spec[
                1, drop_start: drop_start + drop_sizes[idx]
            ] = drop_region_complex
        elif fill == "low":
            drop_region = np.ones(drop_sizes[idx]) * 1e-3
            flat_spec[:, drop_start: drop_start +
                      drop_sizes[idx]] = drop_region
        elif fill == "ones":
            drop_region = np.ones(drop_sizes[idx])
            flat_spec[:, drop_start: drop_start +
                      drop_sizes[idx]] = drop_region
        else:
            raise ValueError(
                "fill expects ffill, bfill, mean, zero, min, max, low, ones. Found {}".format(
                    fill
                )
            )
    new_tensor = flat_spec.reshape(
        tensor.shape[0], tensor.shape[1], tensor.shape[2])
    return new_tensor


def spec_patch_shuffle(
    tensor: np.ndarray,
    patch_size: int,
    shuffle_ratio: float,
) -> np.ndarray:
    """
    Randomly shuffles patches within spectrograms for data augmentation.

    :param tensor: Input spectrogram tensor to shuffle
    :type tensor: np.ndarray
    :param patch_size: Size of each square patch to shuffle
    :type patch_size: int
    :param shuffle_ratio: Fraction of patches to randomly shuffle
    :type shuffle_ratio: float
    :returns: Spectrogram with shuffled patches
    :rtype: np.ndarray
    """
    channels, height, width = tensor.shape
    num_freq_patches = int(height / patch_size)
    num_time_patches = int(width / patch_size)
    num_patches = int(num_freq_patches * num_time_patches)
    num_to_shuffle = int(num_patches * shuffle_ratio)
    patches_to_shuffle = np.random.choice(
        num_patches,
        replace=False,
        size=num_to_shuffle,
    )

    for patch_idx in patches_to_shuffle:
        freq_idx = np.floor(patch_idx / num_freq_patches)
        time_idx = patch_idx % num_time_patches
        patch = tensor[
            :,
            int(freq_idx * patch_size): int(freq_idx * patch_size + patch_size),
            int(time_idx * patch_size): int(time_idx * patch_size + patch_size),
        ]
        patch = patch.reshape(int(2 * patch_size * patch_size))
        np.random.shuffle(patch)
        patch = patch.reshape(2, int(patch_size), int(patch_size))
        tensor[
            :,
            int(freq_idx * patch_size): int(freq_idx * patch_size + patch_size),
            int(time_idx * patch_size): int(time_idx * patch_size + patch_size),
        ] = patch
    return tensor


def spec_translate(
    tensor: np.ndarray,
    time_shift: int,
    freq_shift: int,
) -> np.ndarray:
    """
    Applies time and frequency translation to spectrogram with background noise filling.

    :param tensor: Input spectrogram tensor to translate
    :type tensor: np.ndarray
    :param time_shift: Number of bins to shift in time dimension
    :type time_shift: int
    :param freq_shift: Number of bins to shift in frequency dimension
    :type freq_shift: int
    :returns: Translated spectrogram with noise-filled empty regions
    :rtype: np.ndarray
    """
    # Pre-fill the data with background noise
    new_tensor = np.random.rand(*tensor.shape) * \
        np.percentile(np.abs(tensor), 50)

    # Apply translation
    channels, height, width = tensor.shape
    if time_shift >= 0 and freq_shift >= 0:
        valid_dur = width - time_shift
        valid_bw = height - freq_shift
        new_tensor[:, freq_shift:,
                   time_shift:] = tensor[:, :valid_bw, :valid_dur]
    elif time_shift < 0 and freq_shift >= 0:
        valid_dur = width + time_shift
        valid_bw = height - freq_shift
        new_tensor[:, freq_shift:, :valid_dur] = tensor[:,
                                                        :valid_bw, -time_shift:]
    elif time_shift >= 0 and freq_shift < 0:
        valid_dur = width - time_shift
        valid_bw = height + freq_shift
        new_tensor[:, :valid_bw, time_shift:] = tensor[:, -
                                                       freq_shift:, :valid_dur]
    elif time_shift < 0 and freq_shift < 0:
        valid_dur = width + time_shift
        valid_bw = height + freq_shift
        new_tensor[:, :valid_bw, :valid_dur] = tensor[:, -
                                                      freq_shift:, -time_shift:]

    return new_tensor


def spectrogram_image(
    tensor: np.ndarray,
    black_hot: bool = True
) -> np.ndarray:
    """
    Converts power spectrogram to normalized grayscale image with optional color inversion.

    :param tensor: Input power spectrogram tensor
    :type tensor: np.ndarray
    :param black_hot: Whether to use black-hot colormap (high values appear dark)
    :type black_hot: bool
    :returns: Normalized grayscale image representation of spectrogram
    :rtype: np.ndarray
    """
    spec = 10 * np.log10(tensor+np.finfo(np.float32).tiny)
    img = cv2.normalize(spec, None, 0, 255, cv2.NORM_MINMAX)
    img = img.astype(np.uint8)

    if black_hot:
        img = cv2.bitwise_not(img, img)

    return img
