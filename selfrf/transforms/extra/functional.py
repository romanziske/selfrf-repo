"""
Low-level functional implementations for RF signal transformations and spectrogram processing.

This module provides the core functional implementations used by higher-level transform classes throughout the selfRF library. It contains stateless functions for common RF signal processing operations including amplitude scaling and spectrogram-to-image conversion with various normalization schemes. These functions are designed to work with NumPy arrays and provide the computational backbone for data augmentation pipelines in both self-supervised learning and supervised fine-tuning workflows. The module serves as a bridge between raw signal processing operations and the transform abstractions used in machine learning pipelines. It integrates with other transform modules by providing the underlying mathematical operations that are composed into more complex augmentation strategies.
"""
import numpy as np


def amplitude_scale(tensor: np.ndarray, scale: float) -> np.ndarray:
    """
    Applies uniform amplitude scaling to input tensor for signal magnitude adjustment.

    Multiplies all tensor values by the scaling factor to simulate gain variations or signal strength changes.

    :param tensor: Input tensor of shape (batch_size, vector_length, ...)
    :type tensor: np.ndarray
    :param scale: Multiplicative scaling factor to apply to all tensor elements
    :type scale: float
    :returns: Transformed tensor with scaled amplitude values
    :rtype: np.ndarray
    """
    return tensor * scale
