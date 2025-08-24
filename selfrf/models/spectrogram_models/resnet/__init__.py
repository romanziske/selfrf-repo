"""
ResNet model builder interface for spectrogram-based deep learning in selfRF.

This module exposes the ResNet model construction function tailored for spectrogram inputs, providing a convenient import path for initializing 2D ResNet architectures within the selfRF ecosystem. Its main responsibility is to make the ResNet builder available for use in training, evaluation, and feature extraction workflows involving RF spectrogram data. Typical use-cases include selecting ResNet as a backbone for self-supervised or supervised learning tasks, or as a feature extractor in downstream applications. The module integrates with the broader spectrogram_models package to ensure consistent model instantiation and configuration across the library.
"""

from .resnet import build_resnet2d

__all__ = ["build_resnet2d"]
