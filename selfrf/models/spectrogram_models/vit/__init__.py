"""
Vision Transformer (ViT) model builder interface for spectrogram-based deep learning in selfRF.

This module exposes the ViT model construction function adapted for spectrogram inputs, providing a convenient import path for initializing Vision Transformer architectures within the selfRF ecosystem. Its main responsibility is to make the ViT builder available for use in training, evaluation, and feature extraction workflows involving RF spectrogram data. Typical use-cases include selecting ViT as a backbone for self-supervised or supervised learning tasks, or as a feature extractor in downstream applications. The module integrates with the broader spectrogram_models package to ensure consistent model instantiation and configuration across the library.
"""

from .vit import build_vit2d

__all__ = [
    "build_vit2d",
]
