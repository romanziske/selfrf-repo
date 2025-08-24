"""
Vision Transformer (ViT) model construction utilities for spectrogram-based RF signal processing in selfRF.

This module provides flexible builders for Vision Transformer architectures adapted to handle spectrogram inputs, supporting both classification and feature extraction use-cases. Its main responsibilities include configuring input channels, model variants, output feature dimensions, and pretrained weight management, as well as supporting feature-only extraction for representation learning workflows. Typical use-cases involve initializing ViT backbones for self-supervised pretraining, fine-tuning on classification tasks, or extracting patch embeddings for downstream analysis. The module integrates with the selfRF spectrogram_models package and leverages the timm library for robust model instantiation and compatibility with pretrained weights.
"""

import timm
import logging

logger = logging.getLogger(__name__)

__all__ = ["build_vit2d"]


def build_vit2d(
    input_channels: int,
    n_features: int = 1000,
    version: str = "vit_base_patch16_224.mae",
    pretrained: bool = True,
    **kwargs
):
    """
    Constructs and returns a Vision Transformer (ViT) model for spectrogram inputs.

    Supports both full classification models and feature-only variants for representation learning or transfer learning.

    :param input_channels: Number of input channels, e.g., 1 for magnitude or 2 for complex spectrograms
    :type input_channels: int
    :param n_features: Number of output features for the classification head
    :type n_features: int
    :param version: Specifies the ViT model name from timm (e.g., 'vit_base_patch16_224')
    :type version: str
    :param pretrained: If True, loads pretrained weights from timm (usually ImageNet)
    :type pretrained: bool
    :param kwargs: Additional keyword arguments passed to timm.create_model
    :type kwargs: dict
    :returns: Configured ViT model instance for spectrogram data
    :rtype: torch.nn.Module
    :raises UserWarning: If pretrained weights are loaded with input_channels != 3
    """
    if pretrained and input_channels != 3:
        logger.warning(
            f"Loading pretrained ViT '{version}' but input_channels is {input_channels} (expected 3). "
            "Pretrained weights for the patch embedding layer will likely not be loaded correctly."
        )

    model = timm.create_model(
        version,
        pretrained=pretrained,
        in_chans=input_channels,
        img_size=224,  # Default input size for ViT
        **kwargs
    )

    return model
