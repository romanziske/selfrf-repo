"""
ResNet model construction utilities for spectrogram-based RF signal processing in selfRF.

This module provides flexible builders for 2D ResNet architectures adapted to handle spectrogram inputs, supporting both classification and feature extraction use-cases. Its main responsibilities include configuring input channels, output feature dimensions, and model variants, as well as supporting feature-only extraction for representation learning workflows. Typical use-cases involve initializing ResNet backbones for self-supervised pretraining, fine-tuning on classification tasks, or extracting intermediate feature maps for downstream analysis. The module integrates with the selfRF spectrogram_models package and leverages the timm library for robust model instantiation and pretrained weight management.
"""

import timm
from torch import nn

__all__ = ["build_resnet2d"]


def build_resnet2d(
    input_channels: int,
    n_features: int = 2048,
    version: str = "50",
    features_only: bool = False,
):
    """
    Constructs and returns a 2D ResNet model for spectrogram inputs.

    Supports both full classification models and feature-only variants for representation learning or transfer learning.

    :param input_channels: Number of input channels, typically 2 for complex spectrograms
    :type input_channels: int
    :param n_features: Number of output features or classes for the final layer
    :type n_features: int
    :param version: Specifies the ResNet version (e.g., '18', '34', '50')
    :type version: str
    :param features_only: If True, returns model without final pooling/classification layers
    :type features_only: bool

    :returns: Configured ResNet model instance for spectrogram data
    :rtype: torch.nn.Module
    """
    if features_only:
        # Create model without final pool/fc layer to get the last feature map
        model = timm.create_model(
            "resnet" + version,
            pretrained=False,
            in_chans=input_channels,
            features_only=False,  # Create full model initially
            num_classes=0,       # Remove classifier head
            global_pool=''       # Remove global pooling
        )
        return model
    else:
        model = timm.create_model(
            "resnet" + version,
            in_chans=input_channels,
        )
        model.fc = nn.Linear(model.fc.in_features, n_features)
        return model
