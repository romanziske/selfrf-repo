"""
This module provides the construction utility for 1D ResNet architectures adapted for in-phase and quadrature (IQ) signal data within the selfRF library. Its primary role is to enable the use of popular ResNet variants, originally designed for 2D image data, on 1D RF time-series by leveraging conversion routines and the timm model zoo. The main responsibility of this module is to expose a flexible function for building 1D ResNet models with configurable input channels, output features, and architectural variants, supporting both feature extraction and classification use-cases. Typical use cases include initializing backbone networks for self-supervised learning, classification, or transfer learning on IQ data. This module integrates with other model utility modules and the broader selfRF modeling ecosystem by providing a standardized interface for 1D ResNet instantiation.
"""
import timm
from torch.nn import Linear

from torchsig.models.model_utils.model_utils_1d.conversions_to_1d import convert_2d_model_to_1d


__all__ = ["build_resnet1d"]


def build_resnet1d(
    input_channels: int,
    n_features: int = 2048,
    version: str = "18",
    drop_path_rate: float = 0.2,
    drop_rate: float = 0.3,
    features_only=False,
):
    """
    Constructs and returns a 1D version of the ResNet model for IQ signal data.
    This function adapts a 2D ResNet from timm to 1D using a conversion utility and allows customization of input channels, output features, and dropout rates.
    :param input_channels: Number of 1D input channels, typically 2 for IQ data.
    :param n_features: Number of output features or classes for classification.
    :param version: ResNet variant to use, such as '18', '34', or '50'.
    :param drop_path_rate: Drop path rate for stochastic depth regularization.
    :param drop_rate: Dropout rate for training.
    :param features_only: If True, returns the feature extractor without the final classification layer.

    :returns: A torch.nn.Module representing the configured 1D ResNet model.
    """

    mdl = convert_2d_model_to_1d(
        timm.create_model(
            "resnet" + version,
            in_chans=input_channels,
            drop_path_rate=drop_path_rate,
            drop_rate=drop_rate,
            features_only=features_only
        )
    )
    if not features_only:
        mdl.fc = Linear(mdl.fc.in_features, n_features)
    return mdl
