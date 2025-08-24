"""
This subpackage provides spectrogram-based neural network architectures for selfRF, including ResNet and Vision Transformer (ViT) variants adapted for RF spectrogram data. It offers unified model builders for use in training, fine-tuning, and evaluation workflows.
"""

from .resnet import build_resnet2d
from .vit import build_vit2d

__all__ = [
    "build_resnet2d",
    "build_vit2d",
]
