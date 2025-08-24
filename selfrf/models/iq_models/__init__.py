"""
This subpackage provides neural network architectures for IQ (in-phase and quadrature) RF data, including 1D ResNet and XCiT variants. It offers unified model builders for training, evaluation, and transfer learning on RF time-series signals.
"""

from .resnet import build_resnet1d
