"""
This subpackage provides self-supervised learning (SSL) models for RF spectrograms, including  DenseCL, MAE, and VICRegL. It offers a unified interface for selecting and using SSL algorithms within the selfRF library.
"""

from .densecl import DenseCL
from .mae import MAE
from .vicregl import VICRegL

__all__ = [
    "DenseCL",
    "MAE",
    "VICRegL",
]
