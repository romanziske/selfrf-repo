"""
This subpackage provides all specialized data augmentation and transformation routines for self-supervised learning (SSL) algorithms within the selfRF library. Its primary role is to organize and expose transform classes tailored for SSL methods such as DenseCL, MAE, and VICRegL, supporting both IQ and spectrogram data modalities. The main responsibilities include defining multi-view and algorithm-specific augmentation pipelines, enabling robust and reproducible training for contrastive and reconstruction-based SSL approaches. Typical use cases involve selecting and composing SSL-specific transforms for use in PyTorch dataloaders during pretraining, benchmarking, or transfer learning workflows. This subpackage ensures that all SSL transform components are logically grouped, easily accessible, and seamlessly integrated with the broader selfRF data processing, dataset factory, and model utility ecosystem.
"""

from .densecl_transfrom import DenseCLTransform
from .mae_transfrom import MAETransform
from .vicregl_transform import VICRegLTransform, OptimizedVICRegLTransform

__all__ = [
    "DenseCLTransform",
    "MAETransform",
    "VICRegLTransform",
    "OptimizedVICRegLTransform"
]
