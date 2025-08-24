"""
This subpackage provides a collection of auxiliary and legacy data transformation utilities for use throughout the selfRF library. Its primary role is to organize and expose additional preprocessing, augmentation, and target transformation routines that complement the core and SSL-specific transforms, supporting both IQ and spectrogram data modalities. The main responsibilities include enabling advanced or specialized data manipulation, facilitating compatibility with legacy TorchSig pipelines, and supporting custom target encoding strategies for supervised and self-supervised learning. Typical use cases involve selecting extra transforms for use in model training, evaluation, or inference, or customizing preprocessing for new datasets or experimental workflows. This subpackage ensures that all auxiliary transformation components are logically grouped, easily accessible, and can be seamlessly integrated with other modules in the selfRF ecosystem, including dataset factories, collate functions, and model utilities.
"""

from .transforms import *
from .target_transforms import *
