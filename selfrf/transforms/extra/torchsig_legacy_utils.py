"""
Utility functions for legacy TorchSig transform parameter handling and random distribution generation.

This module provides essential utility functions for managing random parameter distributions used throughout the legacy TorchSig transform implementations preserved in selfRF. It handles the conversion of various parameter specifications (fixed values, ranges, lists, callables) into consistent random distribution functions that can be used by transform classes. The utilities are particularly important for maintaining backward compatibility with older TorchSig versions where parameter handling had different conventions and behaviors compared to modern implementations. These functions serve as the foundation for probabilistic data augmentation strategies where transform parameters need to be randomly sampled from specified distributions during training, enabling reproducible yet varied augmentation patterns. The module integrates with the legacy transform classes by providing the core randomization infrastructure that enables reproducible yet varied data augmentation across different training runs, and supports the broader selfRF ecosystem by ensuring consistent parameter handling across all legacy transform implementations.
"""

__all__ = [
    "get_distribution",
    "FloatParameter",
    "IntParameter",
    "NumericParameter",
    "StrParameter",
]


from functools import partial
import numpy as np

# Built-In
from typing import Callable, Tuple, List

FloatParameter = float | Tuple[float,
                               float] | List[float] | Callable[[int], float]
"""Type alias for float parameter specifications supporting fixed values, ranges, lists, or custom distributions."""

IntParameter = int | Tuple[int, int] | List[int] | Callable[[int], int]
"""Type alias for integer parameter specifications supporting fixed values, ranges, lists, or custom distributions."""

StrParameter = str | List[str]
"""Type alias for string parameter specifications supporting fixed values or lists of choices."""

NumericParameter = FloatParameter | IntParameter
"""Type alias combining both float and integer parameter specifications for numeric transforms."""


def get_distribution(
        params: NumericParameter | StrParameter,
        rng: np.random.Generator = np.random.default_rng()
) -> Callable:
    """
    Creates a random distribution function from various parameter specification formats.

    Converts fixed values, ranges, lists, and callables into consistent distribution functions for use in probabilistic transforms. Supports backward compatibility with legacy TorchSig parameter conventions while providing modern random number generation.

    :param params: Parameter specification defining the distribution type and range
    :type params: NumericParameter | StrParameter
    :param rng: Random number generator instance for sampling operations
    :type rng: np.random.Generator
    :returns: Callable function that generates random values according to specified distribution
    :rtype: Callable
    :raises TypeError: If params type is not supported by the conversion logic
    """
    distribution = params

    if isinstance(params, Callable):
        # custom distribution function
        distribution = params

    if isinstance(params, list):
        # draw samples from uniform distribution from list values
        distribution = partial(rng.choice, params)

    if isinstance(params, tuple):
        # draw samples from uniform distribution from [params[0], params[1])
        distribution = partial(rng.uniform, low=params[0], high=params[1])

    if isinstance(params, (int, float)):
        # draw samples from uniform distribution within [0, params)
        distribution = partial(rng.uniform, high=params)

    return distribution
