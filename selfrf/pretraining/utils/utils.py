"""
General utility functions for pretraining workflows in selfRF.

This module provides helper functions that support dataset management and configuration-driven logic for self-supervised learning experiments. Its main responsibility is to offer reusable utilities, such as class list selection based on configuration, that are needed across multiple pretraining scripts and modules. Typical use-cases include dynamically determining the set of signal classes for training or evaluation based on user-specified configuration options. The module integrates with the broader pretraining pipeline by providing foundational logic that simplifies experiment setup and ensures consistency in dataset handling.
"""

import yaml
from typing import Any, List
from pathlib import Path

from torchsig.signals.signal_lists import TorchSigSignalLists
from torchsig.datasets.dataset_utils import dataset_yaml_name


def get_class_list(config: Any) -> List[str]:
    """
    Returns the list of signal classes based on the configuration.

    If config.family is True, returns the TorchSig family-level signal list.
    Otherwise, reads the signal class list from create_dataset_info.yaml in the dataset directory.

    :param config: Configuration object containing 'family' and 'root' attributes
    :type config: Any
    :returns: List of signal class names to be used for training or evaluation
    :rtype: List[str]
    :raises FileNotFoundError: If create_dataset_info.yaml is not found in dataset directory
    :raises ValueError: If YAML file doesn't contain expected class_list structure
    """
    if config.family:
        return TorchSigSignalLists.family_list

    # Read from create_dataset_info.yaml in dataset directory
    dataset_root = Path(config.root)

    # Look for create_dataset_info.yaml in dataset root and torchsig subdirectory
    yaml_candidates = [
        dataset_root / "torchsig_narrowband_impaired" / "train" / dataset_yaml_name,
        dataset_root / "torchsig" / dataset_yaml_name
    ]

    yaml_file = None
    for candidate in yaml_candidates:
        if candidate.exists():
            yaml_file = candidate
            break

    if yaml_file is None:
        # Fallback to TorchSig all_signals if no YAML found
        print(
            f"Warning: No create_dataset_info.yaml found in {dataset_root}. Using TorchSig all_signals as fallback.")
        return TorchSigSignalLists.all_signals

    try:
        with open(yaml_file, 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)

        # Extract class_list from the YAML structure
        signal_list = None

        # Try different paths in the YAML structure
        if 'read_only' in yaml_data and 'signals' in yaml_data['read_only']:
            signal_list = yaml_data['read_only']['signals'].get('class_list')
        elif 'overrides' in yaml_data:
            signal_list = yaml_data['overrides'].get('class_list')
        elif 'class_list' in yaml_data:
            signal_list = yaml_data['class_list']

        if signal_list is None or not isinstance(signal_list, list):
            raise ValueError(
                f"Could not extract class_list from {yaml_file}. Expected structure not found.")
        return signal_list

    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file {yaml_file}: {e}")
    except Exception as e:
        raise ValueError(f"Error reading signal list from {yaml_file}: {e}")
