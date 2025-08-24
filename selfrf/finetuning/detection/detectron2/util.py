"""
Utility functions and configurations for Detectron2-based RF signal detection training.
"""
import datetime
from enum import Enum
import json
import os
from typing import Dict, NamedTuple, Optional


def create_output_directory(config) -> str:
    """
    Creates a uniquely named output directory and saves configuration information.

    :param config: Configuration object containing training parameters
    :type config: Detectron2Config
    :returns: Path to the created output directory
    :rtype: str
    :raises OSError: If directory creation fails
    :raises IOError: If configuration file cannot be written
    """
    # Create a uniquely named output directory with timestamp
    timestamp: str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir: str = os.path.join(
        config.output_dir,
        f"run_{timestamp}_{str(config.model_type)}"
    )

    os.makedirs(output_dir, exist_ok=True)

    run_info: Dict = {
        "timestamp": timestamp,
        "config": config.__dict__,
        "output_dir": output_dir
    }

    with open(os.path.join(output_dir, "run_info.json"), "w") as f:
        json.dump(run_info, f, indent=2, default=str)

    return output_dir


class ModelConfig(NamedTuple):
    """
    Configuration container for model architecture specifications.

    :param path: Path to config file
    :type path: Optional[str]
    :param is_lazy: Whether model uses LazyConfig system
    :type is_lazy: bool
    :param identifier: Unique identifier for the configuration
    :type identifier: str
    """
    path: Optional[str] = None  # Optional for lazy configs
    is_lazy: bool = False
    identifier: str = ""


class ModelType(Enum):
    """
    Enumeration of supported model architectures with their configurations.
    """
    FASTER_RCNN_R50_FPN = ModelConfig(
        path="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
        is_lazy=False,
        identifier="faster_rcnn_r50_fpn"
    )
    VITDET_VIT_B = ModelConfig(
        is_lazy=True,
        identifier="vitdet_vit_b"
    )
    VITDET_VIT_L = ModelConfig(
        is_lazy=True,
        identifier="vitdet_vit_l"
    )
    VITDET_VIT_H = ModelConfig(
        is_lazy=True,
        identifier="vitdet_vit_h"
    )

    @property
    def config_path(self) -> str:
        """
        Returns the configuration file path for non-lazy configs.

        :returns: Path to configuration file
        :rtype: str
        :raises ValueError: If called on lazy config without path
        """
        if self.is_lazy_config:
            raise ValueError(
                f"Config path not available for lazy config: {self.name}"
            )
        return self.value.path

    @property
    def is_lazy_config(self) -> bool:
        """
        Checks if the model uses lazy configuration system.

        :returns: True if model uses lazy config
        :rtype: bool
        """
        return self.value.is_lazy

    @classmethod
    def from_string(cls, name: str) -> 'ModelType':
        """
        Converts string representation to ModelType enum.

        :param name: String representation of model type
        :type name: str
        :returns: Corresponding ModelType enum value
        :rtype: ModelType
        :raises ValueError: If model type string is not recognized
        """
        try:
            # Convert name to uppercase and replace hyphens with underscores
            enum_name = name.upper().replace('-', '_')
            model_type = cls[enum_name]
            return model_type
        except KeyError:
            # Show available model types in error message
            valid_types = [str(t) for t in cls]
            raise ValueError(
                f"Unknown model type: '{name}'. "
                f"Valid types are: {', '.join(valid_types)}"
            )
