import argparse
import warnings

from detectron2.utils.logger import setup_logger
import torch

from selfrf.finetuning.detection.detectron2.config import Detectron2Config, add_detectron2_config_args, print_config
from selfrf.finetuning.detection.detectron2.inference import inference_dataset
from selfrf.finetuning.detection.detectron2.register import register_dataset
from selfrf.finetuning.detection.detectron2.trainer import do_train
from selfrf.finetuning.detection.detectron2.visualizer import visualize_dataset

setup_logger()


def train(config: Detectron2Config):
    """Register datasets with detectron2."""
    warnings.filterwarnings(
        "ignore",
        message=r".*torch\.cuda\.amp\.autocast.*",
        category=FutureWarning
    )

    # Enable Tensor Core operations for better performance on RTX GPUs
    # Options: 'highest', 'high', 'medium'
    torch.set_float32_matmul_precision('high')

    # Print confirmation
    print(
        f"Tensor Core optimization enabled: {torch.get_float32_matmul_precision()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA capability: {torch.cuda.get_device_capability()}")

    register_dataset(config)

    if config.visualize:
        visualize_dataset(config)
        return

    if config.inference:
        inference_dataset(config)
        return

    do_train(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_detectron2_config_args(parser)
    args = parser.parse_args()

    # Create config from args
    config = Detectron2Config(**vars(args))

    print_config(config)
    train(config)
