"""
Evaluation-specific configuration management for assessing pretrained SSL model representations.

This module provides specialized configuration handling for evaluating self-supervised learning models after pretraining, extending the BaseConfig foundation with evaluation-specific parameters like visualization settings, nearest neighbor analysis, and output path management. It serves as the primary configuration interface for evaluation scripts that assess the quality of learned representations through various metrics and visualization techniques. The EvaluationConfig class manages essential evaluation parameters including model checkpoint paths, t-SNE visualization settings, k-nearest neighbor analysis configuration, and output directory specifications. It integrates with the evaluation pipeline by providing a standardized configuration interface that evaluation scripts can use to configure model loading, representation extraction, and analysis workflows consistently across different experimental setups. The module supports both command-line argument parsing and programmatic configuration creation, enabling flexible usage in both automated evaluation pipelines and interactive analysis workflows for assessing SSL model performance.
"""
import argparse
from dataclasses import dataclass

from .base_config import BaseConfig, add_base_config_args, parse_base_config

DEFAULT_TSNE = True
DEFAULT_KNN = True
DEFAULT_N_NEIGHBORS = 10
DEFAULT_EVALUATION_PATH = './evaluation'


@dataclass
class EvaluationConfig(BaseConfig):
    """
    Configuration class for SSL model evaluation workflows extending BaseConfig with evaluation-specific parameters.

    Inherits all base configuration parameters and adds model loading, visualization, and analysis capabilities for assessing representation quality.

    :param model_path: Path to the pretrained model checkpoint file to evaluate
    :type model_path: str
    :param tsne: Whether to generate t-SNE visualizations of learned representations
    :type tsne: bool
    :param knn: Whether to perform k-nearest neighbor analysis on representations
    :type knn: bool
    :param n_neighbors: Number of neighbors to use in k-NN analysis
    :type n_neighbors: int
    :param evaluation_path: Directory path for saving evaluation results and visualizations
    :type evaluation_path: str
    """
    model_path: str = '.'
    tsne: bool = DEFAULT_TSNE
    knn: bool = DEFAULT_KNN
    n_neighbors: int = DEFAULT_N_NEIGHBORS
    evaluation_path: str = DEFAULT_EVALUATION_PATH


def add_evaluation_config_args(parser: argparse.ArgumentParser) -> None:
    """
    Adds evaluation-specific configuration arguments to an ArgumentParser instance.

    Extends base configuration arguments with evaluation parameters including model paths, visualization settings, and analysis configuration.

    :param parser: ArgumentParser instance to add evaluation configuration arguments to
    :type parser: argparse.ArgumentParser
    """
    add_base_config_args(parser)
    parser.add_argument(
        '--model-path',
        type=str,
        default='.',
    )
    parser.add_argument(
        '--tsne',
        type=lambda x: x.lower() == 'true',
        default=DEFAULT_TSNE,
    )
    parser.add_argument(
        '--knn',
        type=lambda x: x.lower() == 'true',
        default=DEFAULT_KNN,
    )
    parser.add_argument(
        '--n-neighbors',
        type=int,
        default=DEFAULT_N_NEIGHBORS,
    )
    parser.add_argument(
        '--evaluation-path',
        type=str,
        default=DEFAULT_EVALUATION_PATH,
    )


def parse_evaluation_config() -> EvaluationConfig:
    """
    Parses command line arguments into an EvaluationConfig object with proper inheritance handling.

    Creates base configuration first then extends it with evaluation-specific parameters to ensure proper field inheritance and validation.

    :returns: Fully configured EvaluationConfig instance with parsed argument values
    :rtype: EvaluationConfig
    """
    # Create parser with description
    parser = argparse.ArgumentParser(description="Evaluation Config")
    add_evaluation_config_args(parser)

    # First parse the base config (handles num_iq_samples properly)
    base_config = parse_base_config(parser)

    # Get the args again to extract evaluation-specific fields
    args = parser.parse_args()

    # Create EvaluationConfig by combining base config and evaluation args
    evaluation_config = EvaluationConfig(
        **vars(base_config),  # Unpack base config

        # Add evaluation fields
        model_path=args.model_path,
        tsne=args.tsne,
        knn=args.knn,
        n_neighbors=args.n_neighbors,
        evaluation_path=args.evaluation_path
    )

    return evaluation_config
