"""
This module provides factory functions and classes for constructing collate functions used in selfRF pretraining dataloaders. Its role is to abstract and standardize the batching logic for different self-supervised learning (SSL) algorithms, supporting single-view, multi-view, and grid-based data representations. The main responsibilities include selecting the appropriate collate function based on configuration, enabling compatibility with a variety of SSL methods, and ensuring correct batch formatting for distributed and evaluation workflows. Typical use cases involve building collate functions for use in PyTorch dataloaders during SSL training or evaluation, and supporting advanced batching strategies for methods like VICRegL, or MAE. This module integrates with the broader selfRF pretraining pipeline by providing a unified interface for collate function selection and construction.
"""

import torch
from typing import Any, Callable, List, Tuple, Union

from selfrf.pretraining.utils.enums import CollateType
from selfrf.pretraining.config import TrainingConfig, EvaluationConfig


def build_collate_fn(config: Union[TrainingConfig, EvaluationConfig]) -> Callable:
    """
    Builds and returns the appropriate collate function based on the SSL model type in the configuration.
    This function enables dynamic selection of batching logic for different SSL algorithms.

    :param config: Training or evaluation configuration object.
    :returns: A callable collate function for use in PyTorch dataloaders.
    :raises ValueError: If the SSL model's collate type is unknown.
    """
    if isinstance(config, EvaluationConfig):
        return collate_fn_evaluation

    ssl_model = config.ssl_model

    if ssl_model.collate_type == CollateType.MULTI_VIEW:
        return MultiViewCollate()
    elif ssl_model.collate_type == CollateType.SINGLE_VIEW:
        return SingleViewCollate()
    elif ssl_model.collate_type == CollateType.MULTI_VIEW_GRID:
        return MultiViewGridCollate()
    else:
        raise ValueError(f"Unknown collate type for SSL model: {ssl_model}")


def collate_fn_evaluation(batch):
    """
    Collate function for evaluation mode, stacking tensors and targets into batches.

    :param batch: List of (tensor, target) pairs.
    :returns: Tuple of batched tensors and targets.
    """
    tensors, targets = zip(*batch)
    tensors = torch.stack(tensors)
    targets = torch.tensor(targets)
    return tensors, targets


class SingleViewCollate:
    """
    Collate function for single-view SSL methods such as MAE.
    This class stacks single-view samples and their targets into batched tensors.
    """

    def __call__(self, batch: List[Any]) -> torch.Tensor:
        """
        Converts a batch of single-view samples into a batched tensor and target tensor.

        :param batch: List of (tensor, target) pairs or single-tensor samples.
        :returns: Tuple of batched tensors and targets.
        """
        datas, targets = zip(*batch)
        return torch.stack(datas), torch.tensor(targets)


class MultiViewCollate:
    """
    Collate function for multi-view SSL methods.
    This class stacks multiple augmented views and their targets into batched tensors.
    """

    def __call__(self, batch: List[Any]) -> Tuple[torch.Tensor, ...]:
        """
        Converts a batch of multi-view samples into tuples of batched views and targets.

        :param batch: List of ((view1, view2), target) pairs.
        :returns: Tuple of batched views and targets.
        """
        views, targets = zip(*batch)
        view1s, view2s = zip(*views)
        return (
            torch.stack(view1s),
            torch.stack(view2s)
        ), torch.tensor(targets)


class MultiViewGridCollate:
    """
    Collate function for SSL transforms that output multiple views and corresponding grids, such as VICRegL.
    This class stacks lists of view and grid tensors and their targets into batched tensors.
    """

    def __call__(self, batch: List[Any]) -> Tuple[Tuple[List[torch.Tensor], List[torch.Tensor]], torch.Tensor]:
        """
        Converts a batch of multi-view grid samples into tuples of batched views, grids, and targets.
        """
        data_tuples, targets = zip(*batch)

        num_view_types = len(data_tuples[0][0])
        num_grid_types = len(data_tuples[0][1])

        batched_views = []
        for i in range(num_view_types):
            current_view_type_batch = [sample_data_tuple[0][i]
                                       for sample_data_tuple in data_tuples]
            batched_views.append(torch.stack(current_view_type_batch))

        batched_grids = []
        for i in range(num_grid_types):
            current_grid_type_batch = [sample_data_tuple[1][i]
                                       for sample_data_tuple in data_tuples]
            batched_grids.append(torch.stack(current_grid_type_batch))

        #  Handle mixed tuple/integer targets
        processed_targets = []
        for target in targets:
            if isinstance(target, (tuple, list)):
                # Extract first element from tuple: (7, 7) -> 7
                processed_targets.append(target[0])
            else:
                # Keep integer as is: 7 -> 7
                processed_targets.append(target)

        print(f"Processed targets: {processed_targets}")
        batched_targets = torch.tensor(processed_targets)
        return (batched_views, batched_grids), batched_targets
