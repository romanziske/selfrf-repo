"""
DenseCL self-supervised learning model implementation for RF spectrograms in selfRF.

This module provides the DenseCL class, a PyTorch Lightning module for self-supervised contrastive learning using dense local and global representations. Its main responsibility is to enable training of feature extractors on spectrogram data without labels, leveraging both global and local contrastive losses to improve representation quality. Typical use-cases include pretraining backbones for downstream RF signal recognition or detection tasks, as well as benchmarking self-supervised learning methods on RF datasets. The module integrates with the selfRF training pipeline, supporting custom backbones, online linear evaluation, and advanced optimizer/scheduler configurations. It relies on the lightly library for loss functions and momentum encoder utilities, and is designed to work seamlessly with other selfRF SSL models and data modules.
"""

import copy
from typing import List, Tuple

import pytorch_lightning as pl
import torch
from torch.optim import SGD
from torch import Tensor, nn

from lightly.loss import NTXentLoss
from lightly.models.utils import deactivate_requires_grad, update_momentum, select_most_similar
from lightly.models.modules import DenseCLProjectionHead
from lightly.utils.scheduler import cosine_schedule
from lightly.utils.scheduler import CosineWarmupScheduler
from lightly.models.utils import get_weight_decay_parameters


class DenseCL(pl.LightningModule):
    """
    PyTorch Lightning module for DenseCL self-supervised learning on spectrograms.

    This class implements the DenseCL algorithm, combining global and local contrastive losses with momentum encoders for robust representation learning.

    :param backbone: Backbone neural network for feature extraction
    :param batch_size_per_device: Batch size per device for scaling learning rate
    :param start_momentum: Initial momentum value for the momentum encoder
    :param lambda_weight: Weighting factor between global and local losses
    :param input_dim: Input feature dimension for projection heads
    :param hidden_dim: Hidden layer dimension for projection heads
    :param output_dim: Output feature dimension for projection heads
    :param num_classes: Number of classes for optional online linear evaluation
    :param use_online_linear_eval: If True, enables online linear evaluation during training
    """

    def __init__(
        self,
        backbone: nn.Module,
        batch_size_per_device: int,
        start_momentum: float = 0.996,
        lambda_weight: float = 0.5,
        input_dim: int = 2048,
        hidden_dim: int = 2048,
        output_dim: int = 128,
        num_classes: int = 10,
        use_online_linear_eval: bool = False,
    ):
        """
        Initializes the DenseCL module with backbone, projection heads, and loss functions.

        :param backbone: Backbone neural network for feature extraction
        :param batch_size_per_device: Batch size per device for scaling learning rate
        :param start_momentum: Initial momentum value for the momentum encoder
        :param lambda_weight: Weighting factor between global and local losses
        :param input_dim: Input feature dimension for projection heads
        :param hidden_dim: Hidden layer dimension for projection heads
        :param output_dim: Output feature dimension for projection heads
        :param num_classes: Number of classes for optional online linear evaluation
        :param use_online_linear_eval: If True, enables online linear evaluation during training
        """
        super().__init__()
        self.save_hyperparameters(ignore=['backbone'])
        self.batch_size_per_device = batch_size_per_device

        self.backbone = backbone
        self.projection_head_global = DenseCLProjectionHead(
            input_dim, hidden_dim, output_dim)
        self.projection_head_local = DenseCLProjectionHead(
            input_dim, hidden_dim, output_dim)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_global_momentum = copy.deepcopy(
            self.projection_head_global
        )
        self.projection_head_local_momentum = copy.deepcopy(
            self.projection_head_local)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_global_momentum)
        deactivate_requires_grad(self.projection_head_local_momentum)

        self.criterion_global = NTXentLoss(
            temperature=0.2,
            memory_bank_size=(16384*2, output_dim),
        )

        self.criterion_local = NTXentLoss(
            temperature=0.2,
            memory_bank_size=(16384*2, output_dim),
        )

        self.start_momentum = start_momentum
        self.lambda_weight = lambda_weight

    def forward(self, x):
        """
        Computes global and local features and projections for input x.

        :param x: Input tensor of spectrograms
        :type x: torch.Tensor
        :returns: Tuple of (features, global projection, local projection)
        :rtype: tuple
        """
        query_features = self.backbone(x)
        query_global = self.pool(query_features).flatten(start_dim=1)
        query_global = self.projection_head_global(query_global)
        query_features = query_features.flatten(start_dim=2).permute(0, 2, 1)
        query_local = self.projection_head_local(query_features)
        return query_features, query_global, query_local

    @torch.no_grad()
    def forward_momentum(self, x):
        """
        Computes global and local projections using the momentum encoder.

        :param x: Input tensor of spectrograms
        :type x: torch.Tensor
        :returns: Tuple of (features, global projection, local projection)
        :rtype: tuple
        """
        key_features = self.backbone_momentum(x)
        key_global = self.pool(key_features).flatten(start_dim=1)
        key_global = self.projection_head_global_momentum(key_global)
        key_features = key_features.flatten(start_dim=2).permute(0, 2, 1)
        key_local = self.projection_head_local_momentum(key_features)
        return key_features, key_global, key_local

    def training_step(
        self,
        batch: Tuple[List[Tensor], Tensor, List[str]],
        batch_idx: int
    ) -> Tensor:
        """
        Performs a single training step with DenseCL losses and momentum updates.

        :param batch: Tuple containing (views, targets, metadata)
        :type batch: tuple
        :param batch_idx: Index of the current batch
        :type batch_idx: int
        :returns: Training loss value
        :rtype: torch.Tensor
        """
        momentum = cosine_schedule(
            self.global_step,
            int(self.trainer.estimated_stepping_batches),
            self.start_momentum,
            1,
        )
        update_momentum(
            self.backbone,
            self.backbone_momentum,
            m=momentum,
        )
        update_momentum(
            self.projection_head_global,
            self.projection_head_global_momentum,
            m=momentum,
        )
        update_momentum(
            self.projection_head_local,
            self.projection_head_local_momentum,
            m=momentum,
        )

        x_query, x_key = batch[0]
        query_features, query_global, query_local = self.forward(x_query)
        key_features, key_global, key_local = self.forward_momentum(x_key)

        key_local = select_most_similar(
            query_features, key_features, key_local)
        query_local = query_local.flatten(end_dim=1)
        key_local = key_local.flatten(end_dim=1)

        loss_global = self.criterion_global(query_global, key_global)
        loss_local = self.criterion_local(query_local, key_local)
        loss = (1 - self.lambda_weight) * loss_global + \
            self.lambda_weight * loss_local

        self.log("train_loss", loss, prog_bar=True, batch_size=len(batch[1]))
        self.log("loss_global", loss_global)
        self.log("loss_local", loss_local)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'])

        return loss

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler for DenseCL training.

        :returns: List of optimizers and schedulers for PyTorch Lightning
        :rtype: tuple
        """
        params_with_weight_decay, params_without_weight_decay = get_weight_decay_parameters(
            [
                self.backbone,
                self.projection_head_global,
                self.projection_head_local
            ])

        param_groups = [
            {
                "name": "densecl_weight_decay",
                "params": params_with_weight_decay,
                "weight_decay": 1e-4,
            },
            {
                "name": "densecl_no_weight_decay",
                "params": params_without_weight_decay,
                "weight_decay": 0.0
            },
        ]

        optimizer = SGD(
            param_groups,
            lr=0.05 * self.batch_size_per_device * self.trainer.world_size / 256,
            momentum=0.9,
            weight_decay=1e-4,
        )

        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=int(
                    self.trainer.estimated_stepping_batches
                    / (self.trainer.max_epochs or 1)
                    * 10
                ),
                max_epochs=int(self.trainer.estimated_stepping_batches),
            ),
            "interval": "step",
        }
        return [optimizer],
