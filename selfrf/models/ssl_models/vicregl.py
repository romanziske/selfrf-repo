"""
VICRegL self-supervised learning model implementation for RF spectrograms in selfRF.

This module provides the VICRegL class, a PyTorch Lightning module for self-supervised learning using the VICRegL algorithm, which combines global and local invariance regularization for robust representation learning on spectrogram data. Its main responsibilities include defining the model architecture, projection heads, loss computation, and optimizer configuration for VICRegL training. Typical use-cases involve pretraining feature extractors for downstream RF signal recognition or detection tasks, benchmarking self-supervised methods, and experimenting with local-global contrastive learning strategies. The module integrates with the selfRF training pipeline, supports custom backbones, and leverages the lightly library for loss functions, projection heads, and learning rate scheduling. It is designed to work seamlessly with other selfRF SSL models and data modules, enabling flexible experimentation and transfer learning on RF datasets.
"""

import pytorch_lightning as pl
from torch import nn

import torch

from lightly.utils.benchmarking import OnlineLinearClassifier
from lightly.loss import VICRegLLoss
from lightly.models.modules import BarlowTwinsProjectionHead
from lightly.models.modules.heads import VicRegLLocalProjectionHead
from lightly.utils.lars import LARS
from lightly.utils.scheduler import CosineWarmupScheduler
from lightly.models.utils import get_weight_decay_parameters


class VICRegL(pl.LightningModule):
    """
    PyTorch Lightning module for VICRegL self-supervised learning on spectrograms.

    This class implements the VICRegL algorithm, combining global and local invariance regularization for representation learning.

    :param backbone: Backbone neural network for feature extraction.
    :param batch_size_per_device: Batch size per device for scaling learning rate.
    :param input_dim: Input feature dimension for projection heads.
    :param proj_hidden_dim: Hidden layer dimension for global projection head.
    :param proj_out_dim: Output feature dimension for global projection head.
    :param local_proj_hidden_dim: Hidden layer dimension for local projection head.
    :param local_out_dim: Output feature dimension for local projection head.
    :param base_learning_rate: Base learning rate for optimizer.
    :param momentum: Momentum value for optimizer.
    :param weight_decay: Weight decay for optimizer.
    :param use_online_linear_eval: If True, enables online linear evaluation during training.
    :param num_classes: Number of classes for optional online linear evaluation.
    """

    def __init__(
        self,
        backbone: nn.Module,
        batch_size_per_device: int,
        input_dim: int = 2048,
        proj_hidden_dim: int = 2048,
        proj_out_dim: int = 2048,
        local_proj_hidden_dim: int = 128,
        local_out_dim: int = 128,
        base_learning_rate: float = 0.2,
        momentum: float = 0.9,
        weight_decay: float = 1.5e-6,
        use_online_linear_eval: bool = False,
        num_classes: int = 10,
    ):
        """
        Initializes the VICRegL module with backbone, projection heads, and loss function.

        :param backbone: Backbone neural network for feature extraction.
        :param batch_size_per_device: Batch size per device for scaling learning rate.
        :param input_dim: Input feature dimension for projection heads.
        :param proj_hidden_dim: Hidden layer dimension for global projection head.
        :param proj_out_dim: Output feature dimension for global projection head.
        :param local_proj_hidden_dim: Hidden layer dimension for local projection head.
        :param local_out_dim: Output feature dimension for local projection head.
        :param base_learning_rate: Base learning rate for optimizer.
        :param momentum: Momentum value for optimizer.
        :param weight_decay: Weight decay for optimizer.
        :param use_online_linear_eval: If True, enables online linear evaluation during training.
        :param num_classes: Number of classes for optional online linear evaluation.
        """
        super().__init__()
        self.save_hyperparameters(ignore=['backbone'])
        self.backbone = backbone
        self.batch_size_per_device = batch_size_per_device

        self.average_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.projection_head = BarlowTwinsProjectionHead(
            input_dim, proj_hidden_dim, proj_out_dim
        )
        self.local_projection_head = VicRegLLocalProjectionHead(
            input_dim, local_proj_hidden_dim, local_out_dim
        )
        self.criterion = VICRegLLoss()

        self.base_learning_rate = base_learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay

        self.use_online_linear_eval = use_online_linear_eval
        if self.use_online_linear_eval:
            self.online_classifier = OnlineLinearClassifier(
                num_classes=num_classes)

    def forward(self, x):
        """
        Computes global and local projections for input spectrograms.

        :param x: Input tensor of spectrograms.
        :returns: Tuple of (global projection, local projection).
        """
        x = self.backbone(x)
        y_global_pooled = self.average_pool(x).flatten(start_dim=1)
        z_global = self.projection_head(y_global_pooled)
        y_local_patches = x.permute(0, 2, 3, 1)
        z_local_patches = self.local_projection_head(y_local_patches)
        return z_global, z_local_patches

    def training_step(self, batch, batch_index):
        """
        Performs a single training step with VICRegL losses.

        :param batch: Tuple containing (views, grids) and targets.
        :param batch_index: Index of the current batch.
        :returns: Training loss value.
        """

        (views, grids), targets = batch
        features = [self.forward(view) for view in views]

        loss = self.criterion(
            global_view_features=features[:2],
            global_view_grids=grids[:2],
            local_view_features=features[2:],
            local_view_grids=grids[2:],
        )

        if self.use_online_linear_eval:
            # Use features from the first view for online classification
            backbone_features = self.backbone(views[0])
            pooled_features = self.average_pool(
                backbone_features).flatten(start_dim=1)

            cls_loss, cls_log = self.online_classifier.training_step(
                (pooled_features.detach(), targets), batch_index
            )

            # Log classification metrics
            self.log_dict(cls_log, prog_bar=False, batch_size=len(targets))

            # **Optional: Add classification loss to total loss**
            # total_loss = loss + 0.1 * cls_loss  # Scale classification loss
            # return total_loss

        self.log("epoch", self.current_epoch, on_step=False, on_epoch=True)
        self.log(
            "train_loss",
            loss,
            on_step=True,    # log once per training iteration
            on_epoch=True,   # also accumulate & log at the end of the epoch
            prog_bar=True,   # show in tqdm bar
            logger=True,     # send to TensorBoard / WandB / etc.
        )
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'])
        return loss

    def validation_step(self, batch, batch_idx):
        """Online linear evaluation during validation"""
        if not self.use_online_linear_eval:
            return None

        (views, grids), targets = batch
        x = views[0]

        # Extract features
        backbone_features = self.backbone(x)
        features = self.average_pool(backbone_features).flatten(start_dim=1)

        # Use validation_step method
        cls_loss, cls_log = self.online_classifier.validation_step(
            (features.detach(), targets), batch_idx
        )

        # Log validation metrics
        self.log_dict(cls_log, prog_bar=True, batch_size=len(targets))

        return cls_loss

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler for VICRegL training.

        :returns: List of optimizers and schedulers for PyTorch Lightning.
        """
        params_with_weight_decay, params_without_weight_decay = get_weight_decay_parameters(
            [
                self.backbone,
                self.average_pool,
                self.projection_head,
                self.local_projection_head,
            ])

        param_groups = [
            {
                "name": "vicregl_weight_decay",
                "params": params_with_weight_decay,
                "weight_decay": self.weight_decay,
            },
            {
                "name": "vicregl_no_weight_decay",
                "params": params_without_weight_decay,
                "weight_decay": 0.0
            },
        ]

        if self.use_online_linear_eval:
            param_groups.append(
                {
                    "name": "online_classifier",
                    "params": self.online_classifier.parameters(),
                    "weight_decay": 0.0,
                },
            )

        global_batch_size = self.batch_size_per_device * self.trainer.world_size
        base_lr = _get_base_learning_rate(global_batch_size=global_batch_size)

        optimizer = LARS(
            param_groups,
            lr=base_lr * global_batch_size / 256,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

        scheduler = CosineWarmupScheduler(
            optimizer=optimizer,
            warmup_epochs=int(
                self.trainer.estimated_stepping_batches
                / self.trainer.max_epochs
                * 10
            ),
            max_epochs=int(self.trainer.estimated_stepping_batches),
            end_value=0.01,  # Scale base learning rate from 0.2 to 0.002.
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


def _get_base_learning_rate(global_batch_size: int) -> float:
    """
    Returns the base learning rate for training 100 epochs with a given batch size.

    This follows section C.4 in https://arxiv.org/pdf/2105.04906.pdf.

    :param global_batch_size: Total batch size across all devices.
    :returns: Recommended base learning rate for the given batch size.
    """
    if global_batch_size <= 128:
        return 0.8
    elif global_batch_size == 256:
        return 0.5
    elif global_batch_size == 512:
        return 0.4
    else:
        return 0.3
