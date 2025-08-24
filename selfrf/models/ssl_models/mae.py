"""
Masked Autoencoder (MAE) self-supervised learning model implementation for RF spectrograms in selfRF.

This module provides the MAE class, a PyTorch Lightning module for self-supervised pretraining using masked autoencoding on spectrogram data. Its main responsibility is to enable learning of robust feature representations by reconstructing masked patches of input spectrograms, which helps improve downstream performance on RF signal recognition and detection tasks. The MAE integrates a Vision Transformer backbone, a custom decoder, and supports online linear evaluation for benchmarking learned features. Typical use-cases include pretraining backbones for transfer learning, evaluating self-supervised learning strategies, and experimenting with transformer-based architectures on RF datasets. The module is designed to work seamlessly with the selfRF training pipeline and leverages the lightly library for utility functions and benchmarking.
"""

from typing import List, Tuple

import torch
from pytorch_lightning import LightningModule
from timm.models.vision_transformer import VisionTransformer
from torch import Tensor
from torch.nn import Parameter, MSELoss
from torch.optim import AdamW

from lightly.models import utils
from lightly.models.modules import MAEDecoderTIMM, MaskedVisionTransformerTIMM
from lightly.utils.benchmarking import OnlineLinearClassifier
from lightly.utils.scheduler import CosineWarmupScheduler


class MAE(LightningModule):
    """
    PyTorch Lightning module for Masked Autoencoder (MAE) self-supervised learning on spectrograms.

    This class implements masked autoencoding for learning representations by reconstructing masked patches of input spectrograms.

    :param backbone: VisionTransformer backbone for feature extraction.
    :param num_classes: Number of classes for optional online linear evaluation.
    :param batch_size_per_device: Batch size per device for scaling learning rate.
    :param use_online_linear_eval: If True, enables online linear evaluation during training.
    :param decoder_dim: Decoder embedding dimension.
    :param decoder_depth: Number of decoder transformer layers.
    :param decoder_num_heads: Number of decoder attention heads.
    :param mask_ratio: Fraction of patches to mask during training.
    """

    def __init__(
        self,
        backbone: VisionTransformer,
        num_classes: int,
        batch_size_per_device: int,
        use_online_linear_eval: bool = False,
        mask_ratio: float = 0.75,
    ) -> None:
        """
        Initializes the MAE module with backbone, decoder, and loss functions.

        :param backbone: VisionTransformer backbone for feature extraction.
        :param num_classes: Number of classes for optional online linear evaluation.
        :param batch_size_per_device: Batch size per device for scaling learning rate.
        :param use_online_linear_eval: If True, enables online linear evaluation during training.
        :param mask_ratio: Fraction of patches to mask during training.
        :raises TypeError: If backbone is not a VisionTransformer.
        """
        super().__init__()
        if not isinstance(backbone, VisionTransformer):
            raise TypeError(
                f"backbone must be a timm VisionTransformer but is {type(backbone)}")
        self.save_hyperparameters(ignore=["backbone"])

        self.batch_size_per_device = batch_size_per_device
        self.use_online_linear_eval = use_online_linear_eval
        self.mask_ratio = mask_ratio
        decoder_dim = 512

        self.patch_size = backbone.patch_embed.patch_size[0]
        self.sequence_length = backbone.patch_embed.num_patches + backbone.num_prefix_tokens
        mask_token = Parameter(torch.zeros(1, 1, decoder_dim))
        torch.nn.init.normal_(mask_token, std=0.02)

        self.backbone = MaskedVisionTransformerTIMM(vit=backbone)
        self.decoder = MAEDecoderTIMM(
            num_patches=backbone.patch_embed.num_patches,
            patch_size=self.patch_size,
            embed_dim=backbone.embed_dim,
            decoder_embed_dim=decoder_dim,
            decoder_depth=8,
            decoder_num_heads=16,
            mlp_ratio=4.0,
            proj_drop_rate=0.0,
            attn_drop_rate=0.0,
            mask_token=mask_token,
            in_chans=1,
        )
        self.criterion = MSELoss()

        if self.use_online_linear_eval:
            self.online_classifier = OnlineLinearClassifier(
                feature_dim=backbone.embed_dim, num_classes=num_classes
            )
        else:
            self.online_classifier = None

    def forward(self, x: Tensor) -> Tensor:
        """
        Encodes input images using the transformer backbone and returns the [CLS] token features.

        :param x: Input tensor of spectrogram images.
        :returns: Encoded feature tensor for each input.
        """
        return self.backbone.encode(images=x)[:, 0]

    def forward_encoder(self, images, idx_keep=None):
        """
        Encodes images using the transformer backbone, optionally keeping only selected patches.

        :param images: Input tensor of spectrogram images.
        :param idx_keep: Indices of patches to keep (optional).
        :returns: Encoded feature tensor for selected patches.
        """
        return self.backbone.encode(images=images, idx_keep=idx_keep)

    def forward_decoder(self, x_encoded, idx_keep, idx_mask):
        """
        Decodes encoded features to reconstruct masked patches.

        :param x_encoded: Encoded features from the backbone.
        :param idx_keep: Indices of kept patches.
        :param idx_mask: Indices of masked patches.
        :returns: Predicted reconstruction for masked patches.
        """
        # build decoder input
        batch_size = x_encoded.shape[0]
        x_decode = self.decoder.embed(x_encoded)
        x_masked = utils.repeat_token(
            self.decoder.mask_token, (batch_size, self.sequence_length)
        )
        x_masked = utils.set_at_index(
            x_masked, idx_keep, x_decode.type_as(x_masked))

        # decoder forward pass
        x_decoded = self.decoder.decode(x_masked)

        # predict pixel values for masked tokens
        x_pred = utils.get_at_index(x_decoded, idx_mask)
        x_pred = self.decoder.predict(x_pred)
        return x_pred

    def training_step(
        self, batch: Tuple[List[Tensor], Tensor, List[str]], batch_idx: int
    ) -> Tensor:
        """
        Performs a single training step with masked autoencoding and optional online linear evaluation.

        :param batch: Tuple containing (images, targets, metadata).
        :param batch_idx: Index of the current batch.
        :returns: Training loss value.
        """
        images, targets = batch[0], batch[1]

        batch_size = images.shape[0]
        idx_keep, idx_mask = utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.mask_ratio,
            device=images.device,
        )
        features_encoded = self.forward_encoder(images, idx_keep)
        predictions = self.forward_decoder(
            features_encoded, idx_keep, idx_mask)

        patches = utils.patchify(images, self.patch_size)
        target = utils.get_at_index(patches, idx_mask - 1)

        reconstruction_loss = self.criterion(predictions, target)
        self.log(
            "train_loss", reconstruction_loss, prog_bar=True, sync_dist=True, batch_size=len(targets)
        )

        if self.use_online_linear_eval and self.online_classifier:
            cls_features = features_encoded[:, 0]
            cls_loss, cls_log = self.online_classifier.training_step(
                (cls_features.detach(), targets), batch_idx
            )
            self.log_dict(cls_log, sync_dist=True, batch_size=len(targets))
            self.log("train_cls_loss", cls_loss, prog_bar=True,
                     sync_dist=True, batch_size=len(targets))
            total_loss = reconstruction_loss + cls_loss
            self.log("train_total_loss", total_loss, prog_bar=True,
                     sync_dist=True, batch_size=len(targets))

        return reconstruction_loss

    def validation_step(
        self, batch: Tuple[Tensor, Tensor, List[str]], batch_idx: int
    ) -> Tensor:
        """Compute reconstruction loss (and optional probe loss) on the val set."""

        images, targets = batch[0], batch[1]
        batch_size = images.shape[0]

        # recreate random mask, same as in training_step
        idx_keep, idx_mask = utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.mask_ratio,
            device=images.device,
        )

        with torch.no_grad():
            feats = self.forward_encoder(images, idx_keep)
            preds = self.forward_decoder(feats, idx_keep, idx_mask)
            patches = utils.patchify(images, self.patch_size)
            target = utils.get_at_index(patches, idx_mask - 1)
            recon_loss = self.criterion(preds, target)

        # log reconstruction MSE
        self.log("val_loss", recon_loss,
                 prog_bar=True, sync_dist=True, batch_size=len(targets))

        return recon_loss

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler for MAE training.

        :returns: List of optimizers and schedulers for PyTorch Lightning.
        """
        params, params_no_weight_decay = utils.get_weight_decay_parameters(
            [self.backbone, self.decoder]
        )
        optimizer_params = [
            {"name": "mae", "params": params},
            {
                "name": "mae_no_weight_decay",
                "params": params_no_weight_decay,
                "weight_decay": 0.0,
            },
        ]
        if self.use_online_linear_eval and self.online_classifier:
            optimizer_params.append(
                {
                    "name": "online_classifier",
                    "params": self.online_classifier.parameters(),
                    "weight_decay": 0.0,
                }
            )

        optimizer = AdamW(
            optimizer_params,
            lr=1.5e-4 * self.batch_size_per_device * self.trainer.world_size / 256,
            weight_decay=0.05,
            betas=(0.9, 0.95),
        )
        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=(
                    self.trainer.estimated_stepping_batches
                    / self.trainer.max_epochs
                    * 40
                ),
                max_epochs=self.trainer.estimated_stepping_batches,
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]
