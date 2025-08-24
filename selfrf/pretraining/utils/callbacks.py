"""
Custom PyTorch Lightning callbacks for advanced checkpointing in selfRF pretraining.

This module provides specialized callback classes that extend PyTorch Lightning's checkpointing functionality to support saving both full model and backbone-only weights during training. Its main responsibility is to ensure that backbone representations can be easily extracted and reused for downstream tasks or transfer learning scenarios. Typical use-cases include self-supervised pretraining workflows where only the backbone is needed for evaluation or fine-tuning, and experiment management where both full and partial checkpoints are required. The module integrates with Lightning Trainer objects and the broader experiment orchestration pipeline to provide robust and flexible checkpointing options.
"""

from pytorch_lightning.callbacks import ModelCheckpoint
from pathlib import Path
import torch
import os
from typing import Optional


class ModelAndBackboneCheckpoint(ModelCheckpoint):
    """
    PyTorch Lightning callback for saving both full model and backbone-only checkpoints.

    Extends ModelCheckpoint to automatically save the backbone state dict alongside the full model checkpoint.
    Implements top-k tracking for backbone checkpoints to match the full model checkpoint behavior.

    :param args: Positional arguments passed to ModelCheckpoint
    :type args: tuple
    :param kwargs: Keyword arguments passed to ModelCheckpoint
    :type kwargs: dict
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the ModelAndBackboneCheckpoint callback.

        Ensures metric names are not auto-inserted in checkpoint filenames and passes all arguments to the parent class.

        :param args: Positional arguments for ModelCheckpoint
        :type args: tuple
        :param kwargs: Keyword arguments for ModelCheckpoint
        :type kwargs: dict
        """
        super().__init__(auto_insert_metric_name=False, *args, **kwargs)
        # Track backbone checkpoint files for top-k management
        self.backbone_kth_best_model_path = ""
        self.backbone_kth_value = None
        self.backbone_best_k_models = {}

    def _save_checkpoint(self, trainer, filepath):
        """
        Saves both the full model checkpoint and the backbone-only state dict.

        The backbone checkpoint is saved with a modified filename to distinguish it from the full model checkpoint.
        Also manages top-k tracking for backbone checkpoints.

        :param trainer: PyTorch Lightning Trainer instance managing the training loop
        :type trainer: pytorch_lightning.Trainer
        :param filepath: File path where the full model checkpoint will be saved
        :type filepath: str or Path
        """
        # Save the full model checkpoint first
        super()._save_checkpoint(trainer, filepath)

        # Create backbone checkpoint path
        backbone_path = Path(filepath).with_stem(
            f"{Path(filepath).stem}_backbone")

        # Save backbone state dict
        torch.save(trainer.model.backbone.state_dict(), backbone_path)

        # Manage top-k for backbone checkpoints
        self._manage_backbone_top_k(backbone_path, trainer)

    def _manage_backbone_top_k(self, backbone_path: Path, trainer):
        """
        Manages top-k backbone checkpoints by removing older ones when limit is exceeded.

        :param backbone_path: Path to the newly saved backbone checkpoint
        :type backbone_path: Path
        :param trainer: PyTorch Lightning Trainer instance
        :type trainer: pytorch_lightning.Trainer
        """
        if self.save_top_k is None or self.save_top_k == -1:
            # Keep all checkpoints
            return

        if self.save_top_k == 0:
            # Don't keep any checkpoints, remove the one we just saved
            if backbone_path.exists():
                os.remove(backbone_path)
            return

        # Get current metric value (same as used for full model)
        current_score = self._get_metric_value(trainer)

        if current_score is None:
            return

        # Convert to string for consistency with parent class
        backbone_path_str = str(backbone_path)

        # Update backbone best models tracking
        self.backbone_best_k_models[backbone_path_str] = current_score

        if len(self.backbone_best_k_models) > self.save_top_k:
            # Find the worst model to remove
            if self.mode == "min":
                worst_path = max(self.backbone_best_k_models.keys(),
                                 key=lambda k: self.backbone_best_k_models[k])
            else:  # mode == "max"
                worst_path = min(self.backbone_best_k_models.keys(),
                                 key=lambda k: self.backbone_best_k_models[k])

            # Remove the worst backbone checkpoint
            if os.path.exists(worst_path):
                os.remove(worst_path)
                if self.verbose:
                    print(f"Removed backbone checkpoint: {worst_path}")

            # Remove from tracking
            del self.backbone_best_k_models[worst_path]

        # Update kth best tracking
        if len(self.backbone_best_k_models) == self.save_top_k:
            if self.mode == "min":
                self.backbone_kth_value = max(
                    self.backbone_best_k_models.values())
                self.backbone_kth_best_model_path = max(self.backbone_best_k_models.keys(),
                                                        key=lambda k: self.backbone_best_k_models[k])
            else:
                self.backbone_kth_value = min(
                    self.backbone_best_k_models.values())
                self.backbone_kth_best_model_path = min(self.backbone_best_k_models.keys(),
                                                        key=lambda k: self.backbone_best_k_models[k])

    def _get_metric_value(self, trainer) -> Optional[float]:
        """
        Get the current metric value for comparison.

        :param trainer: PyTorch Lightning Trainer instance
        :type trainer: pytorch_lightning.Trainer
        :return: Current metric value or None
        :rtype: Optional[float]
        """
        if self.monitor is None:
            return None

        logs = trainer.callback_metrics
        if self.monitor in logs:
            return float(logs[self.monitor])
        return None

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """
        Called when saving a checkpoint to save extra metadata about backbone checkpoints.

        :param trainer: PyTorch Lightning Trainer instance
        :type trainer: pytorch_lightning.Trainer
        :param pl_module: Lightning module being trained
        :type pl_module: pytorch_lightning.LightningModule
        :param checkpoint: Checkpoint dictionary
        :type checkpoint: dict
        """
        super().on_save_checkpoint(trainer, pl_module, checkpoint)

        # Add backbone checkpoint tracking to the main checkpoint
        checkpoint["backbone_best_k_models"] = self.backbone_best_k_models.copy()
        checkpoint["backbone_kth_best_model_path"] = self.backbone_kth_best_model_path
        checkpoint["backbone_kth_value"] = self.backbone_kth_value

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        """
        Called when loading a checkpoint to restore backbone checkpoint tracking.

        :param trainer: PyTorch Lightning Trainer instance
        :type trainer: pytorch_lightning.Trainer
        :param pl_module: Lightning module being trained
        :type pl_module: pytorch_lightning.LightningModule
        :param checkpoint: Checkpoint dictionary
        :type checkpoint: dict
        """
        super().on_load_checkpoint(trainer, pl_module, checkpoint)

        # Restore backbone checkpoint tracking
        if "backbone_best_k_models" in checkpoint:
            self.backbone_best_k_models = checkpoint["backbone_best_k_models"]
        if "backbone_kth_best_model_path" in checkpoint:
            self.backbone_kth_best_model_path = checkpoint["backbone_kth_best_model_path"]
        if "backbone_kth_value" in checkpoint:
            self.backbone_kth_value = checkpoint["backbone_kth_value"]
