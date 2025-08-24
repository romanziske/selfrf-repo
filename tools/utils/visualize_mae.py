import argparse
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Use lightly's utils directly
from lightly.models import utils as lightly_utils

from selfrf.models.ssl_models.mae import MAE
from selfrf.pretraining.config.training_config import TrainingConfig
from selfrf.pretraining.factories.dataset_factory import build_dataloader
from selfrf.pretraining.factories.model_factory import build_ssl_model
from selfrf.pretraining.factories.transform_factory import TransformFactory
from selfrf.pretraining.utils.enums import BackboneType, DatasetType, SSLModelType


def load_model_checkpoint(model, checkpoint, config):
    """Loads the model state dictionary from a checkpoint."""
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)
        print("Loaded model state_dict from checkpoint.")
    else:
        # Assume the checkpoint itself is the state_dict
        model.load_state_dict(checkpoint)
        print("Loaded model state_dict directly from checkpoint object.")


def visualize_mae_output(original, mask, reconstruction, output_dir, filename_prefix):
    """Generates and saves original, masked, and reconstructed spectrogram images."""
    output_dir.mkdir(parents=True, exist_ok=True)

    original = original.squeeze().cpu().numpy()
    reconstruction = reconstruction.squeeze().cpu().numpy()
    # Use the passed mask tensor directly
    mask_np = mask.squeeze().cpu().numpy()  # Convert mask tensor to numpy

    masked_image = original.copy()
    # Use the numpy mask to gray out areas
    masked_image[mask_np > 0] = 0  # Gray out masked patches with mid-gray

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('MAE Visualization', fontsize=16)

    # Original
    axs[0].imshow(original, cmap='viridis', aspect='auto')
    axs[0].set_title('Original Spectrogram')
    axs[0].axis('off')

    # Masked
    axs[1].imshow(masked_image, cmap='viridis', aspect='auto')
    axs[1].set_title('Masked Spectrogram')
    axs[1].axis('off')

    # Reconstructed
    axs[2].imshow(reconstruction, cmap='viridis', aspect='auto')
    # Correct the index for the reconstructed title (was axs[3])
    axs[2].set_title('Reconstructed Spectrogram')
    axs[2].axis('off')

    # Adjust layout to prevent title overlap
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = output_dir / f"{filename_prefix}_mae_visualization.png"
    plt.savefig(save_path)
    print(f"Saved visualization to {save_path}")
    plt.close(fig)


def main(cli_args, config: TrainingConfig):

    datamodule = build_dataloader(config)
    datamodule.transforms = TransformFactory.create_spectrogram_transform(
        config)

    model: MAE = build_ssl_model(config)

    print(f"Loading weights from {cli_args.checkpoint_path}")
    checkpoint = torch.load(
        cli_args.checkpoint_path,
        map_location=config.device,
        weights_only=False,  # Set to False if it might contain code, True if only weights
    )
    load_model_checkpoint(model, checkpoint, config)

    model = model.to(config.device)
    model.eval()

    datamodule.setup("fit")
    dataloader = datamodule.val_dataloader()

    # 2. Load Data Sample
    target_batch = None
    for i, batch in enumerate(dataloader):
        if i == cli_args.sample_idx:
            target_batch = batch  # Store the correct batch
            break  # Exit loop once the target batch is found

    spectrograms_batch = target_batch[0]
    spectrogram_sample = spectrograms_batch[0].unsqueeze(0)

    spectrogram_sample = spectrogram_sample.to(config.device)

    # 3. Run MAE Forward Pass
    with torch.no_grad():
        # --- Get Mask Indices ---
        batch_size = spectrogram_sample.shape[0]
        # Use sequence_length and mask_ratio from the loaded model
        idx_keep, idx_mask = lightly_utils.random_token_mask(
            size=(batch_size, model.sequence_length),
            mask_ratio=model.mask_ratio,  # Use model's mask ratio
            device=model.device,
        )

        # --- Encode Visible Patches ---
        # Pass idx_keep to the encoder
        features_encoded = model.forward_encoder(
            spectrogram_sample, idx_keep=idx_keep)

        # --- Decode Masked Patches ---
        # Pass encoded features and mask indices to the decoder
        predicted_patches = model.forward_decoder(
            features_encoded, idx_keep, idx_mask)

        # --- Prepare Full Reconstruction ---
        # 1. Get original patches
        original_patches = lightly_utils.patchify(
            spectrogram_sample, model.patch_size)
        # 2. Create tensor for full predicted patches (B, N, P*P*C)
        num_patches = model.sequence_length - \
            model.backbone.vit.num_prefix_tokens  # Exclude CLS token if present
        patch_dim = original_patches.shape[-1]
        full_predicted_patches = torch.zeros(
            batch_size, num_patches, patch_dim, device=model.device)

        # 3. Filter indices and adjust
        cls_tokens = model.backbone.vit.num_prefix_tokens  # Usually 1 for CLS
        batch_indices = torch.arange(
            batch_size, device=model.device).unsqueeze(-1)

        # --- Filter indices to get only IMAGE patch indices ---
        patch_indices_mask = idx_mask[idx_mask >= cls_tokens]
        patch_indices_keep = idx_keep[idx_keep >= cls_tokens]

        # --- Adjust indices to be 0-based for patch tensors ---
        patch_indices_mask_adjusted = patch_indices_mask - cls_tokens
        patch_indices_keep_adjusted = patch_indices_keep - cls_tokens

        # --- Handle Predicted Patches ---
        # Check if decoder predicts CLS token + image patches or just image patches
        num_masked_image_patches = patch_indices_mask.shape[0]
        if predicted_patches.shape[1] == num_masked_image_patches:
            predicted_image_patches = predicted_patches
        elif cls_tokens > 0 and (idx_mask == 0).any() and predicted_patches.shape[1] == num_masked_image_patches + 1:
            # If CLS token was masked and decoder predicted it (usually as the first output), skip it
            predicted_image_patches = predicted_patches[:, 1:]
            print(
                "Note: Assuming first predicted patch corresponds to CLS token and skipping it.")
        else:
            # If shapes don't match and CLS wasn't masked/predicted, raise error or handle differently
            if predicted_patches.shape[1] != num_masked_image_patches:
                raise ValueError(
                    f"Shape mismatch: Predicted patches ({predicted_patches.shape[1]}) "
                    f"vs expected masked image patches ({num_masked_image_patches}). "
                    f"Decoder output format might be unexpected or CLS token handling is wrong."
                )
            else:
                predicted_image_patches = predicted_patches  # Shape matches, proceed

        # Place predicted image patches at masked locations using adjusted indices
        if predicted_image_patches.numel() > 0 and patch_indices_mask_adjusted.numel() > 0:
            full_predicted_patches[batch_indices,
                                   patch_indices_mask_adjusted] = predicted_image_patches

        # 4. Place original patches at unmasked locations
        # Get original patches corresponding to kept image patch indices
        if patch_indices_keep_adjusted.numel() > 0:
            # --- Replace this line ---
            # original_kept_patches = lightly_utils.get_at_index(
            #     original_patches, patch_indices_keep_adjusted)
            # --- With this block ---
            # e.g., (1, 1024, 256)
            B, N_img_patches, D = original_patches.shape
            # patch_indices_keep_adjusted has shape (N_keep_image,)
            # Add batch dim and expand index for gather operation
            indices_for_gather = patch_indices_keep_adjusted.unsqueeze(
                0).unsqueeze(2)  # Shape (1, N_keep_image, 1)
            indices_for_gather = indices_for_gather.expand(
                B, -1, D)  # Shape (1, N_keep_image, 256)
            original_kept_patches = torch.gather(
                original_patches, dim=1, index=indices_for_gather)  # Shape (1, N_keep_image, 256)
            # --- End of replacement ---

            # Place original patches at unmasked locations using adjusted indices
            full_predicted_patches[batch_indices,
                                   patch_indices_keep_adjusted] = original_kept_patches

        # 5. Unpatchify to get the full reconstructed image
        # Use lightly's unpatchify, assuming single channel for spectrogram
        reconstruction_full = lightly_utils.unpatchify(
            full_predicted_patches, model.patch_size, channels=1)

        # --- Prepare Masked Image Visualization ---
        # Create a mask image based on masked IMAGE patches
        mask_viz_patches = torch.zeros(
            batch_size, num_patches, patch_dim, device=model.device)
        if patch_indices_mask_adjusted.numel() > 0:
            # Use the adjusted mask indices for image patches
            # Mark masked patches
            index_scatter_mask_viz = patch_indices_mask_adjusted.unsqueeze(
                0).unsqueeze(-1).expand(batch_size, -1, patch_dim)
            src_mask_viz = torch.ones(
                batch_size, num_masked_image_patches, patch_dim, device=model.device)
            mask_viz_patches.scatter_(
                dim=1, index=index_scatter_mask_viz, src=src_mask_viz)

        # Use lightly's unpatchify for the mask as well
        mask_viz = lightly_utils.unpatchify(
            mask_viz_patches, model.patch_size, channels=1)

    # 4. Visualize
    visualize_mae_output(
        original=spectrogram_sample.squeeze(0),  # Remove batch dim for viz
        mask=mask_viz.squeeze(0),  # Pass the generated mask image
        reconstruction=reconstruction_full.squeeze(
            0),  # Use the full reconstruction
        output_dir=Path(cli_args.output_dir),
        filename_prefix=f"sample_{cli_args.sample_idx}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize MAE input, masked input, and reconstruction.")
    parser.add_argument("--checkpoint_path", required=True, type=str,
                        help="Path to the MAE model checkpoint (.ckpt or .pth).")
    parser.add_argument("--dataset_path", required=True, type=str,
                        help="Path to the root directory of the dataset.")
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="Index of the sample to visualize from the dataset.")
    parser.add_argument("--output_dir", type=str, default="./mae_visualizations",
                        help="Directory to save the output images.")

    cli_args = parser.parse_args()
    config = TrainingConfig(
        root=Path(cli_args.dataset_path),
        dataset=DatasetType.TORCHSIG_NARROWBAND,
        backbone=BackboneType.VIT_BASE,
        ssl_model=SSLModelType.MAE,
        batch_size=1,
        spectrogram=True,
        family=True,
        resize=(224, 224),
    )
    main(cli_args, config=config)
